"""Spend saved force evaluations on previously unsampled cells near observed boundaries."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from study import BOUNDS, STANDARD, boundary_mask, flagged, read_csv, upsample, work


POLICY = {
    'name': 'observed-interface-frontier-v1',
    'max_batch_cells': 3072,
    'max_estimated_batch_rhs': 85_000_000,
    'minimum_reserve_rhs': 10_000_000,
    'tie_seed': 26261,
}


@dataclass(slots=True)
class RefinementArchive:
    labels: np.ndarray
    sampled: np.ndarray
    times: np.ndarray
    minimums: np.ndarray
    costs: np.ndarray
    evaluations: int


def archived_cell(row, grid):
    """Recover the row and column of an archived grid-center initial condition."""
    x, y = float(row['x']), float(row['y'])
    xmin, xmax, ymin, ymax = BOUNDS
    scaled_x = (x - xmin) * grid / (xmax - xmin) - 0.5
    scaled_y = (y - ymin) * grid / (ymax - ymin) - 0.5
    column, line = round(scaled_x), round(scaled_y)
    if (not 0 <= column < grid or not 0 <= line < grid or
            abs(scaled_x - column) > 1e-8 or abs(scaled_y - line) > 1e-8):
        raise ValueError(f'Archived point ({x}, {y}) is outside the {grid}-cell sampling lattice.')
    return line, column


def load_refinement_archive(lc_run, base=32):
    """Reconstruct inherited labels and direct samples using only the adaptive arm's CSVs."""
    raw = Path(lc_run) / 'raw'
    labels = np.full((base, base), 5, dtype=np.uint8)
    times = np.zeros((base, base))
    minimums = np.full((base, base), np.inf)
    costs = np.zeros((base, base), dtype=np.int64)
    sampled = np.zeros((base, base), dtype=bool)
    evaluations = 0
    for depth, name in enumerate(('uniform_0', 'adaptive_1', 'adaptive_2', 'adaptive_3')):
        grid = base * 2**depth
        if depth:
            labels, times, minimums, costs = (upsample(array, 2) for array in (labels, times, minimums, costs))
            sampled = np.zeros((grid, grid), dtype=bool)
        inputs = read_csv(raw / f'{name}.in.csv')
        outputs = read_csv(raw / f'{name}.out.csv')
        for index, (point, record) in enumerate(zip(inputs, outputs, strict=True)):
            if int(point['id']) != index or int(record['id']) != index:
                raise ValueError(f'Archived {name} IDs are missing or out of order.')
            cell = archived_cell(point, grid)
            if sampled[cell]:
                raise ValueError(f'Archived {name} contains a duplicate initial condition at {cell}.')
            label = int(record['label'])
            error = float(record['energy_error'])
            evaluations_in_row = int(record['evals'])
            if not 0 <= label <= 5 or evaluations_in_row < 0:
                raise ValueError(f'Archived {name} has an invalid label or work count at {cell}.')
            labels[cell] = 5 if not math.isfinite(error) or error > 1e-5 else label
            times[cell] = float(record['time'])
            minimums[cell] = float(record['min_r'])
            costs[cell] = evaluations_in_row
            sampled[cell] = True
            evaluations += evaluations_in_row
        if depth == 0 and not np.all(sampled):
            raise ValueError('Archived adaptive base grid is incomplete.')
    return RefinementArchive(labels, sampled, times, minimums, costs, evaluations)


def distance_to(seeds):
    if np.any(seeds):
        return distance_transform_edt(~seeds)
    return np.full(seeds.shape, np.inf)


def priority_map(archive, changed):
    labels = archive.labels
    edges = boundary_mask(labels)
    unresolved = labels == 5
    # Directly observed non-close outcomes probe whether a nearby inherited close
    # label hides a narrow escape channel. Changed labels expand this frontier.
    minority = archive.sampled & (labels < 4)
    d_edges = distance_to(edges)
    d_outcomes = distance_to(minority | changed)
    grazing = (archive.minimums > 0.001) & (archive.minimums < 0.003)
    elapsed = np.nan_to_num(archive.times, nan=0.0, posinf=50.0, neginf=0.0)
    long_lived = np.log1p(np.clip(elapsed, 0, 50)) / np.log1p(50)
    return (12.0/(1 + d_edges) + 5.0/(1 + d_outcomes) + 4.0*unresolved +
            2.0*grazing + 2.0*long_lived)


def select_batch(archive, changed, tie, estimated_work, budget, extra_cost_factor):
    ys, xs = np.nonzero(~archive.sampled)
    if not len(xs):
        return [], 0.0
    priorities = priority_map(archive, changed)[ys, xs]
    order = np.lexsort((tie[ys, xs], -priorities))[:POLICY['max_batch_cells']]
    actualys, actualxs = ys[order], xs[order]
    prior_cost = np.maximum(archive.costs[actualys, actualxs], 1)
    predictions = np.maximum(estimated_work, prior_cost * extra_cost_factor)
    prefix = np.cumsum(predictions)
    take = int(np.searchsorted(prefix, budget, side='right'))
    return [(int(x), int(y)) for y, x in zip(actualys[:take], actualxs[:take])], float(prefix[take - 1]) if take else 0.0


def refine_boundaries(study, lc_run, budget):
    """Return the refined image and a JSON-ready work ledger without accessing validation labels."""
    archive = load_refinement_archive(lc_run, study.base)
    if study.fine != archive.labels.shape[0] or budget < 0:
        raise ValueError('The study grid or boundary refinement budget is invalid.')
    initial_samples = int(np.count_nonzero(archive.sampled))
    tie = np.random.default_rng(POLICY['tie_seed']).permutation(study.fine**2).reshape(archive.labels.shape)
    changed = np.zeros(archive.labels.shape, dtype=bool)
    reserve = min(POLICY['minimum_reserve_rhs'], budget // 16)
    baseline_mean = archive.evaluations / max(1, int(np.count_nonzero(archive.costs)))
    estimated_work = max(25_000.0, 0.55 * baseline_mean)
    extra_cost_factor = 0.70
    spent = 0
    batches = []
    while True:
        remaining = budget - spent
        target = min(POLICY['max_estimated_batch_rhs'], (remaining - reserve)*0.8)
        if target <= 0:
            break
        coords, estimated = select_batch(archive, changed, tie, estimated_work, target, extra_cost_factor)
        if not coords:
            break
        index = len(batches)
        results, elapsed = study.solve_cells(f'boundary_refine_{index:03d}', coords, study.fine, settings=STANDARD)
        if set(results) != set(coords):
            raise RuntimeError('Boundary refinement returned a different set of cells than requested.')
        realized = work(results)
        spent += realized
        if spent > budget:
            raise RuntimeError(f'Boundary refinement exceeded its RHS savings allocation by {spent - budget:,}.')
        revisions = 0
        for (i, j), result in results.items():
            cell = j, i
            label = 5 if flagged(result) else result.label
            revised = label != int(archive.labels[cell])
            revisions += revised
            changed[cell] |= revised
            archive.labels[cell] = label
            archive.sampled[cell] = True
            archive.times[cell] = result.time
            archive.minimums[cell] = result.min_r
            archive.costs[cell] = result.evals
        signal = max(8_000.0, 1.15 * realized / len(coords))
        estimated_work = 0.55 * estimated_work + 0.45 * signal
        ratio = 1.10 * realized / estimated
        extra_cost_factor = float(np.clip(extra_cost_factor * (0.55 + 0.45 * ratio), 0.20, 2.50))
        batches.append({'cells': len(coords), 'rhs': realized, 'estimated_rhs': round(estimated),
                        'revised_labels': revisions, 'seconds': elapsed})
    return archive.labels, {
        'policy': dict(POLICY), 'budget_rhs': budget, 'additional_rhs': spent,
        'unspent_rhs': budget - spent, 'original_adaptive_rhs': archive.evaluations,
        'total_adaptive_rhs': archive.evaluations + spent, 'original_fine_samples': initial_samples,
        'additional_fine_samples': sum(row['cells'] for row in batches),
        'revised_labels': sum(row['revised_labels'] for row in batches),
        'additional_seconds': sum(row['seconds'] for row in batches), 'batches': batches,
    }

"""Run frozen Levi-Civita sampling and live matched-work GPU timings on shifted grids."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

import boundary_refine
import study as core


BASE = 32
FINE = BASE * 8
HORIZON = 50.0
EXTRA_BUDGET = 641_438_440
WORK_GAP_LIMIT = 0.01
ORDER_SEED = 20261120
BOOTSTRAP_SEED = 20261121
SHIFTS = ((0.25, -0.25), (-0.25, 0.125), (0.125, 0.25))


def physical_point(cell, n, shift):
    """Translate each dyadic lattice by fixed fractions of a fine cell."""
    xmin, xmax, ymin, ymax = core.BOUNDS
    x, y = core.cell_point(cell, n)
    return x + shift[0] * ((xmax - xmin) / FINE), y + shift[1] * ((ymax - ymin) / FINE)


class _ShiftedStudy(core.Study):
    def __init__(self, output, binaries, device, shift, *, logical=False):
        super().__init__(Path(output), Path(binaries), device, BASE, HORIZON, 'levi-civita')
        self.shift = tuple(shift)
        self.solver_seconds = 0.0
        self.filter_seconds = 0.0
        self.trajectories = 0
        self.work = 0
        self.solver_batches = []
        self.logical = self.out / 'logical'
        if logical:
            (self.logical / 'raw').mkdir(parents=True)

    def solve_cells(self, name, coords, n, settings=core.STANDARD):
        coords = list(coords)
        result, elapsed = self.solve(name, {cell: physical_point(cell, n, self.shift)
                                            for cell in coords}, settings)
        self.solver_seconds += elapsed
        self.work += core.work(result)
        self.trajectories += len(result)
        if coords:
            self.solver_batches.append(name)
        if name in ('uniform_0', 'adaptive_1', 'adaptive_2', 'adaptive_3'):
            logical_raw = self.logical / 'raw'
            if logical_raw.is_dir():
                core.write_csv(logical_raw / f'{name}.in.csv', ('id', 'x', 'y'),
                               ((index, *core.cell_point(cell, n)) for index, cell in enumerate(coords)))
                physical_output = self.raw / f'{name}.out.csv'
                logical_output = logical_raw / physical_output.name
                if coords:
                    shutil.copyfile(physical_output, logical_output)
                else:
                    core.write_csv(logical_output, ('id', 'label', 'time', 'min_r', 'evals', 'energy_error'), ())
        return result, elapsed

    def filter(self, name, labels, coords, repeat=1, backend='fused'):
        mask, stats, elapsed = super().filter(name, labels, coords, repeat, backend)
        self.filter_seconds += elapsed
        return mask, stats, elapsed


@dataclass(frozen=True)
class _Run:
    labels: np.ndarray
    work: int
    trajectories: int
    wall_seconds: float
    solver_seconds: float
    filter_seconds: float
    schedule_sha256: str
    result_sha256: str
    map_sha256: str

    def metrics(self):
        return {name: value for name, value in vars(self).items() if name != 'labels'}


def _finish(study, labels, elapsed):
    schedule = hashlib.sha256()
    results = hashlib.sha256()
    for name in study.solver_batches:
        for digest, kind in ((schedule, 'in'), (results, 'out')):
            path = study.raw / f'{name}.{kind}.csv'
            digest.update(name.encode('ascii') + b'\0' + bytes.fromhex(core.digest(path)))
    return _Run(labels, study.work, study.trajectories, elapsed, study.solver_seconds,
                study.filter_seconds, schedule.hexdigest(), results.hexdigest(),
                hashlib.sha256(labels.tobytes()).hexdigest())


def _adaptive(output, binaries, device, shift, extra_budget):
    start = time.perf_counter()
    study = _ShiftedStudy(output, binaries, device, shift, logical=True)
    initial, _ = study.solve_cells('uniform_0', core.cells(BASE), BASE)
    levels = study.adaptive(initial)
    labels, ledger = boundary_refine.refine_boundaries(study, study.logical, extra_budget)
    elapsed = time.perf_counter() - start
    if ledger['original_adaptive_rhs'] != levels[-1]['evaluations'] or ledger['total_adaptive_rhs'] != study.work:
        raise RuntimeError('Online adaptive work differs from the frozen refinement ledger.')
    core.write_json(study.out / 'adaptive_ledger.json', ledger)
    return _finish(study, labels, elapsed)


def _uniform_calibration(output, binaries, device, shift, budget):
    start = time.perf_counter()
    study = _ShiftedStudy(output, binaries, device, shift)
    results = []
    for level in range(4):
        side = BASE * (2**level)
        observations, _ = study.solve_cells(f'uniform_{level}', core.cells(side), side)
        results.append(observations)
    grid, spent, trajectories, batches = core.uniform_at_budget(results, BASE, budget)
    schedule = [(BASE, core.cells(BASE)), *((side, coords) for side, coords, _ in batches)]
    if spent > budget:
        raise RuntimeError('The uniform base grid alone exceeded the adaptive work budget.')
    elapsed = time.perf_counter() - start
    core.write_json(study.out / 'matched_prefix.json',
                    {'work': spent, 'trajectories': trajectories,
                     'levels': [{'side': n, 'cells': coords} for n, coords in schedule]})
    return schedule, _Run(grid, spent, trajectories, elapsed, study.solver_seconds,
                          0.0, '', '', hashlib.sha256(grid.tobytes()).hexdigest()), study.work


def _uniform(output, binaries, device, shift, schedule):
    start = time.perf_counter()
    study = _ShiftedStudy(output, binaries, device, shift)
    base, coords = schedule[0]
    results, _ = study.solve_cells('uniform_0', coords, base)
    grid = core.labels_array(results, base)
    for level, (side, coords) in enumerate(schedule[1:], start=1):
        grid = core.upsample(grid, 2)
        results, _ = study.solve_cells(f'uniform_{level}', coords, side)
        for (i, j), result in results.items():
            grid[j, i] = 5 if core.flagged(result) else result.label
    if len(grid) != FINE:
        grid = core.upsample(grid, FINE // len(grid))
    return _finish(study, grid, time.perf_counter() - start)


def _telemetry(device):
    def query(argv):
        try:
            result = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    metrics = ('name', 'driver_version', 'pstate', 'temperature.gpu', 'clocks.current.sm',
               'clocks.current.memory', 'power.draw', 'utilization.gpu', 'memory.used')
    selector = ['nvidia-smi', '-i', str(device)]
    values = query([*selector, '--query-gpu=' + ','.join(metrics), '--format=csv,noheader,nounits'])
    values = [value.strip() for value in values.split(',')] if values else []
    processes = query([*selector, '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    if processes is None:
        process_count = None
    else:
        lines = processes.splitlines()
        process_count = len(lines) if all(line.strip().isdigit() for line in lines) else None
    return {'device_metrics': dict(zip(metrics, values, strict=True)) if len(values) == len(metrics) else None,
            'compute_process_count': process_count}


def _signature(run):
    return run.work, run.trajectories, run.schedule_sha256, run.result_sha256, run.map_sha256


def _ratio_interval(values):
    if not values:
        return None
    ratios = np.asarray(values)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(ratios, size=(10_000, len(ratios)), replace=True)
    lower, upper = np.quantile(np.median(draws, axis=1), (0.025, 0.975))
    return {'median': float(np.median(ratios)), 'ci95': [float(lower), float(upper)]}


def run_grid(output, binaries, device, shift, *, pairs=7, extra_budget=EXTRA_BUDGET):
    """Calibrate once, then run a live warmup pair and repeated paired timings.

    The maps returned are from the fresh calibration and are checked against
    every completed repeat. 'solver_seconds' are solver subprocess wall time,
    including GPU synchronization and CSV handling, not GPU kernel-only time.
    """
    if pairs < 1 or extra_budget <= 0:
        raise ValueError('A positive work budget and at least one timing pair are required.')
    output, binaries, shift = Path(output), Path(binaries), tuple(shift)
    output.mkdir(parents=True)
    adaptive = _adaptive(output / 'calibration_adaptive', binaries, device, shift, extra_budget)
    schedule, uniform, calibration_work = _uniform_calibration(
        output / 'calibration_uniform', binaries, device, shift, adaptive.work)
    work_gap = (adaptive.work - uniform.work) / adaptive.work
    if not 0 <= work_gap <= WORK_GAP_LIMIT:
        metrics = {'status': 'inconclusive_work_match', 'shift': list(shift), 'work_gap': work_gap,
                   'adaptive': adaptive.metrics(), 'uniform': uniform.metrics(), 'pairs': [],
                   'calibration_uniform_evaluations': calibration_work, 'timing_ratio': None}
        core.write_json(output / 'timing.json', metrics)
        return adaptive.labels, uniform.labels, metrics

    def run(arm, path):
        print(f'Holdout {shift} {path.parent.name} {arm}', flush=True)
        before = _telemetry(device)
        if arm == 'adaptive':
            result = _adaptive(path, binaries, device, shift, extra_budget)
        else:
            result = _uniform(path, binaries, device, shift, schedule)
        after = _telemetry(device)
        comparison = adaptive if arm == 'adaptive' else uniform
        matches = ((result.work, result.trajectories, result.map_sha256) ==
                   (comparison.work, comparison.trajectories, comparison.map_sha256))
        return result, {**result.metrics(), 'matches_calibration': matches,
                        'gpu_before': before, 'gpu_after': after}

    seed_offset = SHIFTS.index(shift) if shift in SHIFTS else 0
    adaptive_first = bool(np.random.default_rng(ORDER_SEED + seed_offset).integers(2))
    arms = ('adaptive', 'uniform')
    expected = {'adaptive': _signature(adaptive)}
    warmup = {}
    for arm in arms[::-1] if adaptive_first else arms:
        result, warmup[arm] = run(arm, output / 'warmup' / arm)
        if arm == 'uniform':
            expected[arm] = _signature(result)
    completed = []
    for trial in range(pairs):
        order = arms if (trial % 2 == 0) == adaptive_first else arms[::-1]
        detail = {}
        for arm in order:
            result, detail[arm] = run(arm, output / f'pair_{trial+1:02d}' / arm)
            detail[arm]['matches_live_signature'] = _signature(result) == expected[arm]
        valid = all(arm['matches_calibration'] and arm['matches_live_signature'] for arm in detail.values())
        elapsed_ratio = detail['uniform']['wall_seconds'] / detail['adaptive']['wall_seconds']
        completed.append({'pair': trial + 1, 'order': list(order), **detail, 'valid': valid,
                          'uniform_over_adaptive_wall': elapsed_ratio})
        core.write_json(output / 'timing_progress.json', completed)

    warmup_good = all(row['matches_calibration'] for row in warmup.values())
    valid = warmup_good and all(row['valid'] for row in completed)
    ratios = [row['uniform_over_adaptive_wall'] for row in completed] if valid else []
    metrics = {'status': 'passed' if valid else 'inconclusive_replay_drift', 'shift': list(shift),
               'work_gap': work_gap, 'adaptive': adaptive.metrics(), 'uniform': uniform.metrics(),
               'calibration_uniform_evaluations': calibration_work,
               'warmup': warmup, 'pairs': completed, 'timing_ratio': _ratio_interval(ratios)}
    core.write_json(output / 'timing.json', metrics)
    return adaptive.labels, uniform.labels, metrics


class _PilotReplay:
    base = BASE
    fine = FINE
    adaptive = core.Study.adaptive

    def __init__(self, lc, refined):
        self.lc = Path(lc)
        self.refined = Path(refined)
        self.trajectories = 0
        self.stages = []

    def solve_cells(self, name, coords, n, settings=core.STANDARD):
        using_refine = name.startswith('boundary_refine_')
        raw = (self.refined if using_refine else self.lc) / 'raw'
        inputs = core.read_csv(raw / f'{name}.in.csv')
        outputs = core.read_csv(raw / f'{name}.out.csv')
        self.stages.append(name)
        results = {}
        for index, (cell, point, row) in enumerate(zip(coords, inputs, outputs, strict=True)):
            if (int(point['id']) != index or int(row['id']) != index or
                    physical_point(cell, n, (0, 0)) != (float(point['x']), float(point['y']))):
                raise RuntimeError(f'Zero shift changed coordinates or scheduling in {name}.')
            results[cell] = core.Result(int(row['label']), float(row['time']), float(row['min_r']),
                                        int(row['evals']), float(row['energy_error']), ())
        self.trajectories += len(results)
        return results, 0.0

    def filter(self, name, labels, coords, repeat=1, backend='fused'):
        raw = self.lc / 'raw'
        inputs = core.read_csv(raw / f'{name}.in.csv')
        outputs = core.read_csv(raw / f'{name}.out.csv')
        mask = {}
        for index, ((i, j), sample, result) in enumerate(zip(coords, inputs, outputs, strict=True)):
            neighborhood = [labels.get((i + dx, j + dy), -1)
                            for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
            present = set(neighborhood) - {-1}
            chosen = 5 in present or len(present) > 1
            if (int(sample['id']) != index or int(result['id']) != index or
                    neighborhood != [int(sample[f'c{k}']) for k in range(9)] or
                    int(result['refine']) != int(chosen)):
                raise RuntimeError(f'Zero shift changed neighborhood scheduling in {name}.')
            mask[i, j] = chosen
        return mask, None, 0.0


def validate_zero_shift(lc_archive, refined_archive):
    """Replay the exact pilot sampling decisions without invoking a GPU."""
    lc_archive, refined_archive = Path(lc_archive), Path(refined_archive)
    replay = _PilotReplay(lc_archive, refined_archive)
    initial, _ = replay.solve_cells('uniform_0', core.cells(BASE), BASE)
    levels = replay.adaptive(initial)
    labels, ledger = boundary_refine.refine_boundaries(replay, lc_archive, EXTRA_BUDGET)
    with np.load(lc_archive / 'outcome_grids.npz', allow_pickle=False) as prior:
        if not np.array_equal(levels[-1]['map'], prior['adaptive']):
            raise RuntimeError('Zero-shift initial adaptive image differs from the pilot.')
    with np.load(refined_archive / 'outcome_grids.npz', allow_pickle=False) as prior:
        if not np.array_equal(labels, prior['adaptive']):
            raise RuntimeError('Zero-shift refined image differs from the pilot.')
    expected = sorted(path.name.removesuffix('.in.csv') for path in
                      (refined_archive / 'raw').glob('boundary_refine_*.in.csv'))
    if sorted(name for name in replay.stages if name.startswith('boundary_refine_')) != expected:
        raise RuntimeError('Zero-shift replay performed a different number of refinement batches.')
    return {'passed': True, 'rhs': ledger['total_adaptive_rhs'], 'trajectories': replay.trajectories,
            'refinement_batches': len(expected),
            'adaptive_map_sha256': hashlib.sha256(labels.tobytes()).hexdigest()}

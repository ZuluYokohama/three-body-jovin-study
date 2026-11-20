"""Run the verified Newtonian projection, sampling comparison, and GPU filter benchmark."""

import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from html import escape
import zlib

import numpy as np

from reference import compare_point, integrate, normalized_state_error


ROOT = Path(__file__).resolve().parent
STATE_COLUMNS = tuple(f'{prefix}{i}' for prefix_group in (('x', 'y'), ('vx', 'vy'))
                      for i in (1, 2, 3) for prefix in prefix_group)
LABELS = ('Horizon reached', 'Body 1 escapes', 'Body 2 escapes', 'Body 3 escapes',
          'Close approach', 'Unresolved')
COLORS = ((36, 52, 73), (72, 178, 235), (247, 161, 79), (201, 105, 192),
          (240, 224, 176), (106, 114, 126))
BOUNDS = (0.1, 0.9, 0.05, 0.65)
STANDARD = {'rtol': 1e-9, 'atol': 1e-11, 'hmax': 0.04}
TIGHT = {'rtol': 1e-11, 'atol': 1e-13, 'hmax': 0.02}


@dataclass(frozen=True, slots=True)
class Result:
    label: int
    time: float
    min_r: float
    evals: int
    energy_error: float
    state: tuple
    chart_entries: int = 0
    chart_exits: int = 0
    max_lc_residual: float = 0.0


@dataclass(frozen=True, slots=True)
class Baseline:
    directory: Path
    hashes: dict
    audit_rows: list
    reference: np.ndarray
    adaptive: np.ndarray


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def work(results):
    return sum(result.evals for result in results.values())


def cell_point(cell, n):
    i, j = cell
    xmin, xmax, ymin, ymax = BOUNDS
    return xmin + (i + 0.5)*(xmax-xmin)/n, ymin + (j + 0.5)*(ymax-ymin)/n


def cells(n):
    return [(i, j) for j in range(n) for i in range(n)]


def children(cell):
    i, j = cell
    return [(2*i + di, 2*j + dj) for dj in range(2) for di in range(2)]


def run_command(argv, log, timeout=900):
    start = time.perf_counter()
    result = subprocess.run([str(arg) for arg in argv], capture_output=True, text=True,
                            timeout=timeout, check=False)
    elapsed = time.perf_counter() - start
    log.write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'{argv[0]} exited {result.returncode}. See {log}')
    return result.stdout, elapsed


class Study:
    def __init__(self, output, binaries, device, base, horizon, coordinates):
        self.out = output
        self.binaries = binaries
        self.device = device
        self.base = base
        self.fine = base * 8
        self.horizon = horizon
        self.coordinates = coordinates
        self.raw = output / 'raw'
        self.raw.mkdir(parents=True)

    def solve(self, name, points, settings, horizon=None):
        if not points:
            return {}, 0.0
        input_path, output_path = (self.raw / f'{name}.{extension}.csv' for extension in ('in', 'out'))
        keys = list(points)
        write_csv(input_path, ('id', 'x', 'y'), ((i, *points[key]) for i, key in enumerate(keys)))
        command = [self.binaries / 'solver.exe', input_path, output_path,
                   '--rtol', settings['rtol'], '--atol', settings['atol'], '--hmax', settings['hmax'],
                   '--horizon', self.horizon if horizon is None else horizon, '--device', self.device]
        if self.coordinates == 'levi-civita':
            command.extend(('--coordinates', self.coordinates))
        _, elapsed = run_command(command, self.raw / f'{name}.log')
        rows = read_csv(output_path)
        if [int(row['id']) for row in rows] != list(range(len(keys))):
            raise ValueError(f'Trajectory IDs were missing or out of order for {name}.')
        results = {key: Result(int(row['label']), float(row['time']), float(row['min_r']),
                               int(row['evals']), float(row['energy_error']),
                               tuple(float(row[col]) for col in STATE_COLUMNS),
                               int(row.get('chart_entries', 0)), int(row.get('chart_exits', 0)),
                               float(row.get('max_lc_residual', '0')))
                   for key, row in zip(keys, rows)}
        if any(not 0 <= result.label <= 5 or result.evals < 0 for result in results.values()):
            raise ValueError(f'Invalid label or force-evaluation count in {output_path}.')
        print(f'{name}: {len(results):,} trajectories, {work(results):,} force evaluations, {elapsed:.2f} s',
              flush=True)
        return results, elapsed

    def solve_cells(self, name, coords, n, settings=STANDARD):
        return self.solve(name, {cell: cell_point(cell, n) for cell in coords}, settings)

    def filter(self, name, labels, coords, repeat=1, backend='fused'):
        if not coords:
            return {}, None, 0.0
        input_path, output_path = (self.raw / f'{name}.{extension}.csv' for extension in ('in', 'out'))
        rows = ((index, *(labels.get((i + di, j + dj), -1)
                           for dj in (-1, 0, 1) for di in (-1, 0, 1)))
                for index, (i, j) in enumerate(coords))
        write_csv(input_path, ('id', *(f'c{i}' for i in range(9))), rows)
        stdout, elapsed = run_command(
            [self.binaries / 'filter.exe', input_path, output_path,
             '--repeat', repeat, '--device', self.device, '--backend', backend],
            self.raw / f'{name}.log')
        values = read_csv(output_path)
        if ([int(row['id']) for row in values] != list(range(len(coords))) or
                any(row['refine'] not in ('0', '1') for row in values)):
            raise ValueError(f'Invalid filter rows in {output_path}.')
        stats = json.loads(stdout)
        if not stats['agreement']:
            raise RuntimeError(f'Filter parity failed in {name}.')
        return {cell: row['refine'] == '1' for cell, row in zip(coords, values)}, stats, elapsed

    def gate_gpu(self):
        points = [(0.17, 0.48), (0.41, 0.62), (0.56, 0.34), (0.81, 0.51)]
        horizon = 0.05
        settings = {'rtol': 1e-13, 'atol': 1e-15, 'hmax': 0.001}
        outputs, _ = self.solve('gpu_state_gate', dict(enumerate(points)), settings, horizon)
        errors = []
        for i, point in enumerate(points):
            label, final_time, expected = integrate(point, horizon=horizon, rtol=1e-13,
                                                     atol=1e-15, max_step=0.001)
            output = outputs[i]
            if label != 0 or output.label != 0 or abs(output.time - final_time) > 1e-14:
                raise RuntimeError(f'GPU short-time gate failed to reach the requested horizon for case {i}.')
            errors.append(normalized_state_error(output.state, expected))
        if not np.isfinite(errors).all() or max(errors) > 1e-11:
            raise RuntimeError(f'GPU state parity gate failed: {errors}')
        print(f'GPU state gate passed, max normalized error {max(errors):.3g}', flush=True)
        return {'passed': True, 'fixtures': len(points), 'max_normalized_error': max(errors)}

    def gate_lc(self):
        from lc_checks import check_coupled_passages

        stdout, _ = run_command([self.binaries / 'solver.exe', '--verify-lc', '--device', self.device],
                                self.raw / 'gpu_lc_crossing.log')
        crossing = json.loads(stdout)
        if crossing.get('passed') is not True:
            raise RuntimeError('The analytic CUDA radial collision check failed.')
        coupled = check_coupled_passages(self.solve)
        print(f"GPU Levi-Civita collision and coupled-passage gates passed ({coupled['fixtures']} coupled fixtures)",
              flush=True)
        return {'crossing': crossing, 'coupled': coupled}

    def adaptive(self, initial):
        labels = labels_array(initial, self.base)
        results = initial
        levels = [{'map': labels.copy(), 'evaluations': work(initial), 'seconds': 0.0,
                   'trajectories': len(initial)}]
        for depth in range(3):
            start = time.perf_counter()
            coords = list(results)
            n = self.base * 2**depth
            # Inherited coarse labels give neighbors to newly sampled cells at patch edges.
            current_labels = {(i, j): int(labels[j, i]) for i, j in cells(n)}
            mask, _, _ = self.filter(f'adaptive_filter_{depth}', current_labels, coords)
            selected = [cell for cell in coords if mask[cell] or
                        1e-3 < results[cell].min_r < 3e-3 or flagged(results[cell])]
            coords = [child for cell in selected for child in children(cell)]
            results, _ = self.solve_cells(f'adaptive_{depth+1}', coords, 2*n)
            labels = upsample(labels, 2)
            for (i, j), result in results.items():
                labels[j, i] = 5 if flagged(result) else result.label
            previous = levels[-1]
            levels.append({'map': labels.copy(), 'evaluations': work(results) + previous['evaluations'],
                           'trajectories': len(results) + previous['trajectories'],
                           'seconds': previous['seconds'] + time.perf_counter() - start})
        return levels


def flagged(result):
    return (result.label == 5 or not np.isfinite(result.energy_error) or result.energy_error > 1e-5)


def labels_array(results, n):
    labels = np.full((n, n), 5, dtype=np.uint8)
    for (i, j), result in results.items():
        labels[j, i] = 5 if flagged(result) else result.label
    return labels


def upsample(grid, scale):
    return np.repeat(np.repeat(grid, scale, axis=0), scale, axis=1)


def uniform_at_budget(uniform, base, budget):
    grid = labels_array(uniform[0], base)
    spent = work(uniform[0])
    total = len(uniform[0])
    batches = []
    for level, results in enumerate(uniform[1:]):
        n = base * 2**level
        grid = upsample(grid, 2)
        selected = []
        batches.append((2*n, selected, results))
        # Deterministic outcome-independent permutation spreads a partial level over the plane.
        order = np.random.default_rng(10919 + level).permutation(n*n)
        for ident in order:
            group = children((int(ident) % n, int(ident) // n))
            cost = sum(results[cell].evals for cell in group)
            if spent + cost > budget:
                return upsample(grid, (base*8)//len(grid)), spent, total, batches
            spent += cost
            total += 4
            selected.extend(group)
            for i, j in group:
                result = results[i, j]
                grid[j, i] = 5 if flagged(result) else result.label
    return grid, spent, total, batches


def boundary_mask(labels):
    edges = np.zeros(labels.shape, dtype=bool)
    for axis in (0, 1):
        a, b = (labels[:-1], labels[1:]) if axis == 0 else (labels[:, :-1], labels[:, 1:])
        change = (a != b) & (a != 5) & (b != 5)
        if axis == 0:
            edges[:-1] |= change
            edges[1:] |= change
        else:
            edges[:, :-1] |= change
            edges[:, 1:] |= change
    return edges


def score(predicted, expected, mask):
    eligible = np.count_nonzero(mask)
    if not eligible:
        return {'count': 0, 'matches': 0, 'coverage': None, 'agreement': None}
    matches = int(np.count_nonzero(mask & (predicted == expected)))
    return {'count': int(eligible), 'matches': matches,
            'coverage': float(np.count_nonzero(mask & (predicted != 5))/eligible),
            'agreement': matches/int(eligible)}


def audit_cell(row):
    return int(row['i']), int(row['j'])


def load_baseline(directory, config):
    archived = json.loads((directory / 'evidence_hashes.json').read_text(encoding='utf-8'))
    names = ('manifest.json', 'summary.json', 'cpu_audit.csv', 'outcome_grids.npz',
             'raw/uniform_3.out.csv', 'raw/reference_tight.out.csv')
    hashes = {name: digest(directory / name) for name in names}
    if any(archived.get(name) != value for name, value in hashes.items()):
        raise RuntimeError('The archived baseline does not match its evidence hashes.')
    prior = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    for key in ('base', 'levels', 'horizon', 'standard', 'tight', 'cpu_sample', 'target'):
        if prior[key] != config[key]:
            raise RuntimeError(f'The archived baseline used a different {key}.')
    if prior['source_sha256']['reference.py'] != digest(ROOT / 'reference.py'):
        raise RuntimeError('The independent CPU reference differs from the archived baseline.')
    n = config['base'] * 2**config['levels']
    audit = read_csv(directory / 'cpu_audit.csv')
    points = [audit_cell(row) for row in audit]
    if (len(points) != config['cpu_sample'] or len(set(points)) != len(points) or
            any(not (0 <= i < n and 0 <= j < n) for i, j in points)):
        raise RuntimeError('The archived CPU audit cohort is invalid.')
    with np.load(directory / 'outcome_grids.npz', allow_pickle=False) as grids:
        reference, adaptive = grids['reference'], grids['adaptive']
        if (reference.shape != (n, n) or adaptive.shape != (n, n) or
                not np.array_equal(grids['bounds'], config['domain']) or
                not np.isin(reference, range(len(LABELS))).all() or
                not np.isin(adaptive, range(len(LABELS))).all()):
            raise RuntimeError('The archived outcome grids are incompatible with the paired study.')
    return Baseline(directory, hashes, audit, reference, adaptive)


def archived_sample(path, identifiers):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = {number: row for row in csv.DictReader(stream)
                if (number := int(row['id'])) in identifiers}
    if set(rows) != identifiers:
        raise RuntimeError(f'The archived output omits audit cases: {path}')
    return rows


def paired_audit(study, baseline, loose, tight, audit_rows):
    n = study.fine
    identifiers = {j*n + i for i, j in map(audit_cell, baseline.audit_rows)}
    original_standard = archived_sample(baseline.directory / 'raw/uniform_3.out.csv', identifiers)
    original_tight = archived_sample(baseline.directory / 'raw/reference_tight.out.csv', identifiers)
    changes = []
    paired_rows = []
    reclassified_cpu = 0
    for prior, audit in zip(baseline.audit_rows, audit_rows, strict=True):
        i, j, _, _, _, cpu, cpu_tight, _, _ = audit
        if audit_cell(prior) != (i, j):
            raise RuntimeError('The paired audit case order changed.')
        number = j*n + i
        std_old, tight_old = original_standard[number], original_tight[number]
        std_new, tight_new = loose[i, j], tight[i, j]
        cpu_old, cpu_tight_old = int(prior['cpu']), int(prior['cpu_tight'])
        reclassified_cpu += (cpu, cpu_tight) != (cpu_old, cpu_tight_old)
        old_label = int(tight_old['label'])
        old_problem = cpu_old == cpu_tight_old != 5 and (
            old_label != cpu_tight_old or not np.isfinite(float(tight_old['energy_error'])) or
            float(tight_old['energy_error']) > 1e-5)
        matches_now = cpu == cpu_tight != 5 and tight_new.label == cpu_tight and not flagged(tight_new)
        if old_problem:
            changes.append({'i': i, 'j': j, 'cpu_label': cpu_tight, 'old_gpu_label': old_label,
                            'new_gpu_label': tight_new.label, 'gpu_match': bool(matches_now),
                            'old_min_r': float(tight_old['min_r']), 'new_min_r': tight_new.min_r})
        paired_rows.append((i, j, *cell_point((i, j), n), cpu_old, cpu_tight_old, cpu, cpu_tight,
                            prior['cpu_time'], prior['cpu_tight_time'], audit[7], audit[8],
                            std_old['label'], tight_old['label'], tight_old['time'], tight_old['min_r'],
                            tight_old['energy_error'], std_new.label, tight_new.label, tight_new.time,
                            tight_new.min_r, tight_new.energy_error, tight_new.chart_entries,
                            tight_new.chart_exits, int(old_problem), int(matches_now)))
    write_csv(study.out / 'paired_audit.csv', ('i', 'j', 'x', 'y', 'old_cpu', 'old_cpu_tight', 'cpu', 'cpu_tight',
              'old_cpu_time', 'old_cpu_tight_time', 'cpu_time', 'cpu_tight_time', 'old_gpu_standard',
              'old_gpu_tight', 'old_gpu_time', 'old_min_r', 'old_energy_error', 'gpu_standard', 'gpu_tight',
              'gpu_time', 'min_r', 'energy_error', 'chart_entries', 'chart_exits', 'old_problem', 'matches_now'),
              paired_rows)
    return {'reference_labels_changed_on_replay': reclassified_cpu, 'original_disagreements': changes}


def frozen_scores(baseline, adaptive, uniform, reference):
    old_stable = baseline.reference != 5
    old_boundary = boundary_mask(baseline.reference)
    return {'original_adaptive': score(baseline.adaptive, baseline.reference, old_boundary),
            'new_adaptive': score(adaptive, baseline.reference, old_boundary),
            'new_uniform': score(uniform, baseline.reference, old_boundary),
            'original_stable_cells': int(np.count_nonzero(old_stable)),
            'also_resolved_in_new_reference': int(np.count_nonzero(old_stable & (reference != 5)))}


def stratified_coords(loose, tight, tight_grid, limit):
    rng = np.random.default_rng(11223)
    edges = boundary_mask(tight_grid)
    groups = {}
    for cell, result in tight.items():
        i, j = cell
        key = (result.label, bool(edges[j, i]), loose[cell].label == result.label)
        groups.setdefault(key, []).append(cell)
    draws = [list(rng.permutation(len(group))) for group in groups.values()]
    coords = []
    for index in range(max(map(len, draws))):
        for group, order in zip(groups.values(), draws):
            if index < len(order):
                coords.append(group[order[index]])
            if len(coords) >= limit:
                break
        if len(coords) >= limit:
            break
    return coords


def cpu_audit(study, loose, tight, tight_grid, limit, workers, baseline=None):
    coords = ([audit_cell(row) for row in baseline.audit_rows] if baseline else
              stratified_coords(loose, tight, tight_grid, limit))
    tasks = [(ident, cell_point(cell, study.fine), study.horizon) for ident, cell in enumerate(coords)]
    if workers == 1:
        rows = list(map(compare_point, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(compare_point, tasks))
    stable = matches = 0
    audit_rows = []
    for ident, cpu, cpu_tight, t1, t2 in rows:
        i, j = coords[ident]
        gpu = tight[i, j]
        if cpu == cpu_tight and cpu_tight != 5:
            stable += 1
            matches += cpu_tight == gpu.label and not flagged(gpu)
        if cpu != cpu_tight or cpu_tight == 5 or cpu_tight != gpu.label:
            tight_grid[j, i] = 5
        audit_rows.append((i, j, loose[i, j].label, gpu.label, gpu.time, cpu, cpu_tight, t1, t2))
    write_csv(study.out / 'cpu_audit.csv', ('i', 'j', 'gpu_standard', 'gpu_tight', 'gpu_time',
              'cpu', 'cpu_tight', 'cpu_time', 'cpu_tight_time'), audit_rows)
    print(f'CPU audit: {matches}/{stable} stable-reference matches across {len(rows)} cases', flush=True)
    stats = {'cases': len(rows), 'stable': stable, 'matches': matches,
             'agreement': matches/stable if stable else None}
    if baseline:
        stats['paired'] = paired_audit(study, baseline, loose, tight, audit_rows)
    return stats


def interior_probes(study, adaptive):
    rng = np.random.default_rng(92725)
    n = study.base
    xmin, xmax, ymin, ymax = BOUNDS
    points = {(i, j): (xmin + (i + rng.uniform(0.15, 0.85))*(xmax-xmin)/n,
                       ymin + (j + rng.uniform(0.15, 0.85))*(ymax-ymin)/n) for i, j in cells(n)}
    standard, _ = study.solve('probe_standard', points, STANDARD)
    tight, _ = study.solve('probe_tight', points, TIGHT)
    checks = []
    expected, actual = [], []
    base_grid = adaptive[0]['map']
    edges = boundary_mask(base_grid)
    for (i, j), (x, y) in points.items():
        pos_x = min(study.fine - 1, int(study.fine*(x-xmin)/(xmax-xmin)))
        pos_y = min(study.fine - 1, int(study.fine*(y-ymin)/(ymax-ymin)))
        predicted = int(adaptive[-1]['map'][pos_y, pos_x])
        first, second = standard[i, j], tight[i, j]
        stable = first.label == second.label and not flagged(first) and not flagged(second)
        interior = bool(not edges[j, i] and base_grid[j, i] != 5)
        checks.append((i, j, x, y, predicted, first.label, second.label, int(stable), int(interior)))
        if stable and interior:
            expected.append(second.label)
            actual.append(predicted)
    write_csv(study.out / 'interior_probes.csv', ('i', 'j', 'x', 'y', 'adaptive', 'gpu_standard',
                                               'gpu_tight', 'stable', 'base_interior'), checks)
    info = score(np.asarray(actual), np.asarray(expected), np.ones(len(expected), dtype=bool))
    return {'all_probes': len(points), 'stable_interior': info,
            'missed_region_rate': None if not info['count'] else 1.0 - info['agreement']}


def png_for_grid(grid):
    rgb = np.asarray(COLORS, dtype=np.uint8)[grid[::-1]]
    height, width = grid.shape

    def chunk(kind, data):
        return (struct.pack('!I', len(data)) + kind + data +
                struct.pack('!I', zlib.crc32(kind + data)))

    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(b''.join(b'\x00' + row.tobytes() for row in rgb))) + chunk(b'IEND', b''))


def render_map(path, grids, horizon):
    panel_width, panel_height = 320, 240
    x_left, y_top, spacing = 58, 101, 80
    width, height = 1260, 464
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
             '<title id="title">Planar three-body outcome map</title>',
             '<desc id="desc">Outcome by initial location of the third body, comparing a fine '
             'reference with two sampling methods.</desc>',
             '<rect width="100%" height="100%" fill="#f5f7fa"/>',
             '<g font-family="Segoe UI, Arial, sans-serif" fill="#1f2937">',
             '<text x="58" y="36" font-size="27" font-weight="700">Planar three-body outcome map</text>',
             f'<text x="58" y="61" font-size="15" fill="#435169">Equal masses released from rest · '
             f'outcomes observed through t = {horizon:g}</text>']
    for panel, (title, grid) in enumerate(grids):
        x = x_left + panel*(panel_width + spacing)
        picture = base64.b64encode(png_for_grid(grid)).decode('ascii')
        parts.extend([
            f'<text x="{x}" y="90" font-size="17" font-weight="600">{escape(title)}</text>',
            f'<image x="{x}" y="{y_top}" width="{panel_width}" height="{panel_height}" '
            f'preserveAspectRatio="none" style="image-rendering:pixelated" href="data:image/png;base64,{picture}"/>',
            f'<rect x="{x}" y="{y_top}" width="{panel_width}" height="{panel_height}" '
            'fill="none" stroke="#485468"/>',
            f'<text x="{x}" y="361" font-size="13">0.10</text>',
            f'<text x="{x+panel_width}" y="361" font-size="13" text-anchor="end">0.90</text>',
            f'<text x="{x+panel_width/2}" y="384" font-size="14" text-anchor="middle">Initial x</text>',
            f'<text x="{x-6}" y="{y_top+11}" font-size="13" text-anchor="end">0.65</text>',
            f'<text x="{x-6}" y="{y_top+panel_height}" font-size="13" text-anchor="end">0.05</text>',
            f'<text x="{x-43}" y="221" font-size="14" text-anchor="middle" '
            f'transform="rotate(-90 {x-43} 221)">Initial y</text>',
        ])
    for idx, (label, (r, g, b)) in enumerate(zip(LABELS, COLORS)):
        x = x_left + 192*idx
        parts.extend([f'<rect x="{x}" y="413" width="15" height="15" rx="2" fill="#{r:02x}{g:02x}{b:02x}"/>',
                      f'<text x="{x+22}" y="426" font-size="14">{escape(label)}</text>'])
    path.write_text('\n'.join([*parts, '</g></svg>']), encoding='utf-8')


def jovin_evidence(directory):
    project = json.loads((directory / 'project.json').read_text(encoding='utf-8-sig'))
    chain = json.loads((directory / 'chain.json').read_text(encoding='utf-8-sig'))
    scope = json.loads((directory / 'scope.json').read_text(encoding='utf-8-sig'))
    if not project['passed'] or not chain['valid']:
        raise RuntimeError('A passing Jovin CPU project check and a valid receipt chain are required.')
    for record in scope['files']:
        if digest(ROOT / record['study_path']) != record['sha256']:
            raise RuntimeError(f"The CPU verification scope changed: {record['study_path']}")
    return {**{key: project[key] for key in ('receipt_id', 'native_receipt_id', 'ast_receipt_id',
                                           'trace_receipt_id', 'source_manifest_sha256')},
            'chain_head': chain['head_sha256'], 'chain_receipts': chain['count'],
            'evidence_sha256': {name: digest(directory / name) for name in
                                ('baseline.json', 'ast.json', 'trace.json', 'project.json', 'chain.json', 'scope.json')}}


def get_profile(study):
    profiler = shutil.which('ncu')
    if not profiler:
        return {'status': 'profiler_missing'}
    report = study.raw / 'nsight_tensor.csv'
    try:
        run_command([profiler, '--csv', '--log-file', report, '--metrics',
                     'sm__inst_executed_pipe_tensor.sum', '--launch-count', '9',
                     study.binaries / 'filter.exe', study.raw / 'full_filter.in.csv',
                     study.raw / 'profile_mask.csv', '--backend', 'tensor', '--repeat', '1',
                     '--device', study.device], study.raw / 'nsight.log', timeout=90)
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        detail = str(error)
    else:
        detail = ''
    captured = report.read_text(encoding='utf-8', errors='replace') if report.exists() else ''
    if 'ERR_NVGPUCTRPERM' in captured:
        return {'status': 'counter_permission_denied', 'report': report.name,
                'systems': profile_dispatch(study)}
    if 'sm__inst_executed_pipe_tensor.sum' in captured and '==ERROR==' not in captured and not detail:
        return {'status': 'captured', 'report': report.name}
    return {'status': 'unavailable', 'report': report.name if report.exists() else None, 'detail': detail}


def profile_dispatch(study):
    executable = shutil.which('nsys')
    if not executable:
        install = Path(os.environ.get('ProgramFiles', 'C:/Program Files')) / 'NVIDIA Corporation'
        versions = sorted(install.glob('Nsight Systems*/target-windows-x64/nsys.exe'))
        if not versions:
            return {'status': 'profiler_missing'}
        executable = versions[-1]
    trace_prefix = study.raw / 'tensor_trace'
    stats_prefix = study.raw / 'tensor_kernels'
    try:
        run_command([executable, 'profile', '--trace=cuda,cublas', '--sample=none', '--cpuctxsw=none',
                     '--stats=false', '--force-overwrite=true', f'--output={trace_prefix}',
                     study.binaries / 'filter.exe', study.raw / 'full_filter.in.csv',
                     study.raw / 'nsys_mask.csv', '--backend', 'tensor', '--repeat', '3',
                     '--device', study.device], study.raw / 'nsys_profile.log', timeout=120)
        run_command([executable, 'stats', '--report', 'cuda_gpu_kern_sum', '--format', 'csv',
                     '--force-export=true', '--force-overwrite=true', '--output', stats_prefix,
                     trace_prefix.with_suffix('.nsys-rep')], study.raw / 'nsys_stats.log', timeout=120)
        csv_path = study.raw / 'tensor_kernels_cuda_gpu_kern_sum.csv'
        kernels = read_csv(csv_path)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        return {'status': 'trace_unavailable', 'detail': str(error)}
    matched = [row for row in kernels if 'tensorop' in row.get('Name', '').lower() or
               'tensor_op' in row.get('Name', '').lower()]
    return {'status': 'tensor_kernel_observed' if matched else 'trace_captured',
            'trace': trace_prefix.with_suffix('.nsys-rep').name, 'kernel_summary': csv_path.name,
            'tensor_kernels': matched}


def report_text(summary):
    comparison = summary['comparison'][-1]
    adaptive, uniform = comparison['adaptive'], comparison['uniform']
    audit = summary['cpu_audit']
    tensor = summary['filter']
    probes = summary['interior_probes']
    decision = summary['decision']
    profile = summary['tensor_profile']
    dispatch = profile.get('systems', {}).get('status')

    def pct(value):
        return 'n/a' if value is None else f'{100*value:.2f}%'

    regularized = summary['coordinates'] == 'levi-civita'
    title = 'Levi-Civita three-body study' if regularized else 'Three-body projection and refinement'
    rows = [f'# {title}', '',
            f"The {tensor['device_name']} study sampled a {summary['fine_grid']} × {summary['fine_grid']} initial-condition "
            f"plane for equal masses released from rest, with a horizon of t = {summary['horizon']:g}.", '',
            f"Jovin CPU projection check `{summary['jovin']['receipt_id']}` passed. The GPU short-time "
            f"state check had a maximum normalized error of {summary['gpu_state_gate']['max_normalized_error']:.3g}. "
            f"On the long-horizon CPU audit, {audit['matches']} of {audit['stable']} stable CPU classifications "
            f"matched the tight GPU run across {audit['cases']} checked cases.", '',
            f"The full-grid reference retained {pct(summary['reference_stable_fraction'])} of cells after "
            'tolerance and reference checks. Stable here means the two GPU tolerance settings agreed on the label '
            'and passed the energy drift screen, with sampled discrepancies from the CPU audit excluded. '
            'Unresolved predictions count against agreement on stable reference cells.', '',
            '| Sampling | Force evaluations | Agreement | Boundary agreement | Coverage | Wall time |',
            '|---|---:|---:|---:|---:|---:|',
            f"| Adaptive | {adaptive['evaluations']:,} | {pct(adaptive['all']['agreement'])} | "
            f"{pct(adaptive['boundary']['agreement'])} | {pct(adaptive['all']['coverage'])} | "
            f"{summary['adaptive_seconds']:.2f} s |",
            f"| Uniform at matched budget | {uniform['evaluations']:,} | {pct(uniform['all']['agreement'])} | "
            f"{pct(uniform['boundary']['agreement'])} | {pct(uniform['all']['coverage'])} | "
            f"{summary['uniform_matched_seconds']:.2f} s |", '',
            f"Adaptive disagreed on {pct(probes['missed_region_rate'])} of {probes['stable_interior']['count']} "
            'stable probes in coarse-map interior cells. The preregistered work-savings gate was '
            f"**{decision['work_savings_gate']}**.", '',
            f"The real {tensor['rows']:,}-neighborhood filter produced identical masks. Median time was "
            f"{tensor['fused_ms']['median']:.4f} ms for fused CUDA and {tensor['tensor_ms']['median']:.4f} ms "
            f"for TF32 cuBLAS. Nsight counter status was `{profile['status']}`." +
            (f" Nsight Systems status was `{dispatch}`." if dispatch else ''), '',
            'The escape labels are finite-threshold observations. The close-approach label stops at '
            'pair separation 0.001. Gray cells are unresolved at the tested accuracy.', '',
            '[Outcome map](outcome_map.svg) · [Detailed results](summary.json) · '
            '[CPU cross-checks](cpu_audit.csv) · [Blind probes](interior_probes.csv)', '']
    if regularized:
        boundary = adaptive['boundary']
        frozen = summary['frozen_baseline_scores']
        old, new = frozen['original_adaptive'], frozen['new_adaptive']
        paired = audit['paired']
        differences = paired['original_disagreements']
        recovered = sum(item['gpu_match'] for item in differences)
        label = lambda passed: 'passed' if passed else 'failed'
        rows.extend([
            '## Validation against the original targets', '',
            f"- Boundary gate **{label(decision['boundary_gate'])}** with {boundary['matches']:,} of "
            f"{boundary['count']:,} labels agreeing ({pct(boundary['agreement'])}, target 90%).",
            f"- CPU gate **{label(decision['cpu_audit_gate'])}** with {audit['matches']} of {audit['stable']} "
            f"stable labels agreeing ({pct(audit['agreement'])}, target 95% with at least {audit['cases']//2} stable).",
            f"- On the frozen original boundary, the new adaptive map agreed on {new['matches']:,} of "
            f"{new['count']:,} points ({pct(new['agreement'])}). The archived map agreed on {old['matches']:,} "
            f"of {old['count']:,} points ({pct(old['agreement'])}).",
            f"- On the original {len(differences)} CPU disagreements, {recovered} now agree. The paired audit "
            f"replayed the same {audit['cases']} coordinates and recorded {paired['reference_labels_changed_on_replay']} "
            'changes to the two CPU labels on replay.', '',
            '| Initial-condition cell | CPU outcome | Original GPU outcome | Levi-Civita GPU outcome |',
            '|---|---|---|---|',
            *(f"| ({row['i']}, {row['j']}) | {LABELS[row['cpu_label']]} | {LABELS[row['old_gpu_label']]} | "
              f"{LABELS[row['new_gpu_label']]} |" for row in differences), '',
            'The separate analytic collision fixture crossed the isolated binary in regularized time. '
            'The scored trajectories use the original physical-time stop at separation 0.001.', '',
            '[Paired audit with event times and separations](paired_audit.csv) · '
            '[CUDA collision fixture](raw/gpu_lc_crossing.log)', '',
        ])
    return '\n'.join(rows)


def execute(args):
    base, horizon, sample_count = (8, 8.0, 24) if args.smoke else (32, 50.0, 96)
    study = Study(args.out.resolve(), args.bin.resolve(), args.device, base, horizon, args.coordinates)
    source_paths = [ROOT / name for name in ('study.py', 'reference.py', 'build_cuda.ps1',
                   'cuda/solver.cu', 'cuda/filter.cu', '.jovin/run_cpu.ps1',
                   'cuda/levi_civita.hpp', 'lc_checks.py')]
    source_paths += sorted((ROOT / 'qa').glob('*')) + [ROOT / '.jovin/project.json']
    source_hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in source_paths if path.is_file()}
    binary_hashes = {name: digest(study.binaries / name) for name in ('solver.exe', 'filter.exe')}
    evidence = jovin_evidence(args.jovin.resolve())
    gpu, _ = run_command(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total',
                          '--format=csv,noheader'], study.raw / 'hardware.log')
    config = {'base': base, 'levels': 3, 'horizon': horizon, 'domain': BOUNDS, 'standard': STANDARD,
              'tight': TIGHT, 'device_index': args.device, 'gpus': gpu.splitlines(), 'cpu_sample': sample_count,
              'target': {'agreement': 0.98, 'boundary_agreement': 0.90, 'coverage': 0.95},
              'cpu_gate': {'minimum_agreement': 0.95, 'minimum_stable_cases': sample_count//2},
              'coordinates': args.coordinates, 'close_separation': 0.001,
              'source_sha256': source_hashes, 'binary_sha256': binary_hashes, 'jovin': evidence}
    baseline = load_baseline(args.baseline.resolve(), config) if args.baseline else None
    if baseline:
        config['baseline'] = {'path': str(baseline.directory), 'sha256': baseline.hashes}
    if args.coordinates == 'levi-civita':
        config['chart'] = {'entry_radius': 0.05, 'entry_hierarchy': 4.0,
                           'exit_radius': 0.075, 'exit_hierarchy': 2.0}
    write_json(study.out / 'manifest.json', config)
    gpu_gate = study.gate_gpu()
    lc_gate = study.gate_lc() if args.coordinates == 'levi-civita' else None
    initial, base_seconds = study.solve_cells('uniform_0', cells(base), base)
    adaptive = study.adaptive(initial)
    uniform = [initial]
    uniform_seconds = [base_seconds]
    for level in (1, 2, 3):
        n = base * 2**level
        result, elapsed = study.solve_cells(f'uniform_{level}', cells(n), n)
        uniform.append(result)
        uniform_seconds.append(uniform_seconds[-1] + elapsed)
    tight, _ = study.solve_cells('reference_tight', cells(study.fine), study.fine, TIGHT)
    tight_grid = labels_array(tight, study.fine)
    audit = cpu_audit(study, uniform[-1], tight, tight_grid, sample_count, args.workers, baseline)
    loose_grid = labels_array(uniform[-1], study.fine)
    tight_grid[tight_grid != loose_grid] = 5
    eligible = tight_grid != 5
    boundary = boundary_mask(tight_grid)
    comparisons = []
    uniform_summary = []
    uniform_spent = 0
    for level, (refined, uniform_results) in enumerate(zip(adaptive, uniform)):
        adaptive_map = upsample(refined['map'], study.fine//len(refined['map']))
        matched, spent, total, batches = uniform_at_budget(uniform, base, refined['evaluations'])
        comparisons.append({'depth': level, 'adaptive': {
            'evaluations': refined['evaluations'], 'trajectories': refined['trajectories'],
            'all': score(adaptive_map, tight_grid, eligible), 'boundary': score(adaptive_map, tight_grid, boundary)},
            'uniform': {'evaluations': spent, 'trajectories': total, 'all': score(matched, tight_grid, eligible),
                        'boundary': score(matched, tight_grid, boundary)}})
        uniform_grid = labels_array(uniform_results, base*2**level)
        uniform_grid = upsample(uniform_grid, study.fine//len(uniform_grid))
        uniform_spent += work(uniform_results)
        uniform_summary.append({'depth': level, 'evaluations': uniform_spent, 'seconds': uniform_seconds[level],
                                'all': score(uniform_grid, tight_grid, eligible),
                                'boundary': score(uniform_grid, tight_grid, boundary)})

    replay_start = time.perf_counter()
    for level, (n, selected, full) in enumerate(batches, start=1):
        replay, _ = study.solve_cells(f'uniform_matched_{level}', selected, n)
        if any((result.label, result.evals) != (full[cell].label, full[cell].evals)
               for cell, result in replay.items()):
            raise RuntimeError('Matched-budget uniform replay differed from the corresponding full-grid run.')
    replay_seconds = base_seconds + time.perf_counter() - replay_start
    probes = interior_probes(study, adaptive)
    _, filter_stats, _ = study.filter('full_filter', dict(zip(cells(study.fine), tight_grid.ravel().tolist())),
                                     cells(study.fine), repeat=9, backend='both')
    profile = get_profile(study)
    reference_title = 'Levi-Civita reference' if lc_gate else 'Resolution-checked reference'
    render_map(study.out / 'outcome_map.svg', ((reference_title, tight_grid),
               ('Adaptive refinement', adaptive[-1]['map']), ('Uniform at matched work', matched)), horizon)
    np.savez_compressed(study.out / 'outcome_grids.npz', reference=tight_grid, adaptive=adaptive[-1]['map'],
                        uniform_matched=matched, bounds=np.array(BOUNDS))

    def reaches_target(metrics):
        target = config['target']
        return ((metrics['all']['agreement'] or 0) >= target['agreement'] and
                (metrics['boundary']['agreement'] or 0) >= target['boundary_agreement'] and
                (metrics['all']['coverage'] or 0) >= target['coverage'])

    adapt_min = min((row['adaptive']['evaluations'] for row in comparisons if reaches_target(row['adaptive'])),
                    default=None)
    uniform_min = min((row['evaluations'] for row in [*uniform_summary, *(item['uniform'] for item in comparisons)]
                       if reaches_target(row)), default=None)
    audit_pass = (audit['stable'] >= config['cpu_gate']['minimum_stable_cases'] and
                  (audit['agreement'] or 0) >= config['cpu_gate']['minimum_agreement'])
    boundary_pass = ((comparisons[-1]['adaptive']['boundary']['agreement'] or 0) >=
                     config['target']['boundary_agreement'])
    decision = 'met' if audit_pass and adapt_min and uniform_min and adapt_min < uniform_min else 'not met'
    for path, expected in source_hashes.items():
        if digest(ROOT / path) != expected:
            raise RuntimeError(f'Study source changed during the run: {path}')
    for name, expected in binary_hashes.items():
        if digest(study.binaries / name) != expected:
            raise RuntimeError(f'The CUDA binary changed during the run: {name}')
    summary = {'manifest_sha256': digest(study.out / 'manifest.json'), 'jovin': evidence,
               'coordinates': args.coordinates,
               'horizon': horizon, 'fine_grid': study.fine, 'gpu_state_gate': gpu_gate, 'cpu_audit': audit,
               'reference_stable_fraction': float(np.mean(eligible)),
               'reference_labels': {label: int(np.count_nonzero(tight_grid == code))
                                    for code, label in enumerate(LABELS)},
               'comparison': comparisons, 'uniform_complete_levels': uniform_summary,
               'adaptive_seconds': adaptive[-1]['seconds'] + base_seconds, 'uniform_matched_seconds': replay_seconds,
               'interior_probes': probes, 'filter': filter_stats, 'tensor_profile': profile,
               'decision': {'work_savings_gate': decision, 'cpu_audit_gate': audit_pass,
                            'boundary_gate': boundary_pass, 'both_validation_gates': audit_pass and boundary_pass,
                            'adaptive_target_evaluations': adapt_min, 'uniform_target_evaluations': uniform_min}}
    if baseline:
        for name, expected in baseline.hashes.items():
            if digest(baseline.directory / name) != expected:
                raise RuntimeError(f'The archived baseline changed during the run: {name}')
        summary['frozen_baseline_scores'] = frozen_scores(baseline, adaptive[-1]['map'], matched, tight_grid)
    if lc_gate:
        summary['gpu_lc_gate'] = lc_gate
        summary['chart_use'] = {name: {
            'trajectories_entering': sum(result.chart_entries > 0 for result in run.values()),
            'entries': sum(result.chart_entries for result in run.values()),
            'exits': sum(result.chart_exits for result in run.values()),
            'max_lc_residual': max((result.max_lc_residual for result in run.values()
                                    if np.isfinite(result.max_lc_residual)), default=None)}
            for name, run in (('standard_grid', uniform[-1]), ('tight_grid', tight))}
    write_json(study.out / 'summary.json', summary)
    (study.out / 'report.md').write_text(report_text(summary), encoding='utf-8')
    write_json(study.out / 'evidence_hashes.json', {p.relative_to(study.out).as_posix(): digest(p)
               for p in sorted(study.out.rglob('*')) if p.is_file()})
    print(f'Finished. Report at {study.out / "report.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New directory for evidence and maps')
    parser.add_argument('--bin', type=Path)
    parser.add_argument('--jovin', type=Path)
    parser.add_argument('--coordinates', choices=('edge', 'levi-civita'), default='edge')
    parser.add_argument('--baseline', type=Path, help='Archived edge study used for the paired Levi-Civita scores')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--workers', type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument('--smoke', action='store_true', help='Use a smaller plane and shorter horizon')
    args = parser.parse_args()
    if args.out.exists() or args.workers < 1:
        parser.error('--out must be new and --workers must be positive')
    is_lc = args.coordinates == 'levi-civita'
    args.bin = args.bin or ROOT / ('out/bin-lc' if is_lc else 'out/bin')
    current_jovin = ROOT / 'out/jovin-lc'
    args.jovin = args.jovin or (current_jovin if is_lc or current_jovin.is_dir() else ROOT / 'out/jovin')
    if is_lc:
        args.baseline = args.baseline or ROOT / ('out/study-smoke' if args.smoke else 'out/study')
    elif args.baseline:
        parser.error('--baseline requires --coordinates levi-civita')
    execute(args)


if __name__ == '__main__':
    main()

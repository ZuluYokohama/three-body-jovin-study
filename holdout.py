"""Freeze and run the unseen three-grid Levi-Civita confirmation."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys
import time

import numpy as np

import study as core
import holdout_gpu as gpu
from boundary_refine import POLICY
from precision_audit import TIERS
from refined_study import read_json

ROOT = Path(__file__).resolve().parent
PILOT = ROOT
BASE, FINE = 32, 256
HORIZON, CUTOFF = 50.0, 0.001
EXTRA_WORK = 641_438_440
COHORT_SEED, INTERVAL_SEED = 20261119, 20261120
PAIR_COUNT = 7
SOURCES = (
    'holdout.py', 'holdout_gpu.py', 'holdout_oracle.cpp', 'holdout_oracle.py',
    'verify_holdout.ps1', 'study.py', 'boundary_refine.py', 'reference.py', 'refined_study.py',
    'precision_audit.py', 'lc_checks.py', 'cuda/solver.cu', 'cuda/levi_civita.hpp',
    'cuda/filter.cu', 'qa/verify.cpp', 'qa/CMakeLists.txt', 'qa/native_ast.json',
    'qa/native_plan.json', 'qa/native_trace.json', '.jovin/project.json',
)


def key(row):
    return int(row['grid']), int(row['i']), int(row['j'])


def file_hashes(root, names):
    return {name: core.digest(root / name) for name in names}


def archive_inputs(pilot):
    """Read coordinates only. Prior scores and labels never enter the run adapters."""
    known = set()
    hashes = {}
    for name in ('study', 'study-smoke', 'study-lc', 'study-lc-smoke',
                 'study-lc-refined', 'precision-audit-refined', 'lc-cuda-smoke'):
        folder = pilot / 'out' / name
        for path in sorted(folder.rglob('*.in.csv')):
            rows = core.read_csv(path)
            if rows and {'x', 'y'} <= rows[0].keys():
                hashes[path.relative_to(pilot).as_posix()] = core.digest(path)
                known.update((float(row['x']), float(row['y'])) for row in rows)
        # Some validation runs use pairs.csv or have initial coordinates in the audit itself.
        for path in sorted(folder.glob('*.csv')):
            rows = core.read_csv(path)
            if rows and {'x', 'y'} <= rows[0].keys():
                hashes[path.relative_to(pilot).as_posix()] = core.digest(path)
                known.update((float(row['x']), float(row['y'])) for row in rows)
    for name in ('lc_checks.py', 'study.py'):
        hashes[name] = core.digest(pilot / name)
    from lc_checks import FIXTURES
    known.update(point for point, _ in FIXTURES)
    known.update(((0.17, 0.48), (0.41, 0.62), (0.56, 0.34), (0.81, 0.51)))
    return known, hashes


def cardinal(i, j):
    return [(a, b) for a, b in ((i-1, j), (i+1, j), (i, j-1), (i, j+1))
            if 0 <= a < FINE and 0 <= b < FINE]


def build_cohort(reference):
    mask = core.boundary_mask(reference)
    strata = (np.flatnonzero(mask), np.flatnonzero(~mask))
    rng = np.random.Generator(np.random.PCG64(COHORT_SEED))
    centers, oracle_points = [], {}
    for number, shift in enumerate(gpu.SHIFTS, 1):
        for stratum, population, count in zip(('pilot_boundary', 'pilot_other'), strata, (96, 32)):
            for index in sorted(rng.choice(population, count, replace=False).tolist()):
                i, j = index % FINE, index // FINE
                point = gpu.physical_point((i, j), FINE, shift)
                centers.append((number, i, j, *point, stratum, len(population), count))
                for cell in ((i, j), *cardinal(i, j)):
                    oracle_points[number, *cell] = gpu.physical_point(cell, FINE, shift)
    return centers, [(n, i, j, *oracle_points[n,i,j]) for n,i,j in sorted(oracle_points)]


def prepare(args):
    pilot = args.pilot.resolve()
    target = args.out.resolve()
    if target.exists():
        raise ValueError('The holdout output directory must be new.')
    archives = pilot / 'out'
    lc_archive = archives / 'study-lc'
    refined = archives / 'study-lc-refined'
    other, archive_hashes = archive_inputs(pilot)
    existing_count = len(other)
    visited = set(other)
    for shift in gpu.SHIFTS:
        for side in (32, 64, 128, 256):
            for cell in core.cells(side):
                point = gpu.physical_point(cell, side, shift)
                x, y = point
                if not (core.BOUNDS[0] < x < core.BOUNDS[1] and core.BOUNDS[2] < y < core.BOUNDS[3]):
                    raise RuntimeError(f'Holdout coordinate escaped the original domain: {point}')
                if point in visited:
                    raise RuntimeError(f'Duplicate archived or holdout initial condition: {point}')
                visited.add(point)
    with np.load(lc_archive / 'outcome_grids.npz', allow_pickle=False) as data:
        centers, oracle_points = build_cohort(data['reference'])
    frozen = read_json(refined / 'manifest.json')
    sources = file_hashes(ROOT, SOURCES)
    for file in ('study.py', 'boundary_refine.py', 'precision_audit.py', 'cuda/solver.cu', 'cuda/levi_civita.hpp'):
        if frozen['source_sha256'][file] != sources[file]:
            raise RuntimeError(f'The frozen method differs from the refined pilot: {file}')
    bins = file_hashes(args.bin.resolve(), ('solver.exe', 'filter.exe'))
    if frozen['binary_sha256'] != bins:
        raise RuntimeError('The GPU binaries differ from the refined pilot.')
    zero_shift = gpu.validate_zero_shift(lc_archive, refined)
    target.mkdir(parents=True)
    core.write_csv(target/'scoring_centers.csv', ('grid','i','j','x','y','stratum','population','sampled'),
                   centers)
    core.write_csv(target/'cohort.csv', ('grid','i','j','x','y'), oracle_points)
    core.run_command([sys.executable,'-B',ROOT/'holdout_oracle.py','--cohort',target/'cohort.csv',
                      '--out',target/'oracle','--jobs',str(args.oracle_jobs),'--prepare'],
                     target/'oracle_prepare.log',timeout=180)
    protocol = {
        'experiment': 'three previously unevaluated equal-mass free-fall grid shifts',
        'grid': FINE, 'base': BASE, 'domain': core.BOUNDS, 'shifts_in_fine_cell_widths': gpu.SHIFTS,
        'horizon': HORIZON, 'close_separation': CUTOFF, 'force_budget_extra': EXTRA_WORK,
        'archived_initial_conditions': existing_count, 'unseen_candidates_verified': len(visited)-existing_count,
        'seed': COHORT_SEED, 'weights': 'population/sample count per grid and stratum',
        'centers': len(centers), 'oracle_unique_points': len(oracle_points),
        'rules': {'boundary': .90, 'cpu': .95, 'minimum_boundaries_per_grid': 24,
                  'minimum_stable_per_grid': 48, 'max_relative_work_gap': .01,
                  'reference_drift_limit': 1e-12, 'oracle_seconds_per_solve': 45},
        'oracle': {'arithmetic': 'IEEE binary128', 'tolerances': ['1e-20','1e-24'],
                   'step_cap_factors': [1, .5], 'independent': 'direct body-coordinate extrapolation'},
        'gpu': {'device': args.device, 'coordinates': 'levi-civita', 'standard': core.STANDARD,
                'strict': list(TIERS[-2:]), 'energy_drift_limit': 1e-5,
                'uniform_seed_base': 10919, 'policy': POLICY},
        'timing': {'warmups_per_arm': 1, 'paired_repetitions_per_grid': PAIR_COUNT,
                   'pair_order': 'alternating, first arm varies by grid',
                   'interval_seed': INTERVAL_SEED, 'interval_resamples': 10000,
                   'spatial_block_side': 32},
        'source_sha256': sources, 'binary_sha256': bins,
        'cohort_sha256': file_hashes(target, ('cohort.csv', 'scoring_centers.csv')),
        'oracle_preparation_sha256': file_hashes(target/'oracle', ('manifest.json','holdout_oracle')),
        'pilot_sha256': file_hashes(pilot, ('out/study-lc/outcome_grids.npz',
                                          'out/study-lc-refined/manifest.json',
                                          'out/study-lc-refined/outcome_grids.npz')),
        'archive_input_sha256': archive_hashes,
        'zero_shift_replay': zero_shift,
    }
    core.write_json(target/'protocol.json', protocol)
    print(f"Protocol frozen: {len(centers)} scoring centers, {len(oracle_points)} distinct oracle points", flush=True)


def check_freeze(args):
    target = args.out.resolve()
    protocol = read_json(target/'protocol.json')
    for directory, items in ((ROOT, protocol['source_sha256']),
        (args.bin.resolve(), protocol['binary_sha256']), (target, protocol['cohort_sha256']),
        (target/'oracle',protocol['oracle_preparation_sha256']),
        (args.pilot.resolve(), protocol['pilot_sha256'])):
        if file_hashes(directory, items) != items:
            raise RuntimeError(f'Frozen input changed in {directory}.')
    if protocol['gpu']['device'] != args.device:
        raise RuntimeError('The CUDA device index differs from the frozen one.')
    jovin = args.jovin.resolve()
    check = read_json(jovin/'project.json')
    chain = read_json(jovin/'chain.json')
    entry = next((file for file in check['snapshot']['files'] if file['path'] == 'protocol.json'), None)
    if not check['passed'] or not chain['valid'] or not entry or entry['sha256'] != core.digest(target/'protocol.json'):
        raise RuntimeError('The held-out protocol lacks a matching verified Jovin receipt.')
    return protocol, check, chain


def run_gpu(args):
    check_freeze(args)
    target = args.out.resolve()
    for number in (args.grid,) if args.grid else range(1,4):
        directory = target/f'grid_{number}'
        if directory.exists():
            raise ValueError(f'Use a fresh grid directory: {directory}')
        shift = gpu.SHIFTS[number-1]
        print(f'Grid {number}: calibration, warmups, and {PAIR_COUNT} live pairs', flush=True)
        adaptive, uniform, metrics = gpu.run_grid(directory,args.bin.resolve(),args.device,shift,
                                                 pairs=PAIR_COUNT,extra_budget=EXTRA_WORK)
        if np.shape(adaptive) != (FINE,FINE) or np.shape(uniform) != (FINE,FINE):
            raise RuntimeError('A holdout arm returned the wrong lattice shape.')
        np.savez_compressed(directory/'outcome_grids.npz', adaptive=adaptive, uniform=uniform)
        core.write_json(directory/'timing_summary.json', metrics)


def run_strict(args):
    check_freeze(args)
    target = args.out.resolve()
    if not all((target/f'grid_{number}/timing_summary.json').is_file() for number in range(1,4)):
        raise RuntimeError('Finish all live timing runs before strict validation.')
    source = core.read_csv(target/'scoring_centers.csv')
    run = core.Study(target/'strict_gpu', args.bin.resolve(), args.device, BASE, HORIZON, 'levi-civita')
    rows = []
    for number in range(1,4):
        points = {key(row)[1:]: (float(row['x']),float(row['y'])) for row in source if key(row)[0]==number}
        (first, second) = [run.solve(f'grid_{number}_{name}',points,settings)[0] for name,settings in TIERS[-2:]]
        for (i,j), (x,y) in points.items():
            a,b=first[i,j],second[i,j]
            label = a.label if a.label==b.label and not core.flagged(a) and not core.flagged(b) else 5
            rows.append((number,i,j,x,y,label,a.label,b.label,a.evals,b.evals,a.time,b.time,
                         a.energy_error,b.energy_error))
    core.write_csv(target/'strict_gpu_predictions.csv',
                   ('grid','i','j','x','y','label','p13_label','p14_label','p13_evals','p14_evals',
                    'p13_time','p14_time','p13_energy','p14_energy'), rows)


def run_reference(args):
    check_freeze(args)
    target=args.out.resolve()
    if not (target/'strict_gpu_predictions.csv').is_file():
        raise RuntimeError('Freeze the strict GPU predictions before starting the CPU reference.')
    core.run_command([sys.executable,'-B',ROOT/'holdout_oracle.py','--cohort',target/'cohort.csv',
                      '--out',target/'oracle','--jobs',str(args.oracle_jobs),'--resume'], target/'reference.log',
                      timeout=6*60*60+10*60)




def exact_rows(path, expected):
    data = {}
    for row in core.read_csv(path):
        index = key(row)
        if index in data or index not in expected:
            raise ValueError(f'Duplicate or unexpected coordinate {index} in {path}')
        if (float(row['x']), float(row['y'])) != expected[index]:
            raise ValueError(f'Coordinates changed for {index} in {path}')
        data[index] = row
    if data.keys() != expected.keys():
        raise ValueError(f'Incomplete coordinate set in {path}')
    return data


def weighted(rows, field, domain):
    selected = [row for row in rows if row[domain]]
    total = sum(row['weight'] for row in selected)
    successes = sum(row[field] for row in selected)
    value = sum(row['weight']*row[field] for row in selected)/total if total else None
    return {'agreement': value, 'cases': len(selected), 'matches': successes}


def spatial_interval(rows):
    rng = np.random.Generator(np.random.PCG64(INTERVAL_SEED))
    numerator = np.zeros(10_000)
    denominator = np.zeros(10_000)
    for grid in range(1,4):
        blocks = defaultdict(lambda: [0.0,0.0])
        for row in rows:
            if row['grid'] == grid:
                block = blocks[row['i']//32, row['j']//32]
                if row['boundary']:
                    block[0] += row['weight'] * (row['adaptive_match']-row['uniform_match'])
                    block[1] += row['weight']
        if not blocks:
            continue
        values = np.asarray(list(blocks.values()))
        draws = values[rng.integers(0,len(values),size=(10_000,len(values)))].sum(axis=1)
        numerator += draws[:,0]
        denominator += draws[:,1]
    valid = denominator > 0
    if not np.any(valid):
        return None
    return np.percentile(numerator[valid]/denominator[valid], [2.5,97.5]).tolist()


def timing_summary(grids):
    ratios=[]
    rows=[]
    valid=True
    for grid, timing in enumerate(grids,1):
        values=[]
        for record in timing.get('pairs', []):
            a,u=record['adaptive'],record['uniform']
            rows.append((grid,record['pair'],record['order'],a['wall_seconds'],u['wall_seconds'],
                         a['solver_seconds'],u['solver_seconds'],a['work'],u['work'],int(record['valid'])))
            if record['valid']:
                values.append(u['wall_seconds']/a['wall_seconds'])
        valid &= timing['status']=='passed' and len(values)==PAIR_COUNT
        ratios.append(np.asarray(values))
    if not valid:
        return {'status':'inconclusive', 'uniform_over_adaptive': None,'ci': None,'pairs':len(rows)}, rows
    rng=np.random.Generator(np.random.PCG64(INTERVAL_SEED))
    samples=np.column_stack([values[rng.integers(0,len(values),size=(10_000,len(values)))] for values in ratios])
    a_times=np.array([row[3] for row in rows])
    u_times=np.array([row[4] for row in rows])
    return {'status':'measured','uniform_over_adaptive':float(np.median(np.concatenate(ratios))),
            'ci':np.percentile(np.median(samples, axis=1),[2.5,97.5]).tolist(),'pairs':len(rows),
            'adaptive_median_seconds':float(np.median(a_times)),'uniform_median_seconds':float(np.median(u_times))}, rows


def percent(a):
    return 'inconclusive' if a is None else f'{100*a:.2f}%'


def render_report(payload):
    stats=payload['pooled']
    timing=payload['timing']
    verdict=payload['gates']
    lines=[
        '# Held-out Levi-Civita confirmation','',
        'We reran the same equal-mass free-fall problem on three shifted grids. All sampled starting points were checked against the pilot archives. Each body was followed to physical time 50, a qualifying escape, or the original close-approach distance of 0.001.','',
        '## Accuracy','',
        '| Grid | Adaptive boundary | Uniform boundary | GPU agreement with stable CPU | Stable CPU | Confirmed boundary |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for grid in payload['grids']:
        lines.append(f"| {grid['grid']} | {percent(grid['adaptive_boundary']['agreement'])} | "
                     f"{percent(grid['uniform_boundary']['agreement'])} | {percent(grid['cpu_agreement']['agreement'])} | "
                     f"{grid['cpu_agreement']['cases']} | {grid['adaptive_boundary']['cases']} |")
    lines += [
        '',
        f"The 90% boundary target **{verdict['boundary']}** at {percent(stats['adaptive_boundary']['agreement'])} "
        f"stratum-weighted agreement. Uniform scored {percent(stats['uniform_boundary']['agreement'])}. "
        f"The 95% strict GPU agreement target **{verdict['cpu']}** at {percent(stats['cpu_agreement']['agreement'])}.",
        '',
        f"The CPU reference attempted {payload['oracle']['points']} distinct coordinates. It confirmed "
        f"{payload['oracle']['confirmed']} and left {payload['oracle']['unresolved']} unresolved. Unresolved GPU predictions "
        'count as misses wherever the independent reference is stable. The accuracy estimates use the frozen '
        'sample weights. They describe these grids and the equal-mass free-fall family.',
        '',
    ]
    interval=stats['adaptive_advantage_ci']
    if interval is not None:
        lines += [f"The paired adaptive-minus-uniform 95% spatial-block interval was {100*interval[0]:+.2f} to "
                  f"{100*interval[1]:+.2f} percentage points. The benchmark advantage was **{verdict['advantage']}**.",'']
    lines += ['## Live GPU timing','']
    if timing['status']=='measured':
        lo,hi=timing['ci']
        lines += [
            f"Across {timing['pairs']} paired runs, median wall times were {timing['adaptive_median_seconds']:.2f} s "
            f"for adaptive and {timing['uniform_median_seconds']:.2f} s for uniform. The median paired "
            f"uniform/adaptive ratio was {timing['uniform_over_adaptive']:.3f} (95% interval {lo:.3f}–{hi:.3f}). "
            f"The speedup claim was **{verdict['speedup']}**.",
            '',
        ]
    else:
        lines += ['The matched-work timing comparison was inconclusive. See the per-grid timing status and work gaps.','']
    lines += [
        'The wall clock includes choosing samples, launching GPU programs, transfers, file handling, and building the map. Force evaluations count calls to the dynamics. Two methods can do similar force work and spend different time on those other operations.',
        '',
        'Jovin verified the CPU chart fixtures and the frozen protocol. The independently executed binary128 reference and live CUDA runs carry separate input and output hashes. A label match checks the reported outcome by the specified cutoff and horizon. Event-time differences are recorded separately.',
        '',
        '[Scoring data](scored_centers.csv) · [Live timings](timings.csv) · [Frozen protocol](protocol.json) · [Detailed results](summary.json)',
        '',
    ]
    return '\n'.join(lines)


def score_run(args):
    protocol,jovin,chain=check_freeze(args)
    target=args.out.resolve()
    centers=core.read_csv(target/'scoring_centers.csv')
    oracle_input=core.read_csv(target/'cohort.csv')
    all_points={key(row):(float(row['x']),float(row['y'])) for row in oracle_input}
    center_points={key(row):all_points[key(row)] for row in centers}
    labels=exact_rows(target/'oracle'/'labels.csv',all_points)
    strict=exact_rows(target/'strict_gpu_predictions.csv',center_points)
    maps={}
    grid_timings=[]
    for number in range(1,4):
        directory=target/f'grid_{number}'
        with np.load(directory/'outcome_grids.npz',allow_pickle=False) as data:
            maps[number]=data['adaptive'],data['uniform']
        grid_timings.append(read_json(directory/'timing_summary.json'))
    confirmed={coord:int(row['label']) for coord,row in labels.items() if int(row['label']) in range(5)}
    data=[]
    for row in centers:
        number,i,j=coord=key(row)
        expected=confirmed.get(coord,5)
        neighbors=[confirmed.get((number,a,b),5) for a,b in cardinal(i,j)]
        boundary=expected!=5 and any(label!=expected and label!=5 for label in neighbors)
        unknown=expected==5 or (not boundary and 5 in neighbors)
        adaptive,uniform=maps[number]
        a,u,g=int(adaptive[j,i]),int(uniform[j,i]),int(strict[coord]['label'])
        data.append({'grid':number,'i':i,'j':j,'x':row['x'],'y':row['y'],'stratum':row['stratum'],
                     'weight':int(row['population'])/int(row['sampled']),
                     'reference_label':expected,'stable':int(expected!=5),'boundary':int(boundary),
                     'boundary_unknown':int(unknown),'adaptive_label':a,'uniform_label':u,'strict_gpu_label':g,
                     'adaptive_match':int(expected!=5 and a==expected),
                     'uniform_match':int(expected!=5 and u==expected),'strict_match':int(expected!=5 and g==expected)})
    columns=list(data[0])
    core.write_csv(target/'scored_centers.csv',columns,([row[col] for col in columns] for row in data))

    def accuracy(rows):
        return {'adaptive_boundary':weighted(rows,'adaptive_match','boundary'),
                'uniform_boundary':weighted(rows,'uniform_match','boundary'),
                'cpu_agreement':weighted(rows,'strict_match','stable'),
                'adaptive_all':weighted(rows,'adaptive_match','stable'),
                'uniform_all':weighted(rows,'uniform_match','stable'),
                'unknown_boundary_centers':sum(row['boundary_unknown'] for row in rows),
                'unresolved_cpu_centers':sum(not row['stable'] for row in rows)}

    results=[]
    for number in range(1,4):
        rows=[row for row in data if row['grid']==number]
        results.append({'grid':number, **accuracy(rows),
                        'strata':{name:accuracy([row for row in rows if row['stratum']==name])
                                  for name in ('pilot_boundary','pilot_other')},
                        'timing_status':grid_timings[number-1]['status'],
                        'work_gap':grid_timings[number-1].get('work_gap')})
        reference=np.full((FINE,FINE),5,dtype=np.uint8)
        for (grid,i,j),label in confirmed.items():
            if grid==number:
                reference[j,i]=label
        adaptive,uniform=maps[number]
        core.render_map(target/f'grid_{number}_map.svg',(('Binary128 scoring points',reference),
                          ('Adaptive samples',adaptive),('Uniform at matched work',uniform)),HORIZON)
    pooled=accuracy(data)
    pooled['adaptive_advantage_ci']=spatial_interval(data)
    timing,timing_rows=timing_summary(grid_timings)
    core.write_csv(target/'timings.csv',('grid','pair','order','adaptive_seconds','uniform_seconds',
                   'adaptive_solver_seconds','uniform_solver_seconds','adaptive_evals','uniform_evals','valid'),
                   timing_rows)
    enough_boundary=all(row['adaptive_boundary']['cases']>=24 for row in results)
    enough_stable=all(row['cpu_agreement']['cases']>=48 for row in results)
    matched=all(row['work_gap'] is not None and row['work_gap']<=.01 for row in results)
    interval=pooled['adaptive_advantage_ci']
    gates={
        'boundary':('passed' if pooled['adaptive_boundary']['agreement']>=.90 else 'failed') if enough_boundary else 'inconclusive',
        'cpu':('passed' if pooled['cpu_agreement']['agreement']>=.95 else 'failed') if enough_stable else 'inconclusive',
        'advantage':('supported' if interval[0]>0 else 'inconclusive') if enough_boundary and matched and interval else 'inconclusive',
        'speedup':'supported' if timing['ci'] and timing['ci'][0]>1 else 'inconclusive' if timing['ci'] is None else 'unsupported',
    }
    summary={'gates':gates,'pooled':pooled,'grids':results,'timing':timing,
             'oracle':{'points':len(labels),'confirmed':len(confirmed),'unresolved':len(labels)-len(confirmed)},
             'jovin':{'receipt_id':jovin['receipt_id'],'source_manifest_sha256':jovin['source_manifest_sha256'],
                      'receipt_sha256':jovin['receipt_sha256'],'chain_head_sha256':chain['head_sha256']},
             'protocol_sha256':core.digest(target/'protocol.json')}
    core.write_json(target/'summary.json',summary)
    (target/'report.md').write_text(render_report(summary),encoding='utf-8')
    core.write_json(target/'evidence_hashes.json',{path.relative_to(target).as_posix():core.digest(path)
                    for path in sorted(target.rglob('*')) if path.is_file() and path.name!='evidence_hashes.json'})
    print(f'Confirmation complete: {gates}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--pilot',type=Path,default=PILOT)
    parser.add_argument('--bin',type=Path)
    parser.add_argument('--jovin',type=Path,default=ROOT/'out'/'jovin-holdout')
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--oracle-jobs',type=int,choices=range(1,9),default=8)
    parser.add_argument('--grid',type=int,choices=(1,2,3))
    parser.add_argument('phase',choices=('prepare','gpu','strict','reference','score','execute'))
    args=parser.parse_args()
    args.bin=args.bin or args.pilot/'out'/'bin-lc'
    phases={'prepare':prepare,'gpu':run_gpu,'strict':run_strict,'reference':run_reference,'score':score_run}
    if args.phase=='execute':
        for phase in ('gpu','strict','reference','score'):
            phases[phase](args)
    else:
        phases[args.phase](args)


if __name__=='__main__':
    main()


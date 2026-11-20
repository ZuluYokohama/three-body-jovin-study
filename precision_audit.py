"""Rerun the frozen Levi-Civita audit cohort with tighter FP64 tolerances."""

import argparse
import json
from pathlib import Path

from study import ROOT, Study, audit_cell, digest, flagged, read_csv, work, write_csv, write_json


# These settings and the acceptance rule apply identically to every audit point.
TIERS = (
    ('p12', {'rtol': 1e-12, 'atol': 1e-14, 'hmax': 0.01}),
    ('p13', {'rtol': 1e-13, 'atol': 1e-15, 'hmax': 0.005}),
    ('p14', {'rtol': 3e-14, 'atol': 3e-16, 'hmax': 0.0025}),
)
POLICY = 'Use a label only when the two strictest FP64 settings agree and both pass the original energy screen.'


def run_precision_audit(study, coords, cpu_path):
    """Run every GPU tier before opening the independent CPU outcome file."""
    if not coords or len(set(coords)) != len(coords):
        raise ValueError('The audit coordinates must be unique and nonempty.')
    if any(not (0 <= i < study.fine and 0 <= j < study.fine) for i, j in coords):
        raise ValueError('An audit coordinate lies outside the study grid.')
    if study.coordinates != 'levi-civita':
        raise ValueError('This audit requires the Levi-Civita solver.')

    output = study.out
    measurements = {}
    tier_results = []
    for name, settings in TIERS:
        result, elapsed = study.solve_cells(f'precision_{name}', coords, study.fine, settings)
        tier_results.append(result)
        measurements[name] = {'settings': settings, 'evaluations': work(result), 'wall_seconds': elapsed}

    # Freeze predictions before looking at the reference labels. In particular,
    # do not choose a per-case tier by comparing it to the independent outcome.
    predictions = {}
    for cell in coords:
        labels = tuple(5 if flagged(result[cell]) else result[cell].label for result in tier_results)
        predictions[cell] = (*labels, labels[-1] if labels[-1] == labels[-2] else 5)
    write_csv(output / 'gpu_predictions.csv', ('i', 'j', 'p12', 'p13', 'p14', 'confirmed'),
              ((*cell, *predictions[cell]) for cell in coords))

    reference = read_csv(cpu_path)
    if [audit_cell(row) for row in reference] != list(coords):
        raise ValueError('The CPU reference does not cover the same ordered audit coordinates.')
    entries = []
    stable = matches = unresolved_stable = 0
    resolved = [0] * len(TIERS)
    per_tier_matches = [0] * len(TIERS)
    for cell, row in zip(coords, reference, strict=True):
        labels = predictions[cell]
        cpu, cpu_tight = int(row['cpu']), int(row['cpu_tight'])
        is_stable = cpu == cpu_tight != 5
        stable += is_stable
        match = is_stable and labels[-1] == cpu_tight
        matches += match
        unresolved_stable += is_stable and labels[-1] == 5
        for i, predicted in enumerate(labels[:-1]):
            resolved[i] += predicted != 5
            per_tier_matches[i] += is_stable and predicted == cpu_tight
        entries.append((*cell, cpu, cpu_tight, *labels, int(is_stable), int(match),
                        *(tier[cell].time for tier in tier_results),
                        *(tier[cell].min_r for tier in tier_results),
                        *(tier[cell].energy_error for tier in tier_results)))

    names = [name for name, _ in TIERS]
    write_csv(output / 'audit.csv', ('i', 'j', 'cpu', 'cpu_tight', *names, 'confirmed', 'cpu_stable',
              'match', *(f'{name}_time' for name in names), *(f'{name}_min_r' for name in names),
              *(f'{name}_energy_error' for name in names)), entries)
    for i, name in enumerate(names):
        measurements[name]['resolved'] = resolved[i]
        measurements[name]['matches'] = per_tier_matches[i]
        measurements[name]['agreement'] = per_tier_matches[i]/stable if stable else None
    summary = {'cases': len(coords), 'stable': stable, 'matches': matches,
               'agreement': matches/stable if stable else None,
               'unresolved_cpu_stable': unresolved_stable,
               'passed': bool(stable >= 48 and 100*matches >= 95*stable),
               'gate': {'minimum_stable': 48, 'agreement': 0.95},
               'policy': POLICY, 'arithmetic': 'IEEE FP64', 'tiers': measurements,
               'evaluations': sum(item['evaluations'] for item in measurements.values()),
               'wall_seconds': sum(item['wall_seconds'] for item in measurements.values()),
               'cpu_reference': {'path': str(cpu_path), 'sha256': digest(cpu_path)},
               'precomparison_gpu_predictions_sha256': digest(output / 'gpu_predictions.csv')}
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'out/study')
    parser.add_argument('--cpu', type=Path, default=ROOT / 'out/study-lc/cpu_audit.csv')
    parser.add_argument('--bin', type=Path, default=ROOT / 'out/bin-lc')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    cohort = baseline / 'cpu_audit.csv'
    prior_manifest = baseline / 'manifest.json'
    archive = json.loads((baseline / 'evidence_hashes.json').read_text(encoding='utf-8'))
    if any(digest(path) != archive[path.name] for path in (cohort, prior_manifest)):
        raise RuntimeError('The archived audit cohort or study manifest has changed.')
    config = json.loads(prior_manifest.read_text(encoding='utf-8'))
    coords = [audit_cell(row) for row in read_csv(cohort)]
    if len(coords) != 96 or config['cpu_sample'] != len(coords) or config['horizon'] != 50:
        raise RuntimeError('The run requires the original 96-case, t=50 audit cohort.')

    if args.out.exists():
        parser.error('--out must be a new directory')
    output = args.out.resolve()
    study = Study(output, args.bin.resolve(), args.device, config['base'], config['horizon'], 'levi-civita')
    write_json(output / 'manifest.json', {'policy': POLICY, 'tiers': dict(TIERS),
               'coordinates': study.coordinates, 'horizon': study.horizon,
               'close_separation': 1e-3, 'arithmetic': 'IEEE FP64', 'device': args.device,
               'cohort': {'path': str(cohort), 'sha256': archive[cohort.name]},
               'cpu_reference_path': str(args.cpu.resolve()),
               'solver_sha256': digest(study.binaries / 'solver.exe'),
               'orchestrator_sha256': digest(Path(__file__)), 'study_sha256': digest(ROOT / 'study.py')})
    result = run_precision_audit(study, coords, args.cpu.resolve())
    write_json(output / 'evidence_hashes.json', {path.relative_to(output).as_posix(): digest(path)
               for path in sorted(output.rglob('*')) if path.is_file()})
    print(f"Precision audit: {result['matches']}/{result['stable']} stable CPU outcomes confirmed; "
          f"{result['unresolved_cpu_stable']} stable CPU cases unresolved on GPU; "
          f"{'pass' if result['passed'] else 'fail'}", flush=True)


if __name__ == '__main__':
    main()

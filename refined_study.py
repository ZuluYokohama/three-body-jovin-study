"""Spend the measured Levi-Civita work savings on a paired follow-up study."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import study as core
from boundary_refine import refine_boundaries


ROOT = core.ROOT
FINE = 256
DENSE_LEVELS = (
    {'rtol': 1e-12, 'atol': 1e-14, 'hmax': 0.01},
    {'rtol': 1e-13, 'atol': 1e-15, 'hmax': 0.005},
)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def archive_hashes(directory, names):
    inventory = read_json(directory / 'evidence_hashes.json')
    hashes = {name: core.digest(directory / name) for name in names}
    if any(inventory.get(name) != digest for name, digest in hashes.items()):
        raise RuntimeError(f'Archived evidence does not match its recorded hashes in {directory}.')
    return hashes


def grid_results(directory, name, side):
    results = {}
    with (directory / 'raw' / f'{name}.out.csv').open(newline='', encoding='utf-8') as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if index != int(row['id']) or index >= side*side:
                raise RuntimeError(f'Unexpected order in archived grid {name}.')
            results[index % side, index // side] = core.Result(
                int(row['label']), float(row['time']), float(row['min_r']), int(row['evals']),
                float(row['energy_error']), ())
    if len(results) != side*side:
        raise RuntimeError(f'Incomplete archived grid {name}.')
    return results


def dense_precision(study, loose, tight, reference, audited, budget, audit_cost):
    """Select on recorded GPU indicators and leave the audited reference exclusions intact."""
    primary = []
    grazing = []
    for cell, second in tight.items():
        first = loose[cell]
        if core.flagged(first) or core.flagged(second) or first.label != second.label:
            primary.append(cell)
        elif second.label != 4 and 0.001 < min(first.min_r, second.min_r) <= 0.0012:
            grazing.append(cell)
    queue = primary + grazing
    available = budget - audit_cost
    if available < 0:
        raise RuntimeError('The paired precision audit exhausted the saved reference work.')
    records = []
    spent = 0
    batch_number = 0
    estimate = 500_000
    for start in range(0, len(queue), 128):
        batch = queue[start:start+128]
        if available - spent < max(16_000_000, 2*estimate*len(batch)):
            # Spend the residual only if there is room for measured variation in a smaller batch.
            batch = batch[:max(0, (available - spent - 16_000_000)//(2*estimate))]
        if not batch:
            break
        results = [study.solve_cells(f'dense_precision_{batch_number:03d}_{level}', batch, FINE,
                                     settings)[0] for level, settings in enumerate(DENSE_LEVELS)]
        cost = sum(core.work(level) for level in results)
        spent += cost
        if spent > available:
            raise RuntimeError('Dense precision refinement exceeded the saved reference work.')
        estimate = max(1, round(cost/len(batch)))
        batch_number += 1
        for i, j in batch:
            a, b = (level[i, j] for level in results)
            previous = int(reference[j, i])
            confirm = a.label if a.label == b.label and not core.flagged(a) and not core.flagged(b) else 5
            if (i, j) not in audited:
                reference[j, i] = confirm
            records.append((i, j, previous, confirm, int(reference[j, i]), a.label, b.label,
                            a.evals, b.evals, a.time, b.time, a.min_r, b.min_r,
                            a.energy_error, b.energy_error, int((i, j) in audited)))
        if len(batch) < min(128, len(queue) - start):
            break
    core.write_csv(study.out / 'dense_precision.csv', ('i', 'j', 'previous_reference', 'confirmed_label',
        'new_reference', 'first_label', 'second_label', 'first_evals', 'second_evals', 'first_time',
        'second_time', 'first_min_r', 'second_min_r', 'first_energy_error', 'second_energy_error',
        'audited_coordinate'), records)
    return {'tolerance_disagreements_or_unresolved': len(primary), 'grazing_candidates': len(grazing),
            'rerun_trajectories': len(records), 'force_evaluations': spent,
            'newly_resolved': sum(row[2] == 5 and row[4] != 5 for row in records),
            'lost_resolution': sum(row[2] != 5 and row[4] == 5 for row in records),
            'changed_resolved_label': sum(row[2] != row[4] and 5 not in (row[2], row[4]) for row in records),
            'still_unconfirmed': sum(row[3] == 5 for row in records)}


def compare(prediction, uniform, reference):
    stable, boundary = reference != 5, core.boundary_mask(reference)
    return {'adaptive': {'all': core.score(prediction, reference, stable),
                         'boundary': core.score(prediction, reference, boundary)},
            'uniform': {'all': core.score(uniform, reference, stable),
                        'boundary': core.score(uniform, reference, boundary)}}


def percent(value):
    return 'n/a' if value is None else f'{100*value:.2f}%'


def rate(metrics):
    return f"{percent(metrics['agreement'])} ({metrics['matches']:,}/{metrics['count']:,})"


def report(summary):
    comparison = summary['boundary']
    frozen = comparison['frozen_levi_civita']
    promoted = comparison['precision_reference']
    edge = comparison['frozen_edge']
    cpu = summary['cpu']
    verdict = summary['gates']
    dense = summary['dense_precision']
    work = summary['work']
    return '\n'.join([
        '# Levi-Civita boundary refinement and precision reruns', '',
        f"Adaptive boundary gate **{'passed' if verdict['boundary'] else 'failed'}** at "
        f"{rate(frozen['adaptive']['boundary'])} against the frozen Levi-Civita reference. The target was 90%.", '',
        f"CPU gate **{'passed' if verdict['cpu'] else 'failed'}** at {cpu['matches']}/{cpu['stable']} "
        f"({percent(cpu['agreement'])}) on the original 96-point audit. The target was 95% with at least 48 "
        'stable CPU cases. GPU cases that disagreed between the two strictest settings count as misses.', '',
        '| Reference boundary | Adaptive | Uniform at matched work |', '|---|---:|---:|',
        f"| Frozen Levi-Civita | {rate(frozen['adaptive']['boundary'])} | {rate(frozen['uniform']['boundary'])} |",
        f"| Precision rerun | {rate(promoted['adaptive']['boundary'])} | {rate(promoted['uniform']['boundary'])} |",
        f"| Frozen original edge | {rate(edge['adaptive']['boundary'])} | {rate(edge['uniform']['boundary'])} |", '',
        f"Boundary sampling spent {work['extra_adaptive']:,} of {work['saved_adaptive']:,} available force evaluations. "
        f"Dense and audit reruns spent {work['extra_reference']:,} of {work['saved_reference']:,}. "
        f"The dense precision pass reran {dense['rerun_trajectories']:,} trajectories and resolved "
        f"{dense['newly_resolved']:,} additional cells under the agreement rule. The adaptive sampler used its own "
        'observations. The dense reference was used for scoring.', '',
        'The GPU reruns tightened local error controls in FP64. These are classification checks at the original '
        'time horizon and separation cutoff. Jovin verified the CPU chart and its analytic fixtures and the run '
        'manifest binds the CUDA binary, selection code, settings, archive inputs, and output hashes.', '',
        'Both percentages are pilot targets. Publication claims need a frozen procedure evaluated on independent '
        'cases, convergence and uncertainty results, and a matched-work comparison with a reproducible artifact.', '',
        '[Outcome map](outcome_map.svg) · [Dense precision checks](dense_precision.csv) · [Summary](summary.json)', '',
    ])


def execute(args):
    edge, lc, audit_dir = (path.resolve() for path in (args.edge, args.lc, args.precision_audit))
    binaries = args.bin.resolve()
    edge_summary = read_json(edge / 'summary.json')
    lc_summary = read_json(lc / 'summary.json')
    edge_hashes = archive_hashes(edge, ('summary.json', 'manifest.json', 'cpu_audit.csv',
                                       'outcome_grids.npz', 'raw/reference_tight.out.csv'))
    lc_hashes = archive_hashes(lc, ['summary.json', 'manifest.json', 'cpu_audit.csv', 'outcome_grids.npz',
                                   'raw/reference_tight.out.csv',
                                   *(f'raw/uniform_{level}.out.csv' for level in range(4)),
                                   *(f'raw/{stage}.{direction}.csv'
                                     for stage in ('uniform_0', 'adaptive_1', 'adaptive_2', 'adaptive_3')
                                     for direction in ('in', 'out'))])
    audit_hashes = archive_hashes(audit_dir, ('manifest.json', 'summary.json', 'audit.csv', 'gpu_predictions.csv'))
    audit_manifest = read_json(audit_dir / 'manifest.json')
    audited = {core.audit_cell(row) for row in core.read_csv(edge / 'cpu_audit.csv')}
    cpu = read_json(audit_dir / 'summary.json')
    binary_hashes = {name: core.digest(binaries / name) for name in ('solver.exe', 'filter.exe')}
    if (len(audited) != 96 or cpu['cases'] != 96 or audit_manifest['horizon'] != 50.0 or
            audit_manifest['coordinates'] != 'levi-civita' or audit_manifest['close_separation'] != 0.001 or
            audit_manifest['cohort']['sha256'] != edge_hashes['cpu_audit.csv'] or
            cpu['cpu_reference']['sha256'] != lc_hashes['cpu_audit.csv'] or
            audit_manifest['solver_sha256'] != binary_hashes['solver.exe'] or
            cpu['precomparison_gpu_predictions_sha256'] != audit_hashes['gpu_predictions.csv'] or
            read_json(lc / 'manifest.json')['binary_sha256']['solver.exe'] != binary_hashes['solver.exe']):
        raise RuntimeError('The refined study requires the original frozen CPU cohort and Levi-Civita binary.')
    evidence = core.jovin_evidence(args.jovin.resolve())
    source_names = ('study.py', 'reference.py', 'lc_checks.py', 'precision_audit.py', 'refined_study.py',
                    'boundary_refine.py', 'cuda/levi_civita.hpp', 'cuda/solver.cu')
    sources = {name: core.digest(ROOT / name) for name in source_names}
    if (sources['precision_audit.py'] != audit_manifest['orchestrator_sha256'] or
            sources['study.py'] != audit_manifest['study_sha256']):
        raise RuntimeError('CPU audit source differs from the frozen run.')
    previous_adaptive = lc_summary['comparison'][-1]['adaptive']['evaluations']
    original_adaptive = edge_summary['comparison'][-1]['adaptive']['evaluations']
    edge_tight = core.work(grid_results(edge, 'reference_tight', FINE))
    lc_tight = grid_results(lc, 'reference_tight', FINE)
    adaptive_budget = original_adaptive - previous_adaptive
    reference_budget = edge_tight - core.work(lc_tight)
    if adaptive_budget <= 0 or reference_budget <= 0:
        raise RuntimeError('The archived runs contain no positive work savings to reinvest.')
    with np.load(edge / 'outcome_grids.npz', allow_pickle=False) as arrays:
        edge_reference = arrays['reference']
    with np.load(lc / 'outcome_grids.npz', allow_pickle=False) as arrays:
        frozen_reference = arrays['reference']
        previous_map = arrays['adaptive']
    if edge_reference.shape != (FINE, FINE) or frozen_reference.shape != (FINE, FINE):
        raise RuntimeError('The archived reference planes differ from the fixed 256 × 256 experiment.')
    study = core.Study(args.out.resolve(), binaries, args.device, FINE//8, 50.0, 'levi-civita')
    hardware, _ = core.run_command(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version',
                                    '--format=csv,noheader'], study.raw / 'hardware.log')
    protocol = {'coordinates': 'levi-civita', 'grid': FINE, 'horizon': 50.0, 'close_separation': 0.001,
                'domain': core.BOUNDS, 'device': args.device, 'hardware': hardware.splitlines(),
                'gates': {'boundary': 0.90, 'cpu': 0.95, 'minimum_stable_cpu': 48},
                'boundary_reference': 'frozen prior Levi-Civita reference',
                'adaptive_selection': 'observe archived and newly integrated adaptive samples only',
                'dense_levels': DENSE_LEVELS, 'audit_levels': audit_manifest['tiers'],
                'saved_adaptive': adaptive_budget, 'saved_reference': reference_budget,
                'source_sha256': sources, 'binary_sha256': binary_hashes, 'jovin': evidence,
                'archive_sha256': {str(edge): edge_hashes, str(lc): lc_hashes, str(audit_dir): audit_hashes}}
    core.write_json(study.out / 'manifest.json', protocol)
    gpu_gate = study.gate_gpu()
    lc_gate = study.gate_lc()
    adaptive, ledger = refine_boundaries(study, lc, adaptive_budget)
    lc_uniform = [grid_results(lc, f'uniform_{level}', (FINE//8)*2**level) for level in range(4)]
    if ledger['original_adaptive_rhs'] != previous_adaptive:
        raise RuntimeError('Adaptive work accounting disagrees with the archived run.')
    uniform, matched_work, _, _ = core.uniform_at_budget(lc_uniform, FINE//8, ledger['total_adaptive_rhs'])
    fresh_reference = frozen_reference.copy()
    dense = dense_precision(study, lc_uniform[-1], lc_tight, fresh_reference, audited, reference_budget,
                            cpu['evaluations'])
    scores = {'frozen_levi_civita': compare(adaptive, uniform, frozen_reference),
              'precision_reference': compare(adaptive, uniform, fresh_reference),
              'frozen_edge': compare(adaptive, uniform, edge_reference),
              'previous_levi_civita': core.score(previous_map, frozen_reference,
                                               core.boundary_mask(frozen_reference))}
    boundary_pass = (scores['frozen_levi_civita']['adaptive']['boundary']['agreement'] or 0) >= 0.90
    cpu_pass = cpu['stable'] >= 48 and (cpu['agreement'] or 0) >= 0.95
    if cpu_pass != cpu['passed']:
        raise RuntimeError('The precision audit reported an inconsistent CPU gate.')
    core.render_map(study.out / 'outcome_map.svg', (('Frozen Levi-Civita reference', frozen_reference),
                    ('Budgeted adaptive refinement', adaptive), ('Uniform at matched work', uniform)), 50.0)
    np.savez_compressed(study.out / 'outcome_grids.npz', frozen_reference=frozen_reference,
                        precision_reference=fresh_reference, adaptive=adaptive, uniform_matched=uniform,
                        bounds=np.asarray(core.BOUNDS))
    for name, expected in sources.items():
        if core.digest(ROOT / name) != expected:
            raise RuntimeError(f'Source changed during the study: {name}')
    for folder, hashes in protocol['archive_sha256'].items():
        if any(core.digest(Path(folder) / name) != expected for name, expected in hashes.items()):
            raise RuntimeError(f'An archived result changed during the study: {folder}')
    if any(core.digest(binaries / name) != expected for name, expected in binary_hashes.items()):
        raise RuntimeError('GPU binaries changed during the study.')
    summary = {'boundary': scores, 'cpu': cpu, 'dense_precision': dense, 'adaptive': ledger,
               'gates': {'boundary': boundary_pass, 'cpu': cpu_pass, 'both': boundary_pass and cpu_pass},
               'work': {'saved_adaptive': adaptive_budget, 'saved_reference': reference_budget,
                        'extra_adaptive': ledger['additional_rhs'],
                        'extra_reference': cpu['evaluations'] + dense['force_evaluations'],
                        'total_adaptive': ledger['total_adaptive_rhs'], 'matched_uniform': matched_work},
               'resolved_reference': {'frozen': float(np.mean(frozen_reference != 5)),
                                      'precision': float(np.mean(fresh_reference != 5))},
               'gpu_state_gate': gpu_gate, 'gpu_lc_gate': lc_gate, 'jovin': evidence,
               'precision_audit_artifacts': str(audit_dir), 'manifest_sha256': core.digest(study.out / 'manifest.json')}
    core.write_json(study.out / 'summary.json', summary)
    (study.out / 'report.md').write_text(report(summary), encoding='utf-8')
    core.write_json(study.out / 'evidence_hashes.json', {p.relative_to(study.out).as_posix(): core.digest(p)
                    for p in sorted(study.out.rglob('*')) if p.is_file()})
    print(f'Finished. Report at {study.out / "report.md"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--edge', type=Path, default=ROOT / 'out/study')
    parser.add_argument('--lc', type=Path, default=ROOT / 'out/study-lc')
    parser.add_argument('--precision-audit', type=Path, default=ROOT / 'out/precision-audit-refined')
    parser.add_argument('--jovin', type=Path, default=ROOT / 'out/jovin-refined')
    parser.add_argument('--bin', type=Path, default=ROOT / 'out/bin-lc')
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('--out must be a new directory')
    execute(args)


if __name__ == '__main__':
    main()

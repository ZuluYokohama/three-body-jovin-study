"""Independent physical-time checks for the CUDA Levi-Civita chart."""

from math import isfinite

from reference import integrate, normalized_state_error, pair_distance


HORIZON = 0.38
GPU_SETTINGS = {'rtol': 1e-13, 'atol': 1e-15, 'hmax': 0.001}
CPU_SETTINGS = {'rtol': 3e-14, 'atol': 1e-15, 'max_step': 0.0005}
CLOSE_DISTANCE = 0.001

# Reflections exercise both pair orientations encountered on this plane.
# Points 4 and 6 bracket a tangency to the cutoff near x = 0.2854492.
FIXTURES = (
    ((0.30, 0.40), 0),
    ((0.70, 0.40), 0),
    ((0.35, 0.50), 0),
    ((0.65, 0.50), 0),
    ((0.28553, 0.40), 0),
    ((0.28, 0.40), 4),
    ((0.28537, 0.40), 4),
)


def check_coupled_passages(solve):
    """Check coupled passages and cutoff events against direct Newtonian DOP853."""
    points = {index: point for index, (point, _) in enumerate(FIXTURES)}
    results, _ = solve('gpu_lc_passages', points, GPU_SETTINGS, HORIZON)
    details = []
    for index, (point, expected_label) in enumerate(FIXTURES):
        actual = results[index]
        label, time, expected = integrate(point, horizon=HORIZON, **CPU_SETTINGS)
        if label != expected_label or actual.label != label:
            raise RuntimeError(f'Levi-Civita fixture {point} expected label {expected_label}, '
                               f'got CPU={label} and CUDA={actual.label}.')
        if actual.chart_entries < 1 or not 0.0 <= actual.max_lc_residual <= 1e-8:
            raise RuntimeError(f'Levi-Civita chart diagnostics failed for {point}: {actual}')
        time_error = abs(actual.time - time)
        if not isfinite(time_error) or time_error > 5e-9:
            raise RuntimeError(f'Levi-Civita physical-time error for {point}: {time_error:g}.')

        row = {'point': list(point), 'label': label, 'time': actual.time, 'cpu_time': time,
               'time_error': time_error, 'minimum_separation': actual.min_r,
               'chart_entries': actual.chart_entries, 'chart_exits': actual.chart_exits,
               'max_lc_residual': actual.max_lc_residual}
        if label == 0:
            if actual.chart_exits < 1 or not CLOSE_DISTANCE < actual.min_r < 0.05:
                raise RuntimeError(f'Levi-Civita did not complete the isolated passage for {point}.')
            state_error = normalized_state_error(actual.state, expected)
            if not isfinite(state_error) or state_error > 1e-8:
                raise RuntimeError(f'Levi-Civita reconstructed state error for {point}: {state_error:g}.')
            row['normalized_state_error'] = state_error
        else:
            cutoff = min(pair_distance(actual.state, 0, 1), pair_distance(actual.state, 1, 2),
                         pair_distance(actual.state, 0, 2))
            cutoff_error = abs(cutoff - CLOSE_DISTANCE)
            if not isfinite(cutoff_error) or cutoff_error > 2e-8:
                raise RuntimeError(f'Levi-Civita cutoff localization failed for {point}: {cutoff:g}.')
            row['cutoff_radius_error'] = cutoff_error
        details.append(row)

    passes = [row for row in details if row['label'] == 0]
    cutoffs = [row for row in details if row['label'] == 4]
    return {'passed': True, 'fixtures': len(details), 'completed_passages': len(passes),
            'cutoff_fixtures': len(cutoffs), 'horizon': HORIZON,
            'max_normalized_state_error': max(row['normalized_state_error'] for row in passes),
            'max_cutoff_time_error': max(row['time_error'] for row in cutoffs),
            'max_cutoff_radius_error': max(row['cutoff_radius_error'] for row in cutoffs),
            'max_lc_residual': max(row['max_lc_residual'] for row in details), 'details': details}

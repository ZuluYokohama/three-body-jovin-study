"""Independent body-coordinate integration of the study's Newtonian system."""

from math import hypot, sqrt

import numpy as np
from scipy.integrate import solve_ivp


def initial_state(x, y):
    cx, cy = (1.0 + x) / 3.0, y / 3.0
    return (-cx, -cy, 1.0 - cx, -cy, x - cx, y - cy, *([0.0] * 6))


def derivative(_time, state):
    x1, y1, x2, y2, x3, y3 = map(float, state[:6])
    x12, y12 = x2 - x1, y2 - y1
    x13, y13 = x3 - x1, y3 - y1
    x23, y23 = x3 - x2, y3 - y2
    s12 = x12 * x12 + y12 * y12
    s13 = x13 * x13 + y13 * y13
    s23 = x23 * x23 + y23 * y23
    k12, k13, k23 = (1.0 / (s * sqrt(s)) for s in (s12, s13, s23))
    return (*state[6:], x12*k12 + x13*k13, y12*k12 + y13*k13,
            -x12*k12 + x23*k23, -y12*k12 + y23*k23,
            -x13*k13 - x23*k23, -y13*k13 - y23*k23)


def pair_distance(state, i, j):
    return hypot(state[2*i] - state[2*j], state[2*i+1] - state[2*j+1])


def close_margin(_time, state):
    return min(pair_distance(state, 0, 1), pair_distance(state, 1, 2),
               pair_distance(state, 0, 2)) - 1e-3


close_margin.terminal = True
close_margin.direction = -1


def escape_margin(state, k):
    i, j = (k + 1) % 3, (k + 2) % 3
    d = pair_distance(state, i, j)
    rx = state[2*k] - 0.5*(state[2*i] + state[2*j])
    ry = state[2*k+1] - 0.5*(state[2*i+1] + state[2*j+1])
    vx = state[2*k+6] - 0.5*(state[2*i+6] + state[2*j+6])
    vy = state[2*k+7] - 0.5*(state[2*i+7] + state[2*j+7])
    ux = state[2*i+6] - state[2*j+6]
    uy = state[2*i+7] - state[2*j+7]
    radius = hypot(rx, ry)
    if radius == 0.0 or d == 0.0:
        return -1.0
    return min(radius - 10.0, radius - 10.0*d, (rx*vx + ry*vy)/radius,
               0.5*(vx*vx + vy*vy) - 3.0/radius, 2.0/d - 0.5*(ux*ux + uy*uy))


def escape_event(k):
    def event(_time, state):
        return escape_margin(state, k)
    event.terminal = True
    event.direction = 1
    return event


EVENTS = (close_margin, *(escape_event(k) for k in range(3)))
EVENT_LABELS = (4, 1, 2, 3)


def integrate(point, *, horizon, rtol, atol, max_step):
    kwargs = dict(rtol=rtol, atol=atol, max_step=max_step, events=EVENTS)
    result = solve_ivp(derivative, (0.0, horizon), initial_state(*point),
                       method="DOP853", **kwargs)
    label = 0 if result.success else 5
    final_state = tuple(result.y[:, -1])
    if result.status == 1:
        event = next(i for i, times in enumerate(result.t_events) if len(times))
        label = EVENT_LABELS[event]
        if label in (1, 2, 3):
            candidates = sum(escape_margin(final_state, k) >= -1e-10 for k in range(3))
            if candidates != 1:
                label = 5
    return label, float(result.t[-1]), final_state


def compare_point(task):
    ident, point, horizon = task
    standard = integrate(point, horizon=horizon, rtol=1e-10, atol=1e-12, max_step=0.04)
    tight = integrate(point, horizon=horizon, rtol=1e-12, atol=1e-14, max_step=0.02)
    return ident, standard[0], tight[0], standard[1], tight[1]


def normalized_state_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    return float(np.max(np.abs(actual - expected) / np.maximum(1.0, np.abs(expected))))

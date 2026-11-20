// Independent IEEE binary128 reference for the equal-mass free-fall plane.
// Build on Linux with g++ -std=gnu++17 -O3 holdout_oracle.cpp -lquadmath -o holdout_oracle.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <quadmath.h>

using Real = __float128;
using State = std::array<Real, 12>;
constexpr Real cutoff = 0.001Q;
constexpr Real cutoff_squared = cutoff * cutoff;
constexpr int levels = 8;
constexpr std::array<std::array<int, 2>, 3> pairs = {{{0, 2}, {0, 4}, {2, 4}}};

State force(const State& q) {
    State result{};
    for (int i = 0; i < 6; ++i) result[i] = q[i + 6];
    for (int i = 0; i < 6; i += 2) {
        for (int j = i + 2; j < 6; j += 2) {
            const Real x = q[j] - q[i], y = q[j + 1] - q[i + 1];
            const Real s = x*x + y*y;
            const Real scale = 1 / (s * sqrtq(s));
            result[i + 6] += x * scale;
            result[i + 7] += y * scale;
            result[j + 6] -= x * scale;
            result[j + 7] -= y * scale;
        }
    }
    return result;
}

State midpoint(const State& original, Real span, int n) {
    const Real h = span / n;
    State previous = original, current = original;
    const State initial = force(original);
    for (int i = 0; i < 12; ++i) current[i] += h * initial[i];
    for (int step = 1; step < n; ++step) {
        const State tangent = force(current);
        State next{};
        for (int i = 0; i < 12; ++i) next[i] = previous[i] + 2 * h * tangent[i];
        previous = current;
        current = next;
    }
    const State tangent = force(current);
    for (int i = 0; i < 12; ++i) current[i] = (previous[i] + current[i] + h * tangent[i]) / 2;
    return current;
}

struct Trial { State q; Real error; int order; };

Trial extrapolate(const State& q, Real h, Real tolerance) {
    std::array<State, levels> history{};
    history[0] = midpoint(q, h, 2);
    Real error = HUGE_VALQ;
    for (int j = 1; j < levels; ++j) {
        const State old = history[0];
        history[j] = midpoint(q, h, 2*(j + 1));
        for (int k = j - 1; k >= 0; --k) {
            const Real ratio = Real(j + 1) / (k + 1);
            for (int i = 0; i < 12; ++i) {
                history[k][i] = history[k + 1][i] +
                    (history[k + 1][i] - history[k][i]) / (ratio*ratio - 1);
            }
        }
        error = 0;
        for (int i = 0; i < 12; ++i) {
            error = fmaxq(error, fabsq(history[0][i] - old[i]) /
                                 (tolerance * (1 + fmaxq(fabsq(q[i]), fabsq(history[0][i])))));
        }
        if (j >= 3 && error <= 1) return {history[0], error, 2*(j + 1)};
    }
    return {history[0], error, 2*levels};
}

Real squared_distance(const State& q, int i, int j) {
    const Real x = q[i] - q[j], y = q[i + 1] - q[j + 1];
    return x*x + y*y;
}

Real distance(const State& q, int i, int j) { return sqrtq(squared_distance(q, i, j)); }

Real radial_rate(const State& q, int i, int j) {
    return (q[i]-q[j]) * (q[i+6]-q[j+6]) +
           (q[i+1]-q[j+1]) * (q[i+7]-q[j+7]);
}

Real energy(const State& q) {
    Real e = 0;
    for (int i = 0; i < 6; ++i) e += q[i + 6]*q[i + 6]/2;
    return e - 1/distance(q,0,2) - 1/distance(q,0,4) - 1/distance(q,2,4);
}

int escape(const State& q) {
    int result = 0;
    for (int k = 0; k < 3; ++k) {
        const int i = 2*((k + 1)%3), j = 2*((k + 2)%3), m = 2*k;
        const Real d = distance(q,i,j);
        const Real rx = q[m] - (q[i] + q[j])/2, ry = q[m + 1] - (q[i + 1] + q[j + 1])/2;
        const Real r = sqrtq(rx*rx + ry*ry);
        if (r < 10 || r < 10*d) continue;
        const Real vx = q[m + 6] - (q[i + 6] + q[j + 6])/2;
        const Real vy = q[m + 7] - (q[i + 7] + q[j + 7])/2;
        const Real ux = q[i + 6] - q[j + 6], uy = q[i + 7] - q[j + 7];
        if (rx*vx + ry*vy > 0 && (vx*vx + vy*vy)/2 > 3/r && (ux*ux + uy*uy)/2 < 2/d)
            result = result ? 5 : k + 1;
    }
    return result;
}

struct Event { State q; Real offset; int label; bool interior_close; };

Event detect_event(const State& start, const State& end, Real span, Real tolerance) {
    Event earliest{end, span, 0, false};
    for (const auto& pair : pairs) {
        const int i = pair[0], j = pair[1];
        const Real d0 = squared_distance(start, i, j), d1 = squared_distance(end, i, j);
        Real upper = span;
        State candidate = end;
        const bool inside_end = d1 <= cutoff_squared;
        if (!inside_end) {
            // Limit the costly minimum search to close passages. Every physical step
            // is already capped by both the pair crossing time and free-fall time.
            if (fminq(d0, d1) > 4*cutoff_squared ||
                radial_rate(start, i, j) >= 0 || radial_rate(end, i, j) < 0) continue;
            Real lower = 0;
            for (int iteration = 0; iteration < 48; ++iteration) {
                const Real mid = (lower + upper)/2;
                const Trial trial = extrapolate(start, mid, tolerance);
                if (!finiteq(trial.error) || trial.error > 1) return {start, 0, 5, false};
                if (radial_rate(trial.q, i, j) >= 0) {
                    upper = mid;
                    candidate = trial.q;
                } else {
                    lower = mid;
                }
            }
            if (squared_distance(candidate, i, j) > cutoff_squared) continue;
        }
        Real lower = 0;
        for (int iteration = 0; iteration < 48; ++iteration) {
            const Real mid = (lower + upper)/2;
            const Trial trial = extrapolate(start, mid, tolerance);
            if (!finiteq(trial.error) || trial.error > 1) return {start, 0, 5, false};
            if (squared_distance(trial.q, i, j) <= cutoff_squared) {
                upper = mid;
                candidate = trial.q;
            } else {
                lower = mid;
            }
        }
        if (!earliest.label || upper < earliest.offset) earliest = {candidate, upper, 4, !inside_end};
    }

    if (!escape(end)) return earliest;
    Real upper = earliest.offset;
    State candidate = earliest.q;
    int label = escape(candidate);
    if (!label) return earliest;
    Real lower = 0;
    for (int iteration = 0; iteration < 48; ++iteration) {
        const Real mid = (lower + upper)/2;
        const Trial trial = extrapolate(start, mid, tolerance);
        if (!finiteq(trial.error) || trial.error > 1) return {start, 0, 5, false};
        if (const int detected = escape(trial.q)) {
            upper = mid;
            candidate = trial.q;
            label = detected;
        } else {
            lower = mid;
        }
    }
    if (earliest.label == 4 && upper == earliest.offset) return earliest;
    return {candidate, upper, label, false};
}

struct Stop { State q; Real time, energy_drift; int label, steps; bool interior_close; };

Stop integrate(State q, Real horizon, Real tolerance, Real cap_factor) {
    const Real initial_energy = energy(q);
    const Real maximum_step = 0.04Q * cap_factor;
    Real time = 0, h = maximum_step/2, drift = 0;
    int steps = 0;
    while (time < horizon && steps < 500000) {
        ++steps;
        const Real remaining = horizon - time;
        Real span = fminq(remaining, h);
        for (const auto& pair : pairs) {
            const int i = pair[0], j = pair[1];
            const Real d = distance(q, i, j);
            if (d <= cutoff) return {q,time,drift,4,steps,false};
            const Real vx = q[i+6]-q[j+6], vy = q[i+7]-q[j+7];
            const Real speed = sqrtq(vx*vx + vy*vy);
            const Real freefall_time = d*sqrtq(d/2);
            const Real dynamical = speed > 0 ? fminq(freefall_time, d/speed) : freefall_time;
            span = fminq(span, 0.08Q*cap_factor*dynamical);
        }
        const Trial trial = extrapolate(q, span, tolerance);
        if (!finiteq(trial.error) || trial.error > 1) {
            h = span / 2;
            if (h < 1e-25Q) break;
            continue;
        }
        const Event event = detect_event(q, trial.q, span, tolerance);
        if (event.label == 5 && !event.offset) {
            h = span/2;
            if (h < 1e-25Q) break;
            continue;
        }
        const Real accepted_energy = energy(event.q);
        if (!finiteq(accepted_energy)) break;
        drift = fmaxq(drift, fabsq(accepted_energy - initial_energy) / fmaxq(1, fabsq(initial_energy)));
        time = !event.label && event.offset == remaining ? horizon : time + event.offset;
        q = event.q;
        if (event.label) return {q,time,drift,event.label,steps,event.interior_close};
        h = fminq(maximum_step, span * (trial.error < 0.02Q ? 2 : (trial.error < 0.5Q ? 1.25Q : 1)));
    }
    return {q,time,drift,time == horizon ? 0 : 5,steps,false};
}

State freefall(Real x, Real y) {
    const Real cx = (1+x)/3, cy = y/3;
    return {-cx,-cy,1-cx,-cy,x-cx,y-cy};
}

bool interior_grazing_fixture(Real height, int expected_label) {
    constexpr Real span = 1e-7Q;
    constexpr Real tolerance = 1e-22Q;
    State q{span*250, -height/2, -span*250, height/2, 1, 1, -500, 0, 500, 0, 0, 0};
    const Trial trial = extrapolate(q, span, tolerance);
    if (trial.error > 1 || squared_distance(q, 0, 2) <= cutoff_squared ||
        squared_distance(trial.q, 0, 2) <= cutoff_squared) return false;
    const Event result = detect_event(q, trial.q, span, tolerance);
    return result.label == expected_label &&
           (expected_label == 0 || (result.interior_close && result.offset > 0 && result.offset < span));
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) {
        const Real radius = 1/sqrtq(3);
        const Real frequency = sqrtq(3);
        const Real period = 2*acosq(-1)/frequency;
        State q{};
        for (int i = 0; i < 3; ++i) {
            const Real a = 2*acosq(-1)*i/3;
            q[2*i] = radius*cosq(a); q[2*i + 1] = radius*sinq(a);
            q[2*i + 6] = -frequency*q[2*i + 1]; q[2*i + 7] = frequency*q[2*i];
        }
        const Stop stop = integrate(q, period, 1e-23Q, 1);
        Real error = 0;
        for (int i = 0; i < 12; ++i) error = fmaxq(error, fabsq(q[i]-stop.q[i]));
        const bool inside = interior_grazing_fixture(0.0009999Q, 4);
        const bool outside = interior_grazing_fixture(0.0010001Q, 0);
        const Stop close = integrate(freefall(0.28537, 0.40), 0.38Q, 1e-24Q, 0.5Q);
        const Stop clear = integrate(freefall(0.28553, 0.40), 0.38Q, 1e-24Q, 0.5Q);
        const bool passed = stop.label == 0 && error < 1e-20Q && inside && outside &&
                            close.label == 4 && clear.label == 0;
        std::printf("{\"fixture\":\"equal_mass_equilateral\",\"passed\":%s,\"precision_bits\":%d,"
                    "\"state_error\":%.8g,\"energy_drift\":%.8g,\"steps\":%d,"
                    "\"interior_grazing_detected\":%s,\"exterior_grazing_clear\":%s,"
                    "\"freefall_grazing_inside_label\":%d,\"freefall_grazing_outside_label\":%d}\n",
                    passed ? "true" : "false", FLT128_MANT_DIG, double(error), double(stop.energy_drift),
                    stop.steps, inside ? "true" : "false", outside ? "true" : "false", close.label, clear.label);
        return !passed;
    }
    if (argc != 5 && argc != 6) {
        std::fprintf(stderr,"usage: holdout_oracle x y horizon tolerance [cap_factor]\n");
        return 2;
    }
    // Convert through binary64 to lift exactly the two initial coordinates parsed by CUDA.
    const Real x = std::strtod(argv[1], nullptr), y = std::strtod(argv[2], nullptr);
    const Real horizon = strtoflt128(argv[3], nullptr), tolerance = strtoflt128(argv[4], nullptr);
    const Real cap_factor = argc == 6 ? strtoflt128(argv[5], nullptr) : 1;
    if (!finiteq(x) || !finiteq(y) || !finiteq(horizon) || !finiteq(tolerance) || !finiteq(cap_factor) ||
        horizon <= 0 || tolerance <= 0 || cap_factor <= 0 || cap_factor > 1) return 2;
    const Stop stop = integrate(freefall(x,y),horizon,tolerance,cap_factor);
    char time[64], drift[64];
    quadmath_snprintf(time, sizeof(time), "%.25Qg", stop.time);
    quadmath_snprintf(drift, sizeof(drift), "%.9Qg", stop.energy_drift);
    std::printf("{\"precision_bits\":%d,\"label\":%d,\"time\":%s,\"energy_drift\":%s,\"steps\":%d,"
                "\"interior_close\":%s}\n", FLT128_MANT_DIG, stop.label, time, drift, stop.steps,
                stop.interior_close ? "true" : "false");
    return stop.label == 5;
}

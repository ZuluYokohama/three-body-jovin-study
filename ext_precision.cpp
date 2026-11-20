// Independent IEEE binary128 direct-coordinate diagnostic for the equal-mass plane.
// Build on Linux with g++ -std=gnu++17 -O3 ext_precision.cpp -lquadmath -o ext_precision.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <quadmath.h>

using Real = __float128;
using State = std::array<Real, 12>;
constexpr Real cutoff = 0.001Q;
constexpr int levels = 8;

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

Real distance(const State& q, int i, int j) {
    const Real x = q[i] - q[j], y = q[i + 1] - q[j + 1];
    return sqrtq(x*x + y*y);
}

Real energy(const State& q) {
    Real e = 0;
    for (int i = 0; i < 6; ++i) e += q[i + 6]*q[i + 6]/2;
    return e - 1/distance(q,0,2) - 1/distance(q,0,4) - 1/distance(q,2,4);
}

int outcome(const State& q) {
    if (fminq(distance(q,0,2), fminq(distance(q,0,4), distance(q,2,4))) <= cutoff) return 4;
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

struct Stop { State q; Real time, energy_drift; int label, steps; };

Stop integrate(State q, Real horizon, Real tolerance) {
    const Real initial_energy = energy(q);
    Real time = 0, h = 0.02Q, drift = 0;
    int steps = 0;
    while (time < horizon && steps < 500000) {
        ++steps;
        const Real d = fminq(distance(q,0,2), fminq(distance(q,0,4), distance(q,2,4)));
        Real span = fminq(horizon - time, fminq(h, 0.08Q*d*sqrtq(d/2)));
        const Trial trial = extrapolate(q, span, tolerance);
        if (!finiteq(trial.error) || trial.error > 1) {
            h = span / 2;
            if (h < 1e-25Q) break;
            continue;
        }
        int label = outcome(trial.q);
        State accepted = trial.q;
        if (label) {
            Real low = 0;
            for (int iteration = 0; iteration < 52; ++iteration) {
                const Real mid = (low + span)/2;
                const Trial test = extrapolate(q, mid, tolerance);
                const int event = outcome(test.q);
                if (event) { span = mid; accepted = test.q; label = event; }
                else low = mid;
            }
        }
        drift = fmaxq(drift, fabsq(energy(accepted) - initial_energy) / fmaxq(1, fabsq(initial_energy)));
        time += span;
        q = accepted;
        if (label) return {q,time,drift,label,steps};
        h = fminq(0.04Q, span * (trial.error < 0.02Q ? 2 : (trial.error < 0.5Q ? 1.25Q : 1)));
    }
    return {q,time,drift,time == horizon ? 0 : 5,steps};
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
        const Stop stop = integrate(q, period, 1e-23Q);
        Real error = 0;
        for (int i = 0; i < 12; ++i) error = fmaxq(error, fabsq(q[i]-stop.q[i]));
        std::printf("{\"fixture\":\"equal_mass_equilateral\",\"state_error\":%.8g,\"energy_drift\":%.8g,\"steps\":%d}\n",
                    double(error), double(stop.energy_drift), stop.steps);
        return stop.label != 0 || error > 1e-20Q;
    }
    if (argc != 5) {
        std::fprintf(stderr,"usage: ext_precision x y horizon tolerance\n");
        return 2;
    }
    // Convert through binary64 to lift exactly the two initial coordinates parsed by CUDA.
    const Real x = std::strtod(argv[1], nullptr), y = std::strtod(argv[2], nullptr);
    const Real horizon = strtoflt128(argv[3], nullptr), tolerance = strtoflt128(argv[4], nullptr);
    if (!finiteq(x) || !finiteq(y) || !finiteq(horizon) || !finiteq(tolerance) || horizon <= 0 || tolerance <= 0) return 2;
    const Real cx = (1+x)/3, cy = y/3;
    const State q{-cx,-cy,1-cx,-cy,x-cx,y-cy};
    const Stop stop = integrate(q,horizon,tolerance);
    char time[64], drift[64];
    quadmath_snprintf(time, sizeof(time), "%.25Qg", stop.time);
    quadmath_snprintf(drift, sizeof(drift), "%.9Qg", stop.energy_drift);
    std::printf("{\"precision_bits\":%d,\"label\":%d,\"time\":%s,\"energy_drift\":%s,\"steps\":%d}\n",
                FLT128_MANT_DIG, stop.label, time, drift, stop.steps);
    return stop.label == 5;
}

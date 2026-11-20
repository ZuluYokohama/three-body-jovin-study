#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define LC_HD __host__ __device__
#else
#define LC_HD
#endif

namespace lc {

struct Vec {
    double x;
    double y;
};

LC_HD inline Vec operator+(Vec a, Vec b) { return {a.x + b.x, a.y + b.y}; }
LC_HD inline Vec operator-(Vec a, Vec b) { return {a.x - b.x, a.y - b.y}; }
LC_HD inline Vec operator*(double scale, Vec a) { return {scale * a.x, scale * a.y}; }
LC_HD inline double dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y; }
LC_HD inline double norm2(Vec a) { return dot(a, a); }
LC_HD inline Vec conjugate(Vec a) { return {a.x, -a.y}; }
LC_HD inline Vec multiply(Vec a, Vec b) { return {a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x}; }

struct Jacobi {
    Vec r;
    Vec R;
    Vec p;
    Vec P;
};

struct State {
    Vec u;
    Vec W;
    Vec R;
    Vec P;
    double t;
};

LC_HD inline State enter(Jacobi z, double time) {
    const double radius = sqrt(norm2(z.r));
    Vec u{};
    if (z.r.x >= 0.0) {
        u.x = sqrt(0.5 * (radius + z.r.x));
        u.y = z.r.y / (2.0 * u.x);
    } else {
        u.y = copysign(sqrt(0.5 * (radius - z.r.x)), z.r.y);
        u.x = z.r.y / (2.0 * u.y);
    }
    return {u, 2.0 * multiply(conjugate(u), z.p), z.R, z.P, time};
}

LC_HD inline Jacobi physical(State z) {
    return {multiply(z.u, z.u), z.R, (0.5 / norm2(z.u)) * multiply(z.W, z.u), z.P};
}

LC_HD inline double energy(Jacobi z) {
    return norm2(z.p) + 0.75 * norm2(z.P) - 1.0 / sqrt(norm2(z.r))
        - 1.0 / sqrt(norm2(z.R + 0.5 * z.r)) - 1.0 / sqrt(norm2(z.R - 0.5 * z.r));
}

LC_HD inline double hamiltonian(State z, double E, bool isolated_binary = false) {
    const double rho = norm2(z.u);
    double B = -E;
    if (!isolated_binary) {
        const Vec r = multiply(z.u, z.u);
        B += 0.75 * norm2(z.P) - 1.0 / sqrt(norm2(z.R + 0.5 * r))
            - 1.0 / sqrt(norm2(z.R - 0.5 * r));
    }
    return 0.25 * norm2(z.W) - 1.0 + rho * B;
}

LC_HD inline State rhs(State z, double E, bool isolated_binary = false) {
    const double rho = norm2(z.u);
    if (isolated_binary) {
        return {0.5 * z.W, (2.0 * E) * z.u, {}, {}, rho};
    }
    const Vec half_r = 0.5 * multiply(z.u, z.u);
    const Vec plus = z.R + half_r;
    const Vec minus = z.R - half_r;
    const double plus_inv = 1.0 / sqrt(norm2(plus));
    const double minus_inv = 1.0 / sqrt(norm2(minus));
    const Vec g_plus = (plus_inv * plus_inv * plus_inv) * plus;
    const Vec g_minus = (minus_inv * minus_inv * minus_inv) * minus;
    const double B = 0.75 * norm2(z.P) - plus_inv - minus_inv - E;
    return {0.5 * z.W, (-2.0 * B) * z.u - rho * multiply(conjugate(z.u), g_plus - g_minus),
            (1.5 * rho) * z.P, -rho * (g_plus + g_minus), rho};
}

LC_HD inline State add_scaled(State z, double scale, State direction) {
    return {z.u + scale * direction.u, z.W + scale * direction.W,
            z.R + scale * direction.R, z.P + scale * direction.P, z.t + scale * direction.t};
}

LC_HD inline Vec weighted(Vec a, Vec b, Vec c, Vec d) { return (a + d) + 2.0 * (b + c); }

LC_HD inline State rk4(State z, double ds, double E, bool isolated_binary = false) {
    const State k1 = rhs(z, E, isolated_binary);
    const State k2 = rhs(add_scaled(z, 0.5 * ds, k1), E, isolated_binary);
    const State k3 = rhs(add_scaled(z, 0.5 * ds, k2), E, isolated_binary);
    const State k4 = rhs(add_scaled(z, ds, k3), E, isolated_binary);
    const State total{weighted(k1.u, k2.u, k3.u, k4.u), weighted(k1.W, k2.W, k3.W, k4.W),
                      weighted(k1.R, k2.R, k3.R, k4.R), weighted(k1.P, k2.P, k3.P, k4.P),
                      (k1.t + k4.t) + 2.0 * (k2.t + k3.t)};
    return add_scaled(z, ds / 6.0, total);
}

}  // namespace lc

#undef LC_HD

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>

#include <levi_civita.hpp>

namespace {

using Vector = std::complex<double>;
using Triplet = std::array<Vector, 3>;

struct State {
    Triplet position{};
    Triplet velocity{};
};

constexpr double parity_tolerance = 1e-11;

Vector inverse_square_force(Vector separation) {
    const double distance_squared = std::norm(separation);
    return separation / (distance_squared * std::sqrt(distance_squared));
}

State body_derivative(const State& body) {
    State derivative{body.velocity, {}};
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            if (i != j) {
                derivative.velocity[i] += inverse_square_force(body.position[j] - body.position[i]);
            }
        }
    }
    return derivative;
}

Triplet differences(const Triplet& body) {
    return {body[1] - body[0], body[2] - body[1], body[0] - body[2]};
}

State project(const State& body) {
    return {differences(body.position), differences(body.velocity)};
}

Triplet from_differences(const Triplet& edge) {
    return {(edge[2] - edge[0]) / 3.0, (edge[0] - edge[1]) / 3.0,
            (edge[1] - edge[2]) / 3.0};
}

State reconstruct(const State& edge) {
    return {from_differences(edge.position), from_differences(edge.velocity)};
}

State edge_derivative(const State& edge) {
    const Triplet force{inverse_square_force(edge.position[0]),
                        inverse_square_force(edge.position[1]),
                        inverse_square_force(edge.position[2])};
    const Vector sum = force[0] + force[1] + force[2];
    return {edge.velocity, {sum - 3.0 * force[0], sum - 3.0 * force[1],
                            sum - 3.0 * force[2]}};
}

State isolated_pair_derivative(const State& edge) {
    return {edge.velocity, {-2.0 * inverse_square_force(edge.position[0]),
                            -2.0 * inverse_square_force(edge.position[1]),
                            -2.0 * inverse_square_force(edge.position[2])}};
}

State add_scaled(const State& state, const State& derivative, double scale) {
    State result = state;
    for (std::size_t i = 0; i < 3; ++i) {
        result.position[i] += scale * derivative.position[i];
        result.velocity[i] += scale * derivative.velocity[i];
    }
    return result;
}

template <typename Dynamics>
State rk4(const State& state, double step, Dynamics derivative) {
    const State k1 = derivative(state);
    const State k2 = derivative(add_scaled(state, k1, step / 2.0));
    const State k3 = derivative(add_scaled(state, k2, step / 2.0));
    const State k4 = derivative(add_scaled(state, k3, step));
    State result = add_scaled(state, k1, step / 6.0);
    result = add_scaled(result, k2, step / 3.0);
    result = add_scaled(result, k3, step / 3.0);
    return add_scaled(result, k4, step / 6.0);
}

double normalized_difference(const Triplet& actual, const Triplet& expected) {
    double difference = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        const double scale = std::max({1.0, std::abs(actual[i]), std::abs(expected[i])});
        const double error = std::abs(actual[i] - expected[i]) / scale;
        if (!std::isfinite(error)) {
            return std::numeric_limits<double>::infinity();
        }
        difference = std::max(difference, error);
    }
    return difference;
}

double normalized_difference(const State& actual, const State& expected) {
    return std::max(normalized_difference(actual.position, expected.position),
                    normalized_difference(actual.velocity, expected.velocity));
}

double closure_error(const Triplet& edge) {
    const double scale = std::max({1.0, std::abs(edge[0]), std::abs(edge[1]), std::abs(edge[2])});
    const double error = std::abs(edge[0] + edge[1] + edge[2]) / scale;
    return std::isfinite(error) ? error : std::numeric_limits<double>::infinity();
}

Triplet recenter(Triplet values) {
    const Vector center = (values[0] + values[1] + values[2]) / 3.0;
    for (auto& value : values) {
        value -= center;
    }
    return values;
}

double next_unit(std::uint32_t& seed) {
    seed = 1664525u * seed + 1013904223u;
    return static_cast<double>(seed) / 4294967296.0;
}

State next_fixture(std::uint32_t& seed, std::size_t index) {
    State state;
    const double scale = 0.2 + 2.8 * next_unit(seed);
    const double angle = 2.0 * std::numbers::pi * next_unit(seed);
    const Vector rotation = std::polar(scale, angle);
    const double x = -0.3 + 1.6 * next_unit(seed);
    const double y = (index % 2 == 0 ? 1.0 : -1.0) * (0.25 + next_unit(seed));
    state.position = recenter({Vector{0.0, 0.0}, rotation, rotation * Vector{x, y}});
    if (index % 4 != 0) {
        for (auto& velocity : state.velocity) {
            velocity = {next_unit(seed) - 0.5, next_unit(seed) - 0.5};
        }
        state.velocity = recenter(state.velocity);
    }
    if (index % 3 == 0) {
        std::swap(state.position[1], state.position[2]);
        std::swap(state.velocity[1], state.velocity[2]);
    }
    return state;
}

struct ParityMetrics {
    std::size_t fixtures = 512;
    double reconstruction = 0.0;
    double derivative = 0.0;
    double one_step = 0.0;
    double closure = 0.0;
    double omitted_coupling_error = 0.0;
};

ParityMetrics verify_projection() {
    ParityMetrics metrics;
    std::uint32_t seed = 721013u;
    for (std::size_t index = 0; index < metrics.fixtures; ++index) {
        const State body = next_fixture(seed, index);
        const State edge = project(body);
        const State body_rhs = body_derivative(body);
        const State edge_rhs = edge_derivative(edge);
        const double step = 0.0002 + 0.0008 * next_unit(seed);
        const State stepped_edge = rk4(edge, step, edge_derivative);
        const State stepped_body = rk4(body, step, body_derivative);

        metrics.reconstruction = std::max(metrics.reconstruction,
                                          normalized_difference(reconstruct(edge), body));
        metrics.derivative = std::max({metrics.derivative,
                                       normalized_difference(edge_rhs, project(body_rhs)),
                                       normalized_difference(reconstruct(edge_rhs), body_rhs)});
        metrics.one_step = std::max(metrics.one_step,
                                    normalized_difference(reconstruct(stepped_edge), stepped_body));
        metrics.closure = std::max({metrics.closure, closure_error(stepped_edge.position),
                                    closure_error(stepped_edge.velocity)});
        metrics.omitted_coupling_error = std::max(
            metrics.omitted_coupling_error,
            normalized_difference(isolated_pair_derivative(edge), project(body_rhs)));
    }
    return metrics;
}

State equilateral_initial(double angular_speed) {
    const double height = std::sqrt(3.0) / 2.0;
    State result;
    result.position = {Vector{-0.5, -height / 3.0}, Vector{0.5, -height / 3.0},
                       Vector{0.0, 2.0 * height / 3.0}};
    for (std::size_t i = 0; i < 3; ++i) {
        result.velocity[i] = Vector{0.0, angular_speed} * result.position[i];
    }
    return result;
}

constexpr std::array<std::size_t, 4> orbit_steps{128, 256, 512, 1024};

struct OrbitMetrics {
    std::array<double, orbit_steps.size()> body_errors{};
    std::array<double, orbit_steps.size()> edge_errors{};
};

OrbitMetrics verify_equilateral_orbit() {
    const double angular_speed = std::sqrt(3.0);
    const double period = 2.0 * std::numbers::pi / angular_speed;
    const State initial = equilateral_initial(angular_speed);
    OrbitMetrics metrics;
    for (std::size_t i = 0; i < orbit_steps.size(); ++i) {
        State body = initial;
        State edge = project(initial);
        const double step = period / static_cast<double>(orbit_steps[i]);
        for (std::size_t step_index = 0; step_index < orbit_steps[i]; ++step_index) {
            body = rk4(body, step, body_derivative);
            edge = rk4(edge, step, edge_derivative);
        }
        metrics.body_errors[i] = normalized_difference(body, initial);
        metrics.edge_errors[i] = normalized_difference(reconstruct(edge), initial);
    }
    return metrics;
}

bool fourth_order(const std::array<double, orbit_steps.size()>& errors) {
    if (errors.back() > 1e-7) {
        return false;
    }
    for (std::size_t i = 1; i < errors.size(); ++i) {
        const double ratio = errors[i - 1] / errors[i];
        if (!(ratio > 13.0 && ratio < 19.0)) {
            return false;
        }
    }
    return true;
}

lc::Vec lc_vector(Vector value) {
    return {value.real(), value.imag()};
}

Vector complex_vector(lc::Vec value) {
    return {value.x, value.y};
}

lc::Jacobi to_jacobi(const State& body, std::size_t i, std::size_t j, std::size_t k) {
    const Vector center = (body.position[i] + body.position[j]) / 2.0;
    const Vector center_velocity = (body.velocity[i] + body.velocity[j]) / 2.0;
    return {lc_vector(body.position[j] - body.position[i]),
            lc_vector(body.position[k] - center),
            lc_vector((body.velocity[j] - body.velocity[i]) / 2.0),
            lc_vector((body.velocity[k] - center_velocity) * (2.0 / 3.0))};
}

State from_jacobi(const lc::Jacobi& jacobi, std::size_t i, std::size_t j, std::size_t k) {
    const Vector r = complex_vector(jacobi.r);
    const Vector R = complex_vector(jacobi.R);
    const Vector p = complex_vector(jacobi.p);
    const Vector P = complex_vector(jacobi.P);
    State body{};
    body.position[i] = -R / 3.0 - r / 2.0;
    body.position[j] = -R / 3.0 + r / 2.0;
    body.position[k] = 2.0 * R / 3.0;
    body.velocity[i] = -P / 2.0 - p;
    body.velocity[j] = -P / 2.0 + p;
    body.velocity[k] = P;
    return body;
}

double energy(const State& body) {
    double total = 0.0;
    for (std::size_t i = 0; i < 3; ++i) {
        total += 0.5 * std::norm(body.velocity[i]);
        for (std::size_t j = i + 1; j < 3; ++j) {
            total -= 1.0 / std::abs(body.position[j] - body.position[i]);
        }
    }
    return total;
}

double finite_error(double error) {
    return std::isfinite(error) ? std::abs(error) : std::numeric_limits<double>::infinity();
}

// Differentiate the reconstruction with the complex quotient rule. Compare its
// physical-time derivative with the independent direct Newtonian body equations.
lc::Jacobi projected_rhs(const lc::State& chart, const lc::State& direction) {
    const Vector u = complex_vector(chart.u);
    const Vector W = complex_vector(chart.W);
    const Vector du = complex_vector(direction.u);
    const Vector dW = complex_vector(direction.W);
    const Vector ubar = std::conj(u);
    const double rho = std::norm(u);
    const Vector dr = 2.0 * u * du / rho;
    const Vector dp = (dW * ubar - W * std::conj(du)) / (2.0 * ubar * ubar * rho);
    return {lc_vector(dr), lc_vector(complex_vector(direction.R) / rho),
            lc_vector(dp), lc_vector(complex_vector(direction.P) / rho)};
}

lc::State other_sheet(lc::State state) {
    state.u = {-state.u.x, -state.u.y};
    state.W = {-state.W.x, -state.W.y};
    return state;
}

struct ChartMetrics {
    std::size_t states = 0;
    double reconstruction = 0.0;
    double derivative = 0.0;
    double clock = 0.0;
    double entry_constraint = 0.0;
    double sheet_parity = 0.0;
    double analytic_crossing = 0.0;
    double analytic_clock = 0.0;
    double coupled_passage = 0.0;
    double coupled_constraint = 0.0;
    double coupled_minimum = 1.0;
    bool crossed = true;
    bool coupled_exit = true;
};

void check_chart_states(ChartMetrics& metrics) {
    std::uint32_t seed = 716002u;
    for (std::size_t n = 0; n < 160; ++n) {
        const State body = next_fixture(seed, n);
        const State expected_rhs = body_derivative(body);
        for (std::size_t i = 0; i < 3; ++i) {
            const std::size_t j = (i + 1) % 3;
            const std::size_t k = (i + 2) % 3;
            const lc::Jacobi jacobi = to_jacobi(body, i, j, k);
            const double h = lc::energy(jacobi);
            const lc::State positive = lc::enter(jacobi, 0.25);
            const lc::State negative = other_sheet(positive);
            for (const lc::State& state : {positive, negative}) {
                ++metrics.states;
                const lc::State rhs = lc::rhs(state, h);
                const double rho = std::norm(complex_vector(state.u));
                metrics.reconstruction = std::max(metrics.reconstruction,
                    normalized_difference(from_jacobi(lc::physical(state), i, j, k), body));
                metrics.derivative = std::max(metrics.derivative,
                    normalized_difference(from_jacobi(projected_rhs(state, rhs), i, j, k), expected_rhs));
                metrics.clock = std::max({metrics.clock, finite_error(rhs.t - rho),
                    finite_error(state.t - 0.25)});
                metrics.entry_constraint = std::max({metrics.entry_constraint,
                    finite_error(lc::hamiltonian(state, h)), finite_error(h - energy(body))});
            }
            const lc::State plus_next = lc::rk4(positive, 0.0001, h);
            const lc::State minus_next = lc::rk4(negative, 0.0001, h);
            metrics.sheet_parity = std::max({metrics.sheet_parity, finite_error(plus_next.t - minus_next.t),
                normalized_difference(from_jacobi(lc::physical(plus_next), i, j, k),
                                      from_jacobi(lc::physical(minus_next), i, j, k))});
        }
    }
}

void check_binary_crossing(ChartMetrics& metrics) {
    constexpr int steps = 2048;
    const double end = std::numbers::pi;
    const double ds = end / steps;
    for (double sign : {1.0, -1.0}) {
        // H=-1 with u(0)=±1 and W(0)=0 gives u=±cos(s), W=∓2sin(s).
        lc::State state{{sign, 0.0}, {}, {}, {}, 0.0};
        double last_time = 0.0;
        for (int step = 1; step <= steps; ++step) {
            state = lc::rk4(state, ds, -1.0, true);
            const double s = step * ds;
            const double expected_time = s / 2.0 + std::sin(2.0 * s) / 4.0;
            metrics.analytic_crossing = std::max({metrics.analytic_crossing,
                finite_error(std::abs(complex_vector(state.u) - sign * std::cos(s))),
                finite_error(std::abs(complex_vector(state.W) + 2.0 * sign * std::sin(s))),
                finite_error(lc::hamiltonian(state, -1.0, true))});
            metrics.analytic_clock = std::max(metrics.analytic_clock, finite_error(state.t - expected_time));
            metrics.crossed &= state.t >= last_time;
            last_time = state.t;
            if (step == steps / 2) {
                metrics.crossed &= std::abs(complex_vector(state.u)) < parity_tolerance &&
                                   std::abs(complex_vector(state.W)) > 1.99;
            }
        }
        metrics.crossed &= state.u.x * sign < -0.99;
    }
}

void check_coupled_passage(ChartMetrics& metrics) {
    for (std::size_t i = 0; i < 3; ++i) {
        const std::size_t j = (i + 1) % 3;
        const std::size_t k = (i + 2) % 3;
        const lc::Jacobi jacobi{{0.045, 0.0}, {0.45, 0.7}, {-4.75, 2.0}, {-0.02, 0.05}};
        const State initial = from_jacobi(jacobi, i, j, k);
        const double h = lc::energy(jacobi);
        const lc::State start = lc::enter(jacobi, 0.0);
        for (lc::State chart : {start, other_sheet(start)}) {
            double local_minimum = std::abs(complex_vector(jacobi.r));
            for (int step = 0; step < 20000; ++step) {
                chart = lc::rk4(chart, 0.0001, h);
                const double separation = std::norm(complex_vector(chart.u));
                local_minimum = std::min(local_minimum, separation);
                metrics.coupled_constraint = std::max(metrics.coupled_constraint,
                    finite_error(lc::hamiltonian(chart, h)));
                if (separation >= 0.075) {
                    break;
                }
            }
            metrics.coupled_minimum = std::min(metrics.coupled_minimum, local_minimum);
            metrics.coupled_exit &= std::norm(complex_vector(chart.u)) >= 0.075 &&
                                    local_minimum > 0.001 && local_minimum < 0.015 &&
                                    chart.t > 0.0 && std::isfinite(chart.t);
            if (!metrics.coupled_exit) {
                continue;
            }
            // Resolve the same physical time with the original direct-coordinate RK4 oracle.
            constexpr int body_steps = 16384;
            State body = initial;
            for (int step = 0; step < body_steps; ++step) {
                body = rk4(body, chart.t / body_steps, body_derivative);
            }
            metrics.coupled_passage = std::max(metrics.coupled_passage,
                normalized_difference(from_jacobi(lc::physical(chart), i, j, k), body));
        }
    }
}

bool chart_passed(const ChartMetrics& chart) {
    return chart.states == 960 && chart.reconstruction < parity_tolerance &&
           chart.derivative < parity_tolerance && chart.clock < parity_tolerance &&
           chart.entry_constraint < parity_tolerance && chart.sheet_parity < parity_tolerance &&
           chart.crossed && chart.analytic_crossing < parity_tolerance &&
           chart.analytic_clock < parity_tolerance && chart.coupled_exit &&
           chart.coupled_passage < 1e-9 && chart.coupled_constraint < parity_tolerance;
}

template <typename T, std::size_t N>
void print_array(const std::array<T, N>& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < N; ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << values[i];
    }
    std::cout << ']';
}

}  // namespace

int main() {
    const auto parity = verify_projection();
    const auto orbit = verify_equilateral_orbit();
    ChartMetrics chart;
    check_chart_states(chart);
    check_binary_crossing(chart);
    check_coupled_passage(chart);
    const bool passed = parity.reconstruction < parity_tolerance &&
                        parity.derivative < parity_tolerance &&
                        parity.one_step < parity_tolerance &&
                        parity.closure < parity_tolerance &&
                        parity.omitted_coupling_error > 0.01 &&
                        fourth_order(orbit.body_errors) && fourth_order(orbit.edge_errors) &&
                        chart_passed(chart);
    std::cout << std::setprecision(17) << std::boolalpha
              << "{\"passed\":" << passed << ",\"fixtures\":" << parity.fixtures
              << ",\"normalized_tolerance\":" << parity_tolerance
              << ",\"max_reconstruction_error\":" << parity.reconstruction
              << ",\"max_derivative_error\":" << parity.derivative
              << ",\"max_same_step_error\":" << parity.one_step
              << ",\"max_closure_error\":" << parity.closure
              << ",\"max_omitted_coupling_error\":" << parity.omitted_coupling_error
              << ",\"equilateral_steps\":";
    print_array(orbit_steps);
    std::cout << ",\"equilateral_body_errors\":";
    print_array(orbit.body_errors);
    std::cout << ",\"equilateral_edge_errors\":";
    print_array(orbit.edge_errors);
    std::cout << ",\"levi_civita\":{\"passed\":" << chart_passed(chart)
              << ",\"chart_states\":" << chart.states
              << ",\"max_reconstruction_error\":" << chart.reconstruction
              << ",\"max_newtonian_derivative_error\":" << chart.derivative
              << ",\"max_clock_rhs_error\":" << chart.clock
              << ",\"max_entry_constraint_error\":" << chart.entry_constraint
              << ",\"max_sheet_parity_error\":" << chart.sheet_parity
              << ",\"radial_crossed_on_both_sheets\":" << chart.crossed
              << ",\"max_analytic_radial_error\":" << chart.analytic_crossing
              << ",\"max_analytic_radial_clock_error\":" << chart.analytic_clock
              << ",\"coupled_exit\":" << chart.coupled_exit
              << ",\"coupled_minimum_separation\":" << chart.coupled_minimum
              << ",\"max_coupled_passage_error\":" << chart.coupled_passage
              << ",\"max_coupled_constraint_error\":" << chart.coupled_constraint << "}}\n";
    return passed ? 0 : 1;
}

#include <cuda_runtime.h>
#include <math_constants.h>

#include "levi_civita.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr double close_distance = 1e-3;
constexpr double escape_distance = 10.0;
constexpr double minimum_step = 1e-12;
constexpr int maximum_attempts = 1'000'000;
constexpr int attempts_per_launch = 256;
constexpr int threads_per_block = 128;
constexpr double chart_entry_distance = 0.05;
constexpr double chart_exit_distance = 0.075;
constexpr double maximum_chart_step = 0.05;

enum Label : int {
    active = -1,
    horizon_reached = 0,
    close_approach = 4,
    unresolved = 5,
};

struct Settings {
    double relative_tolerance = 1e-9;
    double absolute_tolerance = 1e-11;
    double maximum_step = 0.04;
    double horizon = 50.0;
    bool levi_civita = false;
};

using lc::Vec;

// The third directed edge is always the negative sum of the first two.
// Storing two edges makes both triangle closure relations exact by construction.
struct Edges {
    Vec r12;
    Vec r23;
    Vec w12;
    Vec w23;
};

struct Trajectory {
    Edges z;
    double time;
    double step;
    double min_r;
    double initial_energy;
    double energy_error;
    unsigned int evaluations;
    int attempts;
    int label;
    lc::State chart;
    double chart_energy;
    double chart_step;
    double max_lc_residual;
    int chart_pair;
    unsigned int chart_entries;
    unsigned int chart_exits;
};

struct Input {
    std::uint64_t id;
    double x;
    double y;
};

struct Geometry {
    Vec r31;
    Vec w31;
    double d12;
    double d23;
    double d31;
};

__device__ Geometry geometry(Edges z) {
    const Vec r31 = -1.0 * (z.r12 + z.r23);
    const Vec w31 = -1.0 * (z.w12 + z.w23);
    return {r31, w31, sqrt(dot(z.r12, z.r12)), sqrt(dot(z.r23, z.r23)),
            sqrt(dot(r31, r31))};
}

__device__ double min_distance(Geometry g) { return fmin(g.d12, fmin(g.d23, g.d31)); }

__device__ double energy(Edges z, Geometry g) {
    const double kinetic = (dot(z.w12, z.w12) + dot(z.w23, z.w23) + dot(g.w31, g.w31)) / 6.0;
    return kinetic - 1.0 / g.d12 - 1.0 / g.d23 - 1.0 / g.d31;
}

__device__ bool escape_candidate(Vec relative_q, Vec relative_v, Vec pair_velocity,
                                double pair_distance) {
    const double distance2 = dot(relative_q, relative_q);
    if (distance2 < escape_distance * escape_distance || distance2 < 100.0 * pair_distance * pair_distance ||
        dot(relative_q, relative_v) <= 0.0) {
        return false;
    }
    const double binary_energy = 0.5 * dot(pair_velocity, pair_velocity) - 2.0 / pair_distance;
    const double escape_energy = 0.5 * dot(relative_v, relative_v) - 3.0 / sqrt(distance2);
    return binary_energy < 0.0 && escape_energy > 0.0;
}

__device__ int outcome(Edges z, Geometry g) {
    if (!isfinite(g.d12) || !isfinite(g.d23) || !isfinite(g.d31)) {
        return unresolved;
    }
    if (min_distance(g) <= close_distance) {
        return close_approach;
    }
    // Multiplying the center-of-mass reconstruction by 3/2 gives the
    // displacement and velocity of each candidate relative to the other pair.
    const bool first = escape_candidate(0.5 * (g.r31 - z.r12), 0.5 * (g.w31 - z.w12), z.w23, g.d23);
    const bool second = escape_candidate(0.5 * (z.r12 - z.r23), 0.5 * (z.w12 - z.w23), g.w31, g.d31);
    const bool third = escape_candidate(0.5 * (z.r23 - g.r31), 0.5 * (z.w23 - g.w31), z.w12, g.d12);
    if (int(first) + int(second) + int(third) == 1) {
        return first ? 1 : (second ? 2 : 3);
    }
    return active;
}

__device__ Vec force(Vec r) {
    const double distance2 = dot(r, r);
    return (1.0 / (distance2 * sqrt(distance2))) * r;
}

__device__ Edges derivative(Edges z) {
    const Vec f12 = force(z.r12);
    const Vec f23 = force(z.r23);
    const Vec f31 = force(-1.0 * (z.r12 + z.r23));
    // This is -A*A^T*f on the two stored edges. In particular, each relative
    // acceleration includes contributions from both of the other pairs.
    return {z.w12, z.w23, (f23 + f31) - 2.0 * f12, (f12 + f31) - 2.0 * f23};
}

__device__ Edges add_scaled(Edges base, double scale, Edges delta) {
    return {base.r12 + scale * delta.r12, base.r23 + scale * delta.r23,
            base.w12 + scale * delta.w12, base.w23 + scale * delta.w23};
}

__device__ Edges rk4(Edges z, double step, unsigned int &evaluations) {
    const Edges k1 = derivative(z);
    const Edges k2 = derivative(add_scaled(z, 0.5 * step, k1));
    const Edges k3 = derivative(add_scaled(z, 0.5 * step, k2));
    const Edges k4 = derivative(add_scaled(z, step, k3));
    evaluations += 4;
    const Edges total{lc::weighted(k1.r12, k2.r12, k3.r12, k4.r12),
                      lc::weighted(k1.r23, k2.r23, k3.r23, k4.r23),
                      lc::weighted(k1.w12, k2.w12, k3.w12, k4.w12),
                      lc::weighted(k1.w23, k2.w23, k3.w23, k4.w23)};
    return add_scaled(z, step / 6.0, total);
}

__device__ double component_error(double old, double coarse, double fine, Settings settings) {
    if (!isfinite(coarse) || !isfinite(fine)) {
        return CUDART_INF;
    }
    const double scale = settings.absolute_tolerance + settings.relative_tolerance * fmax(fabs(old), fabs(fine));
    return fabs(fine - coarse) / (15.0 * scale);
}

__device__ double vector_error(Vec old, Vec coarse, Vec fine, Settings settings) {
    return fmax(component_error(old.x, coarse.x, fine.x, settings),
                component_error(old.y, coarse.y, fine.y, settings));
}

__device__ double error_norm(Edges old, Edges coarse, Edges fine, Settings settings) {
    return fmax(fmax(vector_error(old.r12, coarse.r12, fine.r12, settings),
                     vector_error(old.r23, coarse.r23, fine.r23, settings)),
                fmax(vector_error(old.w12, coarse.w12, fine.w12, settings),
                     vector_error(old.w23, coarse.w23, fine.w23, settings)));
}

__device__ double crossing_limit(double distance, Vec velocity) {
    const double gravity_time = distance * sqrt(0.5 * distance);
    const double speed2 = dot(velocity, velocity);
    const double motion_time = speed2 > 0.0 ? distance / sqrt(speed2) : CUDART_INF;
    return 0.08 * fmin(gravity_time, motion_time);
}

__device__ void observe(Trajectory &trajectory, Edges z, Geometry g) {
    trajectory.min_r = fmin(trajectory.min_r, min_distance(g));
    const double current = energy(z, g);
    if (isfinite(current) && isfinite(trajectory.initial_energy)) {
        const double relative = fabs(current - trajectory.initial_energy) / fmax(1.0, fabs(trajectory.initial_energy));
        trajectory.energy_error = fmax(trajectory.energy_error, relative);
    }
}

__device__ void finish_event(Trajectory &trajectory, Edges start, double time, double step,
                             Edges end, int label) {
    double low = 0.0;
    double high = step;
    const double width = fmax(1e-11, step * 1e-7);
    for (int iteration = 0; iteration < 24 && high - low > width; ++iteration) {
        const double mid = 0.5 * (low + high);
        const Edges candidate = rk4(start, mid, trajectory.evaluations);
        const int candidate_label = outcome(candidate, geometry(candidate));
        if (candidate_label == active) {
            low = mid;
        } else {
            high = mid;
            end = candidate;
            label = candidate_label;
        }
    }
    trajectory.z = end;
    trajectory.time = time + high;
    trajectory.label = label;
    observe(trajectory, end, geometry(end));
}

__device__ lc::Jacobi jacobi(Edges z, Geometry g, int pair) {
    const Vec r[]{z.r12, z.r23, g.r31};
    const Vec w[]{z.w12, z.w23, g.w31};
    const int next = (pair + 1) % 3;
    const int previous = (pair + 2) % 3;
    return {r[pair], 0.5 * (r[next] - r[previous]), 0.5 * w[pair],
            (1.0 / 3.0) * (w[next] - w[previous])};
}

__device__ Edges edge_state(lc::State state, int pair) {
    const lc::Jacobi coordinates = lc::physical(state);
    const Vec r[]{coordinates.r, coordinates.R - 0.5 * coordinates.r,
                  -1.0 * coordinates.R - 0.5 * coordinates.r};
    const Vec w[]{2.0 * coordinates.p, 1.5 * coordinates.P - coordinates.p,
                  -1.5 * coordinates.P - coordinates.p};
    const int first = (3 - pair) % 3;
    const int second = (4 - pair) % 3;
    return {r[first], r[second], w[first], w[second]};
}

__device__ void begin_chart(Trajectory &trajectory, Geometry g, int pair) {
    const lc::Jacobi coordinates = jacobi(trajectory.z, g, pair);
    trajectory.chart = lc::enter(coordinates, 0.0);
    trajectory.chart_energy = lc::energy(coordinates);
    trajectory.chart_step = maximum_chart_step;
    trajectory.chart_pair = pair;
    ++trajectory.chart_entries;
}

__device__ double chart_radius(lc::State state) { return lc::norm2(state.u); }

struct ChartGeometry {
    double binary;
    double third_a;
    double third_b;
};

__device__ ChartGeometry chart_geometry(lc::State state) {
    const Vec half_r = 0.5 * lc::multiply(state.u, state.u);
    return {chart_radius(state), sqrt(lc::norm2(state.R + half_r)), sqrt(lc::norm2(state.R - half_r))};
}

__device__ bool ends_chart(ChartGeometry g) {
    return g.binary >= chart_exit_distance || fmin(g.third_a, g.third_b) < 2.0 * g.binary;
}

constexpr int chart_exit_event = 6;

__device__ int chart_action(lc::State state, const Trajectory &trajectory, Settings settings) {
    const ChartGeometry g = chart_geometry(state);
    if (!isfinite(g.binary) || !isfinite(g.third_a) || !isfinite(g.third_b) || !isfinite(state.t)) {
        return unresolved;
    }
    if (fmin(g.binary, fmin(g.third_a, g.third_b)) <= close_distance) {
        return close_approach;
    }
    const Edges reconstructed = edge_state(state, trajectory.chart_pair);
    const int result = outcome(reconstructed, geometry(reconstructed));
    if (result != active) {
        return result;
    }
    if (trajectory.time + state.t >= settings.horizon) {
        return horizon_reached;
    }
    return ends_chart(g) ? chart_exit_event : active;
}

__device__ lc::State chart_step(lc::State state, double step, Trajectory &trajectory) {
    trajectory.evaluations += 4;
    return lc::rk4(state, step, trajectory.chart_energy);
}

__device__ double chart_error(lc::State start, lc::State coarse, lc::State fine, Settings settings) {
    const double regularized_error = fmax(vector_error(start.u, coarse.u, fine.u, settings),
                                          vector_error(start.W, coarse.W, fine.W, settings));
    const double third_body_error = fmax(vector_error(start.R, coarse.R, fine.R, settings),
                                         vector_error(start.P, coarse.P, fine.P, settings));
    return fmax(fmax(regularized_error, third_body_error), component_error(0.0, coarse.t, fine.t, settings));
}

__device__ void observe_chart(Trajectory &trajectory, lc::State state) {
    const ChartGeometry g = chart_geometry(state);
    trajectory.min_r = fmin(trajectory.min_r, fmin(g.binary, fmin(g.third_a, g.third_b)));
    const double residual = lc::hamiltonian(state, trajectory.chart_energy);
    const double physical_energy = trajectory.chart_energy + residual / g.binary;
    const double denominator = fmax(1.0, fabs(trajectory.initial_energy));
    trajectory.energy_error = fmax(trajectory.energy_error,
                                   fabs(physical_energy - trajectory.initial_energy) / denominator);
    trajectory.max_lc_residual = fmax(trajectory.max_lc_residual, fabs(residual));
}

struct ChartEvent {
    lc::State state;
    int action;
};

__device__ double radial_derivative(lc::State state) { return lc::dot(state.u, state.W); }

__device__ ChartEvent check_chart_segment(Trajectory &trajectory, lc::State start, lc::State end,
                                          double step, Settings settings) {
    double high = step;
    int action = chart_action(end, trajectory, settings);
    double minimum_radius = CUDART_INF;
    double minimum_time = CUDART_INF;

    // The close-distance surface can be crossed twice within one half-step.
    // A bracket on the radial derivative exposes that grazing minimum.
    if (chart_radius(end) > close_distance && radial_derivative(start) < 0.0 &&
        radial_derivative(end) >= 0.0) {
        double low = 0.0;
        double upper = step;
        lc::State minimum = end;
        for (int iteration = 0; iteration < 24 && upper - low > 1e-10; ++iteration) {
            const double mid = 0.5 * (low + upper);
            const lc::State candidate = chart_step(start, mid, trajectory);
            if (radial_derivative(candidate) < 0.0) {
                low = mid;
            } else {
                upper = mid;
                minimum = candidate;
            }
        }
        minimum_radius = chart_radius(minimum);
        minimum_time = minimum.t;
        if (minimum_radius <= close_distance) {
            high = upper;
            end = minimum;
            action = close_approach;
        }
    }

    if (action != active) {
        double low = 0.0;
        const double width = fmax(1e-11, step * 1e-7);
        for (int iteration = 0; iteration < 24 && high - low > width; ++iteration) {
            const double mid = 0.5 * (low + high);
            const lc::State candidate = chart_step(start, mid, trajectory);
            const int candidate_action = chart_action(candidate, trajectory, settings);
            if (candidate_action == active) {
                low = mid;
            } else {
                high = mid;
                end = candidate;
                action = candidate_action;
            }
        }
    }
    if (minimum_time <= end.t) {
        trajectory.min_r = fmin(trajectory.min_r, minimum_radius);
    }
    return {end, action};
}

__device__ void accept_chart(Trajectory &trajectory, ChartEvent event, Settings settings) {
    if (event.action == unresolved || !(chart_radius(event.state) > 0.0)) {
        trajectory.label = unresolved;
        return;
    }
    observe_chart(trajectory, event.state);
    trajectory.z = edge_state(event.state, trajectory.chart_pair);
    trajectory.time = event.action == horizon_reached ? settings.horizon : trajectory.time + event.state.t;
    trajectory.chart = event.state;
    trajectory.chart.t = 0.0;
    if (event.action == chart_exit_event) {
        trajectory.chart_pair = -1;
        trajectory.step = settings.maximum_step;
        ++trajectory.chart_exits;
    } else {
        trajectory.label = event.action;
    }
}

__device__ void attempt_chart(Trajectory &trajectory, Settings settings) {
    const lc::State start = trajectory.chart;
    const ChartGeometry g = chart_geometry(start);
    const double third_distance = fmin(g.third_a, g.third_b);
    const lc::Jacobi coordinates = lc::physical(start);
    const double third_limit = fmin(crossing_limit(g.third_a, 1.5 * coordinates.P + coordinates.p),
                                    crossing_limit(g.third_b, 1.5 * coordinates.P - coordinates.p));
    const double physical_limit = fmin(settings.maximum_step, third_limit);
    const double step = fmin(trajectory.chart_step, fmin(maximum_chart_step, physical_limit / g.binary));
    if (!(step >= minimum_step) || !isfinite(third_distance) || third_distance <= close_distance) {
        trajectory.label = unresolved;
        return;
    }

    ++trajectory.attempts;
    const lc::State coarse = chart_step(start, step, trajectory);
    const double half = 0.5 * step;
    const lc::State middle = chart_step(start, half, trajectory);
    const lc::State fine = chart_step(middle, half, trajectory);
    const double error = chart_error(start, coarse, fine, settings);
    const double factor = error > 0.0 ? 0.9 * pow(error, -0.2) : 2.0;
    trajectory.chart_step = step * fmax(0.1, fmin(2.0, factor));
    if (error > 1.0) {
        return;
    }
    if (!(middle.t > 0.0) || !(fine.t > middle.t) || fine.t > physical_limit) {
        trajectory.chart_step = fmin(trajectory.chart_step, 0.5 * step);
        return;
    }
    const ChartEvent first = check_chart_segment(trajectory, start, middle, half, settings);
    if (first.action != active) {
        accept_chart(trajectory, first, settings);
        return;
    }
    observe_chart(trajectory, middle);
    const ChartEvent second = check_chart_segment(trajectory, middle, fine, half, settings);
    accept_chart(trajectory, second, settings);
}

template <bool UseLC>
__device__ void attempt(Trajectory &trajectory, Settings settings) {
    if (trajectory.attempts >= maximum_attempts) {
        trajectory.label = unresolved;
        return;
    }
    if constexpr (UseLC) {
        if (trajectory.chart_pair >= 0) {
            attempt_chart(trajectory, settings);
            return;
        }
    }
    const Edges z = trajectory.z;
    const Geometry g = geometry(z);
    if constexpr (UseLC) {
        const double distance[]{g.d12, g.d23, g.d31};
        int pair = 0;
        for (int index = 1; index < 3; ++index) {
            if (distance[index] < distance[pair]) {
                pair = index;
            }
        }
        if (distance[pair] <= chart_entry_distance &&
            fmin(distance[(pair + 1) % 3], distance[(pair + 2) % 3]) >= 4.0 * distance[pair]) {
            begin_chart(trajectory, g, pair);
            attempt_chart(trajectory, settings);
            return;
        }
    }
    const double local_limit = fmin(crossing_limit(g.d12, z.w12),
                                    fmin(crossing_limit(g.d23, z.w23), crossing_limit(g.d31, g.w31)));
    const double remaining = settings.horizon - trajectory.time;
    const double step = fmin(remaining, fmin(settings.maximum_step, fmin(trajectory.step, local_limit)));
    if (!(step > 0.0) || trajectory.time + step == trajectory.time ||
        (step < minimum_step && step < remaining)) {
        trajectory.label = unresolved;
        return;
    }

    ++trajectory.attempts;
    const Edges coarse = rk4(z, step, trajectory.evaluations);
    const double half = 0.5 * step;
    const Edges middle = rk4(z, half, trajectory.evaluations);
    const Edges fine = rk4(middle, half, trajectory.evaluations);
    const double error = error_norm(z, coarse, fine, settings);
    const double factor = error > 0.0 ? 0.9 * pow(error, -0.2) : 2.0;
    trajectory.step = step * fmax(0.1, fmin(2.0, factor));
    if (error > 1.0) {
        return;
    }

    const Geometry mid_geometry = geometry(middle);
    const int mid_label = outcome(middle, mid_geometry);
    if (mid_label != active) {
        finish_event(trajectory, z, trajectory.time, half, middle, mid_label);
        return;
    }
    observe(trajectory, middle, mid_geometry);
    const Geometry end_geometry = geometry(fine);
    const int end_label = outcome(fine, end_geometry);
    if (end_label != active) {
        finish_event(trajectory, middle, trajectory.time + half, half, fine, end_label);
        return;
    }
    trajectory.z = fine;
    trajectory.time = step == remaining ? settings.horizon : trajectory.time + step;
    observe(trajectory, fine, end_geometry);
    if (trajectory.time == settings.horizon) {
        trajectory.label = horizon_reached;
    }
}

__global__ void initialize(const Input *inputs, Trajectory *trajectories, int count, Settings settings,
                           unsigned int *unfinished) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const Input input = inputs[index];
    const Edges z{{1.0, 0.0}, {input.x - 1.0, input.y}, {0.0, 0.0}, {0.0, 0.0}};
    const Geometry g = geometry(z);
    const int label = outcome(z, g);
    trajectories[index] = {z, 0.0, settings.maximum_step, min_distance(g), energy(z, g), 0.0, 0, 0, label,
                           {}, 0.0, 0.0, 0.0, -1, 0, 0};
    if (label != active) {
        atomicSub(unfinished, 1u);
    }
}

template <bool UseLC>
__global__ void advance(Trajectory *trajectories, int count, Settings settings, unsigned int *unfinished) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count || trajectories[index].label != active) {
        return;
    }
    Trajectory trajectory = trajectories[index];
    for (int i = 0; i < attempts_per_launch && trajectory.label == active; ++i) {
        attempt<UseLC>(trajectory, settings);
    }
    trajectories[index] = trajectory;
    if (trajectory.label != active) {
        atomicSub(unfinished, 1u);
    }
}

void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        check_cuda(cudaMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    T *data() const { return data_; }

private:
    T *data_ = nullptr;
};

double parse_number(const std::string &text) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
        throw std::invalid_argument("expected a finite number, got " + text);
    }
    return value;
}

std::uint64_t parse_id(const std::string &text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("expected an unsigned integer id, got " + text);
    }
    return std::stoull(text);
}

std::vector<Input> read_samples(const char *path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error(std::string("cannot open input ") + path);
    }
    std::string line;
    if (!std::getline(stream, line) || (line != "id,x,y" && line != "id,x,y\r")) {
        throw std::invalid_argument("input must have header id,x,y");
    }
    std::vector<Input> result;
    std::unordered_set<std::uint64_t> ids;
    std::size_t row = 1;
    while (std::getline(stream, line)) {
        ++row;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        try {
            std::istringstream fields(line);
            std::string id, x, y;
            if (!std::getline(fields, id, ',') || !std::getline(fields, x, ',') || !std::getline(fields, y) ||
                y.find(',') != std::string::npos) {
                throw std::invalid_argument("expected id,x,y");
            }
            const std::uint64_t sample_id = parse_id(id);
            if (!ids.insert(sample_id).second) {
                throw std::invalid_argument("duplicate sample id");
            }
            result.push_back({sample_id, parse_number(x), parse_number(y)});
        } catch (const std::exception &error) {
            throw std::invalid_argument("input row " + std::to_string(row) + ": " + error.what());
        }
    }
    if (!stream.eof()) {
        throw std::runtime_error("failed while reading samples");
    }
    return result;
}

std::vector<Trajectory> solve(const std::vector<Input> &inputs, Settings settings, int device) {
    if (inputs.empty()) {
        return {};
    }
    if (inputs.size() > std::size_t(std::numeric_limits<int>::max() - threads_per_block)) {
        throw std::invalid_argument("too many samples in input batch");
    }
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    DeviceBuffer<Input> device_inputs(inputs.size());
    DeviceBuffer<Trajectory> device_trajectories(inputs.size());
    DeviceBuffer<unsigned int> device_unfinished(1);
    const int count = static_cast<int>(inputs.size());
    const int blocks = (count + threads_per_block - 1) / threads_per_block;
    unsigned int unfinished = static_cast<unsigned int>(count);
    check_cuda(cudaMemcpy(device_inputs.data(), inputs.data(), inputs.size() * sizeof(Input),
                          cudaMemcpyHostToDevice), "upload samples");
    check_cuda(cudaMemcpy(device_unfinished.data(), &unfinished, sizeof(unfinished), cudaMemcpyHostToDevice),
               "initialize unfinished count");
    initialize<<<blocks, threads_per_block>>>(device_inputs.data(), device_trajectories.data(), count, settings,
                                               device_unfinished.data());
    check_cuda(cudaGetLastError(), "launch initialization");
    // Each launch makes bounded progress. The Windows display driver's watchdog
    // never has to wait for the whole integration of a hard trajectory at once.
    while (unfinished != 0) {
        if (settings.levi_civita) {
            advance<true><<<blocks, threads_per_block>>>(device_trajectories.data(), count, settings,
                                                        device_unfinished.data());
        } else {
            advance<false><<<blocks, threads_per_block>>>(device_trajectories.data(), count, settings,
                                                         device_unfinished.data());
        }
        check_cuda(cudaGetLastError(), "launch trajectory advance");
        check_cuda(cudaMemcpy(&unfinished, device_unfinished.data(), sizeof(unfinished), cudaMemcpyDeviceToHost),
                   "read unfinished count");
    }
    std::vector<Trajectory> result(inputs.size());
    check_cuda(cudaMemcpy(result.data(), device_trajectories.data(), result.size() * sizeof(Trajectory),
                          cudaMemcpyDeviceToHost), "download outcomes");
    return result;
}

void write_results(const char *path, const std::vector<Input> &inputs, const std::vector<Trajectory> &results) {
    std::ofstream stream(path);
    if (!stream) {
        throw std::runtime_error(std::string("cannot open output ") + path);
    }
    stream << "id,label,time,min_r,evals,energy_error,x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3"
              ",chart_entries,chart_exits,max_lc_residual\n"
           << std::setprecision(17);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const Trajectory &result = results[i];
        const Edges z = result.z;
        const Vec r31 = -1.0 * (z.r12 + z.r23);
        const Vec w31 = -1.0 * (z.w12 + z.w23);
        const Vec state[]{(1.0 / 3.0) * (r31 - z.r12), (1.0 / 3.0) * (z.r12 - z.r23),
                          (1.0 / 3.0) * (z.r23 - r31), (1.0 / 3.0) * (w31 - z.w12),
                          (1.0 / 3.0) * (z.w12 - z.w23), (1.0 / 3.0) * (z.w23 - w31)};
        stream << inputs[i].id << ',' << result.label << ',' << result.time << ',' << result.min_r << ','
               << result.evaluations << ',' << result.energy_error;
        for (Vec vector : state) {
            stream << ',' << vector.x << ',' << vector.y;
        }
        stream << ',' << result.chart_entries << ',' << result.chart_exits << ',' << result.max_lc_residual << '\n';
    }
    stream.close();
    if (!stream) {
        throw std::runtime_error(std::string("failed to write output ") + path);
    }
}

constexpr int crossing_fixtures = 4;
constexpr int crossing_steps = 4096;

struct CrossingProof {
    lc::State collision;
    lc::State after;
    double max_constraint_residual;
};

__host__ __device__ Vec crossing_direction(int index) {
    const double sign = index % 2 == 0 ? 1.0 : -1.0;
    return index < 2 ? Vec{sign, 0.0} : Vec{0.0, sign};
}

__global__ void crossing_kernel(double pi, CrossingProof *proofs) {
    const int index = threadIdx.x;
    if (index >= crossing_fixtures) {
        return;
    }
    lc::State state{crossing_direction(index), {}, {}, {}, 0.0};
    CrossingProof proof{};
    for (int step = 1; step <= crossing_steps; ++step) {
        state = lc::rk4(state, pi / crossing_steps, -1.0, true);
        proof.max_constraint_residual = fmax(proof.max_constraint_residual, fabs(lc::hamiltonian(state, -1.0, true)));
        if (step == crossing_steps / 2) {
            proof.collision = state;
        }
    }
    proof.after = state;
    proofs[index] = proof;
}

int verify_lc(int device) {
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    DeviceBuffer<CrossingProof> device_proofs(crossing_fixtures);
    const double pi = std::acos(-1.0);
    crossing_kernel<<<1, threads_per_block>>>(pi, device_proofs.data());
    check_cuda(cudaGetLastError(), "launch analytic crossing");
    CrossingProof proofs[crossing_fixtures];
    check_cuda(cudaMemcpy(proofs, device_proofs.data(), sizeof(proofs), cudaMemcpyDeviceToHost), "read crossing proofs");
    double state_error = 0.0;
    double clock_error = 0.0;
    double constraint = 0.0;
    double maximum_radius_at_collision = 0.0;
    for (int index = 0; index < crossing_fixtures; ++index) {
        const CrossingProof &proof = proofs[index];
        const Vec direction = crossing_direction(index);
        const Vec radial_errors[]{proof.collision.u, proof.collision.W + 2.0 * direction,
                                  proof.after.u + direction, proof.after.W};
        for (Vec error : radial_errors) {
            const double norm = std::hypot(error.x, error.y);
            state_error = std::isfinite(norm) ? std::max(state_error, norm) : std::numeric_limits<double>::max();
        }
        clock_error = std::max({clock_error, std::abs(proof.collision.t - 0.25 * pi),
                                std::abs(proof.after.t - 0.5 * pi)});
        constraint = std::max(constraint, proof.max_constraint_residual);
        maximum_radius_at_collision = std::max(maximum_radius_at_collision, lc::norm2(proof.collision.u));
    }
    const bool passed = state_error <= 1e-10 && clock_error <= 1e-10 && constraint <= 1e-10;
    std::cout << std::setprecision(17) << "{\"passed\":" << (passed ? "true" : "false")
              << ",\"fixtures\":" << crossing_fixtures << ",\"steps\":" << crossing_steps
              << ",\"device\":" << device << ",\"max_state_error\":" << state_error
              << ",\"max_clock_error\":" << clock_error << ",\"max_constraint_residual\":" << constraint
              << ",\"max_radius_at_collision\":" << maximum_radius_at_collision << "}\n";
    return passed ? 0 : 1;
}

int parse_device(const std::string &text) {
    const auto index = parse_id(text);
    if (index > std::uint64_t(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("device index exceeds the integer range");
    }
    return static_cast<int>(index);
}

}  // namespace

int main(int argc, char **argv) {
    try {
        if (argc >= 2 && std::string(argv[1]) == "--verify-lc") {
            if (argc == 2) {
                return verify_lc(0);
            }
            if (argc == 4 && std::string(argv[2]) == "--device") {
                return verify_lc(parse_device(argv[3]));
            }
            throw std::invalid_argument("expected --verify-lc [--device INDEX]");
        }
        if (argc < 3 || argc % 2 == 0) {
            std::cerr << "Usage: solver input.csv output.csv [--rtol VALUE] [--atol VALUE]"
                      << " [--hmax VALUE] [--horizon VALUE] [--device INDEX] [--coordinates edge|levi-civita]\n"
                      << "       solver --verify-lc [--device INDEX]\n";
            return 2;
        }
        Settings settings;
        int device = 0;
        for (int i = 3; i < argc; i += 2) {
            const std::string option = argv[i];
            if (option == "--device") {
                device = parse_device(argv[i + 1]);
            } else if (option == "--coordinates") {
                const std::string value = argv[i + 1];
                if (value != "edge" && value != "levi-civita") {
                    throw std::invalid_argument("coordinates must be edge or levi-civita");
                }
                settings.levi_civita = value == "levi-civita";
            } else {
                const double value = parse_number(argv[i + 1]);
                if (!(value > 0.0)) {
                    throw std::invalid_argument(option + " must be greater than zero");
                }
                if (option == "--rtol") {
                    settings.relative_tolerance = value;
                } else if (option == "--atol") {
                    settings.absolute_tolerance = value;
                } else if (option == "--hmax") {
                    settings.maximum_step = value;
                } else if (option == "--horizon") {
                    settings.horizon = value;
                } else {
                    throw std::invalid_argument("unknown option " + option);
                }
            }
        }
        const auto inputs = read_samples(argv[1]);
        const auto results = solve(inputs, settings, device);
        write_results(argv[2], inputs, results);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "solver: " << error.what() << '\n';
        return 1;
    }
}

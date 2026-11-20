#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int neighbors = 9;
constexpr int labels = 6;
constexpr int unresolved = 5;
constexpr int tile_width = 32;
constexpr int input_stride = 64; // Round the 9 x 6 one-hot channels up for GEMM.
constexpr int output_stride = 16;
constexpr int block_size = 256;
using Clock = std::chrono::steady_clock;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + cublasGetStatusString(status));
    }
}

template<class T>
struct DeviceBuffer {
    explicit DeviceBuffer(std::size_t count) {
        check(cudaMalloc(&data, count * sizeof(T)), "cudaMalloc");
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data = nullptr;
};

struct Blas {
    Blas() { check(cublasCreate(&handle), "cublasCreate"); }
    ~Blas() { cublasDestroy(handle); }
    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;

    cublasHandle_t handle = nullptr;
};

__global__ void fused_mask(const std::int8_t* input, std::uint8_t* output, int rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    unsigned int present = 0;
    for (int slot = 0; slot < neighbors; ++slot) {
        const int label = input[row * neighbors + slot];
        if (label >= 0) present |= 1u << label;
    }
    output[row] = (present & (1u << unresolved)) || __popc(present) > 1;
}

__global__ void one_hot(const std::int8_t* input, float* encoded, int rows, int entries) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= entries) return;

    const int row = index / input_stride;
    const int feature = index % input_stride;
    if (row < rows && feature < neighbors * labels) {
        const int slot = feature / labels;
        const int label = feature % labels;
        encoded[index] = input[row * neighbors + slot] == label ? 1.0f : 0.0f;
    } else {
        encoded[index] = 0.0f;
    }
}

__global__ void mask_from_counts(const float* counts, std::uint8_t* output, int rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    const float* row_counts = counts + row * output_stride;
    int distinct = 0;
    for (int label = 0; label < labels; ++label) {
        distinct += row_counts[label] > 0.0f;
    }
    output[row] = row_counts[unresolved] > 0.0f || distinct > 1;
}

int blocks(int entries) {
    return (entries - 1) / block_size + 1;
}

struct TensorFilter {
    explicit TensorFilter(int rows)
        : columns((rows + tile_width - 1) / tile_width * tile_width),
          projection(output_stride * input_stride), encoded(columns * input_stride),
          counts(columns * output_stride) {
        std::vector<float> matrix(output_stride * input_stride, 0.0f);
        for (int slot = 0; slot < neighbors; ++slot) {
            for (int label = 0; label < labels; ++label) {
                matrix[label + output_stride * (slot * labels + label)] = 1.0f;
            }
        }
        check(cudaMemcpy(projection.data, matrix.data(), matrix.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "upload fixed label projection");
    }

    void run(const std::int8_t* input, std::uint8_t* output, int rows) {
        const int entries = columns * input_stride;
        one_hot<<<blocks(entries), block_size>>>(input, encoded.data, rows, entries);
        check(cudaGetLastError(), "launch one-hot encoding");

        // Every input and coefficient is zero or one. Sums of at most nine terms
        // are exact at TF32 input precision with FP32 accumulation.
        const float alpha = 1.0f;
        const float beta = 0.0f;
        check(cublasGemmEx(blas.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           output_stride, columns, input_stride,
                           &alpha, projection.data, CUDA_R_32F, output_stride,
                           encoded.data, CUDA_R_32F, input_stride,
                           &beta, counts.data, CUDA_R_32F, output_stride,
                           CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
              "cuBLAS label counting");

        mask_from_counts<<<blocks(rows), block_size>>>(counts.data, output, rows);
        check(cudaGetLastError(), "launch refinement mask from counts");
    }

    int columns;
    Blas blas;
    DeviceBuffer<float> projection;
    DeviceBuffer<float> encoded;
    DeviceBuffer<float> counts;
};

template<class Integer>
Integer integer(std::string_view token, const char* field) {
    const auto first = token.find_first_not_of(" \t\r");
    if (first == std::string_view::npos) {
        throw std::runtime_error(std::string("Missing integer for ") + field);
    }
    token = token.substr(first, token.find_last_not_of(" \t\r") - first + 1);
    Integer value{};
    const auto [last, error] = std::from_chars(token.data(), token.data() + token.size(), value);
    if (error != std::errc{} || last != token.data() + token.size()) {
        throw std::runtime_error(std::string("Invalid integer for ") + field + ": " + std::string(token));
    }
    return value;
}

struct Options {
    std::string input;
    std::string output;
    std::string backend = "both";
    int repeat = 5;
    int device = 0;
};

constexpr char usage[] = "Usage: filter INPUT.csv OUTPUT.csv [--repeat N] [--device N] [--backend both|fused|tensor]";

Options parse(int argc, char** argv) {
    if (argc < 3 || argc % 2 == 0) throw std::runtime_error(usage);
    Options options{argv[1], argv[2]};
    for (int arg = 3; arg < argc; arg += 2) {
        const std::string_view flag(argv[arg]);
        if (flag == "--repeat") options.repeat = integer<int>(argv[arg + 1], "repeat");
        else if (flag == "--device") options.device = integer<int>(argv[arg + 1], "device");
        else if (flag == "--backend") options.backend = argv[arg + 1];
        else throw std::runtime_error("Unknown option: " + std::string(flag));
    }
    if (options.repeat < 1 || options.repeat > 1000 || options.device < 0) {
        throw std::runtime_error("Repeat must be within 1..1000 and device must be nonnegative");
    }
    if (options.backend != "both" && options.backend != "fused" && options.backend != "tensor") {
        throw std::runtime_error("Backend must be both, fused or tensor");
    }
    return options;
}

struct Fixtures {
    std::vector<std::int64_t> ids;
    std::vector<std::int8_t> input;
    std::vector<std::uint8_t> expected;
};

Fixtures read_fixtures(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    if (!file || !std::getline(file, line)) throw std::runtime_error("Cannot read input: " + path);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "id,c0,c1,c2,c3,c4,c5,c6,c7,c8") throw std::runtime_error("Invalid filter CSV header");

    Fixtures fixtures;
    while (std::getline(file, line)) {
        std::string_view row(line);
        const auto comma = row.find(',');
        if (comma == std::string_view::npos) throw std::runtime_error("Missing filter CSV fields");
        fixtures.ids.push_back(integer<std::int64_t>(row.substr(0, comma), "id"));
        row.remove_prefix(comma + 1);

        bool seen[labels]{};
        for (int slot = 0; slot < neighbors; ++slot) {
            const auto delimiter = row.find(',');
            const bool is_last = slot == neighbors - 1;
            if ((delimiter == std::string_view::npos) != is_last) {
                throw std::runtime_error("Wrong number of columns for filter id " + std::to_string(fixtures.ids.back()));
            }
            const int label = integer<int>(row.substr(0, delimiter), "neighbor label");
            if (label < -1 || label >= labels) throw std::runtime_error("Neighbor label is outside -1..5");
            fixtures.input.push_back(static_cast<std::int8_t>(label));
            if (label >= 0) seen[label] = true;
            if (!is_last) row.remove_prefix(delimiter + 1);
        }
        fixtures.expected.push_back(seen[unresolved] || std::count(std::begin(seen), std::end(seen), true) > 1);
    }
    if (file.bad()) throw std::runtime_error("I/O error reading filter input");
    const auto max_rows = std::numeric_limits<int>::max() / input_stride - tile_width;
    if (fixtures.ids.empty() || fixtures.ids.size() > static_cast<std::size_t>(max_rows)) {
        throw std::runtime_error("Filter CSV requires at least one row and must fit in GPU index space");
    }
    return fixtures;
}

void write_mask(const std::string& path, const Fixtures& fixtures, const std::vector<std::uint8_t>& mask) {
    std::ofstream file(path);
    if (!file) throw std::runtime_error("Cannot write filter output: " + path);
    file << "id,refine\n";
    for (std::size_t i = 0; i < fixtures.ids.size(); ++i) {
        file << fixtures.ids[i] << ',' << static_cast<int>(mask[i]) << '\n';
    }
    file.close();
    if (!file) throw std::runtime_error("I/O error writing filter output");
}

void stats(std::vector<double> values) {
    if (values.empty()) {
        std::cout << "null";
        return;
    }
    std::sort(values.begin(), values.end());
    const auto count = values.size();
    const double median = (values[(count - 1) / 2] + values[count / 2]) / 2.0;
    const auto p95 = static_cast<std::size_t>(std::ceil(0.95 * count)) - 1;
    std::cout << "{\"min\":" << values.front() << ",\"median\":" << median
              << ",\"p95\":" << values[p95] << '}';
}

std::string json_string(std::string_view input) {
    std::string escaped;
    escaped.push_back('"');
    for (const char character : input) {
        if (character == '\\' || character == '"') escaped.push_back('\\');
        if (static_cast<unsigned char>(character) >= 0x20) escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
}

void benchmark(const Options& options, const Fixtures& fixtures) {
    const int rows = static_cast<int>(fixtures.ids.size());
    check(cudaSetDevice(options.device), "select CUDA device");
    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, options.device), "query CUDA device");

    const auto setup_start = Clock::now();
    const DeviceBuffer<std::int8_t> input(fixtures.input.size());
    const DeviceBuffer<std::uint8_t> output(rows);
    std::optional<TensorFilter> tensor;
    if (options.backend != "fused") tensor.emplace(rows);
    check(cudaDeviceSynchronize(), "finish filter setup");
    const auto setup_ms = std::chrono::duration<double, std::milli>(Clock::now() - setup_start).count();

    std::vector<std::uint8_t> mask(rows);
    const auto run = [&](bool tensor_backend) {
        const auto start = Clock::now();
        check(cudaMemcpy(input.data, fixtures.input.data(), fixtures.input.size(), cudaMemcpyHostToDevice),
              "upload neighborhood labels");
        if (tensor_backend) {
            tensor->run(input.data, output.data, rows);
        } else {
            fused_mask<<<blocks(rows), block_size>>>(input.data, output.data, rows);
            check(cudaGetLastError(), "launch fused refinement mask");
        }
        check(cudaMemcpy(mask.data(), output.data, mask.size(), cudaMemcpyDeviceToHost), "download refinement mask");
        const auto elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        const auto mismatch = std::mismatch(mask.begin(), mask.end(), fixtures.expected.begin());
        if (mismatch.first != mask.end()) {
            const auto index = static_cast<std::size_t>(mismatch.first - mask.begin());
            throw std::runtime_error(std::string(tensor_backend ? "Tensor" : "Fused") +
                                     " mask disagrees with CPU at id " + std::to_string(fixtures.ids[index]));
        }
        return elapsed_ms;
    };

    const bool fused_enabled = options.backend != "tensor";
    if (fused_enabled) run(false);
    if (tensor) run(true);

    std::vector<double> fused_ms;
    std::vector<double> tensor_ms;
    for (int repetition = 0; repetition < options.repeat; ++repetition) {
        if (tensor && repetition % 2 == 1) tensor_ms.push_back(run(true));
        if (fused_enabled) fused_ms.push_back(run(false));
        if (tensor && repetition % 2 == 0) tensor_ms.push_back(run(true));
    }
    write_mask(options.output, fixtures, mask);

    std::cout << std::fixed << std::setprecision(6)
              << "{\"rows\":" << rows << ",\"repeat\":" << options.repeat
              << ",\"device\":" << options.device << ",\"device_name\":" << json_string(device.name)
              << ",\"compute_capability\":\"" << device.major << '.' << device.minor << '"'
              << ",\"backend\":" << json_string(options.backend)
              << ",\"agreement\":true,\"setup_ms\":" << setup_ms
              << ",\"refine_count\":" << std::count(mask.begin(), mask.end(), 1)
              << ",\"fused_ms\":";
    stats(fused_ms);
    std::cout << ",\"tensor_ms\":";
    stats(tensor_ms);
    std::cout << "}\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << usage << '\n';
        return 0;
    }
    try {
        const auto options = parse(argc, argv);
        benchmark(options, read_fixtures(options.input));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "filter: " << error.what() << '\n';
        return 1;
    }
}

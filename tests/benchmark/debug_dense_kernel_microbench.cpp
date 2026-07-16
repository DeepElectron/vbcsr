#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "vbcsr/core/detail/kernels/dense_kernels.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

struct Timing {
    double seconds = 0.0;
    int iterations = 0;
    double checksum = 0.0;
};

struct Args {
    int batch = 4096;
    double min_seconds = 0.20;
    int min_iterations = 5;
    int repeats = 5;
    double beta = 1.0;
    bool multivector_rhs_layout = false;
    unsigned seed = 1729;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key(argv[i]);
        auto next_int = [&](int& value) {
            if (++i >= argc) {
                throw std::runtime_error("missing value for " + key);
            }
            value = std::stoi(argv[i]);
        };
        auto next_double = [&](double& value) {
            if (++i >= argc) {
                throw std::runtime_error("missing value for " + key);
            }
            value = std::stod(argv[i]);
        };
        if (key == "--batch") {
            next_int(args.batch);
        } else if (key == "--min-seconds") {
            next_double(args.min_seconds);
        } else if (key == "--min-iterations") {
            next_int(args.min_iterations);
        } else if (key == "--repeats") {
            next_int(args.repeats);
        } else if (key == "--beta") {
            next_double(args.beta);
        } else if (key == "--rhs-layout") {
            if (++i >= argc) {
                throw std::runtime_error("missing value for " + key);
            }
            const std::string value(argv[i]);
            if (value == "compact") {
                args.multivector_rhs_layout = false;
            } else if (value == "multivector") {
                args.multivector_rhs_layout = true;
            } else {
                throw std::runtime_error("--rhs-layout must be compact or multivector");
            }
        } else if (key == "--seed") {
            int value = static_cast<int>(args.seed);
            next_int(value);
            args.seed = static_cast<unsigned>(value);
        } else {
            throw std::runtime_error("unknown argument " + key);
        }
    }
    if (args.batch <= 0 || args.min_seconds <= 0.0 || args.min_iterations <= 0 || args.repeats <= 0) {
        throw std::runtime_error("batch, min-seconds, min-iterations, and repeats must be positive");
    }
    return args;
}

double checksum_stride(const std::vector<double>& values, int stride) {
    double sum = 0.0;
    const int step = std::max(1, stride);
    for (size_t i = 0; i < values.size(); i += static_cast<size_t>(step)) {
        sum += values[i];
    }
    return sum;
}

template <typename Func>
Timing benchmark(Func&& func, int min_iterations, double min_seconds) {
    int iterations = 0;
    auto start = Clock::now();
    double checksum = 0.0;
    while (true) {
        checksum += func();
        ++iterations;
        const auto now = Clock::now();
        const double elapsed = std::chrono::duration<double>(now - start).count();
        if (iterations >= min_iterations && elapsed >= min_seconds) {
            return {elapsed / static_cast<double>(iterations), iterations, checksum};
        }
    }
}

template <int M, int K>
void fixed_gemv_one(const double* A, const double* x, double* y, double beta) {
    vbcsr::FixedBlockKernel<double, M, K>::gemv(A, x, y, 1.0, beta);
}

template <int M, int K>
void fixed_gemm_one(int rhs, const double* A, const double* B, int ldb, double* C, int ldc, double beta) {
    vbcsr::FixedBlockKernel<double, M, K>::gemm(rhs, A, M, B, ldb, C, ldc, 1.0, beta);
}

void mkl_gemv_one(int m, int k, const double* A, const double* x, double* y, double beta) {
    vbcsr::BLASKernel::gemv(m, k, 1.0, A, m, x, 1, beta, y, 1);
}

void mkl_gemm_one(int m, int rhs, int k, const double* A, const double* B, int ldb, double* C, int ldc, double beta) {
    vbcsr::BLASKernel::gemm(m, rhs, k, 1.0, A, m, B, ldb, beta, C, ldc);
}

template <int M, int K>
Timing run_fixed_gemv(const Args& args, const std::vector<double>& A, const std::vector<double>& x, std::vector<double>& y) {
    return benchmark(
        [&]() {
            for (int b = 0; b < args.batch; ++b) {
                fixed_gemv_one<M, K>(
                    A.data() + static_cast<size_t>(b) * M * K,
                    x.data() + static_cast<size_t>(b) * K,
                    y.data() + static_cast<size_t>(b) * M,
                    args.beta);
            }
            return checksum_stride(y, M + 7);
        },
        args.min_iterations,
        args.min_seconds);
}

Timing run_mkl_direct_gemv(int m, int k, const Args& args, const std::vector<double>& A, const std::vector<double>& x, std::vector<double>& y) {
    return benchmark(
        [&]() {
            for (int b = 0; b < args.batch; ++b) {
                mkl_gemv_one(
                    m,
                    k,
                    A.data() + static_cast<size_t>(b) * m * k,
                    x.data() + static_cast<size_t>(b) * k,
                    y.data() + static_cast<size_t>(b) * m,
                    args.beta);
            }
            return checksum_stride(y, m + 7);
        },
        args.min_iterations,
        args.min_seconds);
}

Timing run_mkl_batched_gemv(int m, int k, const Args& args, const std::vector<double>& A, const std::vector<double>& x, std::vector<double>& y) {
    return benchmark(
        [&]() {
            vbcsr::BLASKernel::gemv_batched(
                m,
                k,
                1.0,
                A.data(),
                m,
                m * k,
                x.data(),
                1,
                k,
                args.beta,
                y.data(),
                1,
                m,
                args.batch);
            return checksum_stride(y, m + 7);
        },
        args.min_iterations,
        args.min_seconds);
}

template <int M, int K>
Timing run_fixed_gemm(
    int rhs,
    int ldb,
    int strideb,
    const Args& args,
    const std::vector<double>& A,
    const std::vector<double>& B,
    std::vector<double>& C) {
    const int ldc = M;
    return benchmark(
        [&]() {
            for (int b = 0; b < args.batch; ++b) {
                fixed_gemm_one<M, K>(
                    rhs,
                    A.data() + static_cast<size_t>(b) * M * K,
                    B.data() + static_cast<size_t>(b) * strideb,
                    ldb,
                    C.data() + static_cast<size_t>(b) * M * rhs,
                    ldc,
                    args.beta);
            }
            return checksum_stride(C, M + 11);
        },
        args.min_iterations,
        args.min_seconds);
}

Timing run_mkl_direct_gemm(
    int m,
    int k,
    int rhs,
    int ldb,
    int strideb,
    const Args& args,
    const std::vector<double>& A,
    const std::vector<double>& B,
    std::vector<double>& C) {
    const int ldc = m;
    return benchmark(
        [&]() {
            for (int b = 0; b < args.batch; ++b) {
                mkl_gemm_one(
                    m,
                    rhs,
                    k,
                    A.data() + static_cast<size_t>(b) * m * k,
                    B.data() + static_cast<size_t>(b) * strideb,
                    ldb,
                    C.data() + static_cast<size_t>(b) * m * rhs,
                    ldc,
                    args.beta);
            }
            return checksum_stride(C, m + 11);
        },
        args.min_iterations,
        args.min_seconds);
}

Timing run_mkl_batched_gemm(
    int m,
    int k,
    int rhs,
    int ldb,
    int strideb,
    const Args& args,
    const std::vector<double>& A,
    const std::vector<double>& B,
    std::vector<double>& C) {
    const int ldc = m;
    return benchmark(
        [&]() {
            vbcsr::BLASKernel::gemm_batched(
                m,
                rhs,
                k,
                1.0,
                A.data(),
                m,
                m * k,
                B.data(),
                ldb,
                strideb,
                args.beta,
                C.data(),
                ldc,
                m * rhs,
                args.batch);
            return checksum_stride(C, m + 11);
        },
        args.min_iterations,
        args.min_seconds);
}

std::vector<double> random_values(size_t count, std::mt19937_64& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> values(count);
    for (double& value : values) {
        value = dist(rng);
    }
    return values;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void print_result(
    const char* op,
    const char* kernel,
    int m,
    int k,
    int rhs,
    int batch,
    double seconds,
    double checksum) {
    const double flops_per_block =
        (std::string(op) == "gemv")
            ? 2.0 * static_cast<double>(m) * static_cast<double>(k)
            : 2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(rhs);
    const double gflops = flops_per_block * static_cast<double>(batch) / seconds / 1.0e9;
    std::cout << op << ','
              << kernel << ','
              << m << ','
              << k << ','
              << rhs << ','
              << batch << ','
              << std::setprecision(12) << seconds << ','
              << gflops << ','
              << checksum << '\n';
}

template <int M, int K>
void run_shape(const Args& args, std::mt19937_64& rng) {
    const std::vector<int> rhs_values = {1, 4, 8, 16, 32};
    auto A = random_values(static_cast<size_t>(args.batch) * M * K, rng);
    auto x = random_values(static_cast<size_t>(args.batch) * K, rng);
    std::vector<double> y(static_cast<size_t>(args.batch) * M, 0.0);

    std::vector<double> fixed_samples;
    std::vector<double> mkl_direct_samples;
    std::vector<double> mkl_batched_samples;
    double fixed_checksum = 0.0;
    double mkl_direct_checksum = 0.0;
    double mkl_batched_checksum = 0.0;
    for (int repeat = 0; repeat < args.repeats; ++repeat) {
        std::fill(y.begin(), y.end(), 0.0);
        auto t_fixed = run_fixed_gemv<M, K>(args, A, x, y);
        fixed_samples.push_back(t_fixed.seconds);
        fixed_checksum = t_fixed.checksum;

        std::fill(y.begin(), y.end(), 0.0);
        auto t_mkl = run_mkl_direct_gemv(M, K, args, A, x, y);
        mkl_direct_samples.push_back(t_mkl.seconds);
        mkl_direct_checksum = t_mkl.checksum;

        std::fill(y.begin(), y.end(), 0.0);
        auto t_batched = run_mkl_batched_gemv(M, K, args, A, x, y);
        mkl_batched_samples.push_back(t_batched.seconds);
        mkl_batched_checksum = t_batched.checksum;
    }
    print_result("gemv", "vbcsr_fixed", M, K, 1, args.batch, median(fixed_samples), fixed_checksum);
    print_result("gemv", "mkl_direct", M, K, 1, args.batch, median(mkl_direct_samples), mkl_direct_checksum);
    print_result("gemv", "mkl_strided_batch_ideal", M, K, 1, args.batch, median(mkl_batched_samples), mkl_batched_checksum);

    for (int rhs : rhs_values) {
        const int ldb = args.multivector_rhs_layout ? args.batch * K : K;
        const int strideb = args.multivector_rhs_layout ? K : K * rhs;
        const size_t b_size = args.multivector_rhs_layout
                                  ? static_cast<size_t>(ldb) * rhs
                                  : static_cast<size_t>(args.batch) * K * rhs;
        auto B = random_values(b_size, rng);
        std::vector<double> C(static_cast<size_t>(args.batch) * M * rhs, 0.0);
        fixed_samples.clear();
        mkl_direct_samples.clear();
        mkl_batched_samples.clear();
        for (int repeat = 0; repeat < args.repeats; ++repeat) {
            std::fill(C.begin(), C.end(), 0.0);
            auto t_fixed = run_fixed_gemm<M, K>(rhs, ldb, strideb, args, A, B, C);
            fixed_samples.push_back(t_fixed.seconds);
            fixed_checksum = t_fixed.checksum;

            std::fill(C.begin(), C.end(), 0.0);
            auto t_mkl = run_mkl_direct_gemm(M, K, rhs, ldb, strideb, args, A, B, C);
            mkl_direct_samples.push_back(t_mkl.seconds);
            mkl_direct_checksum = t_mkl.checksum;

            std::fill(C.begin(), C.end(), 0.0);
            auto t_batched = run_mkl_batched_gemm(M, K, rhs, ldb, strideb, args, A, B, C);
            mkl_batched_samples.push_back(t_batched.seconds);
            mkl_batched_checksum = t_batched.checksum;
        }
        print_result("gemm", "vbcsr_fixed", M, K, rhs, args.batch, median(fixed_samples), fixed_checksum);
        print_result("gemm", "mkl_direct", M, K, rhs, args.batch, median(mkl_direct_samples), mkl_direct_checksum);
        print_result("gemm", "mkl_strided_batch_ideal", M, K, rhs, args.batch, median(mkl_batched_samples), mkl_batched_checksum);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
#ifdef VBCSR_USE_MKL
        int one = 1;
        mkl_set_num_threads_(&one);
#endif
        std::mt19937_64 rng(args.seed);
        std::cout << "op,kernel,m,k,rhs,batch,seconds_per_sweep,gflops,checksum\n";
        run_shape<9, 9>(args, rng);
        run_shape<9, 13>(args, rng);
        run_shape<9, 15>(args, rng);
        run_shape<9, 20>(args, rng);
        run_shape<13, 9>(args, rng);
        run_shape<13, 13>(args, rng);
        run_shape<13, 15>(args, rng);
        run_shape<13, 20>(args, rng);
        run_shape<15, 9>(args, rng);
        run_shape<15, 13>(args, rng);
        run_shape<15, 15>(args, rng);
        run_shape<15, 20>(args, rng);
        run_shape<20, 9>(args, rng);
        run_shape<20, 13>(args, rng);
        run_shape<20, 15>(args, rng);
        run_shape<20, 20>(args, rng);
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 2;
    }
    return 0;
}

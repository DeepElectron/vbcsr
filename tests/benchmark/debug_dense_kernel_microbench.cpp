#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "vbcsr/core/detail/kernels/dense_kernels.hpp"
#include "vbcsr/core/detail/kernels/rowmajor_kernels.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

namespace rmk = vbcsr::detail::rowmajor_kernels;

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

void mkl_gemv_one(int m, int k, const double* A, const double* x, double* y, double beta) {
    vbcsr::BLASKernel::gemv(m, k, 1.0, A, m, x, 1, beta, y, 1);
}

void mkl_gemm_one(int m, int rhs, int k, const double* A, const double* B, int ldb, double* C, int ldc, double beta) {
    vbcsr::BLASKernel::gemm(m, rhs, k, 1.0, A, m, B, ldb, beta, C, ldc);
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

std::vector<std::complex<double>> random_values_z(size_t count, std::mt19937_64& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<std::complex<double>> values(count);
    for (auto& value : values) {
        value = std::complex<double>(dist(rng), dist(rng));
    }
    return values;
}

double checksum_stride_z(const std::vector<std::complex<double>>& values, int stride) {
    double sum = 0.0;
    const int step = std::max(1, stride);
    for (size_t i = 0; i < values.size(); i += static_cast<size_t>(step)) {
        sum += values[i].real();
    }
    return sum;
}

// Transpose every col-major batch block (A(i, k) at src[i + k * M]) into
// canonical row-major (dst[i * K + k]). The BLAS lanes consume the col-major
// buffer; the rm lanes consume this transpose, so both time the same logical
// matrices.
template <typename T>
std::vector<T> transpose_blocks(const std::vector<T>& src, int batch, int M, int K) {
    std::vector<T> dst(src.size());
    for (int b = 0; b < batch; ++b) {
        const T* in = src.data() + static_cast<size_t>(b) * M * K;
        T* out = dst.data() + static_cast<size_t>(b) * M * K;
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                out[static_cast<size_t>(i) * K + k] = in[i + static_cast<size_t>(k) * M];
            }
        }
    }
    return dst;
}

// ---------------------------------------------------------------------------
// Correctness verification of the library row-major kernels
// (vbcsr::detail::rowmajor_kernels) against naive row-major references. A
// silently wrong AVX2 path would otherwise fake the benchmark results.
// A is ROW-major: A(i, k) at a[i * K + k].
// ---------------------------------------------------------------------------
template <typename T>
void ref_rm_gemm(int M, int K, int nv, const T* a, const T* x, int ldx, T* c, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int v = 0; v < nv; ++v) {
            T sum{};
            for (int k = 0; k < K; ++k) {
                sum += a[static_cast<size_t>(i) * K + k] * x[static_cast<size_t>(k) * ldx + v];
            }
            c[static_cast<size_t>(i) * ldc + v] += sum;
        }
    }
}

template <typename T>
void ref_rm_gemm_adjoint(int M, int K, int nv, const T* a, const T* x, int ldx, T* c, int ldc) {
    for (int k = 0; k < K; ++k) {
        for (int v = 0; v < nv; ++v) {
            T sum{};
            for (int i = 0; i < M; ++i) {
                sum += vbcsr::ConjHelper<T>::apply(a[static_cast<size_t>(i) * K + k]) *
                       x[static_cast<size_t>(i) * ldx + v];
            }
            c[static_cast<size_t>(k) * ldc + v] += sum;
        }
    }
}

template <typename T>
double max_abs_diff(const std::vector<T>& lhs, const std::vector<T>& rhs) {
    double max_diff = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(lhs[i] - rhs[i]));
    }
    return max_diff;
}

void verify_rowmajor_kernels(std::mt19937_64& rng) {
    const int shapes[][2] = {{1, 1}, {2, 3}, {9, 13}, {13, 9}, {15, 15}, {20, 9}, {9, 20}, {20, 20}};
    const int nv_values[] = {1, 2, 3, 4, 5, 8, 16, 32};
    double err_gemm_d = 0.0, err_adj_d = 0.0, err_gemm_z = 0.0, err_adj_z = 0.0;
    double err_gemv_d = 0.0, err_gemv_adj_d = 0.0, err_gemv_z = 0.0, err_gemv_adj_z = 0.0;
    for (const auto& shape : shapes) {
        const int M = shape[0], K = shape[1];
        for (int nv : nv_values) {
            // ld > nv exercises the leading-dimension handling; entries in the
            // pad columns [nv, ld) must stay untouched, and the full-buffer
            // comparison below checks that too.
            const int ld = nv + 3;

            // double forward
            auto A = random_values(static_cast<size_t>(M) * K, rng);
            auto X = random_values(static_cast<size_t>(K) * ld, rng);
            auto C = random_values(static_cast<size_t>(M) * ld, rng);
            auto C_ref = C;
            rmk::rm_gemm(M, K, nv, A.data(), X.data(), ld, C.data(), ld);
            ref_rm_gemm(M, K, nv, A.data(), X.data(), ld, C_ref.data(), ld);
            err_gemm_d = std::max(err_gemm_d, max_abs_diff(C, C_ref));

            // double adjoint: X has M rows, C has K rows
            auto Xa = random_values(static_cast<size_t>(M) * ld, rng);
            auto Ca = random_values(static_cast<size_t>(K) * ld, rng);
            auto Ca_ref = Ca;
            rmk::rm_gemm_adjoint(M, K, nv, A.data(), Xa.data(), ld, Ca.data(), ld);
            ref_rm_gemm_adjoint(M, K, nv, A.data(), Xa.data(), ld, Ca_ref.data(), ld);
            err_adj_d = std::max(err_adj_d, max_abs_diff(Ca, Ca_ref));

            // complex forward
            auto Az = random_values_z(static_cast<size_t>(M) * K, rng);
            auto Xz = random_values_z(static_cast<size_t>(K) * ld, rng);
            auto Cz = random_values_z(static_cast<size_t>(M) * ld, rng);
            auto Cz_ref = Cz;
            rmk::rm_gemm(M, K, nv, Az.data(), Xz.data(), ld, Cz.data(), ld);
            ref_rm_gemm(M, K, nv, Az.data(), Xz.data(), ld, Cz_ref.data(), ld);
            err_gemm_z = std::max(err_gemm_z, max_abs_diff(Cz, Cz_ref));

            // complex adjoint
            auto Xza = random_values_z(static_cast<size_t>(M) * ld, rng);
            auto Cza = random_values_z(static_cast<size_t>(K) * ld, rng);
            auto Cza_ref = Cza;
            rmk::rm_gemm_adjoint(M, K, nv, Az.data(), Xza.data(), ld, Cza.data(), ld);
            ref_rm_gemm_adjoint(M, K, nv, Az.data(), Xza.data(), ld, Cza_ref.data(), ld);
            err_adj_z = std::max(err_adj_z, max_abs_diff(Cza, Cza_ref));
        }

        // gemv, double: forward y(M) += A x(K); adjoint y(K) += A^H x(M).
        {
            auto A = random_values(static_cast<size_t>(M) * K, rng);
            auto x = random_values(static_cast<size_t>(K), rng);
            auto y = random_values(static_cast<size_t>(M), rng);
            auto y_ref = y;
            rmk::rm_gemv(M, K, A.data(), x.data(), y.data());
            ref_rm_gemm(M, K, 1, A.data(), x.data(), 1, y_ref.data(), 1);
            err_gemv_d = std::max(err_gemv_d, max_abs_diff(y, y_ref));

            auto xa = random_values(static_cast<size_t>(M), rng);
            auto ya = random_values(static_cast<size_t>(K), rng);
            auto ya_ref = ya;
            rmk::rm_gemv_adjoint(M, K, A.data(), xa.data(), ya.data());
            ref_rm_gemm_adjoint(M, K, 1, A.data(), xa.data(), 1, ya_ref.data(), 1);
            err_gemv_adj_d = std::max(err_gemv_adj_d, max_abs_diff(ya, ya_ref));
        }

        // gemv, complex
        {
            auto A = random_values_z(static_cast<size_t>(M) * K, rng);
            auto x = random_values_z(static_cast<size_t>(K), rng);
            auto y = random_values_z(static_cast<size_t>(M), rng);
            auto y_ref = y;
            rmk::rm_gemv(M, K, A.data(), x.data(), y.data());
            ref_rm_gemm(M, K, 1, A.data(), x.data(), 1, y_ref.data(), 1);
            err_gemv_z = std::max(err_gemv_z, max_abs_diff(y, y_ref));

            auto xa = random_values_z(static_cast<size_t>(M), rng);
            auto ya = random_values_z(static_cast<size_t>(K), rng);
            auto ya_ref = ya;
            rmk::rm_gemv_adjoint(M, K, A.data(), xa.data(), ya.data());
            ref_rm_gemm_adjoint(M, K, 1, A.data(), xa.data(), 1, ya_ref.data(), 1);
            err_gemv_adj_z = std::max(err_gemv_adj_z, max_abs_diff(ya, ya_ref));
        }
    }
    const double tol = 1e-11;
    const double worst = std::max(
        {err_gemm_d, err_adj_d, err_gemm_z, err_adj_z,
         err_gemv_d, err_gemv_adj_d, err_gemv_z, err_gemv_adj_z});
    if (worst > tol) {
        std::cerr << "VERIFICATION FAILED: rm_gemm_d err=" << err_gemm_d
                  << " rm_gemm_adj_d err=" << err_adj_d
                  << " rm_gemm_z err=" << err_gemm_z
                  << " rm_gemm_adj_z err=" << err_adj_z
                  << " rm_gemv_d err=" << err_gemv_d
                  << " rm_gemv_adj_d err=" << err_gemv_adj_d
                  << " rm_gemv_z err=" << err_gemv_z
                  << " rm_gemv_adj_z err=" << err_gemv_adj_z << '\n';
        std::exit(3);
    }
    std::cerr << "# rowmajor library kernel verification OK"
              << " (gemm_d " << err_gemm_d << ", gemm_adj_d " << err_adj_d
              << ", gemm_z " << err_gemm_z << ", gemm_adj_z " << err_adj_z
              << ", gemv_d " << err_gemv_d << ", gemv_adj_d " << err_gemv_adj_d
              << ", gemv_z " << err_gemv_z << ", gemv_adj_z " << err_gemv_adj_z << ")\n";
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
    // Col-major blocks feed the BLAS lanes; the row-major transpose feeds the
    // library rm lanes (same logical matrices).
    auto A = random_values(static_cast<size_t>(args.batch) * M * K, rng);
    auto A_rm = transpose_blocks(A, args.batch, M, K);
    auto x = random_values(static_cast<size_t>(args.batch) * K, rng);
    std::vector<double> y(static_cast<size_t>(args.batch) * M, 0.0);

    std::vector<double> rm_samples;
    std::vector<double> mkl_direct_samples;
    std::vector<double> mkl_batched_samples;
    double rm_checksum = 0.0;
    double mkl_direct_checksum = 0.0;
    double mkl_batched_checksum = 0.0;
    for (int repeat = 0; repeat < args.repeats; ++repeat) {
        // Library row-major SpMV: dot-per-row on row-major A (Phase-4 state).
        std::fill(y.begin(), y.end(), 0.0);
        auto t_rm = benchmark(
            [&]() {
                for (int b = 0; b < args.batch; ++b) {
                    rmk::rm_gemv(
                        M, K,
                        A_rm.data() + static_cast<size_t>(b) * M * K,
                        x.data() + static_cast<size_t>(b) * K,
                        y.data() + static_cast<size_t>(b) * M);
                }
                return checksum_stride(y, M + 7);
            },
            args.min_iterations,
            args.min_seconds);
        rm_samples.push_back(t_rm.seconds);
        rm_checksum = t_rm.checksum;

        std::fill(y.begin(), y.end(), 0.0);
        auto t_mkl = run_mkl_direct_gemv(M, K, args, A, x, y);
        mkl_direct_samples.push_back(t_mkl.seconds);
        mkl_direct_checksum = t_mkl.checksum;

        std::fill(y.begin(), y.end(), 0.0);
        auto t_batched = run_mkl_batched_gemv(M, K, args, A, x, y);
        mkl_batched_samples.push_back(t_batched.seconds);
        mkl_batched_checksum = t_batched.checksum;
    }
    print_result("gemv", "rm_library", M, K, 1, args.batch, median(rm_samples), rm_checksum);
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

        // Library row-major gemm: row-major A blocks, row-major X/C with
        // ld = rhs (compact multivector panel).
        auto X_rm = random_values(static_cast<size_t>(args.batch) * K * rhs, rng);
        std::vector<double> C_rm(static_cast<size_t>(args.batch) * M * rhs, 0.0);

        rm_samples.clear();
        mkl_direct_samples.clear();
        mkl_batched_samples.clear();
        for (int repeat = 0; repeat < args.repeats; ++repeat) {
            std::fill(C_rm.begin(), C_rm.end(), 0.0);
            auto t_rm = benchmark(
                [&]() {
                    for (int b = 0; b < args.batch; ++b) {
                        rmk::rm_gemm(
                            M, K, rhs,
                            A_rm.data() + static_cast<size_t>(b) * M * K,
                            X_rm.data() + static_cast<size_t>(b) * K * rhs,
                            rhs,
                            C_rm.data() + static_cast<size_t>(b) * M * rhs,
                            rhs);
                    }
                    return checksum_stride(C_rm, M + 11);
                },
                args.min_iterations,
                args.min_seconds);
            rm_samples.push_back(t_rm.seconds);
            rm_checksum = t_rm.checksum;

            std::fill(C.begin(), C.end(), 0.0);
            auto t_mkl = run_mkl_direct_gemm(M, K, rhs, ldb, strideb, args, A, B, C);
            mkl_direct_samples.push_back(t_mkl.seconds);
            mkl_direct_checksum = t_mkl.checksum;

            std::fill(C.begin(), C.end(), 0.0);
            auto t_batched = run_mkl_batched_gemm(M, K, rhs, ldb, strideb, args, A, B, C);
            mkl_batched_samples.push_back(t_batched.seconds);
            mkl_batched_checksum = t_batched.checksum;
        }
        print_result("gemm", "rm_library", M, K, rhs, args.batch, median(rm_samples), rm_checksum);
        print_result("gemm", "mkl_direct", M, K, rhs, args.batch, median(mkl_direct_samples), mkl_direct_checksum);
        print_result("gemm", "mkl_strided_batch_ideal", M, K, rhs, args.batch, median(mkl_batched_samples), mkl_batched_checksum);
    }
}

// Complex gemm lanes: the library row-major kernel and the MKL references.
// Complex flops = 8*M*K*rhs per block.
template <int M, int K>
void run_shape_z(const Args& args, std::mt19937_64& rng) {
    using Z = std::complex<double>;
    const Z one(1.0, 0.0);
    const Z beta(args.beta, 0.0);
    const std::vector<int> rhs_values = {8, 16, 32};
    auto A = random_values_z(static_cast<size_t>(args.batch) * M * K, rng);
    auto A_rm = transpose_blocks(A, args.batch, M, K);

    for (int rhs : rhs_values) {
        const int ldb = K;
        auto B = random_values_z(static_cast<size_t>(args.batch) * K * rhs, rng);
        std::vector<Z> C(static_cast<size_t>(args.batch) * M * rhs, Z(0.0, 0.0));
        auto X_rm = random_values_z(static_cast<size_t>(args.batch) * K * rhs, rng);
        std::vector<Z> C_rm(static_cast<size_t>(args.batch) * M * rhs, Z(0.0, 0.0));

        std::vector<double> rm_samples, mkl_samples, mklb_samples;
        double rm_checksum = 0.0, mkl_checksum = 0.0, mklb_checksum = 0.0;
        for (int repeat = 0; repeat < args.repeats; ++repeat) {
            std::fill(C_rm.begin(), C_rm.end(), Z(0.0, 0.0));
            auto t_rm = benchmark(
                [&]() {
                    for (int b = 0; b < args.batch; ++b) {
                        rmk::rm_gemm(
                            M, K, rhs,
                            A_rm.data() + static_cast<size_t>(b) * M * K,
                            X_rm.data() + static_cast<size_t>(b) * K * rhs,
                            rhs,
                            C_rm.data() + static_cast<size_t>(b) * M * rhs,
                            rhs);
                    }
                    return checksum_stride_z(C_rm, M + 11);
                },
                args.min_iterations, args.min_seconds);
            rm_samples.push_back(t_rm.seconds);
            rm_checksum = t_rm.checksum;

            std::fill(C.begin(), C.end(), Z(0.0, 0.0));
            auto t_mkl = benchmark(
                [&]() {
                    for (int b = 0; b < args.batch; ++b) {
                        vbcsr::BLASKernel::gemm(
                            M, rhs, K, one,
                            A.data() + static_cast<size_t>(b) * M * K, M,
                            B.data() + static_cast<size_t>(b) * K * rhs, ldb,
                            beta,
                            C.data() + static_cast<size_t>(b) * M * rhs, M);
                    }
                    return checksum_stride_z(C, M + 11);
                },
                args.min_iterations, args.min_seconds);
            mkl_samples.push_back(t_mkl.seconds);
            mkl_checksum = t_mkl.checksum;

            std::fill(C.begin(), C.end(), Z(0.0, 0.0));
            auto t_mklb = benchmark(
                [&]() {
                    vbcsr::BLASKernel::gemm_batched(
                        M, rhs, K, one,
                        A.data(), M, M * K,
                        B.data(), ldb, K * rhs,
                        beta,
                        C.data(), M, M * rhs,
                        args.batch);
                    return checksum_stride_z(C, M + 11);
                },
                args.min_iterations, args.min_seconds);
            mklb_samples.push_back(t_mklb.seconds);
            mklb_checksum = t_mklb.checksum;
        }
        auto print_z = [&](const char* kernel, double seconds, double checksum) {
            const double flops =
                8.0 * static_cast<double>(M) * K * rhs * static_cast<double>(args.batch);
            std::cout << "gemm_z," << kernel << ',' << M << ',' << K << ',' << rhs << ','
                      << args.batch << ',' << std::setprecision(12) << seconds << ','
                      << flops / seconds / 1.0e9 << ',' << checksum << '\n';
        };
        print_z("rm_library", median(rm_samples), rm_checksum);
        print_z("mkl_direct", median(mkl_samples), mkl_checksum);
        print_z("mkl_strided_batch_ideal", median(mklb_samples), mklb_checksum);
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
        verify_rowmajor_kernels(rng);
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
        run_shape_z<9, 9>(args, rng);
        run_shape_z<9, 13>(args, rng);
        run_shape_z<13, 13>(args, rng);
        run_shape_z<13, 20>(args, rng);
        run_shape_z<15, 15>(args, rng);
        run_shape_z<20, 9>(args, rng);
        run_shape_z<20, 20>(args, rng);
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 2;
    }
    return 0;
}

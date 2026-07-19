#ifndef VBCSR_DETAIL_KERNELS_DENSE_KERNELS_HPP
#define VBCSR_DETAIL_KERNELS_DENSE_KERNELS_HPP

#include "blas_api.hpp"

#include <algorithm>
#include <complex>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vbcsr {

// Helper for conjugation
template <typename T>
struct ConjHelper {
    static __attribute__((always_inline)) T apply(T val) { return val; }
};

template <typename T>
struct ConjHelper<std::complex<T>> {
    static __attribute__((always_inline)) std::complex<T> apply(std::complex<T> val) { return std::conj(val); }
};

// The column-major block kernel family (NaiveKernel, TinyBlockKernel,
// FixedBlockKernel, SmartKernel switch tables) was removed in Phase 4 of the
// row-major migration: block storage is now canonical row-major and every
// apply/SpGEMM path uses detail/kernels/rowmajor_kernels.hpp.

// MKL/BLAS Kernel
struct BLASKernel {
    static constexpr bool supports_strided_gemm() {
#ifdef VBCSR_BLAS_HAS_BATCH_GEMM
        return true;
#else
        return false;
#endif
    }

    static constexpr bool supports_strided_gemv() {
#ifdef VBCSR_BLAS_HAS_BATCH_GEMV
        return true;
#else
        return false;
#endif
    }

    // Double
    static void gemv(int m, int n, double alpha, const double* A, int lda, const double* x, int incx, double beta, double* y, int incy, CBLAS_TRANSPOSE trans = CblasNoTrans) {
        cblas_dgemv(CblasColMajor, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
    }

    static void gemm(int m, int n, int k, double alpha, const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc, CBLAS_TRANSPOSE transA = CblasNoTrans, CBLAS_TRANSPOSE transB = CblasNoTrans) {
        cblas_dgemm(CblasColMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    // Complex Double
    static void gemv(int m, int n, std::complex<double> alpha, const std::complex<double>* A, int lda, const std::complex<double>* x, int incx, std::complex<double> beta, std::complex<double>* y, int incy, CBLAS_TRANSPOSE trans = CblasNoTrans) {
        cblas_zgemv(CblasColMajor, trans, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    }

    static void gemm(int m, int n, int k, std::complex<double> alpha, const std::complex<double>* A, int lda, const std::complex<double>* B, int ldb, std::complex<double> beta, std::complex<double>* C, int ldc, CBLAS_TRANSPOSE transA = CblasNoTrans, CBLAS_TRANSPOSE transB = CblasNoTrans) {
        cblas_zgemm(CblasColMajor, transA, transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    }

    static void gemv_batched(
        int m,
        int n,
        double alpha,
        const double* A,
        int lda,
        int stridea,
        const double* x,
        int incx,
        int stridex,
        double beta,
        double* y,
        int incy,
        int stridey,
        int batch_count,
        CBLAS_TRANSPOSE trans = CblasNoTrans) {
#ifdef VBCSR_BLAS_HAS_BATCH_GEMV
        cblas_dgemv_batch_strided(
            CblasColMajor,
            trans,
            static_cast<vbcsr_blas_int>(m),
            static_cast<vbcsr_blas_int>(n),
            alpha,
            A,
            static_cast<vbcsr_blas_int>(lda),
            static_cast<vbcsr_blas_int>(stridea),
            x,
            static_cast<vbcsr_blas_int>(incx),
            static_cast<vbcsr_blas_int>(stridex),
            beta,
            y,
            static_cast<vbcsr_blas_int>(incy),
            static_cast<vbcsr_blas_int>(stridey),
            static_cast<vbcsr_blas_int>(batch_count));
#else
        for (int batch = 0; batch < batch_count; ++batch) {
            gemv(
                m,
                n,
                alpha,
                A + static_cast<size_t>(batch) * stridea,
                lda,
                x + static_cast<size_t>(batch) * stridex,
                incx,
                beta,
                y + static_cast<size_t>(batch) * stridey,
                incy,
                trans);
        }
#endif
    }

    static void gemm_batched(
        int m,
        int n,
        int k,
        double alpha,
        const double* A,
        int lda,
        int stridea,
        const double* B,
        int ldb,
        int strideb,
        double beta,
        double* C,
        int ldc,
        int stridec,
        int batch_count,
        CBLAS_TRANSPOSE transA = CblasNoTrans,
        CBLAS_TRANSPOSE transB = CblasNoTrans) {
#ifdef VBCSR_BLAS_HAS_BATCH_GEMM
        cblas_dgemm_batch_strided(
            CblasColMajor,
            transA,
            transB,
            static_cast<vbcsr_blas_int>(m),
            static_cast<vbcsr_blas_int>(n),
            static_cast<vbcsr_blas_int>(k),
            alpha,
            A,
            static_cast<vbcsr_blas_int>(lda),
            static_cast<vbcsr_blas_int>(stridea),
            B,
            static_cast<vbcsr_blas_int>(ldb),
            static_cast<vbcsr_blas_int>(strideb),
            beta,
            C,
            static_cast<vbcsr_blas_int>(ldc),
            static_cast<vbcsr_blas_int>(stridec),
            static_cast<vbcsr_blas_int>(batch_count));
#else
        for (int batch = 0; batch < batch_count; ++batch) {
            gemm(
                m,
                n,
                k,
                alpha,
                A + static_cast<size_t>(batch) * stridea,
                lda,
                B + static_cast<size_t>(batch) * strideb,
                ldb,
                beta,
                C + static_cast<size_t>(batch) * stridec,
                ldc,
                transA,
                transB);
        }
#endif
    }

    static void gemv_batched(
        int m,
        int n,
        std::complex<double> alpha,
        const std::complex<double>* A,
        int lda,
        int stridea,
        const std::complex<double>* x,
        int incx,
        int stridex,
        std::complex<double> beta,
        std::complex<double>* y,
        int incy,
        int stridey,
        int batch_count,
        CBLAS_TRANSPOSE trans = CblasNoTrans) {
#ifdef VBCSR_BLAS_HAS_BATCH_GEMV
        cblas_zgemv_batch_strided(
            CblasColMajor,
            trans,
            static_cast<vbcsr_blas_int>(m),
            static_cast<vbcsr_blas_int>(n),
            &alpha,
            A,
            static_cast<vbcsr_blas_int>(lda),
            static_cast<vbcsr_blas_int>(stridea),
            x,
            static_cast<vbcsr_blas_int>(incx),
            static_cast<vbcsr_blas_int>(stridex),
            &beta,
            y,
            static_cast<vbcsr_blas_int>(incy),
            static_cast<vbcsr_blas_int>(stridey),
            static_cast<vbcsr_blas_int>(batch_count));
#else
        for (int batch = 0; batch < batch_count; ++batch) {
            gemv(
                m,
                n,
                alpha,
                A + static_cast<size_t>(batch) * stridea,
                lda,
                x + static_cast<size_t>(batch) * stridex,
                incx,
                beta,
                y + static_cast<size_t>(batch) * stridey,
                incy,
                trans);
        }
#endif
    }

    static void gemm_batched(
        int m,
        int n,
        int k,
        std::complex<double> alpha,
        const std::complex<double>* A,
        int lda,
        int stridea,
        const std::complex<double>* B,
        int ldb,
        int strideb,
        std::complex<double> beta,
        std::complex<double>* C,
        int ldc,
        int stridec,
        int batch_count,
        CBLAS_TRANSPOSE transA = CblasNoTrans,
        CBLAS_TRANSPOSE transB = CblasNoTrans) {
#ifdef VBCSR_BLAS_HAS_BATCH_GEMM
        cblas_zgemm_batch_strided(
            CblasColMajor,
            transA,
            transB,
            static_cast<vbcsr_blas_int>(m),
            static_cast<vbcsr_blas_int>(n),
            static_cast<vbcsr_blas_int>(k),
            &alpha,
            A,
            static_cast<vbcsr_blas_int>(lda),
            static_cast<vbcsr_blas_int>(stridea),
            B,
            static_cast<vbcsr_blas_int>(ldb),
            static_cast<vbcsr_blas_int>(strideb),
            &beta,
            C,
            static_cast<vbcsr_blas_int>(ldc),
            static_cast<vbcsr_blas_int>(stridec),
            static_cast<vbcsr_blas_int>(batch_count));
#else
        for (int batch = 0; batch < batch_count; ++batch) {
            gemm(
                m,
                n,
                k,
                alpha,
                A + static_cast<size_t>(batch) * stridea,
                lda,
                B + static_cast<size_t>(batch) * strideb,
                ldb,
                beta,
                C + static_cast<size_t>(batch) * stridec,
                ldc,
                transA,
                transB);
        }
#endif
    }

    static int preferred_parallel_thread_count() {
#ifdef _OPENMP
        return std::max(1, omp_get_max_threads());
#else
        return 1;
#endif
    }

    // Native sparse kernels may call BLAS inside an outer OpenMP region. Clamp the
    // inner BLAS runtime to one thread to avoid oversubscription.
    static void configure_native_threading() {
#ifdef VBCSR_USE_MKL
        int one = 1;
        if (mkl_get_max_threads() != one) {
            mkl_set_num_threads_(&one);
        }
#elif defined(VBCSR_USE_OPENBLAS)
        openblas_set_num_threads(1);
#else
        // Generic BLAS: Do nothing. 
        // We do NOT want to call omp_set_num_threads(1) here because it would disable
        // parallelism for the outer loops (Sparse MVP/MM).
#endif
    }

    // Vendor sparse kernels should own parallelism themselves. For MKL sparse we
    // align the MKL thread pool with the configured OpenMP thread budget so callers
    // only need to manage one thread setting.
    static void configure_vendor_sparse_threading() {
#ifdef VBCSR_USE_MKL
        int threads = preferred_parallel_thread_count();
        if (mkl_get_max_threads() != threads) {
            mkl_set_num_threads_(&threads);
        }
#endif
    }

    static std::string name() {
#ifdef VBCSR_USE_MKL
        return "Intel MKL";
#elif defined(VBCSR_USE_OPENBLAS)
        return "OpenBLAS";
#else
        return "Generic BLAS";
#endif
    }
};


} // namespace vbcsr

#endif

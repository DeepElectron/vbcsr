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

// Zero a buffer with all OpenMP threads: parallel fill bandwidth and
// NUMA-local first touch (a serial fill places every page on the calling
// thread's node). Launches its own parallel region.
template <typename T>
inline void parallel_zero(T* data, size_t count) {
#ifdef _OPENMP
    #pragma omp parallel
    {
        const size_t thread_count = static_cast<size_t>(omp_get_num_threads());
        const size_t thread_id = static_cast<size_t>(omp_get_thread_num());
        const size_t begin = count * thread_id / thread_count;
        const size_t end = count * (thread_id + 1) / thread_count;
        std::fill(data + begin, data + end, T(0));
    }
#else
    std::fill(data, data + count, T(0));
#endif
}

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

    // ---------------------------------------------------------- thread policy
    //
    // One knob for the user: OMP_NUM_THREADS. vbcsr translates that budget to
    // whichever vendor library is linked; nobody sets MKL_NUM_THREADS,
    // OPENBLAS_NUM_THREADS or a threading layer by hand.
    //
    // Every threaded operation in the library is one of three kinds, and each
    // kind needs exactly one thing:
    //
    //  A. Hand-written kernels (CSR/BSR/VBCSR apply, row-major block kernels,
    //     symbolic SpGEMM). They parallelise their own loops with the linked
    //     OpenMP and call no vendor BLAS inside. They need NOTHING - no call,
    //     no guard. Do not add thread configuration to these paths.
    //
    //  B. Vendor-threaded kernels invoked from serial code (MKL/AOCL sparse
    //     handles, standalone dense LAPACK). The vendor pool must hold the
    //     OpenMP budget: call align_vendor_threads() at the entry point. It is
    //     idempotent and costs a comparison in the common case.
    //
    //  C. Dense BLAS/LAPACK called from INSIDE one of our own OpenMP parallel
    //     regions (the SpGEMM numeric phase, the batched dense subgraph
    //     problems of the graph matrix function - currently the only two).
    //     Each such call must run on one thread or the pools multiply.
    //     Instantiate ScopedSerialBLAS before the parallel region; it pins the
    //     vendor pool to one thread and restores the budget when it leaves
    //     scope.
    //
    // The invariant these maintain: the vendor pool always rests at the OpenMP
    // budget; only a live ScopedSerialBLAS ever has it elsewhere. The process
    // therefore ends every vbcsr operation in the same thread state it began.

    // The user's thread budget (OMP_NUM_THREADS / omp_set_num_threads).
    static int preferred_parallel_thread_count() {
#ifdef _OPENMP
        return std::max(1, omp_get_max_threads());
#else
        return 1;
#endif
    }

    // Current size of the linked vendor pool (the OpenMP budget when the
    // vendor exposes no such knob).
    static int vendor_thread_count() {
#ifdef VBCSR_USE_MKL
        return mkl_get_max_threads();
#elif defined(VBCSR_USE_OPENBLAS)
        return openblas_get_num_threads();
#else
        return preferred_parallel_thread_count();
#endif
    }

    static void set_vendor_thread_count(int threads) {
        if (threads < 1) threads = 1;
#ifdef VBCSR_USE_MKL
        if (mkl_get_max_threads() != threads) {
            mkl_set_num_threads(threads);
            mkl_set_num_threads_local(threads);
        }
#elif defined(VBCSR_USE_OPENBLAS)
        if (openblas_get_num_threads() != threads) {
            openblas_set_num_threads(threads);
        }
#else
        (void)threads;  // generic BLAS: OpenMP's own budget is the only lever
#endif
    }

    // Kind B: vendor pool := OpenMP budget.
    static void align_vendor_threads() {
        set_vendor_thread_count(preferred_parallel_thread_count());
    }

    // Kind C: single-threaded vendor BLAS for the lifetime of the object.
    //
    // Also pins OpenMP's max-active-levels to 1 (once, process-wide): a BLAS
    // that is threaded through our own OpenMP runtime has no vendor pool to
    // pin, and the nesting rule is what makes its inner parallel regions run
    // with a team of one.
    //
    // Construct from serial code, before the parallel region.
    class ScopedSerialBLAS {
    public:
        ScopedSerialBLAS() : previous_(vendor_thread_count()) {
#ifdef _OPENMP
            if (!omp_in_parallel()) {
                static const bool once = [] {
                    omp_set_max_active_levels(1);
                    return true;
                }();
                (void)once;
            }
#endif
            set_vendor_thread_count(1);
        }
        ~ScopedSerialBLAS() { set_vendor_thread_count(previous_); }

        ScopedSerialBLAS(const ScopedSerialBLAS&) = delete;
        ScopedSerialBLAS& operator=(const ScopedSerialBLAS&) = delete;

    private:
        int previous_;
    };

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

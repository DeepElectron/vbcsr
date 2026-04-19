#ifndef VBCSR_DETAIL_KERNELS_DENSE_KERNELS_HPP
#define VBCSR_DETAIL_KERNELS_DENSE_KERNELS_HPP

#include "blas_api.hpp"

#include "../../scalar_traits.hpp"

#include <complex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

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

// Naive Kernel (Fallback)
template <typename T>
struct NaiveKernel {
    static void gemv(int m, int n, T alpha, const T* A, int lda, const T* x, int incx, T beta, T* y, int incy, CBLAS_TRANSPOSE trans = CblasNoTrans) {
        if (trans == CblasNoTrans) {
            for (int i = 0; i < m; ++i) {
                T sum = 0;
                for (int j = 0; j < n; ++j) {
                    sum += A[i + j*lda] * x[j*incx];
                }
                y[i*incy] = alpha * sum + beta * y[i*incy];
            }
        } else if (trans == CblasTrans) {
            for (int j = 0; j < n; ++j) {
                T sum = 0;
                for (int i = 0; i < m; ++i) {
                    sum += A[i + j*lda] * x[i*incx];
                }
                y[j*incy] = alpha * sum + beta * y[j*incy];
            }
        } else if (trans == CblasConjTrans) {
            for (int j = 0; j < n; ++j) {
                T sum = 0;
                for (int i = 0; i < m; ++i) {
                    sum += ScalarTraits<T>::conjugate(A[i + j*lda]) * x[i*incx];
                }
                y[j*incy] = alpha * sum + beta * y[j*incy];
            }
        }
    }

    static void gemm(int m, int n, int k, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C, int ldc, CBLAS_TRANSPOSE transA = CblasNoTrans, CBLAS_TRANSPOSE transB = CblasNoTrans) {
        // Only NoTrans/NoTrans implemented for naive fallback for now as it's the most common
        if (transA == CblasNoTrans && transB == CblasNoTrans) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < m; ++i) {
                    T sum = 0;
                    for (int l = 0; l < k; ++l) {
                        sum += A[i + l*lda] * B[l + j*ldb];
                    }
                    C[i + j*ldc] = alpha * sum + beta * C[i + j*ldc];
                }
            }
        } else if ((transA == CblasTrans || transA == CblasConjTrans) && transB == CblasNoTrans) {
            // C = alpha * A^T * B + beta * C
            // A is M_orig x K_orig (stored as K x M if we consider A as the matrix before op)
            // Wait, standard GEMM: C = alpha * op(A) * op(B) + beta * C
            // Arguments m, n, k refer to dimensions of op(A) and op(B).
            // op(A) is m x k. op(B) is k x n. C is m x n.
            
            // If transA != NoTrans:
            // A is k x m (stored). op(A) is m x k.
            // A[l + i*lda] is element (l, i) of stored A, which is (i, l) of op(A).
            
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < m; ++i) {
                    T sum = 0;
                    for (int l = 0; l < k; ++l) {
                        T a_val = A[l + i*lda]; // A is k x m. Row l, Col i.
                        if (transA == CblasConjTrans) a_val = ScalarTraits<T>::conjugate(a_val);
                        sum += a_val * B[l + j*ldb];
                    }
                    C[i + j*ldc] = alpha * sum + beta * C[i + j*ldc];
                }
            }
        } else {
            throw std::runtime_error("NaiveKernel::gemm: Unsupported transpose combination.");
        }
    }
};

// Template Kernel for Fixed Block Sizes
// M, N are compile-time constants
// Tiny Block Kernel (M=1..16) using AVX2 Intrinsics
// Requires __AVX2__ and __FMA__
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

template <typename T, int M>
struct TinyBlockKernel {
    // Fallback for non-double types or if AVX2 not available
    static void gemv(const T* A, const T* x, T* y, T alpha, T beta, int N) {
        // Fallback to compiler vectorization
        #pragma omp simd
        for (int i = 0; i < M; ++i) y[i] *= beta;
        for (int j = 0; j < N; ++j) {
            T x_val = alpha * x[j];
            #pragma omp simd
            for (int i = 0; i < M; ++i) y[i] += A[i + j*M] * x_val;
        }
    }
    
    static void gemm(int n, const T* A, int lda, const T* B, int ldb, T* C, int ldc, T alpha, T beta, int K) {
         // Fallback
        for (int j = 0; j < n; ++j) {
            T* C_col = &C[j*ldc];
            #pragma omp simd
            for (int i = 0; i < M; ++i) C_col[i] *= beta;
            for (int l = 0; l < K; ++l) {
                T b_val = alpha * B[l + j*ldb];
                const T* A_col = &A[l*lda];
                #pragma omp simd
                for (int i = 0; i < M; ++i) C_col[i] += A_col[i] * b_val;
            }
        }
    }

    static void gemv_trans(const T* A, const T* x, T* y, T alpha, T beta, int N) {
        // Fallback generic transpose
        for (int j = 0; j < N; ++j) {
            T dot = 0;
            const T* A_col = &A[j*M];
            #pragma omp simd reduction(+:dot)
            for (int i = 0; i < M; ++i) {
                dot += ConjHelper<T>::apply(A_col[i]) * x[i];
            }
            y[j] = alpha * dot + beta * y[j];
        }
    }

    static void gemm_trans(int n, const T* A, int lda, const T* B, int ldb, T* C, int ldc, T alpha, T beta, int K) {
        // Fallback generic transpose
        for (int j = 0; j < n; ++j) {
            gemv_trans(A, &B[j*ldb], &C[j*ldc], alpha, beta, K);
        }
    }
};

// Specialization for double
template <int M>
struct TinyBlockKernel<double, M> {
    static void gemv(const double* __restrict__ A, const double* __restrict__ x, double* __restrict__ y, double alpha, double beta, int N) {
        // AVX2 registers: ymm0-ymm15 (256-bit, 4 doubles)
        __m256d y0, y1, y2, y3;
        __m256d vbeta = _mm256_set1_pd(beta);
        
        // Safe Read Y
        auto read_y = [&](int offset, __m256d& reg) {
            if (M >= offset + 4) reg = _mm256_loadu_pd(y + offset);
            else {
                double tmp[4] = {0};
                for(int k=0; k<M-offset; ++k) tmp[k] = y[offset+k];
                reg = _mm256_loadu_pd(tmp);
            }
        };
        
        read_y(0, y0);
        if (M > 4) read_y(4, y1);
        if (M > 8) read_y(8, y2);
        if (M > 12) read_y(12, y3);
        
        // Scale
        y0 = _mm256_mul_pd(y0, vbeta);
        if (M > 4) y1 = _mm256_mul_pd(y1, vbeta);
        if (M > 8) y2 = _mm256_mul_pd(y2, vbeta);
        if (M > 12) y3 = _mm256_mul_pd(y3, vbeta);
        
        // Accumulate
        for (int j = 0; j < N; ++j) {
            __m256d vx = _mm256_set1_pd(alpha * x[j]);
            const double* A_col = A + j*M;
            
            auto fma = [&](int offset, __m256d& y_reg) {
                __m256d va;
                if (M >= offset + 4) va = _mm256_loadu_pd(A_col + offset);
                else {
                    double tmp[4] = {0};
                    for(int k=0; k<M-offset; ++k) tmp[k] = A_col[offset+k];
                    va = _mm256_loadu_pd(tmp);
                }
                y_reg = _mm256_fmadd_pd(va, vx, y_reg);
            };
            
            fma(0, y0);
            if (M > 4) fma(4, y1);
            if (M > 8) fma(8, y2);
            if (M > 12) fma(12, y3);
        }
        
        // Store Y back
        auto store_y = [&](int offset, __m256d& reg) {
            if (M >= offset + 4) _mm256_storeu_pd(y + offset, reg);
            else {
                double tmp[4];
                _mm256_storeu_pd(tmp, reg);
                for(int k=0; k<M-offset; ++k) y[offset+k] = tmp[k];
            }
        };
        
        store_y(0, y0);
        if (M > 4) store_y(4, y1);
        if (M > 8) store_y(8, y2);
        if (M > 12) store_y(12, y3);
    }

    static void gemm(int n, const double* __restrict__ A, int lda, const double* __restrict__ B, int ldb, double* __restrict__ C, int ldc, double alpha, double beta, int K) {
        for (int j = 0; j < n; ++j) {
            gemv(A, &B[j*ldb], &C[j*ldc], alpha, beta, K);
        }
    }

    static void gemv_trans(const double* __restrict__ A, const double* __restrict__ x, double* __restrict__ y, double alpha, double beta, int N) {
        // y = alpha * A^T * x + beta * y
        // A: M x N (ColMajor). A^T: N x M.
        // y is length N. x is length M.
        // y[j] += alpha * dot(A_col_j, x)
        
        // Load x into registers if M is small enough?
        // M=1..16. We can load x into ymm registers.
        
        __m256d x0, x1, x2, x3;
        
        // Safe Read X
        auto read_x = [&](int offset, __m256d& reg) {
            if (M >= offset + 4) reg = _mm256_loadu_pd(x + offset);
            else {
                double tmp[4] = {0};
                for(int k=0; k<M-offset; ++k) tmp[k] = x[offset+k];
                reg = _mm256_loadu_pd(tmp);
            }
        };
        
        read_x(0, x0);
        if (M > 4) read_x(4, x1);
        if (M > 8) read_x(8, x2);
        if (M > 12) read_x(12, x3);
        
        for (int j = 0; j < N; ++j) {
            const double* A_col = A + j*M;
            __m256d sum = _mm256_setzero_pd();
            
            auto dot = [&](int offset, __m256d& x_reg) {
                __m256d va;
                if (M >= offset + 4) va = _mm256_loadu_pd(A_col + offset);
                else {
                    double tmp[4] = {0};
                    for(int k=0; k<M-offset; ++k) tmp[k] = A_col[offset+k];
                    va = _mm256_loadu_pd(tmp);
                }
                sum = _mm256_fmadd_pd(va, x_reg, sum);
            };
            
            dot(0, x0);
            if (M > 4) dot(4, x1);
            if (M > 8) dot(8, x2);
            if (M > 12) dot(12, x3);
            
            // Horizontal sum
            double s = 0.0;
            // _mm256_storeu_pd is slow for just reduction.
            // Use hadd?
            // Or just store to stack.
            double tmp[4];
            _mm256_storeu_pd(tmp, sum);
            s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            
            y[j] = alpha * s + beta * y[j];
        }
    }

    static void gemm_trans(int n, const double* __restrict__ A, int lda, const double* __restrict__ B, int ldb, double* __restrict__ C, int ldc, double alpha, double beta, int K) {
        // C = alpha * A^T * B + beta * C
        // A: M x K (lda), B: M x n (ldb), C: K x n (ldc)
        // A^T is K x M.
        
        for (int j = 0; j < n; ++j) {
            gemv_trans(A, &B[j*ldb], &C[j*ldc], alpha, beta, K);
        }
    }
};
#endif

// Template Kernel for Fixed Block Sizes
// M, N are compile-time constants
template <typename T, int M, int N>
struct FixedBlockKernel {
    static void gemv(const T* __restrict__ A, const T* __restrict__ x, T* __restrict__ y, T alpha, T beta) {
#if defined(__AVX2__) && defined(__FMA__)
        if constexpr (M <= 16) {
            TinyBlockKernel<T, M>::gemv(A, x, y, alpha, beta, N);
            return;
        }
#endif // VBCSR_DETAIL_KERNELS_DENSE_KERNELS_HPP
        // y = alpha * A * x + beta * y
        // A is ColMajor: A[i + j*M]
        
        // Scale y
        #pragma omp simd
        for (int i = 0; i < M; ++i) {
            y[i] *= beta;
        }
        
        // Accumulate
        for (int j = 0; j < N; ++j) {
            T x_val = alpha * x[j];
            #pragma omp simd
            for (int i = 0; i < M; ++i) {
                y[i] += A[i + j*M] * x_val;
            }
        }
    }

    static void gemm(int n, const T* __restrict__ A, int lda, const T* __restrict__ B, int ldb, T* __restrict__ C, int ldc, T alpha, T beta) {
#if defined(__AVX2__) && defined(__FMA__)
        if constexpr (M <= 16) {
            TinyBlockKernel<T, M>::gemm(n, A, lda, B, ldb, C, ldc, alpha, beta, N);
            return;
        }
#endif
        // C = alpha * A * B + beta * C
        // A: M x N (template N is K here) (lda), B: N x n (runtime n) (ldb), C: M x n (ldc)
        // Fixed M, K (template N). Runtime n (num_vecs).
        
        for (int j = 0; j < n; ++j) {
            T* C_col = &C[j*ldc];
            
            #pragma omp simd
            for (int i = 0; i < M; ++i) {
                C_col[i] *= beta;
            }
            
            for (int l = 0; l < N; ++l) { // N here is K (block col dim)
                T b_val = alpha * B[l + j*ldb];
                const T* A_col = &A[l*lda];
                
                #pragma omp simd
                for (int i = 0; i < M; ++i) {
                    C_col[i] += A_col[i] * b_val;
                }
            }
        }
    }

    static void gemv_trans(const T* __restrict__ A, const T* __restrict__ x, T* __restrict__ y, T alpha, T beta) {
#if defined(__AVX2__) && defined(__FMA__)
        if constexpr (M <= 16) {
            TinyBlockKernel<T, M>::gemv_trans(A, x, y, alpha, beta, N);
            return;
        }
#endif
        // y = alpha * A^T * x + beta * y
        // A: M x N. A^T: N x M.
        // y: N. x: M.
        
        for (int j = 0; j < N; ++j) {
            T dot = 0;
            const T* A_col = &A[j*M];
            #pragma omp simd reduction(+:dot)
            for (int i = 0; i < M; ++i) {
                dot += ConjHelper<T>::apply(A_col[i]) * x[i];
            }
            y[j] = alpha * dot + beta * y[j];
        }
    }

    static void gemm_trans(int n, const T* __restrict__ A, int lda, const T* __restrict__ B, int ldb, T* __restrict__ C, int ldc, T alpha, T beta) {
#if defined(__AVX2__) && defined(__FMA__)
        if constexpr (M <= 16) {
            TinyBlockKernel<T, M>::gemm_trans(n, A, lda, B, ldb, C, ldc, alpha, beta, N);
            return;
        }
#endif
        // C = alpha * A^T * B + beta * C
        // A: M x N (K). A^T: N x M.
        // B: M x n. C: N x n.
        
        for (int j = 0; j < n; ++j) {
            gemv_trans(A, &B[j*ldb], &C[j*ldc], alpha, beta);
        }
    }
};





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
        mkl_set_num_threads_(&one);
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
        mkl_set_num_threads_(&threads);
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

// Smart Kernel Dispatcher
// Smart Kernel Dispatcher
template <typename T>
struct SmartKernel {
    static constexpr bool supports_batched_gemm() {
        return (std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>) &&
               BLASKernel::supports_strided_gemm();
    }

    static constexpr bool supports_batched_gemv() {
        return (std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>) &&
               BLASKernel::supports_strided_gemv();
    }

    // Dispatch macros
    #define CASE_GEMV(R, C) case C: FixedBlockKernel<T, R, C>::gemv(A, x, y, alpha, beta); break;
    #define SWITCH_GEMV(R) \
        switch(n) { \
            CASE_GEMV(R, 1) CASE_GEMV(R, 2) CASE_GEMV(R, 3) CASE_GEMV(R, 4) CASE_GEMV(R, 5) \
            CASE_GEMV(R, 6) CASE_GEMV(R, 7) CASE_GEMV(R, 8) CASE_GEMV(R, 9) CASE_GEMV(R, 10) \
            CASE_GEMV(R, 11) CASE_GEMV(R, 12) CASE_GEMV(R, 13) CASE_GEMV(R, 14) CASE_GEMV(R, 15) \
            CASE_GEMV(R, 16) \
            default: BLASKernel::gemv(m, n, alpha, A, m, x, 1, beta, y, 1); break; \
        }

    #define CASE_ROW_GEMV(R) case R: SWITCH_GEMV(R); break;

    static void gemv(int m, int n, T alpha, const T* A, int lda, const T* x, int incx, T beta, T* y, int incy) {
        // Only optimized for unit stride and packed A (lda=m)
        if (incx == 1 && incy == 1 && lda == m) {
            switch(m) {
                CASE_ROW_GEMV(1) CASE_ROW_GEMV(2) CASE_ROW_GEMV(3) CASE_ROW_GEMV(4) CASE_ROW_GEMV(5)
                CASE_ROW_GEMV(6) CASE_ROW_GEMV(7) CASE_ROW_GEMV(8) CASE_ROW_GEMV(9) CASE_ROW_GEMV(10)
                CASE_ROW_GEMV(11) CASE_ROW_GEMV(12) CASE_ROW_GEMV(13) CASE_ROW_GEMV(14) CASE_ROW_GEMV(15)
                CASE_ROW_GEMV(16)
                default: BLASKernel::gemv(m, n, alpha, A, lda, x, incx, beta, y, incy); break;
            }
        } else {
            BLASKernel::gemv(m, n, alpha, A, lda, x, incx, beta, y, incy);
        }
    }

    #define CASE_GEMM(R, C) case C: FixedBlockKernel<T, R, C>::gemm(n, A, lda, B, ldb, C_ptr, ldc, alpha, beta); break;
    #define SWITCH_GEMM(R) \
        switch(k) { \
            CASE_GEMM(R, 1) CASE_GEMM(R, 2) CASE_GEMM(R, 3) CASE_GEMM(R, 4) CASE_GEMM(R, 5) \
            CASE_GEMM(R, 6) CASE_GEMM(R, 7) CASE_GEMM(R, 8) CASE_GEMM(R, 9) CASE_GEMM(R, 10) \
            CASE_GEMM(R, 11) CASE_GEMM(R, 12) CASE_GEMM(R, 13) CASE_GEMM(R, 14) CASE_GEMM(R, 15) \
            CASE_GEMM(R, 16) CASE_GEMM(R, 17) CASE_GEMM(R, 18) CASE_GEMM(R, 19) CASE_GEMM(R, 20) \
            CASE_GEMM(R, 21) CASE_GEMM(R, 22) CASE_GEMM(R, 23) CASE_GEMM(R, 24) CASE_GEMM(R, 25) \
            CASE_GEMM(R, 26) CASE_GEMM(R, 27) CASE_GEMM(R, 28) CASE_GEMM(R, 29) CASE_GEMM(R, 30) \
            CASE_GEMM(R, 31) CASE_GEMM(R, 32) CASE_GEMM(R, 33) CASE_GEMM(R, 34) CASE_GEMM(R, 35) \
            CASE_GEMM(R, 36) CASE_GEMM(R, 37) CASE_GEMM(R, 38) CASE_GEMM(R, 39) CASE_GEMM(R, 40) \
            CASE_GEMM(R, 41) CASE_GEMM(R, 44) CASE_GEMM(R, 45) CASE_GEMM(R, 46) CASE_GEMM(R, 49) CASE_GEMM(R, 54) \
            default: BLASKernel::gemm(m, n, k, alpha, A, lda, B, ldb, beta, C_ptr, ldc); break; \
        }

    #define CASE_ROW_GEMM(R) case R: SWITCH_GEMM(R); break;

    static void gemm(int m, int n, int k, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C_ptr, int ldc) {
        // Only optimized for packed A (lda=m) and NoTrans
        if (lda == m) {
            switch(m) {
                CASE_ROW_GEMM(1) CASE_ROW_GEMM(2) CASE_ROW_GEMM(3) CASE_ROW_GEMM(4) CASE_ROW_GEMM(5)
                CASE_ROW_GEMM(6) CASE_ROW_GEMM(7) CASE_ROW_GEMM(8) CASE_ROW_GEMM(9) CASE_ROW_GEMM(10)
                default: BLASKernel::gemm(m, n, k, alpha, A, lda, B, ldb, beta, C_ptr, ldc); break;
            }
        } else {
            BLASKernel::gemm(m, n, k, alpha, A, lda, B, ldb, beta, C_ptr, ldc);
        }
    }

    #define CASE_GEMV_TRANS(R, C) case C: FixedBlockKernel<T, R, C>::gemv_trans(A, x, y, alpha, beta); break;
    #define SWITCH_GEMV_TRANS(R) \
        switch(n) { \
            CASE_GEMV_TRANS(R, 1) CASE_GEMV_TRANS(R, 2) CASE_GEMV_TRANS(R, 3) CASE_GEMV_TRANS(R, 4) CASE_GEMV_TRANS(R, 5) \
            CASE_GEMV_TRANS(R, 6) CASE_GEMV_TRANS(R, 7) CASE_GEMV_TRANS(R, 8) CASE_GEMV_TRANS(R, 9) CASE_GEMV_TRANS(R, 10) \
            CASE_GEMV_TRANS(R, 11) CASE_GEMV_TRANS(R, 12) CASE_GEMV_TRANS(R, 13) CASE_GEMV_TRANS(R, 14) CASE_GEMV_TRANS(R, 15) \
            CASE_GEMV_TRANS(R, 16) \
            default: BLASKernel::gemv(m, n, alpha, A, m, x, 1, beta, y, 1, CblasConjTrans); break; \
        }
    #define CASE_ROW_GEMV_TRANS(R) case R: SWITCH_GEMV_TRANS(R); break;

    static void gemv_trans(int m, int n, T alpha, const T* A, int lda, const T* x, int incx, T beta, T* y, int incy) {
        // y = alpha * A^T * x + beta * y
        if (incx == 1 && incy == 1 && lda == m) {
            switch(m) {
                CASE_ROW_GEMV_TRANS(1) CASE_ROW_GEMV_TRANS(2) CASE_ROW_GEMV_TRANS(3) CASE_ROW_GEMV_TRANS(4) CASE_ROW_GEMV_TRANS(5)
                CASE_ROW_GEMV_TRANS(6) CASE_ROW_GEMV_TRANS(7) CASE_ROW_GEMV_TRANS(8) CASE_ROW_GEMV_TRANS(9) CASE_ROW_GEMV_TRANS(10)
                CASE_ROW_GEMV_TRANS(11) CASE_ROW_GEMV_TRANS(12) CASE_ROW_GEMV_TRANS(13) CASE_ROW_GEMV_TRANS(14) CASE_ROW_GEMV_TRANS(15)
                CASE_ROW_GEMV_TRANS(16)
                default: BLASKernel::gemv(m, n, alpha, A, lda, x, incx, beta, y, incy, CblasConjTrans); break;
            }
        } else {
            BLASKernel::gemv(m, n, alpha, A, lda, x, incx, beta, y, incy, CblasConjTrans);
        }
    }

    #define CASE_GEMM_TRANS(R, C) case C: FixedBlockKernel<T, R, C>::gemm_trans(n, A, lda, B, ldb, C_ptr, ldc, alpha, beta); break;
    #define SWITCH_GEMM_TRANS(R) \
        switch(k) { \
            CASE_GEMM_TRANS(R, 1) CASE_GEMM_TRANS(R, 2) CASE_GEMM_TRANS(R, 3) CASE_GEMM_TRANS(R, 4) CASE_GEMM_TRANS(R, 5) \
            CASE_GEMM_TRANS(R, 6) CASE_GEMM_TRANS(R, 7) CASE_GEMM_TRANS(R, 8) CASE_GEMM_TRANS(R, 9) CASE_GEMM_TRANS(R, 10) \
            default: BLASKernel::gemm(k, n, m, alpha, A, lda, B, ldb, beta, C_ptr, ldc, CblasConjTrans, CblasNoTrans); break; \
        }
    #define CASE_ROW_GEMM_TRANS(R) case R: SWITCH_GEMM_TRANS(R); break;

    static void gemm_trans(int m, int n, int k, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C_ptr, int ldc) {
        // C = alpha * A^T * B + beta * C
        // A: M x K. A^T: K x M.
        // B: M x n. C: K x n.
        // BLAS GEMM: C(MxN) = A(MxK) * B(KxN).
        // Here: C(Kxn) = A^T(KxM) * B(Mxn).
        // BLAS args: M_blas=K, N_blas=n, K_blas=M.
        if (lda == m) {
            switch(m) {
                CASE_ROW_GEMM_TRANS(1) CASE_ROW_GEMM_TRANS(2) CASE_ROW_GEMM_TRANS(3) CASE_ROW_GEMM_TRANS(4) CASE_ROW_GEMM_TRANS(5)
                CASE_ROW_GEMM_TRANS(6) CASE_ROW_GEMM_TRANS(7) CASE_ROW_GEMM_TRANS(8) CASE_ROW_GEMM_TRANS(9) CASE_ROW_GEMM_TRANS(10)
                default: BLASKernel::gemm(k, n, m, alpha, A, lda, B, ldb, beta, C_ptr, ldc, CblasConjTrans, CblasNoTrans); break;
            }
        } else {
            BLASKernel::gemm(k, n, m, alpha, A, lda, B, ldb, beta, C_ptr, ldc, CblasConjTrans, CblasNoTrans);
        }
    }

    static void gemv_batched(
        int m,
        int n,
        T alpha,
        const T* A,
        int lda,
        int stridea,
        const T* x,
        int incx,
        int stridex,
        T beta,
        T* y,
        int incy,
        int stridey,
        int batch_count) {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>) {
            if (BLASKernel::supports_strided_gemv()) {
                BLASKernel::gemv_batched(
                    m, n, alpha, A, lda, stridea, x, incx, stridex, beta, y, incy, stridey, batch_count);
                return;
            }
        }
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
                incy);
        }
    }

    static void gemm_batched(
        int m,
        int n,
        int k,
        T alpha,
        const T* A,
        int lda,
        int stridea,
        const T* B,
        int ldb,
        int strideb,
        T beta,
        T* C_ptr,
        int ldc,
        int stridec,
        int batch_count) {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>) {
            if (BLASKernel::supports_strided_gemm()) {
                BLASKernel::gemm_batched(
                    m, n, k, alpha, A, lda, stridea, B, ldb, strideb, beta, C_ptr, ldc, stridec, batch_count);
                return;
            }
        }
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
                C_ptr + static_cast<size_t>(batch) * stridec,
                ldc);
        }
    }

    static void gemv_trans_batched(
        int m,
        int n,
        T alpha,
        const T* A,
        int lda,
        int stridea,
        const T* x,
        int incx,
        int stridex,
        T beta,
        T* y,
        int incy,
        int stridey,
        int batch_count) {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>) {
            if (BLASKernel::supports_strided_gemv()) {
                BLASKernel::gemv_batched(
                    m,
                    n,
                    alpha,
                    A,
                    lda,
                    stridea,
                    x,
                    incx,
                    stridex,
                    beta,
                    y,
                    incy,
                    stridey,
                    batch_count,
                    CblasConjTrans);
                return;
            }
        }
        for (int batch = 0; batch < batch_count; ++batch) {
            gemv_trans(
                m,
                n,
                alpha,
                A + static_cast<size_t>(batch) * stridea,
                lda,
                x + static_cast<size_t>(batch) * stridex,
                incx,
                beta,
                y + static_cast<size_t>(batch) * stridey,
                incy);
        }
    }

    static void gemm_trans_batched(
        int m,
        int n,
        int k,
        T alpha,
        const T* A,
        int lda,
        int stridea,
        const T* B,
        int ldb,
        int strideb,
        T beta,
        T* C_ptr,
        int ldc,
        int stridec,
        int batch_count) {
        if constexpr (std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>) {
            if (BLASKernel::supports_strided_gemm()) {
                BLASKernel::gemm_batched(
                    k,
                    n,
                    m,
                    alpha,
                    A,
                    lda,
                    stridea,
                    B,
                    ldb,
                    strideb,
                    beta,
                    C_ptr,
                    ldc,
                    stridec,
                    batch_count,
                    CblasConjTrans,
                    CblasNoTrans);
                return;
            }
        }
        for (int batch = 0; batch < batch_count; ++batch) {
            gemm_trans(
                m,
                n,
                k,
                alpha,
                A + static_cast<size_t>(batch) * stridea,
                lda,
                B + static_cast<size_t>(batch) * strideb,
                ldb,
                beta,
                C_ptr + static_cast<size_t>(batch) * stridec,
                ldc);
        }
    }
    
    // Cleanup macros
    #undef CASE_GEMV
    #undef SWITCH_GEMV
    #undef CASE_ROW_GEMV
    #undef CASE_GEMM
    #undef SWITCH_GEMM
    #undef CASE_ROW_GEMM
    #undef CASE_GEMV_TRANS
    #undef SWITCH_GEMV_TRANS
    #undef CASE_ROW_GEMV_TRANS
    #undef CASE_GEMM_TRANS
    #undef SWITCH_GEMM_TRANS
    #undef CASE_ROW_GEMM_TRANS
};




// Default alias
template <typename T>
using DefaultKernel = BLASKernel; 
// Or NaiveKernel<T> if BLAS not available. 
// Ideally controlled by macro.

} // namespace vbcsr

#endif

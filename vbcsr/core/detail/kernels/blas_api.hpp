#ifndef VBCSR_DETAIL_KERNELS_BLAS_API_HPP
#define VBCSR_DETAIL_KERNELS_BLAS_API_HPP

#if defined(VBCSR_BLAS_ILP64) || defined(VBCSR_USE_ILP64)
#include <cstdint>
using vbcsr_blas_int = int64_t;
#else
using vbcsr_blas_int = int;
#endif

#ifdef VBCSR_USE_OPENBLAS
extern "C" void openblas_set_num_threads(int num_threads);
#endif

#ifdef VBCSR_USE_MKL
extern "C" void mkl_set_num_threads_(int* num_threads);
#endif

extern "C" {
    // Basic BLAS signatures
    void cblas_dgemv(const int Order, const int TransA, const vbcsr_blas_int M, const vbcsr_blas_int N,
                     const double alpha, const double *A, const vbcsr_blas_int lda,
                     const double *X, const vbcsr_blas_int incX, const double beta,
                     double *Y, const vbcsr_blas_int incY);

    void cblas_dgemm(const int Order, const int TransA, const int TransB,
                     const vbcsr_blas_int M, const vbcsr_blas_int N, const vbcsr_blas_int K,
                     const double alpha, const double *A, const vbcsr_blas_int lda,
                     const double *B, const vbcsr_blas_int ldb,
                     const double beta, double *C, const vbcsr_blas_int ldc);

    // Complex versions (zgemv, zgemm) usually take void* for alpha/beta/scalars in some implementations
    // or pass by value in others. Standard CBLAS uses void*.
    void cblas_zgemv(const int Order, const int TransA, const vbcsr_blas_int M, const vbcsr_blas_int N,
                     const void *alpha, const void *A, const vbcsr_blas_int lda,
                     const void *X, const vbcsr_blas_int incX, const void *beta,
                     void *Y, const vbcsr_blas_int incY);

    void cblas_zgemm(const int Order, const int TransA, const int TransB,
                     const vbcsr_blas_int M, const vbcsr_blas_int N, const vbcsr_blas_int K,
                     const void *alpha, const void *A, const vbcsr_blas_int lda,
                     const void *B, const vbcsr_blas_int ldb,
                     const void *beta, void *C, const vbcsr_blas_int ldc);

#ifdef VBCSR_BLAS_HAS_BATCH_GEMM
    void cblas_dgemm_batch_strided(const int Order, const int TransA, const int TransB,
                                   const vbcsr_blas_int M, const vbcsr_blas_int N, const vbcsr_blas_int K,
                                   const double alpha, const double *A, const vbcsr_blas_int lda, const vbcsr_blas_int stridea,
                                   const double *B, const vbcsr_blas_int ldb, const vbcsr_blas_int strideb,
                                   const double beta, double *C, const vbcsr_blas_int ldc, const vbcsr_blas_int stridec,
                                   const vbcsr_blas_int batch_size);
    void cblas_zgemm_batch_strided(const int Order, const int TransA, const int TransB,
                                   const vbcsr_blas_int M, const vbcsr_blas_int N, const vbcsr_blas_int K,
                                   const void *alpha, const void *A, const vbcsr_blas_int lda, const vbcsr_blas_int stridea,
                                   const void *B, const vbcsr_blas_int ldb, const vbcsr_blas_int strideb,
                                   const void *beta, void *C, const vbcsr_blas_int ldc, const vbcsr_blas_int stridec,
                                   const vbcsr_blas_int batch_size);
#endif

#ifdef VBCSR_BLAS_HAS_BATCH_GEMV
    void cblas_dgemv_batch_strided(const int Order, const int TransA,
                                   const vbcsr_blas_int M, const vbcsr_blas_int N,
                                   const double alpha, const double *A, const vbcsr_blas_int lda, const vbcsr_blas_int stridea,
                                   const double *X, const vbcsr_blas_int incX, const vbcsr_blas_int stridex,
                                   const double beta, double *Y, const vbcsr_blas_int incY, const vbcsr_blas_int stridey,
                                   const vbcsr_blas_int batch_size);
    void cblas_zgemv_batch_strided(const int Order, const int TransA,
                                   const vbcsr_blas_int M, const vbcsr_blas_int N,
                                   const void *alpha, const void *A, const vbcsr_blas_int lda, const vbcsr_blas_int stridea,
                                   const void *X, const vbcsr_blas_int incX, const vbcsr_blas_int stridex,
                                   const void *beta, void *Y, const vbcsr_blas_int incY, const vbcsr_blas_int stridey,
                                   const vbcsr_blas_int batch_size);
#endif
}

namespace vbcsr {

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

} // namespace vbcsr

#endif // VBCSR_DETAIL_KERNELS_BLAS_API_HPP

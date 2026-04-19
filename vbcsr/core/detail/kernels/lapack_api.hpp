#ifndef VBCSR_DETAIL_KERNELS_LAPACK_API_HPP
#define VBCSR_DETAIL_KERNELS_LAPACK_API_HPP

#include <complex>

namespace vbcsr {

#if defined(VBCSR_USE_ILP64)
    typedef long long vbcsr_lapack_int;
#else
    typedef int vbcsr_lapack_int;
#endif

using vbcsr_complex_double = std::complex<double>;

extern "C" {
    // BLAS Level 3
    void dgemm_(const char* transa, const char* transb, const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, const vbcsr_lapack_int* k, 
                const double* alpha, const double* a, const vbcsr_lapack_int* lda, const double* b, const vbcsr_lapack_int* ldb, 
                const double* beta, double* c, const vbcsr_lapack_int* ldc);

    void zgemm_(const char* transa, const char* transb, const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, const vbcsr_lapack_int* k, 
                const vbcsr_complex_double* alpha, const vbcsr_complex_double* a, const vbcsr_lapack_int* lda, 
                const vbcsr_complex_double* b, const vbcsr_lapack_int* ldb, 
                const vbcsr_complex_double* beta, vbcsr_complex_double* c, const vbcsr_lapack_int* ldc);

    void dtrsm_(const char* side, const char* uplo, const char* transa, const char* diag, 
                const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, const double* alpha, 
                const double* a, const vbcsr_lapack_int* lda, double* b, const vbcsr_lapack_int* ldb);

    void ztrsm_(const char* side, const char* uplo, const char* transa, const char* diag, 
                const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, const vbcsr_complex_double* alpha, 
                const vbcsr_complex_double* a, const vbcsr_lapack_int* lda, vbcsr_complex_double* b, const vbcsr_lapack_int* ldb);

    // Double precision symmetric eigendecomposition
    void dsyev_(const char* jobz, const char* uplo, const vbcsr_lapack_int* n, double* a, const vbcsr_lapack_int* lda, double* w, double* work, const vbcsr_lapack_int* lwork, vbcsr_lapack_int* info);
    void dsyevd_(const char* jobz, const char* uplo, const vbcsr_lapack_int* n, double* a, const vbcsr_lapack_int* lda, double* w, double* work, const vbcsr_lapack_int* lwork, vbcsr_lapack_int* iwork, const vbcsr_lapack_int* liwork, vbcsr_lapack_int* info);
    
    // Complex double Hermitian eigendecomposition
    void zheev_(const char* jobz, const char* uplo, const vbcsr_lapack_int* n, vbcsr_complex_double* a, const vbcsr_lapack_int* lda, double* w, vbcsr_complex_double* work, const vbcsr_lapack_int* lwork, double* rwork, vbcsr_lapack_int* info);
    void zheevd_(const char* jobz, const char* uplo, const vbcsr_lapack_int* n, vbcsr_complex_double* a, const vbcsr_lapack_int* lda, double* w, vbcsr_complex_double* work, const vbcsr_lapack_int* lwork, double* rwork, const vbcsr_lapack_int* lrwork, vbcsr_lapack_int* iwork, const vbcsr_lapack_int* liwork, vbcsr_lapack_int* info);

    // LU Factorization (Double)
    void dgetrf_(const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, double* a, const vbcsr_lapack_int* lda, vbcsr_lapack_int* ipiv, vbcsr_lapack_int* info);
    
    // LU Inversion (Double)
    void dgetri_(const vbcsr_lapack_int* n, double* a, const vbcsr_lapack_int* lda, const vbcsr_lapack_int* ipiv, double* work, const vbcsr_lapack_int* lwork, vbcsr_lapack_int* info);

    // LU Factorization (Complex Double)
    void zgetrf_(const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, vbcsr_complex_double* a, const vbcsr_lapack_int* lda, vbcsr_lapack_int* ipiv, vbcsr_lapack_int* info);

    // LU Inversion (Complex Double)
    void zgetri_(const vbcsr_lapack_int* n, vbcsr_complex_double* a, const vbcsr_lapack_int* lda, const vbcsr_lapack_int* ipiv, vbcsr_complex_double* work, const vbcsr_lapack_int* lwork, vbcsr_lapack_int* info);

    // Cholesky Factorization
    void dpotrf_(const char* uplo, const vbcsr_lapack_int* n, double* a, const vbcsr_lapack_int* lda, vbcsr_lapack_int* info);
    void zpotrf_(const char* uplo, const vbcsr_lapack_int* n, vbcsr_complex_double* a, const vbcsr_lapack_int* lda, vbcsr_lapack_int* info);

    // Rank-Revealing QR (GEQP3)
    void dgeqp3_(const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, double* a, const vbcsr_lapack_int* lda, vbcsr_lapack_int* jpvt, double* tau, double* work, const vbcsr_lapack_int* lwork, vbcsr_lapack_int* info);
    void zgeqp3_(const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, vbcsr_complex_double* a, const vbcsr_lapack_int* lda, vbcsr_lapack_int* jpvt, vbcsr_complex_double* tau, vbcsr_complex_double* work, const vbcsr_lapack_int* lwork, double* rwork, vbcsr_lapack_int* info);

    // SVD (GESVD)
    void dgesvd_(const char* jobu, const char* jobvt, const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, double* a, const vbcsr_lapack_int* lda, double* s, double* u, const vbcsr_lapack_int* ldu, double* vt, const vbcsr_lapack_int* ldvt, double* work, const vbcsr_lapack_int* lwork, vbcsr_lapack_int* info);
    void zgesvd_(const char* jobu, const char* jobvt, const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, vbcsr_complex_double* a, const vbcsr_lapack_int* lda, double* s, vbcsr_complex_double* u, const vbcsr_lapack_int* ldu, vbcsr_complex_double* vt, const vbcsr_lapack_int* ldvt, vbcsr_complex_double* work, const vbcsr_lapack_int* lwork, double* rwork, vbcsr_lapack_int* info);

    // Generate Q from QR (ORGQR/UNGQR)
    void dorgqr_(const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, const vbcsr_lapack_int* k, double* a, const vbcsr_lapack_int* lda, const double* tau, double* work, const vbcsr_lapack_int* lwork, vbcsr_lapack_int* info);
    void zungqr_(const vbcsr_lapack_int* m, const vbcsr_lapack_int* n, const vbcsr_lapack_int* k, vbcsr_complex_double* a, const vbcsr_lapack_int* lda, const vbcsr_complex_double* tau, vbcsr_complex_double* work, const vbcsr_lapack_int* lwork, vbcsr_lapack_int* info);

    // MKL Threading Control (Fortran Interface)
    void mkl_set_num_threads_(int* num_threads);
}

}

#endif // VBCSR_DETAIL_KERNELS_LAPACK_API_HPP

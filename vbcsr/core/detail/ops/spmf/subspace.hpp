#ifndef VBCSR_DETAIL_OPS_SPMF_SUBSPACE_HPP
#define VBCSR_DETAIL_OPS_SPMF_SUBSPACE_HPP

#include "../../../block_csr.hpp"
#include "../../../dist_multivector.hpp"
#include "../../kernels/lapack_api.hpp"
#include <vector>
#include <cmath>
#include <complex>
#include <iostream>
#include <algorithm>
#include <functional>
#include <mpi.h>
#include <type_traits>

namespace vbcsr {

namespace detail {

    // C = alpha * A * B + beta * C
    template <typename T>
    void dense_gemm(int m, int n, int k, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C, int ldc, bool transA = false, bool transB = false) {
        const char* ta = transA ? (std::is_same<T, std::complex<double>>::value ? "C" : "T") : "N";
        const char* tb = transB ? (std::is_same<T, std::complex<double>>::value ? "C" : "T") : "N";
        
        vbcsr_lapack_int m_ = m;
        vbcsr_lapack_int n_ = n;
        vbcsr_lapack_int k_ = k;
        vbcsr_lapack_int lda_ = lda;
        vbcsr_lapack_int ldb_ = ldb;
        vbcsr_lapack_int ldc_ = ldc;

        if constexpr (std::is_same<T, double>::value) {
            dgemm_(ta, tb, &m_, &n_, &k_, &alpha, A, &lda_, B, &ldb_, &beta, C, &ldc_);
        } else if constexpr (std::is_same<T, std::complex<double>>::value) {
             zgemm_(ta, tb, &m_, &n_, &k_, 
                    reinterpret_cast<const vbcsr_complex_double*>(&alpha), 
                    reinterpret_cast<const vbcsr_complex_double*>(A), &lda_, 
                    reinterpret_cast<const vbcsr_complex_double*>(B), &ldb_, 
                    reinterpret_cast<const vbcsr_complex_double*>(&beta), 
                    reinterpret_cast<vbcsr_complex_double*>(C), &ldc_);
        }
    }

} // namespace detail


// Helper to compute f(M) for a dense matrix M
template <typename T>
void dense_matrix_function(int n_in, std::vector<T>& M, std::function<T(double)> func, int k_cols = -1, int col_start_idx = 0) {
    vbcsr_lapack_int n = n_in;
    if (n == 0) return;
    
    int k = (k_cols <= 0) ? n : k_cols;
    
    std::vector<double> w(n); // Eigenvalues
    vbcsr_lapack_int info;
    vbcsr_lapack_int lwork = -1;
    
    if constexpr (std::is_same<T, double>::value) {
        double wkopt;
        vbcsr_lapack_int iwkopt;
        vbcsr_lapack_int lwork_query = -1;
        vbcsr_lapack_int liwork_query = -1;
        dsyevd_("V", "U", &n, (double*)M.data(), &n, w.data(), &wkopt, &lwork_query, &iwkopt, &liwork_query, &info);
        lwork = (vbcsr_lapack_int)wkopt;
        vbcsr_lapack_int liwork = iwkopt;
        std::vector<double> work(lwork);
        std::vector<vbcsr_lapack_int> iwork(liwork);
        dsyevd_("V", "U", &n, (double*)M.data(), &n, w.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
    } else {
        vbcsr_complex_double wkopt;
        double rwkopt;
        vbcsr_lapack_int iwkopt;
        vbcsr_lapack_int lwork_query = -1;
        vbcsr_lapack_int lrwork_query = -1;
        vbcsr_lapack_int liwork_query = -1;
        vbcsr_complex_double* a_ptr = reinterpret_cast<vbcsr_complex_double*>(M.data());
        zheevd_("V", "U", &n, a_ptr, &n, w.data(), &wkopt, &lwork_query, &rwkopt, &lrwork_query, &iwkopt, &liwork_query, &info);
        lwork = (vbcsr_lapack_int)wkopt.real();
        vbcsr_lapack_int lrwork = (vbcsr_lapack_int)rwkopt;
        vbcsr_lapack_int liwork = iwkopt;
        std::vector<vbcsr_complex_double> work(lwork);
        std::vector<double> rwork(lrwork);
        std::vector<vbcsr_lapack_int> iwork(liwork);
        zheevd_("V", "U", &n, a_ptr, &n, w.data(), work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);
    }
    
    if (info != 0) {
        std::cerr << "Error in LAPACK eigendecomposition (info=" << info << ")" << std::endl;
        return;
    }
    
    // M now contains eigenvectors V. w contains eigenvalues.
    // Result_subset = (V * diag(f(w))) * (V^H)[:, col_start_idx : col_start_idx + k]
    std::vector<T> V_scaled = M;
    for (int j = 0; j < n; ++j) {
        T val = func(w[j]);
        for (int i = 0; i < n; ++i) {
            V_scaled[i + j * n] *= val;
        }
    }
    
    std::vector<T> Res(n * k, T(0));
    // C(n x k) = V_scaled(n x n) * M(k x n)^H
    // Select rows [col_start_idx, col_start_idx + k) of V (M).
    // Pointer arithmetic `M.data() + col_start_idx` with stride `n` correctly slices the rows.
    detail::dense_gemm(n, k, n, T(1.0), V_scaled.data(), n, M.data() + col_start_idx, n, T(0.0), Res.data(), n, false, true);
    M = Res;
}

} // namespace vbcsr

#endif // VBCSR_DETAIL_OPS_SPMF_SUBSPACE_HPP

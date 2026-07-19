#ifndef VBCSR_DETAIL_KERNELS_ROWMAJOR_KERNELS_HPP
#define VBCSR_DETAIL_KERNELS_ROWMAJOR_KERNELS_HPP

// Row-major vec-axis block kernels (doc/row_major_migration_plan.md §2.2,
// validated by the Phase-1 microbenchmark gate).
//
// Layout contract:
//   X row-major: X(k, v) at x[k * ldx + v], ldx padded (the multivector's ld).
//   C row-major: C(i, v) at c[i * ldc + v].
//   A row-major within the block: A(i, k) at a[i * K + k] — the canonical
//     block storage (kCanonicalBlockLayout, flipped in Phase 4). A is a
//     broadcast operand, so the layout only shows up in the few A-indexing
//     expressions below.
//
// Semantics: C += op(A) * X with alpha = 1, beta = 1 (callers zero C first),
// matching every live apply call site. The adjoint op is conjugate-transpose.
//
// The rm_gemv/rm_gemv_adjoint pair covers SpMV (single dense vector, no ld):
// forward is a dot per contiguous A row; adjoint streams conj(A row) * x[i].
//
// Callers pass nv = the padded ld when X/C padding lanes are zero-maintained
// (the DistMultiVector invariant): the pad lanes then compute to exact zeros,
// preserving the invariant while keeping every SIMD chunk full.
//
// SIMD dispatch: AVX2+FMA fast paths for double and complex<double>; a plain
// scalar implementation covers every other type and non-AVX2 builds.

#include <complex>
#include <cstddef>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

#include "../../scalar_traits.hpp"

namespace vbcsr::detail {
namespace rowmajor_kernels {

// ---------------------------------------------------------------------------
// Generic scalar reference paths (any T; also the non-AVX2 fallback).
// ---------------------------------------------------------------------------
template <typename T>
inline void rm_gemm_generic(
    int M, int K, int nv, const T* a, const T* x, int ldx, T* c, int ldc) {
    for (int i = 0; i < M; ++i) {
        T* c_row = c + static_cast<size_t>(i) * ldc;
        const T* a_row = a + static_cast<size_t>(i) * K;
        for (int k = 0; k < K; ++k) {
            const T aik = a_row[k];
            const T* x_row = x + static_cast<size_t>(k) * ldx;
            for (int v = 0; v < nv; ++v) {
                c_row[v] += aik * x_row[v];
            }
        }
    }
}

template <typename T>
inline void rm_gemm_adjoint_generic(
    int M, int K, int nv, const T* a, const T* x, int ldx, T* c, int ldc) {
    for (int i = 0; i < M; ++i) {
        const T* a_row = a + static_cast<size_t>(i) * K;
        const T* x_row = x + static_cast<size_t>(i) * ldx;
        for (int k = 0; k < K; ++k) {
            const T aik = ScalarTraits<T>::conjugate(a_row[k]);
            T* c_row = c + static_cast<size_t>(k) * ldc;
            for (int v = 0; v < nv; ++v) {
                c_row[v] += aik * x_row[v];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SpMV pair (single dense vector): y += op(A) * x. Generic scalar paths;
// AVX2 double specializations below.
// ---------------------------------------------------------------------------
template <typename T>
inline void rm_gemv(int M, int K, const T* a, const T* x, T* y) {
    for (int i = 0; i < M; ++i) {
        const T* a_row = a + static_cast<size_t>(i) * K;
        T sum = T(0);
        for (int k = 0; k < K; ++k) {
            sum += a_row[k] * x[k];
        }
        y[i] += sum;
    }
}

template <typename T>
inline void rm_gemv_adjoint(int M, int K, const T* a, const T* x, T* y) {
    for (int i = 0; i < M; ++i) {
        const T* a_row = a + static_cast<size_t>(i) * K;
        const T xi = x[i];
        for (int k = 0; k < K; ++k) {
            y[k] += ScalarTraits<T>::conjugate(a_row[k]) * xi;
        }
    }
}

#if defined(__AVX2__) && defined(__FMA__)

// ---------------------------------------------------------------------------
// double forward: C(M x nv) += A * X. 3-row output tiles x V ymm columns;
// one broadcast per row per k, X chunk loads shared across the tile;
// k-unrolled by 2 (Phase-1 winning configuration).
// ---------------------------------------------------------------------------
template <int V>
inline void rm_gemm_d_chunk(
    int M, int K, const double* a, const double* x, int ldx, double* c, int ldc) {
    static_assert(V >= 1 && V <= 4, "V must be 1..4 ymm");
    int i = 0;
    for (; i + 3 <= M; i += 3) {
        __m256d acc0[V], acc1[V], acc2[V];
        double* c0 = c + static_cast<size_t>(i) * ldc;
        double* c1 = c0 + ldc;
        double* c2 = c1 + ldc;
        for (int v = 0; v < V; ++v) {
            acc0[v] = _mm256_loadu_pd(c0 + 4 * v);
            acc1[v] = _mm256_loadu_pd(c1 + 4 * v);
            acc2[v] = _mm256_loadu_pd(c2 + 4 * v);
        }
        const double* a0 = a + static_cast<size_t>(i) * K;
        const double* a1 = a0 + K;
        const double* a2 = a1 + K;
        int k = 0;
        for (; k + 2 <= K; k += 2) {
            const double* xr0 = x + static_cast<size_t>(k) * ldx;
            const double* xr1 = xr0 + ldx;
            const __m256d b00 = _mm256_set1_pd(a0[k]);
            const __m256d b01 = _mm256_set1_pd(a1[k]);
            const __m256d b02 = _mm256_set1_pd(a2[k]);
            const __m256d b10 = _mm256_set1_pd(a0[k + 1]);
            const __m256d b11 = _mm256_set1_pd(a1[k + 1]);
            const __m256d b12 = _mm256_set1_pd(a2[k + 1]);
            for (int v = 0; v < V; ++v) {
                const __m256d xv0 = _mm256_loadu_pd(xr0 + 4 * v);
                acc0[v] = _mm256_fmadd_pd(b00, xv0, acc0[v]);
                acc1[v] = _mm256_fmadd_pd(b01, xv0, acc1[v]);
                acc2[v] = _mm256_fmadd_pd(b02, xv0, acc2[v]);
                const __m256d xv1 = _mm256_loadu_pd(xr1 + 4 * v);
                acc0[v] = _mm256_fmadd_pd(b10, xv1, acc0[v]);
                acc1[v] = _mm256_fmadd_pd(b11, xv1, acc1[v]);
                acc2[v] = _mm256_fmadd_pd(b12, xv1, acc2[v]);
            }
        }
        for (; k < K; ++k) {
            const double* xr = x + static_cast<size_t>(k) * ldx;
            const __m256d b0 = _mm256_set1_pd(a0[k]);
            const __m256d b1 = _mm256_set1_pd(a1[k]);
            const __m256d b2 = _mm256_set1_pd(a2[k]);
            for (int v = 0; v < V; ++v) {
                const __m256d xv = _mm256_loadu_pd(xr + 4 * v);
                acc0[v] = _mm256_fmadd_pd(b0, xv, acc0[v]);
                acc1[v] = _mm256_fmadd_pd(b1, xv, acc1[v]);
                acc2[v] = _mm256_fmadd_pd(b2, xv, acc2[v]);
            }
        }
        for (int v = 0; v < V; ++v) {
            _mm256_storeu_pd(c0 + 4 * v, acc0[v]);
            _mm256_storeu_pd(c1 + 4 * v, acc1[v]);
            _mm256_storeu_pd(c2 + 4 * v, acc2[v]);
        }
    }
    for (; i < M; ++i) {
        __m256d acc[V];
        double* c0 = c + static_cast<size_t>(i) * ldc;
        const double* ar = a + static_cast<size_t>(i) * K;
        for (int v = 0; v < V; ++v) {
            acc[v] = _mm256_loadu_pd(c0 + 4 * v);
        }
        for (int k = 0; k < K; ++k) {
            const double* xr = x + static_cast<size_t>(k) * ldx;
            const __m256d b0 = _mm256_set1_pd(ar[k]);
            for (int v = 0; v < V; ++v) {
                acc[v] = _mm256_fmadd_pd(b0, _mm256_loadu_pd(xr + 4 * v), acc[v]);
            }
        }
        for (int v = 0; v < V; ++v) {
            _mm256_storeu_pd(c0 + 4 * v, acc[v]);
        }
    }
}

// double adjoint: C(K x nv) += A^T * X. Output rows tile over K; the inner
// contraction walks A rows (the 3 broadcasts per row read adjacent row-major
// elements a_row[k..k+2] — one cache line) and shares X chunk loads across
// the 3-row tile.
template <int V>
inline void rm_gemm_adj_d_chunk(
    int M, int K, const double* a, const double* x, int ldx, double* c, int ldc) {
    static_assert(V >= 1 && V <= 4, "V must be 1..4 ymm");
    int k = 0;
    for (; k + 3 <= K; k += 3) {
        __m256d acc0[V], acc1[V], acc2[V];
        double* c0 = c + static_cast<size_t>(k) * ldc;
        double* c1 = c0 + ldc;
        double* c2 = c1 + ldc;
        for (int v = 0; v < V; ++v) {
            acc0[v] = _mm256_loadu_pd(c0 + 4 * v);
            acc1[v] = _mm256_loadu_pd(c1 + 4 * v);
            acc2[v] = _mm256_loadu_pd(c2 + 4 * v);
        }
        for (int i = 0; i < M; ++i) {
            const double* ar = a + static_cast<size_t>(i) * K + k;
            const double* xr = x + static_cast<size_t>(i) * ldx;
            const __m256d b0 = _mm256_set1_pd(ar[0]);
            const __m256d b1 = _mm256_set1_pd(ar[1]);
            const __m256d b2 = _mm256_set1_pd(ar[2]);
            for (int v = 0; v < V; ++v) {
                const __m256d xv = _mm256_loadu_pd(xr + 4 * v);
                acc0[v] = _mm256_fmadd_pd(b0, xv, acc0[v]);
                acc1[v] = _mm256_fmadd_pd(b1, xv, acc1[v]);
                acc2[v] = _mm256_fmadd_pd(b2, xv, acc2[v]);
            }
        }
        for (int v = 0; v < V; ++v) {
            _mm256_storeu_pd(c0 + 4 * v, acc0[v]);
            _mm256_storeu_pd(c1 + 4 * v, acc1[v]);
            _mm256_storeu_pd(c2 + 4 * v, acc2[v]);
        }
    }
    for (; k < K; ++k) {
        __m256d acc[V];
        double* c0 = c + static_cast<size_t>(k) * ldc;
        for (int v = 0; v < V; ++v) {
            acc[v] = _mm256_loadu_pd(c0 + 4 * v);
        }
        for (int i = 0; i < M; ++i) {
            const double* xr = x + static_cast<size_t>(i) * ldx;
            const __m256d b0 = _mm256_set1_pd(a[static_cast<size_t>(i) * K + k]);
            for (int v = 0; v < V; ++v) {
                acc[v] = _mm256_fmadd_pd(b0, _mm256_loadu_pd(xr + 4 * v), acc[v]);
            }
        }
        for (int v = 0; v < V; ++v) {
            _mm256_storeu_pd(c0 + 4 * v, acc[v]);
        }
    }
}

// ---------------------------------------------------------------------------
// complex<double>: panel design (Phase-1 winner). The swapped/sign-flipped
// copy of X ([-xi, xr] pairs) is built once per call into a thread-local
// panel; the row loop is then pure FMA (2 loads + 2 FMAs per ymm).
// The adjoint flips the sign of the imaginary broadcast (conjugation).
// ---------------------------------------------------------------------------
inline std::vector<double>& rm_z_panel_storage() {
    thread_local std::vector<double> panel;
    return panel;
}

inline void rm_z_build_panel(
    int rows, int nv_vec, const std::complex<double>* x, int ldx, std::vector<double>& panel) {
    const int panel_ld = 2 * nv_vec;
    if (panel.size() < static_cast<size_t>(rows) * panel_ld) {
        panel.resize(static_cast<size_t>(rows) * panel_ld);
    }
    const __m256d neg_even = _mm256_set_pd(0.0, -0.0, 0.0, -0.0);
    const double* x_d = reinterpret_cast<const double*>(x);
    const int ldx_d = 2 * ldx;
    for (int r = 0; r < rows; ++r) {
        const double* xr = x_d + static_cast<size_t>(r) * ldx_d;
        double* dst = panel.data() + static_cast<size_t>(r) * panel_ld;
        for (int d = 0; d + 4 <= panel_ld; d += 4) {
            const __m256d xv = _mm256_loadu_pd(xr + d);
            _mm256_storeu_pd(dst + d, _mm256_xor_pd(_mm256_permute_pd(xv, 0x5), neg_even));
        }
    }
}

// One output row of the complex product: c_row += sum_k (ar +/- i*ai) x_row_k
// with per-k broadcasts; V ymm chunks of the row. Forward-only (output rows
// are i in [0,M), contraction over k in [0,K)); the adjoint path uses
// rm_gemm_z_adj_rows.
template <int V, bool Conjugate>
inline void rm_gemm_z_rows(
    int M,
    int K,
    const std::complex<double>* a,
    const double* x_d,
    int ldx_d,
    const double* panel,
    int panel_ld,
    std::complex<double>* c,
    int ldc,
    int vc) {
    for (int o = 0; o < M; ++o) {
        double* c0 = reinterpret_cast<double*>(c + static_cast<size_t>(o) * ldc + vc);
        __m256d acc[V];
        for (int v = 0; v < V; ++v) {
            acc[v] = _mm256_loadu_pd(c0 + 4 * v);
        }
        for (int q = 0; q < K; ++q) {
            const std::complex<double> aval = a[static_cast<size_t>(o) * K + q];
            const __m256d br = _mm256_set1_pd(aval.real());
            const __m256d bi = _mm256_set1_pd(Conjugate ? -aval.imag() : aval.imag());
            const double* xr = x_d + static_cast<size_t>(q) * ldx_d + 2 * vc;
            const double* xsr = panel + static_cast<size_t>(q) * panel_ld + 2 * vc;
            for (int v = 0; v < V; ++v) {
                acc[v] = _mm256_fmadd_pd(br, _mm256_loadu_pd(xr + 4 * v), acc[v]);
                acc[v] = _mm256_fmadd_pd(bi, _mm256_loadu_pd(xsr + 4 * v), acc[v]);
            }
        }
        for (int v = 0; v < V; ++v) {
            _mm256_storeu_pd(c0 + 4 * v, acc[v]);
        }
    }
}

// Adjoint output rows in pairs: rows (k, k+1) share every X/panel chunk load
// and read ADJACENT row-major A elements a[i*K + k], a[i*K + k + 1] per
// contraction step — versus the single-row path's lone strided loads, which
// measured ~1.6x slower on the vbcsr complex adjoint. Register budget:
// 2V accumulators + 4 broadcasts + 2 chunk loads (V=4 -> 14 ymm).
template <int V, bool Conjugate>
inline void rm_gemm_z_adj_rows(
    int M,
    int K,
    const std::complex<double>* a,
    const double* x_d,
    int ldx_d,
    const double* panel,
    int panel_ld,
    std::complex<double>* c,
    int ldc,
    int vc) {
    int k = 0;
    for (; k + 2 <= K; k += 2) {
        double* c0 = reinterpret_cast<double*>(c + static_cast<size_t>(k) * ldc + vc);
        double* c1 = reinterpret_cast<double*>(c + static_cast<size_t>(k + 1) * ldc + vc);
        __m256d acc0[V], acc1[V];
        for (int v = 0; v < V; ++v) {
            acc0[v] = _mm256_loadu_pd(c0 + 4 * v);
            acc1[v] = _mm256_loadu_pd(c1 + 4 * v);
        }
        for (int i = 0; i < M; ++i) {
            const std::complex<double>* ap = a + static_cast<size_t>(i) * K + k;
            const __m256d br0 = _mm256_set1_pd(ap[0].real());
            const __m256d bi0 = _mm256_set1_pd(Conjugate ? -ap[0].imag() : ap[0].imag());
            const __m256d br1 = _mm256_set1_pd(ap[1].real());
            const __m256d bi1 = _mm256_set1_pd(Conjugate ? -ap[1].imag() : ap[1].imag());
            const double* xr = x_d + static_cast<size_t>(i) * ldx_d + 2 * vc;
            const double* xsr = panel + static_cast<size_t>(i) * panel_ld + 2 * vc;
            for (int v = 0; v < V; ++v) {
                const __m256d xv = _mm256_loadu_pd(xr + 4 * v);
                const __m256d xs = _mm256_loadu_pd(xsr + 4 * v);
                acc0[v] = _mm256_fmadd_pd(br0, xv, acc0[v]);
                acc0[v] = _mm256_fmadd_pd(bi0, xs, acc0[v]);
                acc1[v] = _mm256_fmadd_pd(br1, xv, acc1[v]);
                acc1[v] = _mm256_fmadd_pd(bi1, xs, acc1[v]);
            }
        }
        for (int v = 0; v < V; ++v) {
            _mm256_storeu_pd(c0 + 4 * v, acc0[v]);
            _mm256_storeu_pd(c1 + 4 * v, acc1[v]);
        }
    }
    for (; k < K; ++k) {
        double* c0 = reinterpret_cast<double*>(c + static_cast<size_t>(k) * ldc + vc);
        __m256d acc[V];
        for (int v = 0; v < V; ++v) {
            acc[v] = _mm256_loadu_pd(c0 + 4 * v);
        }
        for (int i = 0; i < M; ++i) {
            const std::complex<double> aval = a[static_cast<size_t>(i) * K + k];
            const __m256d br = _mm256_set1_pd(aval.real());
            const __m256d bi = _mm256_set1_pd(Conjugate ? -aval.imag() : aval.imag());
            const double* xr = x_d + static_cast<size_t>(i) * ldx_d + 2 * vc;
            const double* xsr = panel + static_cast<size_t>(i) * panel_ld + 2 * vc;
            for (int v = 0; v < V; ++v) {
                acc[v] = _mm256_fmadd_pd(br, _mm256_loadu_pd(xr + 4 * v), acc[v]);
                acc[v] = _mm256_fmadd_pd(bi, _mm256_loadu_pd(xsr + 4 * v), acc[v]);
            }
        }
        for (int v = 0; v < V; ++v) {
            _mm256_storeu_pd(c0 + 4 * v, acc[v]);
        }
    }
}

template <bool Conjugate, bool AdjointIndex>
inline void rm_gemm_z_impl(
    int M,
    int K,
    int nv,
    const std::complex<double>* a,
    const std::complex<double>* x,
    int ldx,
    std::complex<double>* c,
    int ldc) {
    const int nv_vec = nv & ~1;
    const int x_rows = AdjointIndex ? M : K;
    if (nv_vec > 0) {
        auto& panel = rm_z_panel_storage();
        rm_z_build_panel(x_rows, nv_vec, x, ldx, panel);
        const int panel_ld = 2 * nv_vec;
        const double* x_d = reinterpret_cast<const double*>(x);
        const int ldx_d = 2 * ldx;
        int vc = 0;
        if constexpr (AdjointIndex) {
            for (; vc + 8 <= nv_vec; vc += 8) {
                rm_gemm_z_adj_rows<4, Conjugate>(
                    M, K, a, x_d, ldx_d, panel.data(), panel_ld, c, ldc, vc);
            }
            if (vc + 4 <= nv_vec) {
                rm_gemm_z_adj_rows<2, Conjugate>(
                    M, K, a, x_d, ldx_d, panel.data(), panel_ld, c, ldc, vc);
                vc += 4;
            }
            if (vc + 2 <= nv_vec) {
                rm_gemm_z_adj_rows<1, Conjugate>(
                    M, K, a, x_d, ldx_d, panel.data(), panel_ld, c, ldc, vc);
                vc += 2;
            }
        } else {
            for (; vc + 8 <= nv_vec; vc += 8) {
                rm_gemm_z_rows<4, Conjugate>(
                    M, K, a, x_d, ldx_d, panel.data(), panel_ld, c, ldc, vc);
            }
            if (vc + 4 <= nv_vec) {
                rm_gemm_z_rows<2, Conjugate>(
                    M, K, a, x_d, ldx_d, panel.data(), panel_ld, c, ldc, vc);
                vc += 4;
            }
            if (vc + 2 <= nv_vec) {
                rm_gemm_z_rows<1, Conjugate>(
                    M, K, a, x_d, ldx_d, panel.data(), panel_ld, c, ldc, vc);
                vc += 2;
            }
        }
    }
    // Odd trailing lane (nv not a multiple of 2): scalar epilogue.
    for (int v = nv_vec; v < nv; ++v) {
        const int out_rows = AdjointIndex ? K : M;
        const int contraction = AdjointIndex ? M : K;
        for (int o = 0; o < out_rows; ++o) {
            std::complex<double> sum(0.0, 0.0);
            for (int q = 0; q < contraction; ++q) {
                std::complex<double> aval = AdjointIndex
                    ? a[static_cast<size_t>(q) * K + o]
                    : a[static_cast<size_t>(o) * K + q];
                if (Conjugate) aval = std::conj(aval);
                sum += aval * x[static_cast<size_t>(q) * ldx + v];
            }
            c[static_cast<size_t>(o) * ldc + v] += sum;
        }
    }
}

#endif  // AVX2

// ---------------------------------------------------------------------------
// Public entry points (runtime dims; dispatch by type).
// ---------------------------------------------------------------------------
template <typename T>
inline void rm_gemm(int M, int K, int nv, const T* a, const T* x, int ldx, T* c, int ldc) {
    rm_gemm_generic(M, K, nv, a, x, ldx, c, ldc);
}

template <typename T>
inline void rm_gemm_adjoint(int M, int K, int nv, const T* a, const T* x, int ldx, T* c, int ldc) {
    rm_gemm_adjoint_generic(M, K, nv, a, x, ldx, c, ldc);
}

#if defined(__AVX2__) && defined(__FMA__)

template <>
inline void rm_gemm<double>(
    int M, int K, int nv, const double* a, const double* x, int ldx, double* c, int ldc) {
    int vc = 0;
    for (; vc + 16 <= nv; vc += 16) {
        rm_gemm_d_chunk<4>(M, K, a, x + vc, ldx, c + vc, ldc);
    }
    if (vc + 8 <= nv) {
        rm_gemm_d_chunk<2>(M, K, a, x + vc, ldx, c + vc, ldc);
        vc += 8;
    }
    if (vc + 4 <= nv) {
        rm_gemm_d_chunk<1>(M, K, a, x + vc, ldx, c + vc, ldc);
        vc += 4;
    }
    for (; vc < nv; ++vc) {
        for (int i = 0; i < M; ++i) {
            const double* ar = a + static_cast<size_t>(i) * K;
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += ar[k] * x[static_cast<size_t>(k) * ldx + vc];
            }
            c[static_cast<size_t>(i) * ldc + vc] += sum;
        }
    }
}

template <>
inline void rm_gemm_adjoint<double>(
    int M, int K, int nv, const double* a, const double* x, int ldx, double* c, int ldc) {
    int vc = 0;
    for (; vc + 16 <= nv; vc += 16) {
        rm_gemm_adj_d_chunk<4>(M, K, a, x + vc, ldx, c + vc, ldc);
    }
    if (vc + 8 <= nv) {
        rm_gemm_adj_d_chunk<2>(M, K, a, x + vc, ldx, c + vc, ldc);
        vc += 8;
    }
    if (vc + 4 <= nv) {
        rm_gemm_adj_d_chunk<1>(M, K, a, x + vc, ldx, c + vc, ldc);
        vc += 4;
    }
    for (; vc < nv; ++vc) {
        for (int k = 0; k < K; ++k) {
            double sum = 0.0;
            for (int i = 0; i < M; ++i) {
                sum += a[static_cast<size_t>(i) * K + k] * x[static_cast<size_t>(i) * ldx + vc];
            }
            c[static_cast<size_t>(k) * ldc + vc] += sum;
        }
    }
}

template <>
inline void rm_gemm<std::complex<double>>(
    int M,
    int K,
    int nv,
    const std::complex<double>* a,
    const std::complex<double>* x,
    int ldx,
    std::complex<double>* c,
    int ldc) {
    rm_gemm_z_impl<false, false>(M, K, nv, a, x, ldx, c, ldc);
}

template <>
inline void rm_gemm_adjoint<std::complex<double>>(
    int M,
    int K,
    int nv,
    const std::complex<double>* a,
    const std::complex<double>* x,
    int ldx,
    std::complex<double>* c,
    int ldc) {
    rm_gemm_z_impl<true, true>(M, K, nv, a, x, ldx, c, ldc);
}

// SpMV forward, double: dot per contiguous A row; 4 rows share each x chunk
// load, merged by the standard hadd/permute 4-accumulator reduction.
template <>
inline void rm_gemv<double>(int M, int K, const double* a, const double* x, double* y) {
    int i = 0;
    for (; i + 4 <= M; i += 4) {
        const double* a0 = a + static_cast<size_t>(i) * K;
        const double* a1 = a0 + K;
        const double* a2 = a1 + K;
        const double* a3 = a2 + K;
        __m256d s0 = _mm256_setzero_pd();
        __m256d s1 = _mm256_setzero_pd();
        __m256d s2 = _mm256_setzero_pd();
        __m256d s3 = _mm256_setzero_pd();
        int k = 0;
        for (; k + 4 <= K; k += 4) {
            const __m256d xv = _mm256_loadu_pd(x + k);
            s0 = _mm256_fmadd_pd(_mm256_loadu_pd(a0 + k), xv, s0);
            s1 = _mm256_fmadd_pd(_mm256_loadu_pd(a1 + k), xv, s1);
            s2 = _mm256_fmadd_pd(_mm256_loadu_pd(a2 + k), xv, s2);
            s3 = _mm256_fmadd_pd(_mm256_loadu_pd(a3 + k), xv, s3);
        }
        const __m256d t01 = _mm256_hadd_pd(s0, s1);
        const __m256d t23 = _mm256_hadd_pd(s2, s3);
        __m256d sums = _mm256_add_pd(
            _mm256_permute2f128_pd(t01, t23, 0x21),
            _mm256_blend_pd(t01, t23, 0b1100));
        if (k < K) {
            double tails[4] = {0.0, 0.0, 0.0, 0.0};
            for (; k < K; ++k) {
                tails[0] += a0[k] * x[k];
                tails[1] += a1[k] * x[k];
                tails[2] += a2[k] * x[k];
                tails[3] += a3[k] * x[k];
            }
            sums = _mm256_add_pd(sums, _mm256_loadu_pd(tails));
        }
        _mm256_storeu_pd(y + i, _mm256_add_pd(_mm256_loadu_pd(y + i), sums));
    }
    for (; i < M; ++i) {
        const double* ar = a + static_cast<size_t>(i) * K;
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += ar[k] * x[k];
        }
        y[i] += sum;
    }
}

// SpMV adjoint, double: stream each contiguous A row scaled by x[i] into y.
template <>
inline void rm_gemv_adjoint<double>(int M, int K, const double* a, const double* x, double* y) {
    for (int i = 0; i < M; ++i) {
        const double* ar = a + static_cast<size_t>(i) * K;
        const __m256d xi = _mm256_set1_pd(x[i]);
        int k = 0;
        for (; k + 4 <= K; k += 4) {
            _mm256_storeu_pd(
                y + k, _mm256_fmadd_pd(_mm256_loadu_pd(ar + k), xi, _mm256_loadu_pd(y + k)));
        }
        for (; k < K; ++k) {
            y[k] += ar[k] * x[i];
        }
    }
}

#endif  // AVX2

}  // namespace rowmajor_kernels
}  // namespace vbcsr::detail

#endif  // VBCSR_DETAIL_KERNELS_ROWMAJOR_KERNELS_HPP

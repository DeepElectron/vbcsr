#ifndef VBCSR_DETAIL_KERNELS_CSR_APPLY_HPP
#define VBCSR_DETAIL_KERNELS_CSR_APPLY_HPP

#include "../../dist_multivector.hpp"
#include "../../dist_vector.hpp"
#include "../../scalar_traits.hpp"
#include "../backend/csr_backend.hpp"

#include <algorithm>
#include <vector>

namespace vbcsr::detail {

namespace {

// Row-major multivector (element (row, vec) at data[row * ld + vec]): the
// X row for a column and the Y row for the output are both contiguous.
template <typename T>
inline void csr_accumulate_dense_entry(
    T value,
    const T* x_data,
    int x_ld,
    int col_offset,
    int num_vecs,
    T* sums) {
    const T* x_row = x_data + static_cast<size_t>(col_offset) * x_ld;
    #pragma omp simd
    for (int vec = 0; vec < num_vecs; ++vec) {
        sums[vec] += value * x_row[vec];
    }
}

template <typename T>
inline void csr_write_dense_row(T* y_data, int y_ld, int row_offset, int num_vecs, const T* sums) {
    T* y_row = y_data + static_cast<size_t>(row_offset) * y_ld;
    for (int vec = 0; vec < num_vecs; ++vec) {
        y_row[vec] = sums[vec];
    }
}

// True when the vendor pages' row windows tile [0, n_rows) contiguously:
// a beta=0 forward apply then writes every owned row exactly once, so the
// serial pre-zero of Y can be skipped (only the ghost tail needs zeroing).
template <typename Cache>
bool csr_vendor_pages_tile_output_rows(const Cache& cache, int n_rows) {
    int previous_row_end = 0;
    for (const auto& page : cache.pages) {
        if (page.batch.row_begin != previous_row_end) {
            return false;
        }
        previous_row_end = page.batch.row_end;
    }
    return previous_row_end == n_rows;
}

template <typename T>
void csr_zero_ghost_tail(DistVector<T>& y) {
    if (y.ghost_size > 0) {
        std::fill(y.data.begin() + y.local_size, y.data.end(), T(0));
    }
}

template <typename T>
void csr_zero_ghost_tail(DistMultiVector<T>& y) {
    if (y.ghost_rows > 0) {
        // Row-major: ghost rows are the buffer tail (padding lanes stay zero
        // by the multivector invariant).
        std::fill(
            y.data.begin() + static_cast<size_t>(y.local_rows) * y.ld,
            y.data.end(),
            T(0));
    }
}

#ifdef VBCSR_HAVE_MKL_SPARSE
inline matrix_descr csr_mkl_descr() {
    matrix_descr descr{};
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;
    return descr;
}

template <typename T>
inline sparse_operation_t csr_mkl_operation(bool adjoint) {
    if (!adjoint) {
        return SPARSE_OPERATION_NON_TRANSPOSE;
    }
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    }
    return SPARSE_OPERATION_TRANSPOSE;
}

template <typename T>
bool csr_try_vendor_mkl_vector(
    const CSRMatrixBackend<T>& backend,
    const CSRVendorCache<T>& cache,
    DistVector<T>& x,
    DistVector<T>& y,
    bool adjoint,
    bool replace_output) {
    const matrix_descr descr = csr_mkl_descr();
    const sparse_operation_t op = csr_mkl_operation<T>(adjoint);

    size_t page_index = 0;
    for (const auto& page : cache.pages) {
        const auto& batch = page.batch;
        // Replace mode writes with beta=0 instead of pre-zero + beta=1
        // (skips two extra Y passes). Forward: pages tile disjoint Y row
        // windows, so every page replaces. Adjoint: every page-product
        // writes the full Y, so only the first page replaces and later
        // pages accumulate.
        const bool replace_this_page =
            replace_output && (!adjoint || page_index == 0);
        sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
        if constexpr (std::is_same_v<T, double>) {
            const double alpha = 1.0;
            const double beta = replace_this_page ? 0.0 : 1.0;
            // Each cached vendor handle is a compact CSR submatrix covering only
            // rows [row_begin, row_end). Forward apply writes into that Y window;
            // adjoint apply instead reads from that X window.
            const double* x_ptr = adjoint ? x.local_data() + batch.row_begin : x.local_data();
            double* y_ptr = adjoint ? y.local_data() : y.local_data() + batch.row_begin;
            status = mkl_sparse_d_mv(op, alpha, page.mkl.mv_handle, descr, x_ptr, beta, y_ptr);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            const MKL_Complex16 alpha{1.0, 0.0};
            const MKL_Complex16 beta{replace_this_page ? 0.0 : 1.0, 0.0};
            const auto* x_ptr = reinterpret_cast<const MKL_Complex16*>(
                adjoint ? x.local_data() + batch.row_begin : x.local_data());
            auto* y_ptr = reinterpret_cast<MKL_Complex16*>(
                adjoint ? y.local_data() : y.local_data() + batch.row_begin);
            status = mkl_sparse_z_mv(op, alpha, page.mkl.mv_handle, descr, x_ptr, beta, y_ptr);
        }

        if (status != SPARSE_STATUS_SUCCESS) {
            return false;
        }
        ++page_index;
    }

    backend.note_vendor_launch(static_cast<uint64_t>(cache.pages.size()));
    return true;
}

template <typename T>
bool csr_try_vendor_mkl_dense(
    const CSRMatrixBackend<T>& backend,
    const CSRVendorCache<T>& cache,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y,
    bool adjoint,
    bool replace_output) {
    const matrix_descr descr = csr_mkl_descr();
    const sparse_operation_t op = csr_mkl_operation<T>(adjoint);
    // Row-major dense operands: this is MKL's fast layout for CSR * dense
    // (measured 3.07x vs column-major on this workload, see the migration
    // plan evidence E1). Leading dims are the multivector's padded ld.
    const sparse_layout_t layout = SPARSE_LAYOUT_ROW_MAJOR;
    const int num_vecs = x.num_vectors;
    const int x_ld = x.ld;
    const int y_ld = y.ld;

    if (!backend.ensure_mkl_mm_handles(cache, num_vecs)) {
        return false;
    }

    size_t page_index = 0;
    for (const auto& page : cache.pages) {
        const auto& batch = page.batch;
        const sparse_matrix_t mm_handle = page.mkl.mm_handle(num_vecs);
        if (mm_handle == nullptr) {
            return false;
        }

        // Same replace-mode rule as the SpMV path above.
        const bool replace_this_page =
            replace_output && (!adjoint || page_index == 0);
        sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
        if constexpr (std::is_same_v<T, double>) {
            const double alpha = 1.0;
            const double beta = replace_this_page ? 0.0 : 1.0;
            // The dense case follows the same rule as SpMV: forward apply writes only
            // the rows owned by this page-local handle, while adjoint consumes only that
            // row window from X and accumulates into the full Y storage. Row
            // windows start at row_begin * ld in row-major storage.
            const double* b_ptr = adjoint
                ? x.data.data() + static_cast<size_t>(batch.row_begin) * x_ld
                : x.data.data();
            double* c_ptr = adjoint
                ? y.data.data()
                : y.data.data() + static_cast<size_t>(batch.row_begin) * y_ld;
            status = mkl_sparse_d_mm(
                op,
                alpha,
                mm_handle,
                descr,
                layout,
                b_ptr,
                static_cast<MKL_INT>(num_vecs),
                static_cast<MKL_INT>(x_ld),
                beta,
                c_ptr,
                static_cast<MKL_INT>(y_ld));
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            const MKL_Complex16 alpha{1.0, 0.0};
            const MKL_Complex16 beta{replace_this_page ? 0.0 : 1.0, 0.0};
            const auto* b_ptr = reinterpret_cast<const MKL_Complex16*>(
                adjoint ? x.data.data() + static_cast<size_t>(batch.row_begin) * x_ld
                        : x.data.data());
            auto* c_ptr = reinterpret_cast<MKL_Complex16*>(
                adjoint ? y.data.data()
                        : y.data.data() + static_cast<size_t>(batch.row_begin) * y_ld);
            status = mkl_sparse_z_mm(
                op,
                alpha,
                mm_handle,
                descr,
                layout,
                b_ptr,
                static_cast<MKL_INT>(num_vecs),
                static_cast<MKL_INT>(x_ld),
                beta,
                c_ptr,
                static_cast<MKL_INT>(y_ld));
        }

        if (status != SPARSE_STATUS_SUCCESS) {
            return false;
        }
        ++page_index;
    }

    backend.note_vendor_launch(static_cast<uint64_t>(cache.pages.size()));
    return true;
}
#endif

#ifdef VBCSR_HAVE_AOCL_SPARSE
template <typename T>
inline aoclsparse_operation csr_aocl_operation(bool adjoint) {
    if (!adjoint) {
        return aoclsparse_operation_none;
    }
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return aoclsparse_operation_conjugate_transpose;
    }
    return aoclsparse_operation_transpose;
}

template <typename T>
bool csr_try_vendor_aocl_mm(
    const CSRMatrixBackend<T>& backend,
    const CSRVendorCache<T>& cache,
    const T* x_ptr,
    int x_ld,
    T* y_ptr,
    int y_ld,
    int num_vecs,
    bool adjoint) {
    const aoclsparse_operation op = csr_aocl_operation<T>(adjoint);
    // Row-major dense operands; ld is the multivector's padded ld and row
    // windows start at row_begin * ld.
    const aoclsparse_order order = aoclsparse_order_row;

    for (const auto& page : cache.pages) {
        const auto& batch = page.batch;
        const size_t x_off = adjoint ? static_cast<size_t>(batch.row_begin) * x_ld : 0;
        const size_t y_off = adjoint ? 0 : static_cast<size_t>(batch.row_begin) * y_ld;
        aoclsparse_status status = aoclsparse_status_not_implemented;
        if constexpr (std::is_same_v<T, double>) {
            status = aoclsparse_dcsrmm(
                op,
                1.0,
                page.aocl.handle,
                cache.aocl_descr.handle,
                order,
                x_ptr + x_off,
                static_cast<aoclsparse_int>(num_vecs),
                static_cast<aoclsparse_int>(x_ld),
                1.0,
                y_ptr + y_off,
                static_cast<aoclsparse_int>(y_ld));
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            const aoclsparse_double_complex alpha{1.0, 0.0};
            const aoclsparse_double_complex beta{1.0, 0.0};
            status = aoclsparse_zcsrmm(
                op,
                alpha,
                page.aocl.handle,
                cache.aocl_descr.handle,
                order,
                reinterpret_cast<const aoclsparse_double_complex*>(x_ptr + x_off),
                static_cast<aoclsparse_int>(num_vecs),
                static_cast<aoclsparse_int>(x_ld),
                beta,
                reinterpret_cast<aoclsparse_double_complex*>(y_ptr + y_off),
                static_cast<aoclsparse_int>(y_ld));
        }

        if (status != aoclsparse_status_success) {
            return false;
        }
    }

    backend.note_vendor_launch(static_cast<uint64_t>(cache.pages.size()));
    return true;
}

template <typename T>
bool csr_try_vendor_aocl_vector(
    const CSRMatrixBackend<T>& backend,
    const CSRVendorCache<T>& cache,
    DistVector<T>& x,
    DistVector<T>& y,
    bool adjoint) {
    if constexpr (std::is_same_v<T, double>) {
        const aoclsparse_operation op = csr_aocl_operation<T>(adjoint);
        const double alpha = 1.0;
        const double beta = 1.0;

        for (const auto& page : cache.pages) {
            const auto& batch = page.batch;
            const aoclsparse_status status = aoclsparse_dcsrmv(
                op,
                &alpha,
                static_cast<aoclsparse_int>(batch.row_count()),
                static_cast<aoclsparse_int>(cache.num_cols),
                static_cast<aoclsparse_int>(batch.nnz_count),
                batch.values,
                batch.cols,
                reinterpret_cast<const aoclsparse_int*>(batch.row_offsets),
                cache.aocl_descr.handle,
                x.local_data() + (adjoint ? batch.row_begin : 0),
                &beta,
                y.local_data() + (adjoint ? 0 : batch.row_begin));
            if (status != aoclsparse_status_success) {
                return false;
            }
        }

        backend.note_vendor_launch(static_cast<uint64_t>(cache.pages.size()));
        return true;
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        // AOCL exposes no complex csrmv; its CSRMM path covers complex, so a
        // DistVector is applied as a 1-column row-major dense operand (ld=1,
        // row r at data[r]).
        return csr_try_vendor_aocl_mm(
            backend,
            cache,
            x.local_data(),
            1,
            y.local_data(),
            1,
            1,
            adjoint);
    } else {
        return false;
    }
}
#endif

template <typename T>
bool csr_mult_try_vendor_bound(
    DistGraph* graph,
    const CSRMatrixBackend<T>& backend,
    DistVector<T>& x,
    DistVector<T>& y) {
    const auto& cache = backend.ensure_vendor_cache(
        graph->adj_ptr,
        graph->adj_ind,
        static_cast<int>(graph->block_sizes.size()));
    if (cache.kind == CSRVendorBackendKind::None) {
        return false;
    }

    switch (cache.kind) {
    case CSRVendorBackendKind::MKL:
#ifdef VBCSR_HAVE_MKL_SPARSE
    {
        const int n_rows =
            graph->adj_ptr.empty() ? 0 : static_cast<int>(graph->adj_ptr.size()) - 1;
        const bool replace = csr_vendor_pages_tile_output_rows(cache, n_rows);
        if (replace) {
            csr_zero_ghost_tail(y);
        } else {
            std::fill(y.data.begin(), y.data.end(), T(0));
        }
        return csr_try_vendor_mkl_vector(backend, cache, x, y, false, replace);
    }
#endif
        break;
    case CSRVendorBackendKind::AOCL:
#ifdef VBCSR_HAVE_AOCL_SPARSE
        std::fill(y.data.begin(), y.data.end(), T(0));
        return csr_try_vendor_aocl_vector(backend, cache, x, y, false);
#endif
        break;
    case CSRVendorBackendKind::None:
    default:
        break;
    }

    return false;
}

template <typename T>
bool csr_mult_dense_try_vendor_bound(
    DistGraph* graph,
    const CSRMatrixBackend<T>& backend,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y) {
    const auto& cache = backend.ensure_vendor_cache(
        graph->adj_ptr,
        graph->adj_ind,
        static_cast<int>(graph->block_sizes.size()));
    if (cache.kind == CSRVendorBackendKind::None) {
        return false;
    }

    switch (cache.kind) {
    case CSRVendorBackendKind::MKL:
#ifdef VBCSR_HAVE_MKL_SPARSE
    {
        const int n_rows =
            graph->adj_ptr.empty() ? 0 : static_cast<int>(graph->adj_ptr.size()) - 1;
        const bool replace = csr_vendor_pages_tile_output_rows(cache, n_rows);
        if (replace) {
            csr_zero_ghost_tail(y);
        } else {
            std::fill(y.data.begin(), y.data.end(), T(0));
        }
        return csr_try_vendor_mkl_dense(backend, cache, x, y, false, replace);
    }
#endif
        break;
    case CSRVendorBackendKind::AOCL:
#ifdef VBCSR_HAVE_AOCL_SPARSE
        std::fill(y.data.begin(), y.data.end(), T(0));
        return csr_try_vendor_aocl_mm(
            backend,
            cache,
            x.data.data(),
            x.ld,
            y.data.data(),
            y.ld,
            x.num_vectors,
            false);
#endif
        break;
    case CSRVendorBackendKind::None:
    default:
        break;
    }

    return false;
}

template <typename T>
bool csr_mult_adjoint_try_vendor_bound(
    DistGraph* graph,
    const CSRMatrixBackend<T>& backend,
    DistVector<T>& x,
    DistVector<T>& y) {
    const auto& cache = backend.ensure_vendor_cache(
        graph->adj_ptr,
        graph->adj_ind,
        static_cast<int>(graph->block_sizes.size()));
    if (cache.kind == CSRVendorBackendKind::None) {
        return false;
    }

    bool ok = false;
    switch (cache.kind) {
    case CSRVendorBackendKind::MKL:
#ifdef VBCSR_HAVE_MKL_SPARSE
    {
        // Adjoint page-products each write the full Y (all owned + ghost
        // columns), so the first page runs with beta=0 and no pre-zero is
        // needed at all. An empty page list leaves Y untouched — zero it.
        const bool replace = !cache.pages.empty();
        if (!replace) {
            std::fill(y.data.begin(), y.data.end(), T(0));
        }
        ok = csr_try_vendor_mkl_vector(backend, cache, x, y, true, replace);
    }
#endif
        break;
    case CSRVendorBackendKind::AOCL:
#ifdef VBCSR_HAVE_AOCL_SPARSE
        std::fill(y.data.begin(), y.data.end(), T(0));
        ok = csr_try_vendor_aocl_vector(backend, cache, x, y, true);
#endif
        break;
    case CSRVendorBackendKind::None:
    default:
        ok = false;
        break;
    }

    if (ok) {
        y.reduce_ghosts();
    }
    return ok;
}

template <typename T>
bool csr_mult_dense_adjoint_try_vendor_bound(
    DistGraph* graph,
    const CSRMatrixBackend<T>& backend,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y) {
    const auto& cache = backend.ensure_vendor_cache(
        graph->adj_ptr,
        graph->adj_ind,
        static_cast<int>(graph->block_sizes.size()));
    if (cache.kind == CSRVendorBackendKind::None) {
        return false;
    }

    bool ok = false;
    switch (cache.kind) {
    case CSRVendorBackendKind::MKL:
#ifdef VBCSR_HAVE_MKL_SPARSE
    {
        // Same first-page beta=0 rule as the adjoint SpMV path above.
        const bool replace = !cache.pages.empty();
        if (!replace) {
            std::fill(y.data.begin(), y.data.end(), T(0));
        }
        ok = csr_try_vendor_mkl_dense(backend, cache, x, y, true, replace);
    }
#endif
        break;
    case CSRVendorBackendKind::AOCL:
#ifdef VBCSR_HAVE_AOCL_SPARSE
        std::fill(y.data.begin(), y.data.end(), T(0));
        ok = csr_try_vendor_aocl_mm(
            backend,
            cache,
            x.data.data(),
            x.ld,
            y.data.data(),
            y.ld,
            x.num_vectors,
            true);
#endif
        break;
    case CSRVendorBackendKind::None:
    default:
        ok = false;
        break;
    }

    if (ok) {
        y.reduce_ghosts();
    }
    return ok;
}

template <typename T>
void csr_mult_native(DistGraph* graph, const CSRMatrixBackend<T>& backend, DistVector<T>& x, DistVector<T>& y) {
    const auto& row_ptr = graph->adj_ptr;
    const int n_rows = row_ptr.empty() ? 0 : static_cast<int>(row_ptr.size()) - 1;
    const int* block_offsets = graph->block_offsets.data();
    const T* x_data = x.local_data();
    T* y_data = y.local_data();

    // Rows split along the backend's stored nnz-weighted partition (falling
    // back to an even split): the same boundaries the value pages were
    // first-touched with, so each thread reads node-local pages.
    #pragma omp parallel
    {
        const auto [row_begin, row_end] =
            thread_domain_range(n_rows, backend.thread_domains);
        for (int row = row_begin; row < row_end; ++row) {
            T sum = T(0);
            backend.for_each_row_slice(row_ptr, graph->adj_ind, row, [&](auto slice) {
                for (uint32_t idx = 0; idx < slice.nnz_count; ++idx) {
                    const int col = slice.cols[idx];
                    sum += slice.values[idx] * x_data[block_offsets[col]];
                }
            });
            y_data[block_offsets[row]] = sum;
        }
    }
}

template <typename T>
void csr_mult_dense_native(
    DistGraph* graph,
    const CSRMatrixBackend<T>& backend,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y) {
    const auto& row_ptr = graph->adj_ptr;
    const int n_rows = row_ptr.empty() ? 0 : static_cast<int>(row_ptr.size()) - 1;
    const int num_vecs = x.num_vectors;
    const int x_ld = x.ld;
    const int y_ld = y.ld;
    const int* block_offsets = graph->block_offsets.data();
    const T* x_data = x.data.data();
    T* y_data = y.data.data();

    #pragma omp parallel
    {
        std::vector<T> sums(static_cast<size_t>(num_vecs), T(0));

        // Same stored-partition split as csr_mult_native above.
        const auto [row_begin, row_end] =
            thread_domain_range(n_rows, backend.thread_domains);
        for (int row = row_begin; row < row_end; ++row) {
            std::fill(sums.begin(), sums.end(), T(0));

            backend.for_each_row_slice(row_ptr, graph->adj_ind, row, [&](auto slice) {
                for (uint32_t idx = 0; idx < slice.nnz_count; ++idx) {
                    const int col = slice.cols[idx];
                    const int col_offset = block_offsets[col];
                    csr_accumulate_dense_entry(slice.values[idx], x_data, x_ld, col_offset, num_vecs, sums.data());
                }
            });

            csr_write_dense_row(y_data, y_ld, block_offsets[row], num_vecs, sums.data());
        }
    }
}

template <typename T>
void csr_mult_adjoint_native(
    DistGraph* graph,
    const CSRMatrixBackend<T>& backend,
    DistVector<T>& x,
    DistVector<T>& y) {
    std::fill(y.data.begin(), y.data.end(), T(0));

    const auto& row_ptr = graph->adj_ptr;
    const int n_rows = row_ptr.empty() ? 0 : static_cast<int>(row_ptr.size()) - 1;

    #pragma omp parallel
    {
        std::vector<T> y_local(y.data.size(), T(0));

        #pragma omp for schedule(static)
        for (int row = 0; row < n_rows; ++row) {
            const T x_value = x.data[graph->block_offsets[row]];
            backend.for_each_row_slice(row_ptr, graph->adj_ind, row, [&](auto slice) {
                for (uint32_t idx = 0; idx < slice.nnz_count; ++idx) {
                    const int col = slice.cols[idx];
                    const int col_offset = graph->block_offsets[col];
                    y_local[static_cast<size_t>(col_offset)] += ScalarTraits<T>::conjugate(slice.values[idx]) * x_value;
                }
            });
        }

        #pragma omp critical
        {
            for (size_t idx = 0; idx < y.data.size(); ++idx) {
                y.data[idx] += y_local[idx];
            }
        }
    }

    y.reduce_ghosts();
}

template <typename T>
void csr_mult_dense_adjoint_native(
    DistGraph* graph,
    const CSRMatrixBackend<T>& backend,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y) {
    std::fill(y.data.begin(), y.data.end(), T(0));

    const auto& row_ptr = graph->adj_ptr;
    const int n_rows = row_ptr.empty() ? 0 : static_cast<int>(row_ptr.size()) - 1;
    const int num_vecs = x.num_vectors;
    const int x_ld = x.ld;
    const int y_ld = y.ld;

    #pragma omp parallel
    {
        std::vector<T> y_local(y.data.size(), T(0));

        #pragma omp for schedule(static)
        for (int row = 0; row < n_rows; ++row) {
            const int row_offset = graph->block_offsets[row];
            backend.for_each_row_slice(row_ptr, graph->adj_ind, row, [&](auto slice) {
                for (uint32_t idx = 0; idx < slice.nnz_count; ++idx) {
                    const int col = slice.cols[idx];
                    const int col_offset = graph->block_offsets[col];
                    const T value = ScalarTraits<T>::conjugate(slice.values[idx]);
                    // Row-major: both the X source row and the Y target row
                    // are contiguous lanes.
                    T* y_row = y_local.data() + static_cast<size_t>(col_offset) * y_ld;
                    const T* x_row = x.data.data() + static_cast<size_t>(row_offset) * x_ld;
                    #pragma omp simd
                    for (int vec = 0; vec < num_vecs; ++vec) {
                        y_row[vec] += value * x_row[vec];
                    }
                }
            });
        }

        #pragma omp critical
        {
            for (size_t idx = 0; idx < y.data.size(); ++idx) {
                y.data[idx] += y_local[idx];
            }
        }
    }

    y.reduce_ghosts();
}

} // namespace


// outer interfaces
template <typename T>
void csr_mult(DistGraph* graph, const CSRMatrixBackend<T>& backend, DistVector<T>& x, DistVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);
    x.sync_ghosts();
    BLASKernel::configure_vendor_sparse_threading();
    if (!csr_mult_try_vendor_bound(graph, backend, x, y)) {
        BLASKernel::configure_native_threading();
        csr_mult_native(graph, backend, x, y);
    }
}

template <typename T>
void csr_mult_dense(DistGraph* graph, const CSRMatrixBackend<T>& backend, DistMultiVector<T>& x, DistMultiVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);
    x.sync_ghosts();
    BLASKernel::configure_vendor_sparse_threading();
    if (!csr_mult_dense_try_vendor_bound(graph, backend, x, y)) {
        BLASKernel::configure_native_threading();
        csr_mult_dense_native(graph, backend, x, y);
    }
}

template <typename T>
void csr_mult_adjoint(DistGraph* graph, const CSRMatrixBackend<T>& backend, DistVector<T>& x, DistVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);
    BLASKernel::configure_vendor_sparse_threading();
    if (!csr_mult_adjoint_try_vendor_bound(graph, backend, x, y)) {
        BLASKernel::configure_native_threading();
        csr_mult_adjoint_native(graph, backend, x, y);
    }
}

template <typename T>
void csr_mult_dense_adjoint(
    DistGraph* graph,
    const CSRMatrixBackend<T>& backend,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);
    BLASKernel::configure_vendor_sparse_threading();
    if (!csr_mult_dense_adjoint_try_vendor_bound(graph, backend, x, y)) {
        BLASKernel::configure_native_threading();
        csr_mult_dense_adjoint_native(graph, backend, x, y);
    }
}

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_KERNELS_CSR_APPLY_HPP

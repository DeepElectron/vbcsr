#ifndef VBCSR_DETAIL_KERNELS_BSR_APPLY_HPP
#define VBCSR_DETAIL_KERNELS_BSR_APPLY_HPP

#include "../../dist_multivector.hpp"
#include "../../dist_vector.hpp"
#include "dense_kernels.hpp"
#include "rowmajor_kernels.hpp"
#include "../backend/bsr_backend.hpp"

#include <algorithm>
#include <cstdlib>
#include <vector>

namespace vbcsr::detail {

// BSR applies (SpMV and dense) route to the native row-major kernels by
// default: with the Phase-4 canonical row-major blocks, the native paths
// measured faster than MKL's BSR kernels on the reference EPYC 7352 setup —
// dense mm 1.65x at 1 thread (~1.03x at 16); mv 1.9x at 1 thread (MKL's mv
// slows ~17% on row-major blocks, while the native contiguous-row dot kernel
// beats even MKL's column-major-block figure). See the migration plan Phase 4
// record. Set VBCSR_BSR_VENDOR=1 (legacy alias: VBCSR_BSR_DENSE_VENDOR) to
// route through the vendor paths instead (re-measure hook; also exercised by
// test_pb_csr's vendor-cache tests).
inline bool bsr_vendor_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("VBCSR_BSR_VENDOR");
        if (value == nullptr) {
            value = std::getenv("VBCSR_BSR_DENSE_VENDOR");
        }
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
namespace {

inline matrix_descr bsr_mkl_descr() {
    matrix_descr descr{};
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;
    return descr;
}

template <typename T>
inline sparse_operation_t bsr_mkl_operation(bool adjoint) {
    if (!adjoint) {
        return SPARSE_OPERATION_NON_TRANSPOSE;
    }
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    }
    return SPARSE_OPERATION_TRANSPOSE;
}

template <typename T>
bool bsr_try_vendor_mkl_vector(
    DistGraph* graph,
    const BSRMatrixBackend<T>& backend,
    const BSRVendorCache<T>& cache,
    DistVector<T>& x,
    DistVector<T>& y,
    bool adjoint) {
    const matrix_descr descr = bsr_mkl_descr();
    const sparse_operation_t op = bsr_mkl_operation<T>(adjoint);

    for (const auto& entry : cache.batches) {
        const size_t row_offset =
            static_cast<size_t>(graph->block_offsets[entry.batch.row_begin]);
        sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
        if constexpr (std::is_same_v<T, double>) {
            const double alpha = 1.0;
            const double beta = 1.0;
            const double* x_ptr = adjoint ? x.local_data() + row_offset : x.local_data();
            double* y_ptr = adjoint ? y.local_data() : y.local_data() + row_offset;
            status = mkl_sparse_d_mv(op, alpha, entry.mkl.mv_handle, descr, x_ptr, beta, y_ptr);
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            const MKL_Complex16 alpha{1.0, 0.0};
            const MKL_Complex16 beta{1.0, 0.0};
            const auto* x_ptr = reinterpret_cast<const MKL_Complex16*>(
                adjoint ? x.local_data() + row_offset : x.local_data());
            auto* y_ptr = reinterpret_cast<MKL_Complex16*>(
                adjoint ? y.local_data() : y.local_data() + row_offset);
            status = mkl_sparse_z_mv(op, alpha, entry.mkl.mv_handle, descr, x_ptr, beta, y_ptr);
        }

        if (status != SPARSE_STATUS_SUCCESS) {
            return false;
        }
    }

    backend.note_vendor_launch(static_cast<uint64_t>(cache.batches.size()));
    return true;
}

template <typename T>
bool bsr_try_vendor_mkl_dense(
    DistGraph* graph,
    const BSRMatrixBackend<T>& backend,
    const BSRVendorCache<T>& cache,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y,
    bool adjoint,
    bool replace_output) {
    const matrix_descr descr = bsr_mkl_descr();
    // Dense operands are ROW major (the multivector layout, padded ld). The
    // BSR block layout inside the vendor handles is also row major (canonical
    // since Phase 4), so block and operand layouts match — the pairing MKL's
    // BSR mm supports with zero-based indexing (see build_bsr_mkl_mm_variant).
    const sparse_layout_t layout = SPARSE_LAYOUT_ROW_MAJOR;
    const int num_vecs = x.num_vectors;
    const int x_ld = x.ld;
    const int y_ld = y.ld;

    const auto run_dense_adjoint_via_mv = [&]() -> bool {
        // The row-major multivector has no contiguous columns, so each vector
        // is staged through temporary contiguous buffers for the mv kernel.
        const sparse_operation_t op = bsr_mkl_operation<T>(true);
        const int x_rows = x.local_rows + x.ghost_rows;
        const int y_rows = y.local_rows + y.ghost_rows;
        std::vector<T> x_col(static_cast<size_t>(x_rows));
        std::vector<T> y_col(static_cast<size_t>(y_rows));
        for (int vec = 0; vec < num_vecs; ++vec) {
            for (int row = 0; row < x_rows; ++row) {
                x_col[row] = x(row, vec);
            }
            for (int row = 0; row < y_rows; ++row) {
                y_col[row] = y(row, vec);
            }
            for (const auto& entry : cache.batches) {
                const size_t row_offset =
                    static_cast<size_t>(graph->block_offsets[entry.batch.row_begin]);
                sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
                if constexpr (std::is_same_v<T, double>) {
                    status = mkl_sparse_d_mv(
                        op, 1.0, entry.mkl.mv_handle, descr,
                        x_col.data() + row_offset, 1.0, y_col.data());
                } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                    const MKL_Complex16 alpha{1.0, 0.0};
                    const MKL_Complex16 beta{1.0, 0.0};
                    status = mkl_sparse_z_mv(
                        op, alpha, entry.mkl.mv_handle, descr,
                        reinterpret_cast<const MKL_Complex16*>(x_col.data() + row_offset),
                        beta,
                        reinterpret_cast<MKL_Complex16*>(y_col.data()));
                }
                if (status != SPARSE_STATUS_SUCCESS) {
                    return false;
                }
            }
            for (int row = 0; row < y_rows; ++row) {
                y(row, vec) = y_col[row];
            }
        }

        backend.note_vendor_launch(
            static_cast<uint64_t>(cache.batches.size()) * static_cast<uint64_t>(num_vecs));
        return true;
    };

    if (adjoint) {
        if (!backend.ensure_mkl_mm_handles(cache, num_vecs)) {
            return run_dense_adjoint_via_mv();
        }

        const sparse_operation_t op = bsr_mkl_operation<T>(true);
        for (const auto& entry : cache.batches) {
            const sparse_matrix_t mm_handle = entry.mkl.mm_handle(num_vecs);
            if (mm_handle == nullptr) {
                return run_dense_adjoint_via_mv();
            }

            const size_t row_offset =
                static_cast<size_t>(graph->block_offsets[entry.batch.row_begin]);
            sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
            if constexpr (std::is_same_v<T, double>) {
                const double alpha = 1.0;
                const double beta = 1.0;
                const double* b_ptr = x.data.data() + row_offset * x_ld;
                double* c_ptr = y.data.data();
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
                const MKL_Complex16 beta{1.0, 0.0};
                const auto* b_ptr = reinterpret_cast<const MKL_Complex16*>(
                    x.data.data() + row_offset * x_ld);
                auto* c_ptr = reinterpret_cast<MKL_Complex16*>(y.data.data());
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
                return run_dense_adjoint_via_mv();
            }
        }

        backend.note_vendor_launch(static_cast<uint64_t>(cache.batches.size()));
        return true;
    }

    if (!backend.ensure_mkl_mm_handles(cache, num_vecs)) {
        return false;
    }

    for (const auto& entry : cache.batches) {
        const sparse_matrix_t mm_handle = entry.mkl.mm_handle(num_vecs);
        if (mm_handle == nullptr) {
            return false;
        }

        const size_t row_offset =
            static_cast<size_t>(graph->block_offsets[entry.batch.row_begin]);
        sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
        if constexpr (std::is_same_v<T, double>) {
            const double alpha = 1.0;
            const double beta = replace_output ? 0.0 : 1.0;
            const double* b_ptr = x.data.data();
            double* c_ptr = y.data.data() + row_offset * y_ld;
            status = mkl_sparse_d_mm(
                SPARSE_OPERATION_NON_TRANSPOSE,
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
            const MKL_Complex16 beta{replace_output ? 0.0 : 1.0, 0.0};
            const auto* b_ptr =
                reinterpret_cast<const MKL_Complex16*>(x.data.data());
            auto* c_ptr = reinterpret_cast<MKL_Complex16*>(y.data.data() + row_offset * y_ld);
            status = mkl_sparse_z_mm(
                SPARSE_OPERATION_NON_TRANSPOSE,
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
    }

    backend.note_vendor_launch(static_cast<uint64_t>(cache.batches.size()));
    return true;
}

template <typename T>
bool bsr_vendor_batches_have_disjoint_output_rows(const BSRVendorCache<T>& cache) {
    int previous_row_end = 0;
    for (const auto& entry : cache.batches) {
        if (entry.batch.row_begin < previous_row_end) {
            return false;
        }
        previous_row_end = entry.batch.row_end;
    }
    return true;
}

template <typename T>
void bsr_zero_output_ghost_rows(DistMultiVector<T>& y) {
    if (y.ghost_rows <= 0) {
        return;
    }
    // Row-major: ghost rows are the buffer tail — one contiguous fill
    // (padding lanes are zero before and after).
    std::fill(
        y.data.begin() + static_cast<size_t>(y.local_rows) * y.ld, y.data.end(), T(0));
}

} // namespace
#endif

// Fixed-block-size lambda dispatch. The apply impls below take runtime dims,
// so they no longer use this; it remains for consumers that instantiate
// genuinely block-size-templated kernels (detail/ops/spmm/bsr.hpp).
template <typename Fn>
decltype(auto) bsr_dispatch_block_size(int block_size, Fn&& fn) {
    switch (block_size) {
        case 2:
            return fn(std::integral_constant<int, 2>{});
        case 4:
            return fn(std::integral_constant<int, 4>{});
        case 8:
            return fn(std::integral_constant<int, 8>{});
        case 16:
            return fn(std::integral_constant<int, 16>{});
        default:
            return fn(std::integral_constant<int, 0>{});
    }
}

// Per-thread row range: the backend's stored work-balanced partition when it
// matches the live team (one source of truth with storage first-touch
// placement), else an even row split. See thread_domain_range.
inline std::pair<int, int> bsr_thread_row_range(
    int n_rows,
    const ThreadDomainPartition& domains) {
    return thread_domain_range(n_rows, domains);
}

// SpMV impls keep the compile-time BlockSize dispatch: rm_gemv's k-loops
// fully unroll when bs is a constant (measured ~8% on bs=8 SpMV), unlike the
// dense impls where the nv-axis loops dominate and runtime dims are neutral.
template <int BlockSize, typename T>
void bsr_mult_impl(DistGraph* graph, const BSRMatrixBackend<T>& backend, DistVector<T>& x, DistVector<T>& y) {
    const int bs = BlockSize == 0 ? backend.block_size : BlockSize;
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);
    x.sync_ghosts();

    const auto& plan = backend.ensure_apply_plan(graph->adj_ptr, graph->adj_ind);
    const auto& domains = backend.ensure_thread_domains(graph->adj_ptr);
    const int n_rows = graph->adj_ptr.empty() ? 0 : static_cast<int>(graph->adj_ptr.size()) - 1;

    #pragma omp parallel
    {
        const auto [thread_row_begin, thread_row_end] = bsr_thread_row_range(n_rows, domains);
#ifdef _OPENMP
        const bool last_thread = omp_get_thread_num() == omp_get_num_threads() - 1;
#else
        const bool last_thread = true;
#endif
        // Zero this thread's Y row range in-region (parallel fill, NUMA-local
        // first touch); the last thread also clears the ghost tail. No
        // barrier: each thread accumulates only into its own rows.
        std::fill(
            y.data.data() + graph->block_offsets[thread_row_begin],
            y.data.data() + graph->block_offsets[thread_row_end],
            T(0));
        if (last_thread) {
            std::fill(
                y.data.data() + graph->block_offsets[n_rows], y.data.data() + y.data.size(), T(0));
        }
        for (const auto& batch_entry : plan.batches) {
            const auto& batch = batch_entry.batch;
            const int row_begin = std::max(batch.row_begin, thread_row_begin);
            const int row_end = std::min(batch.row_end, thread_row_end);
            for (int row = row_begin; row < row_end; ++row) {
                T* y_block = y.local_data() + graph->block_offsets[row];
                const uint32_t block_begin = batch.row_block_start(row);
                const uint32_t block_end = batch.row_block_end(row);
                for (uint32_t local_block = block_begin; local_block < block_end; ++local_block) {
                    const int col = batch.cols[local_block];
                    const T* block = batch.block_ptr(local_block);
                    const T* x_block = x.data.data() + graph->block_offsets[col];
                    rowmajor_kernels::rm_gemv<T>(bs, bs, block, x_block, y_block);
                }
            }
        }
    }
}

template <typename T>
void bsr_mult(DistGraph* graph, const BSRMatrixBackend<T>& backend, DistVector<T>& x, DistVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);
    x.sync_ghosts();

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
    // Vendor path is opt-in (VBCSR_BSR_VENDOR): only then align the MKL pool
    // with the OpenMP budget and build vendor handles.
    if (bsr_vendor_enabled()) {
        BLASKernel::configure_vendor_sparse_threading();
        const auto& cache = backend.ensure_vendor_cache(
            graph->adj_ptr,
            graph->adj_ind,
            static_cast<int>(graph->block_sizes.size()));
        if (cache.kind == BSRVendorBackendKind::MKL) {
            std::fill(y.data.begin(), y.data.end(), T(0));
            if (bsr_try_vendor_mkl_vector(graph, backend, cache, x, y, false)) {
                return;
            }
        }
    }
#endif

    BLASKernel::configure_native_threading();
    bsr_dispatch_block_size(backend.block_size, [&](auto block_tag) {
        constexpr int kBlockSize = decltype(block_tag)::value;
        bsr_mult_impl<kBlockSize>(graph, backend, x, y);
    });
}

template <typename T>
void bsr_mult_dense_impl(DistGraph* graph, const BSRMatrixBackend<T>& backend, DistMultiVector<T>& x, DistMultiVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);
    x.sync_ghosts();

    const auto& plan = backend.ensure_apply_plan(graph->adj_ptr, graph->adj_ind);
    const auto& domains = backend.ensure_thread_domains(graph->adj_ptr);
    const int n_rows = graph->adj_ptr.empty() ? 0 : static_cast<int>(graph->adj_ptr.size()) - 1;
    const int runtime_block_size = backend.block_size;
    const int nv = x.ld;  // padded; pad lanes are zero (invariant)
    const int x_ld = x.ld;
    const int y_ld = y.ld;

    #pragma omp parallel
    {
        const auto [thread_row_begin, thread_row_end] = bsr_thread_row_range(n_rows, domains);
#ifdef _OPENMP
        const bool last_thread = omp_get_thread_num() == omp_get_num_threads() - 1;
#else
        const bool last_thread = true;
#endif
        // Same in-region per-thread zeroing as the SpMV impl above (row
        // offsets scaled by the padded ld; full-ld rows keep the padding
        // lanes zero).
        std::fill(
            y.data.data() + static_cast<size_t>(graph->block_offsets[thread_row_begin]) * y_ld,
            y.data.data() + static_cast<size_t>(graph->block_offsets[thread_row_end]) * y_ld,
            T(0));
        if (last_thread) {
            std::fill(
                y.data.data() + static_cast<size_t>(graph->block_offsets[n_rows]) * y_ld,
                y.data.data() + y.data.size(),
                T(0));
        }
        for (const auto& batch_entry : plan.batches) {
            const auto& batch = batch_entry.batch;
            const int row_begin = std::max(batch.row_begin, thread_row_begin);
            const int row_end = std::min(batch.row_end, thread_row_end);
            for (int row = row_begin; row < row_end; ++row) {
                T* y_rows = y.data.data() +
                    static_cast<size_t>(graph->block_offsets[row]) * y_ld;
                const uint32_t block_begin = batch.row_block_start(row);
                const uint32_t block_end = batch.row_block_end(row);
                for (uint32_t local_block = block_begin; local_block < block_end; ++local_block) {
                    const int col = batch.cols[local_block];
                    rowmajor_kernels::rm_gemm<T>(
                        runtime_block_size,
                        runtime_block_size,
                        nv,
                        batch.block_ptr(local_block),
                        x.data.data() + static_cast<size_t>(graph->block_offsets[col]) * x_ld,
                        x_ld,
                        y_rows,
                        y_ld);
                }
            }
        }
    }
}

template <typename T>
void bsr_mult_dense(DistGraph* graph, const BSRMatrixBackend<T>& backend, DistMultiVector<T>& x, DistMultiVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);
    x.sync_ghosts();

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
    // Vendor path is opt-in (VBCSR_BSR_VENDOR): only then align the MKL pool
    // with the OpenMP budget and build vendor handles.
    if (bsr_vendor_enabled()) {
        BLASKernel::configure_vendor_sparse_threading();
        const auto& cache = backend.ensure_vendor_cache(
            graph->adj_ptr,
            graph->adj_ind,
            static_cast<int>(graph->block_sizes.size()));
        if (cache.kind == BSRVendorBackendKind::MKL) {
            const bool replace_output = bsr_vendor_batches_have_disjoint_output_rows(cache);
            if (!replace_output) {
                std::fill(y.data.begin(), y.data.end(), T(0));
            } else {
                bsr_zero_output_ghost_rows(y);
            }
            if (bsr_try_vendor_mkl_dense(graph, backend, cache, x, y, false, replace_output)) {
                return;
            }
        }
    }
#endif

    BLASKernel::configure_native_threading();
    bsr_mult_dense_impl(graph, backend, x, y);
}

template <int BlockSize, typename T>
void bsr_mult_adjoint_impl(DistGraph* graph, const BSRMatrixBackend<T>& backend, DistVector<T>& x, DistVector<T>& y) {
    const int bs = BlockSize == 0 ? backend.block_size : BlockSize;
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);

    const auto& plan = backend.ensure_apply_plan(graph->adj_ptr, graph->adj_ind);
    const auto& domains = backend.ensure_thread_domains(graph->adj_ptr);
    const int n_rows = graph->adj_ptr.empty() ? 0 : static_cast<int>(graph->adj_ptr.size()) - 1;
    const int thread_count =
#ifdef _OPENMP
        std::max(1, omp_get_max_threads());
#else
        1;
#endif

    if (thread_count == 1) {
        // Single thread: accumulate straight into y, no scatter buffers.
        std::fill(y.data.begin(), y.data.end(), T(0));
        for (const auto& batch_entry : plan.batches) {
            const auto& batch = batch_entry.batch;
            for (int row = batch.row_begin; row < batch.row_end; ++row) {
                const T* x_block = x.local_data() + graph->block_offsets[row];
                const uint32_t block_begin = batch.row_block_start(row);
                const uint32_t block_end = batch.row_block_end(row);
                for (uint32_t local_block = block_begin; local_block < block_end; ++local_block) {
                    const int col = batch.cols[local_block];
                    rowmajor_kernels::rm_gemv_adjoint<T>(
                        bs,
                        bs,
                        batch.block_ptr(local_block),
                        x_block,
                        y.data.data() + graph->block_offsets[col]);
                }
            }
        }
        y.reduce_ghosts();
        return;
    }

    // Buffers are sized inside the parallel region: each thread zero-fills
    // its own scatter buffer (parallel fill, NUMA-local first touch) instead
    // of the calling thread serially zeroing thread_count full-Y copies.
    std::vector<std::vector<T>> thread_buffers(static_cast<size_t>(thread_count));

    #pragma omp parallel
    {
#ifdef _OPENMP
        const int thread_id = omp_get_thread_num();
#else
        const int thread_id = 0;
#endif
        auto& y_local = thread_buffers[static_cast<size_t>(thread_id)];
        y_local.assign(y.data.size(), T(0));
        const auto [thread_row_begin, thread_row_end] = bsr_thread_row_range(n_rows, domains);

        for (const auto& batch_entry : plan.batches) {
            const auto& batch = batch_entry.batch;
            const int row_begin = std::max(batch.row_begin, thread_row_begin);
            const int row_end = std::min(batch.row_end, thread_row_end);
            for (int row = row_begin; row < row_end; ++row) {
                const T* x_block = x.local_data() + graph->block_offsets[row];
                const uint32_t block_begin = batch.row_block_start(row);
                const uint32_t block_end = batch.row_block_end(row);
                for (uint32_t local_block = block_begin; local_block < block_end; ++local_block) {
                    const int col = batch.cols[local_block];
                    const T* block = batch.block_ptr(local_block);
                    T* y_block = y_local.data() + graph->block_offsets[col];
                    rowmajor_kernels::rm_gemv_adjoint<T>(bs, bs, block, x_block, y_block);
                }
            }
        }
    }

    #pragma omp parallel for
    for (size_t index = 0; index < y.data.size(); ++index) {
        T sum = T(0);
        for (const auto& thread_buffer : thread_buffers) {
            // A buffer stays empty when the region ran with fewer threads
            // than thread_count.
            if (!thread_buffer.empty()) {
                sum += thread_buffer[index];
            }
        }
        y.data[index] = sum;
    }

    y.reduce_ghosts();
}

template <typename T>
void bsr_mult_adjoint(DistGraph* graph, const BSRMatrixBackend<T>& backend, DistVector<T>& x, DistVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
    // Vendor path is opt-in (VBCSR_BSR_VENDOR): only then align the MKL pool
    // with the OpenMP budget and build vendor handles.
    if (bsr_vendor_enabled()) {
        BLASKernel::configure_vendor_sparse_threading();
        const auto& cache = backend.ensure_vendor_cache(
            graph->adj_ptr,
            graph->adj_ind,
            static_cast<int>(graph->block_sizes.size()));
        if (cache.kind == BSRVendorBackendKind::MKL) {
            std::fill(y.data.begin(), y.data.end(), T(0));
            if (bsr_try_vendor_mkl_vector(graph, backend, cache, x, y, true)) {
                y.reduce_ghosts();
                return;
            }
        }
    }
#endif

    BLASKernel::configure_native_threading();
    bsr_dispatch_block_size(backend.block_size, [&](auto block_tag) {
        constexpr int kBlockSize = decltype(block_tag)::value;
        bsr_mult_adjoint_impl<kBlockSize>(graph, backend, x, y);
    });
}

template <typename T>
void bsr_mult_dense_adjoint_impl(
    DistGraph* graph,
    const BSRMatrixBackend<T>& backend,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);

    const auto& plan = backend.ensure_apply_plan(graph->adj_ptr, graph->adj_ind);
    const auto& domains = backend.ensure_thread_domains(graph->adj_ptr);
    const int n_rows = graph->adj_ptr.empty() ? 0 : static_cast<int>(graph->adj_ptr.size()) - 1;
    const int runtime_block_size = backend.block_size;
    const int nv = x.ld;
    const int x_ld = x.ld;
    const int y_ld = y.ld;
    const int thread_count =
#ifdef _OPENMP
        std::max(1, omp_get_max_threads());
#else
        1;
#endif

    if (thread_count == 1) {
        // Single thread owns every output row: accumulate straight into y and
        // skip the scatter buffers + merge (two full passes over y otherwise).
        std::fill(y.data.begin(), y.data.end(), T(0));
        for (const auto& batch_entry : plan.batches) {
            const auto& batch = batch_entry.batch;
            for (int row = batch.row_begin; row < batch.row_end; ++row) {
                const T* x_rows = x.data.data() +
                    static_cast<size_t>(graph->block_offsets[row]) * x_ld;
                const uint32_t block_begin = batch.row_block_start(row);
                const uint32_t block_end = batch.row_block_end(row);
                for (uint32_t local_block = block_begin; local_block < block_end; ++local_block) {
                    const int col = batch.cols[local_block];
                    rowmajor_kernels::rm_gemm_adjoint<T>(
                        runtime_block_size,
                        runtime_block_size,
                        nv,
                        batch.block_ptr(local_block),
                        x_rows,
                        x_ld,
                        y.data.data() + static_cast<size_t>(graph->block_offsets[col]) * y_ld,
                        y_ld);
                }
            }
        }
        y.reduce_ghosts();
        return;
    }

    // Row-driven scatter: per-thread ROW-major accumulation buffers (same
    // layout as y), merged with one flat contiguous reduction. Each thread
    // sizes its own buffer in-region (parallel fill, NUMA-local first touch).
    std::vector<std::vector<T>> thread_buffers(static_cast<size_t>(thread_count));

    #pragma omp parallel
    {
#ifdef _OPENMP
        const int thread_id = omp_get_thread_num();
#else
        const int thread_id = 0;
#endif
        auto& y_local = thread_buffers[static_cast<size_t>(thread_id)];
        y_local.assign(y.data.size(), T(0));
        const auto [thread_row_begin, thread_row_end] = bsr_thread_row_range(n_rows, domains);

        for (const auto& batch_entry : plan.batches) {
            const auto& batch = batch_entry.batch;
            const int row_begin = std::max(batch.row_begin, thread_row_begin);
            const int row_end = std::min(batch.row_end, thread_row_end);
            for (int row = row_begin; row < row_end; ++row) {
                const T* x_rows = x.data.data() +
                    static_cast<size_t>(graph->block_offsets[row]) * x_ld;
                const uint32_t block_begin = batch.row_block_start(row);
                const uint32_t block_end = batch.row_block_end(row);
                for (uint32_t local_block = block_begin; local_block < block_end; ++local_block) {
                    const int col = batch.cols[local_block];
                    rowmajor_kernels::rm_gemm_adjoint<T>(
                        runtime_block_size,
                        runtime_block_size,
                        nv,
                        batch.block_ptr(local_block),
                        x_rows,
                        x_ld,
                        y_local.data() + static_cast<size_t>(graph->block_offsets[col]) * y_ld,
                        y_ld);
                }
            }
        }
    }

    #pragma omp parallel for
    for (size_t index = 0; index < y.data.size(); ++index) {
        T sum = T(0);
        for (const auto& thread_buffer : thread_buffers) {
            // A buffer stays empty when the region ran with fewer threads
            // than thread_count.
            if (!thread_buffer.empty()) {
                sum += thread_buffer[index];
            }
        }
        y.data[index] = sum;
    }

    y.reduce_ghosts();
}

template <typename T>
void bsr_mult_dense_adjoint(
    DistGraph* graph,
    const BSRMatrixBackend<T>& backend,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
    // Vendor path is opt-in (VBCSR_BSR_VENDOR): only then align the MKL pool
    // with the OpenMP budget and build vendor handles.
    if (bsr_vendor_enabled()) {
        BLASKernel::configure_vendor_sparse_threading();
        const auto& cache = backend.ensure_vendor_cache(
            graph->adj_ptr,
            graph->adj_ind,
            static_cast<int>(graph->block_sizes.size()));
        if (cache.kind == BSRVendorBackendKind::MKL) {
            std::fill(y.data.begin(), y.data.end(), T(0));
            if (bsr_try_vendor_mkl_dense(graph, backend, cache, x, y, true, false)) {
                y.reduce_ghosts();
                return;
            }
        }
    }
#endif

    BLASKernel::configure_native_threading();
    bsr_mult_dense_adjoint_impl(graph, backend, x, y);
}

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_KERNELS_BSR_APPLY_HPP

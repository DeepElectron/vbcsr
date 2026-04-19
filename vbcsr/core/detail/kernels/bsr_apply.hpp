#ifndef VBCSR_DETAIL_KERNELS_BSR_APPLY_HPP
#define VBCSR_DETAIL_KERNELS_BSR_APPLY_HPP

// TODO: the native kernels can be optimized to use batched GEMM for acceleration
// But since we have vendor MKL path, the priority is less than other features.

#include "../../dist_multivector.hpp"
#include "../../dist_vector.hpp"
#include "dense_kernels.hpp"
#include "../backend/matrix_backend.hpp"

#include <algorithm>
#include <cstring>
#include <vector>
#include <xmmintrin.h>

namespace vbcsr::detail {

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
    bool adjoint) {
    const matrix_descr descr = bsr_mkl_descr();
    const sparse_layout_t layout = SPARSE_LAYOUT_COLUMN_MAJOR;
    const int num_vecs = x.num_vectors;
    const int x_ld = x.local_rows + x.ghost_rows;
    const int y_ld = y.local_rows + y.ghost_rows;

    const auto run_dense_adjoint_via_mv = [&]() -> bool {
        const sparse_operation_t op = bsr_mkl_operation<T>(true);
        for (const auto& entry : cache.batches) {
            const size_t row_offset =
                static_cast<size_t>(graph->block_offsets[entry.batch.row_begin]);
            for (int vec = 0; vec < num_vecs; ++vec) {
                sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
                if constexpr (std::is_same_v<T, double>) {
                    const double alpha = 1.0;
                    const double beta = 1.0;
                    const double* x_ptr =
                        x.data.data() + static_cast<size_t>(vec) * x_ld + row_offset;
                    double* y_ptr = y.data.data() + static_cast<size_t>(vec) * y_ld;
                    status =
                        mkl_sparse_d_mv(op, alpha, entry.mkl.mv_handle, descr, x_ptr, beta, y_ptr);
                } else if constexpr (std::is_same_v<T, std::complex<double>>) {
                    const MKL_Complex16 alpha{1.0, 0.0};
                    const MKL_Complex16 beta{1.0, 0.0};
                    const auto* x_ptr = reinterpret_cast<const MKL_Complex16*>(
                        x.data.data() + static_cast<size_t>(vec) * x_ld + row_offset);
                    auto* y_ptr = reinterpret_cast<MKL_Complex16*>(
                        y.data.data() + static_cast<size_t>(vec) * y_ld);
                    status =
                        mkl_sparse_z_mv(op, alpha, entry.mkl.mv_handle, descr, x_ptr, beta, y_ptr);
                }

                if (status != SPARSE_STATUS_SUCCESS) {
                    return false;
                }
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
                const double* b_ptr = x.data.data() + row_offset;
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
                    x.data.data() + row_offset);
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
            const double beta = 1.0;
            const double* b_ptr = x.data.data();
            double* c_ptr = y.data.data() + row_offset;
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
            const MKL_Complex16 beta{1.0, 0.0};
            const auto* b_ptr =
                reinterpret_cast<const MKL_Complex16*>(x.data.data());
            auto* c_ptr = reinterpret_cast<MKL_Complex16*>(y.data.data() + row_offset);
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

} // namespace
#endif

inline int bsr_default_rhs_tile(int block_size) {
    return block_size <= 8 ? 8 : 4;
}

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

inline std::pair<int, int> bsr_thread_row_range(int n_rows) {
#ifdef _OPENMP
    const int thread_count = omp_get_num_threads();
    const int thread_id = omp_get_thread_num();
#else
    const int thread_count = 1;
    const int thread_id = 0;
#endif
    const int begin = (n_rows * thread_id) / thread_count;
    const int end = (n_rows * (thread_id + 1)) / thread_count;
    return {begin, end};
}

template <typename T>
inline void bsr_pack_rhs_tile(
    T* packed_tile,
    const T* source,
    int block_size,
    int rhs_count,
    int source_ld) {
    for (int rhs = 0; rhs < rhs_count; ++rhs) {
        const T* src_col = source + static_cast<size_t>(rhs) * source_ld;
        T* dst_col = packed_tile + static_cast<size_t>(rhs) * block_size;
        std::memcpy(dst_col, src_col, sizeof(T) * static_cast<size_t>(block_size));
    }
}

template <typename T>
inline void bsr_unpack_rhs_tile(
    T* destination,
    const T* packed_tile,
    int block_size,
    int rhs_count,
    int destination_ld) {
    for (int rhs = 0; rhs < rhs_count; ++rhs) {
        T* dst_col = destination + static_cast<size_t>(rhs) * destination_ld;
        const T* src_col = packed_tile + static_cast<size_t>(rhs) * block_size;
        std::memcpy(dst_col, src_col, sizeof(T) * static_cast<size_t>(block_size));
    }
}

template <int BlockSize, typename T>
inline void bsr_apply_block_gemv(int runtime_block_size, const T* block, const T* x, T* y) {
    if constexpr (BlockSize == 0) {
        SmartKernel<T>::gemv(runtime_block_size, runtime_block_size, T(1), block, runtime_block_size, x, 1, T(1), y, 1);
    } else {
        FixedBlockKernel<T, BlockSize, BlockSize>::gemv(block, x, y, T(1), T(1));
    }
}

template <int BlockSize, typename T>
inline void bsr_apply_block_gemm(
    int runtime_block_size,
    int rhs_count,
    const T* block,
    const T* x,
    int x_ld,
    T* y,
    int y_ld) {
    if constexpr (BlockSize == 0) {
        SmartKernel<T>::gemm(runtime_block_size, rhs_count, runtime_block_size, T(1), block, runtime_block_size, x, x_ld, T(1), y, y_ld);
    } else {
        FixedBlockKernel<T, BlockSize, BlockSize>::gemm(rhs_count, block, BlockSize, x, x_ld, y, y_ld, T(1), T(1));
    }
}

template <int BlockSize, typename T>
inline void bsr_apply_block_gemv_trans(int runtime_block_size, const T* block, const T* x, T* y) {
    if constexpr (BlockSize == 0) {
        SmartKernel<T>::gemv_trans(runtime_block_size, runtime_block_size, T(1), block, runtime_block_size, x, 1, T(1), y, 1);
    } else {
        FixedBlockKernel<T, BlockSize, BlockSize>::gemv_trans(block, x, y, T(1), T(1));
    }
}

template <int BlockSize, typename T>
inline void bsr_apply_block_gemm_trans(
    int runtime_block_size,
    int rhs_count,
    const T* block,
    const T* x,
    int x_ld,
    T* y,
    int y_ld) {
    if constexpr (BlockSize == 0) {
        SmartKernel<T>::gemm_trans(runtime_block_size, rhs_count, runtime_block_size, T(1), block, runtime_block_size, x, x_ld, T(1), y, y_ld);
    } else {
        FixedBlockKernel<T, BlockSize, BlockSize>::gemm_trans(rhs_count, block, BlockSize, x, x_ld, y, y_ld, T(1), T(1));
    }
}

template <int BlockSize, typename T>
void bsr_mult_impl(DistGraph* graph, const BSRMatrixBackend<T>& backend, DistVector<T>& x, DistVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);
    x.sync_ghosts();

    const auto& plan = backend.ensure_apply_plan(graph->adj_ptr, graph->adj_ind);
    const int n_rows = graph->adj_ptr.empty() ? 0 : static_cast<int>(graph->adj_ptr.size()) - 1;
    const int runtime_block_size = backend.block_size;

    std::fill(y.data.begin(), y.data.end(), T(0));

    #pragma omp parallel
    {
        const auto [thread_row_begin, thread_row_end] = bsr_thread_row_range(n_rows);
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
                    bsr_apply_block_gemv<BlockSize>(runtime_block_size, block, x_block, y_block);
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
    BLASKernel::configure_vendor_sparse_threading();
    const auto& cache = backend.ensure_vendor_cache(
        graph->adj_ptr,
        graph->adj_ind,
        static_cast<int>(graph->block_sizes.size()));
    if (cache.kind != BSRVendorBackendKind::None) {
        std::fill(y.data.begin(), y.data.end(), T(0));
        if (cache.kind == BSRVendorBackendKind::MKL &&
            bsr_try_vendor_mkl_vector(graph, backend, cache, x, y, false)) {
            return;
        }
    }
#endif

    BLASKernel::configure_native_threading();
    bsr_dispatch_block_size(backend.block_size, [&](auto block_tag) {
        constexpr int BlockSize = decltype(block_tag)::value;
        bsr_mult_impl<BlockSize>(graph, backend, x, y);
    });
}

template <int BlockSize, typename T>
void bsr_mult_dense_impl(DistGraph* graph, const BSRMatrixBackend<T>& backend, DistMultiVector<T>& x, DistMultiVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);
    x.sync_ghosts();

    const auto& plan = backend.ensure_apply_plan(graph->adj_ptr, graph->adj_ind);
    const int n_rows = graph->adj_ptr.empty() ? 0 : static_cast<int>(graph->adj_ptr.size()) - 1;
    const int runtime_block_size = backend.block_size;
    const int num_vecs = x.num_vectors;
    const int x_ld = x.local_rows + x.ghost_rows;
    const int y_ld = y.local_rows + y.ghost_rows;
    const int rhs_tile = bsr_default_rhs_tile(runtime_block_size);

    std::fill(y.data.begin(), y.data.end(), T(0));

    #pragma omp parallel
    {
        std::vector<T> x_tile(static_cast<size_t>(runtime_block_size) * rhs_tile, T(0));
        const auto [thread_row_begin, thread_row_end] = bsr_thread_row_range(n_rows);

        for (const auto& batch_entry : plan.batches) {
            const auto& batch = batch_entry.batch;
            const int row_begin = std::max(batch.row_begin, thread_row_begin);
            const int row_end = std::min(batch.row_end, thread_row_end);
            for (int row = row_begin; row < row_end; ++row) {
                const uint32_t block_begin = batch.row_block_start(row);
                const uint32_t block_end = batch.row_block_end(row);
                const uint64_t row_offset = graph->block_offsets[row];
                for (int vec_begin = 0; vec_begin < num_vecs; vec_begin += rhs_tile) {
                    const int tile_width = std::min(rhs_tile, num_vecs - vec_begin);
                    T* y_block = y.data.data() + row_offset + static_cast<size_t>(vec_begin) * y_ld;
                    for (uint32_t local_block = block_begin; local_block < block_end; ++local_block) {
                        const int col = batch.cols[local_block];
                        const T* block = batch.block_ptr(local_block);
                        const T* x_block = x.data.data() + graph->block_offsets[col] + static_cast<size_t>(vec_begin) * x_ld;
                        bsr_pack_rhs_tile(x_tile.data(), x_block, runtime_block_size, tile_width, x_ld);
                        bsr_apply_block_gemm<BlockSize>(
                            runtime_block_size,
                            tile_width,
                            block,
                            x_tile.data(),
                            runtime_block_size,
                            y_block,
                            y_ld);
                    }
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
    BLASKernel::configure_vendor_sparse_threading();
    const auto& cache = backend.ensure_vendor_cache(
        graph->adj_ptr,
        graph->adj_ind,
        static_cast<int>(graph->block_sizes.size()));
    if (cache.kind != BSRVendorBackendKind::None) {
        std::fill(y.data.begin(), y.data.end(), T(0));
        if (cache.kind == BSRVendorBackendKind::MKL &&
            bsr_try_vendor_mkl_dense(graph, backend, cache, x, y, false)) {
            return;
        }
    }
#endif

    BLASKernel::configure_native_threading();
    bsr_dispatch_block_size(backend.block_size, [&](auto block_tag) {
        constexpr int BlockSize = decltype(block_tag)::value;
        bsr_mult_dense_impl<BlockSize>(graph, backend, x, y);
    });
}

template <int BlockSize, typename T>
void bsr_mult_adjoint_impl(DistGraph* graph, const BSRMatrixBackend<T>& backend, DistVector<T>& x, DistVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);

    const auto& plan = backend.ensure_apply_plan(graph->adj_ptr, graph->adj_ind);
    const int n_rows = graph->adj_ptr.empty() ? 0 : static_cast<int>(graph->adj_ptr.size()) - 1;
    const int runtime_block_size = backend.block_size;
    const int thread_count =
#ifdef _OPENMP
        std::max(1, omp_get_max_threads());
#else
        1;
#endif

    std::fill(y.data.begin(), y.data.end(), T(0));
    std::vector<std::vector<T>> thread_buffers(
        static_cast<size_t>(thread_count),
        std::vector<T>(y.data.size(), T(0)));

    #pragma omp parallel
    {
#ifdef _OPENMP
        const int thread_id = omp_get_thread_num();
#else
        const int thread_id = 0;
#endif
        auto& y_local = thread_buffers[static_cast<size_t>(thread_id)];
        const auto [thread_row_begin, thread_row_end] = bsr_thread_row_range(n_rows);

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
                    bsr_apply_block_gemv_trans<BlockSize>(runtime_block_size, block, x_block, y_block);
                }
            }
        }
    }

    #pragma omp parallel for
    for (size_t index = 0; index < y.data.size(); ++index) {
        T sum = T(0);
        for (const auto& thread_buffer : thread_buffers) {
            sum += thread_buffer[index];
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
    BLASKernel::configure_vendor_sparse_threading();
    const auto& cache = backend.ensure_vendor_cache(
        graph->adj_ptr,
        graph->adj_ind,
        static_cast<int>(graph->block_sizes.size()));
    if (cache.kind != BSRVendorBackendKind::None) {
        std::fill(y.data.begin(), y.data.end(), T(0));
        if (cache.kind == BSRVendorBackendKind::MKL &&
            bsr_try_vendor_mkl_vector(graph, backend, cache, x, y, true)) {
            y.reduce_ghosts();
            return;
        }
    }
#endif

    BLASKernel::configure_native_threading();
    bsr_dispatch_block_size(backend.block_size, [&](auto block_tag) {
        constexpr int BlockSize = decltype(block_tag)::value;
        bsr_mult_adjoint_impl<BlockSize>(graph, backend, x, y);
    });
}

template <int BlockSize, typename T>
void bsr_mult_dense_adjoint_impl(
    DistGraph* graph,
    const BSRMatrixBackend<T>& backend,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);

    const auto& plan = backend.ensure_apply_plan(graph->adj_ptr, graph->adj_ind);
    const int n_rows = graph->adj_ptr.empty() ? 0 : static_cast<int>(graph->adj_ptr.size()) - 1;
    const int num_vecs = x.num_vectors;
    const int runtime_block_size = backend.block_size;
    const int x_ld = x.local_rows + x.ghost_rows;
    const int y_ld = y.local_rows + y.ghost_rows;
    const int rhs_tile = bsr_default_rhs_tile(runtime_block_size);
    const int thread_count =
#ifdef _OPENMP
        std::max(1, omp_get_max_threads());
#else
        1;
#endif

    std::fill(y.data.begin(), y.data.end(), T(0));

    for (int vec_begin = 0; vec_begin < num_vecs; vec_begin += rhs_tile) {
        const int tile_width = std::min(rhs_tile, num_vecs - vec_begin);
        std::vector<std::vector<T>> thread_buffers(
            static_cast<size_t>(thread_count),
            std::vector<T>(static_cast<size_t>(y_ld) * tile_width, T(0)));

        #pragma omp parallel
        {
#ifdef _OPENMP
            const int thread_id = omp_get_thread_num();
#else
            const int thread_id = 0;
#endif
            auto& y_local = thread_buffers[static_cast<size_t>(thread_id)];
            std::vector<T> x_tile(static_cast<size_t>(runtime_block_size) * rhs_tile, T(0));
            const auto [thread_row_begin, thread_row_end] = bsr_thread_row_range(n_rows);

            for (const auto& batch_entry : plan.batches) {
                const auto& batch = batch_entry.batch;
                const int row_begin = std::max(batch.row_begin, thread_row_begin);
                const int row_end = std::min(batch.row_end, thread_row_end);
                for (int row = row_begin; row < row_end; ++row) {
                    const T* x_block = x.data.data() + graph->block_offsets[row] + static_cast<size_t>(vec_begin) * x_ld;
                    bsr_pack_rhs_tile(x_tile.data(), x_block, runtime_block_size, tile_width, x_ld);
                    const uint32_t block_begin = batch.row_block_start(row);
                    const uint32_t block_end = batch.row_block_end(row);
                    for (uint32_t local_block = block_begin; local_block < block_end; ++local_block) {
                        const int col = batch.cols[local_block];
                        const T* block = batch.block_ptr(local_block);
                        T* y_block = y_local.data() + graph->block_offsets[col];
                        bsr_apply_block_gemm_trans<BlockSize>(
                            runtime_block_size,
                            tile_width,
                            block,
                            x_tile.data(),
                            runtime_block_size,
                            y_block,
                            y_ld);
                    }
                }
            }
        }

        const size_t tile_base = static_cast<size_t>(vec_begin) * y_ld;
        const size_t tile_size = static_cast<size_t>(y_ld) * tile_width;
        #pragma omp parallel for
        for (size_t index = 0; index < tile_size; ++index) {
            T sum = T(0);
            for (const auto& thread_buffer : thread_buffers) {
                sum += thread_buffer[index];
            }
            y.data[tile_base + index] = sum;
        }
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
    BLASKernel::configure_vendor_sparse_threading();
    const auto& cache = backend.ensure_vendor_cache(
        graph->adj_ptr,
        graph->adj_ind,
        static_cast<int>(graph->block_sizes.size()));
    if (cache.kind != BSRVendorBackendKind::None) {
        std::fill(y.data.begin(), y.data.end(), T(0));
        if (cache.kind == BSRVendorBackendKind::MKL &&
            bsr_try_vendor_mkl_dense(graph, backend, cache, x, y, true)) {
            y.reduce_ghosts();
            return;
        }
    }
#endif

    BLASKernel::configure_native_threading();
    bsr_dispatch_block_size(backend.block_size, [&](auto block_tag) {
        constexpr int BlockSize = decltype(block_tag)::value;
        bsr_mult_dense_adjoint_impl<BlockSize>(graph, backend, x, y);
    });
}

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_KERNELS_BSR_APPLY_HPP

#ifndef VBCSR_DETAIL_OPS_SPMM_BSR_HPP
#define VBCSR_DETAIL_OPS_SPMM_BSR_HPP

#include "../../kernels/bsr_apply.hpp"
#include "../../distributed/block_payload_exchange.hpp"
#include "../../distributed/result_graph.hpp"
#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace vbcsr::detail {

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
struct BSRMKLSparseHandleOwner {
    sparse_matrix_t handle = nullptr;

    BSRMKLSparseHandleOwner() = default;
    BSRMKLSparseHandleOwner(const BSRMKLSparseHandleOwner&) = delete;
    BSRMKLSparseHandleOwner& operator=(const BSRMKLSparseHandleOwner&) = delete;

    ~BSRMKLSparseHandleOwner() {
        destroy_mkl_sparse_handle(handle);
    }
};
#endif

template <typename Matrix>
struct BSRSpMMExecutor {
    using T = typename Matrix::value_type;

    template <int BlockSize>
    static void accumulate_product(
        int runtime_block_size,
        const T* a_block,
        const T* b_block,
        T* dest) {
        if constexpr (BlockSize == 0) {
            SmartKernel<T>::gemm(
                runtime_block_size,
                runtime_block_size,
                runtime_block_size,
                T(1),
                a_block,
                runtime_block_size,
                b_block,
                runtime_block_size,
                T(1),
                dest,
                runtime_block_size);
        } else {
            FixedBlockKernel<T, BlockSize, BlockSize>::gemm(
                BlockSize,
                a_block,
                BlockSize,
                b_block,
                BlockSize,
                dest,
                BlockSize,
                T(1),
                T(1));
        }
    }

    static Matrix run(const Matrix& A, const Matrix& B, double threshold) {
#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
        if (threshold <= 0.0) {
            if (auto result = run_mkl_serial(A, B, threshold)) {
                return std::move(*result);
            }
        }
#endif
        return run_generic(A, B, threshold);
    }

private:
    static Matrix make_empty_like_product(const Matrix& A) {
        const int n_rows = static_cast<int>(A.row_ptr().size()) - 1;
        std::vector<std::vector<int>> adjacency(static_cast<size_t>(n_rows));
        DistGraph* c_graph = construct_result_graph(A, adjacency, std::map<int, int>{}, "spmm");

        Matrix C(c_graph);
        C.owns_graph = true;
        C.graph->enable_matrix_lifetime_management();
        C.set_page_size(A.configured_page_size());
        return C;
    }

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
    static bool is_mkl_supported_scalar_type() {
        return std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>;
    }

    static bool can_use_mkl_serial(const Matrix& A, const Matrix& B) {
        const auto& A_backend = A.active_bsr_backend();
        const auto& B_backend = B.active_bsr_backend();
        if (!is_mkl_supported_scalar_type()) {
            return false;
        }
        if (A.graph->size != 1 || B.graph->size != 1) {
            return false;
        }
        if (A_backend.block_size != B_backend.block_size) {
            return false;
        }
        if (A.graph->block_sizes.size() != B.graph->block_sizes.size()) {
            return false;
        }
        if (A.local_block_nnz() > 0 && A_backend.values.page_count() != 1) {
            return false;
        }
        if (B.local_block_nnz() > 0 && B_backend.values.page_count() != 1) {
            return false;
        }
        return true;
    }

    static DistGraph* construct_serial_result_graph(
        const Matrix& A,
        const std::vector<int>& row_ptr,
        const std::vector<int>& local_cols) {
        auto* graph = new DistGraph(A.graph->comm);
        graph->owned_global_indices = A.graph->owned_global_indices;
        graph->global_to_local = A.graph->global_to_local;
        graph->block_displs = A.graph->block_displs;

        const int n_owned = static_cast<int>(graph->owned_global_indices.size());
        graph->block_sizes.assign(
            A.graph->block_sizes.begin(),
            A.graph->block_sizes.begin() + n_owned);
        graph->ghost_global_indices.clear();

        graph->adj_ptr = row_ptr;
        graph->adj_ind = local_cols;

        graph->block_offsets.resize(graph->block_sizes.size() + 1);
        graph->block_offsets[0] = 0;
        for (size_t idx = 0; idx < graph->block_sizes.size(); ++idx) {
            graph->block_offsets[idx + 1] = graph->block_offsets[idx] + graph->block_sizes[idx];
        }

        graph->send_counts.assign(graph->size, 0);
        graph->recv_counts.assign(graph->size, 0);
        graph->send_indices.clear();
        graph->recv_indices.clear();
        graph->send_displs.assign(static_cast<size_t>(graph->size) + 1, 0);
        graph->recv_displs.assign(static_cast<size_t>(graph->size) + 1, 0);
        graph->send_ranks.clear();
        graph->recv_ranks.clear();
        graph->send_counts_scalar.assign(graph->size, 0);
        graph->recv_counts_scalar.assign(graph->size, 0);
        graph->send_displs_scalar.assign(static_cast<size_t>(graph->size) + 1, 0);
        graph->recv_displs_scalar.assign(static_cast<size_t>(graph->size) + 1, 0);
        return graph;
    }

    static BSRPageBatch<const T> full_mkl_batch(const Matrix& matrix) {
        const auto page = matrix.active_bsr_backend().page(matrix.col_ind(), 0);
        BSRPageBatch<const T> batch;
        batch.cols = page.cols;
        batch.values = page.values;
        batch.row_block_offsets = matrix.row_ptr().data();
        batch.block_count = page.block_count;
        batch.block_size = page.block_size;
        batch.block_value_count = page.block_value_count;
        batch.page_index = 0;
        batch.first_block = 0;
        batch.row_begin = 0;
        batch.row_end = static_cast<int>(matrix.row_ptr().size()) - 1;
        return batch;
    }

    static bool build_mkl_raw_handle(
        sparse_matrix_t& out_handle,
        BSRPageBatch<const T> batch,
        int num_block_cols) {
        if (batch.block_count > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
            batch.row_count() > std::numeric_limits<int>::max() ||
            num_block_cols > std::numeric_limits<int>::max()) {
            return false;
        }

        destroy_mkl_sparse_handle(out_handle);

        sparse_status_t status = SPARSE_STATUS_NOT_SUPPORTED;
        const MKL_INT rows = static_cast<MKL_INT>(batch.row_count());
        const MKL_INT cols = static_cast<MKL_INT>(num_block_cols);
        const MKL_INT mkl_block_size = static_cast<MKL_INT>(batch.block_size);
        auto* row_begin = reinterpret_cast<MKL_INT*>(const_cast<int*>(batch.row_block_offsets));
        auto* row_end = row_begin + 1;
        auto* col_index = reinterpret_cast<MKL_INT*>(const_cast<int*>(batch.cols));

        if constexpr (std::is_same_v<T, double>) {
            status = mkl_sparse_d_create_bsr(
                &out_handle,
                SPARSE_INDEX_BASE_ZERO,
                SPARSE_LAYOUT_COLUMN_MAJOR,
                rows,
                cols,
                mkl_block_size,
                row_begin,
                row_end,
                col_index,
                const_cast<double*>(batch.values));
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            status = mkl_sparse_z_create_bsr(
                &out_handle,
                SPARSE_INDEX_BASE_ZERO,
                SPARSE_LAYOUT_COLUMN_MAJOR,
                rows,
                cols,
                mkl_block_size,
                row_begin,
                row_end,
                col_index,
                reinterpret_cast<MKL_Complex16*>(
                    const_cast<std::complex<double>*>(batch.values)));
        } else {
            return false;
        }

        if (status != SPARSE_STATUS_SUCCESS) {
            destroy_mkl_sparse_handle(out_handle);
            return false;
        }
        return true;
    }

    static bool export_mkl_bsr(
        sparse_matrix_t handle,
        sparse_index_base_t& index_base,
        sparse_layout_t& block_layout,
        MKL_INT& rows,
        MKL_INT& cols,
        MKL_INT& exported_block_size,
        MKL_INT*& row_start,
        MKL_INT*& row_end,
        MKL_INT*& col_ind,
        T*& values) {
        if constexpr (std::is_same_v<T, double>) {
            double* raw_values = nullptr;
            const sparse_status_t status = mkl_sparse_d_export_bsr(
                handle,
                &index_base,
                &block_layout,
                &rows,
                &cols,
                &exported_block_size,
                &row_start,
                &row_end,
                &col_ind,
                &raw_values);
            values = raw_values;
            return status == SPARSE_STATUS_SUCCESS;
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            MKL_Complex16* raw_values = nullptr;
            const sparse_status_t status = mkl_sparse_z_export_bsr(
                handle,
                &index_base,
                &block_layout,
                &rows,
                &cols,
                &exported_block_size,
                &row_start,
                &row_end,
                &col_ind,
                &raw_values);
            values = reinterpret_cast<T*>(raw_values);
            return status == SPARSE_STATUS_SUCCESS;
        } else {
            return false;
        }
    }

    static void copy_contiguous_export_values(T* dest, const T* src, size_t count) {
#ifdef _OPENMP
        constexpr size_t kParallelCopyThresholdBytes = 1u << 20;
        if (count * sizeof(T) >= kParallelCopyThresholdBytes &&
            BLASKernel::preferred_parallel_thread_count() > 1) {
            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                const int nth = omp_get_num_threads();
                const size_t begin = (count * static_cast<size_t>(tid)) /
                                     static_cast<size_t>(nth);
                const size_t end = (count * static_cast<size_t>(tid + 1)) /
                                   static_cast<size_t>(nth);
                if (end > begin) {
                    std::memcpy(
                        dest + begin,
                        src + begin,
                        (end - begin) * sizeof(T));
                }
            }
            return;
        }
#endif
        std::memcpy(dest, src, count * sizeof(T));
    }

    static std::unique_ptr<Matrix> make_result_matrix_uninitialized(
        const Matrix& A,
        DistGraph* c_graph,
        size_t block_count,
        int block_size) {
        std::unique_ptr<DistGraph> graph_guard(c_graph);
        std::unique_ptr<Matrix> C(
            new Matrix(
                graph_guard.get(),
                MatrixKind::BSR,
                true,
                typename Matrix::ConstructionToken{}));
        graph_guard.release();

        using BSRBackendStorage = typename Matrix::BSRBackendStorage;
        BSRBackendStorage backend;
        const uint32_t configured_blocks_per_page =
            BSRBackendStorage::max_blocks_per_page(block_size);
        const uint32_t active_blocks_per_page = block_count == 0
            ? configured_blocks_per_page
            : static_cast<uint32_t>(
                  std::min<uint64_t>(
                      static_cast<uint64_t>(block_count),
                      static_cast<uint64_t>(configured_blocks_per_page)));
        backend.initialize_structure_for_complete_overwrite(
            static_cast<uint64_t>(block_count),
            block_size,
            configured_blocks_per_page,
            active_blocks_per_page);
        C->attach_backend(std::move(backend));
        return C;
    }

    static std::unique_ptr<Matrix> run_mkl_serial(
        const Matrix& A,
        const Matrix& B,
        double threshold) {
        if (!can_use_mkl_serial(A, B)) {
            return nullptr;
        }
        if (A.local_block_nnz() == 0 || B.local_block_nnz() == 0) {
            return std::make_unique<Matrix>(make_empty_like_product(A));
        }

        BLASKernel::configure_vendor_sparse_threading();

        BSRMKLSparseHandleOwner a_handle;
        BSRMKLSparseHandleOwner b_handle;
        BSRMKLSparseHandleOwner product_handle;
        BSRMKLSparseHandleOwner converted_handle;
        const bool profile = std::getenv("VBCSR_PROFILE_BSR_SPGEMM") != nullptr;
        const auto t0 = std::chrono::steady_clock::now();

        const BSRPageBatch<const T> a_batch = full_mkl_batch(A);
        const BSRPageBatch<const T> b_batch = full_mkl_batch(B);
        const int a_num_cols = static_cast<int>(A.graph->block_sizes.size());
        const int b_num_cols = static_cast<int>(B.graph->block_sizes.size());
        if (!build_mkl_raw_handle(a_handle.handle, a_batch, a_num_cols) ||
            !build_mkl_raw_handle(b_handle.handle, b_batch, b_num_cols)) {
            return nullptr;
        }
        const auto t_handles = std::chrono::steady_clock::now();

        const sparse_status_t spmm_status =
            mkl_sparse_spmm(
                SPARSE_OPERATION_NON_TRANSPOSE,
                a_handle.handle,
                b_handle.handle,
                &product_handle.handle);
        if (spmm_status != SPARSE_STATUS_SUCCESS || product_handle.handle == nullptr) {
            return nullptr;
        }

        A.active_bsr_backend().note_vendor_launch(1);
        const auto t_spmm = std::chrono::steady_clock::now();

        sparse_matrix_t export_handle = product_handle.handle;
        if (mkl_sparse_order(export_handle) != SPARSE_STATUS_SUCCESS) {
            return nullptr;
        }
        const auto t_order_initial = std::chrono::steady_clock::now();

        sparse_index_base_t index_base = SPARSE_INDEX_BASE_ZERO;
        sparse_layout_t block_layout = SPARSE_LAYOUT_COLUMN_MAJOR;
        MKL_INT rows = 0;
        MKL_INT cols = 0;
        MKL_INT exported_block_size = 0;
        MKL_INT* row_start = nullptr;
        MKL_INT* row_end = nullptr;
        MKL_INT* col_ind = nullptr;
        T* values = nullptr;

        if (!export_mkl_bsr(
                export_handle,
                index_base,
                block_layout,
                rows,
                cols,
                exported_block_size,
                row_start,
                row_end,
                col_ind,
                values)) {
            if (mkl_sparse_convert_bsr(
                    product_handle.handle,
                    static_cast<MKL_INT>(A.active_bsr_backend().block_size),
                    SPARSE_LAYOUT_COLUMN_MAJOR,
                    SPARSE_OPERATION_NON_TRANSPOSE,
                    &converted_handle.handle) != SPARSE_STATUS_SUCCESS ||
                converted_handle.handle == nullptr) {
                return nullptr;
            }
            export_handle = converted_handle.handle;
            if (mkl_sparse_order(export_handle) != SPARSE_STATUS_SUCCESS) {
                return nullptr;
            }
            if (!export_mkl_bsr(
                    export_handle,
                    index_base,
                    block_layout,
                    rows,
                    cols,
                    exported_block_size,
                    row_start,
                    row_end,
                    col_ind,
                    values)) {
                return nullptr;
            }
        }
        const auto t_export = std::chrono::steady_clock::now();

        const int n_rows = static_cast<int>(A.row_ptr().size()) - 1;
        const int block_size = A.active_bsr_backend().block_size;
        if (rows != static_cast<MKL_INT>(n_rows) ||
            exported_block_size != static_cast<MKL_INT>(block_size)) {
            throw std::runtime_error("MKL BSR SpGEMM returned an unexpected block structure");
        }

        const MKL_INT base = index_base == SPARSE_INDEX_BASE_ONE ? 1 : 0;
        const MKL_INT first = row_start[0] - base;
        const MKL_INT last = row_end[n_rows - 1] - base;
        if (last < first) {
            throw std::runtime_error("MKL BSR SpGEMM exported invalid row offsets");
        }
        const size_t exported_nnz = static_cast<size_t>(last - first);
        std::vector<int> c_row_ptr(static_cast<size_t>(n_rows) + 1, 0);
        std::vector<int> c_cols_local(exported_nnz);
        for (int row = 0; row < n_rows; ++row) {
            const int row_begin =
                static_cast<int>((row_start[row] - base) - first);
            const int row_end_offset =
                static_cast<int>((row_end[row] - base) - first);
            c_row_ptr[static_cast<size_t>(row)] = row_begin;
            c_row_ptr[static_cast<size_t>(row) + 1] = row_end_offset;
        }
        for (size_t entry = 0; entry < exported_nnz; ++entry) {
            const int local_col =
                static_cast<int>(col_ind[first + static_cast<MKL_INT>(entry)] - base);
            if (local_col < 0 || local_col >= static_cast<int>(B.graph->block_sizes.size())) {
                throw std::runtime_error("MKL BSR SpGEMM exported an invalid column index");
            }
            c_cols_local[entry] = local_col;
        }
        const auto t_rows = std::chrono::steady_clock::now();

        DistGraph* c_graph = construct_serial_result_graph(A, c_row_ptr, c_cols_local);
        auto C = make_result_matrix_uninitialized(
            A,
            c_graph,
            exported_nnz,
            block_size);
        const auto t_graph = std::chrono::steady_clock::now();

        const size_t values_per_block =
            static_cast<size_t>(block_size) * static_cast<size_t>(block_size);
        auto& C_backend = C->active_bsr_backend();
        if (block_layout != SPARSE_LAYOUT_COLUMN_MAJOR &&
            block_layout != SPARSE_LAYOUT_ROW_MAJOR) {
            throw std::runtime_error("MKL BSR SpGEMM exported an unsupported block layout");
        }
        if (exported_nnz > 0 &&
            block_layout == SPARSE_LAYOUT_COLUMN_MAJOR &&
            C_backend.values.page_count() == 1) {
            auto page = C_backend.page(C->col_ind(), 0);
            copy_contiguous_export_values(
                page.values,
                values + static_cast<size_t>(first) * values_per_block,
                exported_nnz * values_per_block);
        } else {
            #pragma omp parallel for
            for (int slot = 0; slot < static_cast<int>(exported_nnz); ++slot) {
                const T* src =
                    values +
                    (static_cast<size_t>(first) + static_cast<size_t>(slot)) *
                        values_per_block;
                T* dest = C->mutable_block_data(slot);
                if (block_layout == SPARSE_LAYOUT_COLUMN_MAJOR) {
                    std::memcpy(dest, src, values_per_block * sizeof(T));
                } else if (block_layout == SPARSE_LAYOUT_ROW_MAJOR) {
                    for (int row = 0; row < block_size; ++row) {
                        for (int col = 0; col < block_size; ++col) {
                            dest[static_cast<size_t>(col) * block_size + row] =
                                src[static_cast<size_t>(row) * block_size + col];
                        }
                    }
                }
            }
        }
        C->norms_valid = false;
        C->filter_blocks(threshold);
        const auto t_fill = std::chrono::steady_clock::now();

        if (profile) {
            auto seconds = [](auto a, auto b) {
                return std::chrono::duration<double>(b - a).count();
            };
            std::cerr
                << "VBCSR_PROFILE_BSR_SPGEMM"
                << " handles=" << seconds(t0, t_handles)
                << " spmm=" << seconds(t_handles, t_spmm)
                << " order_initial=" << seconds(t_spmm, t_order_initial)
                << " export=" << seconds(t_order_initial, t_export)
                << " rows=" << seconds(t_export, t_rows)
                << " graph=" << seconds(t_rows, t_graph)
                << " fill=" << seconds(t_graph, t_fill)
                << " total=" << seconds(t0, t_fill)
                << std::endl;
        }

        return C;
    }
#endif

    static Matrix run_generic(const Matrix& A, const Matrix& B, double threshold) {
        const auto& A_backend = A.active_bsr_backend();
        const auto& B_backend = B.active_bsr_backend();
        if (A_backend.block_size != B_backend.block_size) {
            throw std::runtime_error("BSR SpMM requires matching uniform block sizes");
        }

        auto metadata = exchange_ghost_metadata(A, B);
        auto sym = symbolic_multiply_filtered(A, B, metadata, threshold);
        auto payload_ctx = fetch_required_block_payloads(B, sym.required_blocks);
        auto ghost_blocks = build_spmm_ghost_blocks(metadata, std::move(payload_ctx));
        auto adjacency = build_spmm_result_adjacency(A, sym);

        const auto& A_norms = A.get_block_norms();
        const auto& B_local_norms = B.get_block_norms();

        const int n_rows = static_cast<int>(A.row_ptr().size()) - 1;
        DistGraph* c_graph = construct_result_graph(A, adjacency, ghost_blocks.sizes, "spmm");

        Matrix C(c_graph);
        C.owns_graph = true;
        C.graph->enable_matrix_lifetime_management();
        C.set_page_size(A.configured_page_size());
        const int block_size = A_backend.block_size;

        bsr_dispatch_block_size(block_size, [&](auto block_tag) {
            constexpr int BlockSize = decltype(block_tag)::value;

            #pragma omp parallel for
            for (int row = 0; row < n_rows; ++row) {
                const int c_start = sym.c_row_ptr[row];
                const int c_end = sym.c_row_ptr[row + 1];
                if (c_start == c_end) {
                    continue;
                }

                std::vector<T*> dest_ptrs(c_end - c_start);
                for (int idx = c_start; idx < c_end; ++idx) {
                    const int global_col = sym.c_col_ind[idx];
                    const int local_col = c_graph->global_to_local.at(global_col);
                    const int dest_start = c_graph->adj_ptr[row];
                    const int dest_end = c_graph->adj_ptr[row + 1];
                    auto begin = c_graph->adj_ind.begin() + dest_start;
                    auto end = c_graph->adj_ind.begin() + dest_end;
                    auto it = std::lower_bound(begin, end, local_col);
                    if (it == end || *it != local_col) {
                        throw std::runtime_error("BSR SpMM could not locate destination block");
                    }
                    const int graph_block_index =
                        static_cast<int>(std::distance(c_graph->adj_ind.begin(), it));
                    dest_ptrs[static_cast<size_t>(idx - c_start)] =
                        C.mutable_block_data(graph_block_index);
                }

                const int a_start = A.row_ptr()[row];
                const int a_end = A.row_ptr()[row + 1];
                const double row_eps = threshold / std::max(1, a_end - a_start);
                const auto sym_begin = sym.c_col_ind.begin() + c_start;
                const auto sym_end = sym.c_col_ind.begin() + c_end;

                auto accumulate_entry = [&](int global_col, const T* a_block, double norm_a, const T* b_block, double norm_b) {
                    if (norm_a * norm_b < row_eps) {
                        return;
                    }
                    auto it = std::lower_bound(sym_begin, sym_end, global_col);
                    if (it == sym_end || *it != global_col) {
                        return;
                    }
                    accumulate_product<BlockSize>(
                        block_size,
                        a_block,
                        b_block,
                        dest_ptrs[static_cast<size_t>(std::distance(sym_begin, it))]);
                };

                for (int slot = a_start; slot < a_end; ++slot) {
                    const double norm_a = A_norms[slot];
                    const T* a_value = A.block_data(slot);
                    const int global_inner = A.graph->get_global_index(A.col_ind()[slot]);

                    if (A.graph->find_owner(global_inner) == A.graph->rank) {
                        const int local_row_b = B.graph->global_to_local.at(global_inner);
                        // DistGraph rows are sorted by local IDs, and ghost local IDs are
                        // owner-grouped, so local traversal order is not guaranteed to be
                        // globally sorted once ghosts are present. Look up each result
                        // destination through the symbolic row instead of assuming a
                        // monotone global-column walk.
                        for (int b_slot = B.row_ptr()[local_row_b]; b_slot < B.row_ptr()[local_row_b + 1]; ++b_slot) {
                            const int global_col = B.graph->get_global_index(B.col_ind()[b_slot]);
                            const double norm_b = B_local_norms[b_slot];
                            accumulate_entry(
                                global_col,
                                a_value,
                                norm_a,
                                B.block_data(b_slot),
                                norm_b);
                        }
                    } else {
                        auto ghost_it = ghost_blocks.rows.find(global_inner);
                        if (ghost_it == ghost_blocks.rows.end()) {
                            continue;
                        }
                        for (const auto& block : ghost_it->second) {
                            accumulate_entry(
                                block.col,
                                a_value,
                                norm_a,
                                block.data,
                                block.norm);
                        }
                    }
                }
            }
        });

        C.filter_blocks(threshold);
        return C;
    }
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_OPS_SPMM_BSR_HPP

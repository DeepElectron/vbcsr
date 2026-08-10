#ifndef VBCSR_DETAIL_OPS_SPMM_BSR_HPP
#define VBCSR_DETAIL_OPS_SPMM_BSR_HPP

#include "../../kernels/bsr_apply.hpp"
#include "../../kernels/dense_kernels.hpp"
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

    // Vendor batch GEMM in the numeric phase, dispatched on block size the way
    // every kernel choice here is. The policy and the measurements behind it
    // live in common.hpp's vendor_batch_profitable, shared with the fused
    // triple product, because the crossover is a property of the machine and
    // the block size and two copies of it would drift.
    static bool spgemm_batch_enabled(int block_size) {
        return vendor_batch_profitable(block_size);
    }

    // All of one inner index's surviving products in one grouped vendor call:
    // same A block, same shape, distinct destinations -- one group, no races.
    template <int BlockSize>
    static void flush_product_batch(
        int runtime_block_size,
        const T* a_block,
        const std::vector<const T*>& b_blocks,
        const std::vector<T*>& c_blocks) {
        const int bs = BlockSize == 0 ? runtime_block_size : BlockSize;
        if (grouped_block_gemm_batch<T>(bs, a_block, b_blocks, c_blocks)) return;
        for (size_t i = 0; i < b_blocks.size(); ++i) {
            accumulate_product<BlockSize>(runtime_block_size, a_block, b_blocks[i], c_blocks[i]);
        }
    }

    template <int BlockSize>
    static void accumulate_product(
        int runtime_block_size,
        const T* a_block,
        const T* b_block,
        T* dest) {
        // dest += a_block * b_block, all bs x bs blocks in canonical
        // row-major storage.
        const int bs = BlockSize == 0 ? runtime_block_size : BlockSize;
        rowmajor_kernels::rm_gemm<T>(bs, bs, bs, a_block, b_block, bs, dest, bs);
    }

    static Matrix run(const Matrix& A, const Matrix& B, double threshold,
                      bool upper_only = false) {
#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
        // The vendor path computes the full product; the triangular
        // restriction only exists in the generic executor.
        if (threshold <= 0.0 && !upper_only) {
            if (auto result = run_mkl_serial(A, B, threshold)) {
                return std::move(*result);
            }
        }
#endif
        return run_generic(A, B, threshold, upper_only);
    }

    // A <- A B, releasing A's value pages as the numeric phase consumes them.
    // Always the generic path: the vendor route computes the whole product from
    // a handle over A's storage, so there is no point at which a prefix of it
    // is provably dead.
    static Matrix run_consuming(Matrix& A, const Matrix& B, double threshold) {
        return run_generic(A, B, threshold, /*upper_only=*/false, /*consume_A=*/&A);
    }

private:
    // A result whose pages are mapped but not yet touched. The numeric loop
    // zeroes each row range as it reaches it (see the chunk loop), so the
    // first-touch placement the ordinary constructor would have done up front
    // still happens, just spread across the phase that writes the values --
    // and the answer is never resident before it has been computed.
    static Matrix make_result_matrix_deferred_touch(const Matrix& A, DistGraph* c_graph,
                                                    int block_size) {
        std::unique_ptr<DistGraph> graph_guard(c_graph);
        Matrix C(graph_guard.get(), MatrixKind::BSR, true, typename Matrix::ConstructionToken{});
        graph_guard.release();
        C.owns_graph = true;
        C.graph->enable_matrix_lifetime_management();
        C.set_page_size(A.configured_page_size());

        typename Matrix::BSRBackendStorage backend;
        backend.initialize_structure_deferred_touch(
            C.graph->adj_ptr, block_size, A.configured_page_size());
        C.attach_backend(std::move(backend));
        return C;
    }

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
        graph->adj_ind.assign(local_cols.data(), local_cols.size());

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
                SPARSE_LAYOUT_ROW_MAJOR,
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
                SPARSE_LAYOUT_ROW_MAJOR,
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

        BLASKernel::align_vendor_threads();

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
        // No mkl_sparse_order (measured ~6x the multiply on CSR, similar cost
        // class here). Default keeps the vendor's per-row export order (see
        // spgemm_sorted_output_enabled in spmm/common.hpp);
        // VBCSR_SPGEMM_SORTED=1 restores sorted columns in the copy-out.
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
                    SPARSE_LAYOUT_ROW_MAJOR,
                    SPARSE_OPERATION_NON_TRANSPOSE,
                    &converted_handle.handle) != SPARSE_STATUS_SUCCESS ||
                converted_handle.handle == nullptr) {
                return nullptr;
            }
            export_handle = converted_handle.handle;
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
        // Optional sorted output (VBCSR_SPGEMM_SORTED=1): unsorted rows sort
        // a permutation that the block copy below places values through.
        std::vector<MKL_INT> value_perm;
        if (spgemm_sorted_output_enabled()) {
            int any_unsorted = 0;
            #pragma omp parallel for reduction(|:any_unsorted)
            for (int row = 0; row < n_rows; ++row) {
                const MKL_INT* src = col_ind + first + c_row_ptr[row];
                const int deg = c_row_ptr[row + 1] - c_row_ptr[row];
                if (!std::is_sorted(src, src + deg)) {
                    any_unsorted = 1;
                }
            }
            if (any_unsorted) {
                value_perm.resize(exported_nnz);
                // Packed (col << 32 | entry) keys sort as plain integers and
                // unpack into sorted columns + the block permutation below.
                std::vector<uint64_t> keys(exported_nnz);
                #pragma omp parallel for
                for (int row = 0; row < n_rows; ++row) {
                    const int row_begin = c_row_ptr[row];
                    const int row_end_off = c_row_ptr[row + 1];
                    const MKL_INT* src = col_ind + first;
                    for (int i = row_begin; i < row_end_off; ++i) {
                        keys[static_cast<size_t>(i)] =
                            (static_cast<uint64_t>(static_cast<uint32_t>(src[i] - base)) << 32) |
                            static_cast<uint32_t>(i);
                    }
                    std::sort(keys.begin() + row_begin, keys.begin() + row_end_off);
                }
                for (size_t entry = 0; entry < exported_nnz; ++entry) {
                    value_perm[entry] = static_cast<MKL_INT>(keys[entry] & 0xffffffffu);
                }
            }
        }
        for (size_t entry = 0; entry < exported_nnz; ++entry) {
            const MKL_INT src_entry = value_perm.empty()
                ? static_cast<MKL_INT>(entry)
                : value_perm[entry];
            const int local_col =
                static_cast<int>(col_ind[first + src_entry] - base);
            if (local_col < 0 || local_col >= static_cast<int>(B.graph->block_sizes.size())) {
                throw std::runtime_error("MKL BSR SpGEMM exported an invalid column index");
            }
            c_cols_local[entry] = local_col;
        }
        const auto t_rows = std::chrono::steady_clock::now();

        DistGraph* c_graph = construct_serial_result_graph(A, c_row_ptr, c_cols_local);
        auto C = make_result_matrix_uninitialized(
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
        // The result's value pages are fresh and untouched, so this copy IS
        // the first touch: split it per row-domain and store the partition so
        // C's placement matches the split later applies use
        // (numa_locality_plan.md — operation results).
        {
            const auto& c_rows = C->row_ptr();
            const int c_n_rows =
                c_rows.empty() ? 0 : static_cast<int>(c_rows.size()) - 1;
            C_backend.thread_domains = build_thread_domain_partition(
                c_n_rows,
                thread_domain_max_threads(),
                [&](int row) { return c_rows[row + 1] - c_rows[row]; });
        }
        if (exported_nnz > 0 &&
            value_perm.empty() &&
            block_layout == SPARSE_LAYOUT_ROW_MAJOR &&
            C_backend.values.page_count() == 1) {
            auto page = C_backend.page(C->col_ind(), 0);
            const T* src = values + static_cast<size_t>(first) * values_per_block;
            const auto& c_rows = C->row_ptr();
            const auto& c_domains = C_backend.thread_domains;
            #pragma omp parallel for schedule(static)
            for (int domain = 0; domain < c_domains.thread_count; ++domain) {
                const size_t begin =
                    static_cast<size_t>(c_rows[c_domains.domain_begin(domain)]) * values_per_block;
                const size_t end =
                    static_cast<size_t>(c_rows[c_domains.domain_end(domain)]) * values_per_block;
                if (end > begin) {
                    copy_contiguous_export_values(page.values + begin, src + begin, end - begin);
                }
            }
        } else {
            #pragma omp parallel for
            for (int slot = 0; slot < static_cast<int>(exported_nnz); ++slot) {
                const size_t src_slot = value_perm.empty()
                    ? static_cast<size_t>(slot)
                    : static_cast<size_t>(value_perm[static_cast<size_t>(slot)]);
                const T* src =
                    values +
                    (static_cast<size_t>(first) + src_slot) * values_per_block;
                T* dest = C->mutable_block_data(slot);
                if (block_layout == SPARSE_LAYOUT_ROW_MAJOR) {
                    std::memcpy(dest, src, values_per_block * sizeof(T));
                } else if (block_layout == SPARSE_LAYOUT_COLUMN_MAJOR) {
                    for (int row = 0; row < block_size; ++row) {
                        for (int col = 0; col < block_size; ++col) {
                            dest[static_cast<size_t>(row) * block_size + col] =
                                src[static_cast<size_t>(col) * block_size + row];
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
                << " export=" << seconds(t_spmm, t_export)
                << " rows=" << seconds(t_export, t_rows)
                << " graph=" << seconds(t_rows, t_graph)
                << " fill=" << seconds(t_graph, t_fill)
                << " total=" << seconds(t0, t_fill)
                << std::endl;
        }

        return C;
    }
#endif

    // `consume_A`, when set, must alias A: the caller is doing A <- A B and has
    // no further use for A's values, so the numeric phase runs in row chunks
    // and hands each chunk's input pages back as it passes them. Row i of the
    // result reads row i of A and rows of B only -- never another row of A --
    // and A is the LEFT operand, so no other rank fetches its blocks either.
    // Those two facts are the whole licence for releasing early.
    //
    // With consume_A null the chunk loop runs exactly once and this is the
    // original single `omp for`: no extra barrier, no dispatch change.
    static Matrix run_generic(const Matrix& A, const Matrix& B, double threshold,
                              bool upper_only = false, Matrix* consume_A = nullptr) {
        const auto& A_backend = A.active_bsr_backend();
        const auto& B_backend = B.active_bsr_backend();
        if (A_backend.block_size != B_backend.block_size) {
            throw std::runtime_error("BSR SpMM requires matching uniform block sizes");
        }

        const bool profile = std::getenv("VBCSR_PROFILE_BSR_SPGEMM") != nullptr;
        auto stamp = [] { return std::chrono::steady_clock::now(); };
        const auto t0 = stamp();
        const double rss0 = profile ? profile_rss_gb() : 0.0;
        auto metadata = exchange_ghost_metadata(A, B);
        auto sym = symbolic_multiply_filtered(A, B, metadata, threshold, upper_only);
        const auto t_symbolic = stamp();
        const double rss_symbolic = profile ? profile_rss_gb() : 0.0;
        auto payload_ctx = fetch_required_block_payloads(B, sym.required_blocks);
        auto ghost_blocks = build_spmm_ghost_blocks(metadata, std::move(payload_ctx));
        const auto t_fetch = stamp();
        const double rss_fetch = profile ? profile_rss_gb() : 0.0;
        auto adjacency = build_spmm_result_adjacency(A, sym);

        const auto& A_norms = A.get_block_norms();
        const auto& B_local_norms = B.get_block_norms();

        const int n_rows = static_cast<int>(A.row_ptr().size()) - 1;
        DistGraph* c_graph = construct_result_graph(A, adjacency, ghost_blocks.sizes, "spmm");

        const int block_size = A_backend.block_size;
        Matrix C = make_result_matrix_deferred_touch(A, c_graph, block_size);

        // One inner index (row, slot) contributes at most once to each
        // destination, so its surviving products form a batch of same-shape
        // GEMMs with distinct outputs: exactly what the vendor's grouped
        // batch API wants. One call per (row, slot) replaces one small GEMM
        // per block pair, which is where a lone bs x bs product cannot reach
        // vendor efficiency. Runtime-off with VBCSR_SPGEMM_BATCH=0; tiny
        // batches fall back to the direct kernel where call overhead would
        // dominate.
        const bool batch_active = spgemm_batch_enabled(A_backend.block_size);

        // Hoisted out of the pair loop: B's global column per slot. Resolving
        // it per block PAIR was measured at a substantial share of the ~1us
        // per-pair overhead that dominated the numeric phase; per slot it is
        // one O(nnz(B)) pass.
        std::vector<int> b_global_cols(B.col_ind().size());
        #pragma omp parallel for
        for (long long i = 0; i < static_cast<long long>(b_global_cols.size()); ++i) {
            b_global_cols[static_cast<size_t>(i)] =
                B.graph->get_global_index(B.col_ind()[static_cast<size_t>(i)]);
        }

        // O(1) destination resolution: a per-thread epoch-tagged scatter array
        // over the global block columns replaces a per-pair binary search into
        // the symbolic row -- the other large share of the per-pair overhead.
        // Falls back to the search when the global block count would make the
        // per-thread arrays unreasonable.
        const int n_global_blocks =
            A.graph->block_displs.empty() ? 0 : A.graph->block_displs.back();
        const bool use_scatter = n_global_blocks > 0 && n_global_blocks <= (1 << 22);

        // 4 chunks, and the count matters more than it looks.
        //
        // Chunking is what turns the peak from |A| + |C| into max(|A|, |C|) plus
        // the unreleased remainder, so more chunks release more. But the zeroing
        // pass below is domain-aligned -- it has to be, it is the placing first
        // touch -- and a chunk spans only (domains / chunks) domains, so at 16
        // chunks it ran on about 3 of 48 threads. That made the CONSUMING path
        // slower than the ordinary one, which is backwards: it does strictly
        // less allocation work.
        //
        // Measured, 2048 rows x bs 13, spmm against spmm_inplace:
        //   16 chunks  0.317 / 0.417   (in-place 1.32x SLOWER)
        //    8 chunks  0.324 / 0.386
        //    4 chunks  0.326 / 0.315   (parity, and 3/4 of A still released)
        //    2 chunks  0.313 / 0.289   (in-place faster, but only half released)
        // 4 is where in-place stops paying for its own memory saving.
        //
        // The real fix is to stop coupling the two granularities: give each
        // thread its own domain to stream and release (a range release rather
        // than a prefix release), so every thread is busy and the release is
        // finer than any chunk count. That is a larger change than a constant.
        // 4 chunks, and the count is machine-INDEPENDENT for a reason worth
        // stating, because the first attempt at this got it wrong.
        //
        // The domain-aligned touch runs on the domains a chunk spans, so the
        // fraction of threads active during it is (domains/chunks)/domains =
        // 1/chunks. That ratio does not depend on the thread count at all. An
        // earlier version derived the count from threads (min(16, threads/12)),
        // which looked like it removed a fitted constant but silently collapsed
        // to a SINGLE chunk at 12 threads or fewer -- no page release at all,
        // which is the entire point of the consuming path. Measured across 2-48
        // threads it was never slower, because it had stopped doing the work.
        //
        // 4 chunks: a quarter of the threads on the touch, three quarters of the
        // operand released progressively. Measured at 48 threads, spmm against
        // spmm_inplace: 16 chunks 0.317/0.417 (in-place 1.32x SLOWER), 8 chunks
        // 0.324/0.386, 4 chunks 0.326/0.315, 2 chunks 0.313/0.289.
        //
        // It is NOT free at low thread counts, and that is the honest trade:
        // 48 threads 0.311/0.313, 12 threads 0.770/0.799, 4 threads
        // 2.036/2.118 -- so about 4% on a small machine, where a quarter of
        // four threads is one thread doing the touch and the per-chunk barriers
        // have less to hide behind. Paid deliberately: this is a memory-driven
        // path, and the alternative rule that avoided the 4% did so by dropping
        // to a single chunk and releasing nothing.
        constexpr int kConsumeChunks = 4;
        const int chunk_rows = consume_A == nullptr
                                   ? std::max(1, n_rows)
                                   : std::max(1, (n_rows + kConsumeChunks - 1) / kConsumeChunks);

        const double rss_alloc = profile ? profile_rss_gb() : 0.0;
        const auto t_numeric0 = stamp();
#ifdef VBCSR_BLAS_HAS_BATCH_GEMM
        // The batch calls run inside the OpenMP row loop: one product batch
        // per thread at a time, vendor pool pinned to one thread meanwhile.
        std::unique_ptr<BLASKernel::ScopedSerialBLAS> serial_blas;
        if (batch_active) serial_blas = std::make_unique<BLASKernel::ScopedSerialBLAS>();
#endif

        bsr_dispatch_block_size(block_size, [&](auto block_tag) {
            constexpr int BlockSize = decltype(block_tag)::value;

            #pragma omp parallel
            {
                std::vector<const T*> bat_b;
                std::vector<T*> bat_c;
                std::vector<int> dest_slot;
                std::vector<int> dest_tag;
                if (use_scatter) {
                    dest_slot.assign(static_cast<size_t>(n_global_blocks), 0);
                    dest_tag.assign(static_cast<size_t>(n_global_blocks), -1);
                }

                // Dynamic, not static: with upper_only the work per row falls
                // linearly across the matrix (row i touches only columns
                // >= i), and a contiguous static split hands the first thread
                // about twice the mean -- measured as the halved product
                // running at two-thirds the per-flop speed of the full one.
                // Cyclic static (static,4) was tried as a zero-overhead
                // alternative and measured WORSE for the triangular case
                // (1.71x vs 1.96x); dynamic's cost on balanced products is
                // inside run-to-run noise.
                // One pass when nothing is being consumed; otherwise enough
                // chunks that the unreleased tail of A stays small, traded
                // against the barrier at each chunk edge confining dynamic
                // balancing to that chunk's rows.
                for (int chunk_begin = 0; chunk_begin < n_rows; chunk_begin += chunk_rows) {
                const int chunk_end = std::min(n_rows, chunk_begin + chunk_rows);
                // The first touch the constructor no longer does. Each domain
                // clears its own share of this chunk, so the page still lands on
                // the node whose thread will later apply that row -- the whole
                // point of the pass this replaces -- while pages beyond the
                // chunk stay unmapped-in.
                {
                    const auto& domains = C.bsr_thread_domains();
                    #pragma omp for schedule(static)
                    for (int domain = 0; domain < domains.thread_count; ++domain) {
                        const int begin = std::max(domains.domain_begin(domain), chunk_begin);
                        const int end = std::min(domains.domain_end(domain), chunk_end);
                        if (end > begin) {
                            C.bsr_zero_row_range(begin, end);
                        }
                    }
                }
                #pragma omp for schedule(dynamic, 4)
                for (int row = chunk_begin; row < chunk_end; ++row) {
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

                    if (use_scatter) {
                        for (int idx = c_start; idx < c_end; ++idx) {
                            dest_slot[static_cast<size_t>(sym.c_col_ind[idx])] = idx - c_start;
                            dest_tag[static_cast<size_t>(sym.c_col_ind[idx])] = row;
                        }
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
                        T* dest;
                        if (use_scatter) {
                            if (dest_tag[static_cast<size_t>(global_col)] != row) {
                                return;
                            }
                            dest = dest_ptrs[static_cast<size_t>(
                                dest_slot[static_cast<size_t>(global_col)])];
                        } else {
                            auto it = std::lower_bound(sym_begin, sym_end, global_col);
                            if (it == sym_end || *it != global_col) {
                                return;
                            }
                            dest = dest_ptrs[static_cast<size_t>(std::distance(sym_begin, it))];
                        }
                        if (batch_active) {
                            bat_b.push_back(b_block);
                            bat_c.push_back(dest);
                            return;
                        }
                        accumulate_product<BlockSize>(block_size, a_block, b_block, dest);
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
                                const int global_col = b_global_cols[static_cast<size_t>(b_slot)];
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
                                continue;  // batch is empty here: flushed at the last slot
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

                        if (batch_active && !bat_b.empty()) {
                            flush_product_batch<BlockSize>(
                                block_size, a_value, bat_b, bat_c);
                            bat_b.clear();
                            bat_c.clear();
                        }
                    }
                }
                // Implicit barrier above: every row below chunk_end is done on
                // every thread, so A's blocks below its row pointer are dead.
                if (consume_A != nullptr) {
                    #pragma omp single
                    {
                        consume_A->release_value_blocks_before(A.row_ptr()[chunk_end]);
                    }
                }
                }
            }
        });

        const auto t_numeric = stamp();
        const double rss_numeric = profile ? profile_rss_gb() : 0.0;
        C.filter_blocks(threshold);

        if (profile) {
            auto seconds = [](auto a, auto b) {
                return std::chrono::duration<double>(b - a).count();
            };
            std::cerr << "VBCSR_PROFILE_BSR_SPGEMM"
                      << " symbolic=" << seconds(t0, t_symbolic)
                      << " fetch=" << seconds(t_symbolic, t_fetch)
                      << " numeric=" << seconds(t_numeric0, t_numeric)
                      << " filter=" << seconds(t_numeric, stamp())
                      << " batch=" << (batch_active ? 1 : 0)
                      << " upper_only=" << (upper_only ? 1 : 0)
                      << " | rssGB in=" << rss0
                      << " symbolic=" << rss_symbolic
                      << " fetch=" << rss_fetch
                      << " alloc=" << rss_alloc
                      << " numeric=" << rss_numeric
                      << " filter=" << profile_rss_gb()
                      << std::endl;
        }
        return C;
    }
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_OPS_SPMM_BSR_HPP

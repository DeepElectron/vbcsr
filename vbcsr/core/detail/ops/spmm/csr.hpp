#ifndef VBCSR_DETAIL_OPS_SPMM_CSR_HPP
#define VBCSR_DETAIL_OPS_SPMM_CSR_HPP

#include "../../backend/csr_vendor_cache.hpp"
#include "../../distributed/block_payload_exchange.hpp"
#include "../../distributed/result_graph.hpp"
#include "../../kernels/dense_kernels.hpp"
#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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

#ifdef VBCSR_HAVE_MKL_SPARSE
struct MKLSparseHandleOwner {
    sparse_matrix_t handle = nullptr;

    MKLSparseHandleOwner() = default;
    MKLSparseHandleOwner(const MKLSparseHandleOwner&) = delete;
    MKLSparseHandleOwner& operator=(const MKLSparseHandleOwner&) = delete;

    ~MKLSparseHandleOwner() {
        destroy_mkl_sparse_handle(handle);
    }
};
#endif

template <typename Matrix>
struct CSRSpMMExecutor {
    using T = typename Matrix::value_type;

    static Matrix run(const Matrix& A, const Matrix& B, double threshold) {
#ifdef VBCSR_HAVE_MKL_SPARSE
        if (auto result = run_mkl_serial(A, B, threshold)) {
            return std::move(*result);
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

#ifdef VBCSR_HAVE_MKL_SPARSE
    static double scalar_abs(const T& value) {
        using std::abs;
        return static_cast<double>(abs(value));
    }

    static bool is_mkl_supported_scalar_type() {
        return std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>;
    }

    static bool can_use_mkl_serial(const Matrix& A, const Matrix& B) {
        if (!is_mkl_supported_scalar_type()) {
            return false;
        }
        if (A.graph->size != 1 || B.graph->size != 1) {
            return false;
        }
        if (A.graph->block_sizes.size() != B.graph->block_sizes.size()) {
            return false;
        }
        if (A.local_block_nnz() > 0 && A.active_csr_backend().page_count() != 1) {
            return false;
        }
        if (B.local_block_nnz() > 0 && B.active_csr_backend().page_count() != 1) {
            return false;
        }
        return true;
    }

    static DistGraph* construct_serial_result_graph(
        const Matrix& A,
        std::vector<int>&& row_ptr,
        std::vector<int>&& local_cols) {
        auto* graph = new DistGraph(A.graph->comm);
        graph->owned_global_indices = A.graph->owned_global_indices;
        graph->global_to_local = A.graph->global_to_local;
        graph->block_displs = A.graph->block_displs;

        const int n_owned = static_cast<int>(graph->owned_global_indices.size());
        graph->block_sizes.assign(
            A.graph->block_sizes.begin(),
            A.graph->block_sizes.begin() + n_owned);
        graph->ghost_global_indices.clear();

        graph->adj_ptr = std::move(row_ptr);
        graph->adj_ind = std::move(local_cols);

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

    static CSRPageBatch<const T> full_mkl_batch(const Matrix& matrix) {
        const auto page = matrix.active_csr_backend().page(matrix.col_ind(), 0);
        CSRPageBatch<const T> batch;
        batch.cols = page.cols;
        batch.values = page.values;
        batch.row_offsets = matrix.row_ptr().data();
        batch.nnz_count = page.nnz_count;
        batch.page_index = 0;
        batch.first_nnz = 0;
        batch.row_begin = 0;
        batch.row_end = static_cast<int>(matrix.row_ptr().size()) - 1;
        return batch;
    }

    static bool export_mkl_csr(
        sparse_matrix_t handle,
        sparse_index_base_t& index_base,
        MKL_INT& rows,
        MKL_INT& cols,
        MKL_INT*& row_start,
        MKL_INT*& row_end,
        MKL_INT*& col_ind,
        T*& values) {
        if constexpr (std::is_same_v<T, double>) {
            double* raw_values = nullptr;
            const sparse_status_t status = mkl_sparse_d_export_csr(
                handle,
                &index_base,
                &rows,
                &cols,
                &row_start,
                &row_end,
                &col_ind,
                &raw_values);
            values = raw_values;
            return status == SPARSE_STATUS_SUCCESS;
        } else if constexpr (std::is_same_v<T, std::complex<double>>) {
            MKL_Complex16* raw_values = nullptr;
            const sparse_status_t status = mkl_sparse_z_export_csr(
                handle,
                &index_base,
                &rows,
                &cols,
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

        // The vendor multiply owns parallelism: align the MKL pool with the
        // OpenMP thread budget (OMP_NUM_THREADS is the single source of
        // truth). Without this the multiply inherits whatever pool size the
        // previous call left behind.
        BLASKernel::configure_vendor_sparse_threading();

        MKLSparseHandleOwner a_handle;
        MKLSparseHandleOwner b_handle;
        MKLSparseHandleOwner c_handle;
        const bool profile = std::getenv("VBCSR_PROFILE_CSR_SPGEMM") != nullptr;
        const auto t0 = std::chrono::steady_clock::now();

        const CSRPageBatch<const T> a_batch = full_mkl_batch(A);
        const CSRPageBatch<const T> b_batch = full_mkl_batch(B);
        const int a_num_cols = static_cast<int>(A.graph->block_sizes.size());
        const int b_num_cols = static_cast<int>(B.graph->block_sizes.size());
        if (!build_csr_mkl_raw_handle(a_handle.handle, a_batch, a_num_cols) ||
            !build_csr_mkl_raw_handle(b_handle.handle, b_batch, b_num_cols)) {
            return nullptr;
        }
        const auto t_handles = std::chrono::steady_clock::now();

        const sparse_status_t spmm_status =
            mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, a_handle.handle, b_handle.handle, &c_handle.handle);
        if (spmm_status != SPARSE_STATUS_SUCCESS || c_handle.handle == nullptr) {
            return nullptr;
        }

        A.active_csr_backend().note_vendor_launch(1);
        const auto t_spmm = std::chrono::steady_clock::now();

        // No mkl_sparse_order here: it was measured at ~6x the multiply
        // itself (migration plan E4). By default the result keeps the
        // vendor's per-row export order — no library consumer requires a
        // matrix's own adjacency sorted (see spgemm_sorted_output_enabled in
        // spmm/common.hpp); VBCSR_SPGEMM_SORTED=1 restores sorted columns via
        // a per-row packed-key sort in the copy-out below.
        sparse_index_base_t index_base = SPARSE_INDEX_BASE_ZERO;
        MKL_INT rows = 0;
        MKL_INT cols = 0;
        MKL_INT* row_start = nullptr;
        MKL_INT* row_end = nullptr;
        MKL_INT* col_ind = nullptr;
        T* values = nullptr;
        if (!export_mkl_csr(
                c_handle.handle,
                index_base,
                rows,
                cols,
                row_start,
                row_end,
                col_ind,
                values)) {
            return nullptr;
        }
        const auto t_export = std::chrono::steady_clock::now();

        const int n_rows = static_cast<int>(A.row_ptr().size()) - 1;
        if (rows != static_cast<MKL_INT>(n_rows)) {
            throw std::runtime_error("MKL CSR SpGEMM returned an unexpected row count");
        }

        const MKL_INT base = index_base == SPARSE_INDEX_BASE_ONE ? 1 : 0;
        std::vector<int> c_row_ptr(static_cast<size_t>(n_rows) + 1, 0);
        std::vector<int> c_cols_local;
        std::vector<T> c_values;
        // dst slot -> src entry offset (relative to `first`); empty when every
        // exported row is already sorted (then values copy straight through).
        std::vector<MKL_INT> value_perm;

        if (threshold <= 0.0) {
            const MKL_INT first = row_start[0] - base;
            const MKL_INT last = row_end[n_rows - 1] - base;
            if (last < first) {
                throw std::runtime_error("MKL CSR SpGEMM exported invalid row offsets");
            }
            const size_t exported_nnz = static_cast<size_t>(last - first);
            for (int row = 0; row < n_rows; ++row) {
                const int row_begin =
                    static_cast<int>((row_start[row] - base) - first);
                const int row_end_offset =
                    static_cast<int>((row_end[row] - base) - first);
                c_row_ptr[static_cast<size_t>(row)] = row_begin;
                c_row_ptr[static_cast<size_t>(row) + 1] = row_end_offset;
            }

            int any_unsorted = 0;
            if (spgemm_sorted_output_enabled()) {
                #pragma omp parallel for reduction(|:any_unsorted)
                for (int row = 0; row < n_rows; ++row) {
                    const MKL_INT* src = col_ind + first + c_row_ptr[row];
                    const int deg = c_row_ptr[row + 1] - c_row_ptr[row];
                    if (!std::is_sorted(src, src + deg)) {
                        any_unsorted = 1;
                    }
                }
            }

            if (!any_unsorted) {
                if (base == 0 && std::is_same_v<MKL_INT, int>) {
                    c_cols_local.assign(col_ind + first, col_ind + last);
                } else {
                    c_cols_local.resize(exported_nnz);
                    for (size_t entry = 0; entry < exported_nnz; ++entry) {
                        const int local_col =
                            static_cast<int>(col_ind[first + static_cast<MKL_INT>(entry)] - base);
                        if (local_col < 0 || local_col >= static_cast<int>(B.graph->block_sizes.size())) {
                            throw std::runtime_error("MKL CSR SpGEMM exported an invalid column index");
                        }
                        c_cols_local[entry] = local_col;
                    }
                }
            } else {
                c_cols_local.resize(exported_nnz);
                value_perm.resize(exported_nnz);
                // Packed (col << 32 | entry) keys sort as plain integers —
                // no gathering comparator — and unpack into sorted columns
                // plus the value permutation in one pass. Measured >2x
                // cheaper than mkl_sparse_order at ~800K nnz.
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
                const int num_cols = static_cast<int>(B.graph->block_sizes.size());
                for (size_t entry = 0; entry < exported_nnz; ++entry) {
                    const int local_col = static_cast<int>(keys[entry] >> 32);
                    if (local_col < 0 || local_col >= num_cols) {
                        throw std::runtime_error("MKL CSR SpGEMM exported an invalid column index");
                    }
                    c_cols_local[entry] = local_col;
                    value_perm[entry] = static_cast<MKL_INT>(keys[entry] & 0xffffffffu);
                }
            }
        } else {
            size_t exported_nnz = 0;
            for (int row = 0; row < n_rows; ++row) {
                const MKL_INT begin = row_start[row] - base;
                const MKL_INT end = row_end[row] - base;
                if (end < begin) {
                    throw std::runtime_error("MKL CSR SpGEMM exported invalid row offsets");
                }
                exported_nnz += static_cast<size_t>(end - begin);
            }
            c_cols_local.reserve(exported_nnz);
            c_values.reserve(exported_nnz);
            std::vector<std::pair<int, T>> row_entries;
            for (int row = 0; row < n_rows; ++row) {
                c_row_ptr[static_cast<size_t>(row)] = static_cast<int>(c_cols_local.size());
                const MKL_INT begin = row_start[row] - base;
                const MKL_INT end = row_end[row] - base;
                row_entries.clear();
                for (MKL_INT idx = begin; idx < end; ++idx) {
                    const T value = values[idx];
                    if (scalar_abs(value) < threshold) {
                        continue;
                    }
                    const int local_col = static_cast<int>(col_ind[idx] - base);
                    if (local_col < 0 || local_col >= static_cast<int>(B.graph->block_sizes.size())) {
                        throw std::runtime_error("MKL CSR SpGEMM exported an invalid column index");
                    }
                    row_entries.emplace_back(local_col, value);
                }
                if (spgemm_sorted_output_enabled()) {
                    std::sort(row_entries.begin(), row_entries.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
                }
                for (const auto& [col, value] : row_entries) {
                    c_cols_local.push_back(col);
                    c_values.push_back(value);
                }
                c_row_ptr[static_cast<size_t>(row) + 1] = static_cast<int>(c_cols_local.size());
            }
        }
        const auto t_rows = std::chrono::steady_clock::now();

        const size_t result_nnz = c_cols_local.size();
        DistGraph* c_graph =
            construct_serial_result_graph(A, std::move(c_row_ptr), std::move(c_cols_local));
        const auto t_graph_construct = std::chrono::steady_clock::now();
        // Token construction + direct backend build: skips the default
        // zero-filled allocation (every value slot is overwritten below) and
        // the set_page_size rebuild.
        auto C = std::unique_ptr<Matrix>(new Matrix(
            c_graph, MatrixKind::CSR, true, typename Matrix::ConstructionToken{}));
        {
            typename Matrix::CSRBackendStorage backend;
            backend.initialize_structure_for_complete_overwrite(
                result_nnz, A.configured_page_size());
            C->attach_backend(std::move(backend));
        }
        const auto t_graph = std::chrono::steady_clock::now();

        if (threshold <= 0.0 && !value_perm.empty()) {
            // Unsorted export: place values through the per-row sort permutation.
            const MKL_INT first = row_start[0] - base;
            if (C->active_csr_backend().page_count() == 1) {
                auto page = C->active_csr_backend().page(C->col_ind(), 0);
                #pragma omp parallel for
                for (int slot = 0; slot < static_cast<int>(result_nnz); ++slot) {
                    page.values[slot] = values[first + value_perm[static_cast<size_t>(slot)]];
                }
                C->norms_valid = false;
            } else {
                #pragma omp parallel for
                for (int slot = 0; slot < static_cast<int>(result_nnz); ++slot) {
                    *C->mutable_block_data(slot) = values[first + value_perm[static_cast<size_t>(slot)]];
                }
            }
        } else if (threshold <= 0.0 && result_nnz > 0 && C->active_csr_backend().page_count() == 1) {
            auto page = C->active_csr_backend().page(C->col_ind(), 0);
            std::memcpy(
                page.values,
                values + (row_start[0] - base),
                result_nnz * sizeof(T));
            C->norms_valid = false;
        } else if (threshold <= 0.0) {
            const MKL_INT first = row_start[0] - base;
            #pragma omp parallel for
            for (int slot = 0; slot < static_cast<int>(result_nnz); ++slot) {
                *C->mutable_block_data(slot) = values[first + static_cast<MKL_INT>(slot)];
            }
        } else if (!c_values.empty() && C->active_csr_backend().page_count() == 1) {
            auto page = C->active_csr_backend().page(C->col_ind(), 0);
            std::memcpy(page.values, c_values.data(), c_values.size() * sizeof(T));
            C->norms_valid = false;
        } else {
            #pragma omp parallel for
            for (int slot = 0; slot < static_cast<int>(c_values.size()); ++slot) {
                *C->mutable_block_data(slot) = c_values[static_cast<size_t>(slot)];
            }
        }
        const auto t_fill = std::chrono::steady_clock::now();

        if (profile) {
            auto seconds = [](auto a, auto b) {
                return std::chrono::duration<double>(b - a).count();
            };
            std::cerr
                << "VBCSR_PROFILE_CSR_SPGEMM"
                << " handles=" << seconds(t0, t_handles)
                << " spmm=" << seconds(t_handles, t_spmm)
                << " export=" << seconds(t_spmm, t_export)
                << " rows=" << seconds(t_export, t_rows)
                << " graph_construct=" << seconds(t_rows, t_graph_construct)
                << " matrix_ctor=" << seconds(t_graph_construct, t_graph)
                << " fill=" << seconds(t_graph, t_fill)
                << " total=" << seconds(t0, t_fill)
                << std::endl;
        }

        return C;
    }
#endif

    static Matrix run_generic(const Matrix& A, const Matrix& B, double threshold) {
        // Known limitation (distributed / non-MKL fallback only): with 1x1
        // blocks the symbolic filter step already performs the work of a full
        // multiplication, so this path costs roughly two multiplies. A fused
        // symbolic+numeric single pass would remove the duplication.
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
                    throw std::runtime_error("CSR SpMM could not locate destination block");
                }
                const int graph_block_index =
                    static_cast<int>(std::distance(c_graph->adj_ind.begin(), it));
                dest_ptrs[idx - c_start] = C.mutable_block_data(graph_block_index);
            }

            const int a_start = A.row_ptr()[row];
            const int a_end = A.row_ptr()[row + 1];
            const double row_eps = threshold / std::max(1, a_end - a_start);
            const auto sym_begin = sym.c_col_ind.begin() + c_start;
            const auto sym_end = sym.c_col_ind.begin() + c_end;

            auto accumulate_entry = [&](int global_col, const T& value) {
                auto it = std::lower_bound(sym_begin, sym_end, global_col);
                if (it == sym_end || *it != global_col) {
                    return;
                }
                *dest_ptrs[static_cast<size_t>(std::distance(sym_begin, it))] += value;
            };

            for (int slot = a_start; slot < a_end; ++slot) {
                const double norm_a = A_norms[slot];
                const T a_value = *A.block_data(slot);
                const int global_inner = A.graph->get_global_index(A.col_ind()[slot]);

                if (A.graph->find_owner(global_inner) == A.graph->rank) {
                    const int local_row_b = B.graph->global_to_local.at(global_inner);
                    for (int b_slot = B.row_ptr()[local_row_b]; b_slot < B.row_ptr()[local_row_b + 1]; ++b_slot) {
                        const double norm_b = B_local_norms[b_slot];
                        if (norm_a * norm_b < row_eps) {
                            continue;
                        }
                        const int global_col = B.graph->get_global_index(B.col_ind()[b_slot]);
                        accumulate_entry(global_col, a_value * (*B.block_data(b_slot)));
                    }
                } else {
                    auto ghost_it = ghost_blocks.rows.find(global_inner);
                    if (ghost_it == ghost_blocks.rows.end()) {
                        continue;
                    }
                    for (const auto& block : ghost_it->second) {
                        if (norm_a * block.norm < row_eps) {
                            continue;
                        }
                        accumulate_entry(block.col, a_value * block.data[0]);
                    }
                }
            }
        }

        C.filter_blocks(threshold);
        return C;
    }
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_OPS_SPMM_CSR_HPP

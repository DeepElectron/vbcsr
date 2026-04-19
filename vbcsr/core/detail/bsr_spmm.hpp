#ifndef VBCSR_DETAIL_BSR_SPMM_HPP
#define VBCSR_DETAIL_BSR_SPMM_HPP

#include "bsr_kernels.hpp"
#include "block_payload_exchange.hpp"
#include "distributed_result_graph.hpp"
#include "spmm_common.hpp"

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

namespace vbcsr::detail {

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
        const auto& A_backend = require_bsr_backend<T, typename Matrix::KernelType>(A.backend_handle_);
        const auto& B_backend = require_bsr_backend<T, typename Matrix::KernelType>(B.backend_handle_);
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

#endif // VBCSR_DETAIL_BSR_SPMM_HPP

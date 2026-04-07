#ifndef VBCSR_DETAIL_BSR_SPMM_HPP
#define VBCSR_DETAIL_BSR_SPMM_HPP

#include "bsr_kernels.hpp"
#include "bsr_result_builder.hpp"
#include "distributed_plans.hpp"
#include "distributed_result_graph.hpp"
#include "spmm_exchange.hpp"

#include <algorithm>
#include <map>
#include <vector>

namespace vbcsr::detail {

template <typename Matrix>
struct BSRSpMMExecutor {
    using T = typename Matrix::value_type;

    static Matrix run(const Matrix& A, const Matrix& B, double threshold) {
        const auto& A_backend = require_bsr_backend<T, typename Matrix::KernelType>(A.backend_handle_);
        const auto& B_backend = require_bsr_backend<T, typename Matrix::KernelType>(B.backend_handle_);
        if (A_backend.block_size != B_backend.block_size) {
            throw std::runtime_error("BSR SpMM requires matching uniform block sizes");
        }

        auto metadata_plan = RowMetadataExchangePlan<Matrix, Matrix>::build(A, B);
        auto sym = symbolic_multiply_filtered(A, B, metadata_plan.metadata(), threshold);
        auto payload_plan = BlockPayloadExchangePlan<Matrix>::fetch_required(B, sym.required_blocks);
        auto assembly_plan = ResultAssemblyPlan<Matrix>::for_spmm(
            A,
            sym,
            metadata_plan.metadata(),
            std::move(payload_plan));

        const auto& A_norms = A.get_block_norms();
        const auto& B_local_norms = B.get_block_norms();

        const int n_rows = static_cast<int>(A.row_ptr().size()) - 1;
        DistGraph* c_graph = assembly_plan.construct_result_graph(A, "spmm");

        BSRResultBuilder<T> builder(c_graph, A.backend_page_settings().bsr_page_size);
        const int block_size = builder.block_size();

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
                const int slot = builder.find_slot(row, local_col);
                dest_ptrs[idx - c_start] = builder.slot_data(slot);
            }

            const auto sym_begin = sym.c_col_ind.begin() + c_start;
            const auto sym_end = sym.c_col_ind.begin() + c_end;
            const int a_start = A.row_ptr()[row];
            const int a_end = A.row_ptr()[row + 1];
            const double row_eps = threshold / std::max(1, a_end - a_start);

            auto find_dest = [&](int global_col) -> T* {
                auto it = std::lower_bound(sym_begin, sym_end, global_col);
                if (it == sym_end || *it != global_col) {
                    return nullptr;
                }
                return dest_ptrs[static_cast<size_t>(std::distance(sym_begin, it))];
            };

            for (int slot = a_start; slot < a_end; ++slot) {
                const double norm_a = A_norms[slot];
                const T* a_value = A.block_data(slot);
                const int global_inner = A.graph->get_global_index(A.col_ind()[slot]);

                if (A.graph->find_owner(global_inner) == A.graph->rank) {
                    const int local_row_b = B.graph->global_to_local.at(global_inner);
                    for (int b_slot = B.row_ptr()[local_row_b]; b_slot < B.row_ptr()[local_row_b + 1]; ++b_slot) {
                        const double norm_b = B_local_norms[b_slot];
                        if (norm_a * norm_b < row_eps) {
                            continue;
                        }
                        T* dest = find_dest(B.graph->get_global_index(B.col_ind()[b_slot]));
                        if (dest == nullptr) {
                            continue;
                        }
                        SmartKernel<T>::gemm(
                            block_size,
                            block_size,
                            block_size,
                            T(1),
                            a_value,
                            block_size,
                            B.block_data(b_slot),
                            block_size,
                            T(1),
                            dest,
                            block_size);
                    }
                } else {
                    auto ghost_it = assembly_plan.ghost_rows().find(global_inner);
                    if (ghost_it == assembly_plan.ghost_rows().end()) {
                        continue;
                    }
                    for (const auto& block : ghost_it->second) {
                        if (norm_a * block.norm < row_eps) {
                            continue;
                        }
                        T* dest = find_dest(block.col);
                        if (dest == nullptr) {
                            continue;
                        }
                        SmartKernel<T>::gemm(
                            block_size,
                            block_size,
                            block_size,
                            T(1),
                            a_value,
                            block_size,
                            block.data,
                            block_size,
                            T(1),
                            dest,
                            block_size);
                    }
                }
            }
        }

        Matrix C = Matrix::template materialize_from_builder<vbcsr::MatrixKind::BSR>(
            c_graph,
            true,
            std::move(builder),
            A.backend_page_settings());
        C.filter_blocks(threshold);
        return C;
    }
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_BSR_SPMM_HPP

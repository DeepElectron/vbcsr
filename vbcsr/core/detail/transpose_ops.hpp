#ifndef VBCSR_DETAIL_TRANSPOSE_OPS_HPP
#define VBCSR_DETAIL_TRANSPOSE_OPS_HPP

#include "distributed_plans.hpp"
#include "vbcsr_result_builder.hpp"

#include <vector>

namespace detail {

template <typename Matrix>
struct CSRTransposeExecutor {
    using T = typename Matrix::value_type;

    static Matrix serial(const Matrix& matrix) {
        const int n_rows = static_cast<int>(matrix.row_ptr.size()) - 1;
        const int n_cols = static_cast<int>(matrix.graph->block_sizes.size());

        std::vector<std::vector<int>> c_adj(n_cols);
        for (int row = 0; row < n_rows; ++row) {
            const int global_row = matrix.graph->get_global_index(row);
            for (int slot = matrix.row_ptr[row]; slot < matrix.row_ptr[row + 1]; ++slot) {
                c_adj[matrix.col_ind[slot]].push_back(global_row);
            }
        }

        std::vector<int> c_owned_globals(n_cols);
        std::vector<int> c_block_sizes(n_cols);
        for (int row = 0; row < n_cols; ++row) {
            c_owned_globals[row] = matrix.graph->get_global_index(row);
            c_block_sizes[row] = matrix.graph->block_sizes[row];
        }

        DistGraph* graph_C = new DistGraph(matrix.graph->comm);
        graph_C->construct_distributed(c_owned_globals, c_block_sizes, c_adj);
        CSRResultBuilder<T> builder(graph_C);

        for (int row = 0; row < n_rows; ++row) {
            const int global_row = matrix.graph->get_global_index(row);
            for (int slot = matrix.row_ptr[row]; slot < matrix.row_ptr[row + 1]; ++slot) {
                if (matrix.block_size_elements(slot) != 1) {
                    throw std::logic_error("CSR transpose expects scalar slot payloads");
                }
                const int dest_row = matrix.col_ind[slot];
                const int dest_col = graph_C->global_to_local.at(global_row);
                const int dest_slot = builder.find_slot(dest_row, dest_col);
                *builder.slot_data(dest_slot) = ScalarTraits<T>::conjugate(*matrix.block_data(slot));
            }
        }

        return Matrix::from_csr_builder(graph_C, true, std::move(builder));
    }

    static Matrix distributed(const Matrix& matrix) {
        auto payload_plan = vbcsr::detail::TransposePayloadExchangePlan<T>::build(matrix);
        auto assembly_plan = vbcsr::detail::ResultAssemblyPlan<Matrix>::for_transpose(
            payload_plan.adjacency(),
            payload_plan.ghost_sizes());
        DistGraph* graph_C = assembly_plan.construct_result_graph(matrix, "transpose");

        CSRResultBuilder<T> builder(graph_C);
        const int* meta_ptr = payload_plan.recv_meta().data();
        const int* meta_end = payload_plan.recv_meta().data() + payload_plan.recv_meta().size();
        const T* value_ptr = payload_plan.recv_values().data();
        while (meta_ptr < meta_end) {
            const int global_row = *meta_ptr++;
            const int global_col = *meta_ptr++;
            const int row_dim = *meta_ptr++;
            const int col_dim = *meta_ptr++;
            if (row_dim != 1 || col_dim != 1) {
                throw std::logic_error("Distributed CSR transpose expects scalar blocks");
            }

            const int local_row = graph_C->global_to_local.at(global_row);
            const int local_col = graph_C->global_to_local.at(global_col);
            const int slot = builder.find_slot(local_row, local_col);
            *builder.slot_data(slot) = ScalarTraits<T>::conjugate(*value_ptr++);
        }

        return Matrix::from_csr_builder(graph_C, true, std::move(builder));
    }

    static Matrix run(const Matrix& matrix) {
        if (matrix.graph->size == 1) {
            return serial(matrix);
        }
        return distributed(matrix);
    }
};

template <typename Matrix>
struct BSRTransposeExecutor {
    using T = typename Matrix::value_type;

    static Matrix serial(const Matrix& matrix) {
        const int n_rows = static_cast<int>(matrix.row_ptr.size()) - 1;
        const int n_cols = static_cast<int>(matrix.graph->block_sizes.size());

        std::vector<std::vector<int>> c_adj(n_cols);
        for (int row = 0; row < n_rows; ++row) {
            const int global_row = matrix.graph->get_global_index(row);
            for (int slot = matrix.row_ptr[row]; slot < matrix.row_ptr[row + 1]; ++slot) {
                c_adj[matrix.col_ind[slot]].push_back(global_row);
            }
        }

        std::vector<int> c_owned_globals(n_cols);
        std::vector<int> c_block_sizes(n_cols);
        for (int row = 0; row < n_cols; ++row) {
            c_owned_globals[row] = matrix.graph->get_global_index(row);
            c_block_sizes[row] = matrix.graph->block_sizes[row];
        }

        DistGraph* graph_C = new DistGraph(matrix.graph->comm);
        graph_C->construct_distributed(c_owned_globals, c_block_sizes, c_adj);
        BSRResultBuilder<T> builder(graph_C);

        #pragma omp parallel for
        for (int row = 0; row < n_rows; ++row) {
            const int global_row = matrix.graph->get_global_index(row);
            for (int slot = matrix.row_ptr[row]; slot < matrix.row_ptr[row + 1]; ++slot) {
                const int dest_row = matrix.col_ind[slot];
                const int dest_col = graph_C->global_to_local.at(global_row);
                const int dest_slot = builder.find_slot(dest_row, dest_col);
                Matrix::write_transposed_conjugate_values(
                    builder.slot_data(dest_slot),
                    matrix.block_data(slot),
                    matrix.graph->block_sizes[row],
                    matrix.graph->block_sizes[matrix.col_ind[slot]]);
            }
        }

        return Matrix::from_bsr_builder(graph_C, true, std::move(builder));
    }

    static Matrix distributed(const Matrix& matrix) {
        auto payload_plan = vbcsr::detail::TransposePayloadExchangePlan<T>::build(matrix);
        auto assembly_plan = vbcsr::detail::ResultAssemblyPlan<Matrix>::for_transpose(
            payload_plan.adjacency(),
            payload_plan.ghost_sizes());
        DistGraph* graph_C = assembly_plan.construct_result_graph(matrix, "transpose");

        BSRResultBuilder<T> builder(graph_C);
        const int* meta_ptr = payload_plan.recv_meta().data();
        const int* meta_end = payload_plan.recv_meta().data() + payload_plan.recv_meta().size();
        const T* value_ptr = payload_plan.recv_values().data();
        while (meta_ptr < meta_end) {
            const int global_row = *meta_ptr++;
            const int global_col = *meta_ptr++;
            const int row_dim = *meta_ptr++;
            const int col_dim = *meta_ptr++;

            const int local_row = graph_C->global_to_local.at(global_row);
            const int local_col = graph_C->global_to_local.at(global_col);
            const int slot = builder.find_slot(local_row, local_col);
            Matrix::write_transposed_conjugate_values(builder.slot_data(slot), value_ptr, col_dim, row_dim);
            value_ptr += static_cast<size_t>(row_dim) * col_dim;
        }

        return Matrix::from_bsr_builder(graph_C, true, std::move(builder));
    }

    static Matrix run(const Matrix& matrix) {
        if (matrix.graph->size == 1) {
            return serial(matrix);
        }
        return distributed(matrix);
    }
};

template <typename Matrix>
struct VBCSRTransposeExecutor {
    using T = typename Matrix::value_type;
    using Kernel = typename Matrix::KernelType;

    static Matrix run(const Matrix& matrix) {
        if (matrix.graph->size == 1) {
            return serial(matrix);
        }
        return distributed(matrix);
    }

private:
    static Matrix serial(const Matrix& matrix) {
        const int n_rows = static_cast<int>(matrix.row_ptr.size()) - 1;
        const int n_cols = static_cast<int>(matrix.graph->block_offsets.size()) - 1;

        std::vector<std::vector<int>> c_adj(n_cols);
        for (int row = 0; row < n_rows; ++row) {
            const int global_row = matrix.graph->get_global_index(row);
            for (int slot = matrix.row_ptr[row]; slot < matrix.row_ptr[row + 1]; ++slot) {
                const int col = matrix.col_ind[slot];
                c_adj[col].push_back(global_row);
            }
        }

        std::vector<int> c_owned_globals(n_cols);
        std::vector<int> c_block_sizes(n_cols);
        for (int row = 0; row < n_cols; ++row) {
            c_owned_globals[row] = matrix.graph->get_global_index(row);
            c_block_sizes[row] = matrix.graph->block_sizes[row];
        }

        DistGraph* graph_C = new DistGraph(matrix.graph->comm);
        graph_C->construct_distributed(c_owned_globals, c_block_sizes, c_adj);

        VBCSRResultBuilder<T, Kernel> builder(graph_C);
        #pragma omp parallel for
        for (int row = 0; row < n_rows; ++row) {
            const int global_row = matrix.graph->get_global_index(row);
            for (int slot = matrix.row_ptr[row]; slot < matrix.row_ptr[row + 1]; ++slot) {
                const int dest_row = matrix.col_ind[slot];
                const int dest_col = graph_C->global_to_local.at(global_row);
                const int dest_slot = builder.find_slot(dest_row, dest_col);
                Matrix::write_transposed_conjugate_values(
                    builder.slot_data(dest_slot),
                    matrix.block_data(slot),
                    matrix.graph->block_sizes[row],
                    matrix.graph->block_sizes[matrix.col_ind[slot]]);
            }
        }
        return Matrix::from_vbcsr_builder(graph_C, true, std::move(builder));
    }

    static Matrix distributed(const Matrix& matrix) {
        auto payload_plan = vbcsr::detail::TransposePayloadExchangePlan<T>::build(matrix);
        auto assembly_plan = vbcsr::detail::ResultAssemblyPlan<Matrix>::for_transpose(
            payload_plan.adjacency(),
            payload_plan.ghost_sizes());
        DistGraph* graph_C = assembly_plan.construct_result_graph(matrix, "transpose");

        VBCSRResultBuilder<T, Kernel> builder(graph_C);
        const int* meta_ptr = payload_plan.recv_meta().data();
        const int* meta_end = payload_plan.recv_meta().data() + payload_plan.recv_meta().size();
        const T* value_ptr = payload_plan.recv_values().data();
        while (meta_ptr < meta_end) {
            const int global_row = *meta_ptr++;
            const int global_col = *meta_ptr++;
            const int row_dim = *meta_ptr++;
            const int col_dim = *meta_ptr++;
            const int local_row = graph_C->global_to_local.at(global_row);
            const int local_col = graph_C->global_to_local.at(global_col);
            const int slot = builder.find_slot(local_row, local_col);
            Matrix::write_transposed_conjugate_values(builder.slot_data(slot), value_ptr, col_dim, row_dim);
            value_ptr += static_cast<size_t>(row_dim) * col_dim;
        }
        return Matrix::from_vbcsr_builder(graph_C, true, std::move(builder));
    }
};

} // namespace detail

#endif // VBCSR_DETAIL_TRANSPOSE_OPS_HPP

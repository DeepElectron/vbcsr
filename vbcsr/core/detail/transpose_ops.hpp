#ifndef VBCSR_DETAIL_TRANSPOSE_OPS_HPP
#define VBCSR_DETAIL_TRANSPOSE_OPS_HPP

#include "distributed_plans.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace vbcsr::detail {

template <typename Matrix>
struct CSRTransposeExecutor {
    using T = typename Matrix::value_type;

    static Matrix serial(const Matrix& matrix) {
        const int n_rows = static_cast<int>(matrix.row_ptr().size()) - 1;
        const int n_cols = static_cast<int>(matrix.graph->block_sizes.size());

        std::vector<std::vector<int>> c_adj(n_cols);
        for (int row = 0; row < n_rows; ++row) {
            const int global_row = matrix.graph->get_global_index(row);
            for (int slot = matrix.row_ptr()[row]; slot < matrix.row_ptr()[row + 1]; ++slot) {
                c_adj[matrix.col_ind()[slot]].push_back(global_row);
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
        Matrix result(graph_C);
        result.owns_graph = true;
        result.graph->enable_matrix_lifetime_management();
        result.set_page_size(matrix.configured_page_size());

        for (int row = 0; row < n_rows; ++row) {
            const int global_row = matrix.graph->get_global_index(row);
            for (int slot = matrix.row_ptr()[row]; slot < matrix.row_ptr()[row + 1]; ++slot) {
                if (matrix.block_size_elements(slot) != 1) {
                    throw std::logic_error("CSR transpose expects scalar slot payloads");
                }
                const int dest_row = matrix.col_ind()[slot];
                const int dest_col = graph_C->global_to_local.at(global_row);
                const int dest_start = graph_C->adj_ptr[dest_row];
                const int dest_end = graph_C->adj_ptr[dest_row + 1];
                auto begin = graph_C->adj_ind.begin() + dest_start;
                auto end = graph_C->adj_ind.begin() + dest_end;
                auto it = std::lower_bound(begin, end, dest_col);
                if (it == end || *it != dest_col) {
                    throw std::runtime_error("CSR transpose could not locate destination block");
                }
                const int dest_graph_block =
                    static_cast<int>(std::distance(graph_C->adj_ind.begin(), it));
                *result.mutable_block_data(dest_graph_block) =
                    ScalarTraits<T>::conjugate(*matrix.block_data(slot));
            }
        }

        return result;
    }

    static Matrix distributed(const Matrix& matrix) {
        auto payload_plan = vbcsr::detail::TransposePayloadExchangePlan<T>::build(matrix);
        auto assembly_plan = vbcsr::detail::ResultAssemblyPlan<Matrix>::for_transpose(
            payload_plan.adjacency(),
            payload_plan.ghost_sizes());
        DistGraph* graph_C = assembly_plan.construct_result_graph(matrix, "transpose");

        Matrix result(graph_C);
        result.owns_graph = true;
        result.graph->enable_matrix_lifetime_management();
        result.set_page_size(matrix.configured_page_size());
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
            const int dest_start = graph_C->adj_ptr[local_row];
            const int dest_end = graph_C->adj_ptr[local_row + 1];
            auto begin = graph_C->adj_ind.begin() + dest_start;
            auto end = graph_C->adj_ind.begin() + dest_end;
            auto it = std::lower_bound(begin, end, local_col);
            if (it == end || *it != local_col) {
                throw std::runtime_error("Distributed CSR transpose could not locate destination block");
            }
            const int graph_block_index =
                static_cast<int>(std::distance(graph_C->adj_ind.begin(), it));
            *result.mutable_block_data(graph_block_index) =
                ScalarTraits<T>::conjugate(*value_ptr++);
        }

        return result;
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
        const int n_rows = static_cast<int>(matrix.row_ptr().size()) - 1;
        const int n_cols = static_cast<int>(matrix.graph->block_sizes.size());

        std::vector<std::vector<int>> c_adj(n_cols);
        for (int row = 0; row < n_rows; ++row) {
            const int global_row = matrix.graph->get_global_index(row);
            for (int slot = matrix.row_ptr()[row]; slot < matrix.row_ptr()[row + 1]; ++slot) {
                c_adj[matrix.col_ind()[slot]].push_back(global_row);
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
        Matrix result(graph_C);
        result.owns_graph = true;
        result.graph->enable_matrix_lifetime_management();
        result.set_page_size(matrix.configured_page_size());

        #pragma omp parallel for
        for (int row = 0; row < n_rows; ++row) {
                const int global_row = matrix.graph->get_global_index(row);
                for (int slot = matrix.row_ptr()[row]; slot < matrix.row_ptr()[row + 1]; ++slot) {
                const int dest_row = matrix.col_ind()[slot];
                const int dest_col = graph_C->global_to_local.at(global_row);
                const int dest_start = graph_C->adj_ptr[dest_row];
                const int dest_end = graph_C->adj_ptr[dest_row + 1];
                auto begin = graph_C->adj_ind.begin() + dest_start;
                auto end = graph_C->adj_ind.begin() + dest_end;
                auto it = std::lower_bound(begin, end, dest_col);
                if (it == end || *it != dest_col) {
                    throw std::runtime_error("BSR transpose could not locate destination block");
                }
                const int dest_graph_block =
                    static_cast<int>(std::distance(graph_C->adj_ind.begin(), it));
                Matrix::write_transposed_conjugate_values(
                    result.mutable_block_data(dest_graph_block),
                    matrix.block_data(slot),
                    matrix.graph->block_sizes[row],
                    matrix.graph->block_sizes[matrix.col_ind()[slot]]);
            }
        }

        return result;
    }

    static Matrix distributed(const Matrix& matrix) {
        auto payload_plan = vbcsr::detail::TransposePayloadExchangePlan<T>::build(matrix);
        auto assembly_plan = vbcsr::detail::ResultAssemblyPlan<Matrix>::for_transpose(
            payload_plan.adjacency(),
            payload_plan.ghost_sizes());
        DistGraph* graph_C = assembly_plan.construct_result_graph(matrix, "transpose");

        Matrix result(graph_C);
        result.owns_graph = true;
        result.graph->enable_matrix_lifetime_management();
        result.set_page_size(matrix.configured_page_size());
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
            const int dest_start = graph_C->adj_ptr[local_row];
            const int dest_end = graph_C->adj_ptr[local_row + 1];
            auto begin = graph_C->adj_ind.begin() + dest_start;
            auto end = graph_C->adj_ind.begin() + dest_end;
            auto it = std::lower_bound(begin, end, local_col);
            if (it == end || *it != local_col) {
                throw std::runtime_error("Distributed BSR transpose could not locate destination block");
            }
            const int graph_block_index =
                static_cast<int>(std::distance(graph_C->adj_ind.begin(), it));
            Matrix::write_transposed_conjugate_values(
                result.mutable_block_data(graph_block_index),
                value_ptr,
                col_dim,
                row_dim);
            value_ptr += static_cast<size_t>(row_dim) * col_dim;
        }

        return result;
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
        const int n_rows = static_cast<int>(matrix.row_ptr().size()) - 1;
        const int n_cols = static_cast<int>(matrix.graph->block_offsets.size()) - 1;

        std::vector<std::vector<int>> c_adj(n_cols);
        for (int row = 0; row < n_rows; ++row) {
            const int global_row = matrix.graph->get_global_index(row);
            for (int slot = matrix.row_ptr()[row]; slot < matrix.row_ptr()[row + 1]; ++slot) {
                const int col = matrix.col_ind()[slot];
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

        Matrix result(graph_C);
        result.owns_graph = true;
        result.graph->enable_matrix_lifetime_management();
        result.set_page_size(matrix.configured_page_size());
        #pragma omp parallel for
        for (int row = 0; row < n_rows; ++row) {
            const int global_row = matrix.graph->get_global_index(row);
            for (int slot = matrix.row_ptr()[row]; slot < matrix.row_ptr()[row + 1]; ++slot) {
                const int dest_row = matrix.col_ind()[slot];
                const int dest_col = graph_C->global_to_local.at(global_row);
                const int dest_start = graph_C->adj_ptr[dest_row];
                const int dest_end = graph_C->adj_ptr[dest_row + 1];
                auto begin = graph_C->adj_ind.begin() + dest_start;
                auto end = graph_C->adj_ind.begin() + dest_end;
                auto it = std::lower_bound(begin, end, dest_col);
                if (it == end || *it != dest_col) {
                    throw std::runtime_error("VBCSR transpose could not locate destination block");
                }
                const int dest_graph_block =
                    static_cast<int>(std::distance(graph_C->adj_ind.begin(), it));
                Matrix::write_transposed_conjugate_values(
                    result.mutable_block_data(dest_graph_block),
                    matrix.block_data(slot),
                    matrix.graph->block_sizes[row],
                    matrix.graph->block_sizes[matrix.col_ind()[slot]]);
            }
        }
        return result;
    }

    static Matrix distributed(const Matrix& matrix) {
        auto payload_plan = vbcsr::detail::TransposePayloadExchangePlan<T>::build(matrix);
        auto assembly_plan = vbcsr::detail::ResultAssemblyPlan<Matrix>::for_transpose(
            payload_plan.adjacency(),
            payload_plan.ghost_sizes());
        DistGraph* graph_C = assembly_plan.construct_result_graph(matrix, "transpose");

        Matrix result(graph_C);
        result.owns_graph = true;
        result.graph->enable_matrix_lifetime_management();
        result.set_page_size(matrix.configured_page_size());
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
            const int dest_start = graph_C->adj_ptr[local_row];
            const int dest_end = graph_C->adj_ptr[local_row + 1];
            auto begin = graph_C->adj_ind.begin() + dest_start;
            auto end = graph_C->adj_ind.begin() + dest_end;
            auto it = std::lower_bound(begin, end, local_col);
            if (it == end || *it != local_col) {
                throw std::runtime_error("Distributed VBCSR transpose could not locate destination block");
            }
            const int graph_block_index =
                static_cast<int>(std::distance(graph_C->adj_ind.begin(), it));
            Matrix::write_transposed_conjugate_values(
                result.mutable_block_data(graph_block_index),
                value_ptr,
                col_dim,
                row_dim);
            value_ptr += static_cast<size_t>(row_dim) * col_dim;
        }
        return result;
    }
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_TRANSPOSE_OPS_HPP

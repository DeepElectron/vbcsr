#ifndef VBCSR_DETAIL_TRANSPOSE_OPS_HPP
#define VBCSR_DETAIL_TRANSPOSE_OPS_HPP

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
                if (matrix.blk_sizes[slot] != 1) {
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
        auto exchange = exchange_transpose_blocks(matrix);
        DistGraph* graph_C = construct_result_graph(
            matrix.graph->comm,
            matrix.graph->owned_global_indices,
            owned_block_sizes(*matrix.graph),
            exchange.adjacency,
            exchange.ghost_dims,
            "transpose");

        CSRResultBuilder<T> builder(graph_C);
        const int* meta_ptr = exchange.recv_meta.data();
        const int* meta_end = exchange.recv_meta.data() + exchange.recv_meta.size();
        const T* value_ptr = exchange.recv_values.data();
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
        auto exchange = exchange_transpose_blocks(matrix);
        DistGraph* graph_C = construct_result_graph(
            matrix.graph->comm,
            matrix.graph->owned_global_indices,
            owned_block_sizes(*matrix.graph),
            exchange.adjacency,
            exchange.ghost_dims,
            "transpose");

        BSRResultBuilder<T> builder(graph_C);
        const int* meta_ptr = exchange.recv_meta.data();
        const int* meta_end = exchange.recv_meta.data() + exchange.recv_meta.size();
        const T* value_ptr = exchange.recv_values.data();
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
struct LegacyTransposeExecutor {
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
        std::vector<std::vector<uint64_t>> c_handles(n_cols);
        for (int row = 0; row < n_rows; ++row) {
            const int global_row = matrix.graph->get_global_index(row);
            for (int slot = matrix.row_ptr[row]; slot < matrix.row_ptr[row + 1]; ++slot) {
                const int col = matrix.col_ind[slot];
                c_adj[col].push_back(global_row);
                c_handles[col].push_back(matrix.blk_handles[slot]);
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

        LegacyMatrixBuilder<T, Kernel> builder(graph_C, true);
        Matrix C = builder.materialize();

        #pragma omp parallel for
        for (int row = 0; row < n_cols; ++row) {
            const int a_cols = c_block_sizes[row];
            const int start = C.row_ptr[row];
            const int end = C.row_ptr[row + 1];

            for (int offset = 0; offset < (end - start); ++offset) {
                const uint64_t src_handle = c_handles[row][offset];
                const T* a_data = matrix.arena.get_ptr(src_handle);
                const int col_C_local = C.col_ind[start + offset];
                const int a_rows = C.graph->block_sizes[col_C_local];

                builder.write_transposed_conjugate_slot(C, start + offset, a_data, a_rows, a_cols);
            }
        }
        C.norms_valid = false;
        return C;
    }

    static Matrix distributed(const Matrix& matrix) {
        auto exchange = exchange_transpose_blocks(matrix);
        DistGraph* graph_C = construct_result_graph(
            matrix.graph->comm,
            matrix.graph->owned_global_indices,
            owned_block_sizes(*matrix.graph),
            exchange.adjacency,
            exchange.ghost_dims,
            "transpose");

        LegacyMatrixBuilder<T, Kernel> builder(graph_C, true);
        Matrix C = builder.materialize();

        int* meta_ptr = exchange.recv_meta.data();
        int* meta_end = exchange.recv_meta.data() + exchange.recv_meta.size();
        T* value_ptr = exchange.recv_values.data();
        while (meta_ptr < meta_end) {
            const int global_row = *meta_ptr++;
            const int global_col = *meta_ptr++;
            const int row_dim = *meta_ptr++;
            const int col_dim = *meta_ptr++;
            const size_t n_elem = static_cast<size_t>(row_dim) * col_dim;

            std::vector<T> block(n_elem);
            const int rows_A = col_dim;
            const int cols_A = row_dim;
            for (int col = 0; col < cols_A; ++col) {
                for (int row = 0; row < rows_A; ++row) {
                    T val = value_ptr[row + col * rows_A];
                    val = ConjHelper<T>::apply(val);
                    block[col + row * row_dim] = val;
                }
            }
            C.add_block(global_row, global_col, block.data(), row_dim, col_dim, AssemblyMode::ADD);
            value_ptr += n_elem;
        }
        C.assemble();
        return C;
    }
};

} // namespace detail

#endif // VBCSR_DETAIL_TRANSPOSE_OPS_HPP

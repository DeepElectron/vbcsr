#ifndef VBCSR_DETAIL_OPS_TRANSPOSE_HPP
#define VBCSR_DETAIL_OPS_TRANSPOSE_HPP

#include "../distributed/result_graph.hpp"

#include "../distributed/mpi_utils.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace vbcsr::detail {

template <typename T>
struct TransposeExchangeResult {
    std::vector<int> recv_meta;
    std::vector<T> recv_values;
    std::vector<std::vector<int>> adjacency;
    std::map<int, int> ghost_dims;
};

template <typename Matrix>
TransposeExchangeResult<typename Matrix::value_type> exchange_transpose_blocks(const Matrix& matrix) {
    using T = typename Matrix::value_type;

    TransposeExchangeResult<T> result;
    if (matrix.graph->size == 1) {
        return result;
    }

    const int size = matrix.graph->size;
    const int n_rows = static_cast<int>(matrix.row_ptr().size()) - 1;

    std::vector<size_t> send_counts(size, 0);
    std::vector<size_t> send_data_counts(size, 0);
    for (int i = 0; i < n_rows; ++i) {
        const int r_dim = matrix.graph->block_sizes[i];
        for (int k = matrix.row_ptr()[i]; k < matrix.row_ptr()[i + 1]; ++k) {
            const int g_col = matrix.graph->get_global_index(matrix.col_ind()[k]);
            const int owner = matrix.graph->find_owner(g_col);
            const int c_dim = matrix.graph->block_sizes[matrix.col_ind()[k]];
            send_counts[owner] += 4;
            send_data_counts[owner] += static_cast<size_t>(r_dim) * c_dim;
        }
    }

    std::vector<size_t> recv_counts(size);
    std::vector<size_t> recv_data_counts(size);
    MPI_Alltoall(send_counts.data(), sizeof(size_t), MPI_BYTE,
                 recv_counts.data(), sizeof(size_t), MPI_BYTE, matrix.graph->comm);
    MPI_Alltoall(send_data_counts.data(), sizeof(size_t), MPI_BYTE,
                 recv_data_counts.data(), sizeof(size_t), MPI_BYTE, matrix.graph->comm);

    std::vector<size_t> sdispls(size + 1, 0);
    std::vector<size_t> rdispls(size + 1, 0);
    std::vector<size_t> sdispls_data(size + 1, 0);
    std::vector<size_t> rdispls_data(size + 1, 0);
    for (int i = 0; i < size; ++i) {
        sdispls[i + 1] = sdispls[i] + send_counts[i];
        rdispls[i + 1] = rdispls[i] + recv_counts[i];
        sdispls_data[i + 1] = sdispls_data[i] + send_data_counts[i];
        rdispls_data[i + 1] = rdispls_data[i] + recv_data_counts[i];
    }

    std::vector<int> send_meta(sdispls[size]);
    std::vector<T> send_values(sdispls_data[size]);
    std::vector<size_t> current_counts(size, 0);
    std::vector<size_t> current_data_counts(size, 0);

    for (int i = 0; i < n_rows; ++i) {
        const int g_row = matrix.graph->owned_global_indices[i];
        const int r_dim = matrix.graph->block_sizes[i];
        for (int k = matrix.row_ptr()[i]; k < matrix.row_ptr()[i + 1]; ++k) {
            const int g_col = matrix.graph->get_global_index(matrix.col_ind()[k]);
            const int owner = matrix.graph->find_owner(g_col);
            const int c_dim = matrix.graph->block_sizes[matrix.col_ind()[k]];

            int* meta_ptr = send_meta.data() + sdispls[owner] + current_counts[owner];
            meta_ptr[0] = g_col;
            meta_ptr[1] = g_row;
            meta_ptr[2] = c_dim;
            meta_ptr[3] = r_dim;
            current_counts[owner] += 4;

            const size_t n_elem = static_cast<size_t>(r_dim) * c_dim;
            std::memcpy(send_values.data() + sdispls_data[owner] + current_data_counts[owner],
                        matrix.block_data(k),
                        n_elem * sizeof(T));
            current_data_counts[owner] += n_elem;
        }
    }

    result.recv_meta.resize(rdispls[size]);
    safe_alltoallv(send_meta.data(), send_counts, sdispls, MPI_INT,
                   result.recv_meta.data(), recv_counts, rdispls, MPI_INT, matrix.graph->comm);

    result.recv_values.resize(rdispls_data[size]);
    std::vector<size_t> send_data_bytes(size);
    std::vector<size_t> recv_data_bytes(size);
    std::vector<size_t> sdispls_data_bytes(size + 1);
    std::vector<size_t> rdispls_data_bytes(size + 1);
    for (int i = 0; i < size; ++i) {
        send_data_bytes[i] = send_data_counts[i] * sizeof(T);
        recv_data_bytes[i] = recv_data_counts[i] * sizeof(T);
        sdispls_data_bytes[i] = sdispls_data[i] * sizeof(T);
        rdispls_data_bytes[i] = rdispls_data[i] * sizeof(T);
    }
    sdispls_data_bytes[size] = sdispls_data[size] * sizeof(T);
    rdispls_data_bytes[size] = rdispls_data[size] * sizeof(T);

    safe_alltoallv(send_values.data(), send_data_bytes, sdispls_data_bytes, MPI_BYTE,
                   result.recv_values.data(), recv_data_bytes, rdispls_data_bytes, MPI_BYTE, matrix.graph->comm);

    result.adjacency.resize(matrix.graph->owned_global_indices.size());
    const int* meta_ptr = result.recv_meta.data();
    for (int i = 0; i < size; ++i) {
        const int* meta_end = meta_ptr + recv_counts[i];
        while (meta_ptr < meta_end) {
            const int g_row = *meta_ptr++;
            const int g_col = *meta_ptr++;
            meta_ptr++;
            const int c_dim = *meta_ptr++;

            auto row_it = matrix.graph->global_to_local.find(g_row);
            if (row_it != matrix.graph->global_to_local.end()) {
                result.adjacency[row_it->second].push_back(g_col);
                result.ghost_dims[g_col] = c_dim;
            }
        }
    }

    for (auto& neighbors : result.adjacency) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    return result;
}

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
        auto exchange = exchange_transpose_blocks(matrix);
        DistGraph* graph_C = construct_result_graph(
            matrix,
            exchange.adjacency,
            exchange.ghost_dims,
            "transpose");

        Matrix result(graph_C);
        result.owns_graph = true;
        result.graph->enable_matrix_lifetime_management();
        result.set_page_size(matrix.configured_page_size());
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
        auto exchange = exchange_transpose_blocks(matrix);
        DistGraph* graph_C = construct_result_graph(
            matrix,
            exchange.adjacency,
            exchange.ghost_dims,
            "transpose");

        Matrix result(graph_C);
        result.owns_graph = true;
        result.graph->enable_matrix_lifetime_management();
        result.set_page_size(matrix.configured_page_size());
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
        auto exchange = exchange_transpose_blocks(matrix);
        DistGraph* graph_C = construct_result_graph(
            matrix,
            exchange.adjacency,
            exchange.ghost_dims,
            "transpose");

        Matrix result(graph_C);
        result.owns_graph = true;
        result.graph->enable_matrix_lifetime_management();
        result.set_page_size(matrix.configured_page_size());
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

#endif // VBCSR_DETAIL_OPS_TRANSPOSE_HPP

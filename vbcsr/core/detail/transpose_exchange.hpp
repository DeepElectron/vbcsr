#ifndef VBCSR_DETAIL_TRANSPOSE_EXCHANGE_HPP
#define VBCSR_DETAIL_TRANSPOSE_EXCHANGE_HPP

#include "../mpi_utils.hpp"

#include <algorithm>
#include <cstring>
#include <map>
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

} // namespace vbcsr::detail

#endif

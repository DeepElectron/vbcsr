#ifndef VBCSR_DETAIL_SPMM_EXCHANGE_HPP
#define VBCSR_DETAIL_SPMM_EXCHANGE_HPP

#include "../mpi_utils.hpp"

#include <cstring>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace vbcsr {

struct BlockMeta {
    int col;
    double norm;
};

struct BlockID {
    int row;
    int col;

    bool operator<(const BlockID& other) const {
        if (row != other.row) {
            return row < other.row;
        }
        return col < other.col;
    }
};

namespace detail {

template <typename T>
struct GhostBlockRef {
    int col;
    const T* data;
    int c_dim;
    double norm;
};

template <typename T>
using GhostBlockData = std::map<BlockID, std::vector<T>>;

using GhostSizes = std::map<int, int>;
using GhostMetadata = std::map<int, std::vector<BlockMeta>>;

template <typename MatrixA, typename MatrixB>
GhostMetadata exchange_ghost_metadata(const MatrixA& A, const MatrixB& B) {
    if (A.graph->size == 1) {
        return {};
    }

    GhostMetadata metadata;
    const int size = A.graph->size;
    const int rank = A.graph->rank;

    std::set<int> needed_rows;
    const int n_rows = static_cast<int>(A.row_ptr().size()) - 1;
    for (int i = 0; i < n_rows; ++i) {
        for (int k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k) {
            const int global_col = A.graph->get_global_index(A.col_ind()[k]);
            if (B.graph->find_owner(global_col) != rank) {
                needed_rows.insert(global_col);
            }
        }
    }

    std::vector<int> send_req_counts(size, 0);
    for (int global_row : needed_rows) {
        const int owner = B.graph->find_owner(global_row);
        send_req_counts[owner]++;
    }

    std::vector<int> recv_req_counts(size);
    MPI_Alltoall(send_req_counts.data(), 1, MPI_INT, recv_req_counts.data(), 1, MPI_INT, A.graph->comm);

    std::vector<int> sdispls(size + 1, 0);
    std::vector<int> rdispls(size + 1, 0);
    for (int i = 0; i < size; ++i) {
        sdispls[i + 1] = sdispls[i] + send_req_counts[i];
        rdispls[i + 1] = rdispls[i] + recv_req_counts[i];
    }

    std::vector<int> send_req_buf(sdispls[size]);
    std::vector<int> current_req_counts(size, 0);
    for (int global_row : needed_rows) {
        const int owner = B.graph->find_owner(global_row);
        send_req_buf[sdispls[owner] + current_req_counts[owner]++] = global_row;
    }

    std::vector<int> recv_req_buf(rdispls[size]);
    MPI_Alltoallv(send_req_buf.data(), send_req_counts.data(), sdispls.data(), MPI_INT,
                  recv_req_buf.data(), recv_req_counts.data(), rdispls.data(), MPI_INT, A.graph->comm);

    const auto& b_norms = B.get_block_norms();
    std::vector<size_t> send_reply_bytes(size, 0);
    int* req_ptr = recv_req_buf.data();
    for (int i = 0; i < size; ++i) {
        int* req_end = req_ptr + recv_req_counts[i];
        while (req_ptr < req_end) {
            const int global_row = *req_ptr++;
            if (B.graph->global_to_local.count(global_row)) {
                const int local_row = B.graph->global_to_local.at(global_row);
                const int n_blocks = B.row_ptr()[local_row + 1] - B.row_ptr()[local_row];
                send_reply_bytes[i] += 2 * sizeof(int) + n_blocks * (sizeof(int) + sizeof(double));
            }
        }
    }

    std::vector<size_t> recv_reply_bytes(size);
    MPI_Alltoall(send_reply_bytes.data(), sizeof(size_t), MPI_BYTE,
                 recv_reply_bytes.data(), sizeof(size_t), MPI_BYTE, A.graph->comm);

    std::vector<size_t> sdispls_reply(size + 1, 0);
    std::vector<size_t> rdispls_reply(size + 1, 0);
    for (int i = 0; i < size; ++i) {
        sdispls_reply[i + 1] = sdispls_reply[i] + send_reply_bytes[i];
        rdispls_reply[i + 1] = rdispls_reply[i] + recv_reply_bytes[i];
    }

    std::vector<char> send_reply_blob(sdispls_reply[size]);
    req_ptr = recv_req_buf.data();
    for (int i = 0; i < size; ++i) {
        char* blob_ptr = send_reply_blob.data() + sdispls_reply[i];
        int* req_end = req_ptr + recv_req_counts[i];
        while (req_ptr < req_end) {
            const int global_row = *req_ptr++;
            if (B.graph->global_to_local.count(global_row)) {
                const int local_row = B.graph->global_to_local.at(global_row);
                const int start = B.row_ptr()[local_row];
                const int end = B.row_ptr()[local_row + 1];
                const int n_blocks = end - start;

                std::memcpy(blob_ptr, &global_row, sizeof(int));
                blob_ptr += sizeof(int);
                std::memcpy(blob_ptr, &n_blocks, sizeof(int));
                blob_ptr += sizeof(int);
                for (int k = start; k < end; ++k) {
                    const int col = B.graph->get_global_index(B.col_ind()[k]);
                    const double norm = b_norms[k];
                    std::memcpy(blob_ptr, &col, sizeof(int));
                    blob_ptr += sizeof(int);
                    std::memcpy(blob_ptr, &norm, sizeof(double));
                    blob_ptr += sizeof(double);
                }
            }
        }
    }

    std::vector<char> recv_reply_blob(rdispls_reply[size]);
    safe_alltoallv(send_reply_blob.data(), send_reply_bytes, sdispls_reply, MPI_BYTE,
                   recv_reply_blob.data(), recv_reply_bytes, rdispls_reply, MPI_BYTE, A.graph->comm);

    for (int i = 0; i < size; ++i) {
        char* blob_ptr = recv_reply_blob.data() + rdispls_reply[i];
        char* blob_end = recv_reply_blob.data() + rdispls_reply[i + 1];
        while (blob_ptr < blob_end) {
            int global_row = 0;
            int n_blocks = 0;
            std::memcpy(&global_row, blob_ptr, sizeof(int));
            blob_ptr += sizeof(int);
            std::memcpy(&n_blocks, blob_ptr, sizeof(int));
            blob_ptr += sizeof(int);
            auto& list = metadata[global_row];
            list.reserve(n_blocks);
            for (int k = 0; k < n_blocks; ++k) {
                BlockMeta meta;
                std::memcpy(&meta.col, blob_ptr, sizeof(int));
                blob_ptr += sizeof(int);
                std::memcpy(&meta.norm, blob_ptr, sizeof(double));
                blob_ptr += sizeof(double);
                list.push_back(meta);
            }
        }
    }

    return metadata;
}

template <typename Matrix>
std::pair<GhostBlockData<typename Matrix::value_type>, GhostSizes> fetch_ghost_blocks(
    const Matrix& matrix,
    const std::vector<BlockID>& required_blocks) {
    using T = typename Matrix::value_type;

    if (matrix.graph->size == 1) {
        return {{}, {}};
    }

    GhostBlockData<T> ghost_data;
    GhostSizes ghost_sizes;
    const int size = matrix.graph->size;

    std::vector<size_t> send_req_counts(size, 0);
    for (const auto& bid : required_blocks) {
        const int owner = matrix.graph->find_owner(bid.row);
        send_req_counts[owner] += 2 * sizeof(int);
    }

    std::vector<size_t> recv_req_counts(size);
    MPI_Alltoall(send_req_counts.data(), sizeof(size_t), MPI_BYTE,
                 recv_req_counts.data(), sizeof(size_t), MPI_BYTE, matrix.graph->comm);

    std::vector<size_t> sdispls(size + 1, 0);
    std::vector<size_t> rdispls(size + 1, 0);
    for (int i = 0; i < size; ++i) {
        sdispls[i + 1] = sdispls[i] + send_req_counts[i];
        rdispls[i + 1] = rdispls[i] + recv_req_counts[i];
    }

    std::vector<int> send_req_buf(sdispls[size] / sizeof(int));
    std::vector<size_t> current_req_counts(size, 0);
    for (const auto& bid : required_blocks) {
        const int owner = matrix.graph->find_owner(bid.row);
        int* ptr = send_req_buf.data() + (sdispls[owner] + current_req_counts[owner]) / sizeof(int);
        ptr[0] = bid.row;
        ptr[1] = bid.col;
        current_req_counts[owner] += 2 * sizeof(int);
    }

    std::vector<int> recv_req_buf(rdispls[size] / sizeof(int));
    safe_alltoallv(send_req_buf.data(), send_req_counts, sdispls, MPI_BYTE,
                   recv_req_buf.data(), recv_req_counts, rdispls, MPI_BYTE, matrix.graph->comm);

    std::vector<size_t> send_reply_bytes(size, 0);
    int* req_ptr = recv_req_buf.data();
    for (int i = 0; i < size; ++i) {
        int* req_end = req_ptr + recv_req_counts[i] / sizeof(int);
        while (req_ptr < req_end) {
            const int g_row = *req_ptr++;
            const int g_col = *req_ptr++;
            if (matrix.graph->global_to_local.count(g_row)) {
                const int l_row = matrix.graph->global_to_local.at(g_row);
                for (int k = matrix.row_ptr()[l_row]; k < matrix.row_ptr()[l_row + 1]; ++k) {
                    if (matrix.graph->get_global_index(matrix.col_ind()[k]) == g_col) {
                        const int r_dim = matrix.graph->block_sizes[l_row];
                        const int c_dim = matrix.graph->block_sizes[matrix.col_ind()[k]];
                        send_reply_bytes[i] += 4 * sizeof(int) + static_cast<size_t>(r_dim) * c_dim * sizeof(T);
                        break;
                    }
                }
            }
        }
    }

    std::vector<size_t> recv_reply_bytes(size);
    MPI_Alltoall(send_reply_bytes.data(), sizeof(size_t), MPI_BYTE,
                 recv_reply_bytes.data(), sizeof(size_t), MPI_BYTE, matrix.graph->comm);

    std::vector<size_t> sdispls_reply(size + 1, 0);
    std::vector<size_t> rdispls_reply(size + 1, 0);
    for (int i = 0; i < size; ++i) {
        sdispls_reply[i + 1] = sdispls_reply[i] + send_reply_bytes[i];
        rdispls_reply[i + 1] = rdispls_reply[i] + recv_reply_bytes[i];
    }

    std::vector<char> send_reply_blob(sdispls_reply[size]);
    req_ptr = recv_req_buf.data();
    for (int i = 0; i < size; ++i) {
        char* blob_ptr = send_reply_blob.data() + sdispls_reply[i];
        int* req_end = req_ptr + recv_req_counts[i] / sizeof(int);
        while (req_ptr < req_end) {
            const int g_row = *req_ptr++;
            const int g_col = *req_ptr++;
            if (matrix.graph->global_to_local.count(g_row)) {
                const int l_row = matrix.graph->global_to_local.at(g_row);
                for (int k = matrix.row_ptr()[l_row]; k < matrix.row_ptr()[l_row + 1]; ++k) {
                    if (matrix.graph->get_global_index(matrix.col_ind()[k]) == g_col) {
                        const int r_dim = matrix.graph->block_sizes[l_row];
                        const int c_dim = matrix.graph->block_sizes[matrix.col_ind()[k]];
                        const size_t n_elem = static_cast<size_t>(r_dim) * c_dim;

                        std::memcpy(blob_ptr, &g_row, sizeof(int));
                        blob_ptr += sizeof(int);
                        std::memcpy(blob_ptr, &g_col, sizeof(int));
                        blob_ptr += sizeof(int);
                        std::memcpy(blob_ptr, &r_dim, sizeof(int));
                        blob_ptr += sizeof(int);
                        std::memcpy(blob_ptr, &c_dim, sizeof(int));
                        blob_ptr += sizeof(int);
                        std::memcpy(blob_ptr, matrix.block_data(k), n_elem * sizeof(T));
                        blob_ptr += n_elem * sizeof(T);
                        break;
                    }
                }
            }
        }
    }

    std::vector<char> recv_reply_blob(rdispls_reply[size]);
    safe_alltoallv(send_reply_blob.data(), send_reply_bytes, sdispls_reply, MPI_BYTE,
                   recv_reply_blob.data(), recv_reply_bytes, rdispls_reply, MPI_BYTE, matrix.graph->comm);

    for (int i = 0; i < size; ++i) {
        char* blob_ptr = recv_reply_blob.data() + rdispls_reply[i];
        char* blob_end = recv_reply_blob.data() + rdispls_reply[i + 1];
        while (blob_ptr < blob_end) {
            int g_row = 0;
            int g_col = 0;
            int r_dim = 0;
            int c_dim = 0;
            std::memcpy(&g_row, blob_ptr, sizeof(int));
            blob_ptr += sizeof(int);
            std::memcpy(&g_col, blob_ptr, sizeof(int));
            blob_ptr += sizeof(int);
            std::memcpy(&r_dim, blob_ptr, sizeof(int));
            blob_ptr += sizeof(int);
            std::memcpy(&c_dim, blob_ptr, sizeof(int));
            blob_ptr += sizeof(int);

            const size_t n_elem = static_cast<size_t>(r_dim) * c_dim;
            std::vector<T> data(n_elem);
            std::memcpy(data.data(), blob_ptr, n_elem * sizeof(T));
            blob_ptr += n_elem * sizeof(T);

            ghost_data[{g_row, g_col}] = std::move(data);
            ghost_sizes[g_col] = c_dim;
        }
    }

    return {std::move(ghost_data), std::move(ghost_sizes)};
}

} // namespace detail
} // namespace vbcsr

#endif

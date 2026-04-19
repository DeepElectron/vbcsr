#ifndef VBCSR_DETAIL_DISTRIBUTED_BLOCK_PAYLOAD_EXCHANGE_HPP
#define VBCSR_DETAIL_DISTRIBUTED_BLOCK_PAYLOAD_EXCHANGE_HPP

#include "block_payload_types.hpp"

#include "mpi_utils.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace vbcsr::detail {

namespace block_payload_detail {

using RequiredColumnsByRow = std::map<int, std::vector<int>>;

inline void normalize_required_columns(RequiredColumnsByRow& required_cols_by_row) {
    for (auto& [row, cols] : required_cols_by_row) {
        (void)row;
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    }
}

inline RequiredColumnsByRow required_columns_from_blocks(const std::vector<BlockID>& required_blocks) {
    RequiredColumnsByRow required_cols_by_row;
    for (const auto& bid : required_blocks) {
        required_cols_by_row[bid.row].push_back(bid.col);
    }
    normalize_required_columns(required_cols_by_row);
    return required_cols_by_row;
}

inline RequiredColumnsByRow required_columns_from_batches(const std::vector<std::vector<int>>& batch_indices) {
    RequiredColumnsByRow required_cols_by_row;
    for (const auto& indices : batch_indices) {
        for (int row_gid : indices) {
            auto& cols = required_cols_by_row[row_gid];
            cols.insert(cols.end(), indices.begin(), indices.end());
        }
    }
    normalize_required_columns(required_cols_by_row);
    return required_cols_by_row;
}

template <typename Matrix>
void append_matching_local_blocks(
        const Matrix& matrix,
        int gid,
        const std::vector<int>& req_cols,
        FetchedBlockContext<typename Matrix::value_type>& ctx) {
    using T = typename Matrix::value_type;

    auto row_it = matrix.graph->global_to_local.find(gid);
    if (row_it == matrix.graph->global_to_local.end()) {
        return;
    }

    const int lid = row_it->second;
    ctx.row_sizes[gid] = matrix.graph->block_sizes[lid];

    const int start = matrix.row_ptr()[lid];
    const int end = matrix.row_ptr()[lid + 1];
    for (int slot = start; slot < end; ++slot) {
        const int col_lid = matrix.col_ind()[slot];
        const int col_gid = matrix.graph->get_global_index(col_lid);
        if (!std::binary_search(req_cols.begin(), req_cols.end(), col_gid)) {
            continue;
        }

        FetchedBlock<T> block;
        block.global_row = gid;
        block.global_col = col_gid;
        block.r_dim = matrix.graph->block_sizes[lid];
        block.c_dim = matrix.graph->block_sizes[col_lid];
        const size_t size = matrix.block_size_elements(slot);
        block.data.resize(size);
        std::memcpy(block.data.data(), matrix.block_data(slot), size * sizeof(T));
        ctx.blocks.push_back(std::move(block));
    }
}

template <typename Matrix>
size_t count_response_bytes(
        const Matrix& matrix,
        const int* req_buffer) {
    using T = typename Matrix::value_type;

    const int num_rows = req_buffer[0];
    size_t bytes = sizeof(int) + static_cast<size_t>(num_rows) * 2 * sizeof(int) + sizeof(int);

    const int* ptr = req_buffer + 1;
    for (int r = 0; r < num_rows; ++r) {
        const int gid = *ptr++;
        const int num_cols = *ptr++;
        const int* cols_begin = ptr;
        const int* cols_end = ptr + num_cols;
        ptr = cols_end;

        auto row_it = matrix.graph->global_to_local.find(gid);
        if (row_it == matrix.graph->global_to_local.end()) {
            continue;
        }

        const int lid = row_it->second;
        for (int slot = matrix.row_ptr()[lid]; slot < matrix.row_ptr()[lid + 1]; ++slot) {
            const int col_lid = matrix.col_ind()[slot];
            const int col_gid = matrix.graph->get_global_index(col_lid);
            if (!std::binary_search(cols_begin, cols_end, col_gid)) {
                continue;
            }
            bytes += 4 * sizeof(int) + matrix.block_size_elements(slot) * sizeof(T);
        }
    }

    return bytes;
}

template <typename Matrix>
void write_response(
        const Matrix& matrix,
        const int* req_buffer,
        char* out) {
    using T = typename Matrix::value_type;

    const int num_rows = req_buffer[0];
    const int* req_start = req_buffer + 1;
    char* ptr = out;

    std::memcpy(ptr, &num_rows, sizeof(int));
    ptr += sizeof(int);

    const int* req_ptr = req_start;
    for (int r = 0; r < num_rows; ++r) {
        const int gid = *req_ptr++;
        const int num_cols = *req_ptr++;
        req_ptr += num_cols;

        int size = 0;
        auto row_it = matrix.graph->global_to_local.find(gid);
        if (row_it != matrix.graph->global_to_local.end()) {
            size = matrix.graph->block_sizes[row_it->second];
        }
        std::memcpy(ptr, &gid, sizeof(int));
        ptr += sizeof(int);
        std::memcpy(ptr, &size, sizeof(int));
        ptr += sizeof(int);
    }
    char* num_blocks_ptr = ptr;
    ptr += sizeof(int);

    int total_blocks = 0;
    req_ptr = req_start;
    for (int r = 0; r < num_rows; ++r) {
        const int gid = *req_ptr++;
        const int num_cols = *req_ptr++;
        const int* cols_begin = req_ptr;
        const int* cols_end = req_ptr + num_cols;
        req_ptr = cols_end;

        auto row_it = matrix.graph->global_to_local.find(gid);
        if (row_it == matrix.graph->global_to_local.end()) {
            continue;
        }

        const int lid = row_it->second;
        for (int slot = matrix.row_ptr()[lid]; slot < matrix.row_ptr()[lid + 1]; ++slot) {
            const int col_lid = matrix.col_ind()[slot];
            const int col_gid = matrix.graph->get_global_index(col_lid);
            if (!std::binary_search(cols_begin, cols_end, col_gid)) {
                continue;
            }

            ++total_blocks;
            const int r_dim = matrix.graph->block_sizes[lid];
            const int c_dim = matrix.graph->block_sizes[col_lid];
            const size_t size = matrix.block_size_elements(slot);

            std::memcpy(ptr, &gid, sizeof(int));
            ptr += sizeof(int);
            std::memcpy(ptr, &col_gid, sizeof(int));
            ptr += sizeof(int);
            std::memcpy(ptr, &r_dim, sizeof(int));
            ptr += sizeof(int);
            std::memcpy(ptr, &c_dim, sizeof(int));
            ptr += sizeof(int);
            std::memcpy(ptr, matrix.block_data(slot), size * sizeof(T));
            ptr += size * sizeof(T);
        }
    }

    std::memcpy(num_blocks_ptr, &total_blocks, sizeof(int));
}

template <typename T>
void unpack_response(
        const char* ptr,
        FetchedBlockContext<T>& ctx) {
    int num_rows = 0;
    std::memcpy(&num_rows, ptr, sizeof(int));
    ptr += sizeof(int);
    for (int k = 0; k < num_rows; ++k) {
        int gid = 0;
        int size = 0;
        std::memcpy(&gid, ptr, sizeof(int));
        ptr += sizeof(int);
        std::memcpy(&size, ptr, sizeof(int));
        ptr += sizeof(int);
        ctx.row_sizes[gid] = size;
    }

    int num_blocks = 0;
    std::memcpy(&num_blocks, ptr, sizeof(int));
    ptr += sizeof(int);
    for (int k = 0; k < num_blocks; ++k) {
        FetchedBlock<T> block;
        std::memcpy(&block.global_row, ptr, sizeof(int));
        ptr += sizeof(int);
        std::memcpy(&block.global_col, ptr, sizeof(int));
        ptr += sizeof(int);
        std::memcpy(&block.r_dim, ptr, sizeof(int));
        ptr += sizeof(int);
        std::memcpy(&block.c_dim, ptr, sizeof(int));
        ptr += sizeof(int);
        block.data.resize(static_cast<size_t>(block.r_dim) * block.c_dim);
        std::memcpy(block.data.data(), ptr, block.data.size() * sizeof(T));
        ptr += block.data.size() * sizeof(T);
        ctx.blocks.push_back(std::move(block));
    }
}

template <typename Matrix>
FetchedBlockContext<typename Matrix::value_type> fetch_blocks_by_row_columns(
        const Matrix& matrix,
        const RequiredColumnsByRow& required_cols_by_row) {
    using T = typename Matrix::value_type;

    FetchedBlockContext<T> ctx;

    const int size = matrix.graph->size;
    const int rank = matrix.graph->rank;
    std::vector<size_t> send_counts(size, 0);
    std::vector<int> send_row_counts(size, 0);
    for (const auto& [gid, cols] : required_cols_by_row) {
        const int owner = matrix.graph->find_owner(gid);
        if (owner < 0 || owner >= size) {
            throw std::runtime_error("Block payload fetch request targets an invalid owner rank");
        }
        if (owner == rank) {
            append_matching_local_blocks(matrix, gid, cols, ctx);
            continue;
        }

        ++send_row_counts[owner];
        send_counts[owner] += 2 + cols.size();
    }
    for (int i = 0; i < size; ++i) {
        if (send_row_counts[i] > 0) {
            ++send_counts[i];
        }
    }

    std::vector<size_t> recv_counts(size);
    if (size > 1) {
        MPI_Alltoall(
            send_counts.data(),
            sizeof(size_t),
            MPI_BYTE,
            recv_counts.data(),
            sizeof(size_t),
            MPI_BYTE,
            matrix.graph->comm);
    } else {
        recv_counts = send_counts;
    }

    std::vector<size_t> sdispls(size + 1, 0);
    std::vector<size_t> rdispls(size + 1, 0);
    for (int i = 0; i < size; ++i) {
        sdispls[i + 1] = sdispls[i] + send_counts[i];
        rdispls[i + 1] = rdispls[i] + recv_counts[i];
    }

    std::vector<int> send_blob(sdispls[size]);
    std::vector<size_t> current_offsets = sdispls;
    for (int i = 0; i < size; ++i) {
        if (send_row_counts[i] > 0) {
            send_blob[current_offsets[i]++] = send_row_counts[i];
        }
    }
    for (const auto& [gid, cols] : required_cols_by_row) {
        const int owner = matrix.graph->find_owner(gid);
        if (owner == rank) {
            continue;
        }

        size_t& offset = current_offsets[owner];
        send_blob[offset++] = gid;
        send_blob[offset++] = static_cast<int>(cols.size());
        for (int col : cols) {
            send_blob[offset++] = col;
        }
    }
    // Request format per rank: [num_rows][row_gid][num_cols][col_1][col_2]...

    std::vector<int> recv_blob(rdispls[size]);
    if (size > 1) {
        safe_alltoallv(
            send_blob.data(),
            send_counts,
            sdispls,
            MPI_INT,
            recv_blob.data(),
            recv_counts,
            rdispls,
            MPI_INT,
            matrix.graph->comm);
    } else {
        recv_blob = send_blob;
    }

    std::vector<size_t> resp_send_counts(size, 0);
    for (int i = 0; i < size; ++i) {
        if (recv_counts[i] == 0) {
            continue;
        }
        resp_send_counts[i] = count_response_bytes(
            matrix,
            recv_blob.data() + rdispls[i]);
    }

    std::vector<size_t> resp_recv_counts(size);
    if (size > 1) {
        MPI_Alltoall(
            resp_send_counts.data(),
            sizeof(size_t),
            MPI_BYTE,
            resp_recv_counts.data(),
            sizeof(size_t),
            MPI_BYTE,
            matrix.graph->comm);
    } else {
        resp_recv_counts = resp_send_counts;
    }

    std::vector<size_t> resp_sdispls(size + 1, 0);
    std::vector<size_t> resp_rdispls(size + 1, 0);
    for (int i = 0; i < size; ++i) {
        resp_sdispls[i + 1] = resp_sdispls[i] + resp_send_counts[i];
        resp_rdispls[i + 1] = resp_rdispls[i] + resp_recv_counts[i];
    }

    std::vector<char> resp_send_blob(resp_sdispls[size]);
    for (int i = 0; i < size; ++i) {
        if (recv_counts[i] == 0) {
            continue;
        }
        write_response(
            matrix,
            recv_blob.data() + rdispls[i],
            resp_send_blob.data() + resp_sdispls[i]);
    }

    std::vector<char> resp_recv_blob(resp_rdispls[size]);
    if (size > 1) {
        safe_alltoallv(
            resp_send_blob.data(),
            resp_send_counts,
            resp_sdispls,
            MPI_BYTE,
            resp_recv_blob.data(),
            resp_recv_counts,
            resp_rdispls,
            MPI_BYTE,
            matrix.graph->comm);
    } else {
        resp_recv_blob = resp_send_blob;
    }

    for (int i = 0; i < size; ++i) {
        if (resp_recv_counts[i] == 0) {
            continue;
        }
        unpack_response(resp_recv_blob.data() + resp_rdispls[i], ctx);
    }

    return ctx;
}

} // namespace block_payload_detail

template <typename Matrix>
FetchedBlockContext<typename Matrix::value_type> fetch_required_block_payloads(
    const Matrix& matrix,
    const std::vector<BlockID>& required_blocks) {
    return block_payload_detail::fetch_blocks_by_row_columns(
        matrix,
        block_payload_detail::required_columns_from_blocks(required_blocks));
}

template <typename Matrix>
FetchedBlockContext<typename Matrix::value_type> fetch_batched_block_payloads(
    const Matrix& matrix,
    const std::vector<std::vector<int>>& batch_indices) {
    return block_payload_detail::fetch_blocks_by_row_columns(
        matrix,
        block_payload_detail::required_columns_from_batches(batch_indices));
}

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_DISTRIBUTED_BLOCK_PAYLOAD_EXCHANGE_HPP

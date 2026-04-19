#ifndef VBCSR_DETAIL_DISTRIBUTED_PLANS_HPP
#define VBCSR_DETAIL_DISTRIBUTED_PLANS_HPP

#include "distributed_result_graph.hpp"
#include "spmm_exchange.hpp"
#include "transpose_exchange.hpp"

#include "../mpi_utils.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>
#include <omp.h>

namespace vbcsr::detail {

struct SymbolicMultiplyResult {
    std::vector<int> c_row_ptr;
    std::vector<int> c_col_ind;
    std::vector<BlockID> required_blocks;
};

template <typename T>
struct FetchedBlock {
    int global_row;
    int global_col;
    int r_dim;
    int c_dim;
    std::vector<T> data;
};

template <typename T>
struct FetchedBlockContext {
    std::vector<FetchedBlock<T>> blocks;
    std::map<int, int> row_sizes;
};

template <typename MatrixA, typename MatrixB>
class RowMetadataExchangePlan {
public:
    static RowMetadataExchangePlan build(const MatrixA& A, const MatrixB& B) {
        RowMetadataExchangePlan plan;
        plan.metadata_ = exchange_ghost_metadata(A, B);
        return plan;
    }

    const GhostMetadata& metadata() const {
        return metadata_;
    }

private:
    GhostMetadata metadata_;
};

template <typename MatrixA, typename MatrixB>
SymbolicMultiplyResult symbolic_multiply_filtered(
    const MatrixA& A,
    const MatrixB& B,
    const GhostMetadata& meta,
    double threshold) {
    SymbolicMultiplyResult res;
    const int n_rows = static_cast<int>(A.row_ptr().size()) - 1;
    res.c_row_ptr.resize(n_rows + 1);
    res.c_row_ptr[0] = 0;

    const auto& A_norms = A.get_block_norms();
    const auto& B_local_norms = B.get_block_norms();

    std::vector<std::vector<int>> thread_cols(n_rows);
    const int max_threads = omp_get_max_threads();
    std::vector<std::set<BlockID>> thread_required(max_threads);

    struct SymbolicHashEntry {
        int key;
        double value;
        int tag;
    };
    const size_t HASH_SIZE = 8192;
    const size_t HASH_MASK = HASH_SIZE - 1;
    const size_t MAX_ROW_NNZ = static_cast<size_t>(HASH_SIZE * 0.7);

    std::vector<std::vector<SymbolicHashEntry>> thread_tables(
        max_threads,
        std::vector<SymbolicHashEntry>(HASH_SIZE, {-1, 0.0, 0}));
    std::vector<std::vector<int>> thread_touched(max_threads);
    std::vector<int> thread_tags(max_threads, 0);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& table = thread_tables[tid];
        auto& touched = thread_touched[tid];
        int& tag = thread_tags[tid];

        #pragma omp for
        for (int row = 0; row < n_rows; ++row) {
            ++tag;
            if (tag == 0) {
                for (auto& entry : table) {
                    entry.tag = 0;
                }
                tag = 1;
            }
            touched.clear();

            const int start = A.row_ptr()[row];
            const int end = A.row_ptr()[row + 1];
            for (int a_slot = start; a_slot < end; ++a_slot) {
                const int global_col_A = A.graph->get_global_index(A.col_ind()[a_slot]);
                const double norm_A = A_norms[a_slot];

                auto process_block = [&](int global_col_B, double norm_B) {
                    size_t h = static_cast<size_t>(global_col_B) & HASH_MASK;
                    size_t count = 0;
                    while (table[h].tag == tag) {
                        if (table[h].key == global_col_B) {
                            table[h].value += norm_A * norm_B;
                            return;
                        }
                        h = (h + 1) & HASH_MASK;
                        if (++count > HASH_SIZE) {
                            throw std::runtime_error("Hash table full in symbolic phase");
                        }
                    }
                    if (touched.size() > MAX_ROW_NNZ) {
                        throw std::runtime_error("Row density exceeds symbolic hash table capacity");
                    }
                    table[h] = {global_col_B, norm_A * norm_B, tag};
                    touched.push_back(static_cast<int>(h));
                };

                if (A.graph->find_owner(global_col_A) == A.graph->rank) {
                    const int local_row_B = B.graph->global_to_local.at(global_col_A);
                    const int start_B = B.row_ptr()[local_row_B];
                    const int end_B = B.row_ptr()[local_row_B + 1];
                    for (int b_slot = start_B; b_slot < end_B; ++b_slot) {
                        process_block(B.graph->get_global_index(B.col_ind()[b_slot]), B_local_norms[b_slot]);
                    }
                } else {
                    auto it = meta.find(global_col_A);
                    if (it != meta.end()) {
                        for (const auto& block_meta : it->second) {
                            process_block(block_meta.col, block_meta.norm);
                        }
                    }
                }
            }

            for (int h_idx : touched) {
                if (table[h_idx].value > threshold) {
                    thread_cols[row].push_back(table[h_idx].key);
                }
            }
            std::sort(thread_cols[row].begin(), thread_cols[row].end());
        }
    }

    for (int row = 0; row < n_rows; ++row) {
        res.c_col_ind.insert(res.c_col_ind.end(), thread_cols[row].begin(), thread_cols[row].end());
        res.c_row_ptr[row + 1] = static_cast<int>(res.c_col_ind.size());
    }

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();

        #pragma omp for
        for (int row = 0; row < n_rows; ++row) {
            const int c_start = res.c_row_ptr[row];
            const int c_end = res.c_row_ptr[row + 1];
            if (c_start == c_end) {
                continue;
            }

            const int start = A.row_ptr()[row];
            const int end = A.row_ptr()[row + 1];
            for (int a_slot = start; a_slot < end; ++a_slot) {
                const int global_col_A = A.graph->get_global_index(A.col_ind()[a_slot]);
                if (A.graph->find_owner(global_col_A) == A.graph->rank) {
                    continue;
                }

                auto it = meta.find(global_col_A);
                if (it == meta.end()) {
                    continue;
                }
                for (const auto& block_meta : it->second) {
                    if (std::binary_search(
                            res.c_col_ind.begin() + c_start,
                            res.c_col_ind.begin() + c_end,
                            block_meta.col)) {
                        thread_required[tid].insert({global_col_A, block_meta.col});
                    }
                }
            }
        }
    }

    std::set<BlockID> final_required;
    for (auto& required : thread_required) {
        final_required.insert(required.begin(), required.end());
    }
    res.required_blocks.assign(final_required.begin(), final_required.end());

    return res;
}

template <typename Matrix>
class BlockPayloadExchangePlan {
public:
    using T = typename Matrix::value_type;

    static BlockPayloadExchangePlan fetch_required(const Matrix& matrix, const std::vector<BlockID>& required_blocks) {
        BlockPayloadExchangePlan plan;
        // unpack the tuple
        std::tie(plan.ghost_data_, plan.ghost_sizes_) = fetch_ghost_blocks(matrix, required_blocks);
        return plan; // plan is a wrapper that have ghost_data and sizes, ghost data is a map from block id to the block data
    }

    static FetchedBlockContext<T> fetch_batch(const Matrix& matrix, const std::vector<std::vector<int>>& batch_indices) {
        return fetch_blocks_impl(matrix, batch_indices);
    }
    // the fetchedBlockContext is another type of block data format

    const GhostBlockData<T>& ghost_data() const {
        return ghost_data_;
    }

    const GhostSizes& ghost_sizes() const {
        return ghost_sizes_;
    }

    std::pair<GhostBlockData<T>, GhostSizes> release() && {
        return {std::move(ghost_data_), std::move(ghost_sizes_)};
    }

private:
    GhostBlockData<T> ghost_data_;
    GhostSizes ghost_sizes_;

    static void serve_fetch_requests(
        const Matrix& matrix,
        const char* req_buffer,
        std::vector<char>& resp_buffer) {
        const int* ptr = reinterpret_cast<const int*>(req_buffer);
        const int num_rows = *ptr++;
        const int* req_start = ptr;

        const size_t header_size = sizeof(int) + static_cast<size_t>(num_rows) * 2 * sizeof(int) + sizeof(int);
        resp_buffer.resize(header_size);
        char* buf_ptr = resp_buffer.data();

        std::memcpy(buf_ptr, &num_rows, sizeof(int));
        buf_ptr += sizeof(int);

        ptr = req_start;
        for (int r = 0; r < num_rows; ++r) {
            const int gid = *ptr++;
            const int num_cols = *ptr++;
            ptr += num_cols;

            int size = 0;
            if (matrix.graph->global_to_local.count(gid)) {
                const int lid = matrix.graph->global_to_local.at(gid);
                size = matrix.graph->block_sizes[lid];
            }
            std::memcpy(buf_ptr, &gid, sizeof(int));
            buf_ptr += sizeof(int);
            std::memcpy(buf_ptr, &size, sizeof(int));
            buf_ptr += sizeof(int);
        }

        int total_blocks = 0;
        ptr = req_start;
        for (int r = 0; r < num_rows; ++r) {
            const int gid = *ptr++;
            const int num_cols = *ptr++;
            std::set<int> req_cols(ptr, ptr + num_cols);
            ptr += num_cols;

            if (!matrix.graph->global_to_local.count(gid)) {
                continue;
            }
            const int lid = matrix.graph->global_to_local.at(gid);
            const int start = matrix.row_ptr()[lid];
            const int end = matrix.row_ptr()[lid + 1];

            for (int slot = start; slot < end; ++slot) {
                const int col_lid = matrix.col_ind()[slot];
                const int col_gid = matrix.graph->get_global_index(col_lid);
                if (!req_cols.count(col_gid)) {
                    continue;
                }

                ++total_blocks;
                const int r_dim = matrix.graph->block_sizes[lid];
                const int c_dim = matrix.graph->block_sizes[col_lid];
                const size_t size = matrix.block_size_elements(slot);

                const size_t old_size = resp_buffer.size();
                resp_buffer.resize(old_size + 4 * sizeof(int) + size * sizeof(T));
                char* block_ptr = resp_buffer.data() + old_size;

                std::memcpy(block_ptr, &gid, sizeof(int));
                block_ptr += sizeof(int);
                std::memcpy(block_ptr, &col_gid, sizeof(int));
                block_ptr += sizeof(int);
                std::memcpy(block_ptr, &r_dim, sizeof(int));
                block_ptr += sizeof(int);
                std::memcpy(block_ptr, &c_dim, sizeof(int));
                block_ptr += sizeof(int);
                std::memcpy(block_ptr, matrix.block_data(slot), size * sizeof(T));
            }
        }

        const size_t num_blocks_offset = sizeof(int) + static_cast<size_t>(num_rows) * 2 * sizeof(int);
        std::memcpy(resp_buffer.data() + num_blocks_offset, &total_blocks, sizeof(int));
    }

    static FetchedBlockContext<T> fetch_blocks_impl(
        const Matrix& matrix,
        const std::vector<std::vector<int>>& batch_indices) {
        FetchedBlockContext<T> ctx;

        std::set<int> all_required_rows;
        for (const auto& indices : batch_indices) {
            all_required_rows.insert(indices.begin(), indices.end());
        }

        std::vector<int> local_rows;
        std::map<int, std::vector<int>> remote_rows_by_rank;
        for (int gid : all_required_rows) {
            const int owner = matrix.graph->find_owner(gid);
            if (owner == matrix.graph->rank) {
                local_rows.push_back(gid);
            } else {
                remote_rows_by_rank[owner].push_back(gid);
            }
        }

        std::map<int, std::set<int>> required_cols_per_row;
        for (const auto& indices : batch_indices) {
            for (int row_gid : indices) {
                required_cols_per_row[row_gid].insert(indices.begin(), indices.end());
            }
        }

        for (int gid : local_rows) {
            if (!matrix.graph->global_to_local.count(gid)) {
                continue;
            }
            const int lid = matrix.graph->global_to_local.at(gid);
            ctx.row_sizes[gid] = matrix.graph->block_sizes[lid];

            const int start = matrix.row_ptr()[lid];
            const int end = matrix.row_ptr()[lid + 1];
            const auto& req_cols = required_cols_per_row[gid];

            for (int slot = start; slot < end; ++slot) {
                const int col_lid = matrix.col_ind()[slot];
                const int col_gid = matrix.graph->get_global_index(col_lid);
                if (!req_cols.count(col_gid)) {
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

        std::vector<size_t> send_counts(matrix.graph->size, 0);
        std::vector<std::vector<int>> send_buffers(matrix.graph->size);
        for (auto& [target, rows] : remote_rows_by_rank) {
            send_buffers[target].push_back(static_cast<int>(rows.size()));
            for (int gid : rows) {
                send_buffers[target].push_back(gid);
                const auto& cols = required_cols_per_row[gid];
                send_buffers[target].push_back(static_cast<int>(cols.size()));
                send_buffers[target].insert(send_buffers[target].end(), cols.begin(), cols.end());
            }
            send_counts[target] = send_buffers[target].size() * sizeof(int);
        }

        std::vector<size_t> recv_counts(matrix.graph->size);
        if (matrix.graph->size > 1) {
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

        std::vector<size_t> sdispls(matrix.graph->size + 1, 0);
        std::vector<size_t> rdispls(matrix.graph->size + 1, 0);
        for (int i = 0; i < matrix.graph->size; ++i) {
            sdispls[i + 1] = sdispls[i] + send_counts[i];
            rdispls[i + 1] = rdispls[i] + recv_counts[i];
        }

        std::vector<char> send_blob(sdispls[matrix.graph->size]);
        for (int i = 0; i < matrix.graph->size; ++i) {
            if (!send_buffers[i].empty()) {
                std::memcpy(
                    send_blob.data() + sdispls[i],
                    send_buffers[i].data(),
                    send_buffers[i].size() * sizeof(int));
            }
        }

        std::vector<char> recv_blob(rdispls[matrix.graph->size]);
        if (matrix.graph->size > 1) {
            safe_alltoallv(
                send_blob.data(),
                send_counts,
                sdispls,
                MPI_BYTE,
                recv_blob.data(),
                recv_counts,
                rdispls,
                MPI_BYTE,
                matrix.graph->comm);
        } else {
            recv_blob = send_blob;
        }

        std::vector<std::vector<char>> resp_buffers(matrix.graph->size);
        std::vector<size_t> resp_send_counts(matrix.graph->size, 0);
        for (int i = 0; i < matrix.graph->size; ++i) {
            if (recv_counts[i] == 0) {
                continue;
            }
            serve_fetch_requests(matrix, recv_blob.data() + rdispls[i], resp_buffers[i]);
            resp_send_counts[i] = resp_buffers[i].size();
        }

        std::vector<size_t> resp_recv_counts(matrix.graph->size);
        if (matrix.graph->size > 1) {
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

        std::vector<size_t> resp_sdispls(matrix.graph->size + 1, 0);
        std::vector<size_t> resp_rdispls(matrix.graph->size + 1, 0);
        for (int i = 0; i < matrix.graph->size; ++i) {
            resp_sdispls[i + 1] = resp_sdispls[i] + resp_send_counts[i];
            resp_rdispls[i + 1] = resp_rdispls[i] + resp_recv_counts[i];
        }

        std::vector<char> resp_send_blob(resp_sdispls[matrix.graph->size]);
        for (int i = 0; i < matrix.graph->size; ++i) {
            if (!resp_buffers[i].empty()) {
                std::memcpy(
                    resp_send_blob.data() + resp_sdispls[i],
                    resp_buffers[i].data(),
                    resp_buffers[i].size());
            }
        }

        std::vector<char> resp_recv_blob(resp_rdispls[matrix.graph->size]);
        if (matrix.graph->size > 1) {
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

        for (int i = 0; i < matrix.graph->size; ++i) {
            if (resp_recv_counts[i] == 0) {
                continue;
            }

            const char* ptr = resp_recv_blob.data() + resp_rdispls[i];
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

        return ctx;
    }
};

template <typename Matrix>
class ResultAssemblyPlan {
public:
    using T = typename Matrix::value_type;

    static ResultAssemblyPlan for_transpose(
        std::vector<std::vector<int>> adjacency,
        GhostSizes ghost_sizes) {
        ResultAssemblyPlan plan;
        plan.adjacency_ = std::move(adjacency);
        plan.ghost_sizes_ = std::move(ghost_sizes);
        return plan;
    }

    static ResultAssemblyPlan for_spmm(
        const Matrix& matrix,
        const SymbolicMultiplyResult& symbolic,
        const GhostMetadata& meta,
        BlockPayloadExchangePlan<Matrix>&& payload_plan) {
        auto [ghost_data, ghost_sizes] = std::move(payload_plan).release();
        return for_spmm(matrix, symbolic, meta, std::move(ghost_data), std::move(ghost_sizes));
    }

    static ResultAssemblyPlan for_spmm(
        const Matrix& matrix,
        const SymbolicMultiplyResult& symbolic,
        const GhostMetadata& meta,
        GhostBlockData<T>&& ghost_data,
        GhostSizes&& ghost_sizes) {
        ResultAssemblyPlan plan;
        plan.ghost_data_ = std::move(ghost_data);
        plan.ghost_sizes_ = std::move(ghost_sizes);

        const int n_rows = static_cast<int>(matrix.graph->owned_global_indices.size());
        plan.adjacency_.resize(n_rows);
        for (int row = 0; row < n_rows; ++row) {
            for (int slot = symbolic.c_row_ptr[row]; slot < symbolic.c_row_ptr[row + 1]; ++slot) {
                plan.adjacency_[row].push_back(symbolic.c_col_ind[slot]);
            }
        }

        for (const auto& [bid, data] : plan.ghost_data_) {
            double norm = 0.0;
            auto meta_it = meta.find(bid.row);
            if (meta_it != meta.end()) {
                for (const auto& block_meta : meta_it->second) {
                    if (block_meta.col == bid.col) {
                        norm = block_meta.norm;
                        break;
                    }
                }
            }
            plan.ghost_rows_[bid.row].push_back(
                {bid.col, data.data(), plan.ghost_sizes_.at(bid.col), norm});
        }

        return plan;
    }

    DistGraph* construct_result_graph(const Matrix& matrix, const char* context) const {
        return detail::construct_result_graph(
            matrix.graph->comm,
            matrix.graph->owned_global_indices,
            owned_block_sizes(*matrix.graph),
            adjacency_,
            ghost_sizes_,
            context);
    }

    const std::vector<std::vector<int>>& adjacency() const {
        return adjacency_;
    }

    const std::map<int, std::vector<GhostBlockRef<T>>>& ghost_rows() const {
        return ghost_rows_;
    }

    const GhostSizes& ghost_sizes() const {
        return ghost_sizes_;
    }

private:
    std::vector<std::vector<int>> adjacency_;
    GhostBlockData<T> ghost_data_;
    GhostSizes ghost_sizes_;
    std::map<int, std::vector<GhostBlockRef<T>>> ghost_rows_;
};

template <typename T>
class TransposePayloadExchangePlan {
public:
    template <typename Matrix>
    static TransposePayloadExchangePlan build(const Matrix& matrix) {
        TransposePayloadExchangePlan plan;
        auto exchange = exchange_transpose_blocks(matrix);
        plan.recv_meta_ = std::move(exchange.recv_meta);
        plan.recv_values_ = std::move(exchange.recv_values);
        plan.adjacency_ = std::move(exchange.adjacency);
        plan.ghost_sizes_ = std::move(exchange.ghost_dims);
        return plan;
    }

    const std::vector<int>& recv_meta() const {
        return recv_meta_;
    }

    const std::vector<T>& recv_values() const {
        return recv_values_;
    }

    const std::vector<std::vector<int>>& adjacency() const {
        return adjacency_;
    }

    const GhostSizes& ghost_sizes() const {
        return ghost_sizes_;
    }

private:
    std::vector<int> recv_meta_;
    std::vector<T> recv_values_;
    std::vector<std::vector<int>> adjacency_;
    GhostSizes ghost_sizes_;
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_DISTRIBUTED_PLANS_HPP

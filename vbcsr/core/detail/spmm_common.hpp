#ifndef VBCSR_DETAIL_SPMM_COMMON_HPP
#define VBCSR_DETAIL_SPMM_COMMON_HPP

#include "block_payload_types.hpp"
#include "../mpi_utils.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>
#include <omp.h>

namespace vbcsr {

struct BlockMeta {
    int col;
    double norm;
};

namespace detail {

struct SymbolicMultiplyResult {
    std::vector<int> c_row_ptr;
    std::vector<int> c_col_ind;
    std::vector<BlockID> required_blocks;
};

template <typename T>
struct GhostBlockRef {
    int col;
    const T* data;
    int c_dim;
    double norm;
};

using GhostSizes = std::map<int, int>;
using GhostMetadata = std::map<int, std::vector<BlockMeta>>;

template <typename T>
struct SpMMGhostBlocks {
    std::vector<FetchedBlock<T>> owned_blocks;
    GhostSizes sizes;
    std::map<int, std::vector<GhostBlockRef<T>>> rows;
};

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
std::vector<std::vector<int>> build_spmm_result_adjacency(
    const Matrix& matrix,
    const SymbolicMultiplyResult& symbolic) {
    const int n_rows = static_cast<int>(matrix.graph->owned_global_indices.size());
    std::vector<std::vector<int>> adjacency(n_rows);
    for (int row = 0; row < n_rows; ++row) {
        for (int slot = symbolic.c_row_ptr[row]; slot < symbolic.c_row_ptr[row + 1]; ++slot) {
            adjacency[row].push_back(symbolic.c_col_ind[slot]);
        }
    }
    return adjacency;
}

template <typename T>
SpMMGhostBlocks<T> build_spmm_ghost_blocks(
    const GhostMetadata& metadata,
    FetchedBlockContext<T>&& payload_ctx) {
    SpMMGhostBlocks<T> ghost_blocks;
    ghost_blocks.owned_blocks = std::move(payload_ctx.blocks);

    for (const auto& block : ghost_blocks.owned_blocks) {
        double norm = 0.0;
        auto meta_it = metadata.find(block.global_row);
        if (meta_it != metadata.end()) {
            for (const auto& block_meta : meta_it->second) {
                if (block_meta.col == block.global_col) {
                    norm = block_meta.norm;
                    break;
                }
            }
        }

        ghost_blocks.sizes[block.global_col] = block.c_dim;
        ghost_blocks.rows[block.global_row].push_back(
            {block.global_col, block.data.data(), block.c_dim, norm});
    }

    return ghost_blocks;
}

} // namespace detail
} // namespace vbcsr

#endif // VBCSR_DETAIL_SPMM_COMMON_HPP

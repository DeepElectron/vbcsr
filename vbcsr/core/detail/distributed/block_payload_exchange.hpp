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

// Locally-owned blocks are handed back as POINTERS INTO THE MATRIX -- the
// caller holds the operand for the duration of its product (the contract
// stated at FetchedBlockRef), so copying them was pure waste.
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

        FetchedBlockRef<T> block;
        block.global_row = gid;
        block.global_col = col_gid;
        block.r_dim = matrix.graph->block_sizes[lid];
        block.c_dim = matrix.graph->block_sizes[col_lid];
        block.data = matrix.block_data(slot);
        ctx.blocks.push_back(block);
    }
}

/// Bytes AND blocks one peer's response holds.
///
/// The count is needed before a byte is packed: `num_blocks` sits in the
/// header, ahead of the blocks it counts, so a sender that streams the
/// response out cannot discover it on the way past.
template <typename Matrix>
size_t count_response_bytes(
        const Matrix& matrix,
        const int* req_buffer,
        int* out_blocks = nullptr) {
    using T = typename Matrix::value_type;

    const int num_rows = req_buffer[0];
    size_t bytes = sizeof(int) + static_cast<size_t>(num_rows) * 2 * sizeof(int) + sizeof(int);
    int blocks = 0;

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
            ++blocks;
        }
    }

    if (out_blocks != nullptr) *out_blocks = blocks;
    return bytes;
}

/// Response slice size, in bytes.
///
/// Both sides must agree, and they do it without a message: the receiver knows
/// each peer's total from the count exchange, so as long as the slice size is
/// the same number everywhere, slice k is the same byte range on both sides.
/// The env override exists for tests that need boundaries to fall inside a
/// block; an Allreduce below makes a rank that sets it differently harmless
/// rather than corrupting.
inline size_t response_slice_bytes() {
    static const size_t bytes = [] {
        const char* env = std::getenv("VBCSR_RESPONSE_SLICE_KB");
        if (env != nullptr) {
            const long long kb = std::atoll(env);
            if (kb > 0) return static_cast<size_t>(kb) << 10;
        }
        return size_t(32) << 20;
    }();
    return bytes;
}

/// Tag for response payloads, distinct from the request exchange's.
inline constexpr int kResponsePayloadTag = 1;

/// Packs one peer's response as a byte stream that can be PAUSED.
///
/// Two things force this shape. The response must not exist whole -- packing
/// it whole put the served blob beside the received one for the length of the
/// exchange, which is why the halo budget used to be halved. And the pack must
/// be resumable, because a rank has to keep every peer supplied at once: the
/// exchange this replaced posted all its sends together, and finishing one
/// peer before starting the next turns a full all-to-all into a queue of
/// point-to-point transfers -- every rank shipping to peer 0 while the rest
/// idle, then all to peer 1. On a network that is an incast, and it cost 3-4x
/// on the fetches large enough to show it.
///
/// So `emit` fills a caller's buffer with the next `capacity` bytes and
/// remembers where it stopped, down to the middle of a block payload. The cut
/// points never reach the receiver, which walks the reassembled segment.
template <typename Matrix>
class ResponseWalker {
public:
    using T = typename Matrix::value_type;

    ResponseWalker(const Matrix& matrix, const int* req_buffer, int total_blocks)
        : matrix_(&matrix), num_rows_(req_buffer[0]) {
        // The prelude -- row count, the row/size table, the block count -- is
        // staged whole because it is small (8 bytes a row) and because
        // total_blocks has to be written ahead of the blocks it counts.
        prelude_.resize(sizeof(int) + static_cast<size_t>(num_rows_) * 2 * sizeof(int) +
                        sizeof(int));
        char* out = prelude_.data();
        std::memcpy(out, &num_rows_, sizeof(int));
        out += sizeof(int);
        const int* req_ptr = req_buffer + 1;
        for (int r = 0; r < num_rows_; ++r) {
            const int gid = *req_ptr++;
            const int num_cols = *req_ptr++;
            req_ptr += num_cols;
            int size = 0;
            auto row_it = matrix.graph->global_to_local.find(gid);
            if (row_it != matrix.graph->global_to_local.end()) {
                size = matrix.graph->block_sizes[row_it->second];
            }
            std::memcpy(out, &gid, sizeof(int));
            out += sizeof(int);
            std::memcpy(out, &size, sizeof(int));
            out += sizeof(int);
        }
        std::memcpy(out, &total_blocks, sizeof(int));

        req_ptr_ = req_buffer + 1;
    }

    bool done() const { return done_; }

    /// Next `capacity` bytes of this response. Returns fewer only when the
    /// response is finished, so every slice but the last is exactly full --
    /// which is what lets the receiver post its slices from the total alone.
    size_t emit(char* out, size_t capacity) {
        size_t written = 0;
        if (prelude_off_ < prelude_.size()) {
            const size_t take = std::min(capacity, prelude_.size() - prelude_off_);
            std::memcpy(out, prelude_.data() + prelude_off_, take);
            prelude_off_ += take;
            written += take;
        }
        while (written < capacity) {
            if (head_off_ == head_len_ && pay_off_ == pay_len_ && !advance()) {
                done_ = true;
                break;
            }
            if (head_off_ < head_len_) {
                const size_t take = std::min(capacity - written, head_len_ - head_off_);
                std::memcpy(out + written, head_ + head_off_, take);
                head_off_ += take;
                written += take;
                continue;
            }
            const size_t take = std::min(capacity - written, pay_len_ - pay_off_);
            std::memcpy(out + written, pay_ + pay_off_, take);
            pay_off_ += take;
            written += take;
        }
        return written;
    }

private:
    static constexpr int kRowUnopened = -1;

    /// Positions the next block the requester asked for, or reports there is
    /// none left.
    bool advance() {
        for (;;) {
            if (lid_ == kRowUnopened) {
                if (row_ >= num_rows_) return false;
                gid_ = *req_ptr_++;
                const int num_cols = *req_ptr_++;
                cols_begin_ = req_ptr_;
                cols_end_ = req_ptr_ + num_cols;
                req_ptr_ = cols_end_;
                auto row_it = matrix_->graph->global_to_local.find(gid_);
                if (row_it == matrix_->graph->global_to_local.end()) {
                    ++row_;
                    continue;  // stays unopened, so the next pass reads the next row
                }
                lid_ = row_it->second;
                slot_ = matrix_->row_ptr()[lid_];
                slot_end_ = matrix_->row_ptr()[lid_ + 1];
            }
            while (slot_ < slot_end_) {
                const int slot = slot_++;
                const int col_lid = matrix_->col_ind()[slot];
                const int col_gid = matrix_->graph->get_global_index(col_lid);
                if (!std::binary_search(cols_begin_, cols_end_, col_gid)) continue;

                const int r_dim = matrix_->graph->block_sizes[lid_];
                const int c_dim = matrix_->graph->block_sizes[col_lid];
                std::memcpy(head_ + 0 * sizeof(int), &gid_, sizeof(int));
                std::memcpy(head_ + 1 * sizeof(int), &col_gid, sizeof(int));
                std::memcpy(head_ + 2 * sizeof(int), &r_dim, sizeof(int));
                std::memcpy(head_ + 3 * sizeof(int), &c_dim, sizeof(int));
                head_off_ = 0;
                head_len_ = 4 * sizeof(int);
                pay_ = reinterpret_cast<const char*>(matrix_->block_data(slot));
                pay_off_ = 0;
                pay_len_ = matrix_->block_size_elements(slot) * sizeof(T);
                return true;
            }
            ++row_;
            lid_ = kRowUnopened;
        }
    }

    const Matrix* matrix_;
    int num_rows_ = 0;
    std::vector<char> prelude_;
    size_t prelude_off_ = 0;

    const int* req_ptr_ = nullptr;
    const int* cols_begin_ = nullptr;
    const int* cols_end_ = nullptr;
    int row_ = 0;
    int gid_ = 0;
    int lid_ = kRowUnopened;
    int slot_ = 0;
    int slot_end_ = 0;

    char head_[4 * sizeof(int)];
    size_t head_off_ = 0;
    size_t head_len_ = 0;
    const char* pay_ = nullptr;
    size_t pay_off_ = 0;
    size_t pay_len_ = 0;
    bool done_ = false;
};

// One response segment's header walk: records every block's metadata and its
// payload offset within the segment WITHOUT touching the payload bytes, so
// the copies can be sized once and run in parallel afterwards.
template <typename T>
void scan_response(
        const char* ptr,
        FetchedBlockContext<T>& ctx,
        std::vector<size_t>& blob_offset,
        const char* segment_begin) {
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
        FetchedBlockRef<T> block;
        std::memcpy(&block.global_row, ptr, sizeof(int));
        ptr += sizeof(int);
        std::memcpy(&block.global_col, ptr, sizeof(int));
        ptr += sizeof(int);
        std::memcpy(&block.r_dim, ptr, sizeof(int));
        ptr += sizeof(int);
        std::memcpy(&block.c_dim, ptr, sizeof(int));
        ptr += sizeof(int);
        block.data = nullptr;  // resolved into the arena after sizing
        ctx.blocks.push_back(block);
        blob_offset.push_back(static_cast<size_t>(ptr - segment_begin));
        ptr += static_cast<size_t>(block.r_dim) * block.c_dim * sizeof(T);
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
    std::vector<int> resp_send_blocks(size, 0);
    for (int i = 0; i < size; ++i) {
        if (recv_counts[i] == 0) {
            continue;
        }
        resp_send_counts[i] = count_response_bytes(
            matrix,
            recv_blob.data() + rdispls[i],
            &resp_send_blocks[i]);
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

    std::vector<size_t> resp_rdispls(size + 1, 0);
    for (int i = 0; i < size; ++i) {
        resp_rdispls[i + 1] = resp_rdispls[i] + resp_recv_counts[i];
    }

    // The response exchange, sliced.
    //
    // Receives are posted whole and up front -- the received bytes ARE the
    // arena the caller keeps, so there is nothing to bound on that side. Sends
    // are streamed through a small fixed pool, so the served response never
    // exists in full and the pack of one slice runs while the last is on the
    // wire.
    //
    // Slice k of a peer's response is the same byte range on both sides
    // because both compute it from the same total and the same slice size, so
    // the cut points need no agreement message. Messages between one pair are
    // non-overtaking, so posting them in order is enough to match them up.
    std::vector<char> resp_recv_blob(resp_rdispls[size]);
    if (size > 1) {
        size_t slice = response_slice_bytes();
        {
            unsigned long long mine = slice, agreed = 0;
            MPI_Allreduce(&mine, &agreed, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                          matrix.graph->comm);
            slice = static_cast<size_t>(agreed);
        }

        std::vector<MPI_Request> recv_reqs;
        for (int i = 0; i < size; ++i) {
            for (size_t off = 0; off < resp_recv_counts[i]; off += slice) {
                const size_t n = std::min(slice, resp_recv_counts[i] - off);
                MPI_Request req;
                MPI_Irecv(resp_recv_blob.data() + resp_rdispls[i] + off,
                          static_cast<int>(n), MPI_BYTE, i, kResponsePayloadTag,
                          matrix.graph->comm, &req);
                recv_reqs.push_back(req);
            }
        }

        // Peers in ROTATED order, so rank r starts at r+1 rather than every
        // rank starting at 0 and making peer 0 an incast target while the rest
        // of the fabric is idle. Round-robin below then keeps them all
        // supplied, which is the concurrency the one-shot exchange had for
        // free by posting every send at once.
        std::vector<int> active;
        std::vector<ResponseWalker<Matrix>> walkers;
        active.reserve(size);
        walkers.reserve(size);
        for (int step = 1; step <= size; ++step) {
            const int i = (rank + step) % size;
            if (recv_counts[i] == 0 || resp_send_counts[i] == 0) continue;
            active.push_back(i);
            walkers.emplace_back(matrix, recv_blob.data() + rdispls[i],
                                 resp_send_blocks[i]);
        }

        // Two slices per peer keeps one on the wire while the next is packed.
        // Capped so a rank's send-side stays a fixed cost -- it does not grow
        // with the halo, the system, or the rank count; past the cap the peers
        // are simply served in rotated waves.
        const size_t pool_slices =
            std::min<size_t>(8, std::max<size_t>(2, 2 * active.size()));
        std::vector<std::vector<char>> pool(pool_slices);
        std::vector<MPI_Request> send_reqs(pool_slices, MPI_REQUEST_NULL);
        for (auto& buf : pool) buf.resize(slice);

        size_t turn = 0;
        while (!active.empty()) {
            int k = -1;
            for (size_t b = 0; b < send_reqs.size(); ++b) {
                if (send_reqs[b] == MPI_REQUEST_NULL) { k = static_cast<int>(b); break; }
            }
            if (k < 0) {
                MPI_Waitany(static_cast<int>(send_reqs.size()), send_reqs.data(), &k,
                            MPI_STATUS_IGNORE);
                if (k < 0) k = 0;  // unreachable: the scan proved one is in flight
            }

            if (turn >= active.size()) turn = 0;
            const size_t which = turn;
            const int peer = active[which];
            const size_t n = walkers[which].emit(pool[static_cast<size_t>(k)].data(), slice);
            if (n > 0) {
                MPI_Isend(pool[static_cast<size_t>(k)].data(), static_cast<int>(n),
                          MPI_BYTE, peer, kResponsePayloadTag, matrix.graph->comm,
                          &send_reqs[static_cast<size_t>(k)]);
            }
            if (walkers[which].done()) {
                active.erase(active.begin() + static_cast<long>(which));
                walkers.erase(walkers.begin() + static_cast<long>(which));
            } else {
                ++turn;
            }
        }

        MPI_Waitall(static_cast<int>(send_reqs.size()), send_reqs.data(),
                    MPI_STATUSES_IGNORE);
        for (auto& buf : pool) release_and_drop(buf);
        MPI_Waitall(static_cast<int>(recv_reqs.size()), recv_reqs.data(),
                    MPI_STATUSES_IGNORE);
    }

    // Unpack in two passes: a serial header walk that records every remote
    // block's metadata and payload location (touching no payload bytes),
    // then one sized arena filled by a PARALLEL copy. The per-block heap
    // vector this replaces was tens of millions of allocations and a
    // single-threaded memcpy of the entire halo at fused-kernel scale.
    const size_t local_blocks = ctx.blocks.size();  // refs into the matrix
    std::vector<size_t> blob_offset;                // remote blocks only
    for (int i = 0; i < size; ++i) {
        if (resp_recv_counts[i] == 0) {
            continue;
        }
        scan_response(resp_recv_blob.data() + resp_rdispls[i], ctx, blob_offset,
                      resp_recv_blob.data());
        // scan_response records offsets relative to the whole blob: pass the
        // blob base as the segment origin so one offset table serves all
        // segments.
    }

    // The blob IS the arena. The copy this replaces made the payload exist
    // twice at once -- blob and arena both live across the memcpy -- so the
    // fetch peaked at double what it delivered, and a full pass over the halo
    // was spent moving bytes that were already in the right layout.
    const size_t remote_blocks = ctx.blocks.size() - local_blocks;
    ctx.arena = std::move(resp_recv_blob);
    for (size_t b = 0; b < remote_blocks; ++b) {
        auto& blk = ctx.blocks[local_blocks + b];
        blk.data = reinterpret_cast<const T*>(ctx.arena.data() +
                                             blob_offset[b]);
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

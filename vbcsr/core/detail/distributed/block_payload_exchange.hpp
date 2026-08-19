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

/// Packs one peer's response as a byte STREAM rather than into a buffer.
///
/// The point is that the whole response never exists at once. Packing it whole
/// cost two things: the served blob sat beside the received one for the length
/// of the exchange -- which is why the halo budget carries a factor of two --
/// and nothing went on the wire until the LAST byte was packed, so a
/// single-threaded memcpy of tens of GB ran with the network idle and then the
/// network ran with the CPU idle.
///
/// `sink.append` takes bytes; it is free to cut them into fixed-size slices
/// wherever it likes, including inside a block, because the receiver walks the
/// reassembled segment and never learns where the cuts were.
template <typename Matrix, typename Sink>
void stream_response(
        const Matrix& matrix,
        const int* req_buffer,
        int total_blocks,
        Sink& sink) {
    using T = typename Matrix::value_type;

    const int num_rows = req_buffer[0];
    const int* req_start = req_buffer + 1;

    sink.append(&num_rows, sizeof(int));

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
        sink.append(&gid, sizeof(int));
        sink.append(&size, sizeof(int));
    }
    sink.append(&total_blocks, sizeof(int));

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

            const int r_dim = matrix.graph->block_sizes[lid];
            const int c_dim = matrix.graph->block_sizes[col_lid];
            const size_t elems = matrix.block_size_elements(slot);

            sink.append(&gid, sizeof(int));
            sink.append(&col_gid, sizeof(int));
            sink.append(&r_dim, sizeof(int));
            sink.append(&c_dim, sizeof(int));
            sink.append(matrix.block_data(slot), elems * sizeof(T));
        }
    }
}

/// Sink that cuts a response into fixed slices and ships each as it fills.
///
/// The buffers are a small fixed pool, so what a rank holds for SENDING is
/// `pool x slice` -- a hundred megabytes or so -- rather than its whole
/// response, whatever the rank count or the system size. Filling the next
/// slice while the last one is in flight is what overlaps the pack with the
/// wire.
class ResponseSliceSink {
public:
    ResponseSliceSink(std::vector<std::vector<char>>& pool,
                      std::vector<MPI_Request>& reqs, size_t slice, int peer,
                      MPI_Comm comm)
        : pool_(pool), reqs_(reqs), slice_(slice), peer_(peer), comm_(comm) {}

    void append(const void* src, size_t n) {
        const char* p = static_cast<const char*>(src);
        while (n > 0) {
            if (current_ < 0) current_ = acquire();
            const size_t room = slice_ - filled_;
            const size_t take = std::min(room, n);
            std::memcpy(pool_[static_cast<size_t>(current_)].data() + filled_, p, take);
            filled_ += take;
            p += take;
            n -= take;
            if (filled_ == slice_) ship();
        }
    }

    /// Ships whatever is left, which is the only slice allowed to be short.
    void finish() {
        if (current_ >= 0 && filled_ > 0) ship();
    }

private:
    int acquire() {
        for (size_t k = 0; k < reqs_.size(); ++k) {
            if (reqs_[k] == MPI_REQUEST_NULL) return static_cast<int>(k);
        }
        int idx = MPI_UNDEFINED;
        MPI_Waitany(static_cast<int>(reqs_.size()), reqs_.data(), &idx,
                    MPI_STATUS_IGNORE);
        // Unreachable -- the scan above proved one request is active, so
        // Waitany has something to complete -- but a negative index here would
        // be a wild write rather than an error.
        return idx >= 0 ? idx : 0;
    }

    void ship() {
        MPI_Isend(pool_[static_cast<size_t>(current_)].data(),
                  static_cast<int>(filled_), MPI_BYTE, peer_, kResponsePayloadTag,
                  comm_, &reqs_[static_cast<size_t>(current_)]);
        current_ = -1;
        filled_ = 0;
    }

    std::vector<std::vector<char>>& pool_;
    std::vector<MPI_Request>& reqs_;
    size_t slice_;
    int peer_;
    MPI_Comm comm_;
    int current_ = -1;
    size_t filled_ = 0;
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

        // Deep enough that a slice is always being packed while another is in
        // flight, shallow enough that pool x slice stays a rounding error
        // against the halo it is serving.
        constexpr size_t kSendPoolSlices = 4;
        // A buffer only ever has to hold a full slice if some peer's response
        // is at least that big; below that the pool costs what it ships.
        size_t widest = 0;
        for (int i = 0; i < size; ++i) {
            widest = std::max(widest, resp_send_counts[i]);
        }
        const size_t buf_bytes = std::min(slice, widest);
        std::vector<std::vector<char>> pool(kSendPoolSlices);
        std::vector<MPI_Request> send_reqs(kSendPoolSlices, MPI_REQUEST_NULL);
        for (auto& buf : pool) buf.resize(buf_bytes);

        for (int i = 0; i < size; ++i) {
            if (recv_counts[i] == 0 || resp_send_counts[i] == 0) {
                continue;
            }
            ResponseSliceSink sink(pool, send_reqs, slice, i, matrix.graph->comm);
            stream_response(matrix, recv_blob.data() + rdispls[i],
                            resp_send_blocks[i], sink);
            sink.finish();
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

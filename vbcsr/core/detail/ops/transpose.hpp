#ifndef VBCSR_DETAIL_OPS_TRANSPOSE_HPP
#define VBCSR_DETAIL_OPS_TRANSPOSE_HPP

#include "../../scalar_traits.hpp"
#include "../distributed/mpi_utils.hpp"
#include "../distributed/result_graph.hpp"
#include "../storage/index_map.hpp"

#include <omp.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <unordered_set>
#include <vector>

namespace vbcsr::detail {

// dest = conj(src^T) for canonical row-major blocks: src is src_rows x
// src_cols with src(i, j) at src[i * src_cols + j]; dest is src_cols x
// src_rows with dest(j, i) at dest[j * src_rows + i].
template <typename T>
void write_transposed_conjugate_block(
    T* dest,
    const T* src,
    int src_rows,
    int src_cols) {
    for (int i = 0; i < src_rows; ++i) {
        for (int j = 0; j < src_cols; ++j) {
            dest[j * src_rows + i] =
                ScalarTraits<T>::conjugate(src[i * src_cols + j]);
        }
    }
}

// Bytes of block payload one rank packs into a single value-exchange round.
// The exchange is chunked to this budget so peak memory is the matrix plus a
// bounded buffer rather than three full copies of it (which is what a single
// pack-everything/alltoallv/unpack-everything pass costs, and what put a
// multi-million-orbital transpose out of reach). The budget bounds BOTH sides
// of a round: what one rank packs to send, and -- through the per-source
// allowances each receiver grants before the rounds start -- what one rank
// takes in.
inline size_t transpose_chunk_bytes() {
    const char* env = std::getenv("VBCSR_TRANSPOSE_CHUNK_MB");
    const long long mb = env ? std::atoll(env) : 0;
    return static_cast<size_t>(mb > 0 ? mb : 256) << 20;
}

// Per-destination layout of the source blocks in one row range, computed in
// parallel. Rows are cut into `chunks` contiguous pieces; each piece counts
// privately, a serial prefix over (piece, destination) fixes the bases, and
// the pack pass then writes with a private cursor per piece. The result is
// byte-identical to a single-threaded row-order pack -- which matters, because
// the receiver relies on that order to match values against metadata.
//
// This replaces the old serial offset pass, which walked every block on one
// thread and measured as the single largest non-MPI phase of the transpose.
struct GroupLayout {
    int chunks = 0;
    std::vector<int> row_begin;       // chunks + 1 row boundaries
    std::vector<size_t> base_blocks;  // chunks * nranks: first block index in group
    std::vector<size_t> base_elems;   // chunks * nranks: first element offset in group
    std::vector<size_t> group_blocks; // nranks
    std::vector<size_t> group_elems;  // nranks
};

// `dest[k]` is the rank that owns slot k's transposed image, or kTransposeSkip
// for slots the caller excludes. Slots destined for `my_rank` are left out of
// the layout entirely: they never enter the exchange.
inline constexpr int kTransposeSkip = -1;

template <typename Matrix>
GroupLayout plan_remote_groups(
    const Matrix& matrix,
    const std::vector<int>& dest,
    int row_lo,
    int row_hi,
    int my_rank,
    int nranks,
    int max_chunks) {
    const DistGraph& graph = *matrix.graph;
    const std::vector<int>& adj_ptr = graph.adj_ptr;

    GroupLayout layout;
    const int n_rows = row_hi - row_lo;
    layout.chunks = std::max(1, std::min(max_chunks, n_rows));
    layout.row_begin.resize(static_cast<size_t>(layout.chunks) + 1);
    for (int c = 0; c <= layout.chunks; ++c) {
        layout.row_begin[static_cast<size_t>(c)] =
            row_lo + static_cast<int>(static_cast<long long>(n_rows) * c / layout.chunks);
    }

    const size_t table = static_cast<size_t>(layout.chunks) * nranks;
    layout.base_blocks.assign(table, 0);
    layout.base_elems.assign(table, 0);
    layout.group_blocks.assign(nranks, 0);
    layout.group_elems.assign(nranks, 0);

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < layout.chunks; ++c) {
        size_t* blocks = layout.base_blocks.data() + static_cast<size_t>(c) * nranks;
        size_t* elems = layout.base_elems.data() + static_cast<size_t>(c) * nranks;
        for (int i = layout.row_begin[static_cast<size_t>(c)];
             i < layout.row_begin[static_cast<size_t>(c) + 1]; ++i) {
            const int r_dim = graph.block_sizes[i];
            for (int k = adj_ptr[i]; k < adj_ptr[i + 1]; ++k) {
                const int owner = dest[static_cast<size_t>(k)];
                if (owner == kTransposeSkip || owner == my_rank) continue;
                blocks[owner] += 1;
                elems[owner] += static_cast<size_t>(r_dim) * graph.block_sizes[graph.adj_ind[k]];
            }
        }
    }

    // Serial prefix over (destination, chunk): tiny (chunks * nranks), and it
    // is what makes the parallel pack reproduce row order exactly.
    for (int q = 0; q < nranks; ++q) {
        size_t run_blocks = 0;
        size_t run_elems = 0;
        for (int c = 0; c < layout.chunks; ++c) {
            const size_t idx = static_cast<size_t>(c) * nranks + q;
            const size_t nb = layout.base_blocks[idx];
            const size_t ne = layout.base_elems[idx];
            layout.base_blocks[idx] = run_blocks;
            layout.base_elems[idx] = run_elems;
            run_blocks += nb;
            run_elems += ne;
        }
        layout.group_blocks[static_cast<size_t>(q)] = run_blocks;
        layout.group_elems[static_cast<size_t>(q)] = run_elems;
    }
    return layout;
}

// C = A^H, or -- when `mirror` -- the Hermitian completion C = A +
// strict_lower(A^H) of an upper-triangle-only A.
//
// Three things this does that a pack-everything transpose does not:
//
//  - Blocks whose column this rank already owns never enter the exchange.
//    They are conjugate-transposed straight from A's storage into C's. On a
//    locality-preserving partition that is the overwhelming majority of the
//    matrix, and the old code round-tripped all of it through a self-send.
//
//  - The value exchange runs in bounded rounds (transpose_chunk_bytes), so
//    peak memory is A + C + one chunk instead of A + C + two full copies.
//    The bound holds on both ends: each receiver grants its sources per-round
//    allowances before the rounds start, so a rank fed by many senders at
//    once still sees one budget's worth per round, not one per sender.
//    Metadata is exchanged whole up front -- it is 16 bytes per block against
//    a block payload, so chunking it would buy nothing -- which lets the
//    receiver resolve every destination slot before any value arrives, and is
//    also what tells it how much each source holds for it, which is what the
//    allowances are cut from.
//
//  - The result graph is built with construct_from_local_csr, so the ghost
//    set, ghost block sizes and partition all come from data already in hand.
//    construct_distributed would have re-derived the ghosts by scanning the
//    adjacency, re-sorted every row, fetched ghost sizes over MPI and
//    allgathered the partition.
//
// `mirror` additionally fuses what spmm_hermitian used to do in three steps
// (full transpose into a second matrix, zero its diagonal, union-axpby into a
// third): A's own blocks are copied in place and only the strict upper
// triangle is mirrored, so no intermediate matrix or union graph is built. A
// is required to be upper-triangle-only; anything in its strict lower triangle
// is ignored rather than mirrored on top of an existing block.
// `consume_source`, when set, must be `&matrix`: the caller is handing this
// function its source to destroy, and each row's storage is released as soon as
// the last pass that reads it is done with it. See the note at stage 6 for the
// ordering that makes "the last pass" a row prefix, and
// BlockSpMat::complete_hermitian_inplace for why the entry point that does this
// is the mutating one -- a const method cannot honestly free its own object.
template <typename Matrix>
Matrix conjugate_transpose(const Matrix& matrix, bool mirror,
                           Matrix* consume_source = nullptr) {
    using T = typename Matrix::value_type;

    DistGraph& ga = *matrix.graph;
    const int my_rank = ga.rank;
    const int nranks = ga.size;
    const int n_owned = static_cast<int>(ga.owned_global_indices.size());
    const std::vector<int>& adj_ptr = ga.adj_ptr;
    const size_t nnz = static_cast<size_t>(adj_ptr[n_owned]);

    // ---- 1. Classify every source block by who owns its transposed image.
    std::vector<int> dest(nnz, kTransposeSkip);
    #pragma omp parallel for schedule(dynamic, 256)
    for (int i = 0; i < n_owned; ++i) {
        const int g_row = ga.owned_global_indices[i];
        for (int k = adj_ptr[i]; k < adj_ptr[i + 1]; ++k) {
            const int local_col = ga.adj_ind[k];
            const int g_col = ga.get_global_index(local_col);
            if (mirror && g_col <= g_row) continue;  // diagonal stays, lower is not ours
            dest[static_cast<size_t>(k)] = (nranks == 1) ? my_rank : ga.find_owner(g_col);
        }
    }

    // ---- 2. Exchange the metadata of the remote blocks, whole.
    // Meta record is the DESTINATION block's (global row, global col, row dim,
    // col dim); the payload sent later is already conjugate-transposed, so the
    // receiver only ever memcpys.
    const int max_chunks = std::max(1, omp_get_max_threads() * 4);
    std::vector<int> recv_meta;
    std::vector<size_t> recv_blocks_from(static_cast<size_t>(nranks), 0);
    if (nranks > 1) {
        const GroupLayout meta_layout =
            plan_remote_groups(matrix, dest, 0, n_owned, my_rank, nranks, max_chunks);

        std::vector<size_t> send_meta_counts(nranks), send_meta_displs(nranks + 1, 0);
        for (int q = 0; q < nranks; ++q) {
            send_meta_counts[static_cast<size_t>(q)] = meta_layout.group_blocks[static_cast<size_t>(q)] * 4;
            send_meta_displs[static_cast<size_t>(q) + 1] =
                send_meta_displs[static_cast<size_t>(q)] + send_meta_counts[static_cast<size_t>(q)];
        }
        std::vector<int> send_meta(send_meta_displs[static_cast<size_t>(nranks)]);

        #pragma omp parallel for schedule(static)
        for (int c = 0; c < meta_layout.chunks; ++c) {
            std::vector<size_t> cursor(static_cast<size_t>(nranks));
            for (int q = 0; q < nranks; ++q) {
                cursor[static_cast<size_t>(q)] =
                    send_meta_displs[static_cast<size_t>(q)] +
                    meta_layout.base_blocks[static_cast<size_t>(c) * nranks + q] * 4;
            }
            for (int i = meta_layout.row_begin[static_cast<size_t>(c)];
                 i < meta_layout.row_begin[static_cast<size_t>(c) + 1]; ++i) {
                const int g_row = ga.owned_global_indices[i];
                const int r_dim = ga.block_sizes[i];
                for (int k = adj_ptr[i]; k < adj_ptr[i + 1]; ++k) {
                    const int owner = dest[static_cast<size_t>(k)];
                    if (owner == kTransposeSkip || owner == my_rank) continue;
                    const int local_col = ga.adj_ind[k];
                    int* meta = send_meta.data() + cursor[static_cast<size_t>(owner)];
                    meta[0] = ga.get_global_index(local_col);  // destination row
                    meta[1] = g_row;                           // destination column
                    meta[2] = ga.block_sizes[local_col];       // destination row dim
                    meta[3] = r_dim;                           // destination col dim
                    cursor[static_cast<size_t>(owner)] += 4;
                }
            }
        }

        std::vector<size_t> recv_meta_counts(nranks), recv_meta_displs(nranks + 1, 0);
        MPI_Alltoall(send_meta_counts.data(), sizeof(size_t), MPI_BYTE,
                     recv_meta_counts.data(), sizeof(size_t), MPI_BYTE, ga.comm);
        for (int q = 0; q < nranks; ++q) {
            recv_blocks_from[static_cast<size_t>(q)] = recv_meta_counts[static_cast<size_t>(q)] / 4;
            recv_meta_displs[static_cast<size_t>(q) + 1] =
                recv_meta_displs[static_cast<size_t>(q)] + recv_meta_counts[static_cast<size_t>(q)];
        }
        recv_meta.resize(recv_meta_displs[static_cast<size_t>(nranks)]);
        safe_alltoallv(send_meta.data(), send_meta_counts, send_meta_displs, MPI_INT,
                       recv_meta.data(), recv_meta_counts, recv_meta_displs, MPI_INT, ga.comm);
    }
    const size_t n_recv = recv_meta.size() / 4;

    // ---- 3. Result ghosts: the remote rows that sent to us, plus -- in
    // mirror mode -- A's own ghost columns, which the kept blocks still name.
    // Every ghost's block size is already in hand, so no fetch is needed.
    std::vector<int> ghosts;
    std::vector<int> ghost_sizes;
    {
        // A ghost is a column this rank does NOT own, and that has to be
        // enforced here rather than assumed.
        //
        // to_local below maps owned globals to [0, n_owned) and then ghosts to
        // n_owned + g. If an id reaches both loops the ghost pass OVERWRITES its
        // owned index with an out-of-range one, and every consumer that sizes an
        // array by n_owned -- `counts` immediately below is the first -- writes
        // past the end. That is a heap-buffer-overflow, and it surfaced as
        // `free(): invalid next size` several phases later in a 4-rank
        // Newton-Schulz run, far from here.
        //
        // The received metadata is the source: a block arrives carrying the
        // sender's row as its destination COLUMN, and nothing guaranteed that
        // column was remote to us. Verified directly -- rank 0 saw g_col=0 with
        // in_owned=1 mapped to local_row=128 against n_owned=128.
        std::unordered_set<int> owned_ids(ga.owned_global_indices.begin(),
                                          ga.owned_global_indices.end());
        std::vector<std::pair<int, int>> candidates;  // (global id, block size)
        candidates.reserve(n_recv + (mirror ? ga.ghost_global_indices.size() : 0));
        for (size_t b = 0; b < n_recv; ++b) {
            const int gid = recv_meta[b * 4 + 1];
            if (owned_ids.count(gid) != 0) continue;  // owned: already in to_local
            candidates.emplace_back(gid, recv_meta[b * 4 + 3]);
        }
        if (mirror) {
            for (size_t g = 0; g < ga.ghost_global_indices.size(); ++g) {
                const int gid = ga.ghost_global_indices[g];
                if (owned_ids.count(gid) != 0) continue;
                candidates.emplace_back(gid,
                                        ga.block_sizes[static_cast<size_t>(n_owned) + g]);
            }
        }
        // construct_from_local_csr requires ghosts ordered by (owner, gid).
        std::sort(candidates.begin(), candidates.end(),
                  [&](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                      const int oa = ga.find_owner(a.first, ga.block_displs);
                      const int ob = ga.find_owner(b.first, ga.block_displs);
                      if (oa != ob) return oa < ob;
                      return a.first < b.first;
                  });
        candidates.erase(std::unique(candidates.begin(), candidates.end(),
                                     [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                                         return a.first == b.first;
                                     }),
                         candidates.end());
        ghosts.reserve(candidates.size());
        ghost_sizes.reserve(candidates.size());
        for (const auto& c : candidates) {
            ghosts.push_back(c.first);
            ghost_sizes.push_back(c.second);
        }
    }

    // Global -> result-local for every column the result can name. Owned
    // blocks keep their source-local index; ghosts are renumbered.
    IndexMap to_local;
    to_local.reserve(static_cast<size_t>(n_owned) + ghosts.size());
    for (int i = 0; i < n_owned; ++i) {
        to_local[ga.owned_global_indices[i]] = i;
    }
    for (size_t g = 0; g < ghosts.size(); ++g) {
        to_local[ghosts[g]] = n_owned + static_cast<int>(g);
    }

    // ---- 4. Result adjacency, in result-local column ids.
    std::vector<int> c_adj_ptr(static_cast<size_t>(n_owned) + 1, 0);
    {
        std::vector<int> counts(static_cast<size_t>(n_owned), 0);
        if (mirror) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n_owned; ++i) {
                counts[static_cast<size_t>(i)] += adj_ptr[i + 1] - adj_ptr[i];
            }
        }
        #pragma omp parallel for schedule(dynamic, 256)
        for (int i = 0; i < n_owned; ++i) {
            for (int k = adj_ptr[i]; k < adj_ptr[i + 1]; ++k) {
                if (dest[static_cast<size_t>(k)] != my_rank) continue;
                const int local_row = to_local.at(ga.get_global_index(ga.adj_ind[k]));
                #pragma omp atomic
                counts[static_cast<size_t>(local_row)] += 1;
            }
        }
        #pragma omp parallel for schedule(static)
        for (size_t b = 0; b < n_recv; ++b) {
            const int local_row = to_local.at(recv_meta[b * 4]);
            #pragma omp atomic
            counts[static_cast<size_t>(local_row)] += 1;
        }
        size_t running = 0;
        for (int i = 0; i < n_owned; ++i) {
            running += static_cast<size_t>(counts[static_cast<size_t>(i)]);
            if (running > static_cast<size_t>(std::numeric_limits<int>::max())) {
                throw std::overflow_error(
                    "transpose adjacency exceeds 2^31 blocks on this rank; distribute over more ranks");
            }
            c_adj_ptr[static_cast<size_t>(i) + 1] = static_cast<int>(running);
        }
    }

    std::vector<int> c_adj_ind(static_cast<size_t>(c_adj_ptr[static_cast<size_t>(n_owned)]));
    {
        std::vector<int> cursor(c_adj_ptr.begin(), c_adj_ptr.end() - 1);
        if (mirror) {
            #pragma omp parallel for schedule(dynamic, 256)
            for (int i = 0; i < n_owned; ++i) {
                int at = cursor[static_cast<size_t>(i)];
                for (int k = adj_ptr[i]; k < adj_ptr[i + 1]; ++k) {
                    c_adj_ind[static_cast<size_t>(at++)] =
                        to_local.at(ga.get_global_index(ga.adj_ind[k]));
                }
                cursor[static_cast<size_t>(i)] = at;
            }
        }
        #pragma omp parallel for schedule(dynamic, 256)
        for (int i = 0; i < n_owned; ++i) {
            for (int k = adj_ptr[i]; k < adj_ptr[i + 1]; ++k) {
                if (dest[static_cast<size_t>(k)] != my_rank) continue;
                const int local_row = to_local.at(ga.get_global_index(ga.adj_ind[k]));
                int at;
                #pragma omp atomic capture
                at = cursor[static_cast<size_t>(local_row)]++;
                c_adj_ind[static_cast<size_t>(at)] = i;
            }
        }
        #pragma omp parallel for schedule(static)
        for (size_t b = 0; b < n_recv; ++b) {
            const int local_row = to_local.at(recv_meta[b * 4]);
            int at;
            #pragma omp atomic capture
            at = cursor[static_cast<size_t>(local_row)]++;
            c_adj_ind[static_cast<size_t>(at)] = to_local.at(recv_meta[b * 4 + 1]);
        }
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < n_owned; ++i) {
            std::sort(c_adj_ind.begin() + c_adj_ptr[static_cast<size_t>(i)],
                      c_adj_ind.begin() + c_adj_ptr[static_cast<size_t>(i) + 1]);
        }
    }

    // ---- 5. Result graph and storage.
    DistGraph* graph_c = new DistGraph(ga.comm);
    {
        NumaVector<int> adj_ind_c;
        adj_ind_c.assign(c_adj_ind.data(), c_adj_ind.size());
        graph_c->construct_from_local_csr(
            ga.owned_global_indices,
            std::vector<int>(ga.block_sizes.begin(), ga.block_sizes.begin() + n_owned),
            ghosts,
            ghost_sizes,
            ga.block_displs,
            std::move(c_adj_ptr),
            std::move(adj_ind_c));
    }
    // Consuming skips the first-touch zero pass; the plain path keeps it. That
    // is a real trade, not an oversight either way.
    //
    // The pass is redundant for correctness in both -- stage 6 writes every
    // block the result graph has, because that graph was built in stage 4 from
    // exactly those blocks -- and what it buys is domain-aligned NUMA
    // placement, which stage 6 cannot reproduce: its local writes are a SCATTER
    // (source row i lands in result row j > i), so no ordering of them places
    // result rows by domain.
    //
    // But when consuming, the pass defeats the whole point: it makes the result
    // fully resident before a single row of the source has been released, so
    // the source and the finished result are both whole at the same moment --
    // which is exactly the peak the caller asked to avoid. Placement is worth
    // less than 5.7 GB of peak on the path that asks for this.
    Matrix result = consume_source != nullptr
                        ? Matrix::make_overwritten_result(graph_c,
                                                          matrix.configured_page_size())
                        : Matrix(graph_c);
    if (consume_source == nullptr) {
        result.owns_graph = true;
        result.graph->enable_matrix_lifetime_management();
        result.set_page_size(matrix.configured_page_size());
    }

    // Position of column `local_col` in result row `local_row`.
    const std::vector<int>& cp = graph_c->adj_ptr;
    const NumaVector<int>& ci = graph_c->adj_ind;
    auto slot_of = [&](int local_row, int local_col) {
        const auto begin = ci.begin() + cp[static_cast<size_t>(local_row)];
        const auto end = ci.begin() + cp[static_cast<size_t>(local_row) + 1];
        const auto it = std::lower_bound(begin, end, local_col);
        return (it == end || *it != local_col)
                   ? -1
                   : static_cast<int>(std::distance(ci.begin(), it));
    };

    int any_missing_dest = 0;

    // ---- 6. The values, remote first and local second.
    //
    // THE ORDER MATTERS ONLY WHEN CONSUMING, and then it is what makes the
    // release possible at all. The source is read by two passes -- the remote
    // pack (6b) and the local writes (6a) -- so a row is dead only once BOTH
    // have seen it. 6b walks rows in byte-bounded rounds whose boundaries are a
    // row prefix, and 6a walks every row; running 6b to completion first leaves
    // "already read by everything" equal to "below the row 6a has reached",
    // which is a prefix and so is exactly what release_values_below_row takes.
    //
    // Interleaving them per round would release earlier and hold less, but it
    // would also weave two exchange-bound loops together for a fraction of one
    // chunk. Not worth the tangle: the chunk is the unit of overshoot either
    // way.
    //
    // For the plain path the order is immaterial -- the two write disjoint
    // slots of a result whose pages are already placed by the zero pass above
    // -- so there is one ordering here rather than two.

    // ---- 6b. Remote blocks, in bounded rounds.
    const auto write_local_rows = [&](int row_lo, int row_hi) {
        if (mirror) {
            #pragma omp parallel for schedule(dynamic, 256) reduction(|:any_missing_dest)
            for (int i = row_lo; i < row_hi; ++i) {
                for (int k = adj_ptr[i]; k < adj_ptr[i + 1]; ++k) {
                    const int slot = slot_of(i, to_local.at(ga.get_global_index(ga.adj_ind[k])));
                    if (slot < 0) { any_missing_dest = 1; continue; }
                    std::memcpy(result.mutable_block_data(slot), matrix.block_data(k),
                                matrix.block_size_elements(k) * sizeof(T));
                }
            }
        }
        #pragma omp parallel for schedule(dynamic, 256) reduction(|:any_missing_dest)
        for (int i = row_lo; i < row_hi; ++i) {
            const int r_dim = ga.block_sizes[i];
            for (int k = adj_ptr[i]; k < adj_ptr[i + 1]; ++k) {
                if (dest[static_cast<size_t>(k)] != my_rank) continue;
                const int local_col = ga.adj_ind[k];
                const int slot = slot_of(to_local.at(ga.get_global_index(local_col)), i);
                if (slot < 0) { any_missing_dest = 1; continue; }
                write_transposed_conjugate_block(result.mutable_block_data(slot),
                                                 matrix.block_data(k),
                                                 r_dim, ga.block_sizes[local_col]);
            }
        }
    };

    // ---- 6b. Remote blocks, in bounded rounds.
    if (nranks > 1) {
        // Destination slot and payload size of every incoming block, resolved
        // once from the metadata so a value round is a pure memcpy.
        std::vector<int> recv_slot(n_recv);
        std::vector<size_t> recv_elems(n_recv);
        #pragma omp parallel for schedule(static) reduction(|:any_missing_dest)
        for (size_t b = 0; b < n_recv; ++b) {
            const int* meta = recv_meta.data() + b * 4;
            const int slot = slot_of(to_local.at(meta[0]), to_local.at(meta[1]));
            if (slot < 0) any_missing_dest = 1;
            recv_slot[b] = slot;
            recv_elems[b] = static_cast<size_t>(meta[2]) * static_cast<size_t>(meta[3]);
        }

        // Start of each source's metadata segment, so a round can address its
        // slice without rescanning the counts.
        std::vector<size_t> recv_base(static_cast<size_t>(nranks) + 1, 0);
        for (int q = 0; q < nranks; ++q) {
            recv_base[static_cast<size_t>(q) + 1] =
                recv_base[static_cast<size_t>(q)] + recv_blocks_from[static_cast<size_t>(q)];
        }

        // Per-source allowances, granted by the RECEIVER.
        //
        // The budget below bounds what one rank packs per round, but a rank is
        // fed by every other rank at once: with P senders each shipping a full
        // budget to the same destination in the same round, the receive buffer
        // is P x the budget -- unbounded in the rank count, and tens of GB on
        // a hot receiver at cluster scale. The metadata exchange already told
        // each receiver exactly how much every source holds for it, so it cuts
        // its budget into per-source shares proportional to those holdings and
        // grants them up front, one Alltoall of 8 bytes per rank. Proportional
        // shares mean every source drains its holding in the same number of
        // rounds, so the exchange still finishes in
        // max over receivers of (incoming / budget) rounds -- the minimum any
        // receive-bounded exchange needs -- rather than being throttled by an
        // arbitrary split.
        //
        // The floor keeps a source with a sliver from being granted next to
        // nothing and stretching the exchange into thousands of rounds. All P
        // sources hitting the floor at once overshoots the budget by P x 64 KB
        // -- noise against the budget at any plausible rank count.
        const size_t budget = transpose_chunk_bytes();
        std::vector<size_t> allowance(static_cast<size_t>(nranks), 0);
        {
            std::vector<size_t> grant(static_cast<size_t>(nranks), 0);
            size_t total_in = 0;
            for (int q = 0; q < nranks; ++q) {
                size_t bytes = 0;
                for (size_t b = recv_base[static_cast<size_t>(q)];
                     b < recv_base[static_cast<size_t>(q) + 1]; ++b) {
                    bytes += recv_elems[b] * sizeof(T);
                }
                grant[static_cast<size_t>(q)] = bytes;
                total_in += bytes;
            }
            constexpr size_t kGrantFloor = size_t(1) << 16;
            for (int q = 0; q < nranks; ++q) {
                size_t share = grant[static_cast<size_t>(q)];
                if (total_in > budget) {
                    share = static_cast<size_t>(
                        static_cast<long double>(share) * budget / total_in);
                }
                grant[static_cast<size_t>(q)] = std::max(share, kGrantFloor);
            }
            MPI_Alltoall(grant.data(), sizeof(size_t), MPI_BYTE,
                         allowance.data(), sizeof(size_t), MPI_BYTE, ga.comm);
        }

        // Round boundaries by byte prefix, against BOTH bounds: the sender's
        // own budget and every destination's allowance.
        //
        // Rows are priced at segment granularity -- plan_remote_groups'
        // prefix table gives, per (segment, destination), the bytes this rank
        // would send -- and a serial walk over the segments closes a round
        // when the next one would push the total over the budget or any
        // destination over its allowance. Boundaries stay a row prefix, which
        // is what the consuming release in stage 6a relies on. A round always
        // takes at least one segment, so a single segment that alone exceeds
        // an allowance ships anyway instead of deadlocking the exchange --
        // irreducible without splitting inside it, and the same caveat the
        // send-side bound always had for a single oversized row.
        //
        // Segment count: fine enough that one segment's overshoot is noise,
        // capped so the prefix table (16 B x segments x nranks) stays small
        // against the buffers it exists to bound.
        std::vector<int> round_bound;
        {
            const int by_table = static_cast<int>(std::max<size_t>(
                1, (size_t(32) << 20) / (2 * sizeof(size_t) * static_cast<size_t>(nranks))));
            const int segments = std::max(max_chunks, std::min(4096, by_table));
            const GroupLayout plan =
                plan_remote_groups(matrix, dest, 0, n_owned, my_rank, nranks, segments);
            auto segment_bytes = [&](int c, int q) {
                const size_t lo =
                    plan.base_elems[static_cast<size_t>(c) * nranks + q];
                const size_t hi =
                    c + 1 < plan.chunks
                        ? plan.base_elems[(static_cast<size_t>(c) + 1) * nranks + q]
                        : plan.group_elems[static_cast<size_t>(q)];
                return (hi - lo) * sizeof(T);
            };
            std::vector<size_t> acc(static_cast<size_t>(nranks), 0);
            std::vector<size_t> seg(static_cast<size_t>(nranks), 0);
            size_t acc_total = 0;
            round_bound.push_back(0);
            for (int c = 0; c < plan.chunks; ++c) {
                size_t seg_total = 0;
                bool over = false;
                for (int q = 0; q < nranks; ++q) {
                    seg[static_cast<size_t>(q)] = segment_bytes(c, q);
                    seg_total += seg[static_cast<size_t>(q)];
                    over |= acc[static_cast<size_t>(q)] + seg[static_cast<size_t>(q)] >
                            allowance[static_cast<size_t>(q)];
                }
                if (acc_total > 0 && (over || acc_total + seg_total > budget)) {
                    round_bound.push_back(plan.row_begin[static_cast<size_t>(c)]);
                    std::fill(acc.begin(), acc.end(), 0);
                    acc_total = 0;
                }
                for (int q = 0; q < nranks; ++q) {
                    acc[static_cast<size_t>(q)] += seg[static_cast<size_t>(q)];
                }
                acc_total += seg_total;
            }
            round_bound.push_back(n_owned);
        }
        long long my_rounds =
            std::max<long long>(1, static_cast<long long>(round_bound.size()) - 1);
        long long rounds = my_rounds;
        MPI_Allreduce(&my_rounds, &rounds, 1, MPI_LONG_LONG, MPI_MAX, ga.comm);
        // The rounds are collective, so a rank that needed fewer runs empty
        // ones rather than dropping out of the exchange.
        while (static_cast<long long>(round_bound.size()) - 1 < rounds) {
            round_bound.push_back(n_owned);
        }

        std::vector<size_t> cursor(static_cast<size_t>(nranks), 0);
        std::vector<T> send_values;
        std::vector<T> recv_values;
        std::vector<size_t> round_offset;
        std::vector<size_t> round_block;
        for (long long r = 0; r < rounds; ++r) {
            const int row_lo = round_bound[static_cast<size_t>(r)];
            const int row_hi = round_bound[static_cast<size_t>(r) + 1];
            const GroupLayout layout =
                plan_remote_groups(matrix, dest, row_lo, row_hi, my_rank, nranks, max_chunks);

            std::vector<size_t> send_counts(nranks), send_displs(nranks + 1, 0);
            for (int q = 0; q < nranks; ++q) {
                send_counts[static_cast<size_t>(q)] = layout.group_elems[static_cast<size_t>(q)];
                send_displs[static_cast<size_t>(q) + 1] =
                    send_displs[static_cast<size_t>(q)] + send_counts[static_cast<size_t>(q)];
            }
            send_values.resize(send_displs[static_cast<size_t>(nranks)]);

            #pragma omp parallel for schedule(static)
            for (int c = 0; c < layout.chunks; ++c) {
                std::vector<size_t> at(static_cast<size_t>(nranks));
                for (int q = 0; q < nranks; ++q) {
                    at[static_cast<size_t>(q)] =
                        send_displs[static_cast<size_t>(q)] +
                        layout.base_elems[static_cast<size_t>(c) * nranks + q];
                }
                for (int i = layout.row_begin[static_cast<size_t>(c)];
                     i < layout.row_begin[static_cast<size_t>(c) + 1]; ++i) {
                    const int r_dim = ga.block_sizes[i];
                    for (int k = adj_ptr[i]; k < adj_ptr[i + 1]; ++k) {
                        const int owner = dest[static_cast<size_t>(k)];
                        if (owner == kTransposeSkip || owner == my_rank) continue;
                        const int c_dim = ga.block_sizes[ga.adj_ind[k]];
                        write_transposed_conjugate_block(
                            send_values.data() + at[static_cast<size_t>(owner)],
                            matrix.block_data(k), r_dim, c_dim);
                        at[static_cast<size_t>(owner)] += static_cast<size_t>(r_dim) * c_dim;
                    }
                }
            }

            // How many blocks each source contributes this round; the receiver
            // cannot derive it, because the slabs follow the sender's rows.
            std::vector<size_t> send_blocks(layout.group_blocks);
            std::vector<size_t> recv_blocks(static_cast<size_t>(nranks));
            MPI_Alltoall(send_blocks.data(), sizeof(size_t), MPI_BYTE,
                         recv_blocks.data(), sizeof(size_t), MPI_BYTE, ga.comm);

            // Values arrive per source in the same row order as the metadata,
            // so this round's blocks from source q are the next recv_blocks[q]
            // entries of q's metadata segment.
            std::vector<size_t> recv_counts(nranks), recv_displs(nranks + 1, 0);
            round_block.clear();
            round_offset.clear();
            for (int q = 0; q < nranks; ++q) {
                size_t group = 0;
                for (size_t j = 0; j < recv_blocks[static_cast<size_t>(q)]; ++j) {
                    const size_t b = recv_base[static_cast<size_t>(q)] + cursor[static_cast<size_t>(q)] + j;
                    round_block.push_back(b);
                    round_offset.push_back(recv_displs[static_cast<size_t>(q)] + group);
                    group += recv_elems[b];
                }
                recv_counts[static_cast<size_t>(q)] = group;
                recv_displs[static_cast<size_t>(q) + 1] = recv_displs[static_cast<size_t>(q)] + group;
            }
            recv_values.resize(recv_displs[static_cast<size_t>(nranks)]);

            std::vector<size_t> send_bytes(nranks), recv_bytes(nranks);
            std::vector<size_t> send_bdispls(nranks + 1), recv_bdispls(nranks + 1);
            for (int q = 0; q <= nranks; ++q) {
                send_bdispls[static_cast<size_t>(q)] = send_displs[static_cast<size_t>(q)] * sizeof(T);
                recv_bdispls[static_cast<size_t>(q)] = recv_displs[static_cast<size_t>(q)] * sizeof(T);
                if (q < nranks) {
                    send_bytes[static_cast<size_t>(q)] = send_counts[static_cast<size_t>(q)] * sizeof(T);
                    recv_bytes[static_cast<size_t>(q)] = recv_counts[static_cast<size_t>(q)] * sizeof(T);
                }
            }
            safe_alltoallv(send_values.data(), send_bytes, send_bdispls, MPI_BYTE,
                           recv_values.data(), recv_bytes, recv_bdispls, MPI_BYTE, ga.comm);

            // Unpack: the payload is already conjugate-transposed.
            const long long n_round = static_cast<long long>(round_block.size());
            #pragma omp parallel for schedule(static)
            for (long long j = 0; j < n_round; ++j) {
                const size_t b = round_block[static_cast<size_t>(j)];
                if (recv_slot[b] < 0) continue;
                std::memcpy(result.mutable_block_data(recv_slot[b]),
                            recv_values.data() + round_offset[static_cast<size_t>(j)],
                            recv_elems[b] * sizeof(T));
            }

            for (int q = 0; q < nranks; ++q) {
                cursor[static_cast<size_t>(q)] += recv_blocks[static_cast<size_t>(q)];
            }
        }
    }

    // ---- 6a. A's own blocks (mirror only) and every block whose transposed
    // image this rank owns: straight from A's storage to C's, no buffers.
    //
    // In chunks when consuming, so the source can be handed back as it is
    // finished with. 4 chunks, the same count filter_blocks and bsr.hpp's
    // consuming path use, and for the reason recorded there: a chunk count
    // derived from the thread count collapses to one on a small machine and
    // stops releasing anything, while a large one costs more in barriers than
    // it saves in overshoot.
    constexpr int kLocalChunks = 4;
    const int chunk_rows = consume_source != nullptr
                               ? std::max(1, (n_owned + kLocalChunks - 1) / kLocalChunks)
                               : n_owned;
    for (int chunk_begin = 0; chunk_begin < n_owned; chunk_begin += chunk_rows) {
        const int chunk_end = std::min(n_owned, chunk_begin + chunk_rows);
        write_local_rows(chunk_begin, chunk_end);
        if (consume_source != nullptr) {
            // Both passes are done with every row below chunk_end: 6b finished
            // above, and the loops in write_local_rows carry their own implicit
            // barrier, so no thread is still reading the range being released.
            consume_source->release_values_below_row(chunk_end);
        }
    }
    if (n_owned == 0) write_local_rows(0, 0);

    if (any_missing_dest) {
        throw std::runtime_error("transpose could not locate destination block");
    }
    return result;
}

template <typename Matrix>
struct CSRTransposeExecutor {
    static Matrix run(const Matrix& matrix) { return conjugate_transpose(matrix, false); }
};

template <typename Matrix>
struct BSRTransposeExecutor {
    static Matrix run(const Matrix& matrix) { return conjugate_transpose(matrix, false); }
};

template <typename Matrix>
struct VBCSRTransposeExecutor {
    static Matrix run(const Matrix& matrix) { return conjugate_transpose(matrix, false); }
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_OPS_TRANSPOSE_HPP

#ifndef VBCSR_DETAIL_OPS_SPMM_COMMON_HPP
#define VBCSR_DETAIL_OPS_SPMM_COMMON_HPP

#include "../../distributed/block_payload_types.hpp"
#include "../../distributed/mpi_utils.hpp"
#include "../../distributed/result_graph.hpp"
#include "../../kernels/dense_kernels.hpp"
#include "../../kernels/rowmajor_kernels.hpp"

#include <algorithm>
#include <cstdint>
#include <complex>
#include <type_traits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>
#include <mpi.h>
#include <omp.h>
#ifdef __linux__
#include <sys/mman.h>
#endif

namespace vbcsr {

struct BlockMeta {
    int col;
    double norm;
};

namespace detail {

/// Resident set size in GB, or 0 where /proc is unavailable.
///
/// For the SpGEMM profile lines: a phase table gives the peak of a whole
/// product, which is enough to know a product is the problem and not enough to
/// know WHICH of its stages allocates. Reading it at the same points the timing
/// stamps are taken turns one run into that answer.
inline double profile_rss_gb() {
    std::FILE* file = std::fopen("/proc/self/statm", "r");
    if (file == nullptr) return 0.0;
    long long total = 0, resident = 0;
    const int read = std::fscanf(file, "%lld %lld", &total, &resident);
    std::fclose(file);
    if (read != 2) return 0.0;
    // sysconf, not a hardcoded 4096: the statm figure is in PAGES, and this
    // silently misreported RSS anywhere the page size is not 4 KiB.
    static const double page_bytes = static_cast<double>(::sysconf(_SC_PAGESIZE));
    return static_cast<double>(resident) * page_bytes / 1073741824.0;
}


// ---------------------------------------------------------------------------
// Block-product dispatch, shared by every SpGEMM-shaped kernel here.
//
// ONE policy, in one place, because the choice is a property of the machine,
// the scalar type and the block size rather than of any executor: below the
// crossover the vendor's fixed per-call cost is not amortised by the flops in
// the call and the row-major intrinsic kernel wins; above it the vendor's
// throughput takes over.
//
// The threshold is PER SCALAR TYPE, which is what that amortisation argument
// predicts -- a complex block product carries about four times the flops of a
// real one at the same block size, so the two cross over in different places.
// End-to-end spmm on identical operands, batch forced off vs on, min of 9 runs,
// ratio = vendor / intrinsic, so below 1 the vendor wins:
//
//   complex<double>   bs 13  1.11   16  1.02   18  1.005  19  0.99   20  1.01
//                     bs 21  0.92   22  0.93   24  0.94   26  0.90   32  0.84
//   double            bs 13  1.89   16  1.57   20  1.23   22  1.13   26  1.00
//                     bs 32  1.13   40  1.10   48  1.00   64  1.08   96  1.05
//
// Complex crosses at bs 21 and the win grows with the block. Real NEVER crosses
// -- out to bs 96 the vendor batch is at best a wash -- so it stays off, and
// that is a measurement rather than an omission.
//
// Err HIGH when re-deriving this elsewhere: the two directions are not
// symmetric. Below the crossover the penalty runs to 2.2x, above it the gain
// tops out near 1.2x, so a threshold set too low costs far more than one set
// too high. VBCSR_SPGEMM_BATCH=1/0 forces it either way, which is how the
// table above is reproduced on a new machine.
template <typename T>
inline bool vendor_batch_profitable(int block_size) {
#ifdef VBCSR_BLAS_HAS_BATCH_GEMM
    // Resolved ONCE. This sits under grouped_block_gemm_batch, which runs per
    // group flush in the innermost contraction of every fused kernel, and a
    // getenv there is a locked lookup through the whole environment on every
    // block group. The cost of that is a process-lifetime override: a benchmark
    // toggles it by re-running, not in-process.
    static const int override_value = [] {
        const char* env = std::getenv("VBCSR_SPGEMM_BATCH");
        if (env == nullptr) return -1;
        return std::strcmp(env, "1") == 0 ? 1 : 0;
    }();
    if (override_value >= 0) return override_value == 1;
    if constexpr (std::is_same_v<T, std::complex<double>>) {
        return block_size >= 21;
    }
    (void)block_size;
    return false;
#else
    (void)block_size;
    return false;
#endif
}

// c_blocks[k] += a_block * b_blocks[k], for a group sharing ONE left operand
// and one square block size, through the vendor's grouped batch.
//
// The group shape is what makes this safe and worth doing: same A, same dims,
// DISTINCT destinations, so one call covers the group with no races and no
// packing -- pointer arrays and beta = 1 write straight into the accumulators.
// Both SpGEMM inner loops and both stages of the fused triple product produce
// exactly this shape naturally.
//
// Returns false when the vendor path does not apply (no batch support, wrong
// scalar type, group too small to pay for the call); the caller then loops.
template <typename T>
bool grouped_block_gemm_batch(int bs, const T* a_block,
                              const std::vector<const T*>& b_blocks,
                              const std::vector<T*>& c_blocks) {
#ifdef VBCSR_BLAS_HAS_BATCH_GEMM
    if constexpr (std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>) {
        const size_t n = b_blocks.size();
        // Below a handful of products the call setup outweighs the gain.
        constexpr size_t kMinBatch = 4;
        if (n < kMinBatch || !vendor_batch_profitable<T>(bs)) return false;
        thread_local std::vector<const T*> a_ptrs;
        a_ptrs.assign(n, a_block);
        const int trans[1] = {111 /*CblasNoTrans*/};
        const vbcsr_blas_int dims[1] = {static_cast<vbcsr_blas_int>(bs)};
        const vbcsr_blas_int group_size[1] = {static_cast<vbcsr_blas_int>(n)};
        if constexpr (std::is_same_v<T, double>) {
            const double alpha[1] = {1.0};
            const double beta[1] = {1.0};
            cblas_dgemm_batch(101 /*CblasRowMajor*/, trans, trans, dims, dims, dims,
                              alpha, const_cast<const double**>(a_ptrs.data()), dims,
                              const_cast<const double**>(b_blocks.data()), dims, beta,
                              const_cast<double**>(c_blocks.data()), dims, 1, group_size);
        } else {
            const T alpha(1.0);
            const T beta(1.0);
            cblas_zgemm_batch(101 /*CblasRowMajor*/, trans, trans, dims, dims, dims,
                              &alpha,
                              reinterpret_cast<const void**>(
                                  const_cast<const T**>(a_ptrs.data())), dims,
                              reinterpret_cast<const void**>(
                                  const_cast<const T**>(b_blocks.data())), dims,
                              &beta,
                              reinterpret_cast<void**>(
                                  const_cast<T**>(c_blocks.data())), dims,
                              1, group_size);
        }
        return true;
    } else {
        (void)bs; (void)a_block; (void)b_blocks; (void)c_blocks;
        return false;
    }
#else
    (void)bs; (void)a_block; (void)b_blocks; (void)c_blocks;
    return false;
#endif
}

// ---------------------------------------------------------------------------
// Row-at-a-time fused-kernel primitives.
//
// Shared by every kernel here that contracts a whole result row in scratch
// instead of materialising an intermediate matrix: the triple product
// (rarh.hpp) and the Hermitian square polynomial (square_polynomial.hpp). They
// live here rather than inside either executor because the two would otherwise
// hold identical copies and drift.

/// Same measure BlockSpMat::compute_block_norms uses, so a drop decided here is
/// bit-for-bit the one filter_blocks would have made afterwards.
template <typename T>
inline double block_squared_magnitude(const T& v) {
    if constexpr (std::is_same<T, std::complex<double>>::value ||
                  std::is_same<T, std::complex<float>>::value) {
        return std::norm(v);
    } else {
        return static_cast<double>(v) * static_cast<double>(v);
    }
}

/// dest(r x c) += lhs(r x m) * rhs(m x c), canonical row-major blocks, through
/// the same AVX2/FMA kernel the spmm executors use at these block sizes.
template <typename T>
inline void fused_gemm_accumulate(T* dest, const T* lhs, const T* rhs,
                                  int r_dim, int m_dim, int c_dim) {
    rowmajor_kernels::rm_gemm<T>(r_dim, m_dim, c_dim, lhs, rhs, c_dim, dest, c_dim);
}

/// Sparse accumulator over global block columns: a tag array for O(1) "have I
/// touched this column", a compact list of what was touched, and one value
/// arena. Cleared per row by walking the touched list, so the per-row cost never
/// scales with the global column count.
template <typename T>
struct FusedRowAccumulator {
    // Global column -> touched index, as a LAZILY-ALLOCATED page table rather
    // than a flat array. A flat tag array costs 4 bytes x N_GLOBAL per
    // accumulator regardless of what is touched, and the triple-product
    // kernels hold two accumulators per THREAD -- at millions of block rows
    // that is per-rank gigabytes growing with the global problem while the
    // work per rank shrinks. Pages allocate on first touch, so memory follows
    // the columns this thread actually reaches (its rows' halo), and the hot
    // path pays one indirection over the flat array.
    static constexpr int kSlotPageBits = 12;  // 4096 columns, 16 KB per page
    std::vector<std::unique_ptr<int[]>> slot_pages;
    std::vector<int> touched;          // global columns, in first-touch order
    std::vector<size_t> value_offset;  // per touched entry, offset into values
    std::vector<int> col_dim;          // per touched entry
    /// Running sum of |contribution| over the products accumulated into each
    /// entry, which upper-bounds its Frobenius norm by the triangle inequality.
    /// Carried instead of measuring the finished block: the factors are already
    /// in hand for the gate that needs it, so this is one add per pair where an
    /// exact norm is a full O(r*m) sweep per column.
    std::vector<double> norm_bound;
    std::vector<T> values;

    void resize(int n_global_blocks) {
        const size_t page = size_t(1) << kSlotPageBits;
        slot_pages.clear();
        slot_pages.resize(
            (static_cast<size_t>(std::max(0, n_global_blocks)) + page - 1) >> kSlotPageBits);
    }

    void clear() {
        for (int column : touched) slot_entry(column) = -1;
        touched.clear();
        value_offset.clear();
        col_dim.clear();
        norm_bound.clear();
        values.clear();
    }

    int& slot_entry(int global_column) {
        auto& page = slot_pages[static_cast<size_t>(global_column) >> kSlotPageBits];
        if (!page) {
            const size_t n = size_t(1) << kSlotPageBits;
            page.reset(new int[n]);
            std::fill(page.get(), page.get() + n, -1);
        }
        return page[static_cast<size_t>(global_column) & ((size_t(1) << kSlotPageBits) - 1)];
    }

    /// Returns the touched-entry SLOT, not the offset: `values` may reallocate
    /// here, so callers must re-read values.data() afterwards.
    int obtain(int global_column, int r_dim, int c_dim) {
        int& slot = slot_entry(global_column);
        if (slot < 0) {
            slot = static_cast<int>(touched.size());
            touched.push_back(global_column);
            value_offset.push_back(values.size());
            col_dim.push_back(c_dim);
            norm_bound.push_back(0.0);
            values.resize(values.size() + static_cast<size_t>(r_dim) * c_dim, T(0));
        }
        return slot;
    }
};

/// One contraction's worth of products against a single fixed left operand --
/// the grouped-batch shape, so above the block size where the vendor wins this
/// becomes one call instead of a loop.
///
/// Destinations are carried as OFFSETS, not pointers: obtain() may grow the
/// accumulator's arena, so a pointer taken when the product was queued can
/// dangle by the time the group is flushed. They are resolved at flush.
template <typename T>
struct FusedProductGroup {
    std::vector<const T*> rhs;
    std::vector<size_t> dest_offset;
    std::vector<int> c_dim;
    std::vector<T*> dest;        // scratch, rebuilt per flush
    bool uniform_square = true;  // every product bs x bs, as the batch needs

    void clear() {
        rhs.clear();
        dest_offset.clear();
        c_dim.clear();
        uniform_square = true;
    }
    void add(const T* block, size_t offset, int cols, int r_dim, int m_dim) {
        rhs.push_back(block);
        dest_offset.push_back(offset);
        c_dim.push_back(cols);
        if (cols != r_dim || m_dim != r_dim) uniform_square = false;
    }
    void flush(std::vector<T>& arena, const T* lhs, int r_dim, int m_dim) {
        if (rhs.empty()) return;
        dest.clear();
        dest.reserve(dest_offset.size());
        for (size_t off : dest_offset) dest.push_back(arena.data() + off);
        if (uniform_square && grouped_block_gemm_batch<T>(r_dim, lhs, rhs, dest)) return;
        for (size_t k = 0; k < rhs.size(); ++k) {
            fused_gemm_accumulate<T>(dest[k], lhs, rhs[k], r_dim, m_dim, c_dim[k]);
        }
    }
};

// SpGEMM results ship with the vendor's export order inside each row by
// default: no library consumer requires a matrix's own adjacency sorted
// (every lower_bound/binary_search target is a separately constructed,
// sorted-by-construction structure — audited in the migration plan Phase 5
// record), and the Python to_scipy boundary sorts scipy-side. Set
// VBCSR_SPGEMM_SORTED=1 to restore sorted-column output (per-row packed-key
// sort in the copy-out; still far cheaper than mkl_sparse_order).
inline bool spgemm_sorted_output_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("VBCSR_SPGEMM_SORTED");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

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
    std::vector<FetchedBlockRef<T>> owned_blocks;
    // Owns the remote payloads owned_blocks/rows point into, as the received
    // bytes (local blocks point into the source matrix; see FetchedBlockRef).
    std::vector<char> arena;
    GhostSizes sizes;
    std::map<int, std::vector<GhostBlockRef<T>>> rows;
};

// Row patterns (columns and block norms) of `B`, for an explicit set of global
// rows, fetched from their owners.
//
// Split out of exchange_ghost_metadata because a fused congruence needs this
// twice against different row sets: once for B over A's column support, then
// again for A over the support of A B -- a set that is only known after the
// first exchange, and that no single matrix's adjacency describes.
template <typename MatrixB>
GhostMetadata fetch_row_patterns(const MatrixB& B, const std::set<int>& needed_rows,
                                 MPI_Comm comm, int size, int rank) {
    GhostMetadata metadata;
    if (size == 1) {
        return metadata;
    }
    (void)rank;
    const MPI_Comm graph_comm = comm;

    std::vector<int> send_req_counts(size, 0);
    for (int global_row : needed_rows) {
        const int owner = B.graph->find_owner(global_row);
        send_req_counts[owner]++;
    }

    std::vector<int> recv_req_counts(size);
    MPI_Alltoall(send_req_counts.data(), 1, MPI_INT, recv_req_counts.data(), 1, MPI_INT, graph_comm);

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
                  recv_req_buf.data(), recv_req_counts.data(), rdispls.data(), MPI_INT, graph_comm);

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
                 recv_reply_bytes.data(), sizeof(size_t), MPI_BYTE, graph_comm);

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
                   recv_reply_blob.data(), recv_reply_bytes, rdispls_reply, MPI_BYTE, graph_comm);

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

// B's row patterns for every row A's columns reach. The original spelling,
// now one line over the primitive above.
template <typename MatrixA, typename MatrixB>
GhostMetadata exchange_ghost_metadata(const MatrixA& A, const MatrixB& B) {
    if (A.graph->size == 1) {
        return {};
    }
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
    return fetch_row_patterns(B, needed_rows, A.graph->comm, A.graph->size, rank);
}


// upper_only restricts the result to global block columns >= the row's
// global block index. The numeric phases resolve every product's destination
// through this symbolic pattern, so restricting it here is what skips the
// lower-triangle products' flops and their ghost fetches -- the caller
// (spmm_hermitian) reconstructs the lower triangle by conjugate transposition.
template <typename MatrixA, typename MatrixB>
SymbolicMultiplyResult symbolic_multiply_filtered(
    const MatrixA& A,
    const MatrixB& B,
    const GhostMetadata& meta,
    double threshold,
    bool upper_only = false) {
    SymbolicMultiplyResult res;
    const int n_rows = static_cast<int>(A.row_ptr().size()) - 1;
    res.c_row_ptr.resize(n_rows + 1);
    res.c_row_ptr[0] = 0;

    const auto& A_norms = A.get_block_norms();
    const auto& B_local_norms = B.get_block_norms();
    // Total global block columns: the ceiling on any row's distinct result
    // columns, and the cap applied to row_width_bound below.
    const int n_global_blocks =
        A.graph->block_displs.empty() ? 0 : A.graph->block_displs.back();

    std::vector<std::vector<int>> thread_cols(n_rows);
    const int max_threads = omp_get_max_threads();
    std::vector<std::set<BlockID>> thread_required(max_threads);

    struct SymbolicHashEntry {
        int key;
        double value;
        int tag;
    };
    // Per-thread tables grow to fit the widest row a thread actually sees: a
    // fixed capacity is a hard wall at high degree (a 500-neighbor graph's
    // C = A*B rows exceed any reasonable constant), while sum-of-B-degrees
    // over the A row is a cheap exact upper bound on the row's result width.
    constexpr size_t kInitialSymbolicHashSize = 8192;

    std::vector<std::vector<SymbolicHashEntry>> thread_tables(
        max_threads,
        std::vector<SymbolicHashEntry>(kInitialSymbolicHashSize, {-1, 0.0, 0}));
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
            const int start = A.row_ptr()[row];
            const int end = A.row_ptr()[row + 1];
            const int global_row = A.graph->get_global_index(row);

            // Upper bound on this row's distinct result columns.
            //
            // The sum below counts multiplication PATHS -- one per (A entry, B
            // entry) pair -- not distinct columns, and on a saturated row the
            // two differ by a factor of the row width. A 4096-block row that
            // reaches every other block accumulates 4096^2 = 16.8M, asks for a
            // 2N+1 table, rounds up to 67.1M entries, and at 24 bytes an entry
            // that is 1.5 GiB PER THREAD -- ~72 GiB across 48 threads for a row
            // that cannot hold more than 4096 distinct columns.
            //
            // Measured on a dense 1024-block row before this cap: 4.61 GB peak
            // against 64 MB of operands at 48 threads, and 0.49 GB at 4 -- the
            // giveaway that it scales with thread count, not with the problem.
            //
            // So cap it by what a row can actually contain. upper_only skips
            // every column below the diagonal before insertion, so its rows can
            // only hold what remains above it.
            const size_t max_distinct =
                upper_only
                    ? static_cast<size_t>(std::max(0, n_global_blocks - global_row))
                    : static_cast<size_t>(std::max(0, n_global_blocks));
            size_t row_width_bound = 0;
            for (int a_slot = start; a_slot < end; ++a_slot) {
                const int global_col_A = A.graph->get_global_index(A.col_ind()[a_slot]);
                if (A.graph->find_owner(global_col_A) == A.graph->rank) {
                    const int local_row_B = B.graph->global_to_local.at(global_col_A);
                    row_width_bound += static_cast<size_t>(
                        B.row_ptr()[local_row_B + 1] - B.row_ptr()[local_row_B]);
                } else {
                    auto it = meta.find(global_col_A);
                    if (it != meta.end()) {
                        row_width_bound += it->second.size();
                    }
                }
            }
            row_width_bound = std::min(row_width_bound, max_distinct);
            const size_t required = std::max<size_t>(
                kInitialSymbolicHashSize, 2 * row_width_bound + 1);
            if (required > table.size()) {
                size_t new_size = table.size();
                while (new_size < required) {
                    if (new_size > std::numeric_limits<size_t>::max() / 2) {
                        throw std::overflow_error("Symbolic hash table size overflow");
                    }
                    new_size <<= 1;
                }
                table.assign(new_size, {-1, 0.0, 0});
                tag = 0;  // fresh tags: every entry of the new table is tag 0
            }
            const size_t hash_mask = table.size() - 1;

            ++tag;
            if (tag == 0) {
                for (auto& entry : table) {
                    entry.tag = 0;
                }
                tag = 1;
            }
            touched.clear();

            for (int a_slot = start; a_slot < end; ++a_slot) {
                const int global_col_A = A.graph->get_global_index(A.col_ind()[a_slot]);
                const double norm_A = A_norms[a_slot];

                auto process_block = [&](int global_col_B, double norm_B) {
                    if (upper_only && global_col_B < global_row) {
                        return;
                    }
                    size_t h = static_cast<size_t>(global_col_B) & hash_mask;
                    size_t count = 0;
                    while (table[h].tag == tag) {
                        if (table[h].key == global_col_B) {
                            table[h].value += norm_A * norm_B;
                            return;
                        }
                        h = (h + 1) & hash_mask;
                        if (++count > table.size()) {
                            // Unreachable when the width bound holds; kept as
                            // a defensive invariant.
                            throw std::runtime_error("Hash table full in symbolic phase");
                        }
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

    // Block counts are int-indexed throughout the structural layer; C = A*B
    // can inflate block counts well past A's, so guard the 2^31 per-rank
    // ceiling explicitly instead of wrapping silently.
    int64_t running_blocks = 0;
    for (int row = 0; row < n_rows; ++row) {
        running_blocks += static_cast<int64_t>(thread_cols[row].size());
        if (running_blocks > static_cast<int64_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(
                "SpGEMM result exceeds 2^31 blocks on this rank; "
                "distribute over more ranks");
        }
        res.c_row_ptr[row + 1] = static_cast<int>(running_blocks);
    }
    res.c_col_ind.resize(static_cast<size_t>(res.c_row_ptr[n_rows]));
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < n_rows; ++row) {
        std::copy(thread_cols[row].begin(), thread_cols[row].end(),
                  res.c_col_ind.begin() + res.c_row_ptr[row]);
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
    ghost_blocks.arena = std::move(payload_ctx.arena);

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
            {block.global_col, block.data, block.c_dim, norm});
    }

    return ghost_blocks;
}

// ---------------------------------------------------------------------------
// Shared pieces of the row-at-a-time fused executors (rarh, square_polynomial,
// rhar). Each of these used to be a verbatim per-executor copy; a divergence in
// any of them is a SILENT value-transposition or wrong-fetch bug, not a compile
// error, so they are shared by construction rather than by review.

/// Rows fetched from other ranks, indexed by global row for O(1) reach in the
/// numeric loop. A map probe per block pair was not affordable: the inner
/// contraction runs over thousands of pairs per row.
///
/// `first` is -1 only for a row nobody answered. Payloads carry only NONEMPTY
/// rows, but the pattern reply names every row whose owner answered --
/// including rows that exist and hold zero blocks. Those must read as
/// found-with-nothing, not as missing: an empty row's contribution to a
/// contraction is zero, and treating it as absent turned it into a spurious
/// "not Hermitian" throw on a valid operand.
template <typename T>
struct FusedRemoteRows {
    std::vector<int> first;   // global row -> first entry; -1 means UNANSWERED
    std::vector<int> count;   // global row -> number of FETCHED entries
    // Global row -> the row's FULL pattern width, before any fetch gate. The
    // stage-one error budget divides by the width of the row it telescopes
    // over, and that bound is stated against the row as it exists, not
    // against the subset a gated fetch shipped -- dividing by the smaller
    // fetched count would loosen the numeric gate exactly where the fetch
    // gate already spent the budget.
    std::vector<int> pattern_width;
    std::vector<int> cols;    // global column of each entry
    std::vector<int> dims;    // its block column dimension
    std::vector<double> norms;  // Frobenius, from the pattern exchange
    std::vector<const T*> data;

    /// Row PRESENCE and full widths, which the patterns settle once for the
    /// whole call. Payloads arrive later and possibly in pieces (see
    /// FusedHaloStream); every row starts answered-with-nothing.
    void init(const GhostMetadata& patterns, int n_global) {
        first.assign(static_cast<size_t>(n_global), -1);
        count.assign(static_cast<size_t>(n_global), 0);
        pattern_width.assign(static_cast<size_t>(n_global), 0);
        cols.clear();
        dims.clear();
        norms.clear();
        data.clear();
        for (const auto& row : patterns) {
            first[static_cast<size_t>(row.first)] = 0;
            pattern_width[static_cast<size_t>(row.first)] =
                static_cast<int>(row.second.size());
        }
    }

    /// Index one delivery of payloads. Entries are APPENDED, so rows indexed by
    /// an earlier call keep pointing at their own arena.
    void append(const SpMMGhostBlocks<T>& ghosts) {
        for (const auto& entry : ghosts.rows) {
            first[static_cast<size_t>(entry.first)] = static_cast<int>(cols.size());
            count[static_cast<size_t>(entry.first)] = static_cast<int>(entry.second.size());
            for (const auto& block : entry.second) {
                cols.push_back(block.col);
                dims.push_back(block.c_dim);
                norms.push_back(block.norm);
                data.push_back(block.data);
            }
        }
    }

    /// Forget a row whose payload arena is going away: back to
    /// answered-with-nothing, which is what an unfetched row of a present
    /// pattern already reads as. The stale entries stay in the flat arrays --
    /// nothing indexes them once `count` is zero -- and they cost the
    /// metadata's 1% of a payload, not the payload.
    void drop_row(int global_row) {
        if (first[static_cast<size_t>(global_row)] < 0) return;
        first[static_cast<size_t>(global_row)] = 0;
        count[static_cast<size_t>(global_row)] = 0;
    }

    void build(const SpMMGhostBlocks<T>& ghosts, const GhostMetadata& patterns,
               int n_global) {
        init(patterns, n_global);
        append(ghosts);
    }
};

/// A halo delivered in pieces and released in pieces, with each remote block
/// crossing the network EXACTLY ONCE.
///
/// The predecessor tiled the output rows and re-fetched whatever the next tile
/// touched. That bounds residency but not traffic, and the two are not
/// symmetric: consecutive output rows are neighbouring atoms whose halos
/// overlap almost completely, so a tile's fetch is nearly the previous tile's
/// fetch again. Traffic became (tiles x halo) where the union is (1 x halo) --
/// measured at 6.2x on a 4-rank moire pattern and ~24x on an 8-rank 178k-atom
/// run, all of it in MPI with the cores idle.
///
/// What the patterns already know is enough to do better. They name every
/// block of every reached row before a single payload moves, so each remote
/// row's FIRST and LAST needing output row are computable up front. Fetch a
/// row when the round holding its first use begins; release it when the round
/// holding its last use ends. Then:
///
///   traffic  = the union reach, the irreducible minimum
///   resident = the LIVE set, what the pattern and the row order actually
///              force to coexist -- never the union, unless the order forces it
///
/// which dominates both the one-shot fetch (same traffic, union resident) and
/// the tiled fetch (same residency at best, multiplied traffic). The budget
/// stops being a policy that trades one against the other and becomes what it
/// should always have been: how much NEW payload a round may bring in.
///
/// The residency floor is then a property of the ordering, not of this class.
/// A locality-preserving order makes the live set a sliding window; a scattered
/// one keeps rows alive from the first output row to the last, and no fetch
/// schedule can shrink that. That is a partitioning problem, and reporting it
/// (see fused_halo_stats_enabled) is more honest than re-fetching to hide it.
template <typename T>
struct FusedHaloStream {
    FusedRemoteRows<T> rows;

    /// One round's delivery, held until the last row in it dies.
    struct Chunk {
        std::vector<char> arena;
        std::vector<int> row_ids;
        long long death_round = -1;
    };
    std::vector<Chunk> chunks;

    size_t fetched_blocks = 0;   // total crossing the network: the union
    size_t live_blocks = 0;      // resident now
    size_t peak_live_blocks = 0; // the live set's high-water mark

    void init(const GhostMetadata& patterns, int n_global) {
        rows.init(patterns, n_global);
    }

    /// Take a delivery, index it, and note when it may go.
    void absorb(SpMMGhostBlocks<T>&& ghosts, long long death_round) {
        Chunk chunk;
        chunk.death_round = death_round;
        chunk.row_ids.reserve(ghosts.rows.size());
        size_t blocks = 0;
        for (const auto& entry : ghosts.rows) {
            chunk.row_ids.push_back(entry.first);
            blocks += entry.second.size();
        }
        rows.append(ghosts);
        chunk.arena = std::move(ghosts.arena);
        fetched_blocks += blocks;
        live_blocks += blocks;
        peak_live_blocks = std::max(peak_live_blocks, live_blocks);
        chunks.push_back(std::move(chunk));
    }

    /// Release every delivery whose last reader has run.
    void retire(long long round) {
        size_t kept = 0;
        for (size_t c = 0; c < chunks.size(); ++c) {
            if (chunks[c].death_round > round) {
                if (kept != c) chunks[kept] = std::move(chunks[c]);
                ++kept;
                continue;
            }
            for (int g : chunks[c].row_ids) {
                live_blocks -= static_cast<size_t>(rows.count[static_cast<size_t>(g)]);
                rows.drop_row(g);
            }
            // Physically back to the OS, not to malloc's free list.
            release_and_drop(chunks[c].arena);
        }
        chunks.resize(kept);
    }
};

/// A dense numbering over the global block ids a rank actually touches.
///
/// The plan-time tables here were all indexed by GLOBAL block id, which costs
/// 57 bytes per global row per rank in the fused kernels -- 285 MB at five
/// million atoms, and not a byte less for adding ranks, because the array is
/// sized by the system rather than by the share. That is a ceiling rather than
/// a slowdown: it does not go away by scaling out.
///
/// The touched set is O(local + halo), orders of magnitude smaller, and for
/// plan-time tables a binary search is affordable -- these are walked once per
/// row or once per pattern, not once per block pair. (The numeric loop's
/// tables are a separate problem: it probes by global id thousands of times
/// per row, and the fix there is to resolve the dense id where the id is
/// STORED rather than where it is used.)
struct FusedDenseIds {
    std::vector<int> ids;  ///< sorted, unique

    void build(std::vector<int>& scratch) {
        std::sort(scratch.begin(), scratch.end());
        scratch.erase(std::unique(scratch.begin(), scratch.end()), scratch.end());
        ids.swap(scratch);
    }

    /// Dense slot of `g`, or -1 where this rank never named it.
    int of(int g) const {
        const auto it = std::lower_bound(ids.begin(), ids.end(), g);
        if (it == ids.end() || *it != g) return -1;
        return static_cast<int>(it - ids.begin());
    }

    size_t size() const { return ids.size(); }
};

/// Chunk of the numeric loop's `schedule(dynamic, ...)` over output rows.
///
/// Named because the round plan has to know it: a round below a few chunks per
/// thread cannot fill the team, and a floor derived from a stale copy of this
/// number would starve the loop silently.
inline constexpr int kFusedRowChunk = 8;


/// Blocks one round may hold, from a byte budget.
///
/// The whole budget, because a fetch now costs about what it delivers.
///
/// It used to cost three times that, and the budget was halved to cover it: a
/// rank held the blob it SERVES its peers, the blob it RECEIVES, and the typed
/// arena the received blob was unpacked into. Charging one of three is what
/// let a "62 GB" budget put well over 100 GB of transient buffers on a node
/// beside 176 GB of operands -- a peak under the limit on paper and over it in
/// practice.
///
/// Both extra copies are gone. The received blob IS the arena, and the served
/// response is streamed out through a fixed pool of slices rather than packed
/// whole, so what a rank holds to serve its peers is a hundred megabytes or so
/// whatever the rank count. Keeping the halving would now be charging twice
/// for a copy that no longer exists -- and the budget is the scarce thing: it
/// decides the round count, and the round count is superlinear in it (halving
/// the budget cost 4x the rounds on the moire pattern, not 2x).
inline size_t fused_block_budget(size_t budget_bytes, int bs_max,
                                 size_t scalar_bytes) {
    if (budget_bytes == 0) return 0;
    const size_t per_block = std::max<size_t>(
        1, static_cast<size_t>(bs_max) * static_cast<size_t>(bs_max) * scalar_bytes);
    return std::max<size_t>(1, budget_bytes / per_block);
}

/// The fetch schedule: where to cut rounds, and whether streaming can hold the
/// budget at all.
///
/// One simulation answers both, because they are the same question. Walk the
/// output rows carrying `held` (fetched, not yet released) and `releasable`
/// (dead, but held until a boundary lets it go). Cut a round exactly when the
/// next row's arrivals would put `held` over the budget -- that is the fewest
/// cuts that keep residency inside it, so rounds stay long and the numeric
/// loop's thread team stays fed. If a cut releases everything releasable and
/// `held` is STILL over, no release schedule can bound this fetch: the rows
/// are all wanted at once, and the caller must re-fetch per tile instead.
///
/// Deciding feasibility by simulation rather than by comparing the budget to
/// the theoretical live-set floor is the point. The floor is what a schedule
/// could achieve with infinitely fine rounds; what a real schedule achieves is
/// higher, because releases only happen at boundaries. Comparing against the
/// floor reported "stream is enough" and then streamed 1.09 GB through a
/// 0.75 GB budget -- the bound quietly not holding, which is how the previous
/// version of this got a job OOM-killed.
///
/// A budget of 0 is the caller declaring the reach fits: one round, no checks.
struct FusedFetchPlan {
    std::vector<int> bound{0};  ///< round boundaries, starting at row 0
    bool refetch = false;       ///< streaming cannot hold the budget
};

inline FusedFetchPlan fused_fetch_plan(const std::vector<size_t>& arrivals,
                                       const std::vector<int>& max_death_at,
                                       int n_rows, size_t block_budget) {
    FusedFetchPlan plan;
    if (block_budget == 0) return plan;

    // Simulate what the RUNTIME does, which is coarser than releasing each row
    // at its last reader: a round's arrivals share one arena, and that arena
    // goes only when the longest-lived row in it is done. Modelling per-row
    // release made the plan optimistic, and the bound then did not hold --
    // 1.09 GB streamed through a 0.75 GB budget, reported as if it fit.
    std::vector<std::pair<size_t, int>> open;  // (blocks, last reader row)
    size_t held = 0;      // blocks in closed rounds still resident
    size_t current = 0;   // blocks arriving for the round being built
    int current_death = -1;

    for (int i = 0; i < n_rows; ++i) {
        const size_t incoming = arrivals[static_cast<size_t>(i)];
        if (held + current + incoming > block_budget) {
            if (current > 0) {
                open.emplace_back(current, current_death);
                held += current;
                current = 0;
                current_death = -1;
            }
            size_t freed = 0;
            std::vector<std::pair<size_t, int>> keep;
            keep.reserve(open.size());
            for (const auto& chunk : open) {
                if (chunk.second < i) {
                    freed += chunk.first;
                } else {
                    keep.push_back(chunk);
                }
            }
            open.swap(keep);
            held -= freed;
            if (i > plan.bound.back()) plan.bound.push_back(i);
            if (held + incoming > block_budget) {
                // Nothing left to release and still over: streaming is out.
                plan.refetch = true;
                plan.bound.assign(1, 0);
                return plan;
            }
        }
        current += incoming;
        if (incoming > 0) {
            current_death = std::max(current_death, max_death_at[static_cast<size_t>(i)]);
        }
    }
    return plan;
}

/// Per-call halo accounting on stderr (VBCSR_FUSED_STATS).
///
/// Traffic, live set, round count and the paper's CV/memA ratio -- all four
/// computable from patterns before payloads move, and all four things whose
/// absence cost an 8-hour production run: 1396 rounds were silent, and the
/// scattered ordering that made the live set the whole union was invisible.
inline bool fused_halo_stats_enabled() {
    static const bool enabled = std::getenv("VBCSR_FUSED_STATS") != nullptr;
    return enabled;
}

/// One line per fused kernel call, from rank 0, worst rank in each column.
inline void fused_report_halo(const char* kernel, MPI_Comm comm, int rank,
                              long long rounds, size_t fetched_blocks,
                              size_t peak_live_blocks, size_t local_blocks,
                              size_t block_bytes, double secs_fetch,
                              double secs_numeric, size_t block_budget,
                              bool refetch) {
    if (!fused_halo_stats_enabled()) return;
    long long mine[3] = {static_cast<long long>(fetched_blocks),
                         static_cast<long long>(peak_live_blocks),
                         static_cast<long long>(local_blocks)};
    long long worst[3] = {0, 0, 0};
    MPI_Reduce(mine, worst, 3, MPI_LONG_LONG, MPI_MAX, 0, comm);
    double secs[2] = {secs_fetch, secs_numeric};
    double secs_worst[2] = {0.0, 0.0};
    MPI_Reduce(secs, secs_worst, 2, MPI_DOUBLE, MPI_MAX, 0, comm);
    double secs_min[2] = {0.0, 0.0}, secs_sum[2] = {0.0, 0.0};
    MPI_Reduce(secs, secs_min, 2, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(secs, secs_sum, 2, MPI_DOUBLE, MPI_SUM, 0, comm);
    int nranks = 1;
    MPI_Comm_size(comm, &nranks);
    if (rank != 0) return;
    const double to_gb = static_cast<double>(block_bytes) / 1073741824.0;
    const double cv_over_mem =
        worst[2] > 0 ? static_cast<double>(worst[0]) / static_cast<double>(worst[2]) : 0.0;
    std::fprintf(stderr,
                 "VBCSR_FUSED_STATS %s mode=%s budget=%.2f GB rounds=%lld "
                 "fetched=%.2f GB "
                 "live=%.2f GB local=%.2f GB CV/memA=%.0f%% fetch=%.1fs "
                 "numeric max/mean/min=%.1f/%.1f/%.1fs\n",
                 kernel,
                 block_budget == 0 ? "UNBOUNDED(VBCSR_FUSED_TILE_MB=0)"
                                   : (refetch ? "refetch"
                                              : (rounds == 1 ? "single" : "stream")),
                 static_cast<double>(block_budget) * to_gb,
                 rounds, static_cast<double>(worst[0]) * to_gb,
                 static_cast<double>(worst[1]) * to_gb,
                 static_cast<double>(worst[2]) * to_gb, cv_over_mem * 100.0,
                 secs_worst[0], secs_worst[1], secs_sum[1] / nranks, secs_min[1]);
}

/// Every block of the rows named in `patterns`, as the payload exchange wants
/// them.
inline std::vector<BlockID> fused_blocks_of(const GhostMetadata& patterns) {
    std::vector<BlockID> blocks;
    for (const auto& row : patterns) {
        for (const auto& meta : row.second) {
            blocks.push_back(BlockID{row.first, meta.col});
        }
    }
    return blocks;
}

/// The blocks of `patterns` whose norm reaches the row's fetch gate -- the
/// payload request a norm-gated halo fetch ships instead of fused_blocks_of.
///
/// The gate must be CONSERVATIVE against the caller's numeric gate: a block
/// skipped here is a block whose every use provably fails the numeric prune,
/// so the fetch changes what travels, never what the kernel would have kept.
/// (Row PRESENCE is untouched -- FusedRemoteRows::build takes it from the
/// patterns -- so probes that only need a row to have been answered, like
/// rarh's hermiticity check, are unaffected by an empty fetch.) A gate of 0
/// keeps everything, which is what a threshold of 0 must produce.
///
/// `min_col` trims columns no output on this rank can use: an upper-only
/// kernel writes C[i, j >= i] for OWNED i, so a fetched entry with a column
/// below the rank's first owned row is dead on arrival whatever its norm.
/// Only sound for fetches whose entries feed the OUTPUT-column position
/// (stage 2 of the triple products, the single stage of the square); a
/// contraction-index fetch must pass 0.
template <typename GateOfRow>
inline std::vector<BlockID> fused_gated_blocks_of(const GhostMetadata& patterns,
                                                  GateOfRow&& gate_of_row,
                                                  int min_col = 0) {
    std::vector<BlockID> blocks;
    for (const auto& row : patterns) {
        const double gate = gate_of_row(row.first);
        for (const auto& meta : row.second) {
            if (meta.col < min_col) continue;
            if (meta.norm >= gate) blocks.push_back(BlockID{row.first, meta.col});
        }
    }
    return blocks;
}

/// Bytes of fetched halo payload one fused-kernel tile round may hold.
///
/// How much NEW payload one fetch round may bring in (VBCSR_FUSED_TILE_MB;
/// 0 fetches the whole reach in a single round).
///
/// This bounds the round, not the residency: rows stay until their last reader
/// has run (FusedHaloStream), so the rounds partition the reach instead of
/// repeating it, and the count is ceil(union / budget) whatever the pattern
/// does. Residency is the live set, which the ordering fixes and no budget can
/// move. Smaller budgets buy finer release granularity at more rounds; they
/// cannot buy back a live set the partition insists on.
inline size_t fused_tile_budget_bytes(MPI_Comm comm = MPI_COMM_NULL) {
    // Explicit wins, including 0 -- which means "hold the whole reach, I know
    // it fits". That is a real option and a loaded gun: it is what a 178k-atom
    // job was run with, and holding the union halo took the node past 250 GiB
    // and got it OOM-killed.
    const char* env = std::getenv("VBCSR_FUSED_TILE_MB");
    if (env != nullptr) {
        const long long mb = std::atoll(env);
        if (mb <= 0) return 0;
        return static_cast<size_t>(mb) << 20;
    }

    // Otherwise derive it from the memory that is actually FREE, now.
    //
    // MemTotal, read once, was wrong in the way that matters: it answers "what
    // could a rank have had" and the halo is fetched when the loop is already
    // holding its operands and staging its result. On a 250 GiB node it kept
    // offering 62 GiB to a kernel whose operands and output already needed
    // ~210. MemAvailable, read per call, answers "what is left", so the budget
    // shrinks as the step fills up.
    //
    // Still only the halo. The operands, the staged result and the rest of the
    // Newton-Schulz working set are outside it, so this bounds one term of the
    // footprint and cannot rescue a rank count that is too small for the rest.
    static const int local_ranks = [comm]() -> int {
        int ranks = 1, initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized && comm != MPI_COMM_NULL) {
            MPI_Comm shared = MPI_COMM_NULL;
            if (MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                                    &shared) == MPI_SUCCESS &&
                shared != MPI_COMM_NULL) {
                MPI_Comm_size(shared, &ranks);
                MPI_Comm_free(&shared);
            }
        }
        return std::max(1, ranks);
    }();

    size_t available = 0;
    if (std::FILE* f = std::fopen("/proc/meminfo", "r")) {
        char label[64];
        unsigned long long kb = 0;
        while (std::fscanf(f, "%63s %llu kB\n", label, &kb) == 2) {
            if (std::strncmp(label, "MemAvailable", 12) == 0) {
                available = static_cast<size_t>(kb) * 1024;
                break;
            }
        }
        std::fclose(f);
    }
    if (available == 0) return size_t(512) << 20;
    return std::max<size_t>(size_t(256) << 20,
                            available / static_cast<size_t>(local_ranks) / 4);
}

/// Output-column tiling of the fused numeric pass (VBCSR_FUSED_OUTPUT_TILE,
/// default on; "0" disables -- the A/B lever the prototype was measured
/// with). Wide accumulator rows spill every cache and pin the numeric loop
/// to the DRAM roof; slicing the DESTINATION columns to an L2-resident range
/// and re-walking the contraction per slice touches each destination block
/// once per slice pass instead of once per pair, while the operand PAYLOADS
/// are still read exactly once -- the re-walk skips out-of-slice blocks on
/// metadata alone (12 bytes against a 2.7 KB payload).
inline bool fused_output_tiling_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("VBCSR_FUSED_OUTPUT_TILE");
        return v == nullptr || v[0] == '\0' || v[0] != '0';
    }();
    return enabled;
}

/// Destination-column slice width, in BLOCK columns, for an accumulator slice
/// that stays in L2.
///
/// The invariant is bytes, not blocks: one slice is `cols * r_dim * c_dim *
/// sizeof(T)`. Hardcoding 384 encoded three things at once -- this machine's
/// L2, 13x13 blocks, and complex<double> -- and is wrong on any of BSR at
/// bs 32, a real scalar type, or a different cache. Derive it instead, from
/// the cache the machine reports and the block size actually in hand.
inline int fused_output_tile_cols(int block_dim, size_t scalar_bytes) {
    static const size_t l2_bytes = [] {
#ifdef _SC_LEVEL2_CACHE_SIZE
        const long reported = ::sysconf(_SC_LEVEL2_CACHE_SIZE);
        if (reported > 0) return static_cast<size_t>(reported);
#endif
        return static_cast<size_t>(1) << 20;
    }();
    const size_t per_col =
        std::max<size_t>(1, static_cast<size_t>(block_dim) *
                                static_cast<size_t>(block_dim) * scalar_bytes);
    const size_t cols = l2_bytes / per_col;
    // Below a few dozen columns the re-walk per slice costs more than the
    // locality it buys, whatever the cache says.
    return static_cast<int>(std::max<size_t>(64, std::min<size_t>(cols, 1 << 20)));
}

/// Round boundaries need no minimum row count.
///
/// They did while a round re-fetched its whole tile: a one-row tile then paid
/// a full halo for one row's work, and a 16-row floor was what kept a 64 MB
/// budget on a 4-rank band from turning a 2.9 s product into 29 s. Charging
/// rounds for NEW payload only removes the reason -- a round that opens on an
/// expensive row fetches that row's newcomers and nothing it already holds --
/// and the floor would now do harm, forcing 16 rows' arrivals through a
/// boundary meant to cap one budget's worth.

/// Global block index -> local row, or -1 where the matrix does not own it.
template <typename Matrix>
std::vector<int> fused_row_of_global(const Matrix& m, int n_global_blocks) {
    std::vector<int> map(static_cast<size_t>(n_global_blocks), -1);
    const int n_rows = static_cast<int>(m.graph->adj_ptr.size()) - 1;
    for (int i = 0; i < n_rows; ++i) {
        map[static_cast<size_t>(m.graph->get_global_index(i))] = i;
    }
    return map;
}

/// Largest block norm over the whole team. Global rather than per row because
/// a stage-one neglect is later multiplied by a block belonging to a row this
/// rank may not own, so only a global bound is sound.
template <typename Matrix>
double fused_global_max_norm(const Matrix& A) {
    const auto& norms = A.get_block_norms();
    double local = 0.0;
    for (double n : norms) local = std::max(local, n);
    if (A.graph->size <= 1) return local;
    double global = local;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, A.graph->comm);
    return global;
}

/// The one drop, on finished values, staging the survivors BY COLUMN and
/// EXACTLY SIZED -- which is what lets fused_assemble copy straight out of the
/// staged rows. Two passes rather than one, and the reason is memory rather
/// than clarity:
///
///   * By column, because the accumulator hands slots back in first-touch
///     order and the result graph is ordered. Reordering in assemble had to
///     build a second buffer per row and swap -- an extra copy of the whole
///     answer, and, since the discarded buffers are over the allocator's
///     dynamic mmap threshold at these row sizes, one glibc keeps rather than
///     returns. Measured at 4096 blocks x 200 neighbours: +2.20 GB high water
///     against a 5.72 GB result, 1.90 GB of it left resident, to permute and
///     nothing else. Sorting the surviving INDICES costs nothing beside it.
///   * Exactly sized, because push_back growth left the staged rows at
///     9.70 GB of capacity for 5.72 GB of answer, and every doubling frees a
///     buffer the allocator then holds.
///
/// ORDER INVARIANT, stated because nothing asserts it: the sort below is by
/// GLOBAL column, while construct_result_graph orders each row's adjacency by
/// LOCAL id -- owned columns (ascending global) first, then ghosts (by owner,
/// then global). The two agree only because (a) ownership is
/// contiguous-ascending in global id (find_owner binary-searches
/// block_displs), and (b) the output is UPPER-ONLY: an owned row's columns
/// are all >= its own global id, so every ghost column in the row is above
/// the owned range and sorts after every owned column in BOTH orders. A
/// full-pattern kernel reusing this staging would break (b) -- its ghost
/// columns below the owned range sort first globally but after the owned
/// locals -- and the positional copy in fused_assemble would silently
/// transpose values onto wrong columns in distributed runs.
template <typename T>
inline void stage_row_in_column_order(const FusedRowAccumulator<T>& acc, int r_dim,
                                      double threshold, std::vector<int>& keep_order,
                                      std::vector<int>& keep_columns,
                                      std::vector<T>& keep_values) {
    keep_order.clear();
    size_t keep_elems = 0;
    for (size_t s = 0; s < acc.touched.size(); ++s) {
        const size_t count = static_cast<size_t>(r_dim) * acc.col_dim[s];
        if (threshold > 0.0) {
            const T* block = acc.values.data() + acc.value_offset[s];
            double sq = 0.0;
            for (size_t e = 0; e < count; ++e) sq += block_squared_magnitude<T>(block[e]);
            if (std::sqrt(sq) < threshold) continue;
        }
        keep_order.push_back(static_cast<int>(s));
        keep_elems += count;
    }
    std::sort(keep_order.begin(), keep_order.end(), [&](int x, int y) {
        return acc.touched[static_cast<size_t>(x)] <
               acc.touched[static_cast<size_t>(y)];
    });
    keep_columns.reserve(keep_order.size());
    keep_values.reserve(keep_elems);
    for (int idx : keep_order) {
        const size_t s = static_cast<size_t>(idx);
        const size_t count = static_cast<size_t>(r_dim) * acc.col_dim[s];
        const T* block = acc.values.data() + acc.value_offset[s];
        keep_columns.push_back(acc.touched[s]);
        keep_values.insert(keep_values.end(), block, block + count);
    }
}

/// Everything after a fused kernel's numeric pass: build the graph the
/// surviving pattern defines and copy the staged values in, positionally.
///
/// The staged pattern IS the result pattern (stage_row_in_column_order's
/// invariant above), so the copy is positional. Deferred first touch: the
/// plain `Matrix C(graph_c)` ran the zero pass over the whole result --
/// making it fully resident -- at the one moment the staged rows still hold
/// the entire answer too. Every block is written by the copy, so there is
/// nothing for the pass to initialise that the copy does not.
///
/// Each row's staging buffer is released the instant it has been copied --
/// to the KERNEL, not just to malloc; see release_and_drop above for why the
/// difference is most of the peak.
///
/// Walked by THREAD DOMAIN, not a dynamic row schedule: this copy is the
/// deferred first touch, so the thread that runs a row decides which node its
/// pages live on, and the forward apply splits this matrix along
/// thread_domains.
template <typename Matrix>
Matrix fused_assemble(const Matrix& A,
                      std::vector<std::vector<int>>& row_columns,
                      std::vector<std::vector<typename Matrix::value_type>>& row_values,
                      const GhostSizes& ghost_sizes, const char* what) {
    using T = typename Matrix::value_type;
    const DistGraph& ga = *A.graph;
    const int n_rows = static_cast<int>(ga.adj_ptr.size()) - 1;

    DistGraph* graph_c = construct_result_graph(A, row_columns, ghost_sizes, what);
    Matrix C = Matrix::make_overwritten_result(graph_c, A.configured_page_size());

    const auto& domains = C.thread_domain_partition();
    const bool aligned = domains.thread_count > 0 &&
                         static_cast<int>(domains.row_bounds.size()) ==
                             domains.thread_count + 1;
    #pragma omp parallel for schedule(static) if (aligned)
    for (int d = 0; d < (aligned ? domains.thread_count : 1); ++d) {
        const int row_lo = aligned ? domains.domain_begin(d) : 0;
        const int row_hi = aligned ? domains.domain_end(d) : n_rows;
        for (int i = row_lo; i < row_hi; ++i) {
            int slot = graph_c->adj_ptr[i];
            size_t at = 0;
            for (size_t s = 0; s < row_columns[i].size(); ++s, ++slot) {
                const size_t count = C.block_size_elements(slot);
                std::memcpy(C.mutable_block_data(slot),
                            row_values[i].data() + at, count * sizeof(T));
                at += count;
            }
            release_and_drop(row_values[i]);
        }
    }
    return C;
}

} // namespace detail
} // namespace vbcsr

#endif // VBCSR_DETAIL_OPS_SPMM_COMMON_HPP

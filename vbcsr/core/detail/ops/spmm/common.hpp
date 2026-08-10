#ifndef VBCSR_DETAIL_OPS_SPMM_COMMON_HPP
#define VBCSR_DETAIL_OPS_SPMM_COMMON_HPP

#include "../../distributed/block_payload_types.hpp"
#include "../../distributed/mpi_utils.hpp"
#include "../../kernels/dense_kernels.hpp"
#include "../../kernels/rowmajor_kernels.hpp"

#include <algorithm>
#include <cstdint>
#include <complex>
#include <type_traits>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <limits>
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
// ONE policy, in one place, because the choice is a property of the machine and
// the block size rather than of any executor: below the crossover the vendor's
// fixed per-call cost is not amortised by a bs^3 product and the row-major
// intrinsic kernel wins; above it the vendor's throughput takes over.
// Measured (benchmark_spgemm, complex, dense):
//
//     bs  5   0.50x        bs 13   0.76-1.04x        bs 26   1.22-1.30x
//
// so the switch sits at bs >= 16. VBCSR_SPGEMM_BATCH=1/0 forces it either way,
// re-read per call so a benchmark can toggle it in-process.
inline bool vendor_batch_profitable(int block_size) {
#ifdef VBCSR_BLAS_HAS_BATCH_GEMM
    // Resolved ONCE. This sits under grouped_block_gemm_batch, which runs per
    // group flush in the innermost contraction of every fused kernel, and a
    // getenv there is a locked lookup through the whole environment on every
    // block group.
    static const int override_value = [] {
        const char* env = std::getenv("VBCSR_SPGEMM_BATCH");
        if (env == nullptr) return -1;
        return std::strcmp(env, "1") == 0 ? 1 : 0;
    }();
    if (override_value >= 0) return override_value == 1;
    return block_size >= 16;
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
        if (n < kMinBatch || !vendor_batch_profitable(bs)) return false;
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
    std::vector<int> slot_of_column;   // global column -> touched index, or -1
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
        slot_of_column.assign(static_cast<size_t>(n_global_blocks), -1);
    }

    void clear() {
        for (int column : touched) slot_of_column[static_cast<size_t>(column)] = -1;
        touched.clear();
        value_offset.clear();
        col_dim.clear();
        norm_bound.clear();
        values.clear();
    }

    /// Returns the touched-entry SLOT, not the offset: `values` may reallocate
    /// here, so callers must re-read values.data() afterwards.
    int obtain(int global_column, int r_dim, int c_dim) {
        int& slot = slot_of_column[static_cast<size_t>(global_column)];
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
    std::vector<FetchedBlock<T>> owned_blocks;
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

#endif // VBCSR_DETAIL_OPS_SPMM_COMMON_HPP

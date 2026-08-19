#ifndef VBCSR_DETAIL_OPS_SPMM_RHAR_HPP
#define VBCSR_DETAIL_OPS_SPMM_RHAR_HPP

#include "../../distributed/block_payload_exchange.hpp"
#include "../../distributed/result_graph.hpp"
#include "../transpose.hpp"
#include "common.hpp"
#include "../../../scalar_traits.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <unordered_map>
#include <stdexcept>
#include <vector>

namespace vbcsr::detail {

// C = A^H B A, upper triangle only, computed one result row at a time, WITHOUT
// ever forming A^H or A^H B.
//
// The sibling of rarh (A B A^H): PETSc carries the same pair as MatRARt and
// MatPtAP. The two are NOT one kernel with a flag, because they rest on
// different operand contracts. rarh reaches its second contraction through
// A[j,l]^H == A[l,j], so it REQUIRES A Hermitian; this kernel requires nothing
// of A -- which is exactly what the Newton-Schulz reform X = Z^H S Z needs,
// where Z is a thresholded polynomial of S and only approximately Hermitian.
// Only B must be Hermitian, so that C is and half of it can be skipped.
//
// Why this is a kernel and not transpose-then-multiply:
//
// The route it replaces materialised TWO matrices the answer never needed:
// a full copy of A^H (spmm's transA path is "transpose, then multiply"), and
// the intermediate B A (or A^H B). On a Newton-Schulz reform both are
// Z-sized -- the densest matrix in the loop -- and both stood beside the
// operands and the answer at the step's peak. Here row i of C is
// (column i of A)^H B A, held one row per thread:
//
//   * Column i of A is not a materialised object. Its LOCAL blocks -- the
//     overwhelming majority on a locality-preserving partition -- are read in
//     place through a transpose index and conjugate-transposed block by block
//     into thread scratch, an O(dim^2) copy against the O(dim^3) products it
//     feeds. Its BOUNDARY blocks (rows owned elsewhere) are pushed by their
//     owners, already conjugate-transposed, in one exchange whose volume is
//     the partition surface -- the same blocks a full transpose would have
//     communicated, without the local copy a full transpose would have made.
//   * The second contraction reads ROWS of A in their natural layout, the
//     same remote-row fetch rarh already does for its outer factor.
//
// Accuracy carries rarh's contract unchanged: pairs are skipped only when a
// per-row budget proves they cannot matter, and the threshold is applied once,
// to finished values -- where the two-product route it replaces dropped at the
// intermediate and then again at the end.
//
// Result: block columns at or above the diagonal, every block's Frobenius
// norm >= threshold. Complete it with complete_hermitian_inplace().
template <typename Matrix>
struct RhARExecutor {
    using T = typename Matrix::value_type;
    // Row-scratch primitives and remote-row plumbing shared with rarh and
    // square_polynomial; see common.hpp.
    using RowAccumulator = FusedRowAccumulator<T>;
    using ProductGroup = FusedProductGroup<T>;
    using RemoteRows = FusedRemoteRows<T>;

    // The boundary of A^H: blocks A[k,i] whose row k lives on another rank and
    // whose column i lives here. Pushed by the row owners -- the receiver
    // cannot request them, because the transpose pattern is exactly what it
    // does not know -- and pushed already conjugate-transposed, so the numeric
    // loop reads them like any other left operand.
    //
    // One whole exchange, not byte-bounded rounds: the volume is the partition
    // surface, the same order as the ghost-row fetches below, which are also
    // whole. The matrix-sized transfers that needed rounds (transpose.hpp) are
    // precisely what this kernel exists to avoid.
    struct PushedBlocks {
        std::vector<int> dest_col;   // global column i; this rank owns it
        std::vector<int> src_row;    // global row k
        std::vector<int> src_dim;    // k's block dimension
        std::vector<double> norms;   // |A[k,i]|_F, from the sender's table
        std::vector<size_t> offset;  // into values
        std::vector<T> values;       // A[k,i]^H, canonical row-major
    };

    static PushedBlocks push_boundary_columns(const Matrix& A) {
        const DistGraph& ga = *A.graph;
        const int rank = ga.rank;
        const int nranks = ga.size;
        const int n_rows = static_cast<int>(ga.adj_ptr.size()) - 1;
        const auto& a_norms = A.get_block_norms();

        // Classify local blocks by the owner of their column.
        std::vector<size_t> send_blocks(static_cast<size_t>(nranks), 0);
        std::vector<size_t> send_elems(static_cast<size_t>(nranks), 0);
        for (int i = 0; i < n_rows; ++i) {
            const int r_dim = ga.block_sizes[i];
            for (int k = ga.adj_ptr[i]; k < ga.adj_ptr[i + 1]; ++k) {
                const int g_col = ga.get_global_index(ga.adj_ind[k]);
                const int owner = ga.find_owner(g_col);
                if (owner == rank) continue;
                send_blocks[static_cast<size_t>(owner)] += 1;
                send_elems[static_cast<size_t>(owner)] +=
                    static_cast<size_t>(r_dim) * ga.block_sizes[ga.adj_ind[k]];
            }
        }

        std::vector<size_t> recv_blocks(static_cast<size_t>(nranks), 0);
        std::vector<size_t> recv_elems(static_cast<size_t>(nranks), 0);
        MPI_Alltoall(send_blocks.data(), sizeof(size_t), MPI_BYTE,
                     recv_blocks.data(), sizeof(size_t), MPI_BYTE, ga.comm);
        MPI_Alltoall(send_elems.data(), sizeof(size_t), MPI_BYTE,
                     recv_elems.data(), sizeof(size_t), MPI_BYTE, ga.comm);

        // Pack: metadata as 3 ints per block, norms separately, payload
        // conjugate-transposed at the sender so the receiver only reads.
        // All three displacement tables are kept in the SAME unit as their
        // count tables (blocks for meta and norms, elements for payload), so
        // the byte conversion below is one multiply for counts and
        // displacements alike. The first version of this kept meta_displs in
        // ints (blocks * 3) and multiplied it by the 12-byte per-block unit --
        // a 3x displacement error that read past the send buffer for every
        // destination after the first, and passed the small-matrix test on
        // heap-garbage luck before a 2000-block chain caught it.
        std::vector<size_t> meta_displs(static_cast<size_t>(nranks) + 1, 0);
        std::vector<size_t> norm_displs(static_cast<size_t>(nranks) + 1, 0);
        std::vector<size_t> elem_displs(static_cast<size_t>(nranks) + 1, 0);
        for (int q = 0; q < nranks; ++q) {
            meta_displs[static_cast<size_t>(q) + 1] =
                meta_displs[static_cast<size_t>(q)] + send_blocks[static_cast<size_t>(q)];
            norm_displs[static_cast<size_t>(q) + 1] =
                norm_displs[static_cast<size_t>(q)] + send_blocks[static_cast<size_t>(q)];
            elem_displs[static_cast<size_t>(q) + 1] =
                elem_displs[static_cast<size_t>(q)] + send_elems[static_cast<size_t>(q)];
        }
        std::vector<int> send_meta(meta_displs[static_cast<size_t>(nranks)] * 3);
        std::vector<double> send_norms(norm_displs[static_cast<size_t>(nranks)]);
        std::vector<T> send_values(elem_displs[static_cast<size_t>(nranks)]);
        {
            std::vector<size_t> meta_at(static_cast<size_t>(nranks));
            for (int q = 0; q < nranks; ++q) {
                meta_at[static_cast<size_t>(q)] = meta_displs[static_cast<size_t>(q)] * 3;
            }
            std::vector<size_t> norm_at(norm_displs.begin(), norm_displs.end() - 1);
            std::vector<size_t> elem_at(elem_displs.begin(), elem_displs.end() - 1);
            for (int i = 0; i < n_rows; ++i) {
                const int g_row = ga.get_global_index(i);
                const int r_dim = ga.block_sizes[i];
                for (int k = ga.adj_ptr[i]; k < ga.adj_ptr[i + 1]; ++k) {
                    const int g_col = ga.get_global_index(ga.adj_ind[k]);
                    const int owner = ga.find_owner(g_col);
                    if (owner == rank) continue;
                    const int c_dim = ga.block_sizes[ga.adj_ind[k]];
                    int* meta = send_meta.data() + meta_at[static_cast<size_t>(owner)];
                    meta[0] = g_col;   // destination column = receiver's row
                    meta[1] = g_row;   // source row = the contraction index
                    meta[2] = r_dim;   // its dimension
                    meta_at[static_cast<size_t>(owner)] += 3;
                    send_norms[norm_at[static_cast<size_t>(owner)]++] =
                        a_norms[static_cast<size_t>(k)];
                    write_transposed_conjugate_block(
                        send_values.data() + elem_at[static_cast<size_t>(owner)],
                        A.block_data(k), r_dim, c_dim);
                    elem_at[static_cast<size_t>(owner)] +=
                        static_cast<size_t>(r_dim) * c_dim;
                }
            }
        }

        std::vector<size_t> rmeta_displs(static_cast<size_t>(nranks) + 1, 0);
        std::vector<size_t> rnorm_displs(static_cast<size_t>(nranks) + 1, 0);
        std::vector<size_t> relem_displs(static_cast<size_t>(nranks) + 1, 0);
        for (int q = 0; q < nranks; ++q) {
            rmeta_displs[static_cast<size_t>(q) + 1] =
                rmeta_displs[static_cast<size_t>(q)] + recv_blocks[static_cast<size_t>(q)];
            rnorm_displs[static_cast<size_t>(q) + 1] =
                rnorm_displs[static_cast<size_t>(q)] + recv_blocks[static_cast<size_t>(q)];
            relem_displs[static_cast<size_t>(q) + 1] =
                relem_displs[static_cast<size_t>(q)] + recv_elems[static_cast<size_t>(q)];
        }

        PushedBlocks pushed;
        const size_t n_recv = rnorm_displs[static_cast<size_t>(nranks)];
        std::vector<int> recv_meta(rmeta_displs[static_cast<size_t>(nranks)] * 3);
        pushed.norms.resize(n_recv);
        pushed.values.resize(relem_displs[static_cast<size_t>(nranks)]);
        {
            std::vector<size_t> sc(static_cast<size_t>(nranks)), sd(static_cast<size_t>(nranks) + 1);
            std::vector<size_t> rc(static_cast<size_t>(nranks)), rd(static_cast<size_t>(nranks) + 1);
            auto bytes = [&](const std::vector<size_t>& blocks_or_elems, size_t unit,
                             std::vector<size_t>& counts, const std::vector<size_t>& displs,
                             std::vector<size_t>& bdispls) {
                for (int q = 0; q < nranks; ++q) {
                    counts[static_cast<size_t>(q)] = blocks_or_elems[static_cast<size_t>(q)] * unit;
                    bdispls[static_cast<size_t>(q)] = displs[static_cast<size_t>(q)] * unit;
                }
                bdispls[static_cast<size_t>(nranks)] = displs[static_cast<size_t>(nranks)] * unit;
            };
            bytes(send_blocks, 3 * sizeof(int), sc, meta_displs, sd);
            bytes(recv_blocks, 3 * sizeof(int), rc, rmeta_displs, rd);
            safe_alltoallv(send_meta.data(), sc, sd, MPI_BYTE,
                           recv_meta.data(), rc, rd, MPI_BYTE, ga.comm);
            bytes(send_blocks, sizeof(double), sc, norm_displs, sd);
            bytes(recv_blocks, sizeof(double), rc, rnorm_displs, rd);
            safe_alltoallv(send_norms.data(), sc, sd, MPI_BYTE,
                           pushed.norms.data(), rc, rd, MPI_BYTE, ga.comm);
            bytes(send_elems, sizeof(T), sc, elem_displs, sd);
            bytes(recv_elems, sizeof(T), rc, relem_displs, rd);
            safe_alltoallv(send_values.data(), sc, sd, MPI_BYTE,
                           pushed.values.data(), rc, rd, MPI_BYTE, ga.comm);
        }

        pushed.dest_col.resize(n_recv);
        pushed.src_row.resize(n_recv);
        pushed.src_dim.resize(n_recv);
        pushed.offset.resize(n_recv);
        size_t at = 0;
        for (size_t b = 0; b < n_recv; ++b) {
            pushed.dest_col[b] = recv_meta[b * 3];
            pushed.src_row[b] = recv_meta[b * 3 + 1];
            pushed.src_dim[b] = recv_meta[b * 3 + 2];
            pushed.offset[b] = at;
            const int local = ga.global_to_local.count(pushed.dest_col[b])
                                  ? ga.global_to_local.at(pushed.dest_col[b])
                                  : -1;
            if (local < 0 || local >= n_rows) {
                throw std::runtime_error(
                    "rhar_upper: pushed block addresses a column this rank does not own");
            }
            at += static_cast<size_t>(ga.block_sizes[local]) * pushed.src_dim[b];
        }
        if (at != pushed.values.size()) {
            throw std::runtime_error(
                "rhar_upper: pushed payload does not match its metadata");
        }
        return pushed;
    }

    // Row i of A^H, assembled without materialising A^H: local blocks of A
    // reached in place through this transpose INDEX (conjugate-transposed at
    // use, block by block, into thread scratch), pushed boundary blocks read
    // as they arrived. CSR-style over owned output rows; `transposed` says
    // which reading a data pointer needs.
    struct ColumnView {
        std::vector<int> ptr;         // n_rows + 1
        std::vector<int> src_row;     // global k: this entry is A^H[i,k]
        std::vector<int> src_dim;     // k's block dimension
        std::vector<double> norm;     // |A^H[i,k]|_F
        std::vector<const T*> data;
        std::vector<char> transposed; // 1: stored as A^H[i,k] (pushed)
                                      // 0: stored as A[k,i], transpose at use
    };

    static ColumnView build_column_view(const Matrix& A, const std::vector<int>& a_row,
                                        const PushedBlocks& pushed) {
        const DistGraph& ga = *A.graph;
        const int rank = ga.rank;
        const int n_rows = static_cast<int>(ga.adj_ptr.size()) - 1;
        const auto& a_norms = A.get_block_norms();

        ColumnView view;
        std::vector<int> counts(static_cast<size_t>(n_rows), 0);
        for (int i = 0; i < n_rows; ++i) {
            for (int k = ga.adj_ptr[i]; k < ga.adj_ptr[i + 1]; ++k) {
                const int g_col = ga.get_global_index(ga.adj_ind[k]);
                if (ga.size > 1 && ga.find_owner(g_col) != rank) continue;
                counts[static_cast<size_t>(a_row[static_cast<size_t>(g_col)])] += 1;
            }
        }
        for (size_t b = 0; b < pushed.dest_col.size(); ++b) {
            counts[static_cast<size_t>(
                a_row[static_cast<size_t>(pushed.dest_col[b])])] += 1;
        }
        view.ptr.assign(static_cast<size_t>(n_rows) + 1, 0);
        for (int i = 0; i < n_rows; ++i) {
            view.ptr[static_cast<size_t>(i) + 1] =
                view.ptr[static_cast<size_t>(i)] + counts[static_cast<size_t>(i)];
        }
        const size_t total = static_cast<size_t>(view.ptr[static_cast<size_t>(n_rows)]);
        view.src_row.resize(total);
        view.src_dim.resize(total);
        view.norm.resize(total);
        view.data.resize(total);
        view.transposed.resize(total);

        std::vector<int> cursor(view.ptr.begin(), view.ptr.end() - 1);
        for (int i = 0; i < n_rows; ++i) {
            const int g_row = ga.get_global_index(i);
            const int r_dim = ga.block_sizes[i];
            for (int k = ga.adj_ptr[i]; k < ga.adj_ptr[i + 1]; ++k) {
                const int g_col = ga.get_global_index(ga.adj_ind[k]);
                if (ga.size > 1 && ga.find_owner(g_col) != rank) continue;
                const int out = a_row[static_cast<size_t>(g_col)];
                const int e = cursor[static_cast<size_t>(out)]++;
                view.src_row[static_cast<size_t>(e)] = g_row;
                view.src_dim[static_cast<size_t>(e)] = r_dim;
                view.norm[static_cast<size_t>(e)] = a_norms[static_cast<size_t>(k)];
                view.data[static_cast<size_t>(e)] = A.block_data(k);
                view.transposed[static_cast<size_t>(e)] = 0;
            }
        }
        for (size_t b = 0; b < pushed.dest_col.size(); ++b) {
            const int out = a_row[static_cast<size_t>(pushed.dest_col[b])];
            const int e = cursor[static_cast<size_t>(out)]++;
            view.src_row[static_cast<size_t>(e)] = pushed.src_row[b];
            view.src_dim[static_cast<size_t>(e)] = pushed.src_dim[b];
            view.norm[static_cast<size_t>(e)] = pushed.norms[b];
            view.data[static_cast<size_t>(e)] = pushed.values.data() + pushed.offset[b];
            view.transposed[static_cast<size_t>(e)] = 1;
        }
        return view;
    }

    // The numeric pass, shared by the serial and distributed entry points.
    // Computes rows [row_lo, row_hi): the tiled distributed path runs it once
    // per tile against that tile's fetched halo.
    static void fused_rows(const Matrix& A, const Matrix& B, double threshold,
                           const ColumnView& cols,
                           const RemoteRows& remote_a, const RemoteRows& remote_b,
                           const std::vector<int>& a_row, const std::vector<int>& b_row,
                           int n_global,
                           const std::vector<double>& a_norms,
                           const std::vector<double>& b_norms,
                           double a_max_norm, int row_lo, int row_hi,
                           std::vector<std::vector<int>>& row_columns,
                           std::vector<std::vector<T>>& row_values,
                           int& missing_row) {
        const DistGraph& ga = *A.graph;
        const DistGraph& gb = *B.graph;
        const std::vector<int>& a_ptr = ga.adj_ptr;
        const auto& a_ind = ga.adj_ind;
        const std::vector<int>& b_ptr = gb.adj_ptr;
        const auto& b_ind = gb.adj_ind;

        #pragma omp parallel reduction(|:missing_row)
        {
            RowAccumulator inner;
            RowAccumulator outer;
            inner.resize(n_global);
            outer.resize(n_global);
            ProductGroup group;
            std::vector<int> keep_order;
            // A^H[i,k] for a local block, built per entry. Reused across
            // entries; a flush reads it before the next entry overwrites it.
            std::vector<T> lhs_scratch;

            auto visit_a = [&](int g_row, auto&& fn) {
                const int local = a_row[static_cast<size_t>(g_row)];
                if (local >= 0) {
                    for (int k = a_ptr[local]; k < a_ptr[local + 1]; ++k) {
                        fn(ga.get_global_index(a_ind[k]), ga.block_sizes[a_ind[k]],
                           A.block_data(k), a_norms[static_cast<size_t>(k)]);
                    }
                    return true;
                }
                if (remote_a.first.empty()) return false;
                const int at = remote_a.first[static_cast<size_t>(g_row)];
                if (at < 0) return false;
                const int n = remote_a.count[static_cast<size_t>(g_row)];
                for (int e = at; e < at + n; ++e) {
                    fn(remote_a.cols[e], remote_a.dims[e], remote_a.data[e],
                       remote_a.norms[static_cast<size_t>(e)]);
                }
                return true;
            };
            auto visit_b = [&](int g_row, auto&& fn) {
                const int local = b_row[static_cast<size_t>(g_row)];
                if (local >= 0) {
                    for (int k = b_ptr[local]; k < b_ptr[local + 1]; ++k) {
                        fn(gb.get_global_index(b_ind[k]), gb.block_sizes[b_ind[k]],
                           B.block_data(k), b_norms[static_cast<size_t>(k)]);
                    }
                    return;
                }
                if (remote_b.first.empty()) return;
                const int at = remote_b.first[static_cast<size_t>(g_row)];
                if (at < 0) return;
                const int n = remote_b.count[static_cast<size_t>(g_row)];
                for (int e = at; e < at + n; ++e) {
                    fn(remote_b.cols[e], remote_b.dims[e], remote_b.data[e],
                       remote_b.norms[static_cast<size_t>(e)]);
                }
            };
            // The FULL pattern width for remote rows, not the fetched entry
            // count: the stage-one budget telescopes over the row as it
            // exists, and a norm-gated fetch shipping fewer blocks must not
            // loosen the gate on the ones it shipped.
            auto count_b = [&](int g_row) -> int {
                const int local = b_row[static_cast<size_t>(g_row)];
                if (local >= 0) return b_ptr[local + 1] - b_ptr[local];
                if (remote_b.first.empty()) return 0;
                const int at = remote_b.first[static_cast<size_t>(g_row)];
                if (at < 0) return 0;
                return remote_b.pattern_width[static_cast<size_t>(g_row)];
            };

            #pragma omp for schedule(dynamic, kFusedRowChunk)
            for (int i = row_lo; i < row_hi; ++i) {
                const int g_row = ga.get_global_index(i);
                const int r_dim = ga.block_sizes[i];
                inner.clear();
                outer.clear();

                // (A^H B)[i, :] -- scratch, never leaves the thread. The left
                // factors are the blocks of column i of A, conjugate-transposed:
                // the pushed ones arrived that way, the local ones are turned
                // here, an O(dim^2) copy against the O(dim^3) products each one
                // feeds across B's row.
                const int col_count = cols.ptr[static_cast<size_t>(i) + 1] -
                                      cols.ptr[static_cast<size_t>(i)];
                for (int e = cols.ptr[static_cast<size_t>(i)];
                     e < cols.ptr[static_cast<size_t>(i) + 1]; ++e) {
                    const int g_mid = cols.src_row[static_cast<size_t>(e)];
                    const int m_dim = cols.src_dim[static_cast<size_t>(e)];
                    const double ah_norm = cols.norm[static_cast<size_t>(e)];
                    const T* lhs = cols.data[static_cast<size_t>(e)];
                    if (!cols.transposed[static_cast<size_t>(e)]) {
                        lhs_scratch.resize(static_cast<size_t>(r_dim) * m_dim);
                        write_transposed_conjugate_block(lhs_scratch.data(), lhs,
                                                         m_dim, r_dim);
                        lhs = lhs_scratch.data();
                    }

                    // Stage-one budget, rarh's bound with the roles turned
                    // around: a skipped pair (k,l) contributes A^H[i,k] B[k,l]
                    // to (A^H B)[i,l], at most eps1, and that error reaches
                    // C[i,j] multiplied by A[l,j] -- so at most eps1 *
                    // a_max_norm, with the (k,l) fan-out being col_count rows
                    // of B each up to |B row k| wide. Scaling per k by
                    // |B row k| keeps the sum telescoping:
                    // sum_k |B_k| * eps1_k * a_max_norm = threshold / 2.
                    const int b_row_count = count_b(g_mid);
                    const double eps1 =
                        (threshold > 0.0 && a_max_norm > 0.0 && b_row_count > 0)
                            ? threshold / (2.0 * std::max(1, col_count) *
                                           static_cast<double>(b_row_count) * a_max_norm)
                            : 0.0;
                    group.clear();
                    visit_b(g_mid, [&](int g_inner, int c_dim, const T* b_block,
                                       double b_norm) {
                        const double pair = ah_norm * b_norm;
                        if (pair < eps1) return;
                        const int slot = inner.obtain(g_inner, r_dim, c_dim);
                        inner.norm_bound[static_cast<size_t>(slot)] += pair;
                        group.add(b_block, inner.value_offset[static_cast<size_t>(slot)],
                                  c_dim, r_dim, m_dim);
                    });
                    group.flush(inner.values, lhs, r_dim, m_dim);
                }

                // C[i, j>=i] = sum_l (A^H B)[i,l] A[l,j], reading ROWS of A in
                // their natural layout -- no hermiticity of A involved, which
                // is the whole reason this kernel exists next to rarh.
                const double eps2 =
                    (threshold > 0.0 && !inner.touched.empty())
                        ? threshold / (2.0 * static_cast<double>(inner.touched.size()))
                        : 0.0;
                for (size_t sIdx = 0; sIdx < inner.touched.size(); ++sIdx) {
                    const int g_inner = inner.touched[sIdx];
                    const int m_dim = inner.col_dim[sIdx];
                    const T* inner_block = inner.values.data() + inner.value_offset[sIdx];
                    const double inner_norm = inner.norm_bound[sIdx];
                    group.clear();
                    // Unlike rarh, a row that cannot be visited here is not an
                    // operand-contract failure -- it is a fetch-set bug in THIS
                    // kernel: every l with inner[l] != 0 was marked reachable
                    // when the fetch set was built. Flagged and thrown as such.
                    const bool found = visit_a(g_inner, [&](int g_col, int c_dim,
                                                            const T* a_block,
                                                            double a_col_norm) {
                        if (g_col < g_row) return;  // upper triangle only
                        if (inner_norm * a_col_norm < eps2) return;
                        const int slot = outer.obtain(g_col, r_dim, c_dim);
                        group.add(a_block, outer.value_offset[static_cast<size_t>(slot)],
                                  c_dim, r_dim, m_dim);
                    });
                    // inner_block is read INSIDE the flush; the group writes
                    // into `outer`, never `inner`, so inner's arena cannot
                    // have moved under it.
                    group.flush(outer.values, inner_block, r_dim, m_dim);
                    if (!found) missing_row = 1;
                }

                // The one drop, on finished values; staging order, exact
                // sizing and the ORDER INVARIANT the positional copy rests on
                // are recorded at stage_row_in_column_order in common.hpp.
                stage_row_in_column_order(outer, r_dim, threshold, keep_order,
                                          row_columns[i], row_values[i]);
            }
        }
    }

    static Matrix run_local(const Matrix& A, const Matrix& B, double threshold) {
        const DistGraph& ga = *A.graph;
        const int n_rows = static_cast<int>(ga.adj_ptr.size()) - 1;
        const int n_global = ga.block_displs.empty() ? n_rows : ga.block_displs.back();
        std::vector<std::vector<int>> row_columns(n_rows);
        std::vector<std::vector<T>> row_values(n_rows);
        const std::vector<int> a_row = fused_row_of_global(A, n_global);
        const PushedBlocks no_pushed;
        const ColumnView cols = build_column_view(A, a_row, no_pushed);
        const RemoteRows none;
        int missing_row = 0;
        fused_rows(A, B, threshold, cols, none, none,
                   a_row, fused_row_of_global(B, n_global),
                   n_global, A.get_block_norms(), B.get_block_norms(),
                   fused_global_max_norm(A), 0, n_rows, row_columns, row_values,
                   missing_row);
        if (missing_row) {
            throw std::runtime_error(
                "rhar_upper: internal error, a row in the product support was not visited");
        }
        GhostSizes no_ghosts;
        return fused_assemble(A, row_columns, row_values, no_ghosts, "rhar");
    }

    // Three exchanges, each surface-sized, each unavoidable at its point:
    // the boundary of A^H has to be pushed before the support of A^H B is
    // knowable, B's rows over that boundary before the support is complete,
    // and A's rows over the support before the second contraction can run.
    // All are collective with a fixed sequence, so a rank that owns nothing
    // still participates in every one.
    static Matrix run_distributed(const Matrix& A, const Matrix& B, double threshold) {
        const DistGraph& ga = *A.graph;
        const DistGraph& gb = *B.graph;
        const int rank = ga.rank;
        const int n_rows = static_cast<int>(ga.adj_ptr.size()) - 1;
        const int n_global = ga.block_displs.back();
        const std::vector<int> a_row = fused_row_of_global(A, n_global);
        const std::vector<int> b_row = fused_row_of_global(B, n_global);

        // ---- The boundary of A^H, pushed by its row owners.
        const PushedBlocks pushed = push_boundary_columns(A);
        const ColumnView cols = build_column_view(A, a_row, pushed);

        // The fetch gates need the same global bound the numeric gate uses;
        // taken here (one Allreduce) instead of at the fused_rows call.
        const double a_max_norm = fused_global_max_norm(A);
        const std::vector<double>& a_norms = A.get_block_norms();
        const std::vector<double>& b_norms = B.get_block_norms();

        // ---- Round 1: the rows of B that the column entries name. Local
        // column entries contract against B rows this rank owns (the
        // partitions agree; see ensure_product_operands_compatible), so only
        // the pushed rows can be remote -- but the GATE is built over every
        // entry, because the reach pass below applies it to local B rows too.
        //
        // A block of B row k is worth shipping only if SOME column entry
        // (i,k) could pass the numeric stage-one gate with it:
        //
        //     ah_norm(i,k) * b_norm(k,l) >= threshold /
        //         (2 * col_count(i) * width_b(k) * a_max_norm)
        //
        // taken over all i, a per-row norm floor with colmax_ah(k) =
        // max_i [col_count(i) * ah_norm(i,k)].
        std::set<int> need_b;
        std::map<int, double> colmax_ah;
        std::vector<double> col_sum(static_cast<size_t>(n_rows), 0.0);
        double s_max = 0.0;      // max_i sum_e ah_norm(i,e)
        for (int i = 0; i < n_rows; ++i) {
            const int cc = static_cast<int>(cols.ptr[static_cast<size_t>(i) + 1] -
                                            cols.ptr[static_cast<size_t>(i)]);
            double row_sum = 0.0;
            for (int e = static_cast<int>(cols.ptr[static_cast<size_t>(i)]);
                 e < static_cast<int>(cols.ptr[static_cast<size_t>(i) + 1]); ++e) {
                const int g_mid = cols.src_row[static_cast<size_t>(e)];
                const double reach =
                    static_cast<double>(cc) * cols.norm[static_cast<size_t>(e)];
                row_sum += cols.norm[static_cast<size_t>(e)];
                auto [it, fresh] = colmax_ah.emplace(g_mid, reach);
                if (!fresh && reach > it->second) it->second = reach;
                if (gb.find_owner(g_mid) != rank) need_b.insert(g_mid);
            }
            col_sum[static_cast<size_t>(i)] = row_sum;
            s_max = std::max(s_max, row_sum);
        }
        GhostMetadata meta_b = fetch_row_patterns(B, need_b, ga.comm, ga.size, rank);

        const auto gate_b = [&](int g_row, int width) -> double {
            if (!(threshold > 0.0) || a_max_norm <= 0.0 || width <= 0) return 0.0;
            auto it = colmax_ah.find(g_row);
            if (it == colmax_ah.end() || !(it->second > 0.0)) {
                return std::numeric_limits<double>::infinity();
            }
            return threshold / (2.0 * static_cast<double>(width) * a_max_norm * it->second);
        };

        // ---- The support of A^H B, from patterns alone, gated exactly as
        // the numeric stage-one gate would: per column, the largest B norm a
        // kept pair could carry. `reached` stays a separate flag because at
        // threshold 0 a zero-norm block is numerically kept and its row must
        // remain answerable (the missing_row check); at threshold > 0 a kept
        // block always has norm >= gate > 0, so reached implies colmax > 0.
        //
        // Dense numberings, not global ones: the mid rows are the distinct
        // entries of cols.src_row and the reachable columns come straight from
        // B's local rows and the fetched patterns, so both tables cost O(halo)
        // instead of a fixed price per row of the whole system (see rarh).
        // Every slot is resolved once per ENTRY -- a search inside the walks
        // below would sit in a cross product.
        FusedDenseIds mid_ids, cand;
        {
            std::vector<int> scratch(cols.src_row.begin(), cols.src_row.end());
            mid_ids.build(scratch);
            scratch.clear();
            for (const auto& row : meta_b) {
                for (const auto& meta : row.second) scratch.push_back(meta.col);
            }
            const int b_rows = static_cast<int>(gb.adj_ptr.size()) - 1;
            for (int lb = 0; lb < b_rows; ++lb) {
                for (int e = gb.adj_ptr[lb]; e < gb.adj_ptr[lb + 1]; ++e) {
                    scratch.push_back(gb.get_global_index(gb.adj_ind[e]));
                }
            }
            cand.build(scratch);
        }
        std::vector<int> src_slot(cols.src_row.size(), -1);
        for (size_t e = 0; e < cols.src_row.size(); ++e) {
            src_slot[e] = mid_ids.of(cols.src_row[e]);
        }
        const size_t b_local_entries = static_cast<size_t>(gb.adj_ptr.back());
        std::unordered_map<int, size_t> b_entry_base;
        size_t b_total_entries = b_local_entries;
        for (const auto& row : meta_b) {
            b_entry_base[row.first] = b_total_entries;
            b_total_entries += row.second.size();
        }
        std::vector<int> entry_cand(b_total_entries, -1);
        for (size_t e = 0; e < b_local_entries; ++e) {
            entry_cand[e] = cand.of(gb.get_global_index(gb.adj_ind[e]));
        }
        for (const auto& row : meta_b) {
            size_t at = b_entry_base[row.first];
            for (const auto& meta : row.second) entry_cand[at++] = cand.of(meta.col);
        }

        std::vector<char> reached(cand.size(), 0);
        std::vector<double> colmax_b(cand.size(), 0.0);
        std::vector<char> row_seen(mid_ids.size(), 0);
        double w_bound = 0.0;  // max_i sum_e width_b(src_row(e)): bound on |touched_i|
        {
            std::vector<double> width_of_row(mid_ids.size(), 0.0);
            for (size_t e = 0; e < cols.src_row.size(); ++e) {
                const int g_mid = cols.src_row[e];
                const int mid = src_slot[e];
                if (mid < 0 || row_seen[static_cast<size_t>(mid)]) continue;
                row_seen[static_cast<size_t>(mid)] = 1;
                const int lb = b_row[static_cast<size_t>(g_mid)];
                if (lb >= 0) {
                    const int width = gb.adj_ptr[lb + 1] - gb.adj_ptr[lb];
                    width_of_row[static_cast<size_t>(mid)] = width;
                    const double gate = gate_b(g_mid, width);
                    for (int k = gb.adj_ptr[lb]; k < gb.adj_ptr[lb + 1]; ++k) {
                        const double norm = b_norms[static_cast<size_t>(k)];
                        if (norm < gate) continue;
                        const int slot = entry_cand[static_cast<size_t>(k)];
                        if (slot < 0) continue;
                        reached[static_cast<size_t>(slot)] = 1;
                        colmax_b[static_cast<size_t>(slot)] =
                            std::max(colmax_b[static_cast<size_t>(slot)], norm);
                    }
                    continue;
                }
                auto pattern = meta_b.find(g_mid);
                if (pattern == meta_b.end()) continue;
                const int width = static_cast<int>(pattern->second.size());
                width_of_row[static_cast<size_t>(mid)] = width;
                const double gate = gate_b(g_mid, width);
                size_t at = b_entry_base.at(g_mid);
                for (const auto& meta : pattern->second) {
                    const int slot = entry_cand[at++];
                    if (meta.norm < gate) continue;
                    if (slot < 0) continue;
                    reached[static_cast<size_t>(slot)] = 1;
                    colmax_b[static_cast<size_t>(slot)] =
                        std::max(colmax_b[static_cast<size_t>(slot)], meta.norm);
                }
            }
            for (int i = 0; i < n_rows; ++i) {
                double w = 0.0;
                for (int e = static_cast<int>(cols.ptr[static_cast<size_t>(i)]);
                     e < static_cast<int>(cols.ptr[static_cast<size_t>(i) + 1]); ++e) {
                    const int mid = src_slot[static_cast<size_t>(e)];
                    if (mid >= 0) w += width_of_row[static_cast<size_t>(mid)];
                }
                w_bound = std::max(w_bound, w);
            }
        }

        // ---- Round 2: the rows of A over that support, with the stage-two
        // gate bounded from the safe side: inner_norm(i,l) <= col_sum(i) *
        // colmax_b(l) <= s_max * colmax_b(l) and |touched_i| <= w_bound.
        std::set<int> need_a;
        for (size_t slot = 0; slot < cand.size(); ++slot) {
            if (!reached[slot]) continue;
            const int g = cand.ids[slot];
            if (ga.find_owner(g) != rank) need_a.insert(g);
        }
        GhostMetadata meta_a = fetch_row_patterns(A, need_a, ga.comm, ga.size, rank);

        const auto gate_a = [&](int g_row) -> double {
            if (!(threshold > 0.0)) return 0.0;
            const int slot = cand.of(g_row);
            const double cm = slot < 0 ? 0.0 : colmax_b[static_cast<size_t>(slot)];
            if (!(cm > 0.0) || !(s_max > 0.0) || !(w_bound > 0.0)) {
                return std::numeric_limits<double>::infinity();
            }
            return threshold / (2.0 * w_bound * s_max * cm);
        };

        // Round 2's entries are output columns of an upper-only product, so
        // columns below the first owned row are dead on arrival. Round 1's
        // are contraction indices; no trim.
        int min_owned_row = std::numeric_limits<int>::max();
        for (int i = 0; i < n_rows; ++i) {
            min_owned_row = std::min(min_owned_row, ga.get_global_index(i));
        }
        if (n_rows == 0) min_owned_row = 0;

        // ---- Payloads, gated and TILED (see rarh for the shape; the only
        // difference is that a tile's needs walk the COLUMN VIEW, since the
        // left factors here are columns of A). Kept payload counts per row:
        const auto kept_b = [&](int g) -> size_t {
            auto it = meta_b.find(g);
            if (it == meta_b.end()) return 0;
            const double gate = gate_b(g, static_cast<int>(it->second.size()));
            size_t n = 0;
            for (const auto& m : it->second) {
                if (m.norm >= gate) ++n;
            }
            return n;
        };
        const auto kept_a = [&](int g) -> size_t {
            auto it = meta_a.find(g);
            if (it == meta_a.end()) return 0;
            const double gate = gate_a(g);
            size_t n = 0;
            for (const auto& m : it->second) {
                if (m.col >= min_owned_row && m.norm >= gate) ++n;
            }
            return n;
        };
        // (col, flat entry index), so the birth/death cross product reads a
        // precomputed slot instead of searching per visit.
        const auto for_each_kept_b_entry = [&](int g_mid, auto&& fn) {
            const int lb = b_row[static_cast<size_t>(g_mid)];
            if (lb >= 0) {
                const double gate = gate_b(g_mid, gb.adj_ptr[lb + 1] - gb.adj_ptr[lb]);
                for (int e = gb.adj_ptr[lb]; e < gb.adj_ptr[lb + 1]; ++e) {
                    if (b_norms[static_cast<size_t>(e)] < gate) continue;
                    fn(gb.get_global_index(gb.adj_ind[e]), static_cast<size_t>(e));
                }
                return;
            }
            auto pattern = meta_b.find(g_mid);
            if (pattern == meta_b.end()) return;
            const double gate = gate_b(g_mid, static_cast<int>(pattern->second.size()));
            size_t at = b_entry_base.at(g_mid);
            for (const auto& m : pattern->second) {
                const size_t here = at++;
                if (m.norm >= gate) fn(m.col, here);
            }
        };

        const auto for_each_kept_b_col = [&](int g_mid, auto&& fn) {
            const int lb = b_row[static_cast<size_t>(g_mid)];
            if (lb >= 0) {
                const double gate = gate_b(g_mid, gb.adj_ptr[lb + 1] - gb.adj_ptr[lb]);
                for (int e = gb.adj_ptr[lb]; e < gb.adj_ptr[lb + 1]; ++e) {
                    if (b_norms[static_cast<size_t>(e)] < gate) continue;
                    fn(gb.get_global_index(gb.adj_ind[e]));
                }
                return;
            }
            auto pattern = meta_b.find(g_mid);
            if (pattern == meta_b.end()) return;
            const double gate = gate_b(g_mid, static_cast<int>(pattern->second.size()));
            for (const auto& m : pattern->second) {
                if (m.norm >= gate) fn(m.col);
            }
        };

        // ---- Use spans and the round plan, as in rarh: one walk of the reach
        // gives every remote row its first and last needing output row, a row
        // is fetched on the round its first use opens and released after the
        // round its last use closes, and a round is charged only for what is
        // BORN on it. Traffic is then the union reach exactly, and residency
        // the live set. (rhar walks A^H's columns rather than A's rows -- the
        // reach is the transposed one -- but the schedule is identical.)
        FusedDenseIds b_ids, a_ids;
        {
            std::vector<int> scratch;
            for (const auto& row : meta_b) scratch.push_back(row.first);
            b_ids.build(scratch);
            scratch.clear();
            for (const auto& row : meta_a) scratch.push_back(row.first);
            a_ids.build(scratch);
        }
        std::vector<int> entry_aslot(b_total_entries, -1);
        for (size_t e = 0; e < b_total_entries; ++e) {
            const int slot = entry_cand[e];
            if (slot >= 0) entry_aslot[e] = a_ids.of(cand.ids[static_cast<size_t>(slot)]);
        }
        std::vector<int> b_birth(b_ids.size(), -1);
        std::vector<int> b_death(b_ids.size(), -1);
        std::vector<int> a_birth(a_ids.size(), -1);
        std::vector<int> a_death(a_ids.size(), -1);
        for (int i = 0; i < n_rows; ++i) {
            for (int e = static_cast<int>(cols.ptr[static_cast<size_t>(i)]);
                 e < static_cast<int>(cols.ptr[static_cast<size_t>(i) + 1]); ++e) {
                const int g_mid = cols.src_row[static_cast<size_t>(e)];
                if (gb.find_owner(g_mid) != rank) {
                    const int s_b = b_ids.of(g_mid);
                    if (s_b >= 0) {
                        if (b_birth[static_cast<size_t>(s_b)] < 0) {
                            b_birth[static_cast<size_t>(s_b)] = i;
                        }
                        b_death[static_cast<size_t>(s_b)] = i;
                    }
                }
                for_each_kept_b_entry(g_mid, [&](int l, size_t entry) {
                    if (ga.find_owner(l) == rank) return;
                    const int s_a = entry_aslot[entry];
                    if (s_a < 0) return;
                    if (a_birth[static_cast<size_t>(s_a)] < 0) {
                        a_birth[static_cast<size_t>(s_a)] = i;
                    }
                    a_death[static_cast<size_t>(s_a)] = i;
                });
            }
        }

        int bs_max = 1;
        for (int s : ga.block_sizes) bs_max = std::max(bs_max, s);
        const size_t budget = fused_tile_budget_bytes(ga.comm);
        const size_t block_budget = fused_block_budget(budget, bs_max, sizeof(T));
        std::vector<std::pair<int, int>> b_born, a_born;  // (birth row, global row)
        std::vector<size_t> arrivals(static_cast<size_t>(n_rows) + 1, 0);
        std::vector<int> max_death_at(static_cast<size_t>(n_rows) + 1, -1);
        for (size_t slot = 0; slot < b_ids.size(); ++slot) {
            const int born_b = b_birth[slot];
            if (born_b < 0) continue;
            const int g = b_ids.ids[slot];
            b_born.emplace_back(born_b, g);
            const size_t blocks = kept_b(g);
            arrivals[static_cast<size_t>(born_b)] += blocks;
            max_death_at[static_cast<size_t>(born_b)] =
                std::max(max_death_at[static_cast<size_t>(born_b)], b_death[slot]);
        }
        for (size_t slot = 0; slot < a_ids.size(); ++slot) {
            const int born_a = a_birth[slot];
            if (born_a < 0) continue;
            const int g = a_ids.ids[slot];
            a_born.emplace_back(born_a, g);
            const size_t blocks = kept_a(g);
            arrivals[static_cast<size_t>(born_a)] += blocks;
            max_death_at[static_cast<size_t>(born_a)] =
                std::max(max_death_at[static_cast<size_t>(born_a)], a_death[slot]);
        }
        std::sort(b_born.begin(), b_born.end());
        std::sort(a_born.begin(), a_born.end());
        const FusedFetchPlan fetch_plan = fused_fetch_plan(arrivals, max_death_at, n_rows, block_budget);
        const bool refetch_halo = fetch_plan.refetch;

        // Refetch regime, as in rarh: tiles whose own halo fits the budget,
        // dropped and re-fetched, for when the live set itself does not fit.
        std::vector<char> bflag(b_ids.size(), 0), aflag(a_ids.size(), 0);
        std::vector<char> midflag(
            static_cast<size_t>(std::max(0, static_cast<int>(gb.adj_ptr.size()) - 1)), 0);
        std::vector<int> btouched, atouched, midtouched;
        const auto reset_flags = [&] {
            for (int t : btouched) bflag[static_cast<size_t>(t)] = 0;
            for (int t : atouched) aflag[static_cast<size_t>(t)] = 0;
            for (int t : midtouched) midflag[static_cast<size_t>(t)] = 0;
            btouched.clear();
            atouched.clear();
            midtouched.clear();
        };
        const auto walk_row = [&](int i, auto&& on_b_row, auto&& on_a_row) {
            for (int e = static_cast<int>(cols.ptr[static_cast<size_t>(i)]);
                 e < static_cast<int>(cols.ptr[static_cast<size_t>(i) + 1]); ++e) {
                const int g_mid = cols.src_row[static_cast<size_t>(e)];
                const int s_b = b_ids.of(g_mid);
                if (s_b >= 0) {
                    if (bflag[static_cast<size_t>(s_b)]) continue;
                    bflag[static_cast<size_t>(s_b)] = 1;
                    btouched.push_back(s_b);
                    on_b_row(g_mid);
                } else {
                    const int lb = b_row[static_cast<size_t>(g_mid)];
                    if (lb < 0 || midflag[static_cast<size_t>(lb)]) continue;
                    midflag[static_cast<size_t>(lb)] = 1;
                    midtouched.push_back(lb);
                }
                for_each_kept_b_entry(g_mid, [&](int l, size_t entry) {
                    const int s_a = entry_aslot[entry];
                    if (s_a < 0 || aflag[static_cast<size_t>(s_a)]) return;
                    aflag[static_cast<size_t>(s_a)] = 1;
                    atouched.push_back(s_a);
                    on_a_row(l);
                });
            }
        };

        std::vector<int> tile_bound;
        if (refetch_halo) {
            tile_bound.push_back(0);
            const auto charge = [&](int i) -> size_t {
                size_t c = 0;
                walk_row(i, [&](int g) { c += kept_b(g); }, [&](int l) { c += kept_a(l); });
                return c;
            };
            size_t acc = 0;
            for (int i = 0; i < n_rows; ++i) {
                size_t row_cost = charge(i);
                if (acc > 0 && acc + row_cost > block_budget &&
                    i - tile_bound.back() >= kFusedRowChunk) {
                    tile_bound.push_back(i);
                    reset_flags();
                    acc = 0;
                    row_cost = charge(i);
                }
                acc += row_cost;
            }
            reset_flags();
        } else {
            tile_bound = fetch_plan.bound;
        }
        tile_bound.push_back(n_rows);

        long long my_tiles = static_cast<long long>(tile_bound.size()) - 1;
        long long rounds = my_tiles;
        MPI_Allreduce(&my_tiles, &rounds, 1, MPI_LONG_LONG, MPI_MAX, ga.comm);
        while (static_cast<long long>(tile_bound.size()) - 1 < rounds) {
            tile_bound.push_back(n_rows);
        }

        const auto round_of_row = [&](int i) -> long long {
            const auto it = std::upper_bound(tile_bound.begin(), tile_bound.end(), i);
            return static_cast<long long>(it - tile_bound.begin()) - 1;
        };

        std::vector<std::vector<int>> row_columns(n_rows);
        std::vector<std::vector<T>> row_values(n_rows);
        GhostSizes ghost_sizes;
        int missing_row = 0;
        FusedHaloStream<T> halo_b, halo_a;
        halo_b.init(meta_b, n_global);
        halo_a.init(meta_a, n_global);
        size_t b_cursor = 0, a_cursor = 0;
        double secs_fetch = 0.0, secs_numeric = 0.0;
        for (long long r = 0; r < rounds; ++r) {
            const int lo = tile_bound[static_cast<size_t>(r)];
            const int hi = tile_bound[static_cast<size_t>(r) + 1];
            std::vector<BlockID> want_b, want_a;
            long long death_b = r, death_a = r;
            if (refetch_halo) {
                for (int i = lo; i < hi; ++i) {
                    walk_row(
                        i,
                        [&](int g) {
                            auto it = meta_b.find(g);
                            if (it == meta_b.end()) return;
                            const double gate =
                                gate_b(g, static_cast<int>(it->second.size()));
                            for (const auto& m : it->second) {
                                if (m.norm >= gate) want_b.push_back(BlockID{g, m.col});
                            }
                        },
                        [&](int l) {
                            auto it = meta_a.find(l);
                            if (it == meta_a.end()) return;
                            const double gate = gate_a(l);
                            for (const auto& m : it->second) {
                                if (m.col >= min_owned_row && m.norm >= gate) {
                                    want_a.push_back(BlockID{l, m.col});
                                }
                            }
                        });
                }
                reset_flags();
            }
            for (; !refetch_halo &&
                   b_cursor < b_born.size() && b_born[b_cursor].first < hi; ++b_cursor) {
                const int g = b_born[b_cursor].second;
                auto it = meta_b.find(g);
                if (it != meta_b.end()) {
                    const double gate = gate_b(g, static_cast<int>(it->second.size()));
                    for (const auto& m : it->second) {
                        if (m.norm >= gate) want_b.push_back(BlockID{g, m.col});
                    }
                }
                death_b = std::max(death_b,
                                   round_of_row(b_death[static_cast<size_t>(b_ids.of(g))]));
            }
            for (; !refetch_halo &&
                   a_cursor < a_born.size() && a_born[a_cursor].first < hi; ++a_cursor) {
                const int g = a_born[a_cursor].second;
                auto it = meta_a.find(g);
                if (it != meta_a.end()) {
                    const double gate = gate_a(g);
                    for (const auto& m : it->second) {
                        if (m.col >= min_owned_row && m.norm >= gate) {
                            want_a.push_back(BlockID{g, m.col});
                        }
                    }
                }
                death_a = std::max(death_a,
                                   round_of_row(a_death[static_cast<size_t>(a_ids.of(g))]));
            }

            const double t_fetch0 = MPI_Wtime();
            auto ghosts_b = build_spmm_ghost_blocks<T>(
                meta_b, fetch_required_block_payloads(B, want_b));
            auto ghosts_a = build_spmm_ghost_blocks<T>(
                meta_a, fetch_required_block_payloads(A, want_a));
            for (const auto& [gid, dim] : ghosts_a.sizes) ghost_sizes[gid] = dim;

            halo_b.absorb(std::move(ghosts_b), death_b);
            halo_a.absorb(std::move(ghosts_a), death_a);
            const double t_num0 = MPI_Wtime();
            secs_fetch += t_num0 - t_fetch0;
            fused_rows(A, B, threshold, cols, halo_a.rows, halo_b.rows,
                       a_row, b_row, n_global, a_norms, b_norms,
                       a_max_norm, lo, hi, row_columns, row_values, missing_row);
            secs_numeric += MPI_Wtime() - t_num0;
            halo_b.retire(r);
            halo_a.retire(r);
        }
        fused_report_halo("rhar", ga.comm, rank, rounds,
                          halo_b.fetched_blocks + halo_a.fetched_blocks,
                          halo_b.peak_live_blocks + halo_a.peak_live_blocks,
                          A.get_block_norms().size(),
                          static_cast<size_t>(bs_max) * static_cast<size_t>(bs_max) * sizeof(T),
                          secs_fetch, secs_numeric, budget, refetch_halo);
        int any_missing = missing_row;
        MPI_Allreduce(&missing_row, &any_missing, 1, MPI_INT, MPI_MAX, ga.comm);
        if (any_missing) {
            throw std::runtime_error(
                "rhar_upper: internal error, a row in the product support was not fetched");
        }

        // Result columns come from A's rows: those of the fetched rows were
        // collected per tile above, those of A's own rows are its graph's
        // ghosts.
        const int n_owned = static_cast<int>(ga.owned_global_indices.size());
        for (size_t g = 0; g < ga.ghost_global_indices.size(); ++g) {
            ghost_sizes[ga.ghost_global_indices[g]] =
                ga.block_sizes[static_cast<size_t>(n_owned) + g];
        }
        return fused_assemble(A, row_columns, row_values, ghost_sizes, "rhar");
    }

    static Matrix run(const Matrix& A, const Matrix& B, double threshold) {
        return A.graph->size == 1 ? run_local(A, B, threshold)
                                  : run_distributed(A, B, threshold);
    }
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_OPS_SPMM_RHAR_HPP

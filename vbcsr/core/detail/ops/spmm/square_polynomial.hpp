#ifndef VBCSR_DETAIL_OPS_SPMM_SQUARE_POLYNOMIAL_HPP
#define VBCSR_DETAIL_OPS_SPMM_SQUARE_POLYNOMIAL_HPP

#include "../../distributed/block_payload_exchange.hpp"
#include "../../distributed/result_graph.hpp"
#include "common.hpp"
#include "../../../scalar_traits.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace vbcsr::detail {

// C = c2 A^2 + c1 A + c0 I, upper triangle only, WITHOUT ever forming A^2.
//
// Why this is a kernel and not a product followed by an axpby:
//
// A^2 IS DENSER THAN THE ANSWER whenever the polynomial contracts. Squaring
// doubles the sparsity radius, so A^2 reaches 2 r_A; but the answer of a
// converging Newton-Schulz polynomial (P = 1.875 I - 1.25 X + 0.375 X^2, driving
// X toward the identity) collapses under the drop threshold while the square
// does not. Measured on a 4096-atom step: X carries 2080 block neighbours per
// row, the P that comes out carries 1628, and the X^2 in between reaches the
// full row -- 39 GB materialised, and then filtered away, to produce 17 GB.
// This kernel holds one row of the square in per-thread scratch instead.
//
// Accuracy runs the same way and is comparable, but NOT uniformly better. The
// two-call route drops the threshold on A^2 and then again on the finished
// polynomial, so its first truncation propagates through the second; here the
// coefficients are applied to the accumulated row and the threshold is applied
// once, to the final block. That removes one truncation, which is why the two
// agree to round-off at threshold 0 and stay the same order apart under one.
//
// It does not make this route pointwise more accurate, and it was documented
// that way in error. The routes drop DIFFERENT blocks: two-call decides on
// |A^2| alone, this one on the finished |c2 A^2 + c1 A + c0 I|, so a block the
// polynomial cancels is kept there and dropped here, and vice versa. On a
// scattered pattern (the `scatter` case in test_square_polynomial) this route
// comes out ~16% worse at matched nnz, at every gate width tried. Expect the
// same order, not a guaranteed improvement.
//
// Contract, the caller's to honour and unchecked:
//   * A is Hermitian, so A^2 and the whole polynomial are Hermitian and only
//     the upper triangle is produced. Complete it with complete_hermitian().
//
// Result: block columns at or above the diagonal, every block's Frobenius norm
// >= threshold.
template <typename Matrix>
struct SquarePolynomialExecutor {
    using T = typename Matrix::value_type;
    using RowAccumulator = FusedRowAccumulator<T>;
    using ProductGroup = FusedProductGroup<T>;

    // Rows fetched from other ranks, indexed by global row for O(1) reach in the
    // numeric loop. `first` is -1 only for a row nobody answered; a row that
    // exists and holds no blocks answers with zero entries, which must read as
    // found-with-nothing.
    struct RemoteRows {
        std::vector<int> first;
        std::vector<int> count;
        std::vector<int> cols;
        std::vector<int> dims;
        std::vector<double> norms;
        std::vector<const T*> data;

        void build(const SpMMGhostBlocks<T>& ghosts, const GhostMetadata& patterns,
                   int n_global) {
            first.assign(static_cast<size_t>(n_global), -1);
            count.assign(static_cast<size_t>(n_global), 0);
            for (const auto& row : patterns) first[static_cast<size_t>(row.first)] = 0;
            for (const auto& entry : ghosts.rows) {
                first[static_cast<size_t>(entry.first)] = static_cast<int>(cols.size());
                count[static_cast<size_t>(entry.first)] =
                    static_cast<int>(entry.second.size());
                for (const auto& block : entry.second) {
                    cols.push_back(block.col);
                    dims.push_back(block.c_dim);
                    norms.push_back(block.norm);
                    data.push_back(block.data);
                }
            }
        }
    };

    static std::vector<BlockID> blocks_of(const GhostMetadata& patterns) {
        std::vector<BlockID> blocks;
        for (const auto& row : patterns) {
            for (const auto& meta : row.second) {
                blocks.push_back(BlockID{row.first, meta.col});
            }
        }
        return blocks;
    }

    static std::vector<int> row_of_global(const Matrix& m, int n_global_blocks) {
        std::vector<int> map(static_cast<size_t>(n_global_blocks), -1);
        const int n_rows = static_cast<int>(m.graph->adj_ptr.size()) - 1;
        for (int i = 0; i < n_rows; ++i) {
            map[static_cast<size_t>(m.graph->get_global_index(i))] = i;
        }
        return map;
    }

    static void fused_rows(const Matrix& A, T c2, T c1, T c0, double threshold,
                           const RemoteRows& remote, const std::vector<int>& a_row,
                           int n_global, const std::vector<double>& a_norms,
                           bool a_any_nonzero,
                           std::vector<std::vector<int>>& row_columns,
                           std::vector<std::vector<T>>& row_values) {
        const DistGraph& ga = *A.graph;
        const int n_rows = static_cast<int>(ga.adj_ptr.size()) - 1;
        const std::vector<int>& a_ptr = ga.adj_ptr;
        const auto& a_ind = ga.adj_ind;

        #pragma omp parallel
        {
            RowAccumulator acc;
            ProductGroup group;
            acc.resize(n_global);
            // Surviving accumulator slots, in column order. Per thread and
            // reused across rows so the ordering pass allocates once.
            std::vector<int> keep_order;

            auto visit = [&](int g_row, auto&& fn) {
                const int local = a_row[static_cast<size_t>(g_row)];
                if (local >= 0) {
                    for (int k = a_ptr[local]; k < a_ptr[local + 1]; ++k) {
                        fn(ga.get_global_index(a_ind[k]), ga.block_sizes[a_ind[k]],
                           A.block_data(k), a_norms[static_cast<size_t>(k)]);
                    }
                    return;
                }
                if (remote.first.empty()) return;
                const int at = remote.first[static_cast<size_t>(g_row)];
                if (at < 0) return;
                const int n = remote.count[static_cast<size_t>(g_row)];
                for (int e = at; e < at + n; ++e) {
                    fn(remote.cols[e], remote.dims[e], remote.data[e],
                       remote.norms[static_cast<size_t>(e)]);
                }
            };

            #pragma omp for schedule(dynamic, 8)
            for (int i = 0; i < n_rows; ++i) {
                const int g_row = ga.get_global_index(i);
                const int r_dim = ga.block_sizes[i];
                acc.clear();

                // Per-row budget for the squared term, set to leave this route
                // exactly as accurate as squaring and then combining.
                //
                // A skipped pair contributes c2*A[i,k]*A[k,j] to the finished
                // block, at most |c2|*eps, and there are at most n_A of them:
                // the block is off by at most n_A*|c2|*eps. The two-call route
                // gates the square at threshold/n_A, so its A^2 is off by at
                // most threshold and the c2*A^2 it goes on to form by at most
                // |c2|*threshold. Equating the two leaves
                //
                //     n_A*|c2|*eps = |c2|*threshold   ->   eps = threshold/n_A
                //
                // with c2 dropping out on both sides. Scaling eps by 1/|c2| (or
                // by any fixed safety factor) only buys accuracy the other route
                // never had, and it is bought where it is most expensive: as X
                // approaches the identity the pair norms pile up just under the
                // gate, so a gate half this wide stops skipping almost anything
                // and the near-converged steps -- the cheap ones -- stay dense.
                // Widening the gate to parity costs nothing measurable: the
                // `scatter` case reports the same error at this width and at
                // half of it, and the 4096-atom iteration's final residual is
                // unchanged, while the near-converged step it speeds up ran 31%
                // faster (16.78s -> 11.64s on sc888 step 2).
                const int a_row_count = a_ptr[i + 1] - a_ptr[i];
                const double eps = (threshold > 0.0 && a_any_nonzero)
                                       ? threshold / std::max(1, a_row_count)
                                       : 0.0;

                // (A^2)[i, :] -- scratch, never leaves the thread.
                for (int ka = a_ptr[i]; ka < a_ptr[i + 1]; ++ka) {
                    const int g_mid = ga.get_global_index(a_ind[ka]);
                    const int m_dim = ga.block_sizes[a_ind[ka]];
                    const T* a_block = A.block_data(ka);
                    const double a_norm = a_norms[static_cast<size_t>(ka)];
                    group.clear();
                    visit(g_mid, [&](int g_col, int c_dim, const T* b_block, double b_norm) {
                        if (g_col < g_row) return;  // upper triangle only
                        if (a_norm * b_norm < eps) return;
                        // No norm_bound here: the drop below is on the finished
                        // value, which has to be swept anyway once c1*A and
                        // c0*I have landed on it, so a running bound would be
                        // maintained and then never read. (rarh does read one,
                        // to gate its second contraction.)
                        const int slot = acc.obtain(g_col, r_dim, c_dim);
                        group.add(b_block, acc.value_offset[static_cast<size_t>(slot)],
                                  c_dim, r_dim, m_dim);
                    });
                    group.flush(acc.values, a_block, r_dim, m_dim);
                }

                // Scale by c2, then add c1*A and c0*I on the same row. A's own
                // columns must be REACHED even where the square left nothing --
                // c1*A[i,j] is a term in its own right -- so they are obtained
                // here rather than assumed present.
                for (size_t s = 0; s < acc.touched.size(); ++s) {
                    T* block = acc.values.data() + acc.value_offset[s];
                    const size_t count = static_cast<size_t>(r_dim) * acc.col_dim[s];
                    for (size_t e = 0; e < count; ++e) block[e] *= c2;
                }
                for (int k = a_ptr[i]; k < a_ptr[i + 1]; ++k) {
                    const int g_col = ga.get_global_index(a_ind[k]);
                    if (g_col < g_row) continue;
                    const int c_dim = ga.block_sizes[a_ind[k]];
                    const int slot = acc.obtain(g_col, r_dim, c_dim);
                    T* dest = acc.values.data() + acc.value_offset[static_cast<size_t>(slot)];
                    const T* src = A.block_data(k);
                    const size_t count = static_cast<size_t>(r_dim) * c_dim;
                    for (size_t e = 0; e < count; ++e) dest[e] += c1 * src[e];
                }
                {
                    // c0 * I on the diagonal block, which the identity forces to
                    // exist whether or not either term reached it.
                    const int slot = acc.obtain(g_row, r_dim, r_dim);
                    T* dest = acc.values.data() + acc.value_offset[static_cast<size_t>(slot)];
                    for (int d = 0; d < r_dim; ++d) dest[d * r_dim + d] += c0;
                }

                // The one drop, on finished values.
                //
                // Staged BY COLUMN and EXACTLY SIZED, which is what lets
                // assemble copy straight out of these rows; see the note there
                // and the matching one in rarh.hpp for what the alternative
                // -- reordering afterwards, into a second buffer per row --
                // cost in peak memory.
                std::vector<int>& keep_columns = row_columns[i];
                std::vector<T>& keep_values = row_values[i];
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
        }
    }

    static Matrix assemble(const Matrix& A, std::vector<std::vector<int>>& row_columns,
                           std::vector<std::vector<T>>& row_values,
                           const GhostSizes& ghost_sizes) {
        const DistGraph& ga = *A.graph;
        const int n_rows = static_cast<int>(ga.adj_ptr.size()) - 1;

        // Column order is load-bearing here, and NOT the optional output
        // ordering that spgemm_sorted_output_enabled governs in the other
        // executors: construct_result_graph sorts every row's columns
        // (result_graph.hpp), while the copy-out below walks row_columns[i] and
        // writes consecutive slots from graph_c->adj_ptr[i]. The two orders have
        // to agree; disagreeing silently transposes values onto the wrong
        // columns, which reaches the caller as a diverged iteration rather than
        // as anything the kernel itself can detect.
        //
        // It is established where the rows are staged rather than by a pass
        // here, because a pass here has to build a second buffer per row and so
        // holds a duplicate of the whole answer; see the note at the drop loop.

        DistGraph* graph_c =
            construct_result_graph(A, row_columns, ghost_sizes, "square_polynomial");
        // Deferred first touch, and built at A's page size rather than allocated
        // at the default and then rebuilt by set_page_size. The plain
        // `Matrix C(graph_c)` here faulted in the whole answer -- while the
        // staging buffers below still held all of it -- before a single block
        // had been written. Every block of C is memcpy'd below, so nothing is
        // left unwritten by skipping the zero pass.
        Matrix C = Matrix::make_overwritten_result(graph_c, A.configured_page_size());

        // Each row's staging buffer is released as soon as it is copied, so the
        // two only overlap by the rows not yet reached. Released to the KERNEL,
        // not just to malloc -- see release_and_drop in common.hpp for why the
        // difference is most of this kernel's peak.
        //
        // Walked by THREAD DOMAIN, not by row with a dynamic schedule. This copy
        // is the deferred first touch, so whichever thread runs a row is the
        // node its pages land on -- and the forward apply later splits this
        // matrix along thread_domains. A dynamic schedule placed them wherever
        // the race fell out, which silently undid the NUMA locality the removed
        // zero pass used to establish. bsr.hpp's SpGEMM has the same structure
        // and already did this correctly; these assembles did not.
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
                std::memcpy(C.mutable_block_data(slot), row_values[i].data() + at,
                            count * sizeof(T));
                at += count;
            }
            release_and_drop(row_values[i]);
        }
        }
        return C;
    }

    // Whether this rank holds any nonzero block at all.
    //
    // This used to be an MPI_Allreduce for the global MAX block norm, because
    // the stage gate divided by it. It no longer does -- the gate is
    // threshold/n_A, with the norm scale cancelling out of the error bound --
    // so the only surviving use is the "is it nonzero" guard below, which needs
    // no collective and no magnitude. Left as a collective it was one
    // synchronisation per call buying a boolean.
    static bool has_nonzero_block(const Matrix& A) {
        for (double n : A.get_block_norms()) {
            if (n > 0.0) return true;
        }
        return false;
    }

    static Matrix run(const Matrix& A, T c2, T c1, T c0, double threshold) {
        const DistGraph& ga = *A.graph;
        const int n_rows = static_cast<int>(ga.adj_ptr.size()) - 1;
        const int n_global = ga.block_displs.empty() ? n_rows : ga.block_displs.back();
        std::vector<std::vector<int>> row_columns(n_rows);
        std::vector<std::vector<T>> row_values(n_rows);

        RemoteRows remote;
        GhostSizes ghost_sizes;
        // Declared HERE, not inside the branch below: RemoteRows stores raw
        // pointers into this payload, so it has to outlive the numeric pass. As
        // a local of the `if` it was destroyed the moment the fetch finished and
        // the kernel then read freed memory -- correct on one rank, where there
        // are no ghosts at all and nothing dangles, and wrong on every other.
        SpMMGhostBlocks<T> ghosts;
        if (ga.size > 1) {
            // One round: row i of A^2 reads rows of A over A's own columns.
            const int rank = ga.rank;
            std::set<int> needed;
            for (int i = 0; i < n_rows; ++i) {
                for (int k = ga.adj_ptr[i]; k < ga.adj_ptr[i + 1]; ++k) {
                    const int g_col = ga.get_global_index(ga.adj_ind[k]);
                    if (ga.find_owner(g_col) != rank) needed.insert(g_col);
                }
            }
            const GhostMetadata meta = fetch_row_patterns(A, needed, ga.comm, ga.size, rank);
            ghosts = build_spmm_ghost_blocks<T>(
                meta, fetch_required_block_payloads(A, blocks_of(meta)));
            remote.build(ghosts, meta, n_global);
            ghost_sizes = ghosts.sizes;
        }
        const int n_owned = static_cast<int>(ga.owned_global_indices.size());
        for (size_t g = 0; g < ga.ghost_global_indices.size(); ++g) {
            ghost_sizes[ga.ghost_global_indices[g]] =
                ga.block_sizes[static_cast<size_t>(n_owned) + g];
        }

        fused_rows(A, c2, c1, c0, threshold, remote, row_of_global(A, n_global), n_global,
                   A.get_block_norms(), has_nonzero_block(A), row_columns, row_values);
        return assemble(A, row_columns, row_values, ghost_sizes);
    }
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_OPS_SPMM_SQUARE_POLYNOMIAL_HPP

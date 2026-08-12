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
#include <limits>
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

    // Shared with rarh and rhar; see common.hpp. A drift between per-executor
    // copies of these was a silent wrong-fetch bug, so they are shared by
    // construction.
    using RemoteRows = FusedRemoteRows<T>;

    // Computes rows [row_lo, row_hi): the tiled distributed path runs it once
    // per tile against that tile's fetched halo.
    static void fused_rows(const Matrix& A, T c2, T c1, T c0, double threshold,
                           const RemoteRows& remote, const std::vector<int>& a_row,
                           int n_global, const std::vector<double>& a_norms,
                           bool a_any_nonzero, int row_lo, int row_hi,
                           std::vector<std::vector<int>>& row_columns,
                           std::vector<std::vector<T>>& row_values) {
        const DistGraph& ga = *A.graph;
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
            for (int i = row_lo; i < row_hi; ++i) {
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

                // The one drop, on finished values; the staging order, its
                // exact sizing and the ORDER INVARIANT the positional copy
                // rests on are recorded at stage_row_in_column_order in
                // common.hpp.
                stage_row_in_column_order(acc, r_dim, threshold, keep_order,
                                          row_columns[i], row_values[i]);
            }
        }
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
        const std::vector<int> a_row = fused_row_of_global(A, n_global);
        const std::vector<double>& a_norms = A.get_block_norms();
        const bool a_nonzero = has_nonzero_block(A);
        GhostSizes ghost_sizes;

        if (ga.size > 1) {
            // Row i of A^2 reads rows of A over A's own columns. Patterns are
            // fetched ONCE, whole -- they are what the gates and the tile plan
            // are computed from -- and payloads are fetched per TILE of output
            // rows, so the halo held at any moment is the tile's, not the
            // union reach (fused_tile_budget_bytes).
            //
            // The payload gate keeps only what the numeric gate could: a
            // block of row k ships iff some local pair (i,k) could pass
            //
            //     a_norm(i,k) * norm(k,l) >= threshold / a_row_count(i)
            //
            // i.e. norm(k,l) >= threshold / max_i [arc(i) * a_norm(i,k)].
            // An all-zero operand runs with eps = 0 and keeps zero-norm pairs,
            // so the gate degenerates to keep-everything there, exactly as at
            // threshold 0. Entries are OUTPUT columns of an upper-only
            // product, so columns below the first owned row are trimmed too.
            const int rank = ga.rank;
            std::set<int> needed;
            std::map<int, double> colmax_a;
            int min_owned_row = std::numeric_limits<int>::max();
            for (int i = 0; i < n_rows; ++i) {
                min_owned_row = std::min(min_owned_row, ga.get_global_index(i));
                const int arc = ga.adj_ptr[i + 1] - ga.adj_ptr[i];
                for (int k = ga.adj_ptr[i]; k < ga.adj_ptr[i + 1]; ++k) {
                    const int g_col = ga.get_global_index(ga.adj_ind[k]);
                    if (ga.find_owner(g_col) == rank) continue;
                    needed.insert(g_col);
                    const double reach =
                        static_cast<double>(arc) * a_norms[static_cast<size_t>(k)];
                    auto [it, fresh] = colmax_a.emplace(g_col, reach);
                    if (!fresh && reach > it->second) it->second = reach;
                }
            }
            if (n_rows == 0) min_owned_row = 0;
            const auto gate = [&](int g_row) -> double {
                if (!(threshold > 0.0) || !a_nonzero) return 0.0;
                auto it = colmax_a.find(g_row);
                if (it == colmax_a.end() || !(it->second > 0.0)) {
                    return std::numeric_limits<double>::infinity();
                }
                return threshold / it->second;
            };
            const GhostMetadata meta = fetch_row_patterns(A, needed, ga.comm, ga.size, rank);

            // Kept (gated + trimmed) block count of one remote row's payload.
            const auto kept_count = [&](int g) -> size_t {
                auto it = meta.find(g);
                if (it == meta.end()) return 0;
                const double g_gate = gate(g);
                size_t n = 0;
                for (const auto& m : it->second) {
                    if (m.col >= min_owned_row && m.norm >= g_gate) ++n;
                }
                return n;
            };

            // ---- Tile plan: contiguous output-row ranges whose deduplicated
            // fetch stays under the budget. Dims are not in the patterns, so
            // the budget is counted in blocks at the largest local block dim
            // -- a bounded over-estimate, erring toward smaller tiles. Closing
            // a tile resets the dedup and re-charges the closing row against
            // the fresh tile. One oversized row overshoots alone, the same
            // caveat every byte-budgeted exchange here carries.
            int bs_max = 1;
            for (int s : ga.block_sizes) bs_max = std::max(bs_max, s);
            const size_t budget = fused_tile_budget_bytes();
            const size_t block_budget =
                budget == 0 ? 0
                            : std::max<size_t>(1, budget / (static_cast<size_t>(bs_max) *
                                                            static_cast<size_t>(bs_max) * sizeof(T)));
            std::vector<int> tile_bound{0};
            std::vector<char> row_flag(static_cast<size_t>(n_global), 0);
            std::vector<int> flagged;
            if (block_budget > 0) {
                const auto charge_row = [&](int i) -> size_t {
                    size_t c = 0;
                    for (int k = ga.adj_ptr[i]; k < ga.adj_ptr[i + 1]; ++k) {
                        const int g_col = ga.get_global_index(ga.adj_ind[k]);
                        if (ga.find_owner(g_col) == rank) continue;
                        if (row_flag[static_cast<size_t>(g_col)]) continue;
                        row_flag[static_cast<size_t>(g_col)] = 1;
                        flagged.push_back(g_col);
                        c += kept_count(g_col);
                    }
                    return c;
                };
                size_t acc = 0;
                for (int i = 0; i < n_rows; ++i) {
                    size_t row_cost = charge_row(i);
                    if (acc > 0 && acc + row_cost > block_budget) {
                        tile_bound.push_back(i);
                        for (int g : flagged) row_flag[static_cast<size_t>(g)] = 0;
                        flagged.clear();
                        acc = 0;
                        row_cost = charge_row(i);
                    }
                    acc += row_cost;
                }
                for (int g : flagged) row_flag[static_cast<size_t>(g)] = 0;
                flagged.clear();
            }
            tile_bound.push_back(n_rows);

            // The rounds are COLLECTIVE: every payload fetch is, so a rank
            // that needed fewer tiles runs empty ones.
            long long my_tiles = static_cast<long long>(tile_bound.size()) - 1;
            long long rounds = my_tiles;
            MPI_Allreduce(&my_tiles, &rounds, 1, MPI_LONG_LONG, MPI_MAX, ga.comm);
            while (static_cast<long long>(tile_bound.size()) - 1 < rounds) {
                tile_bound.push_back(n_rows);
            }

            for (long long r = 0; r < rounds; ++r) {
                const int lo = tile_bound[static_cast<size_t>(r)];
                const int hi = tile_bound[static_cast<size_t>(r) + 1];
                std::vector<BlockID> want;
                for (int i = lo; i < hi; ++i) {
                    for (int k = ga.adj_ptr[i]; k < ga.adj_ptr[i + 1]; ++k) {
                        const int g_col = ga.get_global_index(ga.adj_ind[k]);
                        if (ga.find_owner(g_col) == rank) continue;
                        if (row_flag[static_cast<size_t>(g_col)]) continue;
                        row_flag[static_cast<size_t>(g_col)] = 1;
                        flagged.push_back(g_col);
                        auto it = meta.find(g_col);
                        if (it == meta.end()) continue;
                        const double g_gate = gate(g_col);
                        for (const auto& m : it->second) {
                            if (m.col >= min_owned_row && m.norm >= g_gate) {
                                want.push_back(BlockID{g_col, m.col});
                            }
                        }
                    }
                }
                for (int g : flagged) row_flag[static_cast<size_t>(g)] = 0;
                flagged.clear();

                SpMMGhostBlocks<T> ghosts = build_spmm_ghost_blocks<T>(
                    meta, fetch_required_block_payloads(A, want));
                for (const auto& [gid, dim] : ghosts.sizes) ghost_sizes[gid] = dim;
                RemoteRows remote;
                remote.build(ghosts, meta, n_global);
                fused_rows(A, c2, c1, c0, threshold, remote, a_row, n_global,
                           a_norms, a_nonzero, lo, hi, row_columns, row_values);
                // Physically back to the OS, not to malloc's free list: the
                // next tile's arena would otherwise stack on this one's
                // (release_and_drop records why the dtor alone does not).
                release_and_drop(ghosts.arena);
            }
        } else {
            RemoteRows none;
            fused_rows(A, c2, c1, c0, threshold, none, a_row, n_global,
                       a_norms, a_nonzero, 0, n_rows, row_columns, row_values);
        }
        const int n_owned = static_cast<int>(ga.owned_global_indices.size());
        for (size_t g = 0; g < ga.ghost_global_indices.size(); ++g) {
            ghost_sizes[ga.ghost_global_indices[g]] =
                ga.block_sizes[static_cast<size_t>(n_owned) + g];
        }
        return fused_assemble(A, row_columns, row_values, ghost_sizes, "square_polynomial");
    }
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_OPS_SPMM_SQUARE_POLYNOMIAL_HPP

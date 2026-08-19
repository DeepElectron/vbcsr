#ifndef VBCSR_DETAIL_OPS_SPMM_RARH_HPP
#define VBCSR_DETAIL_OPS_SPMM_RARH_HPP

#include "../../distributed/block_payload_exchange.hpp"
#include "../../distributed/result_graph.hpp"
#include "common.hpp"
#include "../../kernels/rowmajor_kernels.hpp"
#include "../../../scalar_traits.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <type_traits>
#include <cstring>
#include <map>
#include <set>
#include <unordered_map>
#include <stdexcept>
#include <vector>
#include <cstdio>

namespace vbcsr::detail {

// C = A B A^H, computed one result row at a time, never forming A B.
//
// Named for the formula, as the same triple product is elsewhere in sparse
// linear algebra: it is PETSc's MatRARt and hypre's RAP -- R A R^H, with the
// outer factor here being the matrix the method is called on.
//
// Why this is a kernel and not two calls to spmm:
//
// The intermediate A B is DENSER THAN THE ANSWER. Its sparsity radius is
// r_A + r_B; the answer's is 2 r_A + r_B, but the answer of a converging
// product (Newton-Schulz's X <- P X P, driving X toward the identity)
// collapses under the drop threshold while the intermediate does not. Measured
// on a 178k-atom moire overlap: A B carries ~7000 block neighbours per row
// against the answer's handful -- 3.4 TB against megabytes. Two spmm calls
// must materialise that 3.4 TB. This kernel holds one of its rows, ~19 MB, per
// thread.
//
// The second consequence is accuracy, and it runs the same way. Two spmm calls
// DROP at each link -- A B is truncated and the error is then propagated
// through the second product -- and then drop again. Here nothing is truncated
// between the stages: pairs are skipped only when a per-row error budget proves
// they cannot matter (see fused_rows), and the finished result block is the one
// thing the threshold is applied to. Both routes therefore carry the same
// bounded-neglect contract, but this one spends its whole budget once instead
// of twice, which is what makes a looser threshold affordable.
//
// Contract, the caller's to honour and unchecked (there is no cheap test):
//   * A is Hermitian. The kernel uses A[j,l]^H == A[l,j] to reach the second
//     contraction by ROW, which is what removes the transpose that
//     spmm(transA) would have materialised.
//   * B is Hermitian, so that C is, and only the upper triangle is produced.
// Complete it with complete_hermitian().
//
// Result: block columns at or above the diagonal, with every block's Frobenius
// norm >= threshold -- the same rule filter_blocks applies, applied before the
// storage is committed rather than after, so the allocation is the final size.
template <typename Matrix>
struct RARhExecutor {
    using T = typename Matrix::value_type;
    // Row-scratch primitives shared with the square polynomial; see common.hpp.
    using RowAccumulator = FusedRowAccumulator<T>;
    using ProductGroup = FusedProductGroup<T>;
    static double squared_magnitude(const T& v) { return block_squared_magnitude<T>(v); }

    // dest(r x c) += lhs(r x m) * rhs(m x c), canonical row-major blocks.
    //
    // rm_gemm, not a hand-rolled loop: it is the same AVX2/FMA kernel both spmm
    // executors reach for at LCAO block sizes, and its C += A X semantics are
    // exactly this accumulate. The scalar ikj loop this replaced was the whole
    // of the fused kernel's flop-rate deficit against spmm -- ~68 GFLOP/s
    // against ~320 measured on diamond sc888 -- so it was competing with hand
    // written intrinsics, not with batching, which is a thing neither executor
    // does at these sizes: BSR gates the vendor batch at bs >= 16 (measured
    // 0.50x at bs 5, a wash at bs 13, 1.30x at bs 26 -- the vendor's throughput
    // only overtakes its dispatch cost on big blocks) and VBCSR gates it at any
    // dim > 20, where it also has to pack every operand into scratch and copy
    // the results back out.
    //
    // Consequence worth knowing: this call is UNGATED, so above those sizes the
    // fused kernel keeps using rm_gemm where spmm would switch to the vendor
    // batch and win ~1.3x. Nothing in rsatb reaches it (carbon is 13 orbitals),
    // but vbcsr is generic and a heavier basis would.
    //
    // Panel scratch inside rm_gemm is thread_local, so this is safe from inside
    // the OpenMP region below.

    // One contraction's worth of products against a single fixed left operand.
    //
    // Both stages of A B A^H produce exactly this shape -- stage 1 holds A[i,l]
    // fixed across every B[l,m], stage 2 holds the intermediate row block fixed
    // across every A[m,j] -- and in both the destinations are distinct, one per
    // accumulator column. That is the grouped-batch shape, so above the block
    // size where the vendor wins this becomes one call instead of a loop, the
    // same dispatch spmm makes (common.hpp's vendor_batch_profitable).
    //
    // Destinations are carried as OFFSETS, not pointers: obtain() may grow the
    // accumulator's value arena and reallocate it, so a pointer taken when the
    // product was queued can dangle by the time the group is flushed. They are
    // resolved here, after the last obtain().


    // Shared with square_polynomial and rhar; see common.hpp. A drift between
    // per-executor copies of these was a silent wrong-fetch bug, so they are
    // shared by construction.
    using RemoteRows = FusedRemoteRows<T>;

    // The numeric pass, shared by the serial and distributed entry points.
    // `remote_a` / `remote_b` are empty on one rank; on many they carry the
    // rows this rank does not own but must read. Computes rows
    // [row_lo, row_hi): the tiled distributed path runs it once per tile
    // against that tile's fetched halo.
    static void fused_rows(const Matrix& A, const Matrix& B, double threshold,
                           const RemoteRows& remote_a, const RemoteRows& remote_b,
                           const std::vector<int>& a_row, const std::vector<int>& b_row,
                           int n_global,
                           const std::vector<double>& a_norms,
                           const std::vector<double>& b_norms,
                           double a_max_norm, int row_lo, int row_hi,
                           std::vector<std::vector<int>>& row_columns,
                           std::vector<std::vector<T>>& row_values,
                           int& bad_operand) {
        const DistGraph& ga = *A.graph;
        const DistGraph& gb = *B.graph;
        const std::vector<int>& a_ptr = ga.adj_ptr;
        const auto& a_ind = ga.adj_ind;
        const std::vector<int>& b_ptr = gb.adj_ptr;
        const auto& b_ind = gb.adj_ind;

        #pragma omp parallel reduction(|:bad_operand)
        {
            RowAccumulator inner;
            RowAccumulator outer;
            inner.resize(n_global);
            outer.resize(n_global);
            ProductGroup group;
            // Surviving accumulator slots, in column order. Per thread and
            // reused across rows so the ordering pass allocates once.
            std::vector<int> keep_order;

            // One block of a row of A, wherever that row lives. `fn` receives
            // the block's Frobenius norm alongside it, because both gates below
            // need it and re-deriving it per pair would cost more than the
            // product it saves.
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

            // The row width the stage-one gate divides by -- the FULL pattern
            // width, not the fetched entry count: the error budget telescopes
            // over the row as it exists, and a norm-gated fetch shipping
            // fewer blocks must not loosen the gate on the ones it shipped.
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

                // Per-row error budget, in the shape spmm already uses: the sum
                // of what either stage neglects is bounded by `threshold`, so a
                // block that survives the final drop was never built from
                // materially incomplete parts. Half the budget per stage, and
                // stage 1's is divided again by A's largest block norm, because
                // a term dropped there is later multiplied by one more factor
                // of A before it reaches the answer:
                //
                //   |dropped C[i,j]| <= sum_l,m |A[i,l]||B[l,m]||A[m,j]|
                //                    <= n_A * eps1 * max|A|   = threshold/2
                //   plus stage 2's   <= n_T * eps2             = threshold/2
                //
                // This is the ONE contract difference from an exact fused
                // product, and it is the same bounded-neglect contract spmm has
                // carried all along -- so the two-call route this replaces was
                // never exact either, and pruned twice where this prunes once.
                //
                // Worth what it costs only where the operands have a SPREAD of
                // block norms, which is the regime this kernel exists for.
                // Measured, forced-propagate Newton-Schulz on diamond, PXP step
                // 1: 4096 atoms at sptol 1e-4 (Z 2255 nbr/row of 4096, so
                // genuinely sparse) 2096.5s unpruned against 1983.1s pruned, a
                // 5.4% gain for a 0.23% move in the step's residual (8.012e-4 ->
                // 8.031e-4). On 1024 atoms, where the fill-in reaches the
                // periodic image and every row saturates, it is a wash -- the
                // gate almost never fires and there is nothing to win. The
                // unchanged phases of the same run pair bound the machine noise
                // at under 1% (seed 0.9%, X^2 0.6%), which is what makes the
                // 5.4% a result rather than scatter.
                const int a_row_count = a_ptr[i + 1] - a_ptr[i];

                // (A B)[i, :] -- scratch, never leaves the thread.
                for (int ka = a_ptr[i]; ka < a_ptr[i + 1]; ++ka) {
                    const int g_mid = ga.get_global_index(a_ind[ka]);
                    const int m_dim = ga.block_sizes[a_ind[ka]];
                    const T* a_block = A.block_data(ka);
                    const double a_norm = a_norms[static_cast<size_t>(ka)];

                    // Stage-one budget, and it has to carry BOTH fan-outs.
                    //
                    // A skipped pair (k,l) contributes A[i,k] B[k,l] to
                    // (A B)[i,l], at most eps1, and that error reaches C[i,j]
                    // multiplied by A[j,l]^H, so at most eps1 * a_max_norm. The
                    // number of skipped pairs that can reach ONE output block is
                    // not n_A -- it is the (k,l) grid, n_A rows of B each up to
                    // |B row k| wide.
                    //
                    // The old bound divided only by n_A and so was unsound. A
                    // 6x6 scalar example makes it concrete: A all ones, B all
                    // 0.08, threshold 1. The old eps1 is 1/12 = 0.0833, every
                    // pair norm is 0.08, so EVERY pair is skipped and the result
                    // is 0 -- while the exact answer is 36 * 0.08 = 2.88 per
                    // entry, an error of 2.88 against a claimed bound of 1. The
                    // test case `coherent` covers exactly this; the previous
                    // random decaying matrices never accumulated coherently
                    // enough to expose it.
                    //
                    // Scaling per k by |B row k| keeps the sum telescoping:
                    // sum_k |B_k| * eps1_k * a_max_norm = threshold / 2.
                    const int b_row_count = count_b(g_mid);
                    const double eps1 =
                        (threshold > 0.0 && a_max_norm > 0.0 && b_row_count > 0)
                            ? threshold / (2.0 * std::max(1, a_row_count) *
                                           static_cast<double>(b_row_count) * a_max_norm)
                            : 0.0;
                    group.clear();
                    visit_b(g_mid, [&](int g_inner, int c_dim, const T* b_block,
                                       double b_norm) {
                        const double pair = a_norm * b_norm;
                        if (pair < eps1) return;
                        const int slot = inner.obtain(g_inner, r_dim, c_dim);
                        inner.norm_bound[static_cast<size_t>(slot)] += pair;
                        group.add(b_block, inner.value_offset[static_cast<size_t>(slot)],
                                  c_dim, r_dim, m_dim);
                    });
                    group.flush(inner.values, a_block, r_dim, m_dim);
                }

                // C[i, j>=i] = sum_l (A B)[i,l] A[l,j], via A[j,l]^H == A[l,j].
                const double eps2 =
                    (threshold > 0.0 && !inner.touched.empty())
                        ? threshold / (2.0 * static_cast<double>(inner.touched.size()))
                        : 0.0;
                for (size_t sIdx = 0; sIdx < inner.touched.size(); ++sIdx) {
                    const int g_inner = inner.touched[sIdx];
                    const int m_dim = inner.col_dim[sIdx];
                    const T* inner_block = inner.values.data() + inner.value_offset[sIdx];
                    // An UPPER bound, so gating on it prunes conservatively --
                    // it can only ever keep a product the exact norm would have
                    // dropped, never the reverse, which is the direction the
                    // error bound needs.
                    const double inner_norm = inner.norm_bound[sIdx];
                    // The row must still be REACHED even when every one of its
                    // products is negligible: reaching it is what proves A has
                    // this column as a row, which is the Hermitian check below.
                    group.clear();
                    const bool found = visit_a(g_inner, [&](int g_col, int c_dim,
                                                            const T* a_block,
                                                            double a_col_norm) {
                        if (g_col < g_row) return;  // upper triangle only
                        if (inner_norm * a_col_norm < eps2) return;
                        const int slot = outer.obtain(g_col, r_dim, c_dim);
                        group.add(a_block, outer.value_offset[static_cast<size_t>(slot)],
                                  c_dim, r_dim, m_dim);
                    });
                    // inner_block is read INSIDE the flush, so it must still be
                    // valid there: the group writes into `outer`, never `inner`,
                    // so inner's arena cannot have moved under it.
                    group.flush(outer.values, inner_block, r_dim, m_dim);
                    if (!found) bad_operand = 1;
                }

                // The one drop, on finished values; the staging order and its
                // exact sizing carry the peak-memory argument, and the ORDER
                // INVARIANT the positional copy rests on -- both recorded at
                // stage_row_in_column_order in common.hpp.
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
        const RemoteRows none;
        int bad_operand = 0;
        fused_rows(A, B, threshold, none, none,
                   fused_row_of_global(A, n_global), fused_row_of_global(B, n_global),
                   n_global, A.get_block_norms(), B.get_block_norms(),
                   fused_global_max_norm(A), 0, n_rows, row_columns, row_values, bad_operand);
        if (bad_operand) {
            throw std::runtime_error(
                "rarh_upper: the left operand is not Hermitian "
                "(a column appears that is not also a row)");
        }
        const bool profile = std::getenv("VBCSR_PROFILE_RARH") != nullptr;
        const double rss_staged = profile ? profile_rss_gb() : 0.0;
        GhostSizes no_ghosts;
        Matrix out = fused_assemble(A, row_columns, row_values, no_ghosts, "rarh");
        if (profile) {
            std::fprintf(stderr, "VBCSR_PROFILE_RARH(local) rssGB staged=%.2f assembled=%.2f\n",
                         rss_staged, profile_rss_gb());
        }
        return out;
    }

    // Two rounds of exchange, and they cannot be collapsed into one: the second
    // asks for rows of A over the support of A B, and that support is not known
    // until the first round's patterns have arrived. It is the price of never
    // forming A B, and it is paid in metadata and halo rows rather than in a
    // matrix the size of the problem.
    static Matrix run_distributed(const Matrix& A, const Matrix& B, double threshold) {
        const bool profile = std::getenv("VBCSR_PROFILE_RARH") != nullptr;
        const double rss_in = profile ? profile_rss_gb() : 0.0;
        const DistGraph& ga = *A.graph;
        const DistGraph& gb = *B.graph;
        const int rank = ga.rank;
        const int n_rows = static_cast<int>(ga.adj_ptr.size()) - 1;
        const int n_global = ga.block_displs.back();

        // The fetch gates below need the same global bound the numeric gate
        // uses; taken here (one Allreduce) instead of at the fused_rows call.
        const double a_max_norm = fused_global_max_norm(A);
        const std::vector<double>& a_norms = A.get_block_norms();
        const std::vector<double>& b_norms = B.get_block_norms();

        // ---- Round 1: the rows of B that A's columns reach, and -- per
        // reached row -- the strongest gate any local pair can put on its
        // blocks. A block of B row k is worth shipping only if SOME local pair
        // (i,k) could pass the numeric stage-one gate with it:
        //
        //     a_norm(i,k) * b_norm(k,l) >= threshold /
        //         (2 * a_row_count(i) * width_b(k) * a_max_norm)
        //
        // which, taken over all local i, is a per-row norm floor on B[k,l]
        // with colmax_a(k) = max_i [a_row_count(i) * a_norm(i,k)]. A block
        // below the floor fails the numeric gate for EVERY local use, so
        // skipping its payload changes what travels, never what is kept.
        std::set<int> need_b;
        std::map<int, double> colmax_a;  // needed row of B -> max_i arc(i)*a_norm(i,k)
        std::vector<double> a_row_sum(static_cast<size_t>(n_rows), 0.0);
        double s_max = 0.0;      // max_i sum_k a_norm(i,k)
        double w_bound_d = 0.0;  // max_i sum_k width_b(k): upper bound on |touched_i|
        for (int i = 0; i < n_rows; ++i) {
            const int arc = ga.adj_ptr[i + 1] - ga.adj_ptr[i];
            double row_sum = 0.0;
            for (int k = ga.adj_ptr[i]; k < ga.adj_ptr[i + 1]; ++k) {
                const int g_col = ga.get_global_index(ga.adj_ind[k]);
                const double reach = static_cast<double>(arc) * a_norms[static_cast<size_t>(k)];
                row_sum += a_norms[static_cast<size_t>(k)];
                auto [it, fresh] = colmax_a.emplace(g_col, reach);
                if (!fresh && reach > it->second) it->second = reach;
                if (gb.find_owner(g_col) != rank) need_b.insert(g_col);
            }
            a_row_sum[static_cast<size_t>(i)] = row_sum;
            s_max = std::max(s_max, row_sum);
        }
        GhostMetadata meta_b = fetch_row_patterns(B, need_b, ga.comm, ga.size, rank);

        const auto gate_b = [&](int g_row, int width) -> double {
            if (!(threshold > 0.0) || a_max_norm <= 0.0 || width <= 0) return 0.0;
            auto it = colmax_a.find(g_row);
            if (it == colmax_a.end() || !(it->second > 0.0)) {
                return std::numeric_limits<double>::infinity();
            }
            return threshold / (2.0 * static_cast<double>(width) * a_max_norm * it->second);
        };

        // ---- The support of A B, from patterns alone, carrying per column
        // the largest B-block norm that survives its row's fetch gate. A
        // column whose every incoming pair is provably below the numeric gate
        // can never enter a row's accumulator, so it needs neither round-2
        // pattern nor payload -- the local rows of B go through the SAME gate
        // for exactly that reason. (colmax doubles as the reach flag: 0 means
        // unreached or unreachable, and it feeds the round-2 gate below.)
        //
        // `reached` is kept SEPARATELY from colmax_b because the two answer
        // different questions: reached says a kept block exists (at threshold
        // 0 a ZERO-norm block is numerically kept, enters the accumulator,
        // and its row must be answerable for the hermiticity probe), colmax
        // feeds the round-2 norm gate. At threshold > 0 a kept block always
        // has norm >= gate > 0, so reached implies colmax > 0 there.
        //
        // Both are indexed by a dense numbering over the columns that could
        // possibly be reached -- enumerable straight from the patterns and the
        // local rows of B, without the cross product below -- rather than by
        // global id. Same answers, memory proportional to the halo instead of
        // to the system.
        FusedDenseIds cand;
        {
            std::vector<int> scratch;
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
        // One slot lookup per B ENTRY, not per visit. The walk below is the
        // cross product (local rows x A's columns x those rows' columns), so a
        // binary search inside it costs ~1.5e8 searches on a 1024-row rank --
        // about 3 seconds, serial, for an answer that depends only on the
        // entry. Resolved once here into a flat table instead: local blocks
        // first, then each fetched pattern's blocks.
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

        // (col, flat entry index) for the kept blocks of a B row, so callers
        // index whichever slot table they need without searching.
        const auto for_each_kept_b_entry = [&](int g_mid, auto&& fn) {
            auto local = gb.global_to_local.find(g_mid);
            if (local != gb.global_to_local.end() &&
                local->second < static_cast<int>(gb.adj_ptr.size()) - 1) {
                const int lb = local->second;
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
            for (const auto& meta : pattern->second) {
                const size_t here = at++;
                if (meta.norm >= gate) fn(meta.col, here);
            }
        };

        std::vector<char> reached(cand.size(), 0);
        std::vector<double> colmax_b(cand.size(), 0.0);
        for (int i = 0; i < n_rows; ++i) {
            for (int k = ga.adj_ptr[i]; k < ga.adj_ptr[i + 1]; ++k) {
                const int g_mid = ga.get_global_index(ga.adj_ind[k]);
                auto local = gb.global_to_local.find(g_mid);
                if (local != gb.global_to_local.end() && local->second < static_cast<int>(gb.adj_ptr.size()) - 1) {
                    const int lb = local->second;
                    const double gate = gate_b(g_mid, gb.adj_ptr[lb + 1] - gb.adj_ptr[lb]);
                    for (int e = gb.adj_ptr[lb]; e < gb.adj_ptr[lb + 1]; ++e) {
                        const double norm = b_norms[static_cast<size_t>(e)];
                        if (norm < gate) continue;
                        const int slot = entry_cand[static_cast<size_t>(e)];
                        if (slot < 0) continue;
                        reached[static_cast<size_t>(slot)] = 1;
                        colmax_b[static_cast<size_t>(slot)] =
                            std::max(colmax_b[static_cast<size_t>(slot)], norm);
                    }
                    continue;
                }
                auto pattern = meta_b.find(g_mid);
                if (pattern == meta_b.end()) continue;
                const double gate = gate_b(g_mid, static_cast<int>(pattern->second.size()));
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
        }
        for (int i = 0; i < n_rows; ++i) {
            double w = 0.0;
            for (int k = ga.adj_ptr[i]; k < ga.adj_ptr[i + 1]; ++k) {
                const int g_mid = ga.get_global_index(ga.adj_ind[k]);
                auto local = gb.global_to_local.find(g_mid);
                if (local != gb.global_to_local.end() && local->second < static_cast<int>(gb.adj_ptr.size()) - 1) {
                    w += gb.adj_ptr[local->second + 1] - gb.adj_ptr[local->second];
                } else if (auto pattern = meta_b.find(g_mid); pattern != meta_b.end()) {
                    w += static_cast<double>(pattern->second.size());
                }
            }
            w_bound_d = std::max(w_bound_d, w);
        }

        // ---- Round 2: the rows of A over that support. The fetch gate is
        // the stage-two numeric gate with every unknown bounded from the safe
        // side: inner_norm(i,l) <= a_row_sum(i) * colmax_b(l) <= s_max *
        // colmax_b(l), and |touched_i| <= w_bound, so a block A[l,j] below
        //
        //     threshold / (2 * w_bound * s_max * colmax_b(l))
        //
        // fails the numeric eps2 gate for every local row that reaches l.
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
            if (!(cm > 0.0) || !(s_max > 0.0) || !(w_bound_d > 0.0)) {
                return std::numeric_limits<double>::infinity();
            }
            return threshold / (2.0 * w_bound_d * s_max * cm);
        };

        // ---- Payloads, gated and TILED: what provably cannot pass the
        // numeric gates never travels, and what can travels one tile of
        // output rows at a time, so the halo held at any moment is the
        // tile's, not the union reach (fused_tile_budget_bytes). Round 2
        // additionally trims columns below the first owned row -- its entries
        // are output columns of an upper-only product, so those are dead on
        // arrival. Round 1's entries are contraction indices; no trim.
        int min_owned_row = std::numeric_limits<int>::max();
        for (int i = 0; i < n_rows; ++i) {
            min_owned_row = std::min(min_owned_row, ga.get_global_index(i));
        }
        if (n_rows == 0) min_owned_row = 0;

        // Kept payload block count of a remote B / A row, as its gate ships it.
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
        // The kept pattern of a B row -- local through its own gate, remote
        // through the metadata -- feeding both the tile plan's stage-2 charge
        // and the per-tile stage-2 need set.
        const auto for_each_kept_b_col = [&](int g_mid, auto&& fn) {
            auto local = gb.global_to_local.find(g_mid);
            if (local != gb.global_to_local.end() &&
                local->second < static_cast<int>(gb.adj_ptr.size()) - 1) {
                const int lb = local->second;
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

        // ---- Use spans: for every remote row, the FIRST and LAST output row
        // that needs it. Both sides come out of one walk of the same reach the
        // numeric loop will take -- A's columns for the rows of B, and their
        // kept columns for the rows of A -- and the walk is over patterns, so
        // it costs no payload. Output rows are visited in order, which makes
        // "first" the first write and "last" the latest.
        //
        // Keyed by a dense numbering over the rows whose patterns we hold --
        // the only rows that can have a span at all -- rather than by global
        // id, so these four tables cost O(halo) instead of 16 bytes per global
        // row per rank.
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
            for (int k = ga.adj_ptr[i]; k < ga.adj_ptr[i + 1]; ++k) {
                const int g_mid = ga.get_global_index(ga.adj_ind[k]);
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

        // ---- Round plan: cut where the NEW payload reaches the budget. Each
        // row is charged only for what is born on it, so the charges sum to the
        // union reach and the round count is ceil(union / budget) -- no pattern
        // can make a round repeat what an earlier round already brought in.
        int bs_max = 1;
        for (int s : ga.block_sizes) bs_max = std::max(bs_max, s);
        const size_t budget = fused_tile_budget_bytes(ga.comm);
        const size_t block_budget =
            budget == 0 ? 0
                        : std::max<size_t>(1, budget / (static_cast<size_t>(bs_max) *
                                                        static_cast<size_t>(bs_max) * sizeof(T)));
        // Arrivals ordered by the row they are born on, so both the plan below
        // and each round's want list read a slice instead of sweeping the
        // global index once per round.
        std::vector<std::pair<int, int>> b_born, a_born;  // (birth row, global row)
        std::vector<size_t> arrivals(static_cast<size_t>(n_rows) + 1, 0);
        std::vector<size_t> departures(static_cast<size_t>(n_rows) + 1, 0);
        for (size_t slot = 0; slot < b_ids.size(); ++slot) {
            const int born_b = b_birth[slot];
            if (born_b < 0) continue;
            const int g = b_ids.ids[slot];
            b_born.emplace_back(born_b, g);
            const size_t blocks = kept_b(g);
            arrivals[static_cast<size_t>(born_b)] += blocks;
            departures[static_cast<size_t>(b_death[slot])] += blocks;
        }
        for (size_t slot = 0; slot < a_ids.size(); ++slot) {
            const int born_a = a_birth[slot];
            if (born_a < 0) continue;
            const int g = a_ids.ids[slot];
            a_born.emplace_back(born_a, g);
            const size_t blocks = kept_a(g);
            arrivals[static_cast<size_t>(born_a)] += blocks;
            departures[static_cast<size_t>(a_death[slot])] += blocks;
        }
        std::sort(b_born.begin(), b_born.end());
        std::sort(a_born.begin(), a_born.end());
        const bool refetch_halo = fused_must_refetch(arrivals, departures, n_rows, block_budget);

        // ---- Refetch regime: tiles whose OWN halo fits the budget, dropped
        // and re-fetched per tile. Reached only when the live set itself does
        // not fit, where no release schedule can bound residency and the
        // choice is between paying for a block twice and being OOM-killed.
        const int b_local_rows = static_cast<int>(gb.adj_ptr.size()) - 1;
        std::vector<char> bflag(b_ids.size(), 0), aflag(a_ids.size(), 0);
        std::vector<char> midflag(static_cast<size_t>(std::max(0, b_local_rows)), 0);
        std::vector<int> btouched, atouched, midtouched;
        const auto reset_flags = [&] {
            for (int t : btouched) bflag[static_cast<size_t>(t)] = 0;
            for (int t : atouched) aflag[static_cast<size_t>(t)] = 0;
            for (int t : midtouched) midflag[static_cast<size_t>(t)] = 0;
            btouched.clear();
            atouched.clear();
            midtouched.clear();
        };
        // One output row's reach, deduplicated against the tile so far.
        const auto walk_row = [&](int i, auto&& on_b_row, auto&& on_a_row) {
            for (int k = ga.adj_ptr[i]; k < ga.adj_ptr[i + 1]; ++k) {
                const int g_mid = ga.get_global_index(ga.adj_ind[k]);
                const int s_b = b_ids.of(g_mid);
                if (s_b >= 0) {
                    if (bflag[static_cast<size_t>(s_b)]) continue;
                    bflag[static_cast<size_t>(s_b)] = 1;
                    btouched.push_back(s_b);
                    on_b_row(g_mid);
                } else {
                    auto local = gb.global_to_local.find(g_mid);
                    if (local == gb.global_to_local.end() ||
                        local->second >= b_local_rows) {
                        continue;
                    }
                    if (midflag[static_cast<size_t>(local->second)]) continue;
                    midflag[static_cast<size_t>(local->second)] = 1;
                    midtouched.push_back(local->second);
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
                walk_row(i, [&](int g) { c += kept_b(g); },
                         [&](int l) { c += kept_a(l); });
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
            tile_bound = fused_round_plan(arrivals, departures, n_rows, block_budget);
        }
        tile_bound.push_back(n_rows);

        // The rounds are COLLECTIVE (two payload fetches each), so a rank
        // that needed fewer tiles runs empty ones.
        long long my_tiles = static_cast<long long>(tile_bound.size()) - 1;
        long long rounds = my_tiles;
        MPI_Allreduce(&my_tiles, &rounds, 1, MPI_LONG_LONG, MPI_MAX, ga.comm);
        while (static_cast<long long>(tile_bound.size()) - 1 < rounds) {
            tile_bound.push_back(n_rows);
        }

        // Which round a row's last reader falls in, so a delivery knows when
        // it may go. Rows are cut in increasing order, so this is a search over
        // the boundaries.
        const auto round_of_row = [&](int i) -> long long {
            const auto it = std::upper_bound(tile_bound.begin(), tile_bound.end(), i);
            return static_cast<long long>(it - tile_bound.begin()) - 1;
        };

        const double rss_ghosts = profile ? profile_rss_gb() : 0.0;
        const std::vector<int> a_row_map = fused_row_of_global(A, n_global);
        const std::vector<int> b_row_map = fused_row_of_global(B, n_global);
        std::vector<std::vector<int>> row_columns(n_rows);
        std::vector<std::vector<T>> row_values(n_rows);
        GhostSizes ghost_sizes;
        int bad_operand = 0;
        double rss_numeric = rss_ghosts;
        FusedHaloStream<T> halo_b, halo_a;
        halo_b.init(meta_b, n_global);
        halo_a.init(meta_a, n_global);
        size_t b_cursor = 0, a_cursor = 0;
        double secs_fetch = 0.0, secs_numeric = 0.0, secs_wait = 0.0;
        for (long long r = 0; r < rounds; ++r) {
            const int lo = tile_bound[static_cast<size_t>(r)];
            const int hi = tile_bound[static_cast<size_t>(r) + 1];
            // Only what is BORN in this round: everything else either already
            // arrived and is still held, or is not needed yet.
            std::vector<BlockID> want_b, want_a;
            long long death_b = r, death_a = r;
            if (refetch_halo) {
                // Everything this tile reads, and everything goes at its end.
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

            // How long the team spends waiting for its slowest rank, apart
            // from moving bytes. Worth separating because the two have
            // different fixes -- a one-sided transport would remove the wait
            // and nothing else, and measuring it is how we found the wait was
            // never the problem. Only under the stats flag: the barrier itself
            // perturbs what it measures.
            const double t_bar0 = MPI_Wtime();
            if (fused_halo_stats_enabled()) MPI_Barrier(ga.comm);
            const double t_fetch0 = MPI_Wtime();
            secs_wait += t_fetch0 - t_bar0;
            auto ghosts_b = build_spmm_ghost_blocks<T>(
                meta_b, fetch_required_block_payloads(B, want_b));
            auto ghosts_a = build_spmm_ghost_blocks<T>(
                meta_a, fetch_required_block_payloads(A, want_a));
            // Result columns come from A's rows; collected per round since
            // only the round that fetches a row carries its sizes.
            for (const auto& [gid, dim] : ghosts_a.sizes) ghost_sizes[gid] = dim;

            halo_b.absorb(std::move(ghosts_b), death_b);
            halo_a.absorb(std::move(ghosts_a), death_a);
            const double t_num0 = MPI_Wtime();
            secs_fetch += t_num0 - t_fetch0;
            fused_rows(A, B, threshold, halo_a.rows, halo_b.rows,
                       a_row_map, b_row_map, n_global, a_norms, b_norms,
                       a_max_norm, lo, hi, row_columns, row_values, bad_operand);
            rss_numeric = profile ? std::max(rss_numeric, profile_rss_gb()) : 0.0;
            secs_numeric += MPI_Wtime() - t_num0;
            halo_b.retire(r);
            halo_a.retire(r);
        }
        fused_report_halo("rarh", ga.comm, rank, rounds,
                          halo_b.fetched_blocks + halo_a.fetched_blocks,
                          halo_b.peak_live_blocks + halo_a.peak_live_blocks,
                          A.get_block_norms().size(),
                          static_cast<size_t>(bs_max) * static_cast<size_t>(bs_max) * sizeof(T),
                          secs_fetch, secs_numeric, budget, refetch_halo);
        if (fused_halo_stats_enabled()) {
            double w = secs_wait, wmax = 0.0;
            MPI_Reduce(&w, &wmax, 1, MPI_DOUBLE, MPI_MAX, 0, ga.comm);
            double wsum = 0.0;
            MPI_Reduce(&w, &wsum, 1, MPI_DOUBLE, MPI_SUM, 0, ga.comm);
            if (rank == 0) {
                std::fprintf(stderr,
                             "VBCSR_FUSED_STATS rarh   imbalance-wait max=%.1fs "
                             "mean=%.1fs\n",
                             wmax, wsum / ga.size);
            }
        }
        int any_bad = bad_operand;
        MPI_Allreduce(&bad_operand, &any_bad, 1, MPI_INT, MPI_MAX, ga.comm);
        if (any_bad) {
            throw std::runtime_error(
                "rarh_upper: the left operand is not Hermitian "
                "(a column appears that is not also a row)");
        }

        // A's own ghost columns complete the result ghost sizes.
        const int n_owned = static_cast<int>(ga.owned_global_indices.size());
        for (size_t g = 0; g < ga.ghost_global_indices.size(); ++g) {
            ghost_sizes[ga.ghost_global_indices[g]] =
                ga.block_sizes[static_cast<size_t>(n_owned) + g];
        }
        Matrix out = fused_assemble(A, row_columns, row_values, ghost_sizes, "rarh");
        if (profile) {
            std::fprintf(stderr,
                         "VBCSR_PROFILE_RARH rssGB in=%.2f ghosts=%.2f staged=%.2f assembled=%.2f\n",
                         rss_in, rss_ghosts, rss_numeric, profile_rss_gb());
        }
        return out;
    }

    static Matrix run(const Matrix& A, const Matrix& B, double threshold) {
        return A.graph->size == 1 ? run_local(A, B, threshold)
                                  : run_distributed(A, B, threshold);
    }
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_OPS_SPMM_RARH_HPP

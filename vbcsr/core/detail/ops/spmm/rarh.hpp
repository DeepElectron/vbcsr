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
#include <type_traits>
#include <cstring>
#include <map>
#include <set>
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


    // Global block index -> local row, or -1 where the matrix does not own it.
    static std::vector<int> row_of_global(const Matrix& m, int n_global_blocks) {
        std::vector<int> map(static_cast<size_t>(n_global_blocks), -1);
        const int n_rows = static_cast<int>(m.graph->adj_ptr.size()) - 1;
        for (int i = 0; i < n_rows; ++i) {
            map[static_cast<size_t>(m.graph->get_global_index(i))] = i;
        }
        return map;
    }

    // Rows fetched from other ranks, indexed by global row for O(1) reach in
    // the numeric loop. A map probe per block pair was not affordable: the
    // inner contraction runs over thousands of pairs per row.
    struct RemoteRows {
        std::vector<int> first;   // global row -> first entry; -1 means UNANSWERED
        std::vector<int> count;   // global row -> number of entries
        std::vector<int> cols;    // global column of each entry
        std::vector<int> dims;    // its block column dimension
        std::vector<double> norms;  // Frobenius, from the pattern exchange
        std::vector<const T*> data;

        void build(const SpMMGhostBlocks<T>& ghosts, const GhostMetadata& patterns,
                   int n_global) {
            first.assign(static_cast<size_t>(n_global), -1);
            count.assign(static_cast<size_t>(n_global), 0);
            // Payloads carry only NONEMPTY rows, but the pattern reply names
            // every row whose owner answered -- including rows that exist and
            // hold zero blocks. Those must read as found-with-nothing, not as
            // missing: an empty row's contribution to the contraction is zero,
            // and treating it as absent turned it into a spurious
            // "not Hermitian" throw on a valid operand.
            for (const auto& row : patterns) {
                first[static_cast<size_t>(row.first)] = 0;
            }
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
    };

    // Every block of the rows named in `patterns`, as the payload exchange
    // wants them.
    static std::vector<BlockID> blocks_of(const GhostMetadata& patterns) {
        std::vector<BlockID> blocks;
        for (const auto& row : patterns) {
            for (const auto& meta : row.second) {
                blocks.push_back(BlockID{row.first, meta.col});
            }
        }
        return blocks;
    }

    // The numeric pass, shared by the serial and distributed entry points.
    // `remote_a` / `remote_b` are empty on one rank; on many they carry the
    // rows this rank does not own but must read.
    static void fused_rows(const Matrix& A, const Matrix& B, double threshold,
                           const RemoteRows& remote_a, const RemoteRows& remote_b,
                           const std::vector<int>& a_row, const std::vector<int>& b_row,
                           int n_global,
                           const std::vector<double>& a_norms,
                           const std::vector<double>& b_norms,
                           double a_max_norm,
                           std::vector<std::vector<int>>& row_columns,
                           std::vector<std::vector<T>>& row_values,
                           int& bad_operand) {
        const DistGraph& ga = *A.graph;
        const DistGraph& gb = *B.graph;
        const int n_rows = static_cast<int>(ga.adj_ptr.size()) - 1;
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

            // How many blocks visit_b would hand over for a row. The stage-one
            // gate needs it: see the bound where eps1 is set.
            auto count_b = [&](int g_row) -> int {
                const int local = b_row[static_cast<size_t>(g_row)];
                if (local >= 0) return b_ptr[local + 1] - b_ptr[local];
                if (remote_b.first.empty()) return 0;
                const int at = remote_b.first[static_cast<size_t>(g_row)];
                if (at < 0) return 0;
                return remote_b.count[static_cast<size_t>(g_row)];
            };

            #pragma omp for schedule(dynamic, 8)
            for (int i = 0; i < n_rows; ++i) {
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

                // The one drop, on finished values.
                std::vector<int>& keep_columns = row_columns[i];
                std::vector<T>& keep_values = row_values[i];
                for (size_t sIdx = 0; sIdx < outer.touched.size(); ++sIdx) {
                    const size_t count =
                        static_cast<size_t>(r_dim) * outer.col_dim[sIdx];
                    const T* block = outer.values.data() + outer.value_offset[sIdx];
                    if (threshold > 0.0) {
                        double sq = 0.0;
                        for (size_t e = 0; e < count; ++e) sq += squared_magnitude(block[e]);
                        if (std::sqrt(sq) < threshold) continue;
                    }
                    keep_columns.push_back(outer.touched[sIdx]);
                    keep_values.insert(keep_values.end(), block, block + count);
                }
            }
        }
    }

    // Everything after the numeric pass: order each row, build the graph the
    // surviving pattern defines, and copy the staged values in. Shared, so the
    // serial and distributed paths cannot drift on the part that decides the
    // result's structure.
    static Matrix assemble(const Matrix& A,
                           std::vector<std::vector<int>>& row_columns,
                           std::vector<std::vector<T>>& row_values,
                           const GhostSizes& ghost_sizes) {
        const DistGraph& ga = *A.graph;
        const int n_rows = static_cast<int>(ga.adj_ptr.size()) - 1;

        // Sort each row's columns, carrying the values with them, so the
        // result graph and the staged values agree on order.
        #pragma omp parallel for schedule(dynamic, 8)
        for (int i = 0; i < n_rows; ++i) {
            std::vector<int>& columns = row_columns[i];
            if (columns.size() < 2) continue;
            std::vector<int> order(columns.size());
            for (size_t s = 0; s < order.size(); ++s) order[s] = static_cast<int>(s);
            std::sort(order.begin(), order.end(),
                      [&](int x, int y) { return columns[x] < columns[y]; });
            const int r_dim = ga.block_sizes[i];
            auto dim_of = [&](int g_col) {
                auto it = ga.global_to_local.find(g_col);
                if (it != ga.global_to_local.end()) return ga.block_sizes[it->second];
                return ghost_sizes.at(g_col);
            };
            std::vector<int> sorted_columns(columns.size());
            std::vector<T> sorted_values;
            sorted_values.reserve(row_values[i].size());
            // Offsets in the staged buffer, in the pre-sort order.
            std::vector<size_t> offset(columns.size());
            size_t running = 0;
            for (size_t s = 0; s < columns.size(); ++s) {
                offset[s] = running;
                running += static_cast<size_t>(r_dim) * dim_of(columns[s]);
            }
            for (size_t s = 0; s < order.size(); ++s) {
                const int from = order[s];
                sorted_columns[s] = columns[from];
                const size_t count = static_cast<size_t>(r_dim) * dim_of(columns[from]);
                sorted_values.insert(sorted_values.end(),
                                     row_values[i].begin() + offset[from],
                                     row_values[i].begin() + offset[from] + count);
            }
            columns.swap(sorted_columns);
            row_values[i].swap(sorted_values);
        }

        DistGraph* graph_c = construct_result_graph(A, row_columns, ghost_sizes, "rarh");
        // Deferred first touch, as square_polynomial's assemble already does.
        // The plain `Matrix C(graph_c)` here ran the zero pass over the whole
        // result -- making it fully resident -- at the one moment the staged
        // rows below still hold the entire answer too. Every block is written by
        // the positional copy that follows, so there is nothing for the pass to
        // initialise that the copy does not.
        Matrix C = Matrix::make_overwritten_result(graph_c, A.configured_page_size());

        // The staged pattern IS the result pattern, so the copy is positional.
        //
        // Each row's staging buffer is freed the instant it has been copied.
        // Without that, the whole answer exists twice at this point -- once
        // staged per row, once in C -- which on a Newton-Schulz step is tens of
        // GB held for the length of a memcpy loop and was the largest single
        // item in this kernel's peak. Rows are independent heap allocations, so
        // this needs no chunking and no barrier: the thread that copies a row
        // is the one that releases it, and `swap` with an empty vector is the
        // spelling that actually returns the capacity (clear() keeps it).
        //
        // By THREAD DOMAIN, not a dynamic row schedule: this copy is the
        // deferred first touch, so the thread that runs a row decides which node
        // its pages live on, and the forward apply splits this matrix along
        // thread_domains. See the same note in square_polynomial.hpp.
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
            std::vector<T>().swap(row_values[i]);
        }
        }
        return C;
    }

    // Largest block norm of A, over the whole team. Global rather than per row
    // because stage 1's neglect is later multiplied by an A block belonging to
    // a row this rank may not own, so only a global bound is sound.
    static double global_max_norm(const Matrix& A) {
        const auto& norms = A.get_block_norms();
        double local = 0.0;
        for (double n : norms) local = std::max(local, n);
        if (A.graph->size <= 1) return local;
        double global = local;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, A.graph->comm);
        return global;
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
                   row_of_global(A, n_global), row_of_global(B, n_global),
                   n_global, A.get_block_norms(), B.get_block_norms(),
                   global_max_norm(A), row_columns, row_values, bad_operand);
        if (bad_operand) {
            throw std::runtime_error(
                "rarh_upper: the left operand is not Hermitian "
                "(a column appears that is not also a row)");
        }
        const bool profile = std::getenv("VBCSR_PROFILE_RARH") != nullptr;
        const double rss_staged = profile ? profile_rss_gb() : 0.0;
        GhostSizes no_ghosts;
        Matrix out = assemble(A, row_columns, row_values, no_ghosts);
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

        // ---- Round 1: the rows of B that A's columns reach.
        std::set<int> need_b;
        for (int i = 0; i < n_rows; ++i) {
            for (int k = ga.adj_ptr[i]; k < ga.adj_ptr[i + 1]; ++k) {
                const int g_col = ga.get_global_index(ga.adj_ind[k]);
                if (gb.find_owner(g_col) != rank) need_b.insert(g_col);
            }
        }
        GhostMetadata meta_b = fetch_row_patterns(B, need_b, ga.comm, ga.size, rank);

        // ---- The support of A B, from patterns alone. Only its UNION over
        // local rows is kept: that is all the next round needs, and it costs
        // one flag per global block instead of the pattern of every row.
        std::vector<char> reached(static_cast<size_t>(n_global), 0);
        for (int i = 0; i < n_rows; ++i) {
            for (int k = ga.adj_ptr[i]; k < ga.adj_ptr[i + 1]; ++k) {
                const int g_mid = ga.get_global_index(ga.adj_ind[k]);
                auto local = gb.global_to_local.find(g_mid);
                if (local != gb.global_to_local.end() && local->second < static_cast<int>(gb.adj_ptr.size()) - 1) {
                    const int lb = local->second;
                    for (int e = gb.adj_ptr[lb]; e < gb.adj_ptr[lb + 1]; ++e) {
                        reached[static_cast<size_t>(gb.get_global_index(gb.adj_ind[e]))] = 1;
                    }
                    continue;
                }
                auto pattern = meta_b.find(g_mid);
                if (pattern == meta_b.end()) continue;
                for (const auto& meta : pattern->second) {
                    reached[static_cast<size_t>(meta.col)] = 1;
                }
            }
        }

        // ---- Round 2: the rows of A over that support.
        std::set<int> need_a;
        for (int g = 0; g < n_global; ++g) {
            if (reached[static_cast<size_t>(g)] && ga.find_owner(g) != rank) need_a.insert(g);
        }
        GhostMetadata meta_a = fetch_row_patterns(A, need_a, ga.comm, ga.size, rank);

        // ---- Payloads for both rounds.
        auto ghosts_b = build_spmm_ghost_blocks<T>(
            meta_b, fetch_required_block_payloads(B, blocks_of(meta_b)));
        auto ghosts_a = build_spmm_ghost_blocks<T>(
            meta_a, fetch_required_block_payloads(A, blocks_of(meta_a)));

        RemoteRows remote_b, remote_a;
        remote_b.build(ghosts_b, meta_b, n_global);
        remote_a.build(ghosts_a, meta_a, n_global);

        const double rss_ghosts = profile ? profile_rss_gb() : 0.0;
        std::vector<std::vector<int>> row_columns(n_rows);
        std::vector<std::vector<T>> row_values(n_rows);
        int bad_operand = 0;
        fused_rows(A, B, threshold, remote_a, remote_b,
                   row_of_global(A, n_global), row_of_global(B, n_global),
                   n_global, A.get_block_norms(), B.get_block_norms(),
                   global_max_norm(A), row_columns, row_values, bad_operand);
        int any_bad = bad_operand;
        MPI_Allreduce(&bad_operand, &any_bad, 1, MPI_INT, MPI_MAX, ga.comm);
        if (any_bad) {
            throw std::runtime_error(
                "rarh_upper: the left operand is not Hermitian "
                "(a column appears that is not also a row)");
        }

        // Result columns come from A's rows: those of the fetched rows are in
        // ghosts_a.sizes, those of A's own rows are its graph's ghosts.
        const double rss_numeric = profile ? profile_rss_gb() : 0.0;
        GhostSizes ghost_sizes = ghosts_a.sizes;
        const int n_owned = static_cast<int>(ga.owned_global_indices.size());
        for (size_t g = 0; g < ga.ghost_global_indices.size(); ++g) {
            ghost_sizes[ga.ghost_global_indices[g]] =
                ga.block_sizes[static_cast<size_t>(n_owned) + g];
        }
        Matrix out = assemble(A, row_columns, row_values, ghost_sizes);
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

#ifndef VBCSR_DETAIL_OPS_SPMF_GRAPH_FUNCTION_HPP
#define VBCSR_DETAIL_OPS_SPMF_GRAPH_FUNCTION_HPP

#include <set>
#include "../../../block_csr.hpp"
#include "../../kernels/dense_kernels.hpp"
#include "subspace.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <omp.h>

namespace vbcsr {

namespace detail {} // (no file-scope helpers currently)

// Generic driver of the graph matrix-function approximation: for every owned
// block row of the HERMITIAN matrix A, assemble the dense subgraph matrix of
// its whole neighbourhood, hand it to `action`, and scatter the resulting
// block row into Result.
//
//   action(total_dim, M, k_cols, col_start)
//
// receives M as the total_dim x total_dim dense subgraph matrix and must
// replace it with the total_dim x k_cols column-major block
// f(M)[:, col_start : col_start + k_cols] - dense_matrix_function's contract,
// so a specialised kernel (e.g. the thin inverse-square-root action) can be
// substituted for the generic diagonalization without duplicating the graph
// traversal. The action may consume M's storage as its working buffer, and is
// invoked from inside an OpenMP parallel region: it must be thread-safe and
// must not let exceptions escape; failures are its own to account for (see
// graph_matrix_function / graph_inverse_sqrt).
//
// Hermiticity of A (and of f(A)) is relied on twice, not merely assumed:
//  * the dense view handed to the action is BlockSpMat::to_dense's row-major
//    buffer read as column-major, i.e. conj(M) - harmless exactly because
//    f(conj(M)) = conj(f(M)) for Hermitian M and real-valued f;
//  * only the block COLUMN of f(M) is computed, and the wanted block row is
//    extracted as its (conjugation-cancelling) transpose in the scatter step.
//
// Memory: one subgraph problem runs per OpenMP thread and a frugal action
// holds ~2 dim^2 scalars, so the peak working set is OMP_NUM_THREADS x 2 dim^2
// - the thread budget is also the memory lever, and there is no separate cap.
//
// (No default for `verbose`: BlockSpMat befriends this template, and a friend
// declaration counts as the first declaration.)
template <typename T, typename DenseAction>
void graph_function_apply(
    BlockSpMat<T>& A,
    BlockSpMat<T>* Result,
    DenseAction&& action,
    bool verbose) {

    if (Result == nullptr) {
        throw std::invalid_argument("graph_function_apply requires a valid output matrix pointer");
    }
    // Result must be able to HOLD what this writes, which is more than the
    // old check (same comm, same owned COUNT) established: two graphs can
    // agree on both and still disagree on which rows they own, on block sizes,
    // and on adjacency. That mattered because the scatter below goes through
    // add_block, which on a block the output graph does not have prints a
    // warning to stderr and DROPS it -- a silently incomplete f(A) whose only
    // trace is a line nobody reads. The loop also inserts a diagonal that A's
    // own pattern may omit, so "Result has A's pattern" is not sufficient
    // either; it needs pattern(A) union diagonal.
    if (Result->graph != A.graph) {
        if (Result->graph->comm != A.graph->comm ||
            Result->graph->owned_global_indices != A.graph->owned_global_indices) {
            throw std::runtime_error(
                "graph_function_apply: Result and A must own the same rows in the same "
                "order on the same communicator");
        }
        const int n_check = static_cast<int>(A.graph->owned_global_indices.size());
        for (int idx = 0; idx < n_check; ++idx) {
            const int global_row = A.graph->owned_global_indices[idx];
            std::set<int> have;
            for (int k = Result->row_ptr()[idx]; k < Result->row_ptr()[idx + 1]; ++k) {
                have.insert(Result->graph->get_global_index(Result->col_ind()[k]));
            }
            const auto missing = [&](int col) {
                throw std::runtime_error(
                    "graph_function_apply: Result is missing block (" +
                    std::to_string(global_row) + ", " + std::to_string(col) +
                    "). It must cover pattern(A) union the diagonal, or the result "
                    "is silently incomplete.");
            };
            if (have.count(global_row) == 0) missing(global_row);
            for (int k = A.row_ptr()[idx]; k < A.row_ptr()[idx + 1]; ++k) {
                const int col = A.graph->get_global_index(A.col_ind()[k]);
                if (have.count(col) == 0) missing(col);
            }
        }
    }

    DistGraph* graph = A.graph;
    const int rank = graph->rank;
    MPI_Comm comm = graph->comm;
    const bool collective = (comm != MPI_COMM_NULL && comm != MPI_COMM_SELF);

    Result->fill(T(0));

    const int n_owned = static_cast<int>(graph->owned_global_indices.size());
    int n_owned_max = n_owned;
    if (collective) {
        MPI_Allreduce(&n_owned, &n_owned_max, 1, MPI_INT, MPI_MAX, comm);
    }

    // Largest dense subgraph dimension: row i's problem spans the block sizes
    // of its whole neighbourhood (plus its own diagonal block if the sparsity
    // pattern happens to omit it).
    long long max_sub_dim = 0;
    for (int idx = 0; idx < n_owned; ++idx) {
        const int global_row = graph->owned_global_indices[idx];
        long long dim = 0;
        bool has_diag = false;
        for (int k = A.row_ptr()[idx]; k < A.row_ptr()[idx + 1]; ++k) {
            const int col_lid = A.col_ind()[k];
            dim += graph->block_sizes[col_lid];
            if (graph->get_global_index(col_lid) == global_row) has_diag = true;
        }
        if (!has_diag) dim += graph->block_sizes[idx];
        max_sub_dim = std::max(max_sub_dim, dim);
    }
    if (collective) {
        long long max_sub_dim_global = max_sub_dim;
        MPI_Allreduce(&max_sub_dim, &max_sub_dim_global, 1, MPI_LONG_LONG, MPI_MAX, comm);
        max_sub_dim = max_sub_dim_global;
    }

    // Batch size: one dense subgraph problem per OpenMP thread of this process.
    // The batched payload exchange below is collective, so every rank must
    // agree on the batch size.
    int batch_size = BLASKernel::preferred_parallel_thread_count();
    if (collective) {
        int batch_size_min = batch_size;
        MPI_Allreduce(&batch_size, &batch_size_min, 1, MPI_INT, MPI_MIN, comm);
        batch_size = batch_size_min;
    }

    const int nbatch = (n_owned_max + batch_size - 1) / batch_size;
    if (rank == 0 && verbose) {
        std::cout << "graph_function_apply: " << n_owned_max << " rows (max over ranks), max subgraph dim "
                  << max_sub_dim << ", batch size " << batch_size << ", " << nbatch
                  << " batches" << std::endl;
    }

    // One dense problem per OpenMP thread, so the BLAS/LAPACK calls inside
    // must be single-threaded (thread policy kind C, dense_kernels.hpp).
    BLASKernel::ScopedSerialBLAS serial_blas;

    std::vector<std::vector<int>> batch_indices(batch_size);
    std::vector<std::vector<int>> batch_nb_sizes(batch_size);

    // Phase timers (verbose only): fetch and the parallel region are wall
    // times; build/action/scatter are thread-seconds summed over the team.
    const double t_start = omp_get_wtime();
    double t_fetch = 0.0, t_parallel = 0.0;
    double t_build = 0.0, t_action = 0.0, t_scatter = 0.0;

    for (int b = 0; b < nbatch; ++b) {
        // Neighbourhood of each row in the batch: the sorted global indices of
        // its column blocks (with their block sizes), the diagonal ensured.
        // Rows past n_owned are padding (empty request) so the collective
        // fetch stays aligned.
        for (int i = 0; i < batch_size; ++i) {
            std::vector<int>& neighbors = batch_indices[i];
            std::vector<int>& nb_sizes = batch_nb_sizes[i];
            neighbors.clear();
            nb_sizes.clear();
            const int idx = b * batch_size + i;
            if (idx >= n_owned) continue;

            const int global_row = graph->owned_global_indices[idx];
            std::vector<std::pair<int, int>> cols;  // (gid, block size)
            cols.reserve(A.row_ptr()[idx + 1] - A.row_ptr()[idx] + 1);
            bool has_diag = false;
            for (int k = A.row_ptr()[idx]; k < A.row_ptr()[idx + 1]; ++k) {
                const int col_lid = A.col_ind()[k];
                cols.emplace_back(graph->get_global_index(col_lid), graph->block_sizes[col_lid]);
                if (cols.back().first == global_row) has_diag = true;
            }
            // The subgraph must contain the row's own diagonal block even when
            // A's sparsity pattern stores none: the action anchors the
            // extracted block row at the diagonal position, and f(A) has a
            // nonzero diagonal block regardless of A's pattern.
            if (!has_diag) cols.emplace_back(global_row, graph->block_sizes[idx]);
            std::sort(cols.begin(), cols.end());
            for (const auto& c : cols) {
                neighbors.push_back(c.first);
                nb_sizes.push_back(c.second);
            }
        }

        const double t0 = omp_get_wtime();
        auto batch_blocks = detail::fetch_batched_block_payloads(A, batch_indices);
        t_fetch += omp_get_wtime() - t0;

        // Bounded progress: at most ~16 lines however many batches there
        // are. A large graph reaches hundreds of batches, and one line per
        // batch buries every other message in the log (a 11k-row system
        // printed 175 of these).
        const int report_stride = std::max(1, nbatch / 16);
        if (rank == 0 && verbose && (b % report_stride == 0 || b + 1 == nbatch)) {
            std::cout << "graph_function_apply: batch " << (b + 1) << "/" << nbatch
                      << " (rows " << b * batch_size << ".."
                      << std::min(n_owned_max, (b + 1) * batch_size) - 1 << ")" << std::endl;
        }

        const double t1 = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic) reduction(+ : t_build, t_action, t_scatter)
        for (int i = 0; i < batch_size; ++i) {
            const int idx = b * batch_size + i;
            if (idx >= n_owned) continue;

            const int global_row = graph->owned_global_indices[idx];
            const std::vector<int>& neighbors = batch_indices[i];
            const std::vector<int>& nb_sizes = batch_nb_sizes[i];
            const int block_idx = static_cast<int>(
                std::lower_bound(neighbors.begin(), neighbors.end(), global_row) -
                neighbors.begin());

            const int r_dim = nb_sizes[block_idx];
            int total_dim = 0;
            int row_offset = 0;
            for (std::size_t k = 0; k < neighbors.size(); ++k) total_dim += nb_sizes[k];
            for (int k = 0; k < block_idx; ++k) row_offset += nb_sizes[k];

            std::vector<T> M;
            double tp = omp_get_wtime();
            {
                // The sparse submatrix and its dense view coexist only here.
                // Dropping it before the dense solve matters: at a subgraph
                // dimension of ~3800 its payload is ~0.5 dim^2 scalars, and the
                // batch runs one problem per thread.
                BlockSpMat<T> sub_mat = A.construct_submatrix(neighbors, batch_blocks);
                // M is row-major, size total_dim x total_dim; the action reads
                // it column-major (see the Hermiticity note above).
                M = sub_mat.to_dense();
            }

            t_build += omp_get_wtime() - tp;

            // Only the needed columns [row_offset, row_offset + r_dim) of f(M)
            // are formed.
            tp = omp_get_wtime();
            action(total_dim, M, r_dim, row_offset);
            t_action += omp_get_wtime() - tp;
            tp = omp_get_wtime();

            // M is now f(M)[:, row_offset : row_offset + r_dim], column-major.
            // Scatter the block row: block (global_row, col_gid) of f(A) is,
            // by Hermiticity, the transpose of rows [col_offset, col_offset +
            // c_dim) of that column block - read it out of M directly.
            int col_offset = 0;
            std::vector<T> block_data;
            for (std::size_t k = 0; k < neighbors.size(); ++k) {
                const int col_gid = neighbors[k];
                const int c_dim = nb_sizes[k];

                block_data.resize(static_cast<std::size_t>(r_dim) * c_dim);
                for (int r = 0; r < r_dim; ++r) {
                    for (int c = 0; c < c_dim; ++c) {
                        block_data[r * c_dim + c] =
                            M[static_cast<std::size_t>(r) * total_dim + (col_offset + c)];
                    }
                }
                Result->add_block(global_row, col_gid, block_data.data(), r_dim, c_dim,
                                  AssemblyMode::ADD, kCanonicalBlockLayout);
                col_offset += c_dim;
            }
            t_scatter += omp_get_wtime() - tp;
        }
        t_parallel += omp_get_wtime() - t1;
    }

    const double t2 = omp_get_wtime();
    Result->assemble();

    if (rank == 0 && verbose) {
        const double t_end = omp_get_wtime();
        std::cout << "graph_function_apply: wall " << (t_end - t_start) << "s (fetch " << t_fetch
                  << "s, solve region " << t_parallel << "s, assemble " << (t_end - t2)
                  << "s); thread-seconds: build " << t_build << ", action " << t_action
                  << ", scatter " << t_scatter << std::endl;
    }
}

// f(A) for a general scalar function: every subgraph problem is solved by a
// dense diagonalization (the Lanczos variant was removed after measuring
// consistently worse efficiency than the direct dense diagonalization).
//
// Throws if any subgraph eigensolve failed - agreed collectively after the
// loop, so every rank throws together instead of one rank abandoning the
// collective assembly.
template <typename T>
void graph_matrix_function(
    BlockSpMat<T>& A,
    BlockSpMat<T>* Result,
    std::function<T(double)> func,
    bool verbose = false) {
    std::atomic<long long> failed{0};
    graph_function_apply(
        A, Result,
        [&func, &failed](int total_dim, std::vector<T>& M, int k_cols, int col_start) {
            if (!dense_matrix_function(total_dim, M, func, k_cols, col_start)) {
                failed.fetch_add(1, std::memory_order_relaxed);
            }
        },
        verbose);

    long long total_failed = failed.load();
    MPI_Comm comm = A.graph->comm;
    if (comm != MPI_COMM_NULL && comm != MPI_COMM_SELF) {
        long long local = total_failed;
        MPI_Allreduce(&local, &total_failed, 1, MPI_LONG_LONG, MPI_SUM, comm);
    }
    if (total_failed > 0) {
        throw std::runtime_error("graph_matrix_function: " + std::to_string(total_failed) +
                                 " subgraph eigendecompositions failed");
    }
}

}

#endif // VBCSR_DETAIL_OPS_SPMF_GRAPH_FUNCTION_HPP

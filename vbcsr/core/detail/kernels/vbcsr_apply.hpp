#ifndef VBCSR_DETAIL_KERNELS_VBCSR_APPLY_HPP
#define VBCSR_DETAIL_KERNELS_VBCSR_APPLY_HPP

#include "../../dist_graph.hpp"
#include "../../dist_multivector.hpp"
#include "../../dist_vector.hpp"
#include "../backend/vbcsr_backend.hpp"
#include "dense_kernels.hpp"
#include "rowmajor_kernels.hpp"

#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vbcsr::detail {
namespace vbcsr_apply_detail {

template <typename T>
struct ShapeBatchKernel {
    using Backend = VBCSRMatrixBackend<T>;

    static void mult(DistGraph* graph, const Backend& backend, DistVector<T>& x, DistVector<T>& y) {
        BLASKernel::configure_native_threading();
        x.bind_to_graph(graph);
        y.bind_to_graph(graph);
        x.sync_ghosts();
        std::fill(y.data.begin(), y.data.end(), T(0));
        run_mult_row_direct(graph, backend, x, y);
    }

    // Dense apply runs natively on the row-major multivector via the
    // vec-axis kernel family (rowmajor_kernels.hpp). nv is the padded ld:
    // pad lanes are zero on input and compute to exact zeros on output,
    // so every SIMD chunk stays full and the padding invariant holds.
    static void mult_dense(DistGraph* graph, const Backend& backend, DistMultiVector<T>& X, DistMultiVector<T>& Y) {
        BLASKernel::configure_native_threading();
        X.bind_to_graph(graph);
        Y.bind_to_graph(graph);
        X.sync_ghosts();
        std::fill(Y.data.begin(), Y.data.end(), T(0));
        run_mult_dense_row_direct(graph, backend, X, Y);
    }

    static void mult_adjoint(DistGraph* graph, const Backend& backend, DistVector<T>& x, DistVector<T>& y) {
        BLASKernel::configure_native_threading();
        x.bind_to_graph(graph);
        y.bind_to_graph(graph);
        std::fill(y.data.begin(), y.data.end(), T(0));
        run_mult_adjoint_col_direct(graph, backend, x, y);
        y.reduce_ghosts();
    }

    static void mult_dense_adjoint(DistGraph* graph, const Backend& backend, DistMultiVector<T>& X, DistMultiVector<T>& Y) {
        BLASKernel::configure_native_threading();
        X.bind_to_graph(graph);
        Y.bind_to_graph(graph);
        std::fill(Y.data.begin(), Y.data.end(), T(0));
        run_mult_dense_adjoint_col_direct(graph, backend, X, Y);
        Y.reduce_ghosts();
    }

private:
    // Still used by the apply-plan builders (packed-output marking is ignored
    // by the row-major dense paths, which accumulate into Y rows directly).
    static constexpr int kDirectDenseRowDegreeLimit = 16;

    static void run_mult_row(
        DistGraph* graph,
        const Backend& backend,
        DistVector<T>& x,
        DistVector<T>& y,
        int row,
        int block_begin,
        int block_end) {
        const auto& col_ind = graph->adj_ind;
        const auto& block_offsets = graph->block_offsets;
        const auto& block_sizes = graph->block_sizes;
        const int row_dim = block_sizes[row];
        T* y_ptr = y.data.data() + block_offsets[row];
        for (int slot = block_begin; slot < block_end; ++slot) {
            const int col = col_ind[slot];
            rowmajor_kernels::rm_gemv<T>(
                row_dim,
                block_sizes[col],
                backend.block_ptr_for_graph_block(slot),
                x.data.data() + block_offsets[col],
                y_ptr);
        }
    }

    static size_t forward_task_work_units(const typename Backend::ForwardRowTask& task) {
        return std::max<size_t>(size_t(1), task.work);
    }

    static std::vector<int> build_forward_work_chunks(const typename Backend::ForwardApplyPlan& plan) {
        const int row_count = static_cast<int>(plan.rows.size());
        if (row_count == 0) {
            return std::vector<int>{0};
        }

        const int chunk_count = std::max(1, std::min(row_count, max_parallel_threads()));
        std::vector<int> chunks(static_cast<size_t>(chunk_count) + 1, 0);
        const size_t total_units = std::max<size_t>(size_t(1), plan.total_work);
        const size_t target_units =
            (total_units + static_cast<size_t>(chunk_count) - 1) /
            static_cast<size_t>(chunk_count);

        int row = 0;
        for (int chunk = 0; chunk < chunk_count - 1; ++chunk) {
            const int remaining_chunks = chunk_count - chunk - 1;
            size_t chunk_units = 0;
            while (row < row_count - remaining_chunks &&
                   (chunk_units < target_units || row == chunks[static_cast<size_t>(chunk)])) {
                chunk_units += forward_task_work_units(plan.rows[static_cast<size_t>(row)]);
                ++row;
            }
            chunks[static_cast<size_t>(chunk) + 1] = row;
        }
        chunks[static_cast<size_t>(chunk_count)] = row_count;
        return chunks;
    }

    static size_t adjoint_task_work_units(const typename Backend::AdjointColumnTask& task) {
        return std::max<size_t>(size_t(1), task.work);
    }

    static std::vector<int> build_adjoint_work_chunks(const typename Backend::AdjointApplyPlan& plan) {
        const int column_count = static_cast<int>(plan.columns.size());
        if (column_count == 0) {
            return std::vector<int>{0};
        }

        const int chunk_count = std::max(1, std::min(column_count, max_parallel_threads()));
        std::vector<int> chunks(static_cast<size_t>(chunk_count) + 1, 0);
        const size_t total_units = std::max<size_t>(size_t(1), plan.total_work);
        const size_t target_units =
            (total_units + static_cast<size_t>(chunk_count) - 1) /
            static_cast<size_t>(chunk_count);

        int column = 0;
        for (int chunk = 0; chunk < chunk_count - 1; ++chunk) {
            const int remaining_chunks = chunk_count - chunk - 1;
            size_t chunk_units = 0;
            while (column < column_count - remaining_chunks &&
                   (chunk_units < target_units || column == chunks[static_cast<size_t>(chunk)])) {
                chunk_units += adjoint_task_work_units(plan.columns[static_cast<size_t>(column)]);
                ++column;
            }
            chunks[static_cast<size_t>(chunk) + 1] = column;
        }
        chunks[static_cast<size_t>(chunk_count)] = column_count;
        return chunks;
    }

    static void run_mult_row_direct(
        DistGraph* graph,
        const Backend& backend,
        DistVector<T>& x,
        DistVector<T>& y) {
        const int n_rows = static_cast<int>(graph->owned_global_indices.size());
        const auto& plan = backend.ensure_forward_apply_plan(
            graph->adj_ptr,
            graph->adj_ind,
            graph->block_sizes,
            n_rows,
            kDirectDenseRowDegreeLimit);

        const auto chunks = build_forward_work_chunks(plan);
        const int chunk_count = static_cast<int>(chunks.size()) - 1;

        #pragma omp parallel for schedule(static)
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            for (int task_index = chunks[static_cast<size_t>(chunk)];
                 task_index < chunks[static_cast<size_t>(chunk) + 1];
                 ++task_index) {
                const auto& task = plan.rows[static_cast<size_t>(task_index)];
                run_mult_row(
                    graph,
                    backend,
                    x,
                    y,
                    task.row,
                    task.block_begin,
                    task.block_end);
            }
        }
    }

    static void run_mult_dense_row_direct(
        DistGraph* graph,
        const Backend& backend,
        DistMultiVector<T>& X,
        DistMultiVector<T>& Y) {
        const int n_rows = static_cast<int>(graph->owned_global_indices.size());
        const int nv = X.ld;  // padded; pad lanes are zero (invariant)
        const int x_ld = X.ld;
        const int y_ld = Y.ld;
        const auto& plan = backend.ensure_forward_apply_plan(
            graph->adj_ptr,
            graph->adj_ind,
            graph->block_sizes,
            n_rows,
            kDirectDenseRowDegreeLimit);

        const auto chunks = build_forward_work_chunks(plan);
        const int chunk_count = static_cast<int>(chunks.size()) - 1;
        const auto& col_ind = graph->adj_ind;
        const auto& block_offsets = graph->block_offsets;
        const auto& block_sizes = graph->block_sizes;
        const T* x_data = X.data.data();
        T* y_data = Y.data.data();

        #pragma omp parallel for schedule(static)
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            for (int task_index = chunks[static_cast<size_t>(chunk)];
                 task_index < chunks[static_cast<size_t>(chunk) + 1];
                 ++task_index) {
                const auto& task = plan.rows[static_cast<size_t>(task_index)];
                const int row = task.row;
                const int row_dim = task.row_dim;
                T* y_rows = y_data + static_cast<size_t>(block_offsets[row]) * y_ld;
                for (int slot = task.block_begin; slot < task.block_end; ++slot) {
                    const int col = col_ind[slot];
                    rowmajor_kernels::rm_gemm<T>(
                        row_dim,
                        block_sizes[col],
                        nv,
                        backend.block_ptr_for_graph_block(slot),
                        x_data + static_cast<size_t>(block_offsets[col]) * x_ld,
                        x_ld,
                        y_rows,
                        y_ld);
                }
            }
        }
    }

    static void run_mult_adjoint_col_direct(
        DistGraph* graph,
        const Backend& backend,
        DistVector<T>& x,
        DistVector<T>& y) {
        const int n_rows = static_cast<int>(graph->owned_global_indices.size());
        const auto& plan = backend.ensure_adjoint_apply_plan(
            graph->adj_ptr,
            graph->adj_ind,
            graph->block_sizes,
            n_rows,
            kDirectDenseRowDegreeLimit);
        const auto chunks = build_adjoint_work_chunks(plan);
        const int chunk_count = static_cast<int>(chunks.size()) - 1;
        const auto& block_offsets = graph->block_offsets;
        const auto& block_sizes = graph->block_sizes;

        #pragma omp parallel for schedule(static)
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            for (int task_index = chunks[static_cast<size_t>(chunk)];
                 task_index < chunks[static_cast<size_t>(chunk) + 1];
                 ++task_index) {
                const auto& task = plan.columns[static_cast<size_t>(task_index)];
                T* y_ptr = y.data.data() + block_offsets[task.col];
                for (int incoming = task.incoming_begin; incoming < task.incoming_end; ++incoming) {
                    const int slot = plan.incoming_slots[static_cast<size_t>(incoming)];
                    const int row = plan.incoming_rows[static_cast<size_t>(incoming)];
                    rowmajor_kernels::rm_gemv_adjoint<T>(
                        block_sizes[row],
                        task.col_dim,
                        backend.block_ptr_for_graph_block(slot),
                        x.data.data() + block_offsets[row],
                        y_ptr);
                }
            }
        }
    }

    static void run_mult_dense_adjoint_col_direct(
        DistGraph* graph,
        const Backend& backend,
        DistMultiVector<T>& X,
        DistMultiVector<T>& Y) {
        const int n_rows = static_cast<int>(graph->owned_global_indices.size());
        const int nv = X.ld;  // padded; pad lanes are zero (invariant)
        const int x_ld = X.ld;
        const int y_ld = Y.ld;
        const auto& plan = backend.ensure_adjoint_apply_plan(
            graph->adj_ptr,
            graph->adj_ind,
            graph->block_sizes,
            n_rows,
            kDirectDenseRowDegreeLimit);
        const auto chunks = build_adjoint_work_chunks(plan);
        const int chunk_count = static_cast<int>(chunks.size()) - 1;
        const auto& block_offsets = graph->block_offsets;
        const auto& block_sizes = graph->block_sizes;
        const T* x_data = X.data.data();
        T* y_data = Y.data.data();

        #pragma omp parallel for schedule(static)
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            for (int task_index = chunks[static_cast<size_t>(chunk)];
                 task_index < chunks[static_cast<size_t>(chunk) + 1];
                 ++task_index) {
                const auto& task = plan.columns[static_cast<size_t>(task_index)];
                T* y_rows = y_data + static_cast<size_t>(block_offsets[task.col]) * y_ld;
                for (int incoming = task.incoming_begin; incoming < task.incoming_end; ++incoming) {
                    const int slot = plan.incoming_slots[static_cast<size_t>(incoming)];
                    const int row = plan.incoming_rows[static_cast<size_t>(incoming)];
                    rowmajor_kernels::rm_gemm_adjoint<T>(
                        block_sizes[row],
                        task.col_dim,
                        nv,
                        backend.block_ptr_for_graph_block(slot),
                        x_data + static_cast<size_t>(block_offsets[row]) * x_ld,
                        x_ld,
                        y_rows,
                        y_ld);
                }
            }
        }
    }

    // The page-batch apply machinery (run_mult_*_batch*, build_apply_tasks,
    // apply-mode selection, pack/accumulate helpers) was dead code with no
    // callers and was removed during the row-major migration (Phase 2);
    // the live paths are the *_direct functions above.

    static int max_parallel_threads() {
        #ifdef _OPENMP
        return omp_get_max_threads();
        #else
        return 1;
        #endif
    }
};

} // namespace vbcsr_apply_detail

template <typename T>
void vbcsr_mult(
    DistGraph* graph,
    const VBCSRMatrixBackend<T>& backend,
    DistVector<T>& x,
    DistVector<T>& y) {
    vbcsr_apply_detail::ShapeBatchKernel<T>::mult(graph, backend, x, y);
}

template <typename T>
void vbcsr_mult_dense(
    DistGraph* graph,
    const VBCSRMatrixBackend<T>& backend,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y) {
    vbcsr_apply_detail::ShapeBatchKernel<T>::mult_dense(graph, backend, x, y);
}

template <typename T>
void vbcsr_mult_adjoint(
    DistGraph* graph,
    const VBCSRMatrixBackend<T>& backend,
    DistVector<T>& x,
    DistVector<T>& y) {
    vbcsr_apply_detail::ShapeBatchKernel<T>::mult_adjoint(graph, backend, x, y);
}

template <typename T>
void vbcsr_mult_dense_adjoint(
    DistGraph* graph,
    const VBCSRMatrixBackend<T>& backend,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y) {
    vbcsr_apply_detail::ShapeBatchKernel<T>::mult_dense_adjoint(graph, backend, x, y);
}

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_KERNELS_VBCSR_APPLY_HPP

#ifndef VBCSR_DETAIL_VBCSR_KERNELS_HPP
#define VBCSR_DETAIL_VBCSR_KERNELS_HPP

#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vbcsr::detail {

template <typename Matrix>
struct VBCSRShapeBatchExecutor {
    using T = typename Matrix::value_type;
    using ShapeBatch = typename Matrix::VBCSRBackendStorage::ShapeBatchView;

    static void mult(const Matrix& matrix, DistVector<T>& x, DistVector<T>& y) {
        using ExecutionKind = typename Matrix::VBCSRBackendStorage::ExecutionKind;
        x.bind_to_graph(matrix.graph);
        y.bind_to_graph(matrix.graph);
        x.sync_ghosts();
        std::fill(y.data.begin(), y.data.end(), T(0));

        const auto batches = collect_batches(matrix);
        auto thread_acc = make_thread_accumulators(static_cast<int>(y.data.size()));

        #pragma omp parallel for schedule(dynamic)
        for (int batch_idx = 0; batch_idx < static_cast<int>(batches.size()); ++batch_idx) {
            const int tid = current_thread_id();
            auto& accum = thread_acc[tid];
            const auto& batch = batches[batch_idx];
            matrix.active_vbcsr_backend().record_apply_batch(batch.shape_id, batch.block_count());
            switch (batch_execution_kind(batch)) {
                case ExecutionKind::StaticFallback:
                case ExecutionKind::BatchedFallback:
                case ExecutionKind::JIT:
                    run_mult_batch(matrix, batch, x, accum.data());
                    break;
            }
        }

        reduce_thread_accumulators(thread_acc, y.data);
    }

    static void mult_dense(const Matrix& matrix, DistMultiVector<T>& X, DistMultiVector<T>& Y) {
        using ExecutionKind = typename Matrix::VBCSRBackendStorage::ExecutionKind;
        X.bind_to_graph(matrix.graph);
        Y.bind_to_graph(matrix.graph);
        X.sync_ghosts();
        std::fill(Y.data.begin(), Y.data.end(), T(0));

        const auto batches = collect_batches(matrix);
        auto thread_acc = make_thread_accumulators(static_cast<int>(Y.data.size()));
        const int ldb = X.local_rows + X.ghost_rows;
        const int ldc = Y.local_rows + Y.ghost_rows;
        const int num_vecs = X.num_vectors;

        #pragma omp parallel for schedule(dynamic)
        for (int batch_idx = 0; batch_idx < static_cast<int>(batches.size()); ++batch_idx) {
            const int tid = current_thread_id();
            auto& accum = thread_acc[tid];
            const auto& batch = batches[batch_idx];
            matrix.active_vbcsr_backend().record_apply_batch(batch.shape_id, batch.block_count());
            switch (batch_execution_kind(batch)) {
                case ExecutionKind::StaticFallback:
                case ExecutionKind::BatchedFallback:
                case ExecutionKind::JIT:
                    run_mult_dense_batch(matrix, batch, X, ldb, num_vecs, ldc, accum.data());
                    break;
            }
        }

        reduce_thread_accumulators(thread_acc, Y.data);
    }

    static void mult_adjoint(const Matrix& matrix, DistVector<T>& x, DistVector<T>& y) {
        using ExecutionKind = typename Matrix::VBCSRBackendStorage::ExecutionKind;
        x.bind_to_graph(matrix.graph);
        y.bind_to_graph(matrix.graph);
        std::fill(y.data.begin(), y.data.end(), T(0));

        const auto batches = collect_batches(matrix);
        auto thread_acc = make_thread_accumulators(static_cast<int>(y.data.size()));

        #pragma omp parallel for schedule(dynamic)
        for (int batch_idx = 0; batch_idx < static_cast<int>(batches.size()); ++batch_idx) {
            const int tid = current_thread_id();
            auto& accum = thread_acc[tid];
            const auto& batch = batches[batch_idx];
            matrix.active_vbcsr_backend().record_apply_batch(batch.shape_id, batch.block_count());
            switch (batch_execution_kind(batch)) {
                case ExecutionKind::StaticFallback:
                case ExecutionKind::BatchedFallback:
                case ExecutionKind::JIT:
                    run_mult_adjoint_batch(matrix, batch, x, accum.data());
                    break;
            }
        }

        reduce_thread_accumulators(thread_acc, y.data);
        y.reduce_ghosts();
    }

    static void mult_dense_adjoint(const Matrix& matrix, DistMultiVector<T>& X, DistMultiVector<T>& Y) {
        using ExecutionKind = typename Matrix::VBCSRBackendStorage::ExecutionKind;
        X.bind_to_graph(matrix.graph);
        Y.bind_to_graph(matrix.graph);
        std::fill(Y.data.begin(), Y.data.end(), T(0));

        const auto batches = collect_batches(matrix);
        auto thread_acc = make_thread_accumulators(static_cast<int>(Y.data.size()));
        const int ldb = X.local_rows + X.ghost_rows;
        const int ldc = Y.local_rows + Y.ghost_rows;
        const int num_vecs = X.num_vectors;

        #pragma omp parallel for schedule(dynamic)
        for (int batch_idx = 0; batch_idx < static_cast<int>(batches.size()); ++batch_idx) {
            const int tid = current_thread_id();
            auto& accum = thread_acc[tid];
            const auto& batch = batches[batch_idx];
            matrix.active_vbcsr_backend().record_apply_batch(batch.shape_id, batch.block_count());
            switch (batch_execution_kind(batch)) {
                case ExecutionKind::StaticFallback:
                case ExecutionKind::BatchedFallback:
                case ExecutionKind::JIT:
                    run_mult_dense_adjoint_batch(matrix, batch, X, ldb, num_vecs, ldc, accum.data());
                    break;
            }
        }

        reduce_thread_accumulators(thread_acc, Y.data);
        Y.reduce_ghosts();
    }

private:
    static void run_mult_batch(const Matrix& matrix, const ShapeBatch& batch, DistVector<T>& x, T* accum) {
        for (uint32_t idx = 0; idx < batch.block_count(); ++idx) {
            const int slot = batch.logical_slot(idx);
            const int row = matrix.block_row_from_slot(slot);
            const int col = matrix.block_col_from_slot(slot);
            const T* block = batch.block_ptr(idx);
            const T* x_ptr = x.data.data() + matrix.graph->block_offsets[col];
            T* y_ptr = accum + matrix.graph->block_offsets[row];
            SmartKernel<T>::gemv(
                batch.row_dim,
                batch.col_dim,
                T(1),
                block,
                batch.row_dim,
                x_ptr,
                1,
                T(1),
                y_ptr,
                1);
        }
    }

    static void run_mult_dense_batch(
        const Matrix& matrix,
        const ShapeBatch& batch,
        DistMultiVector<T>& X,
        int ldb,
        int num_vecs,
        int ldc,
        T* accum) {
        for (uint32_t idx = 0; idx < batch.block_count(); ++idx) {
            const int slot = batch.logical_slot(idx);
            const int row = matrix.block_row_from_slot(slot);
            const int col = matrix.block_col_from_slot(slot);
            const T* block = batch.block_ptr(idx);
            const T* x_ptr = &X(matrix.graph->block_offsets[col], 0);
            T* y_ptr = accum + matrix.graph->block_offsets[row];
            SmartKernel<T>::gemm(
                batch.row_dim,
                num_vecs,
                batch.col_dim,
                T(1),
                block,
                batch.row_dim,
                x_ptr,
                ldb,
                T(1),
                y_ptr,
                ldc);
        }
    }

    static void run_mult_adjoint_batch(const Matrix& matrix, const ShapeBatch& batch, DistVector<T>& x, T* accum) {
        for (uint32_t idx = 0; idx < batch.block_count(); ++idx) {
            const int slot = batch.logical_slot(idx);
            const int row = matrix.block_row_from_slot(slot);
            const int col = matrix.block_col_from_slot(slot);
            const T* block = batch.block_ptr(idx);
            const T* x_ptr = x.local_data() + matrix.graph->block_offsets[row];
            T* y_ptr = accum + matrix.graph->block_offsets[col];
            SmartKernel<T>::gemv_trans(
                batch.row_dim,
                batch.col_dim,
                T(1),
                block,
                batch.row_dim,
                x_ptr,
                1,
                T(1),
                y_ptr,
                1);
        }
    }

    static void run_mult_dense_adjoint_batch(
        const Matrix& matrix,
        const ShapeBatch& batch,
        DistMultiVector<T>& X,
        int ldb,
        int num_vecs,
        int ldc,
        T* accum) {
        for (uint32_t idx = 0; idx < batch.block_count(); ++idx) {
            const int slot = batch.logical_slot(idx);
            const int row = matrix.block_row_from_slot(slot);
            const int col = matrix.block_col_from_slot(slot);
            const T* block = batch.block_ptr(idx);
            const T* x_ptr = &X(matrix.graph->block_offsets[row], 0);
            T* y_ptr = accum + matrix.graph->block_offsets[col];
            SmartKernel<T>::gemm_trans(
                batch.row_dim,
                num_vecs,
                batch.col_dim,
                T(1),
                block,
                batch.row_dim,
                x_ptr,
                ldb,
                T(1),
                y_ptr,
                ldc);
        }
    }

    static std::vector<ShapeBatch> collect_batches(const Matrix& matrix) {
        std::vector<ShapeBatch> batches;
        matrix.for_each_shape_batch([&](const ShapeBatch& batch) {
            batches.push_back(batch);
        });
        return batches;
    }

    static typename Matrix::VBCSRBackendStorage::ExecutionKind batch_execution_kind(const ShapeBatch& batch) {
        using ExecutionKind = typename Matrix::VBCSRBackendStorage::ExecutionKind;
        if (batch.policy == nullptr) {
            return ExecutionKind::StaticFallback;
        }
        return batch.policy->preferred_execution.load(std::memory_order_relaxed);
    }

    static std::vector<std::vector<T>> make_thread_accumulators(int element_count) {
        int max_threads = 1;
        #ifdef _OPENMP
        max_threads = omp_get_max_threads();
        #endif
        return std::vector<std::vector<T>>(max_threads, std::vector<T>(element_count, T(0)));
    }

    static void reduce_thread_accumulators(const std::vector<std::vector<T>>& thread_acc, std::vector<T>& out) {
        for (const auto& local : thread_acc) {
            for (size_t idx = 0; idx < out.size(); ++idx) {
                out[idx] += local[idx];
            }
        }
    }

    static int current_thread_id() {
        #ifdef _OPENMP
        return omp_get_thread_num();
        #else
        return 0;
        #endif
    }
};

template <typename Matrix>
void vbcsr_mult(const Matrix& matrix, DistVector<typename Matrix::value_type>& x, DistVector<typename Matrix::value_type>& y) {
    VBCSRShapeBatchExecutor<Matrix>::mult(matrix, x, y);
}

template <typename Matrix>
void vbcsr_mult_dense(const Matrix& matrix, DistMultiVector<typename Matrix::value_type>& X, DistMultiVector<typename Matrix::value_type>& Y) {
    VBCSRShapeBatchExecutor<Matrix>::mult_dense(matrix, X, Y);
}

template <typename Matrix>
void vbcsr_mult_adjoint(const Matrix& matrix, DistVector<typename Matrix::value_type>& x, DistVector<typename Matrix::value_type>& y) {
    VBCSRShapeBatchExecutor<Matrix>::mult_adjoint(matrix, x, y);
}

template <typename Matrix>
void vbcsr_mult_dense_adjoint(const Matrix& matrix, DistMultiVector<typename Matrix::value_type>& X, DistMultiVector<typename Matrix::value_type>& Y) {
    VBCSRShapeBatchExecutor<Matrix>::mult_dense_adjoint(matrix, X, Y);
}

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_VBCSR_KERNELS_HPP

#ifndef VBCSR_DETAIL_KERNELS_VBCSR_APPLY_HPP
#define VBCSR_DETAIL_KERNELS_VBCSR_APPLY_HPP

#include "../../dist_graph.hpp"
#include "../../dist_multivector.hpp"
#include "../../dist_vector.hpp"
#include "../backend/matrix_backend.hpp"
#include "dense_kernels.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vbcsr::detail {
namespace vbcsr_apply_detail {

template <typename T, typename Kernel>
struct ShapeBatchKernel {
    using Backend = VBCSRMatrixBackend<T, Kernel>;
    using ApplyMode = typename Backend::ApplyMode;
    using PageBatch = typename Backend::PageBatch;

    // VBCSR apply works in the storage order: blocks are grouped first by
    // shape, then by storage page. A task therefore records both the page batch
    // and the subrange within that batch so large same-shape pages can be split
    // without losing access to compact batched GEMV/GEMM operands.
    //
    // Accumulation uses per-thread output buffers because different tasks may
    // write the same destination row, and adjoint apply writes through columns.
    // The final reduction is cheaper and less fragile than fine-grained locks.
    struct ApplyTask {
        int batch_index = -1;
        uint32_t begin = 0;
        uint32_t count = 0;
    };

    static void mult(DistGraph* graph, const Backend& backend, DistVector<T>& x, DistVector<T>& y) {
        BLASKernel::configure_native_threading();
        x.bind_to_graph(graph);
        y.bind_to_graph(graph);
        x.sync_ghosts();
        std::fill(y.data.begin(), y.data.end(), T(0));

        const auto& batches = backend
                                  .ensure_apply_plan(graph->adj_ptr, graph->adj_ind)
                                  .batches;
        const auto tasks = build_apply_tasks(
            batches,
            SmartKernel<T>::supports_batched_gemv(),
            [](const PageBatch& batch) {
                return static_cast<size_t>(batch.row_dim + batch.col_dim);
            });
        
        auto thread_acc = make_thread_accumulators(static_cast<int>(y.data.size()));

        #pragma omp parallel for schedule(dynamic)
        for (int task_idx = 0; task_idx < static_cast<int>(tasks.size()); ++task_idx) {
            const int tid = current_thread_id();
            auto& accum = thread_acc[tid];
            const auto& task = tasks[task_idx];
            const auto& batch = batches[task.batch_index];
            const ApplyMode mode = select_vector_apply_mode(task.count);
            backend.record_apply_batch(batch.shape_id, mode);
            run_mult_batch(graph, batch, task.begin, task.count, mode, x, accum.data());
        }
        
        reduce_thread_accumulators(thread_acc, y.data);
    }

    static void mult_dense(DistGraph* graph, const Backend& backend, DistMultiVector<T>& X, DistMultiVector<T>& Y) {
        BLASKernel::configure_native_threading();
        X.bind_to_graph(graph);
        Y.bind_to_graph(graph);
        X.sync_ghosts();
        std::fill(Y.data.begin(), Y.data.end(), T(0));

        const auto& batches = backend
                                  .ensure_apply_plan(graph->adj_ptr, graph->adj_ind)
                                  .batches;
        auto thread_acc = make_thread_accumulators(static_cast<int>(Y.data.size()));
        const int ldb = X.local_rows + X.ghost_rows;
        const int ldc = Y.local_rows + Y.ghost_rows;
        const int num_vecs = X.num_vectors;
        const auto tasks = build_apply_tasks(
            batches,
            SmartKernel<T>::supports_batched_gemm(),
            [num_vecs](const PageBatch& batch) {
                return static_cast<size_t>(num_vecs) *
                       static_cast<size_t>(batch.row_dim + batch.col_dim);
            });

        #pragma omp parallel for schedule(dynamic)
        for (int task_idx = 0; task_idx < static_cast<int>(tasks.size()); ++task_idx) {
            const int tid = current_thread_id();
            auto& accum = thread_acc[tid];
            const auto& task = tasks[task_idx];
            const auto& batch = batches[task.batch_index];
            const ApplyMode mode = select_dense_apply_mode(task.count);
            backend.record_apply_batch(batch.shape_id, mode);
            run_mult_dense_batch(graph, batch, task.begin, task.count, mode, X, ldb, num_vecs, ldc, accum.data());
        }

        reduce_thread_accumulators(thread_acc, Y.data);
    }

    static void mult_adjoint(DistGraph* graph, const Backend& backend, DistVector<T>& x, DistVector<T>& y) {
        BLASKernel::configure_native_threading();
        x.bind_to_graph(graph);
        y.bind_to_graph(graph);
        std::fill(y.data.begin(), y.data.end(), T(0));

        const auto& batches = backend
                                  .ensure_apply_plan(graph->adj_ptr, graph->adj_ind)
                                  .batches;
        const auto tasks = build_apply_tasks(
            batches,
            SmartKernel<T>::supports_batched_gemv(),
            [](const PageBatch& batch) {
                return static_cast<size_t>(batch.row_dim + batch.col_dim);
            });
        auto thread_acc = make_thread_accumulators(static_cast<int>(y.data.size()));

        #pragma omp parallel for schedule(dynamic)
        for (int task_idx = 0; task_idx < static_cast<int>(tasks.size()); ++task_idx) {
            const int tid = current_thread_id();
            auto& accum = thread_acc[tid];
            const auto& task = tasks[task_idx];
            const auto& batch = batches[task.batch_index];
            const ApplyMode mode = select_vector_apply_mode(task.count);
            backend.record_apply_batch(batch.shape_id, mode);
            run_mult_adjoint_batch(graph, batch, task.begin, task.count, mode, x, accum.data());
        }

        reduce_thread_accumulators(thread_acc, y.data);
        y.reduce_ghosts();
    }

    static void mult_dense_adjoint(DistGraph* graph, const Backend& backend, DistMultiVector<T>& X, DistMultiVector<T>& Y) {
        BLASKernel::configure_native_threading();
        X.bind_to_graph(graph);
        Y.bind_to_graph(graph);
        std::fill(Y.data.begin(), Y.data.end(), T(0));

        const auto& batches = backend
                                  .ensure_apply_plan(graph->adj_ptr, graph->adj_ind)
                                  .batches;
        auto thread_acc = make_thread_accumulators(static_cast<int>(Y.data.size()));
        const int ldb = X.local_rows + X.ghost_rows;
        const int ldc = Y.local_rows + Y.ghost_rows;
        const int num_vecs = X.num_vectors;
        const auto tasks = build_apply_tasks(
            batches,
            SmartKernel<T>::supports_batched_gemm(),
            [num_vecs](const PageBatch& batch) {
                return static_cast<size_t>(num_vecs) *
                       static_cast<size_t>(batch.row_dim + batch.col_dim);
            });

        #pragma omp parallel for schedule(dynamic)
        for (int task_idx = 0; task_idx < static_cast<int>(tasks.size()); ++task_idx) {
            const int tid = current_thread_id();
            auto& accum = thread_acc[tid];
            const auto& task = tasks[task_idx];
            const auto& batch = batches[task.batch_index];
            const ApplyMode mode = select_dense_apply_mode(task.count);
            backend.record_apply_batch(batch.shape_id, mode);
            run_mult_dense_adjoint_batch(graph, batch, task.begin, task.count, mode, X, ldb, num_vecs, ldc, accum.data());
        }

        reduce_thread_accumulators(thread_acc, Y.data);
        Y.reduce_ghosts();
    }

private:
    // Target scratch budget for one batched micro-kernel launch. This currently
    // controls both temporary buffer size and the granularity of ApplyTask
    // subdivision when a large page batch is split.
    static constexpr size_t kTargetScratchBytes = 1u << 20;

    // this is a serial kernel that runs for each task
    static void run_mult_batch(
        DistGraph* graph,
        const PageBatch& batch,
        uint32_t begin,
        uint32_t count,
        ApplyMode mode,
        DistVector<T>& x,
        T* accum) {
        if (count == 0) {
            return;
        }
        if (mode == ApplyMode::Batched) {
            run_mult_batch_batched(graph, batch, begin, count, x, accum);
            return;
        }
        run_mult_batch_fallback(graph, batch, begin, count, x, accum);
    }

    static void run_mult_batch_fallback(
        DistGraph* graph,
        const PageBatch& batch,
        uint32_t begin,
        uint32_t count,
        DistVector<T>& x,
        T* accum) {
        for (uint32_t offset = 0; offset < count; ++offset) {
            const uint32_t idx = begin + offset;
            const int row = batch.row_block_index(idx);
            const int col = batch.col_block_index(idx);
            const T* block = batch.block_ptr(idx);
            const T* x_ptr = x.data.data() + graph->block_offsets[col];
            T* y_ptr = accum + graph->block_offsets[row];
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
        DistGraph* graph,
        const PageBatch& batch,
        uint32_t begin,
        uint32_t count,
        ApplyMode mode,
        DistMultiVector<T>& X,
        int ldb,
        int num_vecs,
        int ldc,
        T* accum) {
        if (count == 0) {
            return;
        }
        if (mode == ApplyMode::Batched) {
            run_mult_dense_batch_batched(graph, batch, begin, count, X, ldb, num_vecs, ldc, accum);
            return;
        }
        run_mult_dense_batch_fallback(graph, batch, begin, count, X, ldb, num_vecs, ldc, accum);
    }

    static void run_mult_dense_batch_fallback(
        DistGraph* graph,
        const PageBatch& batch,
        uint32_t begin,
        uint32_t count,
        DistMultiVector<T>& X,
        int ldb,
        int num_vecs,
        int ldc,
        T* accum) {
        for (uint32_t offset = 0; offset < count; ++offset) {
            const uint32_t idx = begin + offset;
            const int row = batch.row_block_index(idx);
            const int col = batch.col_block_index(idx);
            const T* block = batch.block_ptr(idx);
            const T* x_ptr = &X(graph->block_offsets[col], 0);
            T* y_ptr = accum + graph->block_offsets[row];
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

    static void run_mult_adjoint_batch(
        DistGraph* graph,
        const PageBatch& batch,
        uint32_t begin,
        uint32_t count,
        ApplyMode mode,
        DistVector<T>& x,
        T* accum) {
        if (count == 0) {
            return;
        }
        if (mode == ApplyMode::Batched) {
            run_mult_adjoint_batch_batched(graph, batch, begin, count, x, accum);
            return;
        }
        run_mult_adjoint_batch_fallback(graph, batch, begin, count, x, accum);
    }

    static void run_mult_adjoint_batch_fallback(
        DistGraph* graph,
        const PageBatch& batch,
        uint32_t begin,
        uint32_t count,
        DistVector<T>& x,
        T* accum) {
        for (uint32_t offset = 0; offset < count; ++offset) {
            const uint32_t idx = begin + offset;
            const int row = batch.row_block_index(idx);
            const int col = batch.col_block_index(idx);
            const T* block = batch.block_ptr(idx);
            const T* x_ptr = x.local_data() + graph->block_offsets[row];
            T* y_ptr = accum + graph->block_offsets[col];
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
        DistGraph* graph,
        const PageBatch& batch,
        uint32_t begin,
        uint32_t count,
        ApplyMode mode,
        DistMultiVector<T>& X,
        int ldb,
        int num_vecs,
        int ldc,
        T* accum) {
        if (count == 0) {
            return;
        }
        if (mode == ApplyMode::Batched) {
            run_mult_dense_adjoint_batch_batched(graph, batch, begin, count, X, ldb, num_vecs, ldc, accum);
            return;
        }
        run_mult_dense_adjoint_batch_fallback(graph, batch, begin, count, X, ldb, num_vecs, ldc, accum);
    }

    static void run_mult_dense_adjoint_batch_fallback(
        DistGraph* graph,
        const PageBatch& batch,
        uint32_t begin,
        uint32_t count,
        DistMultiVector<T>& X,
        int ldb,
        int num_vecs,
        int ldc,
        T* accum) {
        for (uint32_t offset = 0; offset < count; ++offset) {
            const uint32_t idx = begin + offset;
            const int row = batch.row_block_index(idx);
            const int col = batch.col_block_index(idx);
            const T* block = batch.block_ptr(idx);
            const T* x_ptr = &X(graph->block_offsets[row], 0);
            T* y_ptr = accum + graph->block_offsets[col];
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

    static void run_mult_batch_batched(
        DistGraph* graph,
        const PageBatch& batch,
        uint32_t begin,
        uint32_t count,
        DistVector<T>& x,
        T* accum) {
        const int a_stride = batch.row_dim * batch.col_dim;
        const uint32_t chunk_size = choose_chunk_size(
            static_cast<size_t>(batch.row_dim + batch.col_dim),
            count);

        std::vector<T> x_scratch;
        std::vector<T> y_scratch;
        x_scratch.reserve(static_cast<size_t>(chunk_size) * batch.col_dim);
        y_scratch.reserve(static_cast<size_t>(chunk_size) * batch.row_dim);

        for (uint32_t offset = 0; offset < count; offset += chunk_size) {
            const uint32_t local_count = std::min<uint32_t>(chunk_size, count - offset);
            x_scratch.resize(static_cast<size_t>(local_count) * batch.col_dim);
            y_scratch.assign(static_cast<size_t>(local_count) * batch.row_dim, T(0));

            for (uint32_t idx = 0; idx < local_count; ++idx) {
                const int col = batch.col_block_index(begin + offset + idx);
                const T* x_ptr = x.data.data() + graph->block_offsets[col];
                std::memcpy(
                    x_scratch.data() + static_cast<size_t>(idx) * batch.col_dim,
                    x_ptr,
                    static_cast<size_t>(batch.col_dim) * sizeof(T));
            }

            SmartKernel<T>::gemv_batched(
                batch.row_dim,
                batch.col_dim,
                T(1),
                batch.block_ptr(begin + offset),
                batch.row_dim,
                a_stride,
                x_scratch.data(),
                1,
                batch.col_dim,
                T(0),
                y_scratch.data(),
                1,
                batch.row_dim,
                static_cast<int>(local_count));

            for (uint32_t idx = 0; idx < local_count; ++idx) {
                const int row = batch.row_block_index(begin + offset + idx);
                T* y_ptr = accum + graph->block_offsets[row];
                const T* y_local = y_scratch.data() + static_cast<size_t>(idx) * batch.row_dim;
                for (int i = 0; i < batch.row_dim; ++i) {
                    y_ptr[i] += y_local[i];
                }
            }
        }
    }

    static void run_mult_dense_batch_batched(
        DistGraph* graph,
        const PageBatch& batch,
        uint32_t begin,
        uint32_t count,
        DistMultiVector<T>& X,
        int ldb,
        int num_vecs,
        int ldc,
        T* accum) {
        const int a_stride = batch.row_dim * batch.col_dim;
        const int b_stride = batch.col_dim * num_vecs;
        const int c_stride = batch.row_dim * num_vecs;
        const uint32_t chunk_size = choose_chunk_size(
            static_cast<size_t>(num_vecs) * static_cast<size_t>(batch.row_dim + batch.col_dim),
            count);

        std::vector<T> b_scratch;
        std::vector<T> c_scratch;
        b_scratch.reserve(static_cast<size_t>(chunk_size) * b_stride);
        c_scratch.reserve(static_cast<size_t>(chunk_size) * c_stride);

        for (uint32_t offset = 0; offset < count; offset += chunk_size) {
            const uint32_t local_count = std::min<uint32_t>(chunk_size, count - offset);
            b_scratch.resize(static_cast<size_t>(local_count) * b_stride);
            c_scratch.assign(static_cast<size_t>(local_count) * c_stride, T(0));

            for (uint32_t idx = 0; idx < local_count; ++idx) {
                const int col = batch.col_block_index(begin + offset + idx);
                const T* x_ptr = &X(graph->block_offsets[col], 0);
                T* packed_b = b_scratch.data() + static_cast<size_t>(idx) * b_stride;
                pack_multivector_block(x_ptr, ldb, batch.col_dim, num_vecs, packed_b);
            }

            SmartKernel<T>::gemm_batched(
                batch.row_dim,
                num_vecs,
                batch.col_dim,
                T(1),
                batch.block_ptr(begin + offset),
                batch.row_dim,
                a_stride,
                b_scratch.data(),
                batch.col_dim,
                b_stride,
                T(0),
                c_scratch.data(),
                batch.row_dim,
                c_stride,
                static_cast<int>(local_count));

            for (uint32_t idx = 0; idx < local_count; ++idx) {
                const int row = batch.row_block_index(begin + offset + idx);
                T* y_ptr = accum + graph->block_offsets[row];
                const T* y_local = c_scratch.data() + static_cast<size_t>(idx) * c_stride;
                accumulate_multivector_block(y_ptr, ldc, y_local, batch.row_dim, num_vecs);
            }
        }
    }

    static void run_mult_adjoint_batch_batched(
        DistGraph* graph,
        const PageBatch& batch,
        uint32_t begin,
        uint32_t count,
        DistVector<T>& x,
        T* accum) {
        const int a_stride = batch.row_dim * batch.col_dim;
        const uint32_t chunk_size = choose_chunk_size(
            static_cast<size_t>(batch.row_dim + batch.col_dim),
            count);

        std::vector<T> x_scratch;
        std::vector<T> y_scratch;
        x_scratch.reserve(static_cast<size_t>(chunk_size) * batch.row_dim);
        y_scratch.reserve(static_cast<size_t>(chunk_size) * batch.col_dim);

        for (uint32_t offset = 0; offset < count; offset += chunk_size) {
            const uint32_t local_count = std::min<uint32_t>(chunk_size, count - offset);
            x_scratch.resize(static_cast<size_t>(local_count) * batch.row_dim);
            y_scratch.assign(static_cast<size_t>(local_count) * batch.col_dim, T(0));

            for (uint32_t idx = 0; idx < local_count; ++idx) {
                const int row = batch.row_block_index(begin + offset + idx);
                const T* x_ptr = x.local_data() + graph->block_offsets[row];
                std::memcpy(
                    x_scratch.data() + static_cast<size_t>(idx) * batch.row_dim,
                    x_ptr,
                    static_cast<size_t>(batch.row_dim) * sizeof(T));
            }

            SmartKernel<T>::gemv_trans_batched(
                batch.row_dim,
                batch.col_dim,
                T(1),
                batch.block_ptr(begin + offset),
                batch.row_dim,
                a_stride,
                x_scratch.data(),
                1,
                batch.row_dim,
                T(0),
                y_scratch.data(),
                1,
                batch.col_dim,
                static_cast<int>(local_count));

            for (uint32_t idx = 0; idx < local_count; ++idx) {
                const int col = batch.col_block_index(begin + offset + idx);
                T* y_ptr = accum + graph->block_offsets[col];
                const T* y_local = y_scratch.data() + static_cast<size_t>(idx) * batch.col_dim;
                for (int i = 0; i < batch.col_dim; ++i) {
                    y_ptr[i] += y_local[i];
                }
            }
        }
    }

    static void run_mult_dense_adjoint_batch_batched(
        DistGraph* graph,
        const PageBatch& batch,
        uint32_t begin,
        uint32_t count,
        DistMultiVector<T>& X,
        int ldb,
        int num_vecs,
        int ldc,
        T* accum) {
        const int a_stride = batch.row_dim * batch.col_dim;
        const int b_stride = batch.row_dim * num_vecs;
        const int c_stride = batch.col_dim * num_vecs;
        const uint32_t chunk_size = choose_chunk_size(
            static_cast<size_t>(num_vecs) * static_cast<size_t>(batch.row_dim + batch.col_dim),
            count);

        std::vector<T> b_scratch;
        std::vector<T> c_scratch;
        b_scratch.reserve(static_cast<size_t>(chunk_size) * b_stride);
        c_scratch.reserve(static_cast<size_t>(chunk_size) * c_stride);

        for (uint32_t offset = 0; offset < count; offset += chunk_size) {
            const uint32_t local_count = std::min<uint32_t>(chunk_size, count - offset);
            b_scratch.resize(static_cast<size_t>(local_count) * b_stride);
            c_scratch.assign(static_cast<size_t>(local_count) * c_stride, T(0));

            for (uint32_t idx = 0; idx < local_count; ++idx) {
                const int row = batch.row_block_index(begin + offset + idx);
                const T* x_ptr = &X(graph->block_offsets[row], 0);
                T* packed_b = b_scratch.data() + static_cast<size_t>(idx) * b_stride;
                pack_multivector_block(x_ptr, ldb, batch.row_dim, num_vecs, packed_b);
            }

            SmartKernel<T>::gemm_trans_batched(
                batch.row_dim,
                num_vecs,
                batch.col_dim,
                T(1),
                batch.block_ptr(begin + offset),
                batch.row_dim,
                a_stride,
                b_scratch.data(),
                batch.row_dim,
                b_stride,
                T(0),
                c_scratch.data(),
                batch.col_dim,
                c_stride,
                static_cast<int>(local_count));

            for (uint32_t idx = 0; idx < local_count; ++idx) {
                const int col = batch.col_block_index(begin + offset + idx);
                T* y_ptr = accum + graph->block_offsets[col];
                const T* y_local = c_scratch.data() + static_cast<size_t>(idx) * c_stride;
                accumulate_multivector_block(y_ptr, ldc, y_local, batch.col_dim, num_vecs);
            }
        }
    }

    template <typename ScratchFn>
    static std::vector<ApplyTask> build_apply_tasks(
        const std::vector<PageBatch>& batches,
        bool batched_supported,
        ScratchFn&& scratch_elems_for_batch) {
        std::vector<ApplyTask> tasks;
        if (batches.empty()) {
            return tasks;
        }

        // Default scheduling unit is one non-empty shape page. Splitting is only
        // useful for the batched path: scalar fallback already exposes page-level
        // work, while a few huge repeated-shape pages can otherwise starve the
        // OpenMP team. The split size is bounded by the same scratch budget used
        // for packed operands and outputs in the batched micro-kernel launch.
        const bool should_split = batched_supported && should_split_apply_batches(batches.size());
        if (!should_split) {
            tasks.reserve(batches.size());
            for (int batch_index = 0; batch_index < static_cast<int>(batches.size()); ++batch_index) {
                tasks.push_back(ApplyTask{
                    batch_index,
                    0u,
                    batches[batch_index].block_count()});
            }
            return tasks;
        }

        size_t estimated_tasks = 0;
        for (const auto& batch : batches) {
            const uint32_t task_size = choose_chunk_size(
                scratch_elems_for_batch(batch),
                batch.block_count());
            estimated_tasks += (batch.block_count() + task_size - 1u) / task_size;
        }
        tasks.reserve(estimated_tasks);

        for (int batch_index = 0; batch_index < static_cast<int>(batches.size()); ++batch_index) {
            const auto& batch = batches[batch_index];
            const uint32_t task_size = choose_chunk_size(
                scratch_elems_for_batch(batch),
                batch.block_count());
            for (uint32_t begin = 0; begin < batch.block_count(); begin += task_size) {
                tasks.push_back(ApplyTask{
                    batch_index,
                    begin,
                    std::min<uint32_t>(task_size, batch.block_count() - begin)});
            }
        }
        return tasks;
    }

    static ApplyMode select_vector_apply_mode(uint32_t block_count) {
        return block_count > 1 && SmartKernel<T>::supports_batched_gemv()
            ? ApplyMode::Batched
            : ApplyMode::Scalar;
    }

    static ApplyMode select_dense_apply_mode(uint32_t block_count) {
        return block_count > 1 && SmartKernel<T>::supports_batched_gemm()
            ? ApplyMode::Batched
            : ApplyMode::Scalar;
    }

    static bool should_split_apply_batches(size_t batch_count) {
        return max_parallel_threads() > 1 &&
               batch_count < static_cast<size_t>(2 * max_parallel_threads());
    }

    // Estimate how many same-shape blocks fit in one batched micro-kernel launch
    // without growing operand/output scratch beyond the target scratch budget.
    // This is a memory-oriented heuristic, not a full performance model.
    static uint32_t choose_chunk_size(size_t per_block_scratch_elems, uint32_t total_blocks) {
        if (total_blocks == 0) {
            return 1;
        }
        const size_t target_elems = std::max<size_t>(1, kTargetScratchBytes / sizeof(T));
        const size_t blocks = per_block_scratch_elems == 0
            ? static_cast<size_t>(total_blocks)
            : std::max<size_t>(1, target_elems / per_block_scratch_elems);
        return static_cast<uint32_t>(std::min<size_t>(blocks, total_blocks));
    }

    static void pack_multivector_block(const T* src, int src_ld, int rows, int num_vecs, T* dest) {
        for (int vec = 0; vec < num_vecs; ++vec) {
            std::memcpy(
                dest + static_cast<size_t>(vec) * rows,
                src + static_cast<size_t>(vec) * src_ld,
                static_cast<size_t>(rows) * sizeof(T));
        }
    }

    static void accumulate_multivector_block(T* dest, int dest_ld, const T* src, int rows, int num_vecs) {
        for (int vec = 0; vec < num_vecs; ++vec) {
            T* dest_col = dest + static_cast<size_t>(vec) * dest_ld;
            const T* src_col = src + static_cast<size_t>(vec) * rows;
            for (int row = 0; row < rows; ++row) {
                dest_col[row] += src_col[row];
            }
        }
    }

    static std::vector<std::vector<T>> make_thread_accumulators(int element_count) {
        const int max_threads = max_parallel_threads();
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

    static int max_parallel_threads() {
        #ifdef _OPENMP
        return omp_get_max_threads();
        #else
        return 1;
        #endif
    }
};

} // namespace vbcsr_apply_detail

template <typename T, typename Kernel>
void vbcsr_mult(
    DistGraph* graph,
    const VBCSRMatrixBackend<T, Kernel>& backend,
    DistVector<T>& x,
    DistVector<T>& y) {
    vbcsr_apply_detail::ShapeBatchKernel<T, Kernel>::mult(graph, backend, x, y);
}

template <typename T, typename Kernel>
void vbcsr_mult_dense(
    DistGraph* graph,
    const VBCSRMatrixBackend<T, Kernel>& backend,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y) {
    vbcsr_apply_detail::ShapeBatchKernel<T, Kernel>::mult_dense(graph, backend, x, y);
}

template <typename T, typename Kernel>
void vbcsr_mult_adjoint(
    DistGraph* graph,
    const VBCSRMatrixBackend<T, Kernel>& backend,
    DistVector<T>& x,
    DistVector<T>& y) {
    vbcsr_apply_detail::ShapeBatchKernel<T, Kernel>::mult_adjoint(graph, backend, x, y);
}

template <typename T, typename Kernel>
void vbcsr_mult_dense_adjoint(
    DistGraph* graph,
    const VBCSRMatrixBackend<T, Kernel>& backend,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y) {
    vbcsr_apply_detail::ShapeBatchKernel<T, Kernel>::mult_dense_adjoint(graph, backend, x, y);
}

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_KERNELS_VBCSR_APPLY_HPP

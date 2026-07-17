#ifndef VBCSR_DETAIL_OPS_SPMM_VBCSR_HPP
#define VBCSR_DETAIL_OPS_SPMM_VBCSR_HPP

#include "../../distributed/block_payload_exchange.hpp"
#include "../../distributed/result_graph.hpp"
#include "common.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vbcsr::detail {

template <typename Matrix>
struct VBCSRSpMMExecutor {
    using T = typename Matrix::value_type;
    using Kernel = typename Matrix::KernelType;
    using GhostBlockRef = typename Matrix::GhostBlockRef;

private:
    static constexpr size_t kTargetScratchBytes = 1u << 20;

    struct HashEntry {
        int key;
        int graph_block_index;
        uint32_t tag;
    };

    struct ProductBatchKey {
        int row_dim = 0;
        int inner_dim = 0;
        int col_dim = 0;

        bool operator<(const ProductBatchKey& other) const {
            return std::tie(row_dim, inner_dim, col_dim) <
                   std::tie(other.row_dim, other.inner_dim, other.col_dim);
        }
    };

    struct ProductTask {
        const T* a_ptr = nullptr;
        const T* b_ptr = nullptr;
        T* c_ptr = nullptr;
    };

    struct ProductBatch {
        ProductBatchKey key;
        std::vector<ProductTask> tasks;
    };

public:
    static Matrix run(const Matrix& A, const Matrix& B, double threshold) {
        auto metadata = exchange_ghost_metadata(A, B);
        auto sym = symbolic_multiply_filtered(A, B, metadata, threshold);
        auto payload_ctx = fetch_required_block_payloads(B, sym.required_blocks);
        auto ghost_blocks = build_spmm_ghost_blocks(metadata, std::move(payload_ctx));
        auto adjacency = build_spmm_result_adjacency(A, sym);

        const auto& A_norms = A.get_block_norms();
        const auto& B_local_norms = B.get_block_norms();

        DistGraph* c_graph = construct_result_graph(A, adjacency, ghost_blocks.sizes, "spmm");
        Matrix C = make_result_matrix_for_numeric_overwrite(A, c_graph);

        numeric_multiply(A, B, ghost_blocks.rows, C, threshold, A_norms, B_local_norms);
        C.filter_blocks(threshold);
        return C;
    }

    static void run_numeric(
        const Matrix& A,
        const Matrix& B,
        const std::map<int, std::vector<GhostBlockRef>>& ghost_rows,
        Matrix& C,
        double threshold,
        const std::vector<double>& A_norms,
        const std::vector<double>& B_local_norms) {
        numeric_multiply(A, B, ghost_rows, C, threshold, A_norms, B_local_norms);
    }

private:
    static Matrix make_result_matrix_for_numeric_overwrite(const Matrix& A, DistGraph* c_graph) {
        std::unique_ptr<DistGraph> graph_guard(c_graph);
        const MatrixKind result_kind = Matrix::detect_matrix_kind(graph_guard.get());
        if (result_kind != MatrixKind::VBCSR) {
            Matrix C(graph_guard.get());
            C.owns_graph = true;
            C.graph->enable_matrix_lifetime_management();
            C.set_page_size(A.configured_page_size());
            graph_guard.release();
            return C;
        }

        Matrix C(
            graph_guard.get(),
            MatrixKind::VBCSR,
            true,
            typename Matrix::ConstructionToken{});
        graph_guard.release();

        using VBCSRBackendStorage = typename Matrix::VBCSRBackendStorage;
        VBCSRBackendStorage backend(A.configured_page_size());
        const size_t nnz = C.graph->adj_ind.size();
        backend.initialize_graph_block_handles(nnz);

        std::map<std::pair<int, int>, std::vector<int>> graph_blocks_by_shape;
        const int n_rows = static_cast<int>(C.graph->owned_global_indices.size());
        for (int row = 0; row < n_rows; ++row) {
            const int row_dim = C.graph->block_sizes[row];
            for (int graph_block_index = C.graph->adj_ptr[row];
                 graph_block_index < C.graph->adj_ptr[row + 1];
                 ++graph_block_index) {
                const int col = C.graph->adj_ind[graph_block_index];
                const int col_dim = C.graph->block_sizes[col];
                graph_blocks_by_shape[std::make_pair(row_dim, col_dim)].push_back(graph_block_index);
            }
        }

        for (const auto& [shape, graph_blocks] : graph_blocks_by_shape) {
            const int shape_id =
                backend.ensure_shape(shape.first, shape.second, graph_blocks.size());
            backend.append_blocks_for_shape_uninitialized(shape_id, graph_blocks);
        }

        C.attach_backend(std::move(backend));
        return C;
    }

    static void numeric_multiply(
        const Matrix& A,
        const Matrix& B,
        const std::map<int, std::vector<GhostBlockRef>>& ghost_rows,
        Matrix& C,
        double threshold,
        const std::vector<double>& A_norms,
        const std::vector<double>& B_local_norms) {
        const int n_rows = static_cast<int>(A.row_ptr().size()) - 1;

        int max_threads = 1;
        #ifdef _OPENMP
        max_threads = omp_get_max_threads();
        #endif
        const size_t hash_size = choose_hash_table_size(C);
        const size_t hash_mask = hash_size - 1;

        std::vector<std::vector<HashEntry>> thread_tables(
            max_threads,
            std::vector<HashEntry>(hash_size, {-1, -1, 0}));
        std::vector<uint32_t> thread_tags(max_threads, 0);

        const auto run_parallel = [&](auto use_small_product_batches_tag) {
            constexpr bool UseSmallProductBatches =
                decltype(use_small_product_batches_tag)::value;

            #pragma omp parallel
            {
                int tid = 0;
                #ifdef _OPENMP
                tid = omp_get_thread_num();
                #endif
                auto& table = thread_tables[tid];
                uint32_t& tag = thread_tags[tid];
                std::vector<int> small_batch_slots;
                if constexpr (UseSmallProductBatches) {
                    small_batch_slots.assign(small_product_key_count(), -1);
                }
                std::vector<ProductBatch> small_product_batches;
                std::map<ProductBatchKey, std::vector<ProductTask>> product_batches;

                const auto enqueue_product =
                    [&](const ProductBatchKey& key, ProductTask task) {
                        if constexpr (UseSmallProductBatches) {
                            if (is_small_product_key(key)) {
                                const size_t slot = small_product_key_index(key);
                                int batch_index = small_batch_slots[slot];
                                if (batch_index < 0) {
                                    batch_index = static_cast<int>(small_product_batches.size());
                                    small_batch_slots[slot] = batch_index;
                                    small_product_batches.push_back(ProductBatch{key, {}});
                                }
                                small_product_batches[static_cast<size_t>(batch_index)].tasks.push_back(task);
                                return;
                            }
                        }
                        product_batches[key].push_back(task);
                    };

                #pragma omp for
                for (int row = 0; row < n_rows; ++row) {
                    ++tag;
                    if (tag == 0) {
                        for (auto& entry : table) {
                            entry.tag = 0;
                        }
                        tag = 1;
                    }

                    const int c_start = C.row_ptr()[row];
                    const int c_end = C.row_ptr()[row + 1];
                    for (int graph_block_index = c_start; graph_block_index < c_end; ++graph_block_index) {
                        const int local_col = C.col_ind()[graph_block_index];
                        const int global_col = C.graph->get_global_index(local_col);
                        T* c_values = C.mutable_block_data(graph_block_index);
                        std::fill(
                            c_values,
                            c_values + C.block_size_elements(graph_block_index),
                            T(0));

                        size_t h = static_cast<size_t>(global_col) & hash_mask;
                        size_t count = 0;
                        while (table[h].tag == tag) {
                            h = (h + 1) & hash_mask;
                            if (++count > hash_size) {
                                throw std::runtime_error("Hash table is full during VBCSR SpMM population");
                            }
                        }
                        table[h] = {global_col, graph_block_index, tag};
                    }

                    const int a_start = A.row_ptr()[row];
                    const int a_end = A.row_ptr()[row + 1];
                    const int r_dim = A.graph->block_sizes[row];

                    for (int a_graph_block = a_start; a_graph_block < a_end; ++a_graph_block) {
                        const int row_count = a_end - a_start;
                        const double row_eps = threshold / std::max(1, row_count);

                        const int local_col_A = A.col_ind()[a_graph_block];
                        const int global_col_A = A.graph->get_global_index(local_col_A);
                        const T* a_val = A.block_data(a_graph_block);
                        const int inner_dim = A.graph->block_sizes[local_col_A];
                        const double norm_A = A_norms[a_graph_block];

                        if (A.graph->find_owner(global_col_A) == A.graph->rank) {
                            const int local_row_B = B.graph->global_to_local.at(global_col_A);
                            const int b_start = B.row_ptr()[local_row_B];
                            const int b_end = B.row_ptr()[local_row_B + 1];
                            for (int b_graph_block = b_start; b_graph_block < b_end; ++b_graph_block) {
                                const double norm_B = B_local_norms[b_graph_block];
                                if (norm_A * norm_B < row_eps) {
                                    continue;
                                }

                                const int local_col_B = B.col_ind()[b_graph_block];
                                const int global_col_B = B.graph->get_global_index(local_col_B);
                                const T* b_val = B.block_data(b_graph_block);
                                const int c_dim = B.graph->block_sizes[local_col_B];

                                accumulate_product(
                                    table,
                                    tag,
                                    hash_mask,
                                    hash_size,
                                    global_col_B,
                                    [&](int c_graph_block) {
                                        enqueue_product(
                                            ProductBatchKey{r_dim, inner_dim, c_dim},
                                            ProductTask{a_val, b_val, C.mutable_block_data(c_graph_block)});
                                    },
                                    "local");
                            }
                        } else {
                            auto it = ghost_rows.find(global_col_A);
                            if (it == ghost_rows.end()) {
                                continue;
                            }
                            for (const auto& block : it->second) {
                                if (norm_A * block.norm < row_eps) {
                                    continue;
                                }

                                accumulate_product(
                                    table,
                                    tag,
                                    hash_mask,
                                    hash_size,
                                    block.col,
                                    [&](int c_graph_block) {
                                        enqueue_product(
                                            ProductBatchKey{r_dim, inner_dim, block.c_dim},
                                            ProductTask{a_val, block.data, C.mutable_block_data(c_graph_block)});
                                    },
                                    "ghost");
                            }
                        }
                    }
                }

                if constexpr (UseSmallProductBatches) {
                    for (auto& batch : small_product_batches) {
                        run_product_batch_fallback(batch.key, batch.tasks);
                    }
                }

                for (auto& [key, tasks] : product_batches) {
                    if (use_direct_product_kernel(key)) {
                        run_product_batch_fallback(key, tasks);
                        continue;
                    }
                    if (SmartKernel<T>::supports_batched_gemm()) {
                        run_product_batch_batched(key, tasks);
                        continue;
                    }
                    run_product_batch_fallback(key, tasks);
                }
            }
        };

        if (threshold > 0.0) {
            run_parallel(std::true_type{});
        } else {
            run_parallel(std::false_type{});
        }
    }

    static size_t choose_hash_table_size(const Matrix& C) {
        size_t max_row_blocks = 1;
        const auto& row_ptr = C.row_ptr();
        for (size_t row = 0; row + 1 < row_ptr.size(); ++row) {
            const size_t row_blocks =
                static_cast<size_t>(row_ptr[row + 1] - row_ptr[row]);
            max_row_blocks = std::max(max_row_blocks, row_blocks);
        }

        if (max_row_blocks > (std::numeric_limits<size_t>::max() - 1) / 2) {
            throw std::overflow_error("VBCSR SpMM row is too wide for numeric hash table sizing");
        }

        const size_t required = std::max<size_t>(16, 2 * max_row_blocks + 1);
        size_t hash_size = 16;
        while (hash_size < required) {
            if (hash_size > std::numeric_limits<size_t>::max() / 2) {
                throw std::overflow_error("VBCSR SpMM numeric hash table size overflow");
            }
            hash_size <<= 1;
        }
        return hash_size;
    }

    static bool use_direct_product_kernel(const ProductBatchKey& key) {
        return key.row_dim <= 20 && key.inner_dim <= 20 && key.col_dim <= 20;
    }

    static constexpr int kSmallProductDimLimit = 20;

    static constexpr size_t small_product_key_count() {
        return static_cast<size_t>(kSmallProductDimLimit + 1) *
               static_cast<size_t>(kSmallProductDimLimit + 1) *
               static_cast<size_t>(kSmallProductDimLimit + 1);
    }

    static bool is_small_product_key(const ProductBatchKey& key) {
        return key.row_dim > 0 && key.row_dim <= kSmallProductDimLimit &&
               key.inner_dim > 0 && key.inner_dim <= kSmallProductDimLimit &&
               key.col_dim > 0 && key.col_dim <= kSmallProductDimLimit;
    }

    static size_t small_product_key_index(const ProductBatchKey& key) {
        constexpr size_t stride = static_cast<size_t>(kSmallProductDimLimit + 1);
        return (static_cast<size_t>(key.row_dim) * stride +
                static_cast<size_t>(key.inner_dim)) * stride +
               static_cast<size_t>(key.col_dim);
    }

    template <typename F>
    static void accumulate_product(
        std::vector<HashEntry>& table,
        uint32_t tag,
        size_t hash_mask,
        size_t hash_size,
        int global_col,
        F&& update,
        const char* phase) {
        size_t h = static_cast<size_t>(global_col) & hash_mask;
        size_t count = 0;
        while (table[h].tag == tag) {
            if (table[h].key == global_col) {
                update(table[h].graph_block_index);
                break;
            }
            h = (h + 1) & hash_mask;
            if (++count > hash_size) {
                throw std::runtime_error(
                    std::string("Hash table infinite loop detected during VBCSR SpMM numeric phase (") + phase + ")");
            }
        }
    }

    static void run_product_batch_fallback(const ProductBatchKey& key, const std::vector<ProductTask>& tasks) {
        for (const auto& task : tasks) {
            SmartKernel<T>::gemm(
                key.row_dim,
                key.col_dim,
                key.inner_dim,
                T(1),
                task.a_ptr,
                key.row_dim,
                task.b_ptr,
                key.inner_dim,
                T(1),
                task.c_ptr,
                key.row_dim);
        }
    }

    static void run_product_batch_batched(const ProductBatchKey& key, const std::vector<ProductTask>& tasks) {
        if (tasks.empty()) {
            return;
        }

        const int a_stride = key.row_dim * key.inner_dim;
        const int b_stride = key.inner_dim * key.col_dim;
        const int c_stride = key.row_dim * key.col_dim;
        const uint32_t chunk_size = choose_chunk_size(
            static_cast<size_t>(a_stride + b_stride + c_stride),
            static_cast<uint32_t>(tasks.size()));

        std::vector<T> a_scratch;
        std::vector<T> b_scratch;
        std::vector<T> c_scratch;
        a_scratch.reserve(static_cast<size_t>(chunk_size) * a_stride);
        b_scratch.reserve(static_cast<size_t>(chunk_size) * b_stride);
        c_scratch.reserve(static_cast<size_t>(chunk_size) * c_stride);

        for (uint32_t begin = 0; begin < tasks.size(); begin += chunk_size) {
            const uint32_t count = std::min<uint32_t>(chunk_size, static_cast<uint32_t>(tasks.size()) - begin);
            a_scratch.resize(static_cast<size_t>(count) * a_stride);
            b_scratch.resize(static_cast<size_t>(count) * b_stride);
            c_scratch.assign(static_cast<size_t>(count) * c_stride, T(0));

            for (uint32_t idx = 0; idx < count; ++idx) {
                const auto& task = tasks[begin + idx];
                std::memcpy(
                    a_scratch.data() + static_cast<size_t>(idx) * a_stride,
                    task.a_ptr,
                    static_cast<size_t>(a_stride) * sizeof(T));
                std::memcpy(
                    b_scratch.data() + static_cast<size_t>(idx) * b_stride,
                    task.b_ptr,
                    static_cast<size_t>(b_stride) * sizeof(T));
            }

            SmartKernel<T>::gemm_batched(
                key.row_dim,
                key.col_dim,
                key.inner_dim,
                T(1),
                a_scratch.data(),
                key.row_dim,
                a_stride,
                b_scratch.data(),
                key.inner_dim,
                b_stride,
                T(0),
                c_scratch.data(),
                key.row_dim,
                c_stride,
                static_cast<int>(count));

            for (uint32_t idx = 0; idx < count; ++idx) {
                T* dest = tasks[begin + idx].c_ptr;
                const T* src = c_scratch.data() + static_cast<size_t>(idx) * c_stride;
                for (int elem = 0; elem < c_stride; ++elem) {
                    dest[elem] += src[elem];
                }
            }
        }
    }

    static uint32_t choose_chunk_size(size_t per_task_scratch_elems, uint32_t total_tasks) {
        if (total_tasks == 0) {
            return 1;
        }
        const size_t target_elems = std::max<size_t>(1, kTargetScratchBytes / sizeof(T));
        const size_t tasks_per_chunk = per_task_scratch_elems == 0
            ? static_cast<size_t>(total_tasks)
            : std::max<size_t>(1, target_elems / per_task_scratch_elems);
        return static_cast<uint32_t>(std::min<size_t>(tasks_per_chunk, total_tasks));
    }
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_OPS_SPMM_VBCSR_HPP

#ifndef VBCSR_DETAIL_OPS_SPMM_VBCSR_HPP
#define VBCSR_DETAIL_OPS_SPMM_VBCSR_HPP

#include "../../distributed/block_payload_exchange.hpp"
#include "../../distributed/result_graph.hpp"
#include "../../kernels/rowmajor_kernels.hpp"
#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
    // `consume_A`, when set, must alias A: the caller is doing A <- A B and
    // has given up A, so each chunk of A's value pages is handed back as soon
    // as the rows that read them are accumulated AND their queued product
    // batches have flushed -- this executor defers its GEMMs into per-thread
    // batches holding raw pointers into A's storage, so the flush has to move
    // inside the chunk (see numeric_multiply). The ghost exchange completes
    // before the numeric loop and A's blocks are never fetched by another
    // rank, so nothing anyone still needs is released. With consume_A null the
    // chunk loop runs once and this is the non-consuming path unchanged.
    static Matrix run(const Matrix& A, const Matrix& B, double threshold,
                      bool upper_only = false, Matrix* consume_A = nullptr) {
        const bool profile = std::getenv("VBCSR_PROFILE_VBCSR_SPGEMM") != nullptr;
        const auto t0 = std::chrono::steady_clock::now();
        auto metadata = exchange_ghost_metadata(A, B);
        auto sym = symbolic_multiply_filtered(A, B, metadata, threshold, upper_only);
        const auto t_symbolic = std::chrono::steady_clock::now();
        auto payload_ctx = fetch_required_block_payloads(B, sym.required_blocks);
        auto ghost_blocks = build_spmm_ghost_blocks(metadata, std::move(payload_ctx));
        auto adjacency = build_spmm_result_adjacency(A, sym);
        const auto t_adjacency = std::chrono::steady_clock::now();

        const auto& A_norms = A.get_block_norms();
        const auto& B_local_norms = B.get_block_norms();
        const auto t_norms = std::chrono::steady_clock::now();

        DistGraph* c_graph = construct_result_graph(A, adjacency, ghost_blocks.sizes, "spmm");
        const auto t_graph = std::chrono::steady_clock::now();
        Matrix C = make_result_matrix_for_numeric_overwrite(
            A, c_graph, /*defer_zero=*/consume_A != nullptr);
        const auto t_structure = std::chrono::steady_clock::now();

        numeric_multiply(A, B, ghost_blocks.rows, C, threshold, A_norms, B_local_norms,
                         consume_A);
        const auto t_numeric = std::chrono::steady_clock::now();
        C.filter_blocks(threshold);
        const auto t_filter = std::chrono::steady_clock::now();

        if (profile) {
            auto seconds = [](auto a, auto b) {
                return std::chrono::duration<double>(b - a).count();
            };
            std::cerr
                << "VBCSR_PROFILE_VBCSR_SPGEMM"
                << " symbolic=" << seconds(t0, t_symbolic)
                << " adjacency=" << seconds(t_symbolic, t_adjacency)
                << " norms=" << seconds(t_adjacency, t_norms)
                << " graph=" << seconds(t_norms, t_graph)
                << " structure=" << seconds(t_graph, t_structure)
                << " numeric=" << seconds(t_structure, t_numeric)
                << " filter=" << seconds(t_numeric, t_filter)
                << " total=" << seconds(t0, t_filter)
                << std::endl;
        }
        return C;
    }

    static Matrix run_consuming(Matrix& A, const Matrix& B, double threshold) {
        return run(A, B, threshold, /*upper_only=*/false, /*consume_A=*/&A);
    }

    static void run_numeric(
        const Matrix& A,
        const Matrix& B,
        const std::map<int, std::vector<GhostBlockRef>>& ghost_rows,
        Matrix& C,
        double threshold,
        const std::vector<double>& A_norms,
        const std::vector<double>& B_local_norms) {
        numeric_multiply(A, B, ghost_rows, C, threshold, A_norms, B_local_norms,
                         /*consume_A=*/nullptr);
    }

private:
    static Matrix make_result_matrix_for_numeric_overwrite(const Matrix& A, DistGraph* c_graph,
                                                           bool defer_zero) {
        if (defer_zero) {
            // The consuming path. The eager pass below would make the whole
            // result resident before a single row of the operand has been
            // released -- the source and the finished result both whole at
            // once, which is exactly the peak the caller asked to avoid. Every
            // destination block is std::fill'ed by the numeric loop, so the
            // pass is redundant for correctness in both modes; what it buys is
            // deterministic placement, and the consuming numeric loop
            // re-establishes that with a per-chunk domain-aligned touch (the
            // same shape bsr.hpp and filter_blocks use).
            // make_overwritten_result is the deferred build for every kind and
            // already handles the empty-rank graph without extra collectives.
            return Matrix::make_overwritten_result(c_graph, A.configured_page_size());
        }
        std::unique_ptr<DistGraph> graph_guard(c_graph);
        const MatrixKind result_kind = Matrix::detect_matrix_kind(graph_guard.get());
        if (result_kind != MatrixKind::VBCSR) {
            // Construct with the kind in hand, not `Matrix C(graph)`: that
            // constructor runs detect_matrix_kind itself, which is COLLECTIVE,
            // and repeating it here costs every rank two more Allreduce for an
            // answer already on the line above. Safe as it stood -- the branch
            // is on the globally reduced kind, so no rank takes it alone -- but
            // the same shape inside make_overwritten_result, where the branch
            // WAS local, hung the job. Not a pattern to leave lying around.
            Matrix C(graph_guard.get(), result_kind, true,
                     typename Matrix::ConstructionToken{});
            graph_guard.release();
            C.allocate_from_graph();
            C.set_page_size(A.configured_page_size());
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
        // Shared first-touch structure build: the numeric fill below then
        // overwrites pages already placed on the threads that later apply
        // this result (numa_locality_plan.md — operation results).
        //
        // NOT deferred, unlike the CSR and BSR results, and the difference is
        // deliberate. The pass is redundant for CORRECTNESS here -- the numeric
        // loop std::fills every destination block before accumulating into it --
        // so it looks like pure waste, and it does cost a fully resident result
        // before any value exists. What it buys is DETERMINISTIC placement: this
        // pass walks the thread-domain partition that the forward apply plan
        // later reuses, while the numeric loop below is schedule(dynamic, 4), so
        // deferring would hand each page to whichever thread happened to win that
        // row and leave the apply reading across nodes.
        //
        // Closing this properly means giving the numeric loop a domain-aligned
        // static schedule (then its fill IS the placing touch and the pass can go
        // via build_first_touch_structure(defer_zero=true) + zero_domain), which
        // trades the load balance dynamic scheduling was chosen for. That is a
        // measurement, not a cleanup, so it is left alone rather than guessed at.
        backend.build_first_touch_structure(
            C.graph->adj_ptr,
            C.graph->adj_ind,
            C.graph->block_sizes,
            static_cast<int>(C.graph->owned_global_indices.size()));

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
        const std::vector<double>& B_local_norms,
        Matrix* consume_A) {
        const int n_rows = static_cast<int>(A.row_ptr().size()) - 1;

        // Numeric products call BLAS (gemm_batched) from inside the OpenMP
        // region below (thread policy kind C, dense_kernels.hpp).
        BLASKernel::ScopedSerialBLAS serial_blas;

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

        // 4 chunks when consuming, one otherwise -- the constant, its
        // machine-independence, and the trade it prices are recorded at
        // kConsumeChunks in bsr.hpp, which measured them.
        constexpr int kConsumeChunks = 4;
        const int chunk_rows = (consume_A == nullptr || n_rows <= 0)
                                   ? std::max(1, n_rows)
                                   : std::max(1, (n_rows + kConsumeChunks - 1) / kConsumeChunks);
        // The placing touch below wants the result's thread domains; guarded,
        // like every other consumer of the partition, so an unaligned build
        // degrades to unplaced pages rather than wrong ones.
        const auto& domains = C.thread_domain_partition();
        const bool domains_aligned = domains.thread_count > 0 &&
                                     static_cast<int>(domains.row_bounds.size()) ==
                                         domains.thread_count + 1;

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

                size_t queued_tasks = 0;
                const auto enqueue_product =
                    [&](const ProductBatchKey& key, ProductTask task) {
                        ++queued_tasks;
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

                // This thread's queued products, run and cleared. Once per
                // numeric pass when nothing is consumed; once per CHUNK when
                // consuming, because the tasks hold raw pointers into A's
                // storage and the release at the chunk edge would pull the
                // pages out from under them.
                const auto flush_batches = [&] {
                    if constexpr (UseSmallProductBatches) {
                        for (auto& batch : small_product_batches) {
                            run_product_batch_fallback(batch.key, batch.tasks);
                            small_batch_slots[small_product_key_index(batch.key)] = -1;
                        }
                        small_product_batches.clear();
                    }
                    for (auto& [key, tasks] : product_batches) {
                        if (use_direct_product_kernel(key)) {
                            run_product_batch_fallback(key, tasks);
                            continue;
                        }
                        if constexpr (supports_batched_products()) {
                            run_product_batch_batched(key, tasks);
                        } else {
                            run_product_batch_fallback(key, tasks);
                        }
                    }
                    product_batches.clear();
                };

                for (int chunk_begin = 0; chunk_begin < n_rows; chunk_begin += chunk_rows) {
                const int chunk_end = std::min(n_rows, chunk_begin + chunk_rows);
                // The first touch the deferred constructor no longer does,
                // per chunk and domain-aligned so each page still lands on
                // the node whose thread later applies that row, while pages
                // beyond the chunk stay unmapped. Placement and parallelism
                // separated exactly as in filter_blocks' copy_kept_blocks:
                // the touch is static over domains, the numeric loop below
                // stays dynamic over the whole chunk.
                if (consume_A != nullptr && domains_aligned) {
                    #pragma omp for schedule(static)
                    for (int d = 0; d < domains.thread_count; ++d) {
                        const int lo = std::max(domains.domain_begin(d), chunk_begin);
                        const int hi = std::min(domains.domain_end(d), chunk_end);
                        if (lo < hi) C.touch_row_range(lo, hi);
                    }
                }

                // Dynamic, not static: under upper_only the per-row work falls
                // linearly across the matrix, and a static split roughly
                // doubles the first thread's share (measured on the BSR
                // executor as the halved product losing a third of its speed).
                #pragma omp for schedule(dynamic, 4)
                for (int row = chunk_begin; row < chunk_end; ++row) {
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

                    // Row-boundary flush cap. The queue holds 24 bytes per
                    // block PAIR, and unbounded it holds every pair of the
                    // whole pass: measured at 2048 rows x ~820 x ~700
                    // neighbours, ~29 GB of task queues beside a 4.6 GB
                    // result -- six times the answer, growing with pairs
                    // (~flops), the fastest-growing quantity there is.
                    // Accumulating past this cap buys nothing: execution
                    // chunks every batch to kTargetScratchBytes anyway, so a
                    // key's grouping gain saturates thousands of tasks below
                    // it. The cap only defers flushes long enough for SPARSE
                    // rows to keep amortising the per-key map walk. Checked
                    // at row boundaries, so the true bound is the cap plus
                    // one row's pairs.
                    constexpr size_t kMaxQueuedTasks = size_t(1) << 18;
                    if (queued_tasks >= kMaxQueuedTasks) {
                        flush_batches();
                        queued_tasks = 0;
                    }
                }

                flush_batches();

                // The omp for above ends on an implicit barrier, but the flush
                // is per-thread work after it, so an explicit barrier is what
                // proves every thread's tasks reading A have run before the
                // release. single carries its own exit barrier, so no thread
                // enters the next chunk's touch while the release is in
                // flight. The boundary is a ROW; how that becomes a byte range
                // is the backend's business (release_values_below_row).
                if (consume_A != nullptr) {
                    #pragma omp barrier
                    #pragma omp single
                    {
                        consume_A->release_values_below_row(chunk_end);
                    }
                }
                }  // chunk
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

    // Vendor strided-batch gemm exists for double and complex<double> only;
    // every other scalar type takes the row-major fallback loop.
    static constexpr bool supports_batched_products() {
        return (std::is_same_v<T, double> || std::is_same_v<T, std::complex<double>>) &&
               BLASKernel::supports_strided_gemm();
    }

    static void run_product_batch_fallback(const ProductBatchKey& key, const std::vector<ProductTask>& tasks) {
        // Block product C(row_dim x col_dim) += A(row_dim x inner_dim) *
        // B(inner_dim x col_dim), all blocks in canonical row-major storage:
        // rm_gemm with X = B (ldx = col_dim) and C compact (ldc = col_dim).
        for (const auto& task : tasks) {
            rowmajor_kernels::rm_gemm<T>(
                key.row_dim,
                key.inner_dim,
                key.col_dim,
                task.a_ptr,
                task.b_ptr,
                key.col_dim,
                task.c_ptr,
                key.col_dim);
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

            // Row-major blocks through the column-major vendor batch:
            // C_rm = A_rm * B_rm  <=>  C_cm' = B_cm' * A_cm' on the transposed
            // (= raw row-major) buffers, so swap the operands and dims.
            BLASKernel::gemm_batched(
                key.col_dim,
                key.row_dim,
                key.inner_dim,
                T(1),
                b_scratch.data(),
                key.col_dim,
                b_stride,
                a_scratch.data(),
                key.inner_dim,
                a_stride,
                T(0),
                c_scratch.data(),
                key.col_dim,
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

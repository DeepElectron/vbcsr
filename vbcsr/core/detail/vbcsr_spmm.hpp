#ifndef VBCSR_DETAIL_VBCSR_SPMM_HPP
#define VBCSR_DETAIL_VBCSR_SPMM_HPP

#include "distributed_plans.hpp"
#include "vbcsr_result_builder.hpp"

#include <map>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace detail {

template <typename Matrix>
struct VBCSRSpMMExecutor {
    using T = typename Matrix::value_type;
    using Kernel = typename Matrix::KernelType;
    using GhostBlockRef = typename Matrix::GhostBlockRef;

private:
    struct HashEntry {
        int key;
        int slot;
        int tag;
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

public:
    static Matrix run(const Matrix& A, const Matrix& B, double threshold) {
        auto metadata_plan = RowMetadataExchangePlan<Matrix, Matrix>::build(A, B);
        auto sym = symbolic_multiply_filtered(A, B, metadata_plan.metadata(), threshold);
        auto payload_plan = BlockPayloadExchangePlan<Matrix>::fetch_required(B, sym.required_blocks);
        auto assembly_plan = ResultAssemblyPlan<Matrix>::for_spmm(
            A,
            sym,
            metadata_plan.metadata(),
            std::move(payload_plan));

        const auto& A_norms = A.get_block_norms();
        const auto& B_local_norms = B.get_block_norms();

        DistGraph* c_graph = assembly_plan.construct_result_graph(A, "spmm");
        VBCSRResultBuilder<T, Kernel> builder(c_graph);
        Matrix C = Matrix::from_vbcsr_builder(c_graph, true, std::move(builder));

        numeric_multiply(A, B, assembly_plan.ghost_rows(), C, threshold, A_norms, B_local_norms);
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
    static void numeric_multiply(
        const Matrix& A,
        const Matrix& B,
        const std::map<int, std::vector<GhostBlockRef>>& ghost_rows,
        Matrix& C,
        double threshold,
        const std::vector<double>& A_norms,
        const std::vector<double>& B_local_norms) {
        using ExecutionKind = typename Matrix::VBCSRBackendStorage::ExecutionKind;
        const int n_rows = static_cast<int>(A.row_ptr.size()) - 1;

        const int max_threads = omp_get_max_threads();
        const size_t HASH_SIZE = 8192;
        const size_t HASH_MASK = HASH_SIZE - 1;

        std::vector<std::vector<HashEntry>> thread_tables(
            max_threads,
            std::vector<HashEntry>(HASH_SIZE, {-1, -1, 0}));
        std::vector<int> thread_tags(max_threads, 0);

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& table = thread_tables[tid];
            int& tag = thread_tags[tid];
            std::map<ProductBatchKey, std::vector<ProductTask>> product_batches;

            #pragma omp for
            for (int row = 0; row < n_rows; ++row) {
                ++tag;
                if (tag == 0) {
                    for (auto& entry : table) {
                        entry.tag = 0;
                    }
                    tag = 1;
                }

                const int c_start = C.row_ptr[row];
                const int c_end = C.row_ptr[row + 1];
                for (int slot = c_start; slot < c_end; ++slot) {
                    const int local_col = C.col_ind[slot];
                    const int global_col = C.graph->get_global_index(local_col);

                    size_t h = static_cast<size_t>(global_col) & HASH_MASK;
                    size_t count = 0;
                    while (table[h].tag == tag) {
                        h = (h + 1) & HASH_MASK;
                        if (++count > HASH_SIZE) {
                            throw std::runtime_error("Hash table is full during VBCSR SpMM population");
                        }
                    }
                    table[h] = {global_col, slot, tag};
                }

                const int a_start = A.row_ptr[row];
                const int a_end = A.row_ptr[row + 1];
                const int r_dim = A.graph->block_sizes[row];

                for (int a_slot = a_start; a_slot < a_end; ++a_slot) {
                    const int row_count = a_end - a_start;
                    const double row_eps = threshold / std::max(1, row_count);

                    const int local_col_A = A.col_ind[a_slot];
                    const int global_col_A = A.graph->get_global_index(local_col_A);
                    const T* a_val = A.block_data(a_slot);
                    const int inner_dim = A.graph->block_sizes[local_col_A];
                    const double norm_A = A_norms[a_slot];

                    if (A.graph->find_owner(global_col_A) == A.graph->rank) {
                        const int local_row_B = B.graph->global_to_local.at(global_col_A);
                        const int b_start = B.row_ptr[local_row_B];
                        const int b_end = B.row_ptr[local_row_B + 1];
                        for (int b_slot = b_start; b_slot < b_end; ++b_slot) {
                            const double norm_B = B_local_norms[b_slot];
                            if (norm_A * norm_B < row_eps) {
                                continue;
                            }

                            const int local_col_B = B.col_ind[b_slot];
                            const int global_col_B = B.graph->get_global_index(local_col_B);
                            const T* b_val = B.block_data(b_slot);
                            const int c_dim = B.graph->block_sizes[local_col_B];

                            accumulate_product(
                                table,
                                tag,
                                HASH_MASK,
                                HASH_SIZE,
                                global_col_B,
                                [&](int c_slot) {
                                    product_batches[ProductBatchKey{r_dim, inner_dim, c_dim}].push_back(
                                        ProductTask{a_val, b_val, C.mutable_block_data(c_slot)});
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
                                HASH_MASK,
                                HASH_SIZE,
                                block.col,
                                [&](int c_slot) {
                                    product_batches[ProductBatchKey{r_dim, inner_dim, block.c_dim}].push_back(
                                        ProductTask{a_val, block.data, C.mutable_block_data(c_slot)});
                                },
                                "ghost");
                        }
                    }
                }
            }

            for (auto& [key, tasks] : product_batches) {
                A.active_vbcsr_backend().record_spmm_batch(
                    key.row_dim,
                    key.inner_dim,
                    key.col_dim,
                    tasks.size());
                switch (A.active_vbcsr_backend().execution_kind_for_spmm_triple(
                    key.row_dim,
                    key.inner_dim,
                    key.col_dim)) {
                    case ExecutionKind::StaticFallback:
                    case ExecutionKind::BatchedFallback:
                    case ExecutionKind::JIT:
                        run_product_batch(key, tasks);
                        break;
                }
            }
        }
    }

    template <typename F>
    static void accumulate_product(
        std::vector<HashEntry>& table,
        int tag,
        size_t hash_mask,
        size_t hash_size,
        int global_col,
        F&& update,
        const char* phase) {
        size_t h = static_cast<size_t>(global_col) & hash_mask;
        size_t count = 0;
        while (table[h].tag == tag) {
            if (table[h].key == global_col) {
                update(table[h].slot);
                break;
            }
            h = (h + 1) & hash_mask;
            if (++count > hash_size) {
                throw std::runtime_error(
                    std::string("Hash table infinite loop detected during VBCSR SpMM numeric phase (") + phase + ")");
            }
        }
    }

    static void run_product_batch(const ProductBatchKey& key, const std::vector<ProductTask>& tasks) {
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
};

} // namespace detail

#endif // VBCSR_DETAIL_VBCSR_SPMM_HPP

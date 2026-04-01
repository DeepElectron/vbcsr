#ifndef VBCSR_DETAIL_LEGACY_SPMM_HPP
#define VBCSR_DETAIL_LEGACY_SPMM_HPP

#include <map>
#include <vector>

namespace detail {

template <typename Matrix>
struct LegacySpMMExecutor {
    using T = typename Matrix::value_type;
    using Kernel = typename Matrix::KernelType;
    using GhostBlockRef = typename Matrix::GhostBlockRef;

private:
    struct HashEntry {
        int key;
        int slot;
        int tag;
    };

public:

    static Matrix run(const Matrix& A, const Matrix& B, double threshold) {
        auto meta = A.exchange_ghost_metadata(B);

        const auto& A_norms = A.get_block_norms();
        const auto& B_local_norms = B.get_block_norms();

        auto sym = A.symbolic_multiply_filtered(B, meta, threshold);
        auto [ghost_data_map, ghost_sizes] = B.fetch_ghost_blocks(sym.required_blocks);

        std::map<int, std::vector<GhostBlockRef>> ghost_rows;
        for (const auto& [bid, data] : ghost_data_map) {
            const int c_dim = ghost_sizes.at(bid.col);

            double norm = 0.0;
            if (meta.count(bid.row)) {
                for (const auto& m : meta.at(bid.row)) {
                    if (m.col == bid.col) {
                        norm = m.norm;
                        break;
                    }
                }
            }

            ghost_rows[bid.row].push_back({bid.col, data.data(), c_dim, norm});
        }

        std::vector<std::vector<int>> adj(A.graph->owned_global_indices.size());
        const int n_rows = static_cast<int>(A.row_ptr.size()) - 1;
        for (int row = 0; row < n_rows; ++row) {
            for (int slot = sym.c_row_ptr[row]; slot < sym.c_row_ptr[row + 1]; ++slot) {
                adj[row].push_back(sym.c_col_ind[slot]);
            }
        }

        DistGraph* c_graph = construct_result_graph(
            A.graph->comm,
            A.graph->owned_global_indices,
            owned_block_sizes(*A.graph),
            adj,
            ghost_sizes,
            "spmm");

        LegacyMatrixBuilder<T, Kernel> builder(c_graph, true);
        Matrix C = builder.materialize();

        numeric_multiply(A, B, ghost_rows, C, threshold, A_norms, B_local_norms);
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
                            throw std::runtime_error("Hash table is full during SpMM population");
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
                                    T* c_val = C.mutable_block_data(c_slot);
                                    SmartKernel<T>::gemm(
                                        r_dim,
                                        c_dim,
                                        inner_dim,
                                        T(1),
                                        a_val,
                                        r_dim,
                                        b_val,
                                        inner_dim,
                                        T(1),
                                        c_val,
                                        r_dim);
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
                                    T* c_val = C.mutable_block_data(c_slot);
                                    SmartKernel<T>::gemm(
                                        r_dim,
                                        block.c_dim,
                                        inner_dim,
                                        T(1),
                                        a_val,
                                        r_dim,
                                        block.data,
                                        inner_dim,
                                        T(1),
                                        c_val,
                                        r_dim);
                                },
                                "ghost");
                        }
                    }
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
                    std::string("Hash table infinite loop detected during SpMM numeric phase (") + phase + ")");
            }
        }
    }
};

} // namespace detail

#endif // VBCSR_DETAIL_LEGACY_SPMM_HPP

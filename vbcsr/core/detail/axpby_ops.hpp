#ifndef VBCSR_DETAIL_AXPBY_OPS_HPP
#define VBCSR_DETAIL_AXPBY_OPS_HPP

#include "../dist_graph.hpp"
#include "bsr_result_builder.hpp"
#include "csr_result_builder.hpp"
#include "distributed_result_graph.hpp"
#include "vbcsr_result_builder.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <type_traits>
#include <vector>

namespace vbcsr::detail {

template <typename Matrix>
struct CSRAxpbyExecutor {
    using T = typename Matrix::value_type;

    static void run(Matrix& self, const Matrix& X, T alpha, T beta) {
        Matrix::ensure_csr_binary_compatibility(self, X);

        const bool local_same_structure = (self.row_ptr() == X.row_ptr() && self.col_ind() == X.col_ind());
        int same_structure_flag = local_same_structure ? 1 : 0;
        if (self.graph->size > 1) {
            MPI_Allreduce(MPI_IN_PLACE, &same_structure_flag, 1, MPI_INT, MPI_MIN, self.graph->comm);
        }
        const bool same_structure = (same_structure_flag == 1);
        if (same_structure) {
            #pragma omp parallel for
            for (size_t slot = 0; slot < self.local_block_nnz(); ++slot) {
                T* dest = self.mutable_block_data(static_cast<int>(slot));
                const T* src = X.block_data(static_cast<int>(slot));
                *dest = alpha * (*src) + beta * (*dest);
            }
            self.norms_valid = false;
            return;
        }

        if (beta == T(0)) {
            DistGraph* new_graph = X.graph->duplicate();
            CSRResultBuilder<T> builder(new_graph, X.backend_page_settings().csr_page_size);
            #pragma omp parallel for
            for (size_t slot = 0; slot < X.local_block_nnz(); ++slot) {
                *builder.slot_data(static_cast<int>(slot)) = alpha * (*X.block_data(static_cast<int>(slot)));
            }
            self.template replace_with_builder<vbcsr::MatrixKind::CSR>(
                new_graph,
                true,
                std::move(builder),
                X.backend_page_settings());
            return;
        }

        const int n_rows = static_cast<int>(self.row_ptr().size()) - 1;
        std::vector<std::vector<int>> new_adj(n_rows);
        #pragma omp parallel for
        for (int row = 0; row < n_rows; ++row) {
            auto& cols = new_adj[row];
            cols.reserve((self.row_ptr()[row + 1] - self.row_ptr()[row]) + (X.row_ptr()[row + 1] - X.row_ptr()[row]));
            for (int slot = self.row_ptr()[row]; slot < self.row_ptr()[row + 1]; ++slot) {
                cols.push_back(self.graph->get_global_index(self.col_ind()[slot]));
            }
            for (int slot = X.row_ptr()[row]; slot < X.row_ptr()[row + 1]; ++slot) {
                cols.push_back(X.graph->get_global_index(X.col_ind()[slot]));
            }
            std::sort(cols.begin(), cols.end());
            cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        }

        DistGraph* new_graph = new DistGraph(self.graph->comm);
        new_graph->construct_distributed(
            self.graph->owned_global_indices,
            owned_block_sizes(*self.graph),
            new_adj);

        CSRResultBuilder<T> builder(new_graph, self.backend_page_settings().csr_page_size);
        #pragma omp parallel for
        for (int row = 0; row < n_rows; ++row) {
            std::map<int, T> row_values;
            for (int slot = self.row_ptr()[row]; slot < self.row_ptr()[row + 1]; ++slot) {
                row_values[self.graph->get_global_index(self.col_ind()[slot])] = beta * (*self.block_data(slot));
            }
            for (int slot = X.row_ptr()[row]; slot < X.row_ptr()[row + 1]; ++slot) {
                row_values[X.graph->get_global_index(X.col_ind()[slot])] += alpha * (*X.block_data(slot));
            }
            for (const auto& [global_col, value] : row_values) {
                const int local_col = new_graph->global_to_local.at(global_col);
                const int dest_slot = builder.find_slot(row, local_col);
                *builder.slot_data(dest_slot) = value;
            }
        }

        self.template replace_with_builder<vbcsr::MatrixKind::CSR>(
            new_graph,
            true,
            std::move(builder),
            self.backend_page_settings());
    }
};

template <typename Matrix>
struct BSRAxpbyExecutor {
    using T = typename Matrix::value_type;

    static void run(Matrix& self, const Matrix& X, T alpha, T beta) {
        Matrix::ensure_bsr_binary_compatibility(self, X);

        const bool local_same_structure = (self.row_ptr() == X.row_ptr() && self.col_ind() == X.col_ind());
        int same_structure_flag = local_same_structure ? 1 : 0;
        if (self.graph->size > 1) {
            MPI_Allreduce(MPI_IN_PLACE, &same_structure_flag, 1, MPI_INT, MPI_MIN, self.graph->comm);
        }
        const bool same_structure = (same_structure_flag == 1);
        if (same_structure) {
            #pragma omp parallel for
            for (size_t slot = 0; slot < self.local_block_nnz(); ++slot) {
                T* dest = self.mutable_block_data(static_cast<int>(slot));
                const T* src = X.block_data(static_cast<int>(slot));
                const size_t size = self.block_size_elements(static_cast<int>(slot));
                for (size_t idx = 0; idx < size; ++idx) {
                    dest[idx] = alpha * src[idx] + beta * dest[idx];
                }
            }
            self.norms_valid = false;
            return;
        }

        if (beta == T(0)) {
            DistGraph* new_graph = X.graph->duplicate();
            BSRResultBuilder<T> builder(new_graph, X.backend_page_settings().bsr_page_size);
            #pragma omp parallel for
            for (size_t slot = 0; slot < X.local_block_nnz(); ++slot) {
                T* dest = builder.slot_data(static_cast<int>(slot));
                const T* src = X.block_data(static_cast<int>(slot));
                const size_t size = X.block_size_elements(static_cast<int>(slot));
                for (size_t idx = 0; idx < size; ++idx) {
                    dest[idx] = alpha * src[idx];
                }
            }
            self.template replace_with_builder<vbcsr::MatrixKind::BSR>(
                new_graph,
                true,
                std::move(builder),
                X.backend_page_settings());
            return;
        }

        const int n_rows = static_cast<int>(self.row_ptr().size()) - 1;
        std::vector<std::vector<int>> new_adj(n_rows);
        #pragma omp parallel for
        for (int row = 0; row < n_rows; ++row) {
            auto& cols = new_adj[row];
            cols.reserve((self.row_ptr()[row + 1] - self.row_ptr()[row]) + (X.row_ptr()[row + 1] - X.row_ptr()[row]));
            for (int slot = self.row_ptr()[row]; slot < self.row_ptr()[row + 1]; ++slot) {
                cols.push_back(self.graph->get_global_index(self.col_ind()[slot]));
            }
            for (int slot = X.row_ptr()[row]; slot < X.row_ptr()[row + 1]; ++slot) {
                cols.push_back(X.graph->get_global_index(X.col_ind()[slot]));
            }
            std::sort(cols.begin(), cols.end());
            cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        }

        DistGraph* new_graph = new DistGraph(self.graph->comm);
        new_graph->construct_distributed(
            self.graph->owned_global_indices,
            owned_block_sizes(*self.graph),
            new_adj);

        BSRResultBuilder<T> builder(new_graph, self.backend_page_settings().bsr_page_size);
        #pragma omp parallel for
        for (int row = 0; row < n_rows; ++row) {
            for (int slot = self.row_ptr()[row]; slot < self.row_ptr()[row + 1]; ++slot) {
                const int local_col = new_graph->global_to_local.at(self.graph->get_global_index(self.col_ind()[slot]));
                const int dest_slot = builder.find_slot(row, local_col);
                T* dest = builder.slot_data(dest_slot);
                const T* src = self.block_data(slot);
                const size_t size = self.block_size_elements(slot);
                for (size_t idx = 0; idx < size; ++idx) {
                    dest[idx] = beta * src[idx];
                }
            }

            for (int slot = X.row_ptr()[row]; slot < X.row_ptr()[row + 1]; ++slot) {
                const int local_col = new_graph->global_to_local.at(X.graph->get_global_index(X.col_ind()[slot]));
                const int dest_slot = builder.find_slot(row, local_col);
                T* dest = builder.slot_data(dest_slot);
                const T* src = X.block_data(slot);
                const size_t size = X.block_size_elements(slot);
                for (size_t idx = 0; idx < size; ++idx) {
                    dest[idx] += alpha * src[idx];
                }
            }
        }

        self.template replace_with_builder<vbcsr::MatrixKind::BSR>(
            new_graph,
            true,
            std::move(builder),
            self.backend_page_settings());
    }
};

template <typename Matrix>
struct VBCSRAxpbyExecutor {
    using T = typename Matrix::value_type;

    static void run(Matrix& self, const Matrix& X, T alpha, T beta) {
        if (beta == T(0)) {
            if (self.graph == X.graph) {
                copy_scaled_structure(self, X, alpha);
                return;
            }

            const bool same_structure = (self.row_ptr() == X.row_ptr() && self.col_ind() == X.col_ind());
            if (same_structure) {
                copy_scaled_structure(self, X, alpha);
                return;
            }

            self = X.duplicate();
            self.scale(alpha);
            return;
        }

        bool same_graph = (self.graph == X.graph);
        bool same_structure = false;
        if (same_graph) {
            if (self.row_ptr().size() == X.row_ptr().size() && self.col_ind().size() == X.col_ind().size()) {
                if (self.row_ptr() == X.row_ptr() && self.col_ind() == X.col_ind()) {
                    same_structure = true;
                }
            }
        }

        if (same_structure) {
            #pragma omp parallel for
            for (size_t i = 0; i < self.local_block_nnz(); ++i) {
                T* block = self.mutable_block_data(static_cast<int>(i));
                const T* block_x = X.block_data(static_cast<int>(i));
                const size_t size = self.block_size_elements(static_cast<int>(i));
                for (size_t j = 0; j < size; ++j) {
                    block[j] = alpha * block_x[j] + beta * block[j];
                }
            }
            self.norms_valid = false;
            return;
        }

        const int n_rows = static_cast<int>(self.row_ptr().size()) - 1;
        if (static_cast<int>(X.row_ptr().size()) - 1 != n_rows) {
            throw std::runtime_error("Matrix row count mismatch in axpby");
        }

        const int x_n_owned = static_cast<int>(X.graph->owned_global_indices.size());
        const int x_n_ghost = static_cast<int>(X.graph->ghost_global_indices.size());
        const int x_total_cols = x_n_owned + x_n_ghost;

        std::vector<int> x_to_this(x_total_cols, -1);
        bool x_is_subset = true;

        for (int i = 0; i < x_n_owned; ++i) {
            const int gid = X.graph->owned_global_indices[i];
            if (self.graph->global_to_local.count(gid)) {
                x_to_this[i] = self.graph->global_to_local.at(gid);
            } else {
                x_is_subset = false;
            }
        }
        for (int i = 0; i < x_n_ghost; ++i) {
            const int gid = X.graph->ghost_global_indices[i];
            if (self.graph->global_to_local.count(gid)) {
                x_to_this[x_n_owned + i] = self.graph->global_to_local.at(gid);
            } else {
                x_is_subset = false;
            }
        }

        int local_subset = x_is_subset ? 1 : 0;
        int global_subset = 0;
        if (self.graph->size > 1) {
            MPI_Allreduce(&local_subset, &global_subset, 1, MPI_INT, MPI_MIN, self.graph->comm);
        } else {
            global_subset = local_subset;
        }
        x_is_subset = (global_subset == 1);

        if (x_is_subset) {
            bool sparsity_subset = true;
            #pragma omp parallel for reduction(&&:sparsity_subset)
            for (int row = 0; row < n_rows; ++row) {
                if (!sparsity_subset) {
                    continue;
                }
                int y_start = self.row_ptr()[row];
                const int y_end = self.row_ptr()[row + 1];
                const int x_start = X.row_ptr()[row];
                const int x_end = X.row_ptr()[row + 1];

                int y_k = y_start;
                for (int x_k = x_start; x_k < x_end; ++x_k) {
                    const int x_col_local = X.col_ind()[x_k];
                    const int target_col = x_to_this[x_col_local];

                    while (y_k < y_end && self.col_ind()[y_k] < target_col) {
                        ++y_k;
                    }
                    if (y_k == y_end || self.col_ind()[y_k] != target_col) {
                        sparsity_subset = false;
                        break;
                    }
                }
            }

            int local_ss = sparsity_subset ? 1 : 0;
            int global_ss = 0;
            if (self.graph->size > 1) {
                MPI_Allreduce(&local_ss, &global_ss, 1, MPI_INT, MPI_MIN, self.graph->comm);
            } else {
                global_ss = local_ss;
            }
            if (global_ss == 1) {
                self.scale(beta);
                #pragma omp parallel for
                for (int row = 0; row < n_rows; ++row) {
                    int y_start = self.row_ptr()[row];
                    const int y_end = self.row_ptr()[row + 1];
                    const int x_start = X.row_ptr()[row];
                    const int x_end = X.row_ptr()[row + 1];

                    int y_k = y_start;
                    for (int x_k = x_start; x_k < x_end; ++x_k) {
                        const int x_col_local = X.col_ind()[x_k];
                        const int target_col = x_to_this[x_col_local];

                        while (y_k < y_end && self.col_ind()[y_k] < target_col) {
                            ++y_k;
                        }
                        T* y_ptr = self.mutable_block_data(y_k);
                        const T* x_ptr = X.block_data(x_k);
                        const int size = static_cast<int>(self.block_size_elements(y_k));

                        for (int j = 0; j < size; ++j) {
                            y_ptr[j] += alpha * x_ptr[j];
                        }
                    }
                }
                self.norms_valid = false;
                return;
            }
        }

        std::vector<std::vector<int>> new_adj(n_rows);
        #pragma omp parallel for
        for (int row = 0; row < n_rows; ++row) {
            std::vector<int>& cols = new_adj[row];
            const int y_count = self.row_ptr()[row + 1] - self.row_ptr()[row];
            const int x_count = X.row_ptr()[row + 1] - X.row_ptr()[row];
            cols.reserve(y_count + x_count);

            for (int slot = self.row_ptr()[row]; slot < self.row_ptr()[row + 1]; ++slot) {
                cols.push_back(self.graph->get_global_index(self.col_ind()[slot]));
            }
            for (int slot = X.row_ptr()[row]; slot < X.row_ptr()[row + 1]; ++slot) {
                cols.push_back(X.graph->get_global_index(X.col_ind()[slot]));
            }

            std::sort(cols.begin(), cols.end());
            cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        }

        DistGraph* new_graph = new DistGraph(self.graph->comm);
        new_graph->construct_distributed(self.graph->owned_global_indices, self.graph->block_sizes, new_adj);

        int new_total_cols = static_cast<int>(new_graph->global_to_local.size());
        if (static_cast<int>(new_graph->block_sizes.size()) < new_total_cols) {
            new_graph->block_sizes.resize(new_total_cols);
        }

        #pragma omp parallel for
        for (int idx = 0; idx < new_total_cols; ++idx) {
            const int gid = new_graph->get_global_index(idx);
            int size = 0;

            auto it_this = self.graph->global_to_local.find(gid);
            if (it_this != self.graph->global_to_local.end()) {
                size = self.graph->block_sizes[it_this->second];
            } else {
                auto it_x = X.graph->global_to_local.find(gid);
                if (it_x != X.graph->global_to_local.end()) {
                    size = X.graph->block_sizes[it_x->second];
                }
            }
            new_graph->block_sizes[idx] = size;
        }

        std::vector<int> new_row_ptr;
        std::vector<int> new_col_ind;
        new_graph->get_matrix_structure(new_row_ptr, new_col_ind);
        VBCSRResultBuilder<T, typename Matrix::KernelType> builder(
            new_graph,
            self.backend_page_settings().vbcsr_page_size);

        #pragma omp parallel for
        for (int row = 0; row < n_rows; ++row) {
            const int y_start = self.row_ptr()[row];
            const int y_end = self.row_ptr()[row + 1];
            const int x_start = X.row_ptr()[row];
            const int x_end = X.row_ptr()[row + 1];
            const int start = new_row_ptr[row];
            const int end = new_row_ptr[row + 1];

            int y_k = y_start;
            int x_k = x_start;

            for (int slot = start; slot < end; ++slot) {
                const int col = new_col_ind[slot];

                while (y_k < y_end && canonical_less(self.col_ind()[y_k], self.graph, col, new_graph)) {
                    ++y_k;
                }
                const bool in_y = (y_k < y_end && !canonical_less(col, new_graph, self.col_ind()[y_k], self.graph));

                while (x_k < x_end && canonical_less(X.col_ind()[x_k], X.graph, col, new_graph)) {
                    ++x_k;
                }
                const bool in_x = (x_k < x_end && !canonical_less(col, new_graph, X.col_ind()[x_k], X.graph));

                T* dest_ptr = builder.slot_data(slot);
                const size_t sz = static_cast<size_t>(new_graph->block_sizes[row]) *
                                  static_cast<size_t>(new_graph->block_sizes[col]);

                if (in_y) {
                    const T* y_ptr = self.block_data(y_k);
                    if (beta == T(1)) {
                        std::memcpy(dest_ptr, y_ptr, sz * sizeof(T));
                    } else {
                        for (size_t j = 0; j < sz; ++j) {
                            dest_ptr[j] = beta * y_ptr[j];
                        }
                    }
                    if (in_x) {
                        const T* x_ptr = X.block_data(x_k);
                        for (size_t j = 0; j < sz; ++j) {
                            dest_ptr[j] += alpha * x_ptr[j];
                        }
                    }
                } else if (in_x) {
                    const T* x_ptr = X.block_data(x_k);
                    for (size_t j = 0; j < sz; ++j) {
                        dest_ptr[j] = alpha * x_ptr[j];
                    }
                } else {
                    std::memset(dest_ptr, 0, sz * sizeof(T));
                }
            }
        }

        self.template replace_with_builder<vbcsr::MatrixKind::VBCSR>(
            new_graph,
            true,
            std::move(builder),
            self.backend_page_settings());
    }

private:
    static void copy_scaled_structure(Matrix& self, const Matrix& X, T alpha) {
        #pragma omp parallel for
        for (size_t i = 0; i < self.local_block_nnz(); ++i) {
            T* block = self.mutable_block_data(static_cast<int>(i));
            const T* block_x = X.block_data(static_cast<int>(i));
            const size_t size = self.block_size_elements(static_cast<int>(i));
            for (size_t j = 0; j < size; ++j) {
                block[j] = alpha * block_x[j];
            }
        }
        self.block_norms = X.block_norms;
        self.norms_valid = false;

        bool update_norm = false;
        double alpha_real = 0;
        if (X.norms_valid) {
            if (alpha == T(1)) {
                self.norms_valid = true;
            }

            if (!self.norms_valid) {
                update_norm = true;
                if constexpr (std::is_same<T, std::complex<double>>::value || std::is_same<T, std::complex<float>>::value) {
                    if (alpha.imag() != 0.0) {
                        update_norm = false;
                    } else {
                        alpha_real = alpha.real();
                    }
                } else {
                    alpha_real = alpha;
                }
            }

            if (update_norm) {
                #pragma omp parallel for
                for (size_t i = 0; i < self.block_norms.size(); ++i) {
                    self.block_norms[i] = X.block_norms[i] * alpha_real;
                }
                self.norms_valid = true;
            }
        }
    }

    static bool canonical_less(int col_local, DistGraph* graph, int col_other_local, DistGraph* graph_other) {
        const int n_owned = static_cast<int>(graph->owned_global_indices.size());
        const int n_owned_other = static_cast<int>(graph_other->owned_global_indices.size());

        const bool ghost = col_local >= n_owned;
        const bool ghost_other = col_other_local >= n_owned_other;

        if (ghost != ghost_other) {
            return !ghost;
        }

        const int gid = graph->get_global_index(col_local);
        const int gid_other = graph_other->get_global_index(col_other_local);

        if (!ghost) {
            return gid < gid_other;
        }

        const int owner = graph->find_owner(gid);
        const int owner_other = graph_other->find_owner(gid_other);
        if (owner != owner_other) {
            return owner < owner_other;
        }
        return gid < gid_other;
    }
};

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_AXPBY_OPS_HPP

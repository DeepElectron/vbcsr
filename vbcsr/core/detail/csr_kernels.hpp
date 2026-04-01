#ifndef VBCSR_DETAIL_CSR_KERNELS_HPP
#define VBCSR_DETAIL_CSR_KERNELS_HPP

#include "../dist_multivector.hpp"
#include "../dist_vector.hpp"
#include "../scalar_traits.hpp"
#include "backend_handle.hpp"

#include <algorithm>
#include <vector>

namespace vbcsr::detail {

template <typename T>
void csr_mult(DistGraph* graph, const CSRMatrixBackend<T>& backend, DistVector<T>& x, DistVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);
    x.sync_ghosts();

    const auto& row_ptr = graph->adj_ptr;
    const int n_rows = row_ptr.empty() ? 0 : static_cast<int>(row_ptr.size()) - 1;

    #pragma omp parallel for
    for (int row = 0; row < n_rows; ++row) {
        T sum = T(0);
        backend.for_each_row_segment(row_ptr, row, [&](auto page, auto) {
            for (uint32_t idx = 0; idx < page.nnz; ++idx) {
                const int col = page.cols[idx];
                sum += page.vals[idx] * x.data[graph->block_offsets[col]];
            }
        });
        y.data[graph->block_offsets[row]] = sum;
    }
}

template <typename T>
void csr_mult_dense(DistGraph* graph, const CSRMatrixBackend<T>& backend, DistMultiVector<T>& x, DistMultiVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);
    x.sync_ghosts();

    const auto& row_ptr = graph->adj_ptr;
    const int n_rows = row_ptr.empty() ? 0 : static_cast<int>(row_ptr.size()) - 1;
    const int num_vecs = x.num_vectors;
    const int x_ld = x.local_rows + x.ghost_rows;
    const int y_ld = y.local_rows + y.ghost_rows;

    #pragma omp parallel for
    for (int row = 0; row < n_rows; ++row) {
        const int row_offset = graph->block_offsets[row];
        for (int vec = 0; vec < num_vecs; ++vec) {
            y.data[vec * y_ld + row_offset] = T(0);
        }

        backend.for_each_row_segment(row_ptr, row, [&](auto page, auto) {
            for (uint32_t idx = 0; idx < page.nnz; ++idx) {
                const int col = page.cols[idx];
                const int col_offset = graph->block_offsets[col];
                const T value = page.vals[idx];
                for (int vec = 0; vec < num_vecs; ++vec) {
                    y.data[vec * y_ld + row_offset] += value * x.data[vec * x_ld + col_offset];
                }
            }
        });
    }
}

template <typename T>
void csr_mult_adjoint(DistGraph* graph, const CSRMatrixBackend<T>& backend, DistVector<T>& x, DistVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);

    std::fill(y.data.begin(), y.data.end(), T(0));

    const auto& row_ptr = graph->adj_ptr;
    const int n_rows = row_ptr.empty() ? 0 : static_cast<int>(row_ptr.size()) - 1;

    #pragma omp parallel
    {
        std::vector<T> y_local(y.data.size(), T(0));

        #pragma omp for
        for (int row = 0; row < n_rows; ++row) {
            const T x_value = x.data[graph->block_offsets[row]];
            backend.for_each_row_segment(row_ptr, row, [&](auto page, auto) {
                for (uint32_t idx = 0; idx < page.nnz; ++idx) {
                    const int col = page.cols[idx];
                    const int col_offset = graph->block_offsets[col];
                    y_local[col_offset] += ScalarTraits<T>::conjugate(page.vals[idx]) * x_value;
                }
            });
        }

        #pragma omp critical
        {
            for (size_t idx = 0; idx < y.data.size(); ++idx) {
                y.data[idx] += y_local[idx];
            }
        }
    }

    y.reduce_ghosts();
}

template <typename T>
void csr_mult_dense_adjoint(
    DistGraph* graph,
    const CSRMatrixBackend<T>& backend,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);

    std::fill(y.data.begin(), y.data.end(), T(0));

    const auto& row_ptr = graph->adj_ptr;
    const int n_rows = row_ptr.empty() ? 0 : static_cast<int>(row_ptr.size()) - 1;
    const int num_vecs = x.num_vectors;
    const int x_ld = x.local_rows + x.ghost_rows;
    const int y_ld = y.local_rows + y.ghost_rows;

    #pragma omp parallel
    {
        std::vector<T> y_local(y.data.size(), T(0));

        #pragma omp for
        for (int row = 0; row < n_rows; ++row) {
            const int row_offset = graph->block_offsets[row];
            backend.for_each_row_segment(row_ptr, row, [&](auto page, auto) {
                for (uint32_t idx = 0; idx < page.nnz; ++idx) {
                    const int col = page.cols[idx];
                    const int col_offset = graph->block_offsets[col];
                    const T value = ScalarTraits<T>::conjugate(page.vals[idx]);
                    for (int vec = 0; vec < num_vecs; ++vec) {
                        y_local[vec * y_ld + col_offset] += value * x.data[vec * x_ld + row_offset];
                    }
                }
            });
        }

        #pragma omp critical
        {
            for (size_t idx = 0; idx < y.data.size(); ++idx) {
                y.data[idx] += y_local[idx];
            }
        }
    }

    y.reduce_ghosts();
}

} // namespace vbcsr::detail

#endif // VBCSR_DETAIL_CSR_KERNELS_HPP

#ifndef VBCSR_DETAIL_BSR_KERNELS_HPP
#define VBCSR_DETAIL_BSR_KERNELS_HPP

#include "../dist_multivector.hpp"
#include "../dist_vector.hpp"
#include "../kernels.hpp"
#include "backend_handle.hpp"

#include <algorithm>
#include <cstring>
#include <vector>
#include <xmmintrin.h>

namespace vbcsr::detail {

template <typename T>
inline void bsr_block_gemv(int block_size, const T* block, const T* x, T* y, T alpha, T beta) {
    switch (block_size) {
        case 2:
            FixedBlockKernel<T, 2, 2>::gemv(block, x, y, alpha, beta);
            break;
        case 4:
            FixedBlockKernel<T, 4, 4>::gemv(block, x, y, alpha, beta);
            break;
        case 8:
            FixedBlockKernel<T, 8, 8>::gemv(block, x, y, alpha, beta);
            break;
        case 16:
            FixedBlockKernel<T, 16, 16>::gemv(block, x, y, alpha, beta);
            break;
        default:
            SmartKernel<T>::gemv(block_size, block_size, alpha, block, block_size, x, 1, beta, y, 1);
            break;
    }
}

template <typename T>
inline void bsr_block_gemm(
    int block_size,
    int num_vecs,
    const T* block,
    const T* x,
    int ldb,
    T* y,
    int ldc,
    T alpha,
    T beta) {
    switch (block_size) {
        case 2:
            FixedBlockKernel<T, 2, 2>::gemm(num_vecs, block, block_size, x, ldb, y, ldc, alpha, beta);
            break;
        case 4:
            FixedBlockKernel<T, 4, 4>::gemm(num_vecs, block, block_size, x, ldb, y, ldc, alpha, beta);
            break;
        case 8:
            FixedBlockKernel<T, 8, 8>::gemm(num_vecs, block, block_size, x, ldb, y, ldc, alpha, beta);
            break;
        case 16:
            FixedBlockKernel<T, 16, 16>::gemm(num_vecs, block, block_size, x, ldb, y, ldc, alpha, beta);
            break;
        default:
            SmartKernel<T>::gemm(block_size, num_vecs, block_size, alpha, block, block_size, x, ldb, beta, y, ldc);
            break;
    }
}

template <typename T>
inline void bsr_block_gemv_trans(int block_size, const T* block, const T* x, T* y, T alpha, T beta) {
    switch (block_size) {
        case 2:
            FixedBlockKernel<T, 2, 2>::gemv_trans(block, x, y, alpha, beta);
            break;
        case 4:
            FixedBlockKernel<T, 4, 4>::gemv_trans(block, x, y, alpha, beta);
            break;
        case 8:
            FixedBlockKernel<T, 8, 8>::gemv_trans(block, x, y, alpha, beta);
            break;
        case 16:
            FixedBlockKernel<T, 16, 16>::gemv_trans(block, x, y, alpha, beta);
            break;
        default:
            SmartKernel<T>::gemv_trans(block_size, block_size, alpha, block, block_size, x, 1, beta, y, 1);
            break;
    }
}

template <typename T>
inline void bsr_block_gemm_trans(
    int block_size,
    int num_vecs,
    const T* block,
    const T* x,
    int ldb,
    T* y,
    int ldc,
    T alpha,
    T beta) {
    switch (block_size) {
        case 2:
            FixedBlockKernel<T, 2, 2>::gemm_trans(num_vecs, block, block_size, x, ldb, y, ldc, alpha, beta);
            break;
        case 4:
            FixedBlockKernel<T, 4, 4>::gemm_trans(num_vecs, block, block_size, x, ldb, y, ldc, alpha, beta);
            break;
        case 8:
            FixedBlockKernel<T, 8, 8>::gemm_trans(num_vecs, block, block_size, x, ldb, y, ldc, alpha, beta);
            break;
        case 16:
            FixedBlockKernel<T, 16, 16>::gemm_trans(num_vecs, block, block_size, x, ldb, y, ldc, alpha, beta);
            break;
        default:
            SmartKernel<T>::gemm_trans(block_size, num_vecs, block_size, alpha, block, block_size, x, ldb, beta, y, ldc);
            break;
    }
}

template <typename T>
void bsr_mult(DistGraph* graph, const BSRMatrixBackend<T>& backend, DistVector<T>& x, DistVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);
    x.sync_ghosts();

    const auto& row_ptr = graph->adj_ptr;
    const int n_rows = row_ptr.empty() ? 0 : static_cast<int>(row_ptr.size()) - 1;
    const int block_size = backend.block_size;

    auto run_generic = [&]() {
        #pragma omp parallel for
        for (int row = 0; row < n_rows; ++row) {
            T* y_block = y.local_data() + graph->block_offsets[row];
            std::memset(y_block, 0, sizeof(T) * block_size);
            backend.for_each_row_segment(row_ptr, row, [&](auto page, auto) {
                for (uint32_t idx = 0; idx < page.nblocks; ++idx) {
                    const int col = page.cols[idx];
                    const T* block = page.vals + static_cast<size_t>(idx) * page.block_elems;
                    const T* x_block = x.data.data() + graph->block_offsets[col];
                    SmartKernel<T>::gemv(block_size, block_size, T(1), block, block_size, x_block, 1, T(1), y_block, 1);
                }
            });
        }
    };

    auto run_fixed = [&](auto block_tag) {
        constexpr int BlockSize = decltype(block_tag)::value;
        #pragma omp parallel for
        for (int row = 0; row < n_rows; ++row) {
            T* y_block = y.local_data() + graph->block_offsets[row];
            std::memset(y_block, 0, sizeof(T) * BlockSize);
            backend.for_each_row_segment(row_ptr, row, [&](auto page, auto) {
                for (uint32_t idx = 0; idx < page.nblocks; ++idx) {
                    const int col = page.cols[idx];
                    const T* block = page.vals + static_cast<size_t>(idx) * page.block_elems;
                    const T* x_block = x.data.data() + graph->block_offsets[col];
                    FixedBlockKernel<T, BlockSize, BlockSize>::gemv(block, x_block, y_block, T(1), T(1));
                }
            });
        }
    };

    switch (block_size) {
        case 2:
            run_fixed(std::integral_constant<int, 2>{});
            return;
        case 4:
            run_fixed(std::integral_constant<int, 4>{});
            return;
        case 8:
            run_fixed(std::integral_constant<int, 8>{});
            return;
        case 16:
            run_fixed(std::integral_constant<int, 16>{});
            return;
        default:
            run_generic();
            return;
    }
}

template <typename T>
void bsr_mult_dense(DistGraph* graph, const BSRMatrixBackend<T>& backend, DistMultiVector<T>& x, DistMultiVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);
    x.sync_ghosts();

    const auto& row_ptr = graph->adj_ptr;
    const int n_rows = row_ptr.empty() ? 0 : static_cast<int>(row_ptr.size()) - 1;
    const int block_size = backend.block_size;
    const int num_vecs = x.num_vectors;
    const int x_ld = x.local_rows + x.ghost_rows;
    const int y_ld = y.local_rows + y.ghost_rows;

    auto run_generic = [&]() {
        #pragma omp parallel for
        for (int row = 0; row < n_rows; ++row) {
            T* y_block = &y(graph->block_offsets[row], 0);
            bool first = true;
            backend.for_each_row_segment(row_ptr, row, [&](auto page, auto) {
                for (uint32_t idx = 0; idx < page.nblocks; ++idx) {
                    const int col = page.cols[idx];
                    const T* block = page.vals + static_cast<size_t>(idx) * page.block_elems;
                    const T* x_block = &x(graph->block_offsets[col], 0);
                    SmartKernel<T>::gemm(
                        block_size,
                        num_vecs,
                        block_size,
                        T(1),
                        block,
                        block_size,
                        x_block,
                        x_ld,
                        first ? T(0) : T(1),
                        y_block,
                        y_ld);
                    first = false;
                }
            });
            if (first) {
                for (int vec = 0; vec < num_vecs; ++vec) {
                    std::memset(y_block + vec * y_ld, 0, sizeof(T) * block_size);
                }
            }
        }
    };

    auto run_fixed = [&](auto block_tag) {
        constexpr int BlockSize = decltype(block_tag)::value;
        #pragma omp parallel for
        for (int row = 0; row < n_rows; ++row) {
            T* y_block = &y(graph->block_offsets[row], 0);
            bool first = true;
            backend.for_each_row_segment(row_ptr, row, [&](auto page, auto) {
                for (uint32_t idx = 0; idx < page.nblocks; ++idx) {
                    const int col = page.cols[idx];
                    const T* block = page.vals + static_cast<size_t>(idx) * page.block_elems;
                    const T* x_block = &x(graph->block_offsets[col], 0);
                    FixedBlockKernel<T, BlockSize, BlockSize>::gemm(
                        num_vecs,
                        block,
                        BlockSize,
                        x_block,
                        x_ld,
                        y_block,
                        y_ld,
                        T(1),
                        first ? T(0) : T(1));
                    first = false;
                }
            });
            if (first) {
                for (int vec = 0; vec < num_vecs; ++vec) {
                    std::memset(y_block + vec * y_ld, 0, sizeof(T) * BlockSize);
                }
            }
        }
    };

    switch (block_size) {
        case 2:
            run_fixed(std::integral_constant<int, 2>{});
            return;
        case 4:
            run_fixed(std::integral_constant<int, 4>{});
            return;
        case 8:
            run_fixed(std::integral_constant<int, 8>{});
            return;
        case 16:
            run_fixed(std::integral_constant<int, 16>{});
            return;
        default:
            run_generic();
            return;
    }
}

template <typename T>
void bsr_mult_adjoint(DistGraph* graph, const BSRMatrixBackend<T>& backend, DistVector<T>& x, DistVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);

    std::fill(y.data.begin(), y.data.end(), T(0));

    const auto& row_ptr = graph->adj_ptr;
    const int n_rows = row_ptr.empty() ? 0 : static_cast<int>(row_ptr.size()) - 1;
    const int block_size = backend.block_size;

    #pragma omp parallel
    {
        std::vector<T> y_local(y.data.size(), T(0));

        #pragma omp for
        for (int row = 0; row < n_rows; ++row) {
            const T* x_block = x.local_data() + graph->block_offsets[row];
            backend.for_each_row_segment(row_ptr, row, [&](auto page, auto) {
                for (uint32_t idx = 0; idx < page.nblocks; ++idx) {
                    const int col = page.cols[idx];
                    const T* block = page.vals + static_cast<size_t>(idx) * page.block_elems;
                    T* y_block = y_local.data() + graph->block_offsets[col];
                    bsr_block_gemv_trans(block_size, block, x_block, y_block, T(1), T(1));
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
void bsr_mult_dense_adjoint(
    DistGraph* graph,
    const BSRMatrixBackend<T>& backend,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);

    std::fill(y.data.begin(), y.data.end(), T(0));

    const auto& row_ptr = graph->adj_ptr;
    const int n_rows = row_ptr.empty() ? 0 : static_cast<int>(row_ptr.size()) - 1;
    const int num_vecs = x.num_vectors;
    const int block_size = backend.block_size;
    const int x_ld = x.local_rows + x.ghost_rows;
    const int y_ld = y.local_rows + y.ghost_rows;

    #pragma omp parallel
    {
        std::vector<T> y_local(y.data.size(), T(0));

        #pragma omp for
        for (int row = 0; row < n_rows; ++row) {
            const T* x_block = &x(graph->block_offsets[row], 0);
            backend.for_each_row_segment(row_ptr, row, [&](auto page, auto) {
                for (uint32_t idx = 0; idx < page.nblocks; ++idx) {
                    const int col = page.cols[idx];
                    const T* block = page.vals + static_cast<size_t>(idx) * page.block_elems;
                    T* y_block = y_local.data() + graph->block_offsets[col];
                    bsr_block_gemm_trans(block_size, num_vecs, block, x_block, x_ld, y_block, y_ld, T(1), T(1));
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

#endif // VBCSR_DETAIL_BSR_KERNELS_HPP

#include "../block_csr.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using namespace vbcsr;

namespace {

void build_banded_adjacency(int n_rows, int half_bandwidth, std::vector<std::vector<int>>& adj) {
    adj.resize(static_cast<size_t>(n_rows));
    for (int row = 0; row < n_rows; ++row) {
        const int begin = std::max(0, row - half_bandwidth);
        const int end = std::min(n_rows - 1, row + half_bandwidth);
        auto& row_adj = adj[static_cast<size_t>(row)];
        row_adj.reserve(static_cast<size_t>(end - begin + 1));
        for (int col = begin; col <= end; ++col) {
            row_adj.push_back(col);
        }
    }
}

template <typename T>
void csr_mult_native_benchmark(
    DistGraph* graph,
    const detail::CSRMatrixBackend<T>& backend,
    DistVector<T>& x,
    DistVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);
    x.sync_ghosts();

    const auto& row_ptr = graph->adj_ptr;
    const int n_rows = row_ptr.empty() ? 0 : static_cast<int>(row_ptr.size()) - 1;
    const int* block_offsets = graph->block_offsets.data();
    const T* x_data = x.local_data();
    T* y_data = y.local_data();

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < n_rows; ++row) {
        T sum = T(0);
        backend.for_each_row_slice(row_ptr, graph->adj_ind, row, [&](auto slice) {
            for (uint32_t idx = 0; idx < slice.nnz_count; ++idx) {
                sum += slice.values[idx] * x_data[block_offsets[slice.cols[idx]]];
            }
        });
        y_data[block_offsets[row]] = sum;
    }
}

template <typename T>
void csr_mult_dense_native_benchmark(
    DistGraph* graph,
    const detail::CSRMatrixBackend<T>& backend,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y) {
    x.bind_to_graph(graph);
    y.bind_to_graph(graph);
    x.sync_ghosts();

    const auto& row_ptr = graph->adj_ptr;
    const int n_rows = row_ptr.empty() ? 0 : static_cast<int>(row_ptr.size()) - 1;
    const int num_vecs = x.num_vectors;
    const int x_ld = x.local_rows + x.ghost_rows;
    const int y_ld = y.local_rows + y.ghost_rows;
    const int* block_offsets = graph->block_offsets.data();
    const T* x_data = x.data.data();
    T* y_data = y.data.data();

    #pragma omp parallel
    {
        std::vector<T> sums(static_cast<size_t>(num_vecs), T(0));

        #pragma omp for schedule(static)
        for (int row = 0; row < n_rows; ++row) {
            std::fill(sums.begin(), sums.end(), T(0));
            backend.for_each_row_slice(row_ptr, graph->adj_ind, row, [&](auto slice) {
                for (uint32_t idx = 0; idx < slice.nnz_count; ++idx) {
                    const int col_offset = block_offsets[slice.cols[idx]];
                    for (int vec = 0; vec < num_vecs; ++vec) {
                        sums[static_cast<size_t>(vec)] +=
                            slice.values[idx] * x_data[static_cast<size_t>(vec * x_ld + col_offset)];
                    }
                }
            });

            for (int vec = 0; vec < num_vecs; ++vec) {
                y_data[static_cast<size_t>(vec * y_ld + block_offsets[row])] = sums[static_cast<size_t>(vec)];
            }
        }
    }
}

template <typename Fn>
double time_kernel(int iterations, Fn&& fn) {
    const auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        fn();
    }
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int n_rows = argc > 1 ? std::atoi(argv[1]) : 20000;
    const int half_bandwidth = argc > 2 ? std::atoi(argv[2]) : 8;
    const int num_vecs = argc > 3 ? std::atoi(argv[3]) : 16;
    const int iterations = argc > 4 ? std::atoi(argv[4]) : 10;
    const uint32_t nnz_per_page = argc > 5 ? static_cast<uint32_t>(std::atoi(argv[5])) : 4096;

    std::vector<std::vector<int>> adj;
    build_banded_adjacency(n_rows, half_bandwidth, adj);

    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(n_rows, std::vector<int>(static_cast<size_t>(n_rows), 1), adj);

    detail::CSRMatrixBackend<double> backend;
    backend.initialize_structure(graph.adj_ind.size(), nnz_per_page);
    for (int slot = 0; slot < static_cast<int>(graph.adj_ind.size()); ++slot) {
        const int col = graph.adj_ind[static_cast<size_t>(slot)];
        *backend.value_ptr(slot) = 1.0 / (1.0 + std::abs(col - (slot % n_rows)));
    }

    const auto& cache = backend.ensure_vendor_cache(
        graph.adj_ptr,
        graph.adj_ind,
        static_cast<int>(graph.block_sizes.size()));

    DistVector<double> x(&graph);
    DistVector<double> y_native(&graph);
    DistVector<double> y_vendor(&graph);
    for (int i = 0; i < x.full_size(); ++i) {
        x.data[static_cast<size_t>(i)] = std::sin(0.1 * static_cast<double>(i));
    }

    DistMultiVector<double> X(&graph, num_vecs);
    DistMultiVector<double> Y_native(&graph, num_vecs);
    DistMultiVector<double> Y_vendor(&graph, num_vecs);
    for (int vec = 0; vec < num_vecs; ++vec) {
        for (int row = 0; row < X.local_rows + X.ghost_rows; ++row) {
            X(row, vec) = std::cos(0.01 * static_cast<double>(vec * (X.local_rows + X.ghost_rows) + row));
        }
    }

    const double mult_native = time_kernel(iterations, [&] {
        csr_mult_native_benchmark(&graph, backend, x, y_native);
    });
    backend.reset_vendor_launch_count();
    const double mult_dispatch = time_kernel(iterations, [&] {
        detail::csr_mult(&graph, backend, x, y_vendor);
    });

    const double dense_native = time_kernel(iterations, [&] {
        csr_mult_dense_native_benchmark(&graph, backend, X, Y_native);
    });
    backend.reset_vendor_launch_count();
    const double dense_dispatch = time_kernel(iterations, [&] {
        detail::csr_mult_dense(&graph, backend, X, Y_vendor);
    });

    double max_err_vec = 0.0;
    for (int i = 0; i < y_native.full_size(); ++i) {
        max_err_vec = std::max(max_err_vec, std::abs(y_native.data[static_cast<size_t>(i)] - y_vendor.data[static_cast<size_t>(i)]));
    }

    double max_err_dense = 0.0;
    for (size_t i = 0; i < Y_native.data.size(); ++i) {
        max_err_dense = std::max(max_err_dense, std::abs(Y_native.data[i] - Y_vendor.data[i]));
    }

    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "CSR vendor benchmark\n";
        std::cout << "  rows=" << n_rows
                  << " half_bandwidth=" << half_bandwidth
                  << " num_vecs=" << num_vecs
                  << " iterations=" << iterations
                  << " nnz_per_page=" << nnz_per_page << "\n";
        std::cout << "  detected_backend=" << backend.vendor_backend_name()
                  << " page_cache_pages=" << cache.pages.size() << "\n";
        std::cout << "  mult_native_s=" << mult_native
                  << " mult_dispatch_s=" << mult_dispatch
                  << " max_err=" << max_err_vec << "\n";
        std::cout << "  mult_dense_native_s=" << dense_native
                  << " mult_dense_dispatch_s=" << dense_dispatch
                  << " max_err=" << max_err_dense << "\n";
    }

    MPI_Finalize();
    return 0;
}

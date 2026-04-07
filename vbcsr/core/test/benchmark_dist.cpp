#include "../block_csr.hpp"
#include <iostream>
#include <vector>
#include <mpi.h>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <limits>

using namespace vbcsr;

// Use BLAS Kernel for performance
using Kernel = BLASKernel;

// Deterministic value generators
double get_mat_val(int global_row, int global_col, int r_idx, int c_idx) {
    // Value at block (global_row, global_col), element (r_idx, c_idx)
    // Simple deterministic function
    return std::sin(global_row * 1000.0 + r_idx) * std::cos(global_col * 1000.0 + c_idx);
}

double get_vec_val(int global_col, int c_idx, int vec_idx = 0) {
    return std::cos(global_col * 1000.0 + c_idx + vec_idx);
}

template <typename T>
void bsr_mult_native_benchmark(
    DistGraph* graph,
    const detail::BSRMatrixBackend<T>& backend,
    DistVector<T>& x,
    DistVector<T>& y) {
    BLASKernel::configure_native_threading();
    detail::bsr_dispatch_block_size(backend.block_size, [&](auto block_tag) {
        constexpr int BlockSize = decltype(block_tag)::value;
        detail::bsr_mult_impl<BlockSize>(graph, backend, x, y);
    });
}

template <typename T>
void bsr_mult_dense_native_benchmark(
    DistGraph* graph,
    const detail::BSRMatrixBackend<T>& backend,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y) {
    BLASKernel::configure_native_threading();
    detail::bsr_dispatch_block_size(backend.block_size, [&](auto block_tag) {
        constexpr int BlockSize = decltype(block_tag)::value;
        detail::bsr_mult_dense_impl<BlockSize>(graph, backend, x, y);
    });
}

template <typename T>
void bsr_mult_adjoint_native_benchmark(
    DistGraph* graph,
    const detail::BSRMatrixBackend<T>& backend,
    DistVector<T>& x,
    DistVector<T>& y) {
    BLASKernel::configure_native_threading();
    detail::bsr_dispatch_block_size(backend.block_size, [&](auto block_tag) {
        constexpr int BlockSize = decltype(block_tag)::value;
        detail::bsr_mult_adjoint_impl<BlockSize>(graph, backend, x, y);
    });
}

template <typename T>
void bsr_mult_dense_adjoint_native_benchmark(
    DistGraph* graph,
    const detail::BSRMatrixBackend<T>& backend,
    DistMultiVector<T>& x,
    DistMultiVector<T>& y) {
    BLASKernel::configure_native_threading();
    detail::bsr_dispatch_block_size(backend.block_size, [&](auto block_tag) {
        constexpr int BlockSize = decltype(block_tag)::value;
        detail::bsr_mult_dense_adjoint_impl<BlockSize>(graph, backend, x, y);
    });
}

double mpi_max_double(MPI_Comm comm, double value) {
    double reduced = 0.0;
    MPI_Allreduce(&value, &reduced, 1, MPI_DOUBLE, MPI_MAX, comm);
    return reduced;
}

template <typename Fn>
double benchmark_max_seconds(MPI_Comm comm, int n_iter, Fn&& fn) {
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int i = 0; i < n_iter; ++i) {
        fn();
    }
    const double elapsed = MPI_Wtime() - start;
    return mpi_max_double(comm, elapsed);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Benchmark Parameters
    int n_global_blocks = 1000; // Reduced default for quick verification
    int block_size = 50;        
    int n_vecs = 5;             
    int n_iter = 10;            
    uint32_t page_size = std::numeric_limits<uint32_t>::max();

    if (argc > 1) n_global_blocks = std::atoi(argv[1]);
    if (argc > 2) block_size = std::atoi(argv[2]);
    if (argc > 3) n_vecs = std::atoi(argv[3]);
    if (argc > 4) page_size = static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10));
    if (argc > 5) n_iter = std::atoi(argv[5]);

    if (rank == 0) {
        std::cout << "Benchmark & Verification Configuration:" << std::endl;
        std::cout << "  Ranks: " << size << std::endl;
        std::cout << "  Global Blocks: " << n_global_blocks << std::endl;
        std::cout << "  Global Scalar Dimension: " << (n_global_blocks * block_size) << std::endl;
        std::cout << "  Block Size: " << block_size << std::endl;
        std::cout << "  RHS Vectors: " << n_vecs << std::endl;
        std::cout << "  Requested Page Size: "
                  << (page_size == std::numeric_limits<uint32_t>::max() ? std::string("max") : std::to_string(page_size))
                  << std::endl;
        std::cout << "  Iterations: " << n_iter << std::endl;
    }

    // 1. Distributed Graph Construction
    // 1D Stencil: i connected to i-1, i, i+1
    
    int blocks_per_rank = n_global_blocks / size;
    int remainder = n_global_blocks % size;
    
    int my_start = rank * blocks_per_rank + std::min(rank, remainder);
    int my_count = blocks_per_rank + (rank < remainder ? 1 : 0);
    int my_end = my_start + my_count;

    std::vector<int> my_owned_indices(my_count);
    std::vector<int> my_block_sizes(my_count, block_size);
    std::vector<std::vector<int>> my_adj(my_count);

    for (int i = 0; i < my_count; ++i) {
        int gid = my_start + i;
        my_owned_indices[i] = gid;
        
        // Neighbors
        if (gid > 0) my_adj[i].push_back(gid - 1);
        my_adj[i].push_back(gid); // Self
        if (gid < n_global_blocks - 1) my_adj[i].push_back(gid + 1);
    }

    double t0 = MPI_Wtime();
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_distributed(my_owned_indices, my_block_sizes, my_adj);
    double t_graph = MPI_Wtime() - t0;
    
    if (rank == 0) std::cout << "Graph Construction Time: " << t_graph << " s" << std::endl;

    // 2. Backend Assembly
    detail::BSRMatrixBackend<double> backend;
    backend.initialize_structure(graph.adj_ind.size(), block_size, page_size);
    
    // Fill with deterministic values
    int n_owned = graph.owned_global_indices.size();
    
    t0 = MPI_Wtime();
    for (int i = 0; i < n_owned; ++i) {
        int gid_r = graph.owned_global_indices[i];
        int start = graph.adj_ptr[i];
        int end = graph.adj_ptr[i + 1];
        for (int k = start; k < end; ++k) {
            double* data = backend.block_ptr(k);
            int lid_c = graph.adj_ind[k];
            
            // Resolve GID for column
            int gid_c;
            if (lid_c < n_owned) {
                gid_c = graph.owned_global_indices[lid_c];
            } else {
                gid_c = graph.ghost_global_indices[lid_c - n_owned];
            }
            
            int r_dim = block_size;
            int c_dim = block_size; // Uniform
            
            for (int c = 0; c < c_dim; ++c) {
                for (int r = 0; r < r_dim; ++r) {
                    data[c * r_dim + r] = get_mat_val(gid_r, gid_c, r, c);
                }
            }
        }
    }
    double t_assembly = MPI_Wtime() - t0;
    if (rank == 0) std::cout << "Matrix Assembly Time: " << t_assembly << " s" << std::endl;

    const auto& cache = backend.ensure_vendor_cache(
        graph.adj_ptr,
        graph.adj_ind,
        static_cast<int>(graph.block_sizes.size()));

    if (rank == 0) {
        std::cout << "  Matrix Kind: BSR" << std::endl;
        std::cout << "  Active Blocks Per Page: " << backend.active_blocks_per_page() << std::endl;
        std::cout << "  Vendor Backend: " << backend.vendor_backend_name() << std::endl;
        std::cout << "  Vendor Batches: " << cache.batches.size() << std::endl;
    }

    const auto fill_forward_reference = [&](int gid_r, int vec_idx, std::vector<double>& y_ref) {
        std::fill(y_ref.begin(), y_ref.end(), 0.0);
        std::vector<int> neighbors;
        if (gid_r > 0) neighbors.push_back(gid_r - 1);
        neighbors.push_back(gid_r);
        if (gid_r < n_global_blocks - 1) neighbors.push_back(gid_r + 1);

        for (int gid_c : neighbors) {
            for (int c = 0; c < block_size; ++c) {
                const double x_val = get_vec_val(gid_c, c, vec_idx);
                for (int r = 0; r < block_size; ++r) {
                    y_ref[r] += get_mat_val(gid_r, gid_c, r, c) * x_val;
                }
            }
        }
    };

    const auto fill_adjoint_reference = [&](int gid_c, int vec_idx, std::vector<double>& y_ref) {
        std::fill(y_ref.begin(), y_ref.end(), 0.0);
        std::vector<int> source_rows;
        if (gid_c > 0) source_rows.push_back(gid_c - 1);
        source_rows.push_back(gid_c);
        if (gid_c < n_global_blocks - 1) source_rows.push_back(gid_c + 1);

        for (int gid_r : source_rows) {
            for (int r = 0; r < block_size; ++r) {
                const double x_val = get_vec_val(gid_r, r, vec_idx);
                for (int c = 0; c < block_size; ++c) {
                    y_ref[c] += get_mat_val(gid_r, gid_c, r, c) * x_val;
                }
            }
        }
    };

    // 3. MatVec Benchmark & Verify
    DistVector<double> x(&graph), y_vendor(&graph), y_native(&graph);
    
    // Fill x
    double* x_ptr = x.local_data();
    for (int i = 0; i < n_owned; ++i) {
        int gid = graph.owned_global_indices[i];
        for (int k = 0; k < block_size; ++k) {
            x_ptr[i * block_size + k] = get_vec_val(gid, k);
        }
    }
    y_vendor.set_constant(0.0);
    y_native.set_constant(0.0);

    detail::bsr_mult(&graph, backend, x, y_vendor);
    bsr_mult_native_benchmark(&graph, backend, x, y_native);

    double max_vendor_ref_err = 0.0;
    double max_native_ref_err = 0.0;
    double max_vendor_native_diff = 0.0;
    double* y_vendor_ptr = y_vendor.local_data();
    double* y_native_ptr = y_native.local_data();
    for (int i = 0; i < n_owned; ++i) {
        int gid_r = graph.owned_global_indices[i];
        std::vector<double> y_ref(block_size, 0.0);
        fill_forward_reference(gid_r, 0, y_ref);
        for (int r = 0; r < block_size; ++r) {
            const double vendor_val = y_vendor_ptr[i * block_size + r];
            const double native_val = y_native_ptr[i * block_size + r];
            max_vendor_ref_err = std::max(max_vendor_ref_err, std::abs(vendor_val - y_ref[r]));
            max_native_ref_err = std::max(max_native_ref_err, std::abs(native_val - y_ref[r]));
            max_vendor_native_diff = std::max(
                max_vendor_native_diff,
                std::abs(vendor_val - native_val));
        }
    }

    const double global_vendor_ref_err = mpi_max_double(MPI_COMM_WORLD, max_vendor_ref_err);
    const double global_native_ref_err = mpi_max_double(MPI_COMM_WORLD, max_native_ref_err);
    const double global_vendor_native_diff = mpi_max_double(MPI_COMM_WORLD, max_vendor_native_diff);

    if (rank == 0) {
        std::cout << "MatVec Vendor-vs-Ref Max Error: " << global_vendor_ref_err << std::endl;
        std::cout << "MatVec Native-vs-Ref Max Error: " << global_native_ref_err << std::endl;
        std::cout << "MatVec Vendor-vs-Native Max Diff: " << global_vendor_native_diff << std::endl;
        if (std::max({global_vendor_ref_err, global_native_ref_err, global_vendor_native_diff}) > 1e-12) {
            std::cout << "  VERIFICATION FAILED" << std::endl;
        }
        else std::cout << "  VERIFICATION PASSED" << std::endl;
    }

    backend.reset_vendor_launch_count();
    const double native_vec_seconds =
        benchmark_max_seconds(MPI_COMM_WORLD, n_iter, [&] {
            bsr_mult_native_benchmark(&graph, backend, x, y_native);
        });
    backend.reset_vendor_launch_count();
    const double vendor_vec_seconds =
        benchmark_max_seconds(MPI_COMM_WORLD, n_iter, [&] {
            detail::bsr_mult(&graph, backend, x, y_vendor);
        });
    const uint64_t matvec_vendor_launches = backend.get_vendor_launch_count();
    
    if (rank == 0) {
        std::cout << "MatVec Native Time (avg): " << (native_vec_seconds / n_iter) << " s" << std::endl;
        std::cout << "MatVec Vendor Time (avg): " << (vendor_vec_seconds / n_iter) << " s" << std::endl;
        std::cout << "MatVec Speedup (native/vendor): " << (native_vec_seconds / vendor_vec_seconds) << std::endl;
        double flops = (double)n_global_blocks * 3.0 * 2.0 * block_size * block_size;
        std::cout << "MatVec Native GFLOPS: " << (flops / (native_vec_seconds / n_iter)) * 1e-9 << std::endl;
        std::cout << "MatVec Vendor GFLOPS: " << (flops / (vendor_vec_seconds / n_iter)) * 1e-9 << std::endl;
        std::cout << "MatVec Vendor Launch Count: " << matvec_vendor_launches << std::endl;
    }

    // 4. MatMat Benchmark & Verify
    DistMultiVector<double> X(&graph, n_vecs);
    DistMultiVector<double> Y_vendor(&graph, n_vecs);
    DistMultiVector<double> Y_native(&graph, n_vecs);
    
    // Init X
    for (int v = 0; v < n_vecs; ++v) {
        double* col = X.col_data(v);
        for (int i = 0; i < n_owned; ++i) {
            int gid = graph.owned_global_indices[i];
            for (int k = 0; k < block_size; ++k) {
                col[i * block_size + k] = get_vec_val(gid, k, v);
            }
        }
    }
    
    detail::bsr_mult_dense(&graph, backend, X, Y_vendor);
    bsr_mult_dense_native_benchmark(&graph, backend, X, Y_native);

    max_vendor_ref_err = 0.0;
    max_native_ref_err = 0.0;
    max_vendor_native_diff = 0.0;
    for (int v = 0; v < n_vecs; ++v) {
        double* vendor_col_ptr = Y_vendor.col_data(v);
        double* native_col_ptr = Y_native.col_data(v);
        for (int i = 0; i < n_owned; ++i) {
            int gid_r = graph.owned_global_indices[i];
            std::vector<double> y_ref(block_size, 0.0);
            fill_forward_reference(gid_r, v, y_ref);
            for (int r = 0; r < block_size; ++r) {
                const double vendor_val = vendor_col_ptr[i * block_size + r];
                const double native_val = native_col_ptr[i * block_size + r];
                max_vendor_ref_err = std::max(max_vendor_ref_err, std::abs(vendor_val - y_ref[r]));
                max_native_ref_err = std::max(max_native_ref_err, std::abs(native_val - y_ref[r]));
                max_vendor_native_diff = std::max(
                    max_vendor_native_diff,
                    std::abs(vendor_val - native_val));
            }
        }
    }

    const double global_matmat_vendor_ref_err = mpi_max_double(MPI_COMM_WORLD, max_vendor_ref_err);
    const double global_matmat_native_ref_err = mpi_max_double(MPI_COMM_WORLD, max_native_ref_err);
    const double global_matmat_vendor_native_diff = mpi_max_double(MPI_COMM_WORLD, max_vendor_native_diff);

    if (rank == 0) {
        std::cout << "MatMat Vendor-vs-Ref Max Error: " << global_matmat_vendor_ref_err << std::endl;
        std::cout << "MatMat Native-vs-Ref Max Error: " << global_matmat_native_ref_err << std::endl;
        std::cout << "MatMat Vendor-vs-Native Max Diff: " << global_matmat_vendor_native_diff << std::endl;
        if (std::max({global_matmat_vendor_ref_err, global_matmat_native_ref_err, global_matmat_vendor_native_diff}) > 1e-12) {
            std::cout << "  VERIFICATION FAILED" << std::endl;
        }
        else std::cout << "  VERIFICATION PASSED" << std::endl;
    }

    backend.reset_vendor_launch_count();
    const double native_dense_seconds =
        benchmark_max_seconds(MPI_COMM_WORLD, n_iter, [&] {
            bsr_mult_dense_native_benchmark(&graph, backend, X, Y_native);
        });
    backend.reset_vendor_launch_count();
    const double vendor_dense_seconds =
        benchmark_max_seconds(MPI_COMM_WORLD, n_iter, [&] {
            detail::bsr_mult_dense(&graph, backend, X, Y_vendor);
        });
    const uint64_t matmat_vendor_launches = backend.get_vendor_launch_count();
    
    if (rank == 0) {
        std::cout << "MatMat Native Time (avg): " << (native_dense_seconds / n_iter) << " s" << std::endl;
        std::cout << "MatMat Vendor Time (avg): " << (vendor_dense_seconds / n_iter) << " s" << std::endl;
        std::cout << "MatMat Speedup (native/vendor): " << (native_dense_seconds / vendor_dense_seconds) << std::endl;
        double flops = (double)n_global_blocks * 3.0 * 2.0 * block_size * block_size * n_vecs;
        std::cout << "MatMat Native GFLOPS: " << (flops / (native_dense_seconds / n_iter)) * 1e-9 << std::endl;
        std::cout << "MatMat Vendor GFLOPS: " << (flops / (vendor_dense_seconds / n_iter)) * 1e-9 << std::endl;
        std::cout << "MatMat Vendor Launch Count: " << matmat_vendor_launches << std::endl;
    }

    // 5. Adjoint MatVec Benchmark & Verify
    DistVector<double> x_adj(&graph), y_adj_vendor(&graph), y_adj_native(&graph);
    double* x_adj_ptr = x_adj.local_data();
    for (int i = 0; i < n_owned; ++i) {
        const int gid = graph.owned_global_indices[i];
        for (int k = 0; k < block_size; ++k) {
            x_adj_ptr[i * block_size + k] = get_vec_val(gid, k, 17);
        }
    }

    detail::bsr_mult_adjoint(&graph, backend, x_adj, y_adj_vendor);
    bsr_mult_adjoint_native_benchmark(&graph, backend, x_adj, y_adj_native);

    max_vendor_ref_err = 0.0;
    max_native_ref_err = 0.0;
    max_vendor_native_diff = 0.0;
    double* y_adj_vendor_ptr = y_adj_vendor.local_data();
    double* y_adj_native_ptr = y_adj_native.local_data();
    for (int i = 0; i < n_owned; ++i) {
        const int gid_c = graph.owned_global_indices[i];
        std::vector<double> y_ref(block_size, 0.0);
        fill_adjoint_reference(gid_c, 17, y_ref);
        for (int c = 0; c < block_size; ++c) {
            const double vendor_val = y_adj_vendor_ptr[i * block_size + c];
            const double native_val = y_adj_native_ptr[i * block_size + c];
            max_vendor_ref_err = std::max(max_vendor_ref_err, std::abs(vendor_val - y_ref[c]));
            max_native_ref_err = std::max(max_native_ref_err, std::abs(native_val - y_ref[c]));
            max_vendor_native_diff = std::max(
                max_vendor_native_diff,
                std::abs(vendor_val - native_val));
        }
    }

    const double global_adj_vendor_ref_err = mpi_max_double(MPI_COMM_WORLD, max_vendor_ref_err);
    const double global_adj_native_ref_err = mpi_max_double(MPI_COMM_WORLD, max_native_ref_err);
    const double global_adj_vendor_native_diff = mpi_max_double(MPI_COMM_WORLD, max_vendor_native_diff);

    if (rank == 0) {
        std::cout << "Adjoint MatVec Vendor-vs-Ref Max Error: " << global_adj_vendor_ref_err << std::endl;
        std::cout << "Adjoint MatVec Native-vs-Ref Max Error: " << global_adj_native_ref_err << std::endl;
        std::cout << "Adjoint MatVec Vendor-vs-Native Max Diff: " << global_adj_vendor_native_diff << std::endl;
        if (std::max({global_adj_vendor_ref_err, global_adj_native_ref_err, global_adj_vendor_native_diff}) > 1e-12) {
            std::cout << "  VERIFICATION FAILED" << std::endl;
        } else {
            std::cout << "  VERIFICATION PASSED" << std::endl;
        }
    }

    backend.reset_vendor_launch_count();
    const double native_adj_vec_seconds =
        benchmark_max_seconds(MPI_COMM_WORLD, n_iter, [&] {
            bsr_mult_adjoint_native_benchmark(&graph, backend, x_adj, y_adj_native);
        });
    backend.reset_vendor_launch_count();
    const double vendor_adj_vec_seconds =
        benchmark_max_seconds(MPI_COMM_WORLD, n_iter, [&] {
            detail::bsr_mult_adjoint(&graph, backend, x_adj, y_adj_vendor);
        });
    const uint64_t adjoint_matvec_vendor_launches = backend.get_vendor_launch_count();

    if (rank == 0) {
        std::cout << "Adjoint MatVec Native Time (avg): " << (native_adj_vec_seconds / n_iter) << " s" << std::endl;
        std::cout << "Adjoint MatVec Vendor Time (avg): " << (vendor_adj_vec_seconds / n_iter) << " s" << std::endl;
        std::cout << "Adjoint MatVec Speedup (native/vendor): " << (native_adj_vec_seconds / vendor_adj_vec_seconds) << std::endl;
        const double flops = (double)n_global_blocks * 3.0 * 2.0 * block_size * block_size;
        std::cout << "Adjoint MatVec Native GFLOPS: " << (flops / (native_adj_vec_seconds / n_iter)) * 1e-9 << std::endl;
        std::cout << "Adjoint MatVec Vendor GFLOPS: " << (flops / (vendor_adj_vec_seconds / n_iter)) * 1e-9 << std::endl;
        std::cout << "Adjoint MatVec Vendor Launch Count: " << adjoint_matvec_vendor_launches << std::endl;
    }

    // 6. Adjoint MatMat Benchmark & Verify
    DistMultiVector<double> X_adj(&graph, n_vecs);
    DistMultiVector<double> Y_adj_vendor(&graph, n_vecs);
    DistMultiVector<double> Y_adj_native(&graph, n_vecs);

    for (int v = 0; v < n_vecs; ++v) {
        double* col = X_adj.col_data(v);
        for (int i = 0; i < n_owned; ++i) {
            const int gid = graph.owned_global_indices[i];
            for (int k = 0; k < block_size; ++k) {
                col[i * block_size + k] = get_vec_val(gid, k, 100 + v);
            }
        }
    }

    detail::bsr_mult_dense_adjoint(&graph, backend, X_adj, Y_adj_vendor);
    bsr_mult_dense_adjoint_native_benchmark(&graph, backend, X_adj, Y_adj_native);

    max_vendor_ref_err = 0.0;
    max_native_ref_err = 0.0;
    max_vendor_native_diff = 0.0;
    for (int v = 0; v < n_vecs; ++v) {
        double* vendor_col_ptr = Y_adj_vendor.col_data(v);
        double* native_col_ptr = Y_adj_native.col_data(v);
        for (int i = 0; i < n_owned; ++i) {
            const int gid_c = graph.owned_global_indices[i];
            std::vector<double> y_ref(block_size, 0.0);
            fill_adjoint_reference(gid_c, 100 + v, y_ref);
            for (int c = 0; c < block_size; ++c) {
                const double vendor_val = vendor_col_ptr[i * block_size + c];
                const double native_val = native_col_ptr[i * block_size + c];
                max_vendor_ref_err = std::max(max_vendor_ref_err, std::abs(vendor_val - y_ref[c]));
                max_native_ref_err = std::max(max_native_ref_err, std::abs(native_val - y_ref[c]));
                max_vendor_native_diff = std::max(
                    max_vendor_native_diff,
                    std::abs(vendor_val - native_val));
            }
        }
    }

    const double global_adj_dense_vendor_ref_err = mpi_max_double(MPI_COMM_WORLD, max_vendor_ref_err);
    const double global_adj_dense_native_ref_err = mpi_max_double(MPI_COMM_WORLD, max_native_ref_err);
    const double global_adj_dense_vendor_native_diff = mpi_max_double(MPI_COMM_WORLD, max_vendor_native_diff);

    if (rank == 0) {
        std::cout << "Adjoint MatMat Vendor-vs-Ref Max Error: " << global_adj_dense_vendor_ref_err << std::endl;
        std::cout << "Adjoint MatMat Native-vs-Ref Max Error: " << global_adj_dense_native_ref_err << std::endl;
        std::cout << "Adjoint MatMat Vendor-vs-Native Max Diff: " << global_adj_dense_vendor_native_diff << std::endl;
        if (std::max({global_adj_dense_vendor_ref_err, global_adj_dense_native_ref_err, global_adj_dense_vendor_native_diff}) > 1e-12) {
            std::cout << "  VERIFICATION FAILED" << std::endl;
        } else {
            std::cout << "  VERIFICATION PASSED" << std::endl;
        }
    }

    backend.reset_vendor_launch_count();
    const double native_adj_dense_seconds =
        benchmark_max_seconds(MPI_COMM_WORLD, n_iter, [&] {
            bsr_mult_dense_adjoint_native_benchmark(&graph, backend, X_adj, Y_adj_native);
        });
    backend.reset_vendor_launch_count();
    const double vendor_adj_dense_seconds =
        benchmark_max_seconds(MPI_COMM_WORLD, n_iter, [&] {
            detail::bsr_mult_dense_adjoint(&graph, backend, X_adj, Y_adj_vendor);
        });
    const uint64_t adjoint_matmat_vendor_launches = backend.get_vendor_launch_count();

    if (rank == 0) {
        std::cout << "Adjoint MatMat Native Time (avg): " << (native_adj_dense_seconds / n_iter) << " s" << std::endl;
        std::cout << "Adjoint MatMat Vendor Time (avg): " << (vendor_adj_dense_seconds / n_iter) << " s" << std::endl;
        std::cout << "Adjoint MatMat Speedup (native/vendor): " << (native_adj_dense_seconds / vendor_adj_dense_seconds) << std::endl;
        const double flops = (double)n_global_blocks * 3.0 * 2.0 * block_size * block_size * n_vecs;
        std::cout << "Adjoint MatMat Native GFLOPS: " << (flops / (native_adj_dense_seconds / n_iter)) * 1e-9 << std::endl;
        std::cout << "Adjoint MatMat Vendor GFLOPS: " << (flops / (vendor_adj_dense_seconds / n_iter)) * 1e-9 << std::endl;
        std::cout << "Adjoint MatMat Vendor Launch Count: " << adjoint_matmat_vendor_launches << std::endl;
    }

    MPI_Finalize();
    return 0;
}

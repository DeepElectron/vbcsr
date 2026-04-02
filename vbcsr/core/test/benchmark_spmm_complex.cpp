#include "../block_csr.hpp"

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>
#include <vector>

using namespace vbcsr;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_blocks = 2000;
    int blocks_per_row = 200;
    double threshold = 3e3;
    int iterations = 3;
    bool make_contiguous = false;

    if (argc > 1) n_blocks = std::atoi(argv[1]);
    if (argc > 2) blocks_per_row = std::atoi(argv[2]);
    if (argc > 3) threshold = std::atof(argv[3]);
    if (argc > 4) iterations = std::max(1, std::atoi(argv[4]));
    if (argc > 5) make_contiguous = std::atoi(argv[5]) != 0;

    std::mt19937 gen_struct(12345);

    std::vector<int> block_size_options = {9, 13, 15, 20};
    std::uniform_int_distribution<> size_dist(0, static_cast<int>(block_size_options.size()) - 1);

    std::vector<int> block_sizes(n_blocks);
    for (int i = 0; i < n_blocks; ++i) {
        block_sizes[i] = block_size_options[size_dist(gen_struct)];
    }

    std::vector<std::vector<int>> adj(n_blocks);
    std::uniform_int_distribution<> col_dist(0, n_blocks - 1);
    for (int row = 0; row < n_blocks; ++row) {
        std::set<int> row_cols;
        while (static_cast<int>(row_cols.size()) < blocks_per_row) {
            row_cols.insert(col_dist(gen_struct));
        }
        adj[row].assign(row_cols.begin(), row_cols.end());
    }

    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, adj);

    using T = std::complex<double>;
    BlockSpMat<T> A(&graph);
    BlockSpMat<T> B(&graph);

    std::mt19937 gen_data(12345 + rank);
    std::uniform_real_distribution<> data_dist(-1.0, 1.0);
    const int n_owned = static_cast<int>(graph.owned_global_indices.size());

    for (int local_row = 0; local_row < n_owned; ++local_row) {
        const int global_row = graph.owned_global_indices[local_row];
        for (int global_col : adj[global_row]) {
            const int rows = block_sizes[global_row];
            const int cols = block_sizes[global_col];
            std::vector<T> block(static_cast<size_t>(rows) * cols);

            for (auto& value : block) {
                value = T(data_dist(gen_data), data_dist(gen_data));
            }
            A.add_block(global_row, global_col, block.data(), rows, cols, AssemblyMode::INSERT, MatrixLayout::RowMajor);

            for (auto& value : block) {
                value = T(data_dist(gen_data), data_dist(gen_data));
            }
            B.add_block(global_row, global_col, block.data(), rows, cols, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        }
    }

    A.assemble();
    B.assemble();

    if (make_contiguous) {
        A.contiguous();
        B.contiguous();
    }

    if (rank == 0) {
        std::cout << "Starting complex SpMM benchmark..." << std::endl;
        std::cout << "  Ranks: " << size << std::endl;
        std::cout << "  Blocks: " << n_blocks << std::endl;
        std::cout << "  Blocks per row: " << blocks_per_row << std::endl;
        std::cout << "  Threshold: " << threshold << std::endl;
        std::cout << "  Iterations: " << iterations << std::endl;
        std::cout << "  A kind: " << static_cast<int>(A.matrix_kind()) << std::endl;
        std::cout << "  A contiguous: " << A.is_contiguous() << std::endl;
        std::cout << "  B contiguous: " << B.is_contiguous() << std::endl;
        std::cout << "  Strided GEMM available: " << BLASKernel::supports_strided_gemm() << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto warmup_start = std::chrono::high_resolution_clock::now();
    BlockSpMat<T> warmup = A.spmm(B, threshold);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto warmup_end = std::chrono::high_resolution_clock::now();

    long long local_blocks = static_cast<long long>(warmup.local_block_nnz());
    long long global_blocks = 0;
    MPI_Reduce(&local_blocks, &global_blocks, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    const double warmup_local_seconds = std::chrono::duration<double>(warmup_end - warmup_start).count();
    double warmup_seconds = 0.0;
    MPI_Reduce(&warmup_local_seconds, &warmup_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double total_seconds = 0.0;
    for (int iter = 0; iter < iterations; ++iter) {
        MPI_Barrier(MPI_COMM_WORLD);
        const auto t0 = std::chrono::high_resolution_clock::now();
        BlockSpMat<T> result = A.spmm(B, threshold);
        MPI_Barrier(MPI_COMM_WORLD);
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double local_seconds = std::chrono::duration<double>(t1 - t0).count();
        double iter_seconds = 0.0;
        MPI_Reduce(&local_seconds, &iter_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            total_seconds += iter_seconds;
        }
    }

    if (rank == 0) {
        std::cout << "Profiling Results (seconds):" << std::endl;
        std::cout << "  Warmup SpMM: " << warmup_seconds << " (excluded from average)" << std::endl;
        std::cout << "  Average SpMM: " << (total_seconds / iterations) << std::endl;
        std::cout << "  Result blocks: " << global_blocks << std::endl;
    }

    MPI_Finalize();
    return 0;
}

#include "../block_csr.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace vbcsr;

namespace {

double block_value(int row_gid, int col_gid, int r, int c) {
    return std::sin(0.25 * row_gid + 0.5 * col_gid + 0.125 * r - 0.0625 * c);
}

template <typename Fn>
double benchmark_seconds(Fn&& fn, int iterations) {
    const auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        fn();
    }
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count() / std::max(1, iterations);
}

double max_abs_diff(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    double max_diff = 0.0;
    const size_t n = std::min(lhs.size(), rhs.size());
    for (size_t i = 0; i < n; ++i) {
        max_diff = std::max(max_diff, std::abs(lhs[i] - rhs[i]));
    }
    return max_diff;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n_blocks = 240;
    int n_vecs = 8;
    int n_apply_iters = 20;
    int n_spmm_iters = 3;

    if (argc > 1) n_blocks = std::atoi(argv[1]);
    if (argc > 2) n_vecs = std::atoi(argv[2]);
    if (argc > 3) n_apply_iters = std::atoi(argv[3]);
    if (argc > 4) n_spmm_iters = std::atoi(argv[4]);

    std::vector<int> block_sizes(n_blocks);
    for (int i = 0; i < n_blocks; ++i) {
        block_sizes[i] = (i % 4 < 2) ? 2 : 3;
    }

    std::vector<std::vector<int>> adj(n_blocks);
    for (int row = 0; row < n_blocks; ++row) {
        std::vector<int> cols = {
            row,
            (row + 1) % n_blocks,
            (row + 2) % n_blocks,
            (row + n_blocks - 1) % n_blocks,
            (row + n_blocks - 2) % n_blocks
        };
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        adj[row] = std::move(cols);
    }

    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(n_blocks, block_sizes, adj);

    BlockSpMat<double> matrix(&graph);
    for (int row = 0; row < n_blocks; ++row) {
        for (int col : adj[row]) {
            const int rows = block_sizes[row];
            const int cols = block_sizes[col];
            std::vector<double> block(static_cast<size_t>(rows) * cols);
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    block[static_cast<size_t>(r) * cols + c] = block_value(row, col, r, c);
                }
            }
            matrix.add_block(row, col, block.data(), rows, cols, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        }
    }
    matrix.assemble();

    DistVector<double> x(&graph);
    for (size_t i = 0; i < x.data.size(); ++i) {
        x.data[i] = std::cos(0.01 * static_cast<double>(i));
    }

    DistMultiVector<double> X(&graph, n_vecs);
    // Row-major storage: element (row, vec) at data[row * ld + vec].
    const int x_total_rows = X.local_rows + X.ghost_rows;
    for (int vec = 0; vec < n_vecs; ++vec) {
        for (int i = 0; i < x_total_rows; ++i) {
            X(i, vec) = std::sin(0.02 * static_cast<double>(i) + 0.1 * vec);
        }
    }

    DistVector<double> y(&graph);
    DistMultiVector<double> Y(&graph, n_vecs);

    const double matvec_s = benchmark_seconds([&] { matrix.mult(x, y); }, n_apply_iters);
    const double matmat_s = benchmark_seconds([&] { matrix.mult_dense(X, Y); }, n_apply_iters);

    const double spmm_s = benchmark_seconds([&] {
        BlockSpMat<double> tmp = matrix.spmm_self(0.0);
        (void)tmp;
    }, n_spmm_iters);

    if (rank == 0) {
        std::cout << "Repeated-shape VBCSR benchmark" << std::endl;
        std::cout << "  Blocks: " << n_blocks << std::endl;
        std::cout << "  RHS vectors: " << n_vecs << std::endl;
        std::cout << "  Matrix kind: " << static_cast<int>(matrix.matrix_kind()) << std::endl;
        std::cout << "  Batched GEMM available: " << BLASKernel::supports_strided_gemm() << std::endl;
        std::cout << "  Batched GEMV available: " << BLASKernel::supports_strided_gemv() << std::endl;
        std::cout << "  mult avg s: " << matvec_s << std::endl;
        std::cout << "  mult_dense avg s: " << matmat_s << std::endl;
        std::cout << "  spmm_self avg s: " << spmm_s << std::endl;
    }

    MPI_Finalize();
    return 0;
}

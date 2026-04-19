#include "../block_csr.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <set>
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
            for (int c = 0; c < cols; ++c) {
                for (int r = 0; r < rows; ++r) {
                    block[static_cast<size_t>(c) * rows + r] = block_value(row, col, r, c);
                }
            }
            matrix.add_block(row, col, block.data(), rows, cols, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        }
    }
    matrix.assemble();

    DistVector<double> x(&graph);
    for (size_t i = 0; i < x.data.size(); ++i) {
        x.data[i] = std::cos(0.01 * static_cast<double>(i));
    }

    DistMultiVector<double> X(&graph, n_vecs);
    for (int vec = 0; vec < n_vecs; ++vec) {
        for (size_t i = 0; i < X.data.size() / static_cast<size_t>(n_vecs); ++i) {
            X.data[static_cast<size_t>(vec) * (X.data.size() / static_cast<size_t>(n_vecs)) + i] =
                std::sin(0.02 * static_cast<double>(i) + 0.1 * vec);
        }
    }

    DistVector<double> y(&graph);
    DistMultiVector<double> Y(&graph, n_vecs);
    auto apply_counts = [](const auto& matrix) {
        std::set<int> seen_shapes;
        std::pair<size_t, size_t> counts{0, 0};
        matrix.for_each_shape_batch([&](const auto& batch) {
            if (seen_shapes.insert(batch.shape_id).second) {
                counts.first += static_cast<size_t>(batch.scalar_apply_batch_count());
                counts.second += static_cast<size_t>(batch.batched_apply_batch_count());
            }
        });
        return counts;
    };

    const auto matvec_before = apply_counts(matrix);
    const double matvec_s = benchmark_seconds([&] { matrix.mult(x, y); }, n_apply_iters);
    const auto matvec_after = apply_counts(matrix);
    const size_t matvec_scalar = matvec_after.first - matvec_before.first;
    const size_t matvec_batched = matvec_after.second - matvec_before.second;

    const auto matmat_before = apply_counts(matrix);
    const double matmat_s = benchmark_seconds([&] { matrix.mult_dense(X, Y); }, n_apply_iters);
    const auto matmat_after = apply_counts(matrix);
    const size_t matmat_scalar = matmat_after.first - matmat_before.first;
    const size_t matmat_batched = matmat_after.second - matmat_before.second;

    const double spmm_s = benchmark_seconds([&] {
        BlockSpMat<double> tmp = matrix.spmm_self(0.0);
        (void)tmp;
    }, n_spmm_iters);

    if (rank == 0) {
        std::cout << "Repeated-shape VBCSR benchmark" << std::endl;
        std::cout << "  Blocks: " << n_blocks << std::endl;
        std::cout << "  RHS vectors: " << n_vecs << std::endl;
        std::cout << "  Matrix kind: " << static_cast<int>(matrix.matrix_kind()) << std::endl;
        std::cout << "  Batched GEMM available: " << SmartKernel<double>::supports_batched_gemm() << std::endl;
        std::cout << "  Batched GEMV available: " << SmartKernel<double>::supports_batched_gemv() << std::endl;
        std::cout << "  Auto-batched mult: " << (matvec_batched > 0) << std::endl;
        std::cout << "  Auto-batched mult_dense: " << (matmat_batched > 0) << std::endl;
        std::cout << "  mult avg s: " << matvec_s << std::endl;
        std::cout << "  mult scalar launches: " << matvec_scalar << std::endl;
        std::cout << "  mult batched launches: " << matvec_batched << std::endl;
        std::cout << "  mult_dense avg s: " << matmat_s << std::endl;
        std::cout << "  mult_dense scalar launches: " << matmat_scalar << std::endl;
        std::cout << "  mult_dense batched launches: " << matmat_batched << std::endl;
        std::cout << "  spmm_self avg s: " << spmm_s << std::endl;
    }

    MPI_Finalize();
    return 0;
}

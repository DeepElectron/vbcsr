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

    BlockSpMat<double> unpacked(&graph);
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
            unpacked.add_block(row, col, block.data(), rows, cols, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        }
    }
    unpacked.assemble();

    BlockSpMat<double> packed = unpacked.duplicate();
    packed.contiguous();

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

    DistVector<double> y_unpacked(&graph);
    DistVector<double> y_packed(&graph);
    DistMultiVector<double> Y_unpacked(&graph, n_vecs);
    DistMultiVector<double> Y_packed(&graph, n_vecs);
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

    unpacked.mult(x, y_unpacked);
    packed.mult(x, y_packed);
    unpacked.mult_dense(X, Y_unpacked);
    packed.mult_dense(X, Y_packed);

    BlockSpMat<double> spmm_unpacked = unpacked.spmm_self(0.0);
    BlockSpMat<double> spmm_packed = packed.spmm_self(0.0);

    const double matvec_diff = max_abs_diff(y_unpacked.data, y_packed.data);
    const double matmat_diff = max_abs_diff(Y_unpacked.data, Y_packed.data);
    const double spmm_diff = max_abs_diff(spmm_unpacked.to_dense(), spmm_packed.to_dense());

    const auto unpacked_matvec_before = apply_counts(unpacked);
    const double unpacked_matvec_s = benchmark_seconds([&] { unpacked.mult(x, y_unpacked); }, n_apply_iters);
    const auto unpacked_matvec_after = apply_counts(unpacked);
    const size_t unpacked_matvec_scalar = unpacked_matvec_after.first - unpacked_matvec_before.first;
    const size_t unpacked_matvec_batched = unpacked_matvec_after.second - unpacked_matvec_before.second;

    const auto packed_matvec_before = apply_counts(packed);
    const double packed_matvec_s = benchmark_seconds([&] { packed.mult(x, y_packed); }, n_apply_iters);
    const auto packed_matvec_after = apply_counts(packed);
    const size_t packed_matvec_scalar = packed_matvec_after.first - packed_matvec_before.first;
    const size_t packed_matvec_batched = packed_matvec_after.second - packed_matvec_before.second;

    const auto unpacked_matmat_before = apply_counts(unpacked);
    const double unpacked_matmat_s = benchmark_seconds([&] { unpacked.mult_dense(X, Y_unpacked); }, n_apply_iters);
    const auto unpacked_matmat_after = apply_counts(unpacked);
    const size_t unpacked_matmat_scalar = unpacked_matmat_after.first - unpacked_matmat_before.first;
    const size_t unpacked_matmat_batched = unpacked_matmat_after.second - unpacked_matmat_before.second;

    const auto packed_matmat_before = apply_counts(packed);
    const double packed_matmat_s = benchmark_seconds([&] { packed.mult_dense(X, Y_packed); }, n_apply_iters);
    const auto packed_matmat_after = apply_counts(packed);
    const size_t packed_matmat_scalar = packed_matmat_after.first - packed_matmat_before.first;
    const size_t packed_matmat_batched = packed_matmat_after.second - packed_matmat_before.second;

    const double unpacked_spmm_s = benchmark_seconds([&] {
        BlockSpMat<double> tmp = unpacked.spmm_self(0.0);
        (void)tmp;
    }, n_spmm_iters);
    const double packed_spmm_s = benchmark_seconds([&] {
        BlockSpMat<double> tmp = packed.spmm_self(0.0);
        (void)tmp;
    }, n_spmm_iters);

    if (rank == 0) {
        std::cout << "Repeated-shape VBCSR benchmark" << std::endl;
        std::cout << "  Blocks: " << n_blocks << std::endl;
        std::cout << "  RHS vectors: " << n_vecs << std::endl;
        std::cout << "  Matrix kind: " << static_cast<int>(unpacked.matrix_kind()) << std::endl;
        std::cout << "  Unpacked contiguous: " << unpacked.is_contiguous() << std::endl;
        std::cout << "  Packed contiguous: " << packed.is_contiguous() << std::endl;
        std::cout << "  Batched GEMM available: " << SmartKernel<double>::supports_batched_gemm() << std::endl;
        std::cout << "  Batched GEMV available: " << SmartKernel<double>::supports_batched_gemv() << std::endl;
        std::cout << "  Unpacked auto-batched mult: " << (unpacked_matvec_batched > 0) << std::endl;
        std::cout << "  Unpacked auto-batched mult_dense: " << (unpacked_matmat_batched > 0) << std::endl;
        std::cout << "  Max diff (mult): " << matvec_diff << std::endl;
        std::cout << "  Max diff (mult_dense): " << matmat_diff << std::endl;
        std::cout << "  Max diff (spmm_self): " << spmm_diff << std::endl;
        std::cout << "  mult unpacked avg s: " << unpacked_matvec_s << std::endl;
        std::cout << "  mult unpacked scalar launches: " << unpacked_matvec_scalar << std::endl;
        std::cout << "  mult unpacked batched launches: " << unpacked_matvec_batched << std::endl;
        std::cout << "  mult packed avg s: " << packed_matvec_s << std::endl;
        std::cout << "  mult packed scalar launches: " << packed_matvec_scalar << std::endl;
        std::cout << "  mult packed batched launches: " << packed_matvec_batched << std::endl;
        std::cout << "  mult_dense unpacked avg s: " << unpacked_matmat_s << std::endl;
        std::cout << "  mult_dense unpacked scalar launches: " << unpacked_matmat_scalar << std::endl;
        std::cout << "  mult_dense unpacked batched launches: " << unpacked_matmat_batched << std::endl;
        std::cout << "  mult_dense packed avg s: " << packed_matmat_s << std::endl;
        std::cout << "  mult_dense packed scalar launches: " << packed_matmat_scalar << std::endl;
        std::cout << "  mult_dense packed batched launches: " << packed_matmat_batched << std::endl;
        std::cout << "  spmm_self unpacked avg s: " << unpacked_spmm_s << std::endl;
        std::cout << "  spmm_self packed avg s: " << packed_spmm_s << std::endl;
    }

    MPI_Finalize();
    return 0;
}

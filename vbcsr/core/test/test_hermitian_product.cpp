#include "../block_csr.hpp"
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

using namespace vbcsr;
using cd = std::complex<double>;

namespace {

// Global matrix: two 2x2 blocks (4x4 scalars), dense block structure so the
// distributed transpose genuinely exchanges off-diagonal blocks between ranks.
// A = [[ 1   2i   5   6  ]
//      [ 3   4    7i   8  ]
//      [ 9   10   13   14i]
//      [ 11  12i  15   16 ]]
const cd kA[4][4] = {
    {{1, 0}, {0, 2}, {5, 0}, {6, 0}},
    {{3, 0}, {4, 0}, {0, 7}, {8, 0}},
    {{9, 0}, {10, 0}, {13, 0}, {0, 14}},
    {{11, 0}, {0, 12}, {15, 0}, {16, 0}},
};

cd adjoint_entry(int r, int c) { return std::conj(kA[c][r]); }

// Value at (owned-local scalar row, global scalar col) of the local dense
// view, resolving the column offset through the matrix graph so the check is
// independent of local ghost ordering. Structurally absent blocks read 0.
template <typename Mat>
cd dense_entry(const Mat& m, const std::vector<cd>& dense, int local_row, int g_col) {
    const int g_block = g_col / 2;
    const auto it = m.graph->global_to_local.find(g_block);
    if (it == m.graph->global_to_local.end()) return cd(0, 0);
    const int my_cols = m.graph->block_offsets.back();
    const int col = m.graph->block_offsets[it->second] + (g_col % 2);
    return dense[local_row * my_cols + col];
}

template <typename Mat>
int check_adjoint(const Mat& m, int first_global_row, int n_rows, const char* label, int rank) {
    const std::vector<cd> dense = m.to_dense();
    int failures = 0;
    for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < 4; ++c) {
            const cd got = dense_entry(m, dense, r, c);
            const cd expected = adjoint_entry(first_global_row + r, c);
            if (std::abs(got - expected) > 1e-9) {
                std::cout << "[rank " << rank << "] " << label << " mismatch at ("
                          << first_global_row + r << "," << c << "): got " << got
                          << " expected " << expected << std::endl;
                ++failures;
            }
        }
    }
    if (failures == 0) {
        std::cout << "[rank " << rank << "] " << label << " OK" << std::endl;
    }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Rank r owns global block row r; at np=1 rank 0 owns both blocks.
    // Ownership must match construct_distributed's positional partition
    // (rank r owns global ids [displs[r], displs[r+1])).
    std::vector<int> owned, sizes;
    std::vector<std::vector<int>> adj;
    if (size == 1) {
        owned = {0, 1};
        sizes = {2, 2};
        adj = {{0, 1}, {0, 1}};
    } else if (rank < 2) {
        owned = {rank};
        sizes = {2};
        adj = {{0, 1}};
    }

    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_distributed(owned, sizes, adj);

    BlockSpMat<cd> A(&graph);
    for (int gb : owned) {
        for (int cb = 0; cb < 2; ++cb) {
            cd block[4]; // RowMajor 2x2 (canonical block layout)
            for (int r = 0; r < 2; ++r)
                for (int c = 0; c < 2; ++c)
                    block[r * 2 + c] = kA[2 * gb + r][2 * cb + c];
            A.add_block(gb, cb, block, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        }
    }
    A.assemble();

    int failures = 0;
    const int first_row = owned.empty() ? 0 : owned.front() * 2;
    const int n_local_rows = static_cast<int>(owned.size()) * 2;

    // Test 1: A.transpose() must equal the conjugate transpose of A.
    auto AH = A.transpose();
    failures += check_adjoint(AH, first_row, n_local_rows, "transpose", rank);

    // Test 2: C = A^H * I via spmm(transA) must equal A^H as well.
    BlockSpMat<cd> I(&graph);
    const cd eye[4] = {{1, 0}, {0, 0}, {0, 0}, {1, 0}};
    for (int gb : owned) {
        I.add_block(gb, gb, eye, 2, 2, AssemblyMode::INSERT);
    }
    I.assemble();

    auto C = A.spmm(I, 0.0, true, false);
    failures += check_adjoint(C, first_row, n_local_rows, "spmm transA", rank);

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << (global_failures ? "Hermitian product FAILED" : "Hermitian product OK")
                  << std::endl;
    }

    MPI_Finalize();
    return global_failures > 0 ? 1 : 0;
}

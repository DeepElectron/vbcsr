// release_values_below_row must actually return memory, on every matrix kind.
//
// The operation is allowed to be a no-op -- a heap-backed buffer cannot give
// anything back, and neither can a prefix shorter than an OS page -- so a
// broken implementation and a legitimately-quiet one look identical from the
// outside. That is not hypothetical here: the design note records an earlier
// whole-page-only release that "never fired" because BSR puts the whole matrix
// in one page, and nothing noticed. So this asserts BYTES, on a matrix big
// enough that a real release cannot come out empty.
//
// The three kinds reach three different implementations behind one contract:
// CSR and BSR convert the row to an index into their single value array, VBCSR
// has to release a prefix of every shape's buffer separately and rounds down to
// a thread-domain edge. The point of the test is that the caller says "row" and
// all three answer.
#include "vbcsr/core/block_csr.hpp"

#include <complex>
#include <cstdio>
#include <numeric>
#include <vector>

using namespace vbcsr;
using T = std::complex<double>;

namespace {

// Rows x degree blocks, with `sizes` giving each row's block dimension. One
// distinct size makes BSR (or CSR at size 1); several make VBCSR.
BlockSpMat<T> build(DistGraph& graph, int n_rows, int degree,
                    const std::vector<int>& sizes) {
    std::vector<std::vector<int>> adj(static_cast<size_t>(n_rows));
    for (int r = 0; r < n_rows; ++r) {
        for (int k = 0; k < degree; ++k) {
            adj[static_cast<size_t>(r)].push_back((r + k) % n_rows);
        }
        std::sort(adj[static_cast<size_t>(r)].begin(), adj[static_cast<size_t>(r)].end());
    }
    graph.construct_serial(n_rows, sizes, adj);
    BlockSpMat<T> m(&graph);
    m.fill_random();
    return m;
}

int check(const char* label, MatrixKind expect, int n_rows, int degree,
          const std::vector<int>& sizes) {
    DistGraph graph(MPI_COMM_SELF);
    BlockSpMat<T> m = build(graph, n_rows, degree, sizes);
    int failures = 0;

    if (m.matrix_kind() != expect) {
        std::printf("  %-8s FAILED: expected a different backend kind\n", label);
        return 1;
    }

    // Nothing below row 0, and nothing for a negative row.
    if (m.release_values_below_row(0) != 0 || m.release_values_below_row(-5) != 0) {
        std::printf("  %-8s FAILED: released something below row 0\n", label);
        ++failures;
    }

    // Half the matrix. Any correct implementation returns a large fraction of
    // half the payload; VBCSR rounds down to a domain edge, so the assertion is
    // "a substantial part of half", not an exact figure.
    const uint64_t total_bytes =
        static_cast<uint64_t>(m.local_scalar_nnz()) * sizeof(T);
    const uint64_t released = m.release_values_below_row(n_rows / 2);
    const double fraction = static_cast<double>(released) / static_cast<double>(total_bytes);
    std::printf("  %-8s released %8.1f MB of %8.1f MB payload (%.0f%% at the halfway row)\n",
                label, released / 1048576.0, total_bytes / 1048576.0,
                100.0 * fraction);
    if (released == 0) {
        std::printf("  %-8s FAILED: released nothing\n", label);
        ++failures;
    }
    // Well under half would mean the prefix was mostly missed. A domain edge
    // costs at most one domain, so this is loose enough for any thread count
    // and still fails a release that only found one page.
    if (fraction < 0.25) {
        std::printf("  %-8s FAILED: released far less than the half asked for\n", label);
        ++failures;
    }

    // Blocks AT OR ABOVE the boundary must still be readable and unchanged --
    // the release hands back pages, it does not shrink the matrix.
    const DistGraph& g = *m.graph;
    double checksum = 0.0;
    for (int row = n_rows / 2; row < n_rows; ++row) {
        for (int k = g.adj_ptr[row]; k < g.adj_ptr[row + 1]; ++k) {
            const T* block = m.block_data(k);
            checksum += std::abs(block[0]);
        }
    }
    if (!(checksum > 0.0)) {
        std::printf("  %-8s FAILED: rows above the boundary read as empty\n", label);
        ++failures;
    }
    return failures;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int failures = 0;

    // Big enough that a release cannot be lost in page rounding: ~1 GB of
    // payload per case at these settings.
    const int n_rows = 2048;
    const int degree = 200;

    failures += check("CSR", MatrixKind::CSR, n_rows, degree,
                      std::vector<int>(n_rows, 1));
    failures += check("BSR", MatrixKind::BSR, n_rows, degree,
                      std::vector<int>(n_rows, 13));

    // Several distinct block sizes, so the store packs by shape and the row
    // prefix is a prefix of every shape's buffer rather than of one array.
    std::vector<int> mixed(static_cast<size_t>(n_rows));
    for (int r = 0; r < n_rows; ++r) {
        mixed[static_cast<size_t>(r)] = 9 + (r % 5);
    }
    failures += check("VBCSR", MatrixKind::VBCSR, n_rows, degree, mixed);

    std::printf(failures == 0 ? "test_release_below_row PASSED\n"
                              : "test_release_below_row FAILED\n");
    MPI_Finalize();
    return failures == 0 ? 0 : 1;
}

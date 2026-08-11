// The two fused upper-triangle kernels against a dense reference.
//
// Both stage a row through an accumulator that hands its columns back in
// FIRST-TOUCH order while the result graph is built in COLUMN order. Those two
// orders disagreeing does not throw and does not produce garbage -- it writes
// each block onto a neighbouring column, which surfaces to a caller as a
// Newton-Schulz iteration that converges more slowly than it should. Nothing
// short of a value-by-value reference catches it.
//
// vbcsr had no coverage of either kernel; rsatb's calc/test/test_rarh.cpp and
// test_square_polynomial.cpp do cover them, from outside the library that owns
// them. This is not a copy of those. Theirs uses 6 uniform blocks, which at 4
// ranks leaves at most 2 rows per rank -- few enough that touch order coincides
// with column order and the check goes blind: with the ordering deliberately
// removed, test_rarh fails at 1 and 2 ranks and PASSES at 4. This one fails at
// 1, 2, 4 and 5.
//
// The difference is the inputs, which are deliberately awkward:
//   * VARIED block sizes, so a misplaced block is a size mismatch and not just
//     wrong numbers, and so the row's value offsets are not a stride.
//   * Adjacency listed OUT OF ORDER per row, since first-touch order follows
//     the order the pattern is walked in.
//   * Run at whatever rank count it is given. One rank exercises run_local; two
//     or more exercise run_distributed, where the columns a row reaches come
//     back from two rounds of exchange and the order they arrive in has nothing
//     to do with the order the result needs.
//   * Every check run TWICE above one rank: once over a balanced partition, and
//     once over one that leaves half the ranks owning no rows at all. The
//     starved shape is what any run with more ranks than block rows produces,
//     and it used to hang the job outright.
//
// Operand values come from a hash of (block, entry) rather than from a
// generator, so every rank derives the same matrix for the blocks it owns
// without agreeing on anything.
#include "../block_csr.hpp"

#include <complex>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace vbcsr;

namespace {

constexpr int kBlocks = 9;
using T = std::complex<double>;

uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

double unit(uint64_t h) {
    return 2.0 * (static_cast<double>(h >> 11) / 9007199254740992.0) - 1.0;
}

// Entry (a,b) of the upper-triangle block (p,q), p <= q, of matrix `which`.
T raw(int which, int p, int q, int a, int b) {
    const uint64_t h = mix(mix(mix(static_cast<uint64_t>(which) * 1000003u +
                                   static_cast<uint64_t>(p) * 1009u +
                                   static_cast<uint64_t>(q)) +
                               static_cast<uint64_t>(a)) +
                           static_cast<uint64_t>(b) * 7919u);
    return T(unit(h), unit(mix(h)));
}

// Entry (a,b) of block (i,j) of a HERMITIAN matrix built from `raw`.
T hermitian_entry(int which, int i, int j, int a, int b) {
    if (i > j) return std::conj(hermitian_entry(which, j, i, b, a));
    if (i < j) return raw(which, i, j, a, b);
    if (a > b) return std::conj(raw(which, i, i, b, a));
    if (a < b) return raw(which, i, i, a, b);
    return T(raw(which, i, i, a, a).real(), 0.0);
}

// Dense n x n image of the blocks THIS RANK owns; zero elsewhere, so a sum
// across ranks is the whole matrix.
std::vector<T> densify(const BlockSpMat<T>& m, const std::vector<int>& sizes,
                       const std::vector<int>& offsets, int n) {
    std::vector<T> dense(static_cast<size_t>(n) * n, T(0));
    const DistGraph& g = *m.graph;
    const int rows = static_cast<int>(g.adj_ptr.size()) - 1;
    for (int i = 0; i < rows; ++i) {
        const int g_row = g.get_global_index(i);
        for (int k = g.adj_ptr[i]; k < g.adj_ptr[i + 1]; ++k) {
            const int g_col = g.get_global_index(g.adj_ind[k]);
            const T* block = m.block_data(k);
            const int r_dim = sizes[static_cast<size_t>(g_row)];
            const int c_dim = sizes[static_cast<size_t>(g_col)];
            for (int r = 0; r < r_dim; ++r) {
                for (int c = 0; c < c_dim; ++c) {
                    dense[static_cast<size_t>(offsets[g_row] + r) * n + offsets[g_col] + c] =
                        block[static_cast<size_t>(r) * c_dim + c];
                }
            }
        }
    }
    return dense;
}

void sum_over_ranks(std::vector<T>& dense) {
    MPI_Allreduce(MPI_IN_PLACE, dense.data(), static_cast<int>(dense.size()),
                  MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
}

std::vector<T> dense_mul(const std::vector<T>& a, const std::vector<T>& b, int n) {
    std::vector<T> c(static_cast<size_t>(n) * n, T(0));
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            const T aik = a[static_cast<size_t>(i) * n + k];
            if (aik == T(0)) continue;
            for (int j = 0; j < n; ++j) {
                c[static_cast<size_t>(i) * n + j] += aik * b[static_cast<size_t>(k) * n + j];
            }
        }
    }
    return c;
}

std::vector<T> dense_adjoint(const std::vector<T>& a, int n) {
    std::vector<T> t(static_cast<size_t>(n) * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            t[static_cast<size_t>(i) * n + j] = std::conj(a[static_cast<size_t>(j) * n + i]);
        }
    }
    return t;
}

double max_abs_diff(const std::vector<T>& x, const std::vector<T>& y) {
    double worst = 0.0;
    for (size_t i = 0; i < x.size(); ++i) worst = std::max(worst, std::abs(x[i] - y[i]));
    return worst;
}

// Block sizes chosen so no two neighbours are equal: a block written to the
// wrong column is then also the wrong shape.
const std::vector<int>& block_sizes() {
    static const std::vector<int> sizes = {2, 3, 1, 4, 2, 3, 1, 2, 3};
    return sizes;
}

// Symmetric pattern, since both operands must be Hermitian. Each row's columns
// are listed DESCENDING, so anything relying on the input happening to be
// sorted fails here.
const std::vector<std::vector<int>>& adjacency() {
    static const std::vector<std::vector<int>> adj = [] {
        std::vector<std::vector<int>> a(kBlocks);
        auto link = [&](int i, int j) {
            a[static_cast<size_t>(i)].push_back(j);
            if (i != j) a[static_cast<size_t>(j)].push_back(i);
        };
        for (int i = 0; i < kBlocks; ++i) link(i, i);
        link(0, 1); link(1, 2); link(2, 3); link(3, 4); link(4, 5);
        link(5, 6); link(6, 7); link(7, 8); link(0, 4); link(2, 6); link(1, 7);
        for (auto& row : a) {
            std::sort(row.begin(), row.end(), std::greater<int>());
            row.erase(std::unique(row.begin(), row.end()), row.end());
        }
        return a;
    }();
    return adj;
}

// The rows this rank owns, as a contiguous ascending slice.
//
// `workers` is how many of the ranks get any rows at all, counted from rank 0;
// the rest own nothing. workers == nranks spreads the remainder rather than
// piling it on the low ranks, so no rank is starved by accident. workers <
// nranks starves the top ranks ON PURPOSE -- see the empty-rank case in main.
std::vector<int> my_rows(int rank, int nranks, int workers) {
    if (workers > nranks) workers = nranks;
    if (workers < 1) workers = 1;
    if (rank >= workers) return {};
    const int base = kBlocks / workers;
    const int extra = kBlocks % workers;
    const int begin = rank * base + std::min(rank, extra);
    const int end = begin + base + (rank < extra ? 1 : 0);
    std::vector<int> rows;
    for (int i = begin; i < end; ++i) rows.push_back(i);
    return rows;
}

// One full pass of the checks over a given partition. Returns the failure count,
// already agreed across ranks.
int run_case(const char* label, const std::vector<int>& rows, int rank) {
    const std::vector<int>& sizes = block_sizes();
    const std::vector<std::vector<int>>& full_adj = adjacency();
    std::vector<int> offsets(kBlocks + 1, 0);
    for (int i = 0; i < kBlocks; ++i) offsets[i + 1] = offsets[i] + sizes[static_cast<size_t>(i)];
    const int n = offsets[kBlocks];

    std::vector<int> owned_sizes;
    std::vector<std::vector<int>> adj;
    for (int i : rows) {
        owned_sizes.push_back(sizes[static_cast<size_t>(i)]);
        adj.push_back(full_adj[static_cast<size_t>(i)]);
    }

    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_distributed(rows, owned_sizes, adj);

    auto build = [&](BlockSpMat<T>& m, int which) {
        for (int i : rows) {
            for (int j : full_adj[static_cast<size_t>(i)]) {
                const int r = sizes[static_cast<size_t>(i)];
                const int c = sizes[static_cast<size_t>(j)];
                std::vector<T> block(static_cast<size_t>(r) * c);
                for (int a = 0; a < r; ++a) {
                    for (int b = 0; b < c; ++b) {
                        block[static_cast<size_t>(a) * c + b] = hermitian_entry(which, i, j, a, b);
                    }
                }
                m.add_block(i, j, block.data(), r, c, AssemblyMode::INSERT);
            }
        }
    };

    BlockSpMat<T> A(&graph);
    BlockSpMat<T> B(&graph);
    build(A, 0);
    build(B, 1);

    std::vector<T> a_dense = densify(A, sizes, offsets, n);
    std::vector<T> b_dense = densify(B, sizes, offsets, n);
    sum_over_ranks(a_dense);
    sum_over_ranks(b_dense);

    int failures = 0;
    const double tol = 1e-11;
    auto report = [&](const char* what, double worst) {
        if (rank == 0) std::printf("  %-52s max |diff| = %.3e\n", what, worst);
        if (!(worst < tol)) {
            if (rank == 0) std::printf("    FAILED\n");
            ++failures;
        }
    };
    (void)label;

    // C = A B A^H, at threshold 0 so every block the pattern allows survives
    // and the comparison is exact rather than up-to-dropped-blocks.
    const std::vector<T> rarh_want =
        dense_mul(dense_mul(a_dense, b_dense, n), dense_adjoint(a_dense, n), n);
    {
        const BlockSpMat<T> C = A.rarh_upper(B, 0.0).complete_hermitian();
        std::vector<T> got = densify(C, sizes, offsets, n);
        sum_over_ranks(got);
        report("rarh_upper vs dense A B A^H", max_abs_diff(got, rarh_want));
    }

    // The in-place completion must land on the SAME matrix, not merely a
    // correct one: it releases the upper triangle's pages as it mirrors them,
    // and a release that ran even slightly ahead of the last read would show up
    // here as zeroed blocks rather than as a crash -- MADV_DONTNEED hands back
    // zero-filled pages, so the memory stays addressable and the damage is
    // silent. Compared against the dense reference AND against what the
    // non-consuming path produces, because "both wrong the same way" is the one
    // failure a self-comparison would miss.
    {
        BlockSpMat<T> C = A.rarh_upper(B, 0.0);
        C.complete_hermitian_inplace();
        std::vector<T> got = densify(C, sizes, offsets, n);
        sum_over_ranks(got);
        report("complete_hermitian_inplace vs dense A B A^H",
               max_abs_diff(got, rarh_want));

        std::vector<T> reference =
            densify(A.rarh_upper(B, 0.0).complete_hermitian(), sizes, offsets, n);
        sum_over_ranks(reference);
        report("complete_hermitian_inplace vs the copying completion",
               max_abs_diff(got, reference));
    }

    // c2 A^2 + c1 A + c0 I. The coefficients are REAL because the kernel
    // produces only the upper triangle and completes by conjugate mirror: A and
    // A^2 are Hermitian, but a complex scalar times a Hermitian matrix is not,
    // so complex coefficients would violate the contract rather than test it.
    // Newton-Schulz's are real for the same reason.
    {
        const T c2(0.75, 0.0), c1(-1.5, 0.0), c0(2.0, 0.0);
        const BlockSpMat<T> P =
            A.square_polynomial_upper(c2, c1, c0, 0.0).complete_hermitian();
        std::vector<T> got = densify(P, sizes, offsets, n);
        sum_over_ranks(got);
        std::vector<T> want = dense_mul(a_dense, a_dense, n);
        for (size_t i = 0; i < want.size(); ++i) want[i] = c2 * want[i] + c1 * a_dense[i];
        for (int i = 0; i < n; ++i) want[static_cast<size_t>(i) * n + i] += c0;
        report("square_polynomial_upper vs dense c2 A^2 + c1 A + c0 I",
               max_abs_diff(got, want));
    }

    // Every block the kernel KEEPS under a threshold must still be the exact
    // dense answer -- the threshold decides which blocks exist, never their
    // values -- and none of them may come from the lower triangle.
    {
        const double threshold = 1e-2;
        const BlockSpMat<T> C = A.rarh_upper(B, threshold);
        const std::vector<T> want =
            dense_mul(dense_mul(a_dense, b_dense, n), dense_adjoint(a_dense, n), n);
        const DistGraph& g = *C.graph;
        const int rows = static_cast<int>(g.adj_ptr.size()) - 1;
        double worst = 0.0;
        long long kept = 0;
        for (int i = 0; i < rows; ++i) {
            const int g_row = g.get_global_index(i);
            for (int k = g.adj_ptr[i]; k < g.adj_ptr[i + 1]; ++k) {
                const int g_col = g.get_global_index(g.adj_ind[k]);
                if (g_col < g_row) {
                    std::printf("[rank %d] FAILED: lower-triangle block (%d,%d) kept\n",
                                rank, g_row, g_col);
                    ++failures;
                }
                const T* block = C.block_data(k);
                const int r_dim = sizes[static_cast<size_t>(g_row)];
                const int c_dim = sizes[static_cast<size_t>(g_col)];
                for (int r = 0; r < r_dim; ++r) {
                    for (int c = 0; c < c_dim; ++c) {
                        worst = std::max(
                            worst, std::abs(block[static_cast<size_t>(r) * c_dim + c] -
                                            want[static_cast<size_t>(offsets[g_row] + r) * n +
                                                 offsets[g_col] + c]));
                    }
                }
                ++kept;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &kept, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        report("rarh_upper at threshold, kept blocks vs dense", worst);
        // The same blocks must survive however the rows are spread: the
        // threshold is applied to finished blocks, which no partition changes.
        if (kept != 43) {
            if (rank == 0) {
                std::printf("    FAILED: %lld blocks kept, expected 43\n", kept);
            }
            ++failures;
        }
    }

    int any = failures;
    MPI_Allreduce(MPI_IN_PLACE, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return any;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int failures = 0;

    if (rank == 0) std::printf("balanced partition, %d rank(s):\n", nranks);
    failures += run_case("balanced", my_rows(rank, nranks, nranks), rank);

    // A partition that leaves the top ranks owning NOTHING.
    //
    // Not a contrived shape: any run with more ranks than block rows produces
    // it, and so does an unlucky ParMETIS split. It used to hang the job with
    // no output and no error -- make_overwritten_result reached
    // BlockSpMat(DistGraph*) only on a rank whose graph came out empty, and
    // that constructor settles the matrix kind by MPI_Allreduce, so the starved
    // rank entered two collectives nobody else did and blocked there forever.
    //
    // Half the ranks are starved rather than one, so block_displs carries a run
    // of repeated entries and find_owner has to resolve an owner across them.
    if (nranks > 1) {
        const int workers = std::max(1, nranks / 2);
        if (rank == 0) {
            std::printf("starved partition, %d rank(s), %d owning rows:\n", nranks, workers);
        }
        failures += run_case("starved", my_rows(rank, nranks, workers), rank);
    }

    int any = failures;
    MPI_Allreduce(MPI_IN_PLACE, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf(any == 0 ? "test_fused_upper PASSED\n" : "test_fused_upper FAILED\n");
    }
    MPI_Finalize();
    return any == 0 ? 0 : 1;
}

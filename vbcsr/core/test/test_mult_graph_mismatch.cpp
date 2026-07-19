#include "../block_csr.hpp"
#include "../dist_vector.hpp"
#include <iostream>
#include <vector>

using namespace vbcsr;

// Contract under test: BlockSpMat::mult must throw when the vector's graph
// has a different owned block structure than the matrix's graph, even when
// the scalar sizes coincide. Here graph A partitions 2 scalar rows as two
// 1x1 blocks while graph B uses one 2x2 block, so the total size matches
// but the block semantics differ. The mismatch is detected against the
// replicated block partition (DistGraph::block_displs), so every rank
// throws and no rank is left stranded in a collective.
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) std::cout << "This test requires at least 2 ranks." << std::endl;
        MPI_Finalize();
        return 0;
    }

    // Rank 0: A={0,1} sz={1,1}. B={0} sz={2}.
    // Rank 1: A={} sz={}. B={} sz={}.
    std::vector<int> owned_A, sizes_A;
    std::vector<int> owned_B, sizes_B;

    if (rank == 0) {
        owned_A = {0, 1};
        sizes_A = {1, 1};
        owned_B = {0};
        sizes_B = {2};
    }

    std::vector<std::vector<int>> adj_A(owned_A.size());
    std::vector<std::vector<int>> adj_B(owned_B.size());
    if (rank == 0) {
        adj_B[0] = {0}; // diagonal entry so the identity block is part of B's graph
    }

    DistGraph graph_A(MPI_COMM_WORLD);
    graph_A.construct_distributed(owned_A, sizes_A, adj_A);

    DistGraph graph_B(MPI_COMM_WORLD);
    graph_B.construct_distributed(owned_B, sizes_B, adj_B);

    // Vector x bound to Graph A
    DistVector<double> x(&graph_A);
    if (rank == 0) {
        x[0] = 10.0; // Block 0
        x[1] = 20.0; // Block 1
    }

    // Matrix M bound to Graph B: identity on its single 2x2 block.
    BlockSpMat<double> M(&graph_B);
    if (rank == 0) {
        int gid = 0;
        double val[4] = {1.0, 0.0, 0.0, 1.0};
        M.add_block(gid, gid, val, 2, 2, AssemblyMode::INSERT);
    }
    M.assemble();

    DistVector<double> y(&graph_B);

    int failures = 0;
    int caught = 0;
    try {
        M.mult(x, y); // must throw: x's graph does not match M's graph
    } catch (const std::exception& e) {
        std::cout << "Rank " << rank << " caught expected exception: " << e.what() << std::endl;
        caught = 1;
    }

    // The mismatch verdict comes from replicated partition data, so all ranks
    // must throw. Require it on every rank to also catch a rank-asymmetric
    // (deadlock-prone) detection regression.
    MPI_Allreduce(MPI_IN_PLACE, &caught, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (caught == 0) failures++;

    if (rank == 0) {
        if (caught == 0) {
            std::cout << "FAILURE: mult did not throw on every rank despite graph mismatch (2 blocks vs 1 block)." << std::endl;
        } else {
            std::cout << "PASS: mult rejected the mismatched vector graph on all ranks." << std::endl;
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, &failures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return failures > 0 ? 1 : 0;
}

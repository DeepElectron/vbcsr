#include "../block_csr.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

using namespace vbcsr;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Simple Asymmetric Graph
    // 0 connects to 1.
    // 1 connects to 0, 2.
    // 2 connects to 1.
    
    // A^2 Row Counts:
    // 0: {0, 2} -> 2
    // 2: {0, 2} -> 2
    // Wait, this is symmetric counts.
    
    // Use Diamond-Tail again.
    // 0: 4 blocks.
    // 3: 5 blocks.
    
    std::vector<int> owned = {0, 1, 2, 3, 4};
    std::vector<int> sizes = {1, 1, 1, 1, 1};
    std::vector<std::vector<int>> adj(5);
    adj[0] = {0, 1, 2};
    adj[1] = {0, 1, 3};
    adj[2] = {0, 2, 3};
    adj[3] = {1, 2, 3, 4};
    adj[4] = {3, 4};
    
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_distributed(owned, sizes, adj);
    
    BlockSpMat<double, BLASKernel> A(&graph);
    
    double one = 1.0;
    double v_L = 0.8; // Large value
    double v_S = 0.2;  // Value to be asymmetrically filtered
    
    // 0
    A.add_block(0, 0, &one, 1, 1, AssemblyMode::INSERT);
    A.add_block(0, 1, &v_L, 1, 1, AssemblyMode::INSERT);
    A.add_block(0, 2, &v_S, 1, 1, AssemblyMode::INSERT);
    
    // 1
    A.add_block(1, 0, &v_L, 1, 1, AssemblyMode::INSERT);
    A.add_block(1, 1, &one, 1, 1, AssemblyMode::INSERT);
    A.add_block(1, 3, &one, 1, 1, AssemblyMode::INSERT);
    
    // 2
    A.add_block(2, 0, &v_S, 1, 1, AssemblyMode::INSERT);
    A.add_block(2, 2, &one, 1, 1, AssemblyMode::INSERT);
    A.add_block(2, 3, &one, 1, 1, AssemblyMode::INSERT);
    
    // 3
    A.add_block(3, 1, &one, 1, 1, AssemblyMode::INSERT);
    A.add_block(3, 2, &one, 1, 1, AssemblyMode::INSERT);
    A.add_block(3, 3, &one, 1, 1, AssemblyMode::INSERT);
    A.add_block(3, 4, &one, 1, 1, AssemblyMode::INSERT);
    
    // 4
    A.add_block(4, 3, &one, 1, 1, AssemblyMode::INSERT);
    A.add_block(4, 4, &one, 1, 1, AssemblyMode::INSERT);
    
    A.assemble();
    
    // Threshold setup
    // Row 0 count 4. eps = thresh/4.
    // Row 3 count 5. eps = thresh/5.
    // We want S=0.2 to be dropped in Row 0, kept in Row 3.
    // thresh/5 < 0.2 < thresh/4.
    // 0.8 < thresh < 1.0.
    // Set thresh = 0.9.
    // eps0 = 0.225. (0.2 < 0.225 -> Dropped)
    // eps3 = 0.18.  (0.2 > 0.18  -> Kept)
    
    double thresh = 0.9;
    auto C = A.spmm(A, thresh);
    
    if (rank == 0) {
        auto dense = C.to_dense();
        // C_03 at 0*5 + 3 = 3.
        // C_30 at 3*5 + 0 = 15.
        
        double c03 = dense[3];
        double c30 = dense[15];
        
        std::cout << "C_03 = " << c03 << std::endl;
        std::cout << "C_30 = " << c30 << std::endl;
        
        // Expected with bug:
        // C_03 = L = 0.25 (S dropped)
        // C_30 = L+S = 0.45 (S kept)
        // Diff = 0.2.
        
        if (std::abs(c03 - c30) > 1e-9) {
            std::cout << "FAILURE: Asymmetry detected. Diff = " << std::abs(c03 - c30) << std::endl;
        } else {
            std::cout << "SUCCESS: Symmetric." << std::endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}

#include "../block_csr.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <map>

using namespace vbcsr;

// Helper to check equality
bool check_value(const BlockSpMat<double, NaiveKernel<double>>& mat, 
                 int global_row, int global_col, double expected_val,
                 double tol = 1e-12) {
    
    int n_owned = mat.row_ptr.size() - 1;
    for (int i = 0; i < n_owned; ++i) {
        int gid_r = mat.graph->owned_global_indices[i];
        if (gid_r != global_row) continue;
        
        int start = mat.row_ptr[i];
        int end = mat.row_ptr[i+1];
        
        for (int k = start; k < end; ++k) {
            int lid_c = mat.col_ind[k];
            int gid_c = mat.graph->get_global_index(lid_c);
            
            if (gid_c == global_col) {
                double* data = mat.arena.get_ptr(mat.blk_handles[k]);
                // Check first element
                if (std::abs(data[0] - expected_val) > tol) {
                    std::cerr << "Mismatch at (" << global_row << ", " << global_col << "): "
                              << "Got " << data[0] << ", Expected " << expected_val << std::endl;
                    return false;
                }
                return true;
            }
        }
    }
    if (expected_val == 0.0) return true;
    std::cerr << "Block (" << global_row << ", " << global_col << ") not found!" << std::endl;
    return false;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // if (size < 2) {
    //     if (rank == 0) std::cout << "This test requires at least 2 ranks." << std::endl;
    //     MPI_Finalize();
    //     return 0;
    // }

    // Scenario:
    // Rank 0 owns row 0.
    // Graph A: Row 0 connected to Col 0 (local 0) and Col 1 (local 1).
    // Graph B: Row 0 connected to Col 0 (local 0) and Col 2 (local 1).
    
    int n_blocks = 3;
    std::vector<int> block_sizes = {1, 1, 1}; // 1x1 blocks for simplicity
    
    std::vector<int> owned_indices;
    std::vector<int> my_sizes;
    std::vector<std::vector<int>> my_adj;
    
    if (size == 1) {
        // Serial case: Rank 0 owns everything {0, 1, 2} (Sorted)
        // GID 0 -> LID 0
        // GID 1 -> LID 1
        // GID 2 -> LID 2
        
        // Graph A: 0->{0, 1}
        // Graph B: 0->{0, 2}
        owned_indices = {0, 1, 2};
        my_sizes = {1, 1, 1};
        my_adj.resize(3);
        
        // LID 0 (GID 0) connects to 0, 1
        my_adj[0] = {0, 1}; 
        
        // LID 1 (GID 1) connects to 1
        my_adj[1] = {1};    
        
        // LID 2 (GID 2) connects to 2
        my_adj[2] = {2};    
    } else {
        if (rank == 0) {
            owned_indices = {0};
            my_sizes = {1};
            my_adj = {{0, 1}}; // Graph A connects to 0 and 1
        } else {
            owned_indices = {1, 2};
            my_sizes = {1, 1};
            my_adj = {{1}, {2}};
        }
    }
    
    DistGraph graph_A(MPI_COMM_WORLD);
    graph_A.construct_distributed(owned_indices, my_sizes, my_adj);
    
    if (size == 1) {
        my_adj[0] = {0, 2}; // Graph B: Row GID 0 (LID 0) connects to 0, 2
    } else {
        if (rank == 0) {
            my_adj = {{0, 2}}; // Graph B connects to 0 and 2
        }
    }
    DistGraph graph_B(MPI_COMM_WORLD);
    graph_B.construct_distributed(owned_indices, my_sizes, my_adj);
    
    // Check indices on Rank 0
    if (rank == 0) {
        // Graph A: 0->0 (owned), 1->1 (ghost)
        // Graph B: 0->0 (owned), 2->2 (ghost)
        // In Graph A, global 1 should be local 1.
        // In Graph B, global 2 should be local 1 (since it's the first ghost).
        
        int l1_A = graph_A.global_to_local.at(1);
        int l2_B = graph_B.global_to_local.at(2);
        
        std::cout << "Rank 0: Graph A global 1 -> local " << l1_A << std::endl;
        std::cout << "Rank 0: Graph B global 2 -> local " << l2_B << std::endl;
        
        if (l1_A != l2_B) {
            std::cout << "WARNING: Local indices did not collide as expected. Test might not reproduce bug." << std::endl;
        }
    }
    
    // Create Matrices
    BlockSpMat<double, NaiveKernel<double>> A(&graph_A);
    BlockSpMat<double, NaiveKernel<double>> B(&graph_B);
    
    if (rank == 0) {
        // A: (0,0)=1.0, (0,1)=2.0
        double val_00 = 1.0; A.add_block(0, 0, &val_00, 1, 1, AssemblyMode::INSERT);
        double val_01 = 2.0; A.add_block(0, 1, &val_01, 1, 1, AssemblyMode::INSERT);
        
        // B: (0,0)=10.0, (0,2)=20.0
        double val_b00 = 10.0; B.add_block(0, 0, &val_b00, 1, 1, AssemblyMode::INSERT);
        double val_b02 = 20.0; B.add_block(0, 2, &val_b02, 1, 1, AssemblyMode::INSERT);
    }
    A.assemble();
    B.assemble();
    
    // A = A + B
    // Expected:
    // (0,0) = 1 + 10 = 11
    // (0,1) = 2 (unchanged)
    // (0,2) = 20 (new)
    
    // Current Bug:
    // A has (0,1) at local col 1.
    // B has (0,2) at local col 1.
    // Naive axpy sees local col 1 in both.
    // Adds B(0,2) to A(0,1).
    // Result: A(0,1) = 2 + 20 = 22. A(0,2) missing.
    
    try {
        A.axpy(1.0, B);
    } catch (const std::exception& e) {
        std::cout << "Exception during axpy: " << e.what() << std::endl;
    }
    
    if (rank == 0) {
        bool pass = true;
        if (!check_value(A, 0, 0, 11.0)) pass = false;
        if (!check_value(A, 0, 1, 2.0)) {
            std::cout << "FAILURE: A(0,1) corrupted!" << std::endl;
            pass = false;
        }
        if (!check_value(A, 0, 2, 20.0)) {
            std::cout << "FAILURE: A(0,2) missing or wrong!" << std::endl;
            pass = false;
        }
        
        if (pass) std::cout << "Test PASSED" << std::endl;
        else std::cout << "Test FAILED" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}

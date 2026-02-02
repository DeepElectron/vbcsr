#include "../dist_csr.hpp"
#include "../block_csr.hpp"
#include <iostream>
#include <cassert>
#include <mpi.h>

using namespace vbcsr;

void test_simple_conversion() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Construct a simple diagonal matrix with blocks
    // 2 blocks per rank. Block sizes = [2, 2]
    int n_blocks_per_rank = 2;
    int n_global_blocks = n_blocks_per_rank * size;
    std::vector<int> global_block_sizes(n_global_blocks, 2);
    
    // Diagonal adjacency (identity block structure)
    // Plus one off-diagonal block: (i, (i+1)%N) to test ghosts
    std::vector<std::vector<int>> global_adj(n_global_blocks);
    for (int i = 0; i < n_global_blocks; ++i) {
        global_adj[i].push_back(i); // Diagonal
        global_adj[i].push_back((i + 1) % n_global_blocks); // Off-diagonal
    }
    
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_global_blocks, global_block_sizes, global_adj);
    
    BlockSpMat<double> mat(&graph);
    
    // Fill values
    // Diagonal blocks: Identity * 2.0
    // Off-diagonal: All 1.0
    for (int i = 0; i < n_blocks_per_rank; ++i) {
        int l_blk = i; // local block index (owned)
        int g_blk = graph.get_global_index(l_blk);
        
        // Add Diagonal
        std::vector<double> diag_vals(4, 0.0); // 2x2
        diag_vals[0] = 2.0; diag_vals[3] = 2.0; // Identity-ish
        mat.add_block(g_blk, g_blk, diag_vals.data(), 2, 2, AssemblyMode::INSERT, MatrixLayout::ColMajor);
        
        // Add Off-diagonal
        int neighbor = (g_blk + 1) % n_global_blocks;
        std::vector<double> off_vals(4, 1.0); // All 1s
        mat.add_block(g_blk, neighbor, off_vals.data(), 2, 2, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    }
    mat.assemble();
    
    // Convert
    DistCSR<double> csr = convert_to_csr(mat);
    
    // Verify
    // Each rank owns 2 blocks of size 2 -> 4 rows.
    int expected_local_rows = 4;
    assert(csr.n_local_rows == expected_local_rows);
    assert(csr.n_global_rows == 4 * size);
    
    // Check row pointers
    // Each row has 2 blocks of width 2 => 4 non-zeros
    // Total NNZ per rank = 4 rows * 4 nnz = 16
    assert(csr.values.size() == 16);
    assert(csr.row_ptr.size() == expected_local_rows + 1);
    
    for (int i = 0; i < expected_local_rows; ++i) {
        int nnz_row = csr.row_ptr[i+1] - csr.row_ptr[i];
        assert(nnz_row == 4);
    }
    
    // Check values and indices for first row of first block
    // Row 0 (global depends on rank)
    // Should have cols: [GlobalCol(Diagonal), GlobalCol(Neighbor)]
    // Diagonal cols: start_row + 0, start_row + 1
    // Neighbors ...
    
    int my_global_start = rank * 4;
    
    // Check local row 0 (Global Row `my_global_start`)
    // Block 0 is Global Block `rank * 2`.
    // It connects to `rank * 2` and `rank * 2 + 1`.
    // WAIT, logic above:
    // global_adj[i].push_back(i);
    // global_adj[i].push_back((i + 1) % n_global_blocks);
    
    // My first block is GID = rank*2. 
    // Neighbors: rank*2 (Diag) and rank*2+1 (Next).
    // wait is (i+1)%N global block index.
    
    // e.g. Rank 0. Block 0 (G0) -> G0, G1.
    // G0 is diag. G1 is off-diag? No, G1 is just the next block. 
    // Wait, G1 is OWNED by Rank 0! 
    // So for Rank 0, Block 0, both neighbors are local.
    
    // Let's check Rank 0, Block 1 (G1).
    // Neighbors: G1, G2.
    // G2 is owned by Rank 1. (Ghost).
    
    // Verification for Rank 0, Row 2 (First row of Block G1):
    // Cols should be indices of G1 (2,3) and G2 (4,5).
    // Values: Diag (2,0) -> 2.0 at 2. Off (1,1) -> 1.0 at 4,5...
    
    // Let's iterate and print errors
    double tol = 1e-9;
    
    /*
    for(int i=0; i<csr.n_local_rows; ++i) {
        int row_width = csr.row_ptr[i+1] - csr.row_ptr[i];
        std::cout << "Rank " << rank << " Row " << i << ": ";
        for(int k=0; k<row_width; ++k) {
             std::cout << "(" << csr.col_ind[csr.row_ptr[i]+k] << "," << csr.values[csr.row_ptr[i]+k] << ") ";
        }
        std::cout << std::endl;
    }
    */

    // Numerical Verification Loop
    for (int r = 0; r < expected_local_rows; ++r) {
        int global_row = my_global_start + r;
        
        // Find which block this row belongs to
        int local_blk_idx = r / 2; // block size is 2
        int row_in_blk = r % 2;
        int global_blk_idx = graph.get_global_index(local_blk_idx); // Global block index (diagonal element)
        
        int row_start = csr.row_ptr[r];
        int row_end = csr.row_ptr[r+1];
        
        // We expect exactly two blocks for this row:
        // 1. Diagonal block (global_blk_idx)
        // 2. Off-diagonal block ((global_blk_idx + 1) % n_global_blocks)
        
        // Expected global columns for these blocks:
        // Diag block starts at: global_blk_idx * 2
        // Off-diag block starts at: ((global_blk_idx + 1) % n_global_blocks) * 2
        
        int diag_col_start = global_blk_idx * 2;
        int off_col_start = ((global_blk_idx + 1) % n_global_blocks) * 2;
        
        // Let's check each non-zero in the row
        for (int k = row_start; k < row_end; ++k) {
            int col = csr.col_ind[k];
            double val = csr.values[k];
            
            // Determine if this col belongs to Diagonal or Off-Diagonal block
            // Note: Since block size is 2, col / 2 gives the glboal block index of the column
            int col_blk_idx = col / 2;
            int col_in_blk = col % 2;
            
            if (col_blk_idx == global_blk_idx) {
                // Diagonal Block
                // Logic: diag_vals = [2, 0, 0, 2] (RowMajor view of 2x2 identity)
                // But in add_block we used ColMajor!
                // diag_vals data passed: [2, 0, 0, 2]
                // 2x2 ColMajor Matrix:
                // M(0,0)=2, M(1,0)=0
                // M(0,1)=0, M(1,1)=2
                
                // So expected value for (row_in_blk, col_in_blk)
                if (row_in_blk == col_in_blk) {
                    assert(std::abs(val - 2.0) < tol);
                } else {
                    assert(std::abs(val - 0.0) < tol);
                }
            } else if (col_blk_idx == (global_blk_idx + 1) % n_global_blocks) {
                // Off-Diagonal Block
                // Logic: off_vals = [1, 1, 1, 1]
                // All entries are 1.0
                assert(std::abs(val - 1.0) < tol);
            } else {
                std::cerr << "Rank " << rank << " Found unexpected column " << col 
                          << " (Block " << col_blk_idx << ") for row " << global_row << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    if (rank == 0) std::cout << "Test Simple Conversion Passed Numerical checks." << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    try {
        test_simple_conversion();
    } catch (const std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    MPI_Finalize();
    return 0;
}

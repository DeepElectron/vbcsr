#include "../block_csr.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <mpi.h>

using namespace vbcsr;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Create a simple diagonal matrix distributed across processes
    // Global size: size * 10
    int N = size * 10;
    int local_size = 10;
    
    std::vector<int> block_sizes(local_size, 2); // 2x2 blocks
    std::vector<std::vector<int>> adj(local_size);
    
    // Diagonal + off-diagonal
    for(int i=0; i<local_size; ++i) {
        int global_row = rank * local_size + i;
        adj[i].push_back(global_row); // Diagonal
        if(global_row > 0) adj[i].push_back(global_row - 1);
        if(global_row < N - 1) adj[i].push_back(global_row + 1);
    }
    
    std::vector<int> my_indices(local_size);
    for(int i=0; i<local_size; ++i) my_indices[i] = rank * local_size + i;

    DistGraph* graph = new DistGraph(MPI_COMM_WORLD);
    graph->construct_distributed(my_indices, block_sizes, adj);
    
    BlockSpMat<double> mat(graph);
    mat.owns_graph = true;
    
    // Fill with some data
    for(int i=0; i<local_size; ++i) {
        int global_row = rank * local_size + i;
        std::vector<double> data(4, 1.0); // 2x2 block of 1s
        
        // Diagonal
        mat.add_block(global_row, global_row, data.data(), 2, 2);
        
        // Off-diagonal
        if(global_row > 0) mat.add_block(global_row, global_row - 1, data.data(), 2, 2);
        if(global_row < N - 1) mat.add_block(global_row, global_row + 1, data.data(), 2, 2);
    }
    mat.assemble();
    
    // Test Batched Extraction
    if(rank == 0) {
        std::cout << "Testing batched extraction..." << std::endl;
        
        // Batch 1: First 5 rows
        std::vector<int> indices1 = {0, 1, 2, 3, 4};
        
        // Batch 2: Rows 3, 4, 5, 6, 7 (Overlapping)
        std::vector<int> indices2 = {3, 4, 5, 6, 7};
        
        // Batch 3: Last 5 rows
        std::vector<int> indices3;
        for(int i=N-5; i<N; ++i) indices3.push_back(i);
        
        std::vector<std::vector<int>> batches = {indices1, indices2, indices3};
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0) std::cout << "DEBUG: Calling extract_submatrix_batched" << std::endl;
    
    auto results = mat.extract_submatrix_batched(batches);
    
    if(rank == 0) std::cout << "DEBUG: Returned from extract_submatrix_batched" << std::endl;
        
        assert(results.size() == 3);
        
        // Verify Result 1
        auto& sub1 = results[0];
        // Should be 5x5 blocks (tridiagonal structure)
        // Row 0: (0,0), (0,1)
        // Row 1: (1,0), (1,1), (1,2)
        // ...
        // Row 4: (4,3), (4,4) (5 is not in indices1)
        
        // Check a value
        // sub1 is serial, so local indices 0..4 map to global 0..4
        // Block (0,0) -> Global (0,0) -> Should exist
        // Block (0,1) -> Global (0,1) -> Should exist
        // Block (4,5) -> Global (4,5) -> Should NOT exist in submatrix (5 not in indices)
        
        // Let's check nnz blocks
        // Row 0: 2 blocks
        // Row 1: 3 blocks
        // Row 2: 3 blocks
        // Row 3: 3 blocks
        // Row 4: 2 blocks (4,3), (4,4). (4,5) is filtered out.
        // Total: 13 blocks
        
        int nnz_blocks = 0;
        for(int i=0; i<5; ++i) {
            nnz_blocks += sub1.row_ptr[i+1] - sub1.row_ptr[i];
        }
        std::cout << "Batch 1 NNZ Blocks: " << nnz_blocks << " (Expected 13)" << std::endl;
        assert(nnz_blocks == 13);
        
        // Verify Result 2
        auto& sub2 = results[1];
        // Indices: 3, 4, 5, 6, 7
        // Row 3 (idx 0): (3,2), (3,3), (3,4). (3,2) is NOT in indices2. So (3,3), (3,4). -> 2 blocks
        // Row 4 (idx 1): (4,3), (4,4), (4,5). All in indices2. -> 3 blocks
        // Row 5 (idx 2): (5,4), (5,5), (5,6). All in indices2. -> 3 blocks
        // Row 6 (idx 3): (6,5), (6,6), (6,7). All in indices2. -> 3 blocks
        // Row 7 (idx 4): (7,6), (7,7), (7,8). (7,8) NOT in indices2. -> 2 blocks
        // Total: 13 blocks
        
        nnz_blocks = 0;
        for(int i=0; i<5; ++i) {
            nnz_blocks += sub2.row_ptr[i+1] - sub2.row_ptr[i];
        }
        std::cout << "Batch 2 NNZ Blocks: " << nnz_blocks << " (Expected 13)" << std::endl;
        assert(nnz_blocks == 13);
        
        std::cout << "Batched extraction test passed!" << std::endl;
    } else {
        // Other ranks participate in extraction
        std::vector<std::vector<int>> batch; // Empty for others
        // Wait, extract_submatrix_batched is collective? 
        // Yes, it uses MPI_Alltoall, so all ranks must call it.
        // But the input `batch_indices` is only relevant for the caller who wants the result?
        // No, the implementation assumes `batch_indices` is provided by the caller who wants the submatrices.
        // If other ranks pass empty batch, they just serve data.
        // Correct.
        mat.extract_submatrix_batched(batch);
    }
    
    MPI_Finalize();
    return 0;
}

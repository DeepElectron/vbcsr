#include <iostream>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <random>
#include <complex>
#include "../block_csr.hpp"
#include "../kernels.hpp"

using namespace vbcsr;

// Helper to print dense matrix
template <typename T>
void print_dense(const std::vector<T>& A, int rows, int cols, const std::string& name) {
    std::cout << name << ":" << std::endl;
    for(int i=0; i<rows; ++i) {
        for(int j=0; j<cols; ++j) {
            std::cout << A[i*cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if(rank == 0) std::cout << "Running Subgraph Extraction/Insertion Tests..." << std::endl;
    
    // Setup: 2 Ranks (or 1)
    // Global Matrix: 4 blocks.
    // Rank 0: 0, 1
    // Rank 1: 2, 3
    // Block size = 2.
    
    int n_blocks = 4;
    int block_size = 2;
    std::vector<int> block_sizes(n_blocks, block_size);
    
    int rows_per_rank = n_blocks / size;
    int remainder = n_blocks % size;
    int my_start = rank * rows_per_rank + std::min(rank, remainder);
    int my_count = rows_per_rank + (rank < remainder ? 1 : 0);
    
    std::vector<int> my_indices;
    for(int i=0; i<my_count; ++i) my_indices.push_back(my_start + i);
    
    // Full connectivity
    std::vector<std::vector<int>> adj(my_count);
    for(int i=0; i<my_count; ++i) {
        for(int j=0; j<n_blocks; ++j) {
            adj[i].push_back(j);
        }
    }
    
    std::vector<int> my_local_block_sizes(my_count, block_size);
    DistGraph* graph = new DistGraph(MPI_COMM_WORLD);
    graph->construct_distributed(my_indices, my_local_block_sizes, adj);
    
    BlockSpMat<double, SmartKernel<double>> A(graph);
    
    // Fill with unique values: val = row * 10 + col
    for(int i=0; i<my_count; ++i) {
        int r = my_start + i;
        for(int j=0; j<n_blocks; ++j) {
            std::vector<double> data(block_size * block_size);
            for(int k=0; k<block_size*block_size; ++k) {
                data[k] = r * 10.0 + j + k * 0.1;
            }
            A.add_block(r, j, data.data(), block_size, block_size);
        }
    }
    A.assemble();
    
    // Test 1: Extract Submatrix {0, 2} (Mixed Local/Remote if size > 1)
    std::vector<int> sub_indices = {0, 2};
    if (n_blocks > 2) sub_indices = {0, 2}; // Ensure valid
    
    // Only Rank 0 performs extraction test
    if (rank == 0) {
        std::cout << "Rank 0 extracting submatrix {0, 2}..." << std::endl;
        auto sub = A.extract_submatrix(sub_indices);
        
        // Verify Dimensions
        // 2 blocks * 2 size = 4x4 dense
        auto dense = sub.to_dense();
        if (dense.size() != 16) {
            std::cout << "FAILED: Dense size mismatch. Got " << dense.size() << ", expected 16." << std::endl;
        } else {
            std::cout << "Dense size correct." << std::endl;
        }
        
        // Verify Values
        // (0,0) -> 0.0, 0.1...
        // (0,2) -> 2.0, 2.1...
        // (2,0) -> 20.0, 20.1...
        // (2,2) -> 22.0, 22.1...
        
        // Check (0,0) block (top-left in dense)
        if (std::abs(dense[0] - 0.0) > 1e-9) std::cout << "FAILED: (0,0) value mismatch. Got " << dense[0] << std::endl;
        
        // Check (0,2) block (top-right in dense, row 0, col 2*2=4.. wait. Dense is 4x4)
        // Submatrix indices: 0->0, 2->1.
        // Block (0,1) in submatrix corresponds to Global (0,2).
        // Dense offset: Row 0..1, Col 2..3.
        // dense[0*4 + 2] should be A(0,2)[0,0] = 0*10 + 2 + 0 = 2.0
        if (std::abs(dense[2] - 2.0) > 1e-9) std::cout << "FAILED: (0,2) value mismatch. Got " << dense[2] << std::endl;
        
        // Check (2,0) block (bottom-left, Row 2..3, Col 0..1)
        // dense[2*4 + 0] should be A(2,0)[0,0] = 2*10 + 0 + 0 = 20.0
        if (std::abs(dense[8] - 20.0) > 1e-9) std::cout << "FAILED: (2,0) value mismatch. Got " << dense[8] << std::endl;
        
        std::cout << "Extraction Verified." << std::endl;
        
        // Test 2: Modify and Insert
        std::cout << "Modifying submatrix and inserting..." << std::endl;
        
        // Add 1000 to all elements
        for(auto& v : dense) v += 1000.0;
        sub.from_dense(dense);
        
        A.insert_submatrix(sub, sub_indices);
        std::cout << "Insertion called." << std::endl;
    } else {
        // Other ranks participate in communication
        // They need to call extract/insert?
        // No, extract/insert are collective-ish?
        // extract_submatrix uses MPI_Alltoall, so ALL ranks must call it?
        // Wait, my implementation uses collective MPI_Alltoall.
        // So ALL ranks must call extract_submatrix, even if they don't want the result?
        // YES. The implementation assumes collective participation for requests.
        // But if Rank 1 passes empty indices, it just serves requests.
        
        // So we need to sync calls.
        std::vector<int> empty;
        A.extract_submatrix(empty);
        
        // insert_submatrix also uses assemble() which is collective.
        // But insert_submatrix is called on the object.
        // If Rank 1 doesn't have a submatrix to insert, what does it do?
        // It must participate in assemble().
        // But insert_submatrix calls assemble() internally.
        // So Rank 1 must call A.assemble() or A.insert_submatrix(empty_sub)?
        // A.assemble() is what's needed.
        // But we can't easily sync "one rank inserts, others wait".
        // The design of insert_submatrix implies the caller drives the insertion.
        // If it calls assemble(), everyone must call assemble().
        // So Rank 1 must call A.assemble().
        
        A.assemble();
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Verify Insertion Results (Global)
    // Rank 0 checks local (0,0)
    // Rank 1 checks local (2,0) or (2,2)
    
    // Check (0,0) on Rank 0
    if (rank == 0) {
        // Should be 1000.0
        // Find block (0,0)
        // It's local.
        // We can peek A.val
        // But let's use a helper or just trust the logic if no crash.
        // Actually, let's verify.
        // We need to find offset of block (0,0).
        // It's likely the first block.
        if (std::abs(A.val[0] - 1000.0) > 1e-9) std::cout << "FAILED: Rank 0 update mismatch. Got " << A.val[0] << std::endl;
        else std::cout << "Rank 0 Update Verified." << std::endl;
    }
    
    // Check (2,0) on Rank 1 (if size > 1)
    if (size > 1 && rank == 1) {
        // Row 2 is owned by Rank 1.
        // Block (2,0) was updated by Rank 0.
        // Find block (2,0).
        // It's in A.val.
        // Row 2 is local row 0 on Rank 1.
        // Col 0 is... somewhere.
        // Iterate to find col 0.
        int start = A.row_ptr[0];
        int end = A.row_ptr[1];
        bool found = false;
        for(int k=start; k<end; ++k) {
            // We need to map global 0 to local.
            if (A.graph->global_to_local.count(0)) {
                int lid_0 = A.graph->global_to_local.at(0);
                if (A.col_ind[k] == lid_0) {
                    size_t offset = A.blk_ptr[k];
                    if (std::abs(A.val[offset] - 1020.0) > 1e-9) { // Original 20.0 + 1000
                            std::cout << "FAILED: Rank 1 update mismatch. Got " << A.val[offset] << " Expected 1020.0" << std::endl;
                    } else {
                            std::cout << "Rank 1 Update Verified." << std::endl;
                    }
                    found = true;
                }
            }
        }
        if (!found) std::cout << "FAILED: Rank 1 did not find block (2,0)" << std::endl;
    }
    
    if(rank == 0) std::cout << "Test Finished." << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    // Test 3: Robustness (Insert non-existent block)
    if (rank == 0) {
        std::cerr << "Rank 0: Testing Robustness..." << std::endl;
        // Create a dummy serial submatrix
        std::vector<int> dummy_indices = {100}; // 100 doesn't exist
        std::vector<int> dummy_sizes = {2};
        std::vector<std::vector<int>> dummy_adj(1);
        dummy_adj[0].push_back(0); // Self-loop
        
        DistGraph* dummy_graph = new DistGraph(MPI_COMM_SELF);
        dummy_graph->construct_serial(1, dummy_sizes, dummy_adj);
        
        BlockSpMat<double, SmartKernel<double>> dummy_sub(dummy_graph);
        std::vector<double> data(4, 1.0);
        dummy_sub.add_block(100, 100, data.data(), 2, 2);
        dummy_sub.assemble();
        
        std::cerr << "Rank 0: Dummy submatrix assembled." << std::endl;

        // Try to insert this into A
        int owner = A.graph->find_owner(100);
        std::cerr << "Rank 0: Owner of 100 is: " << owner << std::endl;
        
        std::cerr << "Rank 0: Calling insert_submatrix..." << std::endl;
        A.insert_submatrix(dummy_sub, dummy_indices);
        std::cerr << "Rank 0: insert_submatrix returned." << std::endl;
        std::cout << "Robustness Test Passed (No Crash)." << std::endl;
    } else {
        std::cerr << "Rank 1: Waiting for assemble..." << std::endl;
        // Rank 1 must participate in assemble() called by insert_submatrix
        A.assemble();
        std::cerr << "Rank 1: assemble returned." << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    // Test 4: Unsorted Indices {2, 0}
    if (rank == 0) {
        std::cout << "Rank 0: Testing Unsorted Indices {2, 0}..." << std::endl;
        std::vector<int> unsorted_indices = {2, 0};
        
        // Extract
        auto sub = A.extract_submatrix(unsorted_indices);
        auto dense = sub.to_dense();
        
        // Expected mapping if UNSORTED is respected:
        // Row 0 -> Global 2
        // Row 1 -> Global 0
        // (0,0) -> Global (2,2)
        // (1,1) -> Global (0,0)
        
        // Global (2,2) value: 2*10 + 2 = 22.0 + 1000.0 (from Test 2) = 1022.0
        
        double val_00 = dense[0]; // Submatrix (0,0) -> Global (2,2)
        
        // Let's check the first element of Block (0,0)
        if (std::abs(val_00 - 1022.0) > 1e-9) {
            std::cout << "FAILED: Unsorted extraction mismatch. Sub(0,0) [Global 2,2] got " << val_00 << " expected 1022.0." << std::endl;
        } else {
            std::cout << "Unsorted Extraction Verified (Row 0 = Global 2)." << std::endl;
        }
        
        // Insert back with same unsorted indices
        // Modify Sub(0,0) [Global 2,2] -> Add 5000
        // Modify Sub(1,1) [Global 0,0] -> Add 5000
        
        // We need to modify the blocks.
        // Submatrix is serial.
        // Block (0,0) is local 0,0.
        // Block (1,1) is local 1,1.
        
        std::vector<double> mod_data(4, 5000.0);
        sub.add_block(0, 0, mod_data.data(), 2, 2, AssemblyMode::ADD);
        sub.add_block(1, 1, mod_data.data(), 2, 2, AssemblyMode::ADD);
        sub.assemble();
        
        A.insert_submatrix(sub, unsorted_indices);
        std::cout << "Unsorted Insertion called." << std::endl;
        
    } else {
         std::vector<int> empty;
         A.extract_submatrix(empty);
         A.assemble();
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Verify Unsorted Insertion
    if (rank == 1) {
        // Check Global (2,2). Should be 1022.0 + 5000.0 = 6022.0
        // Find block (2,2). Row 2 is local 0.
        int start = A.row_ptr[0];
        int end = A.row_ptr[1];
        bool found = false;
        for(int k=start; k<end; ++k) {
            if (A.graph->global_to_local.count(2)) {
                 int lid_2 = A.graph->global_to_local.at(2);
                 if (A.col_ind[k] == lid_2) {
                     size_t offset = A.blk_ptr[k];
                     if (std::abs(A.val[offset] - 6022.0) > 1e-9) {
                         std::cout << "FAILED: Global (2,2) update mismatch. Got " << A.val[offset] << " Expected 6022.0" << std::endl;
                     } else {
                         std::cout << "Global (2,2) Update Verified." << std::endl;
                     }
                     found = true;
                 }
            }
        }
        if (!found) std::cout << "FAILED: Rank 1 did not find block (2,2)" << std::endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    // Test 5: Explicit Serial Test (MPI_COMM_SELF)
    // Run on all ranks independently
    {
        if (rank == 0) std::cout << "Running Explicit Serial Test (MPI_COMM_SELF)..." << std::endl;
        
        // Create a small serial matrix
        int n_serial_blocks = 2;
        int serial_blk_size = 2;
        std::vector<int> serial_sizes(n_serial_blocks, serial_blk_size);
        std::vector<std::vector<int>> serial_adj(n_serial_blocks);
        for(int i=0; i<n_serial_blocks; ++i) {
            for(int j=0; j<n_serial_blocks; ++j) {
                serial_adj[i].push_back(j);
            }
        }
        
        DistGraph* serial_graph = new DistGraph(MPI_COMM_SELF);
        serial_graph->construct_serial(n_serial_blocks, serial_sizes, serial_adj);
        
        BlockSpMat<double, SmartKernel<double>> S(serial_graph);
        
        // Fill
        for(int i=0; i<n_serial_blocks; ++i) {
            for(int j=0; j<n_serial_blocks; ++j) {
                std::vector<double> data(serial_blk_size * serial_blk_size, 1.0);
                S.add_block(i, j, data.data(), serial_blk_size, serial_blk_size);
            }
        }
        S.assemble();
        
        // Extract {1, 0} (Unsorted)
        std::vector<int> serial_indices = {1, 0};
        auto sub_serial = S.extract_submatrix(serial_indices);
        auto dense_serial = sub_serial.to_dense();
        
        // Verify (0,0) of submatrix -> Global (1,1) of S -> Value 1.0
        if (std::abs(dense_serial[0] - 1.0) > 1e-9) {
             std::cout << "Rank " << rank << " FAILED: Serial extraction value mismatch." << std::endl;
        }
        
        // Modify and Insert
        // Add 10.0 to all
        for(auto& v : dense_serial) v += 10.0;
        sub_serial.from_dense(dense_serial);
        
        S.insert_submatrix(sub_serial, serial_indices);
        
        // Verify S(1,1) -> Should be 11.0
        // S(1,1) is local row 1, col 1.
        int start = S.row_ptr[1];
        int end = S.row_ptr[2];
        bool found = false;
        for(int k=start; k<end; ++k) {
            if (S.col_ind[k] == 1) { // Local col 1 is global 1
                size_t offset = S.blk_ptr[k];
                if (std::abs(S.val[offset] - 11.0) > 1e-9) {
                    std::cout << "Rank " << rank << " FAILED: Serial update mismatch. Got " << S.val[offset] << std::endl;
                }
                found = true;
            }
        }
        if (!found) std::cout << "Rank " << rank << " FAILED: Serial block (1,1) not found." << std::endl;
        
        if (rank == 0) std::cout << "Explicit Serial Test Finished." << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cerr << "Finalizing..." << std::endl;
    
    MPI_Finalize();
    return 0;
}

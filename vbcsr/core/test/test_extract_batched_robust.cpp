#include "../block_csr.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <mpi.h>
#include <algorithm>
#include <random>
#include <set>
#include <map>

using namespace vbcsr;

// Helper to generate expected value
double expected_value(int global_row, int global_col) {
    return (double)(global_row * 10000.0 + global_col);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Setup Large Matrix
    int local_rows = 100; // Total N = 100 * size (e.g., 400)
    int N = size * local_rows;
    
    // Block sizes: deterministic but varying
    std::vector<int> block_sizes(local_rows);
    for(int i=0; i<local_rows; ++i) {
        int global_row = rank * local_rows + i;
        block_sizes[i] = (global_row % 3) + 1;
    }
    
    std::vector<std::vector<int>> adj(local_rows);
    std::vector<int> my_indices(local_rows);
    
    // Plant a dense block at [0..19, 0..19] (Global)
    // Plant a diagonal
    // Plant random noise
    
    for(int i=0; i<local_rows; ++i) {
        int global_row = rank * local_rows + i;
        my_indices[i] = global_row;
        
        std::set<int> cols;
        
        // Diagonal
        cols.insert(global_row);
        
        // Dense Block [0..19]
        if(global_row < 20) {
            for(int c=0; c<20; ++c) cols.insert(c);
        }
        
        // Random Noise (Deterministic)
        std::mt19937 gen(global_row * 123);
        std::uniform_int_distribution<> dis(0, N-1);
        for(int k=0; k<5; ++k) cols.insert(dis(gen));
        
        adj[i].assign(cols.begin(), cols.end());
    }

    DistGraph* graph = new DistGraph(MPI_COMM_WORLD);
    graph->construct_distributed(my_indices, block_sizes, adj);
    
    BlockSpMat<double> mat(graph);
    mat.owns_graph = true;
    
    // Fill data with pattern
    for(int i=0; i<local_rows; ++i) {
        int global_row = rank * local_rows + i;
        int r_dim = block_sizes[i];
        
        for(int global_col : adj[i]) {
            int c_dim = (global_col % 3) + 1;
            std::vector<double> data(r_dim * c_dim);
            for(int k=0; k<data.size(); ++k) data[k] = expected_value(global_row, global_col);
            mat.add_block(global_row, global_col, data.data(), r_dim, c_dim);
        }
    }
    mat.assemble();
    
    // 2. Define Robust Batches
    std::vector<std::vector<int>> batches;
    
    if(rank == 0) {
        std::cout << "Testing robust batched extraction..." << std::endl;
        
        // Case 1: Scrambled Order
        // Request rows {50, 10, 90}
        batches.push_back({50, 10, 90});
        
        // Case 2: Dense Block Subset
        // Request rows {0, 5, 10, 15}
        batches.push_back({0, 5, 10, 15});
        
        // Case 3: Large Batch (First 100 rows)
        std::vector<int> b3;
        for(int i=0; i<100; ++i) b3.push_back(i);
        batches.push_back(b3);
        
        // Case 4: Stress Test (50 random small batches)
        std::mt19937 g_stress(999);
        std::uniform_int_distribution<> d_stress(0, N-1);
        for(int k=0; k<50; ++k) {
            std::vector<int> b;
            int sz = (d_stress(g_stress) % 10) + 1;
            for(int j=0; j<sz; ++j) b.push_back(d_stress(g_stress));
            batches.push_back(b);
        }
        
        auto results = mat.extract_submatrix_batched(batches);
        
        assert(results.size() == batches.size());
        
        // Verification Logic
        for(size_t b_idx=0; b_idx < batches.size(); ++b_idx) {
            const auto& indices = batches[b_idx];
            const auto& sub = results[b_idx];
            
            // 1. Check Dimensions
            if(sub.row_ptr.size() != indices.size() + 1) {
                std::cout << "Batch " << b_idx << ": Row ptr size mismatch." << std::endl;
                exit(1);
            }
            
            // 2. Check Data Integrity
            for(size_t i=0; i<indices.size(); ++i) {
                int global_row = indices[i];
                int start = sub.row_ptr[i];
                int end = sub.row_ptr[i+1];
                
                for(int k=start; k<end; ++k) {
                    int col_lid = sub.col_ind[k];
                    if(col_lid >= indices.size()) {
                         std::cout << "Batch " << b_idx << ": Col index out of bounds." << std::endl;
                         exit(1);
                    }
                    int global_col = indices[col_lid];
                    
                    // Check value
                    const double* data = sub.val.data() + sub.blk_ptr[k];
                    double expected = expected_value(global_row, global_col);
                    
                    if(std::abs(data[0] - expected) > 1e-6) {
                        std::cout << "Batch " << b_idx << ": Value mismatch at (" << global_row << "," << global_col << "). Expected " << expected << ", Got " << data[0] << std::endl;
                        exit(1);
                    }
                }
            }
        }
        std::cout << "Robust batched extraction test passed! Verified " << batches.size() << " batches." << std::endl;
        
    } else {
        mat.extract_submatrix_batched(batches);
    }

    MPI_Finalize();
    return 0;
}

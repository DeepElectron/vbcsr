#include "../block_csr.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <algorithm>
#include <map>

using namespace vbcsr;

// Helper to fill vector with random numbers
void fill_random(std::vector<double>& v, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (auto& val : v) val = dis(gen);
}

// Helper to check equality
bool check_equal(const BlockSpMat<double, NaiveKernel<double>>& mat, 
                 const std::map<std::pair<int,int>, std::vector<double>>& ref_data,
                 int n_blocks, const std::vector<int>& block_sizes,
                 double tol = 1e-12) {
    
    int n_owned = mat.row_ptr.size() - 1;
    for (int i = 0; i < n_owned; ++i) {
        int gid_r = mat.graph->owned_global_indices[i];
        int start = mat.row_ptr[i];
        int end = mat.row_ptr[i+1];
        
        for (int k = start; k < end; ++k) {
            int lid_c = mat.col_ind[k];
            int gid_c = mat.graph->get_global_index(lid_c);
            
            if (ref_data.find({gid_r, gid_c}) == ref_data.end()) {
                // Check if block is zero (might happen if we add explicit zeros, but we shouldn't)
                // Actually, axpby might introduce explicit zeros if alpha*X + beta*Y cancels out.
                // But we don't prune zeros in axpby.
                // So if it's not in ref, it's an error unless ref implies zero.
                // Let's assume ref_data contains all non-zeros.
                // Actually, if we merge, we might have explicit zeros.
                // So we should check if values are small.
                const double* data = mat.block_data(k);
                int r_dim = block_sizes[gid_r];
                int c_dim = block_sizes[gid_c];
                for(int j=0; j<r_dim*c_dim; ++j) {
                    if (std::abs(data[j]) > tol) {
                        std::cerr << "Unexpected non-zero block at " << gid_r << ", " << gid_c << std::endl;
                        return false;
                    }
                }
            } else {
                const auto& ref_block = ref_data.at({gid_r, gid_c});
                const double* data = mat.block_data(k);
                int r_dim = block_sizes[gid_r];
                int c_dim = block_sizes[gid_c];
                for(int j=0; j<r_dim*c_dim; ++j) {
                    if (std::abs(data[j] - ref_block[j]) > tol) {
                        std::cerr << "Value mismatch at " << gid_r << ", " << gid_c << " idx " << j 
                                  << " got " << data[j] << " expected " << ref_block[j] << std::endl;
                        return false;
                    }
                }
            }
        }
    }
    
    // Check if we missed any blocks in ref
    for (const auto& kv : ref_data) {
        int r = kv.first.first;
        int c = kv.first.second;
        // Check if r is owned
        bool owned = false;
        for(int i=0; i<n_owned; ++i) {
            if (mat.graph->owned_global_indices[i] == r) {
                owned = true;
                // Check if block exists
                bool found = false;
                int start = mat.row_ptr[i];
                int end = mat.row_ptr[i+1];
                for(int k=start; k<end; ++k) {
                    if (mat.graph->get_global_index(mat.col_ind[k]) == c) {
                        found = true; break;
                    }
                }
                if (!found) {
                    // Check if ref block is zero
                    bool is_zero = true;
                    for(double v : kv.second) if(std::abs(v) > tol) is_zero = false;
                    if (!is_zero) {
                        std::cerr << "Missing block at " << r << ", " << c << std::endl;
                        return false;
                    }
                }
                break;
            }
        }
    }
    
    return true;
}

void run_test(int rank, int size) {
    if (rank == 0) std::cout << "Running axpby tests..." << std::endl;
    
    // 4 blocks
    std::vector<int> block_sizes = {2, 2, 2, 2};
    int n_blocks = 4;
    
    // Graph 1: Full
    std::vector<std::vector<int>> adj_full = {
        {0, 1, 2, 3},
        {0, 1, 2, 3},
        {0, 1, 2, 3},
        {0, 1, 2, 3}
    };
    
    // Graph 2: Diagonal
    std::vector<std::vector<int>> adj_diag = {
        {0}, {1}, {2}, {3}
    };
    
    // Graph 3: Upper Triangular
    std::vector<std::vector<int>> adj_upper = {
        {0, 1, 2, 3},
        {1, 2, 3},
        {2, 3},
        {3}
    };
    
    // Graph 4: Lower Triangular
    std::vector<std::vector<int>> adj_lower = {
        {0},
        {0, 1},
        {0, 1, 2},
        {0, 1, 2, 3}
    };

    DistGraph g_full(MPI_COMM_WORLD); g_full.construct_serial(n_blocks, block_sizes, adj_full);
    DistGraph g_diag(MPI_COMM_WORLD); g_diag.construct_serial(n_blocks, block_sizes, adj_diag);
    DistGraph g_upper(MPI_COMM_WORLD); g_upper.construct_serial(n_blocks, block_sizes, adj_upper);
    DistGraph g_lower(MPI_COMM_WORLD); g_lower.construct_serial(n_blocks, block_sizes, adj_lower);
    
    auto create_mat = [&](DistGraph* g, double val_start) {
        BlockSpMat<double, NaiveKernel<double>> mat(g);
        int n_owned = g->owned_global_indices.size();
        for(int i=0; i<n_owned; ++i) {
            int r = g->owned_global_indices[i];
            for(int k=g->adj_ptr[i]; k<g->adj_ptr[i+1]; ++k) {
                int c = g->get_global_index(g->adj_ind[k]);
                std::vector<double> block(4, val_start + r + c * 0.1);
                mat.add_block(r, c, block.data(), 2, 2, AssemblyMode::INSERT);
            }
        }
        mat.assemble();
        return mat;
    };
    
    auto get_data = [&](const BlockSpMat<double, NaiveKernel<double>>& mat) {
        std::map<std::pair<int,int>, std::vector<double>> data;
        int n_owned = mat.row_ptr.size() - 1;
        for(int i=0; i<n_owned; ++i) {
            int r = mat.graph->owned_global_indices[i];
            for(int k=mat.row_ptr[i]; k<mat.row_ptr[i+1]; ++k) {
                int c = mat.graph->get_global_index(mat.col_ind[k]);
                const double* data_T = mat.block_data(k);
                std::vector<double> blk(data_T, data_T + 4);
                data[{r,c}] = blk;
            }
        }
        return data;
    };

    // Test 1: Same Graph (A = B)
    {
        if (rank == 0) std::cout << "Test 1: Same Graph..." << std::endl;
        auto Y = create_mat(&g_full, 1.0);
        auto X = create_mat(&g_full, 2.0);
        auto Y_ref = get_data(Y);
        auto X_ref = get_data(X);
        
        double alpha = 0.5, beta = 2.0;
        
        Y.axpby(alpha, X, beta);
        
        // Update Ref
        for(auto& kv : Y_ref) {
            auto& y_blk = kv.second;
            auto& x_blk = X_ref[kv.first];
            for(size_t j=0; j<y_blk.size(); ++j) y_blk[j] = alpha * x_blk[j] + beta * y_blk[j];
        }
        
        if (!check_equal(Y, Y_ref, n_blocks, block_sizes)) {
            if (rank == 0) std::cout << "Test 1 FAILED" << std::endl;
            exit(1);
        }
    }
    
    // Test 2: X Subgraph of Y (Y full, X diag)
    {
        if (rank == 0) std::cout << "Test 2: X Subgraph of Y..." << std::endl;
        auto Y = create_mat(&g_full, 1.0);
        auto X = create_mat(&g_diag, 2.0);
        auto Y_ref = get_data(Y);
        auto X_ref = get_data(X);
        
        double alpha = 0.5, beta = 2.0;
        Y.axpby(alpha, X, beta);
        
        for(auto& kv : Y_ref) {
            auto& y_blk = kv.second;
            if (X_ref.count(kv.first)) {
                auto& x_blk = X_ref[kv.first];
                for(size_t j=0; j<y_blk.size(); ++j) y_blk[j] = alpha * x_blk[j] + beta * y_blk[j];
            } else {
                for(size_t j=0; j<y_blk.size(); ++j) y_blk[j] = beta * y_blk[j];
            }
        }
        
        if (!check_equal(Y, Y_ref, n_blocks, block_sizes)) {
            if (rank == 0) std::cout << "Test 2 FAILED" << std::endl;
            exit(1);
        }
    }
    
    // Test 3: Y Subgraph of X (Y diag, X full)
    {
        if (rank == 0) std::cout << "Test 3: Y Subgraph of X..." << std::endl;
        auto Y = create_mat(&g_diag, 1.0);
        auto X = create_mat(&g_full, 2.0);
        auto Y_ref = get_data(Y);
        auto X_ref = get_data(X);
        
        double alpha = 0.5, beta = 2.0;
        Y.axpby(alpha, X, beta);
        
        // Y should grow to X
        std::map<std::pair<int,int>, std::vector<double>> New_ref;
        for(auto& kv : X_ref) {
            std::vector<double> blk(4);
            if (Y_ref.count(kv.first)) {
                auto& y_blk = Y_ref[kv.first];
                auto& x_blk = kv.second;
                for(size_t j=0; j<4; ++j) blk[j] = alpha * x_blk[j] + beta * y_blk[j];
            } else {
                auto& x_blk = kv.second;
                for(size_t j=0; j<4; ++j) blk[j] = alpha * x_blk[j];
            }
            New_ref[kv.first] = blk;
        }
        
        if (!check_equal(Y, New_ref, n_blocks, block_sizes)) {
            if (rank == 0) std::cout << "Test 3 FAILED" << std::endl;
            exit(1);
        }
    }
    
    // Test 4: Different Graphs (Upper + Lower = Full + Diag overlap)
    {
        if (rank == 0) std::cout << "Test 4: Different Graphs..." << std::endl;
        auto Y = create_mat(&g_upper, 1.0);
        auto X = create_mat(&g_lower, 2.0);
        auto Y_ref = get_data(Y);
        auto X_ref = get_data(X);
        
        double alpha = 0.5, beta = 2.0;
        Y.axpby(alpha, X, beta);
        
        std::map<std::pair<int,int>, std::vector<double>> New_ref;
        
        // Union keys
        std::set<std::pair<int,int>> keys;
        for(auto& kv : Y_ref) keys.insert(kv.first);
        for(auto& kv : X_ref) keys.insert(kv.first);
        
        for(auto& key : keys) {
            std::vector<double> blk(4, 0.0);
            bool in_y = Y_ref.count(key);
            bool in_x = X_ref.count(key);
            
            if (in_y && in_x) {
                for(size_t j=0; j<4; ++j) blk[j] = alpha * X_ref[key][j] + beta * Y_ref[key][j];
            } else if (in_y) {
                for(size_t j=0; j<4; ++j) blk[j] = beta * Y_ref[key][j];
            } else {
                for(size_t j=0; j<4; ++j) blk[j] = alpha * X_ref[key][j];
            }
            New_ref[key] = blk;
        }
        
        if (!check_equal(Y, New_ref, n_blocks, block_sizes)) {
            if (rank == 0) std::cout << "Test 4 FAILED" << std::endl;
            exit(1);
        }
    }
    
    // Test 5: Scalars (beta=0)
    {
        if (rank == 0) std::cout << "Test 5: Beta = 0..." << std::endl;
        auto Y = create_mat(&g_diag, 1.0); // Sparse
        auto X = create_mat(&g_full, 2.0); // Full
        
        // Y = alpha * X + 0 * Y -> Y should become X structure
        double alpha = 0.5;
        Y.axpby(alpha, X, 0.0);
        
        auto X_ref = get_data(X);
        std::map<std::pair<int,int>, std::vector<double>> New_ref;
        for(auto& kv : X_ref) {
            std::vector<double> blk(4);
            for(size_t j=0; j<4; ++j) blk[j] = alpha * kv.second[j];
            New_ref[kv.first] = blk;
        }
        
        if (!check_equal(Y, New_ref, n_blocks, block_sizes)) {
            if (rank == 0) std::cout << "Test 5 FAILED" << std::endl;
            exit(1);
        }
    }
    
    // Test 6: Scalars (alpha=0)
    {
        if (rank == 0) std::cout << "Test 6: Alpha = 0..." << std::endl;
        auto Y = create_mat(&g_full, 1.0);
        auto X = create_mat(&g_diag, 2.0);
        auto Y_ref = get_data(Y);
        
        double beta = 0.5;
        Y.axpby(0.0, X, beta);
        
        for(auto& kv : Y_ref) {
            for(size_t j=0; j<4; ++j) kv.second[j] *= beta;
        }
        
        if (!check_equal(Y, Y_ref, n_blocks, block_sizes)) {
            if (rank == 0) std::cout << "Test 6 FAILED" << std::endl;
            exit(1);
        }
    }

    if (rank == 0) std::cout << "All tests PASSED!" << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    run_test(rank, size);
    
    MPI_Finalize();
    return 0;
}

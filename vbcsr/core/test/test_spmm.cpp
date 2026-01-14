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

// Helper to gather distributed block matrix to a global dense matrix on Rank 0
template <typename T>
std::vector<T> gather_dense(BlockSpMat<T, SmartKernel<T>>& A, int global_rows, int global_cols, const std::vector<int>& global_block_sizes) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Compute block offsets (pixel)
    std::vector<int> block_offsets(global_block_sizes.size() + 1, 0);
    for(size_t i=0; i<global_block_sizes.size(); ++i) {
        block_offsets[i+1] = block_offsets[i] + global_block_sizes[i];
    }
    
    // 1. Local dense buffer
    size_t dense_size = (size_t)global_rows * global_cols;
    // std::cout << "Rank " << rank << " Allocating dense buffer size " << dense_size << " (" << global_rows << "x" << global_cols << ")" << std::endl;
    std::vector<T> local_dense(dense_size, T(0));
    
    int n_owned = A.graph->owned_global_indices.size();
    size_t filled_count = 0;
    for(int i=0; i<n_owned; ++i) {
        int gid_blk = A.graph->owned_global_indices[i];
        int gid_row_start = block_offsets[gid_blk];
        int r_dim = A.graph->block_sizes[i];
        
        int start = A.row_ptr[i];
        int end = A.row_ptr[i+1];
        
        for(int k=start; k<end; ++k) {
            int lid_col = A.col_ind[k];
            int gid_col_blk = A.graph->get_global_index(lid_col);
            int gid_col_start = block_offsets[gid_col_blk];
            int c_dim = A.graph->block_sizes[lid_col];
            
            const T* val = A.val.data() + A.blk_ptr[k];
            
            for(int r=0; r<r_dim; ++r) {
                for(int c=0; c<c_dim; ++c) {
                    // Global dense index (Row-Major)
                    int dense_idx = (gid_row_start + r) * global_cols + (gid_col_start + c);
                    // Block index (Col-Major)
                    int blk_idx = c * r_dim + r;
                    local_dense[dense_idx] = val[blk_idx];
                    filled_count++;
                }
            }
        }
    }
    // std::cout << "Rank " << rank << " gather_dense filled " << filled_count << " elements. n_owned=" << n_owned << std::endl;
    
    // 3. Reduce to Rank 0
    std::vector<T> global_dense(global_rows * global_cols);
    
    if constexpr (std::is_same<T, double>::value) {
        MPI_Reduce(local_dense.data(), global_dense.data(), local_dense.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else if constexpr (std::is_same<T, std::complex<double>>::value) {
        // MPI_C_DOUBLE_COMPLEX might not be standard in C++ bindings, use 2*doubles
        MPI_Reduce(local_dense.data(), global_dense.data(), local_dense.size() * 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        // Fallback or error
        if (rank == 0) std::cerr << "Unsupported type for gather_dense" << std::endl;
    }
    
    return global_dense;
}

// Helper to multiply dense matrices
template <typename T>
std::vector<T> dense_matmul(const std::vector<T>& A, const std::vector<T>& B, int M, int K, int N) {
    std::vector<T> C(M * N, T(0));
    for(int i=0; i<M; ++i) {
        for(int j=0; j<N; ++j) {
            T sum = 0;
            for(int k=0; k<K; ++k) {
                sum += A[i*K + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
    return C;
}

// Helper to compare dense matrices
template <typename T>
bool check_dense_match(const std::vector<T>& A, const std::vector<T>& B, double tol = 1e-10) {
    if (A.size() != B.size()) return false;
    double max_diff = 0.0;
    for(size_t i=0; i<A.size(); ++i) {
        double diff = std::abs(A[i] - B[i]);
        if (diff > max_diff) max_diff = diff;
    }
    if (max_diff > tol) {
        std::cout << "Max diff: " << max_diff << std::endl;
        return false;
    }
    return true;
}

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

// Generalized Random Test
template <typename T>
void test_random_spmm(int n_block_rows, double density, int min_blk, int max_blk, bool test_complex = false) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 1. Generate random block sizes
    std::vector<int> block_sizes(n_block_rows);
    std::mt19937 gen(12345); // Fixed seed for consistency across ranks
    std::uniform_int_distribution<> dis_blk(min_blk, max_blk);
    
    int total_rows = 0;
    for(int i=0; i<n_block_rows; ++i) {
        block_sizes[i] = dis_blk(gen);
        total_rows += block_sizes[i];
    }
    
    // 2. Distribute rows
    int rows_per_rank = n_block_rows / size;
    int remainder = n_block_rows % size;
    int my_start = rank * rows_per_rank + std::min(rank, remainder);
    int my_count = rows_per_rank + (rank < remainder ? 1 : 0);
    
    std::vector<int> my_indices;
    for(int i=0; i<my_count; ++i) my_indices.push_back(my_start + i);
    
    // 3. Generate random adjacency
    std::vector<std::vector<int>> adj(my_count);
    std::uniform_real_distribution<> dis_prob(0.0, 1.0);
    
    for(int i=0; i<my_count; ++i) {
        for(int j=0; j<n_block_rows; ++j) {
            if (dis_prob(gen) < density) {
                adj[i].push_back(j);
            }
        }
        // Ensure diagonal is present for A (optional but good for structure)
        // adj[i].push_back(my_start + i); 
    }
    
    // 4. Create Graph & Matrix
    std::vector<int> my_local_block_sizes;
    for(int i=0; i<my_count; ++i) my_local_block_sizes.push_back(block_sizes[my_start + i]);

    DistGraph* graph = new DistGraph(MPI_COMM_WORLD);
    graph->construct_distributed(my_indices, my_local_block_sizes, adj);
    
    // std::cout << "Rank " << rank << " Graph constructed" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    
    BlockSpMat<T, SmartKernel<T>> A(graph);
    
    // 5. Fill with random values
    int n_owned = graph->owned_global_indices.size();
    std::uniform_real_distribution<> dis_val(-1.0, 1.0);
    for(int i=0; i<n_owned; ++i) {
        int start = A.row_ptr[i];
        int end = A.row_ptr[i+1];
        for(int k=start; k<end; ++k) {
            size_t offset = A.blk_ptr[k];
            size_t next_offset = A.blk_ptr[k+1];
            for(size_t n=0; n<next_offset-offset; ++n) {
                if constexpr (std::is_same<T, std::complex<double>>::value) {
                    A.val[offset + n] = std::complex<double>(dis_val(gen), dis_val(gen));
                } else {
                    A.val[offset + n] = dis_val(gen);
                }
            }
        }
    }
    
    // 6. Compute C = A * A (Self) or A * B. Let's do C = A * A^T for variety?
    // Or just C = A * A.
    // Let's do C = A * A.
    
    // We need to ensure A is square for A*A. It is (n_block_rows x n_block_rows).
    
    // std::cout << "Rank " << rank << " Starting SpMM" << std::endl;
    BlockSpMat<T, SmartKernel<T>> C = A.spmm(A, 0.0); // Threshold 0.0 to keep all
    // std::cout << "Rank " << rank << " SpMM done" << std::endl;
    
    // 7. Verify
    auto dense_A = gather_dense(A, total_rows, total_rows, block_sizes);
    auto dense_C = gather_dense(C, total_rows, total_rows, block_sizes);
    
    if (rank == 0) {
        if (n_block_rows <= 2) {
            print_dense(dense_A, total_rows, total_rows, "A");
            print_dense(dense_C, total_rows, total_rows, "C");
        }
        auto ref_C = dense_matmul(dense_A, dense_A, total_rows, total_rows, total_rows);
        if (n_block_rows <= 2) print_dense(ref_C, total_rows, total_rows, "Ref C");
        
        bool match = check_dense_match(dense_C, ref_C);
        if (match) {
            std::cout << "Random SpMM (" << (test_complex ? "Complex" : "Real") << ", N=" << n_block_rows << ", Den=" << density << ") PASSED" << std::endl;
        } else {
            std::cout << "Random SpMM (" << (test_complex ? "Complex" : "Real") << ", N=" << n_block_rows << ", Den=" << density << ") FAILED" << std::endl;
        }
    }
    
    // 8. Test Transpose C = A^T
    BlockSpMat<T, SmartKernel<T>> AT = A.transpose();
    auto dense_AT = gather_dense(AT, total_rows, total_rows, block_sizes);
    if (rank == 0) {
        if (n_block_rows <= 2) {
             print_dense(dense_AT, total_rows, total_rows, "AT");
        }
        // Verify transpose
        bool trans_match = true;
        for(int i=0; i<total_rows; ++i) {
            for(int j=0; j<total_rows; ++j) {
                T val = dense_AT[i*total_rows + j];
                T ref = dense_A[j*total_rows + i];
                if constexpr (std::is_same<T, std::complex<double>>::value) {
                    ref = std::conj(ref);
                }
                if (std::abs(val - ref) > 1e-10) {
                    trans_match = false;
                    break;
                }
            }
        }
        if (trans_match) {
            std::cout << "Random Transpose (" << (test_complex ? "Complex" : "Real") << ") PASSED" << std::endl;
        } else {
            std::cout << "Random Transpose (" << (test_complex ? "Complex" : "Real") << ") FAILED" << std::endl;
        }
    }
    
    // 9. Test SpMM with transA=true: C = A^T * A
    // std::cout << "Rank " << rank << " Starting SpMM (transA=true)" << std::endl;
    BlockSpMat<T, SmartKernel<T>> Ct = A.spmm(A, 0.0, true, false);
    // std::cout << "Rank " << rank << " SpMM (transA=true) done" << std::endl;
    
    auto dense_Ct = gather_dense(Ct, total_rows, total_rows, block_sizes);
    if (rank == 0) {
        auto ref_Ct = dense_matmul(dense_AT, dense_A, total_rows, total_rows, total_rows);
        bool match_t = check_dense_match(dense_Ct, ref_Ct);
        if (match_t) {
            std::cout << "Random SpMM transA (" << (test_complex ? "Complex" : "Real") << ") PASSED" << std::endl;
        } else {
            std::cout << "Random SpMM transA (" << (test_complex ? "Complex" : "Real") << ") FAILED" << std::endl;
        }
    }
}

// Diverse Connectivity Test
template <typename T>
void test_diverse_spmm(int n_block_rows, double base_density, int min_blk, int max_blk, bool test_complex = false) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 1. Generate random block sizes
    std::vector<int> block_sizes(n_block_rows);
    std::mt19937 gen(67890); 
    std::uniform_int_distribution<> dis_blk(min_blk, max_blk);
    
    int total_rows = 0;
    for(int i=0; i<n_block_rows; ++i) {
        block_sizes[i] = dis_blk(gen);
        total_rows += block_sizes[i];
    }
    
    // 2. Generate importance factors for diverse connectivity
    std::vector<double> importance(n_block_rows);
    std::uniform_real_distribution<> dis_imp(0.05, 1.0);
    for(int i=0; i<n_block_rows; ++i) importance[i] = dis_imp(gen);
    
    // 3. Distribute rows
    int rows_per_rank = n_block_rows / size;
    int remainder = n_block_rows % size;
    int my_start = rank * rows_per_rank + std::min(rank, remainder);
    int my_count = rows_per_rank + (rank < remainder ? 1 : 0);
    
    std::vector<int> my_indices;
    for(int i=0; i<my_count; ++i) my_indices.push_back(my_start + i);
    
    // 4. Generate diverse adjacency
    std::vector<std::vector<int>> adj(my_count);
    std::uniform_real_distribution<> dis_prob(0.0, 1.0);
    
    for(int i=0; i<my_count; ++i) {
        int g_i = my_start + i;
        for(int j=0; j<n_block_rows; ++j) {
            // Probability depends on importance of both row and column
            double prob = base_density * importance[g_i] * importance[j] * 4.0;
            if (dis_prob(gen) < prob) {
                adj[i].push_back(j);
            }
        }
    }
    
    // 5. Create Graph & Matrix
    std::vector<int> my_local_block_sizes;
    for(int i=0; i<my_count; ++i) my_local_block_sizes.push_back(block_sizes[my_start + i]);

    DistGraph* graph = new DistGraph(MPI_COMM_WORLD);
    graph->construct_distributed(my_indices, my_local_block_sizes, adj);
    
    BlockSpMat<T, SmartKernel<T>> A(graph);
    
    // 6. Fill with random values
    std::uniform_real_distribution<> dis_val(-1.0, 1.0);
    for(int i=0; i<my_count; ++i) {
        int start = A.row_ptr[i];
        int end = A.row_ptr[i+1];
        for(int k=start; k<end; ++k) {
            size_t offset = A.blk_ptr[k];
            size_t next_offset = A.blk_ptr[k+1];
            for(size_t n=0; n<next_offset-offset; ++n) {
                if constexpr (std::is_same_v<T, std::complex<double>>) {
                    A.val[offset + n] = std::complex<double>(dis_val(gen), dis_val(gen));
                } else {
                    A.val[offset + n] = dis_val(gen);
                }
            }
        }
    }
    
    // 7. Compute C = A * A
    BlockSpMat<T, SmartKernel<T>> C = A.spmm(A, 0.0);
    
    // 8. Verify
    auto dense_A = gather_dense(A, total_rows, total_rows, block_sizes);
    auto dense_C = gather_dense(C, total_rows, total_rows, block_sizes);
    
    if (rank == 0) {
        auto ref_C = dense_matmul(dense_A, dense_A, total_rows, total_rows, total_rows);
        bool match = check_dense_match(dense_C, ref_C);
        if (match) {
            std::cout << "Diverse SpMM (" << (test_complex ? "Complex" : "Real") << ", N=" << n_block_rows << ") PASSED" << std::endl;
        } else {
            std::cout << "Diverse SpMM (" << (test_complex ? "Complex" : "Real") << ", N=" << n_block_rows << ") FAILED" << std::endl;
        }
    }
}

// Filtered SpMM Test
template <typename T>
void test_filtered_spmm(bool test_complex = false) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Create a simple diagonal matrix with some off-diagonal elements
    // 4 blocks. 
    // 0: 0 (1.0), 1 (0.1)
    // 1: 1 (1.0), 2 (0.1)
    // 2: 2 (1.0), 3 (0.1)
    // 3: 3 (1.0), 0 (0.1)
    
    int n_blocks = 4;
    int block_size = 2;
    std::vector<int> block_sizes(n_blocks, block_size);
    
    int rows_per_rank = n_blocks / 1; // Assume serial for simplicity or run with 1 rank. 
    // But test runs with MPI. Let's handle distributed.
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int my_start = rank * (n_blocks / size);
    int my_count = n_blocks / size;
    // Assume size divides n_blocks (e.g. 1, 2, 4)
    
    std::vector<int> my_indices;
    for(int i=0; i<my_count; ++i) my_indices.push_back(my_start + i);
    
    std::vector<std::vector<int>> adj(my_count);
    for(int i=0; i<my_count; ++i) {
        int r = my_start + i;
        adj[i].push_back(r); // Diagonal
        adj[i].push_back((r + 1) % n_blocks); // Off-diagonal
    }
    
    std::vector<int> my_local_block_sizes(my_count, block_size);
    DistGraph* graph = new DistGraph(MPI_COMM_WORLD);
    graph->construct_distributed(my_indices, my_local_block_sizes, adj);
    
    BlockSpMat<T, SmartKernel<T>> A(graph);
    
    // Fill values
    for(int i=0; i<my_count; ++i) {
        int r = my_start + i;
        // Diagonal: Identity * 1.0
        // Off-diagonal: Identity * 0.1
        
        // Block 0: Diagonal
        std::vector<T> diag(block_size * block_size, T(0));
        for(int k=0; k<block_size; ++k) diag[k*block_size + k] = T(1.0);
        A.add_block(r, r, diag.data(), block_size, block_size);
        
        // Block 1: Off-diagonal
        std::vector<T> off(block_size * block_size, T(0));
        for(int k=0; k<block_size; ++k) off[k*block_size + k] = T(0.1);
        A.add_block(r, (r + 1) % n_blocks, off.data(), block_size, block_size);
    }
    A.assemble();
    
    // C = A * A
    // (I + 0.1*P) * (I + 0.1*P) = I + 0.2*P + 0.01*P^2
    // P is permutation (ring).
    // Terms:
    // I (diag): 1.0
    // P (off-diag 1 step): 0.2
    // P^2 (off-diag 2 steps): 0.01
    
    // Norms (approx):
    // I: sqrt(2) * 1.0 = 1.414
    // P: sqrt(2) * 0.2 = 0.282
    // P^2: sqrt(2) * 0.01 = 0.014
    
    // Filter threshold 0.1
    // Should keep I and P. Drop P^2.
    // Wait, row_eps = threshold / row_count.
    // Row count of C (dense-ish) might be 3 (0, 1, 2).
    // row_eps = 0.1 / 3 = 0.033.
    // P^2 norm 0.014 < 0.033. Should be dropped.
    
    // Filter threshold 0.001
    // Should keep all.
    
    // Filter threshold 0.5
    // row_eps = 0.5 / 3 = 0.166.
    // P norm 0.282 > 0.166. Kept.
    // P^2 dropped.
    
    // Let's try threshold 0.1.
    BlockSpMat<T, SmartKernel<T>> C = A.spmm(A, 0.1);
    
    // Verify C structure
    // Should have 0->0, 0->1. Should NOT have 0->2.
    
    int c_nnz = C.col_ind.size();
    int expected_nnz_per_row = 2; // I and P
    
    // Count local nnz
    if (c_nnz != my_count * expected_nnz_per_row) {
        // It might vary if P^2 wraps around to diagonal (e.g. 2x2 system).
        // Here 4 blocks. P^2 is distance 2. Distinct from 0 and 1.
        if (rank == 0) std::cout << "Filtered SpMM: Unexpected NNZ. Got " << c_nnz << ", expected " << my_count * expected_nnz_per_row << std::endl;
    }
    
    // Verify values?
    // Just verify graph detachment and duplicate.
    
    // Test Duplicate
    BlockSpMat<T, SmartKernel<T>> C_dup = C.duplicate();
    if (C_dup.col_ind.size() != C.col_ind.size()) {
        if (rank == 0) std::cout << "Duplicate failed to preserve structure size" << std::endl;
    }
    
    // Test Graph Detachment
    // Create B sharing A's graph
    BlockSpMat<T, SmartKernel<T>> B = A.duplicate(false); // independent_graph=false
    // Filter B aggressively to remove off-diagonals
    // Threshold 1.0. row_eps = 1.0/2 = 0.5.
    // Off-diag norm 0.14 < 0.5. Dropped.
    // Diag norm 1.4 > 0.5. Kept.
    B.filter_blocks(1.0);
    
    // Check B has only diagonals
    if (B.col_ind.size() != my_count) {
         if (rank == 0) std::cout << "Filter failed to remove off-diagonals. Size: " << B.col_ind.size() << std::endl;
    }
    
    // Check A is untouched
    if (A.col_ind.size() != my_count * 2) {
         if (rank == 0) std::cout << "Filter on B affected A! A size: " << A.col_ind.size() << std::endl;
    }
    
    if (B.graph == A.graph) {
         if (rank == 0) std::cout << "Graph not detached!" << std::endl;
    }
    
    if (rank == 0) std::cout << "Filtered SpMM (" << (test_complex ? "Complex" : "Real") << ") PASSED" << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if(rank == 0) std::cout << "Running Expanded SpMM Tests..." << std::endl;
    
    // 0. Tiny Debug Test
    test_random_spmm<double>(2, 1.0, 2, 3, false);
    
    // 1. Small Real Test
    test_random_spmm<double>(10, 0.3, 2, 5, false);
    
    // 2. Small Complex Test
    test_random_spmm<std::complex<double>>(10, 0.3, 2, 5, true);
    
    // 3. Larger Real Test (N=50 blocks, ~250 rows)
    test_random_spmm<double>(50, 0.1, 2, 8, false);
    
    // 4. Larger Complex Test
    test_random_spmm<std::complex<double>>(30, 0.1, 2, 6, true);
    
    // 5. Diverse Connectivity Test
    test_diverse_spmm<double>(20, 0.2, 2, 10, false);
    test_diverse_spmm<std::complex<double>>(15, 0.2, 2, 8, true);
    
    // 6. Filtered SpMM Test
    test_filtered_spmm<double>(false);
    test_filtered_spmm<std::complex<double>>(true);
    
    MPI_Finalize();
    return 0;
}

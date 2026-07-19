#include "../block_csr.hpp"
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <set>
#include <algorithm>

using namespace vbcsr;

// Simple timer class
struct Timer {
    std::string name;
    std::chrono::high_resolution_clock::time_point start;
    Timer(const std::string& n) : name(n), start(std::chrono::high_resolution_clock::now()) {}
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cout << "[Profile] " << name << ": " << duration << " s" << std::endl;
        }
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) std::cout << "Starting Axpby Benchmark..." << std::endl;

    // Parameters for large matrix
    int n_blocks = 10000;
    int blocks_per_row = 100;
    std::vector<int> block_sizes(n_blocks, 16); // Fixed block size 16
    
    std::mt19937 gen(12345);
    
    // 1. Construct Graphs
    // Graph A: Random sparsity
    std::vector<std::vector<int>> adj_A(n_blocks);
    std::uniform_int_distribution<> col_dist(0, n_blocks - 1);
    for(int i=0; i<n_blocks; ++i) {
        std::set<int> cols;
        while(cols.size() < blocks_per_row) cols.insert(col_dist(gen));
        adj_A[i].assign(cols.begin(), cols.end());
    }
    
    DistGraph graph_A(MPI_COMM_WORLD);
    graph_A.construct_serial(n_blocks, block_sizes, adj_A);
    
    // Graph B: Subset of A (50% of A's blocks)
    std::vector<std::vector<int>> adj_B(n_blocks);
    for(int i=0; i<n_blocks; ++i) {
        for(int col : adj_A[i]) {
            if (col % 2 == 0) adj_B[i].push_back(col);
        }
    }
    DistGraph graph_B(MPI_COMM_WORLD);
    graph_B.construct_serial(n_blocks, block_sizes, adj_B);
    
    // Graph C: Different (Shifted cols)
    std::vector<std::vector<int>> adj_C(n_blocks);
    for(int i=0; i<n_blocks; ++i) {
        std::set<int> cols;
        while(cols.size() < blocks_per_row) cols.insert((col_dist(gen) + 1) % n_blocks);
        adj_C[i].assign(cols.begin(), cols.end());
    }
    DistGraph graph_C(MPI_COMM_WORLD);
    graph_C.construct_serial(n_blocks, block_sizes, adj_C);

    // 2. Create Matrices
    using T = double;
    BlockSpMat<T> mat_A(&graph_A);
    BlockSpMat<T> mat_B(&graph_B); // Subset
    BlockSpMat<T> mat_C(&graph_C); // Different
    
    auto fill_mat = [&](BlockSpMat<T>& mat, const std::vector<std::vector<int>>& adj) {
        int n_owned = mat.graph->owned_global_indices.size();
        std::vector<T> block(16*16, 1.0);
        for(int i=0; i<n_owned; ++i) {
            int r = mat.graph->owned_global_indices[i];
            for(int c : adj[r]) {
                mat.add_block(r, c, block.data(), 16, 16, AssemblyMode::INSERT);
            }
        }
        mat.assemble();
    };
    
    fill_mat(mat_A, adj_A);
    fill_mat(mat_B, adj_B);
    fill_mat(mat_C, adj_C);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "Matrices constructed. Running benchmarks..." << std::endl;

    // Case 1: Same Structure (A += alpha * A)
    {
        BlockSpMat<T> Y = mat_A.duplicate();
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::cout << "\n--- Case 1: Same Structure ---" << std::endl;
        {
            Timer t("Axpby Same Structure");
            Y.axpby(0.5, mat_A, 1.0);
        }
    }
    
    // Case 2: X Subgraph of Y (A += alpha * B) -> B is subset of A
    {
        BlockSpMat<T> Y = mat_A.duplicate();
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::cout << "\n--- Case 2: X Subgraph of Y ---" << std::endl;
        {
            Timer t("Axpby X subset Y");
            Y.axpby(0.5, mat_B, 1.0);
        }
    }
    
    // Case 3: Y Subgraph of X (B += alpha * A) -> B grows to A
    {
        BlockSpMat<T> Y = mat_B.duplicate();
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::cout << "\n--- Case 3: Y Subgraph of X ---" << std::endl;
        {
            Timer t("Axpby Y subset X (Realloc)");
            Y.axpby(0.5, mat_A, 1.0);
        }
    }
    
    // Case 4: Different Graphs (A += alpha * C) -> Union
    {
        BlockSpMat<T> Y = mat_A.duplicate();
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::cout << "\n--- Case 4: Different Graphs (Union) ---" << std::endl;
        {
            Timer t("Axpby Different Graphs (Union)");
            Y.axpby(0.5, mat_C, 1.0);
        }
    }

    MPI_Finalize();
    return 0;
}

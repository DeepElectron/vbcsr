#include <mpi.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "block_csr.hpp"

using namespace vbcsr;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Create a DistGraph with N blocks
    int N = 10; // Total global blocks
    int block_size = 2; // Each block is 2x2
    std::vector<int> global_block_sizes(N, block_size);
    
    // Simple adjacency: Diagonal only initially
    std::vector<std::vector<int>> global_adj(N);
    for(int i=0; i<N; ++i) {
        global_adj[i].push_back(i);
    }

    DistGraph* graph = new DistGraph(MPI_COMM_WORLD);
    graph->construct_serial(N, global_block_sizes, global_adj);

    // 2. Create BlockSpMat
    BlockSpMat<double> mat(graph);
    
    // 3. Populate diagonal blocks
    std::vector<double> block_data(block_size * block_size, 1.0);
    for(int i=0; i<N; ++i) {
        mat.add_block(i, i, block_data.data(), block_size, block_size, AssemblyMode::INSERT);
    }
    mat.assemble();

    // 4. Check Density (Diagonal)
    // NNZ blocks = N
    // Total blocks = N * N
    // Density = N / N^2 = 1/N
    double expected_density = 1.0 / N;
    double density = mat.get_block_density();
    
    if (rank == 0) {
        std::cout << "Diagonal Matrix Density: " << density << " (Expected: " << expected_density << ")" << std::endl;
        if (std::abs(density - expected_density) > 1e-9) {
            std::cerr << "Density check FAILED for diagonal matrix!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // 5. Create a NEW Full Graph for Full Matrix Test
    std::vector<std::vector<int>> full_adj(N);
    for(int i=0; i<N; ++i) {
        for(int j=0; j<N; ++j) {
            full_adj[i].push_back(j);
        }
    }
    
    DistGraph* full_graph = new DistGraph(MPI_COMM_WORLD);
    full_graph->construct_serial(N, global_block_sizes, full_adj);
    
    BlockSpMat<double> full_mat(full_graph);
    
    // Populate all blocks (values don't matter for density check, but good practice)
    for(int i=0; i<N; ++i) {
        for(int j=0; j<N; ++j) {
            full_mat.add_block(i, j, block_data.data(), block_size, block_size, AssemblyMode::INSERT);
        }
    }
    full_mat.assemble();
    
    // 6. Check Density (Full)
    // NNZ blocks = N * N
    // Density = 1.0
    expected_density = 1.0;
    density = full_mat.get_block_density();
    
    if (rank == 0) {
        std::cout << "Full Matrix Density: " << density << " (Expected: " << expected_density << ")" << std::endl;
        if (std::abs(density - expected_density) > 1e-9) {
            std::cerr << "Density check FAILED for full matrix!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "All density tests PASSED!" << std::endl;
    }

    MPI_Finalize();
    return 0;
}

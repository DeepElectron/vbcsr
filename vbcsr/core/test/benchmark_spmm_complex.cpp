#include "../block_csr.hpp"
#include <iostream>
#include <chrono>
#include <random>
#include <complex>
#include <set>

using namespace vbcsr;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // a. 2000 row/col blocks, 200 blocks per row
    int n_blocks = 2000;
    int blocks_per_row = 200;
    
    // Use a fixed seed for graph structure so all ranks agree
    std::mt19937 gen_struct(12345);
    
    // b. block size [9, 13, 15, 20]
    std::vector<int> block_size_options = {9, 13, 15, 20};
    std::uniform_int_distribution<> size_dist(0, block_size_options.size() - 1);
    
    std::vector<int> block_sizes(n_blocks);
    for(int i=0; i<n_blocks; ++i) block_sizes[i] = block_size_options[size_dist(gen_struct)];

    // d. random sparsity
    std::vector<std::vector<int>> adj(n_blocks);
    std::uniform_int_distribution<> col_dist(0, n_blocks - 1);
    for(int i=0; i<n_blocks; ++i) {
        std::set<int> row_cols;
        while(row_cols.size() < blocks_per_row) {
            row_cols.insert(col_dist(gen_struct));
        }
        adj[i].assign(row_cols.begin(), row_cols.end());
    }

    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(n_blocks, block_sizes, adj);

    // c. complex data type
    using T = std::complex<double>;
    BlockSpMat<T> A(&graph);
    BlockSpMat<T> B(&graph);

    // Use rank-dependent seed for data values
    std::mt19937 gen_data(12345 + rank);
    std::uniform_real_distribution<> data_dist(-1.0, 1.0);
    int n_owned = graph.owned_global_indices.size();
    
    for(int i=0; i<n_owned; ++i) {
        int gid_r = graph.owned_global_indices[i];
        for(int gid_c : adj[gid_r]) {
            int rows = block_sizes[gid_r];
            int cols = block_sizes[gid_c];
            std::vector<T> block(rows * cols);
            for(auto& val : block) val = T(data_dist(gen_data), data_dist(gen_data));
            A.add_block(gid_r, gid_c, block.data(), rows, cols, AssemblyMode::INSERT, MatrixLayout::RowMajor);
            
            for(auto& val : block) val = T(data_dist(gen_data), data_dist(gen_data));
            B.add_block(gid_r, gid_c, block.data(), rows, cols, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        }
    }
    A.assemble();
    B.assemble();

    if(rank == 0) std::cout << "Starting SpMM Benchmark (Complex, 2000x2000 blocks, 200 blocks/row)..." << std::endl;

    double threshold = 3e3;
    
    // e. profiling
    MPI_Barrier(MPI_COMM_WORLD);
    auto t_start = std::chrono::high_resolution_clock::now();
    
    auto t1 = std::chrono::high_resolution_clock::now();
    auto metadata_plan = detail::RowMetadataExchangePlan<BlockSpMat<T>, BlockSpMat<T>>::build(A, B);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    auto sym = detail::symbolic_multiply_filtered(A, B, metadata_plan.metadata(), threshold);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t3 = std::chrono::high_resolution_clock::now();
    
    auto payload_plan = detail::BlockPayloadExchangePlan<BlockSpMat<T>>::fetch_required(B, sym.required_blocks);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t4 = std::chrono::high_resolution_clock::now();

    auto assembly_plan = detail::ResultAssemblyPlan<BlockSpMat<T>>::for_spmm(
        A,
        sym,
        metadata_plan.metadata(),
        std::move(payload_plan));
    DistGraph* c_graph = assembly_plan.construct_result_graph(A, "spmm");
    
    BlockSpMat<T> C(c_graph);
    C.owns_graph = true;
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto t5 = std::chrono::high_resolution_clock::now();
    
    const auto& A_norms = A.get_block_norms();
    const auto& B_local_norms = B.get_block_norms();
    A.numeric_multiply(B, assembly_plan.ghost_rows(), C, threshold, A_norms, B_local_norms);
    MPI_Barrier(MPI_COMM_WORLD);
    auto t6 = std::chrono::high_resolution_clock::now();
    
    auto t_end = std::chrono::high_resolution_clock::now();
    
    if(rank == 0) {
        std::cout << "Profiling Results (seconds):" << std::endl;
        std::cout << "  Exchange Metadata: " << std::chrono::duration<double>(t2 - t1).count() << std::endl;
        std::cout << "  Symbolic Phase:    " << std::chrono::duration<double>(t3 - t2).count() << std::endl;
        std::cout << "  Fetch Ghost Blocks: " << std::chrono::duration<double>(t4 - t3).count() << std::endl;
        std::cout << "  Allocate C:        " << std::chrono::duration<double>(t5 - t4).count() << std::endl;
        std::cout << "  Numeric Phase:     " << std::chrono::duration<double>(t6 - t5).count() << std::endl;
        std::cout << "  Total SpMM:        " << std::chrono::duration<double>(t_end - t_start).count() << std::endl;
        
        int total_nnz = 0;
        for(int i=0; i<C.graph->owned_global_indices.size(); ++i) {
            total_nnz += C.row_ptr[i+1] - C.row_ptr[i];
        }
        std::cout << "  C blocks (local rank 0): " << total_nnz << std::endl;
    }

    MPI_Finalize();
    return 0;
}

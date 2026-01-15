#include "../block_csr.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

using namespace vbcsr;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Test 1: Transpose Correctness
    // A = [[1, 2i], [3, 4]]
    // A^H = [[1, 3], [-2i, 4]]
    
    std::vector<int> owned = {0};
    std::vector<int> sizes = {2};
    std::vector<std::vector<int>> adj(1);
    adj[0] = {0};
    
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_distributed(owned, sizes, adj);
    
    BlockSpMat<std::complex<double>, BLASKernel> A(&graph);
    std::complex<double> data_A[4] = {
        {1.0, 0.0}, {3.0, 0.0}, // Col 0: 1, 3
        {0.0, 2.0}, {4.0, 0.0}  // Col 1: 2i, 4
    };
    A.add_block(0, 0, data_A, 2, 2, AssemblyMode::INSERT);
    A.assemble();
    
    auto AH = A.transpose();
    
    if (rank == 0) {
        auto dense = AH.to_dense(); // RowMajor
        // Expected:
        // Row 0: 1, 3
        // Row 1: -2i, 4
        
        std::complex<double> v00 = dense[0];
        std::complex<double> v01 = dense[1];
        std::complex<double> v10 = dense[2];
        std::complex<double> v11 = dense[3];
        
        std::cout << "A^H:" << std::endl;
        std::cout << v00 << " " << v01 << std::endl;
        std::cout << v10 << " " << v11 << std::endl;
        
        bool ok = true;
        if (std::abs(v00 - 1.0) > 1e-9) ok = false;
        if (std::abs(v01 - 3.0) > 1e-9) ok = false;
        if (std::abs(v10 - std::complex<double>(0, -2)) > 1e-9) ok = false;
        if (std::abs(v11 - 4.0) > 1e-9) ok = false;
        
        if (ok) std::cout << "Transpose OK" << std::endl;
        else std::cout << "Transpose FAILED" << std::endl;
    }
    
    // Test 2: SpMM with TransA
    // C = A^H * I = A^H
    BlockSpMat<std::complex<double>, BLASKernel> I(&graph);
    std::complex<double> data_I[4] = {
        {1.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {1.0, 0.0}
    };
    I.add_block(0, 0, data_I, 2, 2, AssemblyMode::INSERT);
    I.assemble();
    
    auto C = A.spmm(I, 0.0, true, false);
    
    if (rank == 0) {
        auto dense = C.to_dense();
        std::complex<double> v10 = dense[2];
        std::cout << "C_10 (should be -2i): " << v10 << std::endl;
        if (std::abs(v10 - std::complex<double>(0, -2)) < 1e-9) {
            std::cout << "SpMM TransA OK" << std::endl;
        } else {
            std::cout << "SpMM TransA FAILED" << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}

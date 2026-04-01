#include "../block_csr.hpp"
#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cmath>
#include <complex>

using namespace vbcsr;

// Helper to fill vector with random numbers
void fill_random(std::vector<double>& v, int seed) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = (double)i * 0.1;
}

std::vector<double> dense_matvec(
    const std::vector<double>& A,
    int rows,
    int cols,
    const std::vector<double>& x) {
    std::vector<double> y(rows, 0.0);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            y[row] += A[row * cols + col] * x[col];
        }
    }
    return y;
}

std::vector<double> dense_matvec_transpose(
    const std::vector<double>& A,
    int rows,
    int cols,
    const std::vector<double>& x) {
    std::vector<double> y(cols, 0.0);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            y[col] += A[row * cols + col] * x[row];
        }
    }
    return y;
}

std::vector<double> dense_matmat(
    const std::vector<double>& A,
    int rows,
    int cols,
    const std::vector<double>& X,
    int x_ld,
    int num_vecs) {
    std::vector<double> Y(rows * num_vecs, 0.0);
    for (int vec = 0; vec < num_vecs; ++vec) {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                Y[vec * rows + row] += A[row * cols + col] * X[vec * x_ld + col];
            }
        }
    }
    return Y;
}

std::vector<double> dense_matmat_transpose(
    const std::vector<double>& A,
    int rows,
    int cols,
    const std::vector<double>& X,
    int x_ld,
    int num_vecs) {
    std::vector<double> Y(cols * num_vecs, 0.0);
    for (int vec = 0; vec < num_vecs; ++vec) {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                Y[vec * cols + col] += A[row * cols + col] * X[vec * x_ld + row];
            }
        }
    }
    return Y;
}

void assert_close(const std::vector<double>& got, const std::vector<double>& expected, double tol = 1e-12) {
    assert(got.size() == expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        assert(std::abs(got[i] - expected[i]) < tol);
    }
}

void test_basic_spmv() {
    std::cout << "Testing Basic SpMV..." << std::endl;
    
    // 1. Setup Graph
    // 2 blocks: 0->0, 0->1
    std::vector<std::vector<int>> global_adj = {{0, 1}, {}};
    std::vector<int> block_sizes = {2, 2};
    
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, block_sizes, global_adj);
    
    BlockSpMat<double> mat(&graph);
    
    // Fill data
    // Block (0,0): Identity
    double d00[] = {1.0, 0.0, 0.0, 1.0};
    mat.add_block(0, 0, d00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    
    // Block (0,1): All 1s
    double d01[] = {1.0, 1.0, 1.0, 1.0};
    mat.add_block(0, 1, d01, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    
    // Vector X = [1, 2, 3, 4]
    DistVector<double> x(&graph);
    double* x_ptr = x.local_data();
    x_ptr[0] = 1.0; x_ptr[1] = 2.0; // Block 0
    x_ptr[2] = 3.0; x_ptr[3] = 4.0; // Block 1
    
    DistVector<double> y(&graph);
    
    mat.mult(x, y);
    
    // Y = A*X
    // Y[0..1] = [1 0; 0 1]*[1;2] + [1 1; 1 1]*[3;4]
    //         = [1; 2] + [7; 7] = [8; 9]
    
    double* y_ptr = y.local_data();
    // std::cout << "Y: " << y_ptr[0] << " " << y_ptr[1] << std::endl;
    
    assert(std::abs(y_ptr[0] - 8.0) < 1e-9);
    assert(std::abs(y_ptr[1] - 9.0) < 1e-9);
    
    std::cout << "PASSED" << std::endl;
}

void test_axpby_structure_mismatch() {
    std::cout << "Testing AXPBY (Structure Mismatch)..." << std::endl;
    
    std::vector<int> block_sizes = {2};
    DistGraph graph(MPI_COMM_WORLD);
    
    // Y has (0,0)
    std::vector<std::vector<int>> adj_Y = {{0}};
    graph.construct_serial(1, block_sizes, adj_Y);
    BlockSpMat<double> Y(&graph);
    double d00[] = {1.0, 1.0, 1.0, 1.0};
    Y.add_block(0, 0, d00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    
    // X has (0,0)
    // We need a separate graph/matrix for X? 
    // Or just modify Y's structure?
    // Let's make X have SAME graph but DIFFERENT structure (subset/superset).
    // But BlockSpMat is tied to graph structure.
    // To test mismatch, we need to manually modify X's topology or use a graph that allows it.
    // Wait, allocate_from_graph sets up full graph structure.
    // So X and Y usually have SAME structure if they share graph.
    // To test mismatch, we must filter one of them.
    
    // Filter Y to remove (0,0) -> Empty
    Y.filter_blocks(10.0); // Norm is sqrt(4)=2. Filter > 10 removes it.
    assert(Y.col_ind.empty());
    
    // Now Y is empty. X has (0,0).
    BlockSpMat<double> X(&graph); // X has (0,0)
    X.add_block(0, 0, d00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    
    // Y = 1.0 * X + 1.0 * Y
    // Y should become X
    Y.axpby(1.0, X, 1.0);
    
    if (Y.graph->owned_global_indices.size() > 0) {
        assert(Y.col_ind.size() == 1);
        assert(Y.col_ind[0] == 0);
        
        // Check values
        const double* ptr = Y.block_data(0);
        assert(ptr[0] == 1.0);
    } else {
        assert(Y.col_ind.empty());
    }
    
    std::cout << "PASSED" << std::endl;
}

void test_memory_reuse() {
    std::cout << "Testing Memory Reuse..." << std::endl;
    
    std::vector<std::vector<int>> adj = {{0}};
    std::vector<int> block_sizes = {2};
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(1, block_sizes, adj);
    
    BlockSpMat<double> mat(&graph);
    
    // Filter out
    mat.filter_blocks(100.0); // Remove everything
    
    BlockSpMat<double> X(&graph); // Has (0,0) if owned
    if (graph.owned_global_indices.size() > 0) {
        double d00[] = {1.0, 1.0, 1.0, 1.0};
        X.add_block(0, 0, d00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    }
    
    // Y (mat) is empty.
    // Y = X
    mat.axpby(1.0, X, 0.0);
    
    if (mat.graph->owned_global_indices.size() > 0) {
        assert(mat.col_ind.size() == 1);
    } else {
        assert(mat.col_ind.empty());
    }
    std::cout << "PASSED" << std::endl;
}

void test_transpose() {
    std::cout << "Testing Transpose..." << std::endl;
    
    // Create a simple 2x2 block matrix
    // Block sizes: [2, 2]
    // 0 1
    // 2 3
    std::vector<int> block_sizes = {2, 2};
    std::vector<std::vector<int>> adj = {{0, 1}, {0, 1}};
    
    DistGraph* graph = new DistGraph(MPI_COMM_WORLD);
    graph->construct_distributed({0, 1}, block_sizes, adj);
    
    BlockSpMat<double> mat(graph);
    mat.owns_graph = true;
    
    // Fill with data
    // (0,0): [1 2; 3 4]
    // (0,1): [5 6; 7 8]
    // (1,0): [9 10; 11 12]
    // (1,1): [13 14; 15 16]
    
    std::vector<double> b00 = {1, 3, 2, 4}; // ColMajor: 1,2 row 0; 3,4 row 1? No.
    // ColMajor: col 0: 1, 3. col 1: 2, 4. -> [1 2; 3 4]
    std::vector<double> b01 = {5, 7, 6, 8}; // [5 6; 7 8]
    std::vector<double> b10 = {9, 11, 10, 12}; // [9 10; 11 12]
    std::vector<double> b11 = {13, 15, 14, 16}; // [13 14; 15 16]
    
    mat.add_block(0, 0, b00.data(), 2, 2);
    mat.add_block(0, 1, b01.data(), 2, 2);
    mat.add_block(1, 0, b10.data(), 2, 2);
    mat.add_block(1, 1, b11.data(), 2, 2);
    mat.assemble();
    
    BlockSpMat<double> mat_T = mat.transpose();
    assert(mat_T.matrix_kind() == MatrixKind::BSR);
    assert(mat_T.row_ptr == mat.row_ptr);
    assert(mat_T.col_ind == mat.col_ind);

    auto check_block = [](const std::vector<double>& got, const std::vector<double>& expected) {
        assert(got.size() == expected.size());
        for (size_t i = 0; i < got.size(); ++i) {
            assert(std::abs(got[i] - expected[i]) < 1e-12);
        }
    };

    check_block(mat_T.get_block(0, 0), std::vector<double>({1, 3, 2, 4}));
    check_block(mat_T.get_block(0, 1), std::vector<double>({9, 11, 10, 12}));
    check_block(mat_T.get_block(1, 0), std::vector<double>({5, 7, 6, 8}));
    check_block(mat_T.get_block(1, 1), std::vector<double>({13, 15, 14, 16}));

    std::cout << "PASSED" << std::endl;
}

void test_real_imag_extract() {
    std::cout << "Testing Real/Imag Extraction..." << std::endl;

    std::vector<int> block_sizes = {2};
    std::vector<std::vector<int>> adj = {{0}};
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(1, block_sizes, adj);

    BlockSpMat<std::complex<double>> mat(&graph);
    std::complex<double> block[] = {
        {1.0, 2.0}, {3.0, 4.0},
        {5.0, 6.0}, {7.0, 8.0}
    };
    mat.add_block(0, 0, block, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.assemble();

    auto real = mat.get_real();
    auto imag = mat.get_imag();

    assert(real.matrix_kind() == MatrixKind::BSR);
    assert(imag.matrix_kind() == MatrixKind::BSR);
    assert(real.row_ptr == mat.row_ptr);
    assert(imag.col_ind == mat.col_ind);

    const std::vector<double> expected_real = {1.0, 3.0, 5.0, 7.0};
    const std::vector<double> expected_imag = {2.0, 4.0, 6.0, 8.0};
    const std::vector<double> got_real = real.get_block(0, 0);
    const std::vector<double> got_imag = imag.get_block(0, 0);
    for (size_t i = 0; i < expected_real.size(); ++i) {
        assert(std::abs(got_real[i] - expected_real[i]) < 1e-12);
        assert(std::abs(got_imag[i] - expected_imag[i]) < 1e-12);
    }

    std::cout << "PASSED" << std::endl;
}

void test_move_semantics_backend_binding() {
    std::cout << "Testing Move Semantics..." << std::endl;

    std::vector<int> block_sizes = {2};
    std::vector<std::vector<int>> adj = {{0}};
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(1, block_sizes, adj);

    BlockSpMat<double> original(&graph);
    double block[] = {1.0, 2.0, 3.0, 4.0};
    original.add_block(0, 0, block, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    original.assemble();

    BlockSpMat<double> moved(std::move(original));
    const std::vector<double> moved_block = moved.get_block(0, 0);
    assert(moved.matrix_kind() == MatrixKind::BSR);
    assert((moved_block == std::vector<double>{1.0, 2.0, 3.0, 4.0}));

    BlockSpMat<double> assigned(&graph);
    assigned = std::move(moved);
    const std::vector<double> assigned_block = assigned.get_block(0, 0);
    assert(assigned.matrix_kind() == MatrixKind::BSR);
    assert((assigned_block == std::vector<double>{1.0, 2.0, 3.0, 4.0}));

    DistVector<double> x(&graph);
    DistVector<double> y(&graph);
    x.local_data()[0] = 1.0;
    x.local_data()[1] = 1.0;
    assigned.mult(x, y);
    assert(std::abs(y.local_data()[0] - 3.0) < 1e-12);
    assert(std::abs(y.local_data()[1] - 7.0) < 1e-12);

    std::cout << "PASSED" << std::endl;
}

void test_bsr_backend_dispatch_kernels() {
    std::cout << "Testing BSR Backend Kernels..." << std::endl;

    std::vector<int> block_sizes = {2, 2};
    std::vector<std::vector<int>> adj = {{0, 1}, {0, 1}};
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, block_sizes, adj);

    BlockSpMat<double> mat(&graph);
    assert(mat.matrix_kind() == MatrixKind::BSR);
    assert(mat.row_ptr == std::vector<int>({0, 2, 4}));
    assert(mat.col_ind == std::vector<int>({0, 1, 0, 1}));

    double a00[] = {1.0, 2.0, 3.0, 4.0};
    double a01[] = {5.0, 6.0, 7.0, 8.0};
    double a10[] = {2.0, 0.0, 1.0, 2.0};
    double a11[] = {0.0, 1.0, 4.0, 3.0};
    mat.add_block(0, 0, a00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(0, 1, a01, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(1, 0, a10, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(1, 1, a11, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.assemble();

    const std::vector<double> dense = mat.to_dense();
    const int rows = graph.block_offsets[graph.owned_global_indices.size()];
    const int cols = graph.block_offsets.back();

    DistVector<double> x(&graph);
    DistVector<double> y(&graph);
    x.local_data()[0] = 1.0;
    x.local_data()[1] = 2.0;
    x.local_data()[2] = 3.0;
    x.local_data()[3] = 4.0;
    mat.mult(x, y);
    assert_close(
        std::vector<double>(y.local_data(), y.local_data() + rows),
        dense_matvec(dense, rows, cols, std::vector<double>(x.local_data(), x.local_data() + cols)));

    DistMultiVector<double> X(&graph, 2);
    DistMultiVector<double> Y(&graph, 2);
    X(0, 0) = 1.0;
    X(1, 0) = 2.0;
    X(2, 0) = 3.0;
    X(3, 0) = 4.0;
    X(0, 1) = 5.0;
    X(1, 1) = 6.0;
    X(2, 1) = 7.0;
    X(3, 1) = 8.0;
    mat.mult_dense(X, Y);
    const std::vector<double> expected_dense = dense_matmat(dense, rows, cols, X.data, cols, 2);
    for (int vec = 0; vec < 2; ++vec) {
        for (int row = 0; row < rows; ++row) {
            assert(std::abs(Y(row, vec) - expected_dense[vec * rows + row]) < 1e-12);
        }
    }

    DistVector<double> x_adj(&graph);
    DistVector<double> y_adj(&graph);
    x_adj.local_data()[0] = 2.0;
    x_adj.local_data()[1] = -1.0;
    x_adj.local_data()[2] = 4.0;
    x_adj.local_data()[3] = 3.0;
    mat.mult_adjoint(x_adj, y_adj);
    assert_close(
        std::vector<double>(y_adj.local_data(), y_adj.local_data() + cols),
        dense_matvec_transpose(dense, rows, cols, std::vector<double>(x_adj.local_data(), x_adj.local_data() + rows)));

    DistMultiVector<double> X_adj(&graph, 2);
    DistMultiVector<double> Y_adj(&graph, 2);
    X_adj(0, 0) = 2.0;
    X_adj(1, 0) = -1.0;
    X_adj(2, 0) = 4.0;
    X_adj(3, 0) = 3.0;
    X_adj(0, 1) = 1.0;
    X_adj(1, 1) = 0.0;
    X_adj(2, 1) = -2.0;
    X_adj(3, 1) = 5.0;
    mat.mult_dense_adjoint(X_adj, Y_adj);
    const std::vector<double> expected_dense_adj = dense_matmat_transpose(dense, rows, cols, X_adj.data, rows, 2);
    for (int vec = 0; vec < 2; ++vec) {
        for (int col = 0; col < cols; ++col) {
            assert(std::abs(Y_adj(col, vec) - expected_dense_adj[vec * cols + col]) < 1e-12);
        }
    }

    std::cout << "PASSED" << std::endl;
}

void test_bsr_axpby_same_structure_handle_stability() {
    std::cout << "Testing BSR AXPBY Handle Stability..." << std::endl;

    std::vector<int> block_sizes = {2};
    std::vector<std::vector<int>> adj = {{0}};
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(1, block_sizes, adj);

    BlockSpMat<double> Y(&graph);
    BlockSpMat<double> X(&graph);
    double y00[] = {1.0, 0.0, 0.0, 1.0};
    double x00[] = {2.0, 1.0, 1.0, 2.0};
    Y.add_block(0, 0, y00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    X.add_block(0, 0, x00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    Y.assemble();
    X.assemble();

    const double* before = Y.block_data(0);
    Y.axpby(2.0, X, 3.0);

    assert(Y.matrix_kind() == MatrixKind::BSR);
    assert(Y.block_data(0) == before);
    assert(Y.get_block(0, 0) == std::vector<double>({7.0, 2.0, 2.0, 7.0}));

    std::cout << "PASSED" << std::endl;
}

void test_bsr_axpby_structure_change_preserves_family() {
    std::cout << "Testing BSR Structure Change Family Preservation..." << std::endl;

    std::vector<int> block_sizes = {2, 2};
    DistGraph graph_y(MPI_COMM_SELF);
    DistGraph graph_x(MPI_COMM_SELF);
    graph_y.construct_serial(2, block_sizes, {{0}, {1}});
    graph_x.construct_serial(2, block_sizes, {{1}, {0}});

    BlockSpMat<double> Y(&graph_y);
    BlockSpMat<double> X(&graph_x);
    double y00[] = {1.0, 0.0, 0.0, 1.0};
    double y11[] = {2.0, 0.0, 0.0, 2.0};
    double x01[] = {3.0, 0.0, 0.0, 3.0};
    double x10[] = {4.0, 0.0, 0.0, 4.0};
    Y.add_block(0, 0, y00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    Y.add_block(1, 1, y11, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    X.add_block(0, 1, x01, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    X.add_block(1, 0, x10, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    Y.assemble();
    X.assemble();

    Y.axpby(2.0, X, 3.0);

    assert(Y.matrix_kind() == MatrixKind::BSR);
    assert(Y.logical_row_ptr() == std::vector<int>({0, 2, 4}));
    assert(Y.logical_col_ind() == std::vector<int>({0, 1, 0, 1}));
    assert(Y.get_block(0, 0) == std::vector<double>({3.0, 0.0, 0.0, 3.0}));
    assert(Y.get_block(0, 1) == std::vector<double>({6.0, 0.0, 0.0, 6.0}));
    assert(Y.get_block(1, 0) == std::vector<double>({8.0, 0.0, 0.0, 8.0}));
    assert(Y.get_block(1, 1) == std::vector<double>({6.0, 0.0, 0.0, 6.0}));

    std::cout << "PASSED" << std::endl;
}

void test_bsr_backend_dispatch_distributed() {
    std::cout << "Testing BSR Backend Kernels (Distributed)..." << std::endl;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> owned;
    std::vector<std::vector<int>> local_adj;
    if (size == 1) {
        owned = {0, 1};
        local_adj = {{0, 1}, {0, 1}};
    } else if (rank == 0) {
        owned = {0};
        local_adj = {{0, 1}};
    } else if (rank == 1) {
        owned = {1};
        local_adj = {{0, 1}};
    }

    DistGraph* graph = new DistGraph(MPI_COMM_WORLD);
    graph->construct_distributed(owned, std::vector<int>(owned.size(), 2), local_adj);

    BlockSpMat<double> mat(graph);
    mat.owns_graph = true;
    assert(mat.matrix_kind() == MatrixKind::BSR);
    assert(mat.row_ptr == mat.logical_row_ptr());
    assert(mat.col_ind == mat.logical_col_ind());

    auto owns = [&](int gid) {
        return std::find(owned.begin(), owned.end(), gid) != owned.end();
    };

    if (owns(0)) {
        double a00[] = {1.0, 2.0, 3.0, 4.0};
        double a01[] = {5.0, 6.0, 7.0, 8.0};
        mat.add_block(0, 0, a00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        mat.add_block(0, 1, a01, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    }
    if (owns(1)) {
        double a10[] = {2.0, 0.0, 1.0, 2.0};
        double a11[] = {0.0, 1.0, 4.0, 3.0};
        mat.add_block(1, 0, a10, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        mat.add_block(1, 1, a11, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    }
    mat.assemble();

    const std::vector<double> dense = mat.to_dense();
    const int rows = graph->block_offsets[graph->owned_global_indices.size()];
    const int cols = graph->block_offsets.back();

    DistVector<double> x(graph);
    DistVector<double> y(graph);
    for (size_t i = 0; i < owned.size(); ++i) {
        const int offset = graph->block_offsets[static_cast<int>(i)];
        x.local_data()[offset] = owned[i] * 10.0 + 1.0;
        x.local_data()[offset + 1] = owned[i] * 10.0 + 2.0;
    }
    mat.mult(x, y);
    assert_close(
        std::vector<double>(y.local_data(), y.local_data() + rows),
        dense_matvec(dense, rows, cols, std::vector<double>(x.data.begin(), x.data.begin() + cols)));

    DistMultiVector<double> X(graph, 2);
    DistMultiVector<double> Y(graph, 2);
    for (size_t i = 0; i < owned.size(); ++i) {
        const int offset = graph->block_offsets[static_cast<int>(i)];
        X(offset, 0) = owned[i] * 10.0 + 1.0;
        X(offset + 1, 0) = owned[i] * 10.0 + 2.0;
        X(offset, 1) = -(owned[i] * 10.0 + 3.0);
        X(offset + 1, 1) = owned[i] * 10.0 + 4.0;
    }
    mat.mult_dense(X, Y);
    const std::vector<double> expected_dense = dense_matmat(dense, rows, cols, X.data, cols, 2);
    for (int vec = 0; vec < 2; ++vec) {
        for (int row = 0; row < rows; ++row) {
            assert(std::abs(Y(row, vec) - expected_dense[vec * rows + row]) < 1e-12);
        }
    }

    std::cout << "PASSED" << std::endl;
}

void test_bsr_transpose_native_distributed() {
    std::cout << "Testing BSR Transpose (Distributed Native)..." << std::endl;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> owned;
    std::vector<std::vector<int>> local_adj;
    if (size == 1) {
        owned = {0, 1};
        local_adj = {{0, 1}, {0}};
    } else if (rank == 0) {
        owned = {0};
        local_adj = {{0, 1}};
    } else if (rank == 1) {
        owned = {1};
        local_adj = {{0}};
    }

    DistGraph* graph = new DistGraph(MPI_COMM_WORLD);
    graph->construct_distributed(owned, std::vector<int>(owned.size(), 2), local_adj);

    BlockSpMat<double> mat(graph);
    mat.owns_graph = true;

    auto owns = [&](int gid) {
        return std::find(owned.begin(), owned.end(), gid) != owned.end();
    };

    if (owns(0)) {
        double a00[] = {1.0, 2.0, 3.0, 4.0};
        double a01[] = {5.0, 6.0, 7.0, 8.0};
        mat.add_block(0, 0, a00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        mat.add_block(0, 1, a01, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    }
    if (owns(1)) {
        double a10[] = {9.0, 10.0, 11.0, 12.0};
        mat.add_block(1, 0, a10, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    }
    mat.assemble();

    BlockSpMat<double> mat_t = mat.transpose();
    assert(mat_t.matrix_kind() == MatrixKind::BSR);
    assert(mat_t.row_ptr == mat_t.logical_row_ptr());
    assert(mat_t.col_ind == mat_t.logical_col_ind());

    if (size == 1) {
        assert(mat_t.row_ptr == std::vector<int>({0, 2, 3}));
        assert(mat_t.col_ind == std::vector<int>({0, 1, 0}));
        assert(mat_t.get_block(0, 0) == std::vector<double>({1.0, 3.0, 2.0, 4.0}));
        assert(mat_t.get_block(0, 1) == std::vector<double>({9.0, 11.0, 10.0, 12.0}));
        assert(mat_t.get_block(1, 0) == std::vector<double>({5.0, 7.0, 6.0, 8.0}));
    } else if (rank == 0) {
        assert(mat_t.row_ptr == std::vector<int>({0, 2}));
        const int col0 = mat_t.graph->global_to_local.at(0);
        const int col1 = mat_t.graph->global_to_local.at(1);
        assert(mat_t.get_block(0, col0) == std::vector<double>({1.0, 3.0, 2.0, 4.0}));
        assert(mat_t.get_block(0, col1) == std::vector<double>({9.0, 11.0, 10.0, 12.0}));
    } else if (rank == 1) {
        assert(mat_t.row_ptr == std::vector<int>({0, 1}));
        const int col0 = mat_t.graph->global_to_local.at(0);
        assert(mat_t.get_block(0, col0) == std::vector<double>({5.0, 7.0, 6.0, 8.0}));
    } else {
        assert(mat_t.row_ptr == std::vector<int>({0}));
        assert(mat_t.col_ind.empty());
    }

    std::cout << "PASSED" << std::endl;
}

void test_bsr_spmm_native_distributed() {
    std::cout << "Testing BSR SpMM (Native)..." << std::endl;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> owned;
    std::vector<std::vector<int>> adj_a;
    std::vector<std::vector<int>> adj_b;
    if (size == 1) {
        owned = {0, 1};
        adj_a = {{0, 1}, {1}};
        adj_b = {{0}, {0, 1}};
    } else if (rank == 0) {
        owned = {0};
        adj_a = {{0, 1}};
        adj_b = {{0}};
    } else if (rank == 1) {
        owned = {1};
        adj_a = {{1}};
        adj_b = {{0, 1}};
    }

    DistGraph* graph_a = new DistGraph(MPI_COMM_WORLD);
    DistGraph* graph_b = new DistGraph(MPI_COMM_WORLD);
    graph_a->construct_distributed(owned, std::vector<int>(owned.size(), 2), adj_a);
    graph_b->construct_distributed(owned, std::vector<int>(owned.size(), 2), adj_b);

    BlockSpMat<double> A(graph_a);
    BlockSpMat<double> B(graph_b);
    A.owns_graph = true;
    B.owns_graph = true;

    auto owns = [&](int gid) {
        return std::find(owned.begin(), owned.end(), gid) != owned.end();
    };

    const std::vector<double> I = {1.0, 0.0, 0.0, 1.0};
    const std::vector<double> TWO_I = {2.0, 0.0, 0.0, 2.0};
    const std::vector<double> THREE_I = {3.0, 0.0, 0.0, 3.0};
    const std::vector<double> FOUR_I = {4.0, 0.0, 0.0, 4.0};
    const std::vector<double> FIVE_I = {5.0, 0.0, 0.0, 5.0};
    const std::vector<double> SIX_I = {6.0, 0.0, 0.0, 6.0};

    if (owns(0)) {
        A.add_block(0, 0, I.data(), 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        A.add_block(0, 1, TWO_I.data(), 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        B.add_block(0, 0, FOUR_I.data(), 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    }
    if (owns(1)) {
        A.add_block(1, 1, THREE_I.data(), 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        B.add_block(1, 0, FIVE_I.data(), 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        B.add_block(1, 1, SIX_I.data(), 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    }
    A.assemble();
    B.assemble();

    BlockSpMat<double> C = A.spmm(B, 0.0);
    assert(C.matrix_kind() == MatrixKind::BSR);
    assert(C.row_ptr == C.logical_row_ptr());
    assert(C.col_ind == C.logical_col_ind());

    if (size == 1) {
        assert(C.row_ptr == std::vector<int>({0, 2, 4}));
        assert(C.col_ind == std::vector<int>({0, 1, 0, 1}));
        assert(C.get_block(0, 0) == std::vector<double>({14.0, 0.0, 0.0, 14.0}));
        assert(C.get_block(0, 1) == std::vector<double>({12.0, 0.0, 0.0, 12.0}));
        assert(C.get_block(1, 0) == std::vector<double>({15.0, 0.0, 0.0, 15.0}));
        assert(C.get_block(1, 1) == std::vector<double>({18.0, 0.0, 0.0, 18.0}));
    } else if (rank == 0) {
        assert(C.row_ptr == std::vector<int>({0, 2}));
        const int col0 = C.graph->global_to_local.at(0);
        const int col1 = C.graph->global_to_local.at(1);
        assert(C.get_block(0, col0) == std::vector<double>({14.0, 0.0, 0.0, 14.0}));
        assert(C.get_block(0, col1) == std::vector<double>({12.0, 0.0, 0.0, 12.0}));
    } else if (rank == 1) {
        assert(C.row_ptr == std::vector<int>({0, 2}));
        const int col0 = C.graph->global_to_local.at(0);
        const int col1 = C.graph->global_to_local.at(1);
        assert(C.get_block(0, col0) == std::vector<double>({15.0, 0.0, 0.0, 15.0}));
        assert(C.get_block(0, col1) == std::vector<double>({18.0, 0.0, 0.0, 18.0}));
    } else {
        assert(C.row_ptr == std::vector<int>({0}));
        assert(C.col_ind.empty());
    }

    std::cout << "PASSED" << std::endl;
}

void test_csr_backend_family_preservation() {
    std::cout << "Testing CSR Backend Family..." << std::endl;

    std::vector<int> block_sizes = {1, 1};
    std::vector<std::vector<int>> adj = {{0, 1}, {1}};
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, block_sizes, adj);

    BlockSpMat<double> mat(&graph);
    assert(mat.matrix_kind() == MatrixKind::CSR);

    double a00[] = {2.0};
    double a01[] = {3.0};
    double a11[] = {5.0};
    mat.add_block(0, 0, a00, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(0, 1, a01, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(1, 1, a11, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.assemble();

    assert(mat.get_block(0, 0) == std::vector<double>({2.0}));
    assert(mat.get_block(0, 1) == std::vector<double>({3.0}));
    assert(mat.local_scalar_nnz() == 3);

    BlockSpMat<double> mat_t = mat.transpose();
    assert(mat_t.matrix_kind() == MatrixKind::CSR);
    assert(mat_t.get_block(0, 0) == std::vector<double>({2.0}));
    assert(mat_t.get_block(1, 0) == std::vector<double>({3.0}));
    assert(mat_t.get_block(1, 1) == std::vector<double>({5.0}));

    BlockSpMat<double> prod = mat.spmm_self(0.0);
    assert(prod.matrix_kind() == MatrixKind::CSR);
    assert(prod.get_block(0, 0) == std::vector<double>({4.0}));
    assert(prod.get_block(0, 1) == std::vector<double>({21.0}));
    assert(prod.get_block(1, 1) == std::vector<double>({25.0}));

    std::cout << "PASSED" << std::endl;
}

void test_csr_backend_dispatch_kernels() {
    std::cout << "Testing CSR Backend Kernels..." << std::endl;

    std::vector<int> block_sizes = {1, 1};
    std::vector<std::vector<int>> adj = {{0, 1}, {0, 1}};
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, block_sizes, adj);

    BlockSpMat<double> mat(&graph);
    assert(mat.matrix_kind() == MatrixKind::CSR);

    double a00[] = {2.0};
    double a01[] = {3.0};
    double a10[] = {4.0};
    double a11[] = {5.0};
    mat.add_block(0, 0, a00, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(0, 1, a01, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(1, 0, a10, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(1, 1, a11, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.assemble();

    DistVector<double> x(&graph);
    DistVector<double> y(&graph);
    x.local_data()[0] = 7.0;
    x.local_data()[1] = 11.0;
    mat.mult(x, y);
    assert(std::abs(y.local_data()[0] - 47.0) < 1e-12);
    assert(std::abs(y.local_data()[1] - 83.0) < 1e-12);

    DistMultiVector<double> X(&graph, 2);
    DistMultiVector<double> Y(&graph, 2);
    X(0, 0) = 1.0;
    X(1, 0) = 2.0;
    X(0, 1) = 3.0;
    X(1, 1) = 4.0;
    mat.mult_dense(X, Y);
    assert(std::abs(Y(0, 0) - 8.0) < 1e-12);
    assert(std::abs(Y(1, 0) - 14.0) < 1e-12);
    assert(std::abs(Y(0, 1) - 18.0) < 1e-12);
    assert(std::abs(Y(1, 1) - 32.0) < 1e-12);

    DistVector<double> x_adj(&graph);
    DistVector<double> y_adj(&graph);
    x_adj.local_data()[0] = 13.0;
    x_adj.local_data()[1] = 17.0;
    mat.mult_adjoint(x_adj, y_adj);
    assert(std::abs(y_adj.local_data()[0] - 94.0) < 1e-12);
    assert(std::abs(y_adj.local_data()[1] - 124.0) < 1e-12);

    DistMultiVector<double> X_adj(&graph, 2);
    DistMultiVector<double> Y_adj(&graph, 2);
    X_adj(0, 0) = 5.0;
    X_adj(1, 0) = 7.0;
    X_adj(0, 1) = 11.0;
    X_adj(1, 1) = 13.0;
    mat.mult_dense_adjoint(X_adj, Y_adj);
    assert(std::abs(Y_adj(0, 0) - 38.0) < 1e-12);
    assert(std::abs(Y_adj(1, 0) - 50.0) < 1e-12);
    assert(std::abs(Y_adj(0, 1) - 74.0) < 1e-12);
    assert(std::abs(Y_adj(1, 1) - 98.0) < 1e-12);

    std::cout << "PASSED" << std::endl;
}

void test_csr_transpose_native_serial() {
    std::cout << "Testing CSR Transpose (Serial Native)..." << std::endl;

    std::vector<int> block_sizes = {1, 1, 1};
    std::vector<std::vector<int>> adj = {{0, 2}, {0}, {1}};
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(3, block_sizes, adj);

    BlockSpMat<double> mat(&graph);
    double a00[] = {2.0};
    double a02[] = {3.0};
    double a10[] = {5.0};
    double a21[] = {7.0};
    mat.add_block(0, 0, a00, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(0, 2, a02, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(1, 0, a10, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(2, 1, a21, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.assemble();

    BlockSpMat<double> mat_t = mat.transpose();
    assert(mat_t.matrix_kind() == MatrixKind::CSR);
    assert(mat_t.row_ptr == std::vector<int>({0, 2, 3, 4}));
    assert(mat_t.col_ind == std::vector<int>({0, 1, 2, 0}));
    assert(mat_t.logical_row_ptr() == std::vector<int>({0, 2, 3, 4}));
    assert(mat_t.logical_col_ind() == std::vector<int>({0, 1, 2, 0}));
    assert(mat_t.row_ptr == mat_t.logical_row_ptr());
    assert(mat_t.col_ind == mat_t.logical_col_ind());
    assert(mat_t.get_block(0, 0) == std::vector<double>({2.0}));
    assert(mat_t.get_block(0, 1) == std::vector<double>({5.0}));
    assert(mat_t.get_block(1, 2) == std::vector<double>({7.0}));
    assert(mat_t.get_block(2, 0) == std::vector<double>({3.0}));

    std::cout << "PASSED" << std::endl;
}

void test_csr_transpose_native_distributed() {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::cout << "Testing CSR Transpose (Distributed Native)..." << std::endl;

    std::vector<int> block_sizes = {1, 1};
    std::vector<int> owned;
    std::vector<std::vector<int>> local_adj;
    if (size == 1) {
        owned = {0, 1};
        local_adj = {{0, 1}, {0}};
    } else if (rank == 0) {
        owned = {0};
        local_adj = {{0, 1}};
    } else if (rank == 1) {
        owned = {1};
        local_adj = {{0}};
    }

    DistGraph* graph = new DistGraph(MPI_COMM_WORLD);
    graph->construct_distributed(owned, std::vector<int>(owned.size(), 1), local_adj);

    BlockSpMat<double> mat(graph);
    mat.owns_graph = true;

    auto owns = [&](int gid) {
        return std::find(owned.begin(), owned.end(), gid) != owned.end();
    };

    if (owns(0)) {
        double a00[] = {2.0};
        double a01[] = {3.0};
        mat.add_block(0, 0, a00, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        mat.add_block(0, 1, a01, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    }
    if (owns(1)) {
        double a10[] = {4.0};
        mat.add_block(1, 0, a10, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    }
    mat.assemble();

    BlockSpMat<double> mat_t = mat.transpose();
    assert(mat_t.matrix_kind() == MatrixKind::CSR);
    assert(mat_t.row_ptr == mat_t.logical_row_ptr());
    assert(mat_t.col_ind == mat_t.logical_col_ind());

    if (size == 1) {
        assert(mat_t.row_ptr == std::vector<int>({0, 2, 3}));
        assert(mat_t.col_ind == std::vector<int>({0, 1, 0}));
        assert(mat_t.get_block(0, 0) == std::vector<double>({2.0}));
        assert(mat_t.get_block(0, 1) == std::vector<double>({4.0}));
        assert(mat_t.get_block(1, 0) == std::vector<double>({3.0}));
    } else if (rank == 0) {
        assert(mat_t.row_ptr == std::vector<int>({0, 2}));
        assert(mat_t.col_ind.size() == 2);
        const int col0 = mat_t.graph->global_to_local.at(0);
        const int col1 = mat_t.graph->global_to_local.at(1);
        assert(mat_t.get_block(0, col0) == std::vector<double>({2.0}));
        assert(mat_t.get_block(0, col1) == std::vector<double>({4.0}));
    } else if (rank == 1) {
        assert(mat_t.row_ptr == std::vector<int>({0, 1}));
        const int col0 = mat_t.graph->global_to_local.at(0);
        assert(mat_t.get_block(0, col0) == std::vector<double>({3.0}));
    } else {
        assert(mat_t.row_ptr == std::vector<int>({0}));
        assert(mat_t.col_ind.empty());
    }

    std::cout << "PASSED" << std::endl;
}

void test_csr_axpby_same_structure_native() {
    std::cout << "Testing CSR AXPBY (Same Structure Native)..." << std::endl;

    std::vector<int> block_sizes = {1, 1};
    std::vector<std::vector<int>> adj = {{0, 1}, {1}};
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, block_sizes, adj);

    BlockSpMat<double> Y(&graph);
    BlockSpMat<double> X(&graph);
    double y00[] = {1.0};
    double y01[] = {2.0};
    double y11[] = {3.0};
    double x00[] = {10.0};
    double x01[] = {20.0};
    double x11[] = {30.0};
    Y.add_block(0, 0, y00, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    Y.add_block(0, 1, y01, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    Y.add_block(1, 1, y11, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    X.add_block(0, 0, x00, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    X.add_block(0, 1, x01, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    X.add_block(1, 1, x11, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    Y.assemble();
    X.assemble();

    const double* before00 = Y.block_data(0);
    const double* before01 = Y.block_data(1);
    const double* before11 = Y.block_data(2);

    Y.axpby(2.0, X, 3.0);

    assert(Y.matrix_kind() == MatrixKind::CSR);
    assert(Y.block_data(0) == before00);
    assert(Y.block_data(1) == before01);
    assert(Y.block_data(2) == before11);
    assert(Y.get_block(0, 0) == std::vector<double>({23.0}));
    assert(Y.get_block(0, 1) == std::vector<double>({46.0}));
    assert(Y.get_block(1, 1) == std::vector<double>({69.0}));

    std::cout << "PASSED" << std::endl;
}

void test_csr_axpby_structure_change_native() {
    std::cout << "Testing CSR AXPBY (Structure Change Native)..." << std::endl;

    std::vector<int> block_sizes = {1, 1};
    DistGraph graph_y(MPI_COMM_SELF);
    DistGraph graph_x(MPI_COMM_SELF);
    graph_y.construct_serial(2, block_sizes, {{0}, {1}});
    graph_x.construct_serial(2, block_sizes, {{1}, {0}});

    BlockSpMat<double> Y(&graph_y);
    BlockSpMat<double> X(&graph_x);
    double y00[] = {1.0};
    double y11[] = {2.0};
    double x01[] = {10.0};
    double x10[] = {20.0};
    Y.add_block(0, 0, y00, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    Y.add_block(1, 1, y11, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    X.add_block(0, 1, x01, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    X.add_block(1, 0, x10, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    Y.assemble();
    X.assemble();

    Y.axpby(2.0, X, 3.0);

    assert(Y.matrix_kind() == MatrixKind::CSR);
    assert(Y.logical_row_ptr() == std::vector<int>({0, 2, 4}));
    assert(Y.logical_col_ind() == std::vector<int>({0, 1, 0, 1}));
    assert(Y.row_ptr == Y.logical_row_ptr());
    assert(Y.col_ind == Y.logical_col_ind());
    assert(Y.get_block(0, 0) == std::vector<double>({3.0}));
    assert(Y.get_block(0, 1) == std::vector<double>({20.0}));
    assert(Y.get_block(1, 0) == std::vector<double>({40.0}));
    assert(Y.get_block(1, 1) == std::vector<double>({6.0}));

    std::cout << "PASSED" << std::endl;
}

void test_csr_spmm_native_distributed() {
    std::cout << "Testing CSR SpMM (Native)..." << std::endl;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<int> owned;
    std::vector<std::vector<int>> adj_a;
    std::vector<std::vector<int>> adj_b;
    if (size == 1) {
        owned = {0, 1};
        adj_a = {{0, 1}, {1}};
        adj_b = {{0}, {0, 1}};
    } else if (rank == 0) {
        owned = {0};
        adj_a = {{0, 1}};
        adj_b = {{0}};
    } else if (rank == 1) {
        owned = {1};
        adj_a = {{1}};
        adj_b = {{0, 1}};
    }

    DistGraph* graph_a = new DistGraph(MPI_COMM_WORLD);
    DistGraph* graph_b = new DistGraph(MPI_COMM_WORLD);
    graph_a->construct_distributed(owned, std::vector<int>(owned.size(), 1), adj_a);
    graph_b->construct_distributed(owned, std::vector<int>(owned.size(), 1), adj_b);

    BlockSpMat<double> A(graph_a);
    BlockSpMat<double> B(graph_b);
    A.owns_graph = true;
    B.owns_graph = true;

    auto owns = [&](int gid) {
        return std::find(owned.begin(), owned.end(), gid) != owned.end();
    };

    if (owns(0)) {
        double a00[] = {2.0};
        double a01[] = {3.0};
        double b00[] = {7.0};
        A.add_block(0, 0, a00, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        A.add_block(0, 1, a01, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        B.add_block(0, 0, b00, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    }
    if (owns(1)) {
        double a11[] = {5.0};
        double b10[] = {11.0};
        double b11[] = {13.0};
        A.add_block(1, 1, a11, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        B.add_block(1, 0, b10, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        B.add_block(1, 1, b11, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    }
    A.assemble();
    B.assemble();

    BlockSpMat<double> C = A.spmm(B, 0.0);
    assert(C.matrix_kind() == MatrixKind::CSR);
    assert(C.row_ptr == C.logical_row_ptr());
    assert(C.col_ind == C.logical_col_ind());

    if (size == 1) {
        assert(C.row_ptr == std::vector<int>({0, 2, 4}));
        assert(C.col_ind == std::vector<int>({0, 1, 0, 1}));
        assert(C.get_block(0, 0) == std::vector<double>({47.0}));
        assert(C.get_block(0, 1) == std::vector<double>({39.0}));
        assert(C.get_block(1, 0) == std::vector<double>({55.0}));
        assert(C.get_block(1, 1) == std::vector<double>({65.0}));
    } else if (rank == 0) {
        assert(C.row_ptr == std::vector<int>({0, 2}));
        const int col0 = C.graph->global_to_local.at(0);
        const int col1 = C.graph->global_to_local.at(1);
        assert(C.get_block(0, col0) == std::vector<double>({47.0}));
        assert(C.get_block(0, col1) == std::vector<double>({39.0}));
    } else if (rank == 1) {
        assert(C.row_ptr == std::vector<int>({0, 2}));
        const int col0 = C.graph->global_to_local.at(0);
        const int col1 = C.graph->global_to_local.at(1);
        assert(C.get_block(0, col0) == std::vector<double>({55.0}));
        assert(C.get_block(0, col1) == std::vector<double>({65.0}));
    } else {
        assert(C.row_ptr == std::vector<int>({0}));
        assert(C.col_ind.empty());
    }

    std::cout << "PASSED" << std::endl;
}

void test_spmm() {
    std::cout << "Testing SpMM (Self)..." << std::endl;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    std::vector<int> block_sizes = {2, 2};
    std::vector<std::vector<int>> adj = {{0}, {1}}; // Global adjacency: 0->0, 1->1
    
    // Partition ownership
    std::vector<int> owned;
    if (size == 1) {
        owned = {0, 1};
    } else {
        if (rank == 0) owned = {0};
        else if (rank == 1) owned = {1};
        // Ranks > 1 own nothing
    }
    
    // Adjust local adjacency based on owned
    std::vector<std::vector<int>> local_adj;
    for (int global_row : owned) {
        local_adj.push_back(adj[global_row]);
    }
    
    DistGraph* graph = new DistGraph(MPI_COMM_WORLD);
    graph->construct_distributed(owned, block_sizes, local_adj);
    
    BlockSpMat<double> mat(graph);
    mat.owns_graph = true;
    
    std::vector<double> identity = {1, 0, 0, 1}; 
    
    for (size_t i = 0; i < owned.size(); ++i) {
        int global_row = owned[i];
        // Add diagonal block (col = global_row)
        mat.add_block(global_row, global_row, identity.data(), 2, 2);
    }
    mat.assemble();
    
    // C = A * A = I * I = I
    BlockSpMat<double> C = mat.spmm_self(0.0);
    
    // Verify C
    // Should have same structure as A (which is I)
    // So local size should match owned size
    
    if (C.col_ind.size() != owned.size()) {
        std::cout << "Rank " << rank << " C size: " << C.col_ind.size() 
                  << " Expected: " << owned.size() << std::endl;
    }
    assert(C.col_ind.size() == owned.size());
    
    for(size_t k=0; k<C.col_ind.size(); ++k) {
        // Check diagonal values
        const double* d = C.block_data(static_cast<int>(k));
        assert(d[0] == 1.0 && d[3] == 1.0); // Diagonal
        assert(d[1] == 0.0 && d[2] == 0.0); // Off-diagonal
    }
    
    std::cout << "PASSED" << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        test_basic_spmv();
    }
    test_axpby_structure_mismatch();
    test_memory_reuse();
    if (size == 1) {
        test_transpose();
    }
    test_real_imag_extract();
    test_move_semantics_backend_binding();
    test_bsr_backend_dispatch_kernels();
    test_bsr_axpby_same_structure_handle_stability();
    test_bsr_axpby_structure_change_preserves_family();
    test_bsr_backend_dispatch_distributed();
    test_bsr_transpose_native_distributed();
    test_bsr_spmm_native_distributed();
    test_csr_backend_family_preservation();
    test_csr_backend_dispatch_kernels();
    test_csr_transpose_native_serial();
    test_csr_transpose_native_distributed();
    test_csr_axpby_same_structure_native();
    test_csr_axpby_structure_change_native();
    test_csr_spmm_native_distributed();
    test_spmm();
    
    MPI_Finalize();
    return 0;
}

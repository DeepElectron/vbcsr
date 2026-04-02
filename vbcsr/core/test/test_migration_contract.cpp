#include "../block_csr.hpp"
#include "../dist_multivector.hpp"
#include "../dist_vector.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace vbcsr;

namespace {

void assert_close(const std::vector<double>& lhs, const std::vector<double>& rhs, double tol = 1e-12) {
    assert(lhs.size() == rhs.size());
    for (size_t i = 0; i < lhs.size(); ++i) {
        assert(std::abs(lhs[i] - rhs[i]) < tol);
    }
}

void assert_matrix_dense_close(const BlockSpMat<double>& lhs, const BlockSpMat<double>& rhs, double tol = 1e-12) {
    assert_close(lhs.to_dense(), rhs.to_dense(), tol);
}

void fill_reference_matrix(BlockSpMat<double>& mat) {
    const double b00[] = {1.0, 2.0, 3.0, 4.0};
    const double b01[] = {5.0, 6.0, 7.0, 8.0};
    const double b11[] = {9.0, 10.0, 11.0, 12.0};

    mat.add_block(0, 0, b00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(0, 1, b01, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(1, 1, b11, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.assemble();
}

void test_add_across_equivalent_graphs() {
    DistGraph g1(MPI_COMM_SELF);
    DistGraph g2(MPI_COMM_SELF);
    g1.construct_serial(2, {2, 2}, {{0, 1}, {1}});
    g2.construct_serial(2, {2, 2}, {{0, 1}, {1}});

    BlockSpMat<double> A(&g1);
    BlockSpMat<double> B(&g2);
    fill_reference_matrix(A);
    fill_reference_matrix(B);

    BlockSpMat<double> C = A.add(B, 1.0, 1.0);
    assert(C.matrix_kind() == MatrixKind::BSR);
    assert_close(C.get_block(0, 0), std::vector<double>({2.0, 4.0, 6.0, 8.0}));
    assert_close(C.get_block(0, 1), std::vector<double>({10.0, 12.0, 14.0, 16.0}));
    assert_close(C.get_block(1, 1), std::vector<double>({18.0, 20.0, 22.0, 24.0}));
}

void test_copy_from_rejects_mismatched_structure() {
    DistGraph dense_graph(MPI_COMM_SELF);
    DistGraph diag_graph(MPI_COMM_SELF);
    dense_graph.construct_serial(2, {1, 1}, {{0, 1}, {1}});
    diag_graph.construct_serial(2, {1, 1}, {{0}, {1}});

    BlockSpMat<double> dense(&dense_graph);
    BlockSpMat<double> diag(&diag_graph);

    const double one = 1.0;
    dense.add_block(0, 0, &one, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    dense.add_block(0, 1, &one, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    dense.add_block(1, 1, &one, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    dense.assemble();

    diag.add_block(0, 0, &one, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    diag.add_block(1, 1, &one, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    diag.assemble();

    bool threw = false;
    try {
        dense.copy_from(diag);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

void test_mult_optimized_matches_mult() {
    DistGraph g(MPI_COMM_SELF);
    g.construct_serial(2, {2, 2}, {{0, 1}, {1}});

    BlockSpMat<double> mat(&g);
    fill_reference_matrix(mat);

    DistVector<double> x(&g);
    DistVector<double> y_ref(&g);
    DistVector<double> y_opt(&g);

    double* x_ptr = x.local_data();
    x_ptr[0] = 1.0;
    x_ptr[1] = 2.0;
    x_ptr[2] = 3.0;
    x_ptr[3] = 4.0;

    mat.mult(x, y_ref);
    mat.mult_optimized(x, y_opt);

    assert_close(
        std::vector<double>(y_ref.local_data(), y_ref.local_data() + y_ref.local_size),
        std::vector<double>(y_opt.local_data(), y_opt.local_data() + y_opt.local_size));
}

void test_construct_serial_root_only() {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int global_blocks = std::max(2, size * 2);
    std::vector<int> block_sizes;
    std::vector<std::vector<int>> adjacency;
    if (rank == 0) {
        block_sizes.assign(global_blocks, 1);
        adjacency.resize(global_blocks);
        for (int gid = 0; gid < global_blocks; ++gid) {
            adjacency[gid].push_back(gid);
            adjacency[gid].push_back((gid + 1) % global_blocks);
        }
    }

    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(rank == 0 ? global_blocks : 0, block_sizes, adjacency);

    const int base = global_blocks / size;
    const int remainder = global_blocks % size;
    const int expected_owned = base + (rank < remainder ? 1 : 0);
    assert(static_cast<int>(graph.owned_global_indices.size()) == expected_owned);
    assert(static_cast<int>(graph.adj_ptr.size()) == expected_owned + 1);
    assert(!graph.block_sizes.empty());
}

void test_dist_multivector_duplicate() {
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, {1, 1}, {{0}, {1}});

    DistMultiVector<double> mv(&graph, 2);
    mv.set_constant(1.0);

    DistMultiVector<double> dup = mv.duplicate();
    mv.scale(3.0);

    for (int col = 0; col < dup.num_vectors; ++col) {
        const double* dup_col = dup.col_data(col);
        const double* mv_col = mv.col_data(col);
        for (int row = 0; row < dup.local_rows; ++row) {
            assert(std::abs(dup_col[row] - 1.0) < 1e-12);
            assert(std::abs(mv_col[row] - 3.0) < 1e-12);
        }
    }
}

BlockSpMat<double> make_shared_graph_duplicate_from_owned_source() {
    DistGraph* graph = new DistGraph(MPI_COMM_SELF);
    graph->construct_serial(2, {1, 1}, {{0, 1}, {1}});

    BlockSpMat<double> owner(graph);
    owner.owns_graph = true;

    const double one = 1.0;
    const double two = 2.0;
    const double three = 3.0;
    owner.add_block(0, 0, &one, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    owner.add_block(0, 1, &two, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    owner.add_block(1, 1, &three, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    owner.assemble();

    return owner.duplicate(false);
}

void test_duplicate_false_keeps_shared_owned_graph_alive() {
    BlockSpMat<double> duplicate = make_shared_graph_duplicate_from_owned_source();

    DistVector<double> x(duplicate.graph);
    DistVector<double> y(duplicate.graph);
    x.local_data()[0] = 1.0;
    x.local_data()[1] = 1.0;
    duplicate.mult(x, y);

    assert(std::abs(y.local_data()[0] - 3.0) < 1e-12);
    assert(std::abs(y.local_data()[1] - 3.0) < 1e-12);
}

void test_duplicate_and_copy_from_require_assembled_remote_state() {
    int size = 1;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (size < 2) {
        return;
    }

    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(2, {1, 1}, {{0}, {1}});

    BlockSpMat<double> mat(&graph);
    BlockSpMat<double> target(&graph);
    const double one = 1.0;

    if (rank == 0) {
        mat.add_block(1, 1, &one, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);

        bool duplicate_threw = false;
        try {
            (void)mat.duplicate();
        } catch (const std::runtime_error&) {
            duplicate_threw = true;
        }
        assert(duplicate_threw);

        bool copy_threw = false;
        try {
            target.copy_from(mat);
        } catch (const std::runtime_error&) {
            copy_threw = true;
        }
        assert(copy_threw);
    }
}

void test_filter_blocks_requires_assembled_remote_state() {
    int size = 1;
    int rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (size < 2) {
        return;
    }

    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(2, {1, 1}, {{0}, {1}});

    BlockSpMat<double> mat(&graph);
    const double one = 1.0;

    if (rank == 0) {
        mat.add_block(1, 1, &one, 1, 1, AssemblyMode::INSERT, MatrixLayout::ColMajor);

        bool threw = false;
        try {
            mat.filter_blocks(0.5);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        assert(threw);
    }
}

void test_utility_surface_matches_expected_dense_behavior() {
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, {2, 3}, {{0, 1}, {0, 1}});

    BlockSpMat<double> mat(&graph);
    const double b00[] = {1.0, 3.0, 2.0, 4.0};
    const double b01[] = {5.0, 8.0, 6.0, 9.0, 7.0, 10.0};
    const double b10[] = {11.0, 13.0, 15.0, 12.0, 14.0, 16.0};
    const double b11[] = {
        17.0, 20.0, 23.0,
        18.0, 21.0, 24.0,
        19.0, 22.0, 25.0};

    mat.add_block(0, 0, b00, 2, 2, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    mat.add_block(0, 1, b01, 2, 3, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    mat.add_block(1, 0, b10, 3, 2, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    mat.add_block(1, 1, b11, 3, 3, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    mat.assemble();

    std::vector<double> dense = mat.to_dense();
    assert(dense.size() == 25);
    assert(mat.matrix_kind() == MatrixKind::VBCSR);
    assert_close(mat.get_block(0, 1), std::vector<double>({5.0, 6.0, 7.0, 8.0, 9.0, 10.0}));
    assert_close(mat.get_values(), std::vector<double>({
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0}));

    BlockSpMat<double> from_dense = mat.duplicate();
    std::vector<double> replacement(25);
    for (size_t i = 0; i < replacement.size(); ++i) {
        replacement[i] = static_cast<double>(i);
    }
    from_dense.from_dense(replacement);
    assert_close(from_dense.to_dense(), replacement);

    DistVector<double> diag(&graph);
    double* diag_ptr = diag.local_data();
    for (int i = 0; i < diag.local_size; ++i) {
        diag_ptr[i] = 1.0;
    }

    BlockSpMat<double> shifted = mat.duplicate();
    BlockSpMat<double> shifted_ref = mat.duplicate();
    shifted.scale(2.0);
    shifted_ref.scale(2.0);
    shifted.shift(0.5);
    shifted_ref.shift(0.5);
    shifted.add_diagonal(diag);
    shifted_ref.add_diagonal(diag);
    assert_matrix_dense_close(shifted, shifted_ref);

    BlockSpMat<double> sub = mat.extract_submatrix({0, 1});
    assert_matrix_dense_close(sub, mat);

    BlockSpMat<double> inserted = mat.duplicate();
    BlockSpMat<double> sub_scaled = sub.duplicate();
    sub_scaled.scale(0.5);
    inserted.insert_submatrix(sub_scaled, {0, 1});
    assert_matrix_dense_close(inserted, sub_scaled);
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    test_add_across_equivalent_graphs();
    test_copy_from_rejects_mismatched_structure();
    test_mult_optimized_matches_mult();
    test_construct_serial_root_only();
    test_dist_multivector_duplicate();
    test_duplicate_false_keeps_shared_owned_graph_alive();
    test_duplicate_and_copy_from_require_assembled_remote_state();
    test_filter_blocks_requires_assembled_remote_state();
    test_utility_surface_matches_expected_dense_behavior();

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "test_migration_contract PASSED" << std::endl;
    }

    MPI_Finalize();
    return 0;
}

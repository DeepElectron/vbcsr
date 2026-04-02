#include "../block_csr.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace vbcsr;

namespace {

template <typename T>
using RealType = typename ScalarTraits<T>::real_type;

struct MatrixProfile {
    const char* name;
    MatrixKind kind;
    std::vector<int> block_sizes;
    std::vector<std::vector<int>> adj_a;
    std::vector<std::vector<int>> adj_b;
};

template <typename T>
constexpr bool is_complex_v = std::is_same_v<T, std::complex<double>>;

template <typename T>
const char* dtype_name() {
    if constexpr (std::is_same_v<T, double>) {
        return "double";
    }
    return "complex<double>";
}

template <typename T>
T draw_value(std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(-0.75, 0.75);
    if constexpr (is_complex_v<T>) {
        return T(dist(gen), dist(gen));
    }
    return static_cast<T>(dist(gen));
}

template <typename T>
T axpby_alpha_value() {
    if constexpr (is_complex_v<T>) {
        return T(0.5, -0.25);
    }
    return T(0.5);
}

template <typename T>
T axpby_beta_value() {
    if constexpr (is_complex_v<T>) {
        return T(-1.25, 0.75);
    }
    return T(-1.25);
}

template <typename T>
T scale_alpha_value() {
    if constexpr (is_complex_v<T>) {
        return T(-0.5, 0.25);
    }
    return T(-0.5);
}

template <typename T>
T shift_alpha_value() {
    if constexpr (is_complex_v<T>) {
        return T(0.2, -0.1);
    }
    return T(0.2);
}

template <typename T>
double component_abs(const T& value) {
    return std::abs(value);
}

template <typename T>
void assert_close(
    const std::vector<T>& got,
    const std::vector<T>& expected,
    double abs_tol = 1e-11,
    double rel_tol = 1e-10) {
    assert(got.size() == expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        const double scale = std::max(1.0, component_abs(expected[i]));
        const double tol = abs_tol + rel_tol * scale;
        if (component_abs(got[i] - expected[i]) > tol) {
            std::cerr << "Mismatch at index " << i
                      << ": got=" << got[i]
                      << " expected=" << expected[i]
                      << " tol=" << tol << std::endl;
        }
        assert(component_abs(got[i] - expected[i]) <= tol);
    }
}

std::vector<int> block_offsets(const std::vector<int>& block_sizes) {
    std::vector<int> offsets(block_sizes.size() + 1, 0);
    for (size_t i = 0; i < block_sizes.size(); ++i) {
        offsets[i + 1] = offsets[i] + block_sizes[i];
    }
    return offsets;
}

int total_scalars(const std::vector<int>& block_sizes) {
    return std::accumulate(block_sizes.begin(), block_sizes.end(), 0);
}

template <typename T>
std::vector<T> dense_adjoint(const std::vector<T>& dense, int rows, int cols) {
    std::vector<T> result(cols * rows, T(0));
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            T value = dense[row * cols + col];
            if constexpr (is_complex_v<T>) {
                value = std::conj(value);
            }
            result[col * rows + row] = value;
        }
    }
    return result;
}

template <typename T>
std::vector<T> dense_matvec(
    const std::vector<T>& dense,
    int rows,
    int cols,
    const std::vector<T>& x) {
    std::vector<T> y(rows, T(0));
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            y[row] += dense[row * cols + col] * x[col];
        }
    }
    return y;
}

template <typename T>
std::vector<T> dense_matvec_adjoint(
    const std::vector<T>& dense,
    int rows,
    int cols,
    const std::vector<T>& x) {
    std::vector<T> y(cols, T(0));
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            T value = dense[row * cols + col];
            if constexpr (is_complex_v<T>) {
                value = std::conj(value);
            }
            y[col] += value * x[row];
        }
    }
    return y;
}

template <typename T>
std::vector<T> dense_matmat(
    const std::vector<T>& dense,
    int rows,
    int cols,
    const std::vector<T>& x_col_major,
    int num_vecs) {
    std::vector<T> y(rows * num_vecs, T(0));
    for (int vec = 0; vec < num_vecs; ++vec) {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                y[vec * rows + row] += dense[row * cols + col] * x_col_major[vec * cols + col];
            }
        }
    }
    return y;
}

template <typename T>
std::vector<T> dense_matmat_adjoint(
    const std::vector<T>& dense,
    int rows,
    int cols,
    const std::vector<T>& x_col_major,
    int num_vecs) {
    std::vector<T> y(cols * num_vecs, T(0));
    for (int vec = 0; vec < num_vecs; ++vec) {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                T value = dense[row * cols + col];
                if constexpr (is_complex_v<T>) {
                    value = std::conj(value);
                }
                y[vec * cols + col] += value * x_col_major[vec * rows + row];
            }
        }
    }
    return y;
}

template <typename T>
std::vector<T> dense_matmul(
    const std::vector<T>& a,
    int a_rows,
    int a_cols,
    const std::vector<T>& b,
    int b_cols) {
    std::vector<T> c(a_rows * b_cols, T(0));
    for (int row = 0; row < a_rows; ++row) {
        for (int inner = 0; inner < a_cols; ++inner) {
            const T a_value = a[row * a_cols + inner];
            for (int col = 0; col < b_cols; ++col) {
                c[row * b_cols + col] += a_value * b[inner * b_cols + col];
            }
        }
    }
    return c;
}

template <typename T>
std::vector<T> dense_axpby(const std::vector<T>& x, const std::vector<T>& y, T alpha, T beta) {
    assert(x.size() == y.size());
    std::vector<T> result(x.size(), T(0));
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = alpha * x[i] + beta * y[i];
    }
    return result;
}

template <typename T>
void dense_scale_in_place(std::vector<T>& dense, T alpha) {
    for (T& value : dense) {
        value *= alpha;
    }
}

template <typename T>
void dense_shift_in_place(std::vector<T>& dense, int rows, int cols, T alpha) {
    const int diag_size = std::min(rows, cols);
    for (int i = 0; i < diag_size; ++i) {
        dense[i * cols + i] += alpha;
    }
}

template <typename T>
void dense_add_diagonal_in_place(std::vector<T>& dense, int rows, int cols, const std::vector<T>& diag) {
    const int diag_size = std::min(rows, cols);
    assert(static_cast<int>(diag.size()) >= diag_size);
    for (int i = 0; i < diag_size; ++i) {
        dense[i * cols + i] += diag[i];
    }
}

template <typename T>
std::vector<T> dense_conjugated(const std::vector<T>& dense) {
    std::vector<T> result(dense);
    if constexpr (is_complex_v<T>) {
        for (T& value : result) {
            value = std::conj(value);
        }
    }
    return result;
}

template <typename T>
std::vector<RealType<T>> dense_real_part(const std::vector<T>& dense) {
    std::vector<RealType<T>> result(dense.size(), RealType<T>(0));
    for (size_t i = 0; i < dense.size(); ++i) {
        if constexpr (is_complex_v<T>) {
            result[i] = dense[i].real();
        } else {
            result[i] = dense[i];
        }
    }
    return result;
}

template <typename T>
std::vector<RealType<T>> dense_imag_part(const std::vector<T>& dense) {
    std::vector<RealType<T>> result(dense.size(), RealType<T>(0));
    if constexpr (is_complex_v<T>) {
        for (size_t i = 0; i < dense.size(); ++i) {
            result[i] = dense[i].imag();
        }
    }
    return result;
}

template <typename T>
std::vector<T> dense_commutator_diagonal(
    const std::vector<T>& dense,
    const std::vector<int>& sizes,
    const std::vector<T>& diag) {
    const int rows = total_scalars(sizes);
    std::vector<T> result(rows * rows, T(0));
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < rows; ++col) {
            result[row * rows + col] = dense[row * rows + col] * (diag[col] - diag[row]);
        }
    }
    return result;
}

template <typename T>
struct BuiltMatrix {
    BlockSpMat<T> matrix;
    std::vector<T> dense;
};

template <typename T>
BuiltMatrix<T> build_serial_matrix(
    const std::vector<int>& sizes,
    const std::vector<std::vector<int>>& adjacency,
    int seed) {
    auto graph = std::make_unique<DistGraph>(MPI_COMM_SELF);
    graph->construct_serial(static_cast<int>(sizes.size()), sizes, adjacency);

    BlockSpMat<T> matrix(graph.get());
    matrix.owns_graph = true;
    graph.release();

    const std::vector<int> offsets = block_offsets(sizes);
    const int rows = offsets.back();
    std::vector<T> dense(rows * rows, T(0));
    std::mt19937 gen(seed);

    for (size_t block_row = 0; block_row < adjacency.size(); ++block_row) {
        const int row_dim = sizes[block_row];
        const int row_offset = offsets[block_row];
        for (int block_col : adjacency[block_row]) {
            const int col_dim = sizes[block_col];
            const int col_offset = offsets[block_col];
            std::vector<T> block(row_dim * col_dim, T(0));
            for (T& value : block) {
                value = draw_value<T>(gen);
            }
            matrix.add_block(
                static_cast<int>(block_row),
                block_col,
                block.data(),
                row_dim,
                col_dim,
                AssemblyMode::INSERT,
                MatrixLayout::RowMajor);
            for (int r = 0; r < row_dim; ++r) {
                for (int c = 0; c < col_dim; ++c) {
                    dense[(row_offset + r) * rows + (col_offset + c)] = block[r * col_dim + c];
                }
            }
        }
    }

    matrix.assemble();
    return BuiltMatrix<T>{std::move(matrix), std::move(dense)};
}

template <typename T>
std::vector<T> controlled_random_vector(int size, int seed) {
    std::mt19937 gen(seed);
    std::vector<T> result(size, T(0));
    for (T& value : result) {
        value = draw_value<T>(gen);
    }
    return result;
}

template <typename T>
void load_vector(DistVector<T>& vec, const std::vector<T>& values) {
    assert(vec.local_size == static_cast<int>(values.size()));
    std::copy(values.begin(), values.end(), vec.data.begin());
}

template <typename T>
void load_multivector(DistMultiVector<T>& mv, const std::vector<T>& values) {
    assert(mv.data.size() == values.size());
    std::copy(values.begin(), values.end(), mv.data.begin());
}

template <typename T>
void assert_matrix_dense_close(
    const BlockSpMat<T>& matrix,
    const std::vector<T>& expected_dense,
    double abs_tol = 1e-11,
    double rel_tol = 1e-10) {
    assert_close(matrix.to_dense(), expected_dense, abs_tol, rel_tol);
}

template <typename T>
void run_reference_suite(const MatrixProfile& profile) {
    const int rows = total_scalars(profile.block_sizes);
    const int num_vecs = 3;

    auto a = build_serial_matrix<T>(profile.block_sizes, profile.adj_a, 101);
    auto b = build_serial_matrix<T>(profile.block_sizes, profile.adj_b, 202);
    auto same = build_serial_matrix<T>(profile.block_sizes, profile.adj_a, 303);

    assert(a.matrix.matrix_kind() == profile.kind);
    assert(b.matrix.matrix_kind() == profile.kind);
    assert(same.matrix.matrix_kind() == profile.kind);
    assert_matrix_dense_close(a.matrix, a.dense);
    assert_matrix_dense_close(b.matrix, b.dense);

    const std::vector<T> x_values = controlled_random_vector<T>(rows, 404);
    DistVector<T> x(a.matrix.graph);
    DistVector<T> y(a.matrix.graph);
    DistVector<T> y_opt(a.matrix.graph);
    load_vector(x, x_values);
    a.matrix.mult(x, y);
    a.matrix.mult_optimized(x, y_opt);
    const std::vector<T> expected_y = dense_matvec(a.dense, rows, rows, x_values);
    assert_close(
        std::vector<T>(y.local_data(), y.local_data() + y.local_size),
        expected_y);
    assert_close(
        std::vector<T>(y_opt.local_data(), y_opt.local_data() + y_opt.local_size),
        expected_y);

    const std::vector<T> x_adj_values = controlled_random_vector<T>(rows, 505);
    DistVector<T> x_adj(a.matrix.graph);
    DistVector<T> y_adj(a.matrix.graph);
    load_vector(x_adj, x_adj_values);
    a.matrix.mult_adjoint(x_adj, y_adj);
    const std::vector<T> expected_adj = dense_matvec_adjoint(a.dense, rows, rows, x_adj_values);
    assert_close(
        std::vector<T>(y_adj.local_data(), y_adj.local_data() + y_adj.local_size),
        expected_adj);

    const std::vector<T> multi_values = controlled_random_vector<T>(rows * num_vecs, 606);
    DistMultiVector<T> x_multi(a.matrix.graph, num_vecs);
    DistMultiVector<T> y_multi(a.matrix.graph, num_vecs);
    load_multivector(x_multi, multi_values);
    a.matrix.mult_dense(x_multi, y_multi);
    assert_close(y_multi.data, dense_matmat(a.dense, rows, rows, multi_values, num_vecs));

    const std::vector<T> multi_adj_values = controlled_random_vector<T>(rows * num_vecs, 707);
    DistMultiVector<T> x_multi_adj(a.matrix.graph, num_vecs);
    DistMultiVector<T> y_multi_adj(a.matrix.graph, num_vecs);
    load_multivector(x_multi_adj, multi_adj_values);
    a.matrix.mult_dense_adjoint(x_multi_adj, y_multi_adj);
    assert_close(y_multi_adj.data, dense_matmat_adjoint(a.dense, rows, rows, multi_adj_values, num_vecs));

    BlockSpMat<T> transposed = a.matrix.transpose();
    assert(transposed.matrix_kind() == profile.kind);
    assert_matrix_dense_close(transposed, dense_adjoint(a.dense, rows, rows));

    BlockSpMat<T> spmm = a.matrix.spmm(b.matrix, 0.0);
    assert(spmm.matrix_kind() == profile.kind);
    assert_matrix_dense_close(spmm, dense_matmul(a.dense, rows, rows, b.dense, rows));

    const std::vector<T> dense_a_adj = dense_adjoint(a.dense, rows, rows);
    const std::vector<T> dense_b_adj = dense_adjoint(b.dense, rows, rows);
    BlockSpMat<T> spmm_trans_a = a.matrix.spmm(b.matrix, 0.0, true, false);
    assert(spmm_trans_a.matrix_kind() == profile.kind);
    assert_matrix_dense_close(spmm_trans_a, dense_matmul(dense_a_adj, rows, rows, b.dense, rows));

    BlockSpMat<T> spmm_trans_b = a.matrix.spmm(b.matrix, 0.0, false, true);
    assert(spmm_trans_b.matrix_kind() == profile.kind);
    assert_matrix_dense_close(spmm_trans_b, dense_matmul(a.dense, rows, rows, dense_b_adj, rows));

    BlockSpMat<T> spmm_self = a.matrix.spmm_self(0.0);
    assert(spmm_self.matrix_kind() == profile.kind);
    assert_matrix_dense_close(spmm_self, dense_matmul(a.dense, rows, rows, a.dense, rows));

    BlockSpMat<T> spmm_self_trans = a.matrix.spmm_self(0.0, true);
    assert(spmm_self_trans.matrix_kind() == profile.kind);
    assert_matrix_dense_close(spmm_self_trans, dense_matmul(dense_a_adj, rows, rows, a.dense, rows));

    const T axpby_alpha = axpby_alpha_value<T>();
    const T axpby_beta = axpby_beta_value<T>();
    BlockSpMat<T> same_structure = a.matrix.duplicate();
    same_structure.axpby(axpby_alpha, same.matrix, axpby_beta);
    assert(same_structure.matrix_kind() == profile.kind);
    assert_matrix_dense_close(same_structure, dense_axpby(same.dense, a.dense, axpby_alpha, axpby_beta));

    BlockSpMat<T> union_structure = a.matrix.duplicate();
    union_structure.axpby(axpby_alpha, b.matrix, axpby_beta);
    assert(union_structure.matrix_kind() == profile.kind);
    assert_matrix_dense_close(union_structure, dense_axpby(b.dense, a.dense, axpby_alpha, axpby_beta));

    BlockSpMat<T> added = a.matrix.add(b.matrix, 0.75, -1.5);
    assert(added.matrix_kind() == profile.kind);
    assert_matrix_dense_close(added, dense_axpby(a.dense, b.dense, T(0.75), T(-1.5)));

    const T scale_alpha = scale_alpha_value<T>();
    BlockSpMat<T> scaled = a.matrix.duplicate();
    std::vector<T> expected_scaled = a.dense;
    dense_scale_in_place(expected_scaled, scale_alpha);
    scaled.scale(scale_alpha);
    assert_matrix_dense_close(scaled, expected_scaled);

    const T shift_alpha = shift_alpha_value<T>();
    BlockSpMat<T> shifted = a.matrix.duplicate();
    std::vector<T> expected_shifted = a.dense;
    dense_shift_in_place(expected_shifted, rows, rows, shift_alpha);
    shifted.shift(shift_alpha);
    assert_matrix_dense_close(shifted, expected_shifted);

    const std::vector<T> diag_values = controlled_random_vector<T>(rows, 808);
    DistVector<T> diag(a.matrix.graph);
    load_vector(diag, diag_values);
    BlockSpMat<T> with_diagonal = a.matrix.duplicate();
    std::vector<T> expected_with_diagonal = a.dense;
    dense_add_diagonal_in_place(expected_with_diagonal, rows, rows, diag_values);
    with_diagonal.add_diagonal(diag);
    assert_matrix_dense_close(with_diagonal, expected_with_diagonal);

    BlockSpMat<T> commutator = a.matrix.duplicate();
    commutator.fill(T(0));
    a.matrix.commutator_diagonal(diag, commutator);
    assert_matrix_dense_close(
        commutator,
        dense_commutator_diagonal(a.dense, profile.block_sizes, diag_values));

    if constexpr (is_complex_v<T>) {
        BlockSpMat<T> conjugated = a.matrix.duplicate();
        conjugated.conjugate();
        assert_matrix_dense_close(conjugated, dense_conjugated(a.dense));

        auto real_part = a.matrix.get_real();
        auto imag_part = a.matrix.get_imag();
        assert(real_part.matrix_kind() == profile.kind);
        assert(imag_part.matrix_kind() == profile.kind);
        assert_close(real_part.to_dense(), dense_real_part(a.dense));
        assert_close(imag_part.to_dense(), dense_imag_part(a.dense));
    }
}

void run_all_reference_suites() {
    const std::vector<MatrixProfile> profiles = {
        {
            "csr",
            MatrixKind::CSR,
            {1, 1, 1, 1},
            {{0, 1, 3}, {0, 1, 2}, {1, 2, 3}, {0, 3}},
            {{0, 2}, {1, 2, 3}, {0, 2, 3}, {1, 3}},
        },
        {
            "bsr",
            MatrixKind::BSR,
            {2, 2, 2, 2},
            {{0, 1, 3}, {0, 1, 2}, {1, 2, 3}, {0, 3}},
            {{0, 2}, {1, 2, 3}, {0, 2, 3}, {1, 3}},
        },
        {
            "vbcsr",
            MatrixKind::VBCSR,
            {2, 3, 1, 2},
            {{0, 1, 3}, {0, 1, 2}, {1, 2, 3}, {0, 3}},
            {{0, 2}, {1, 2, 3}, {0, 2, 3}, {1, 3}},
        },
    };

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    for (const MatrixProfile& profile : profiles) {
        run_reference_suite<double>(profile);
        run_reference_suite<std::complex<double>>(profile);
        if (world_rank == 0) {
            std::cout << "Numerical reference suite passed for " << profile.name
                      << " in double and complex<double>" << std::endl;
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    run_all_reference_suites();

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "test_numeric_reference PASSED" << std::endl;
    }

    MPI_Finalize();
    return 0;
}

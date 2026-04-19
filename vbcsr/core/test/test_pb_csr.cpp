#include "../block_csr.hpp"
#include <iostream>
#include <cassert>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cmath>
#include <complex>
#include <map>
#include <set>
#include <type_traits>

using namespace vbcsr;

template <typename U, typename = void>
struct has_clear_member : std::false_type {};

template <typename U>
struct has_clear_member<U, std::void_t<decltype(std::declval<U&>().clear())>> : std::true_type {};

template <typename U, typename = void>
struct has_blk_sizes_member : std::false_type {};

template <typename U>
struct has_blk_sizes_member<U, std::void_t<decltype(std::declval<U>().blk_sizes)>> : std::true_type {};

using TestMatrix = BlockSpMat<double>;
using TestRowPtrView = decltype(std::declval<TestMatrix&>().row_ptr());

static_assert(std::is_const_v<std::remove_reference_t<decltype(std::declval<TestMatrix&>().row_ptr()[0])>>);
static_assert(!std::is_assignable_v<decltype(std::declval<TestMatrix&>().row_ptr()[0]), int>);
static_assert(!has_clear_member<TestRowPtrView>::value);
static_assert(!has_blk_sizes_member<detail::BSRMatrixBackend<double>>::value);
static_assert(!has_blk_sizes_member<detail::VBCSRMatrixBackend<double, DefaultKernel<double>>>::value);

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

std::vector<double> dense_matmul(
    const std::vector<double>& A,
    int a_rows,
    int a_cols,
    const std::vector<double>& B,
    int b_cols) {
    std::vector<double> C(a_rows * b_cols, 0.0);
    for (int row = 0; row < a_rows; ++row) {
        for (int inner = 0; inner < a_cols; ++inner) {
            const double a_val = A[row * a_cols + inner];
            for (int col = 0; col < b_cols; ++col) {
                C[row * b_cols + col] += a_val * B[inner * b_cols + col];
            }
        }
    }
    return C;
}

void assert_close(const std::vector<double>& got, const std::vector<double>& expected, double tol = 1e-12) {
    assert(got.size() == expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        assert(std::abs(got[i] - expected[i]) < tol);
    }
}

template <typename T>
T conjugate_if_needed(const T& value) {
    return value;
}

template <typename T>
std::complex<T> conjugate_if_needed(const std::complex<T>& value) {
    return std::conj(value);
}

template <typename T>
std::vector<T> dense_matvec_adjoint_generic(
    const std::vector<T>& A,
    int rows,
    int cols,
    const std::vector<T>& x) {
    std::vector<T> y(cols, T(0));
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            y[col] += conjugate_if_needed(A[row * cols + col]) * x[row];
        }
    }
    return y;
}

template <typename T>
std::vector<T> dense_matvec_generic(
    const std::vector<T>& A,
    int rows,
    int cols,
    const std::vector<T>& x) {
    std::vector<T> y(rows, T(0));
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            y[row] += A[row * cols + col] * x[col];
        }
    }
    return y;
}

template <typename T>
std::vector<T> dense_matmat_generic(
    const std::vector<T>& A,
    int rows,
    int cols,
    const std::vector<T>& X,
    int x_ld,
    int num_vecs) {
    std::vector<T> Y(static_cast<size_t>(rows) * num_vecs, T(0));
    for (int vec = 0; vec < num_vecs; ++vec) {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                Y[static_cast<size_t>(vec) * rows + row] += A[row * cols + col] * X[static_cast<size_t>(vec) * x_ld + col];
            }
        }
    }
    return Y;
}

template <typename T>
std::vector<T> dense_matmat_adjoint_generic(
    const std::vector<T>& A,
    int rows,
    int cols,
    const std::vector<T>& X,
    int x_ld,
    int num_vecs) {
    std::vector<T> Y(static_cast<size_t>(cols) * num_vecs, T(0));
    for (int vec = 0; vec < num_vecs; ++vec) {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                Y[static_cast<size_t>(vec) * cols + col] +=
                    conjugate_if_needed(A[row * cols + col]) * X[static_cast<size_t>(vec) * x_ld + row];
            }
        }
    }
    return Y;
}

template <typename T>
void assert_close_generic(const std::vector<T>& got, const std::vector<T>& expected, double tol = 1e-12) {
    assert(got.size() == expected.size());
    for (size_t i = 0; i < got.size(); ++i) {
        assert(std::abs(got[i] - expected[i]) < tol);
    }
}

template <typename T>
T make_test_value(double real, double imag = 0.0) {
    if constexpr (std::is_same_v<T, std::complex<double>> ||
                  std::is_same_v<T, std::complex<float>>) {
        return T(real, imag);
    } else {
        (void)imag;
        return static_cast<T>(real);
    }
}

template <typename T>
double sq_norm_generic(const T& value) {
    if constexpr (std::is_same_v<T, std::complex<double>> ||
                  std::is_same_v<T, std::complex<float>>) {
        return std::norm(value);
    } else {
        return static_cast<double>(value * value);
    }
}

template <typename T>
int dense_row_count(const DistGraph& graph) {
    return graph.block_offsets[graph.owned_global_indices.size()];
}

template <typename T>
int dense_col_count(const DistGraph& graph) {
    return graph.block_offsets.back();
}

template <typename T>
std::vector<T> dense_matmul_generic(
    const std::vector<T>& A,
    int a_rows,
    int a_cols,
    const std::vector<T>& B,
    int b_cols) {
    std::vector<T> C(static_cast<size_t>(a_rows) * b_cols, T(0));
    for (int row = 0; row < a_rows; ++row) {
        for (int inner = 0; inner < a_cols; ++inner) {
            const T a_val = A[static_cast<size_t>(row) * a_cols + inner];
            for (int col = 0; col < b_cols; ++col) {
                C[static_cast<size_t>(row) * b_cols + col] +=
                    a_val * B[static_cast<size_t>(inner) * b_cols + col];
            }
        }
    }
    return C;
}

template <typename T>
std::vector<T> dense_transpose_conjugate_generic(
    const std::vector<T>& A,
    int rows,
    int cols) {
    std::vector<T> AT(static_cast<size_t>(cols) * rows, T(0));
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            AT[static_cast<size_t>(col) * rows + row] =
                conjugate_if_needed(A[static_cast<size_t>(row) * cols + col]);
        }
    }
    return AT;
}

template <typename T>
std::vector<typename ScalarTraits<T>::real_type> dense_real_part(
    const std::vector<T>& A) {
    using RealT = typename ScalarTraits<T>::real_type;
    std::vector<RealT> out(A.size(), RealT(0));
    for (size_t idx = 0; idx < A.size(); ++idx) {
        if constexpr (std::is_same_v<T, std::complex<double>> ||
                      std::is_same_v<T, std::complex<float>>) {
            out[idx] = A[idx].real();
        } else {
            out[idx] = static_cast<RealT>(A[idx]);
        }
    }
    return out;
}

template <typename T>
std::vector<typename ScalarTraits<T>::real_type> dense_imag_part(
    const std::vector<T>& A) {
    using RealT = typename ScalarTraits<T>::real_type;
    std::vector<RealT> out(A.size(), RealT(0));
    for (size_t idx = 0; idx < A.size(); ++idx) {
        if constexpr (std::is_same_v<T, std::complex<double>> ||
                      std::is_same_v<T, std::complex<float>>) {
            out[idx] = A[idx].imag();
        }
    }
    return out;
}

template <typename T>
std::vector<T> dense_conjugated(const std::vector<T>& A) {
    std::vector<T> out = A;
    for (auto& value : out) {
        value = conjugate_if_needed(value);
    }
    return out;
}

template <typename T>
std::vector<T> dense_commutator_diagonal(
    const std::vector<T>& A,
    int rows,
    int cols,
    const std::vector<T>& diag) {
    std::vector<T> out(A.size(), T(0));
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            out[static_cast<size_t>(row) * cols + col] =
                A[static_cast<size_t>(row) * cols + col] *
                (diag[col] - diag[row]);
        }
    }
    return out;
}

template <typename T>
T generated_block_entry(
    int matrix_id,
    int row,
    int col,
    int r,
    int c) {
    double sign = ((matrix_id + row + 2 * col + r + c) % 2 == 0) ? 1.0 : -1.0;
    double real = sign * (0.8 + 0.45 * matrix_id + 0.55 * row + 0.35 * col + 0.09 * r + 0.04 * c);
    double imag = sign * (0.2 + 0.18 * matrix_id - 0.06 * row + 0.05 * col + 0.03 * r - 0.02 * c);
    if (matrix_id == 0 && row == 2 && col == 1) {
        real *= 0.01;
        imag *= 0.01;
    }
    return make_test_value<T>(real, imag);
}

template <typename T>
std::vector<T> generated_block_row_major(
    int matrix_id,
    int row,
    int col,
    int row_dim,
    int col_dim) {
    std::vector<T> block(static_cast<size_t>(row_dim) * col_dim, T(0));
    for (int r = 0; r < row_dim; ++r) {
        for (int c = 0; c < col_dim; ++c) {
            block[static_cast<size_t>(r) * col_dim + c] =
                generated_block_entry<T>(matrix_id, row, col, r, c);
        }
    }
    return block;
}

template <typename T>
void write_block_col_major_from_row_major(
    T* dest,
    const std::vector<T>& src,
    int row_dim,
    int col_dim) {
    for (int r = 0; r < row_dim; ++r) {
        for (int c = 0; c < col_dim; ++c) {
            dest[static_cast<size_t>(c) * row_dim + r] =
                src[static_cast<size_t>(r) * col_dim + c];
        }
    }
}

template <typename T>
std::vector<T> generated_dense_matrix(
    const DistGraph& graph,
    int matrix_id) {
    const int rows = dense_row_count<T>(graph);
    const int cols = dense_col_count<T>(graph);
    std::vector<T> dense(static_cast<size_t>(rows) * cols, T(0));
    const int n_rows = static_cast<int>(graph.owned_global_indices.size());
    for (int row = 0; row < n_rows; ++row) {
        const int row_dim = graph.block_sizes[row];
        const int row_offset = graph.block_offsets[row];
        for (int graph_block_index = graph.adj_ptr[row];
             graph_block_index < graph.adj_ptr[row + 1];
             ++graph_block_index) {
            const int col = graph.adj_ind[graph_block_index];
            const int col_dim = graph.block_sizes[col];
            const int col_offset = graph.block_offsets[col];
            const auto block = generated_block_row_major<T>(
                matrix_id,
                row,
                col,
                row_dim,
                col_dim);
            for (int r = 0; r < row_dim; ++r) {
                for (int c = 0; c < col_dim; ++c) {
                    dense[static_cast<size_t>(row_offset + r) * cols + (col_offset + c)] =
                        block[static_cast<size_t>(r) * col_dim + c];
                }
            }
        }
    }
    return dense;
}

template <typename T>
void populate_matrix_with_generated_blocks(
    BlockSpMat<T>& matrix,
    const DistGraph& graph,
    int matrix_id) {
    const int n_rows = static_cast<int>(graph.owned_global_indices.size());
    for (int row = 0; row < n_rows; ++row) {
        const int row_dim = graph.block_sizes[row];
        for (int graph_block_index = graph.adj_ptr[row];
             graph_block_index < graph.adj_ptr[row + 1];
             ++graph_block_index) {
            const int col = graph.adj_ind[graph_block_index];
            const int col_dim = graph.block_sizes[col];
            const auto block = generated_block_row_major<T>(
                matrix_id,
                row,
                col,
                row_dim,
                col_dim);
            write_block_col_major_from_row_major(
                matrix.mutable_block_data(graph_block_index),
                block,
                row_dim,
                col_dim);
        }
    }
}

template <typename T>
BlockSpMat<T> build_generated_matrix_via_api(
    DistGraph* graph,
    int matrix_id,
    uint32_t page_size) {
    BlockSpMat<T> matrix(graph);
    matrix.set_page_size(page_size);
    const int n_rows = static_cast<int>(graph->owned_global_indices.size());
    for (int row = 0; row < n_rows; ++row) {
        const int row_dim = graph->block_sizes[row];
        for (int graph_block_index = graph->adj_ptr[row];
             graph_block_index < graph->adj_ptr[row + 1];
             ++graph_block_index) {
            const int col = graph->adj_ind[graph_block_index];
            const int col_dim = graph->block_sizes[col];
            auto block = generated_block_row_major<T>(
                matrix_id,
                row,
                col,
                row_dim,
                col_dim);
            matrix.add_block(
                graph->get_global_index(row),
                graph->get_global_index(col),
                block.data(),
                row_dim,
                col_dim,
                AssemblyMode::INSERT,
                MatrixLayout::RowMajor);
        }
    }
    matrix.assemble();
    return matrix;
}

template <typename T>
BlockSpMat<T> build_generated_matrix_via_backend(
    DistGraph* graph,
    MatrixKind kind,
    int matrix_id,
    uint32_t page_size) {
    BlockSpMat<T> matrix(graph);
    matrix.set_page_size(page_size);
    assert(matrix.matrix_kind() == kind);
    populate_matrix_with_generated_blocks<T>(matrix, *graph, matrix_id);
    return matrix;
}

template <typename T>
void fill_generated_vector(std::vector<T>& data, double base_shift) {
    for (size_t idx = 0; idx < data.size(); ++idx) {
        const double real = base_shift + 0.17 * static_cast<double>(idx + 1);
        const double imag = -0.11 * static_cast<double>(idx + 1);
        data[idx] = make_test_value<T>(real, imag);
    }
}

template <typename T>
void fill_generated_multivector(
    std::vector<T>& data,
    int ld,
    int num_vecs,
    double base_shift) {
    for (int vec = 0; vec < num_vecs; ++vec) {
        for (int row = 0; row < ld; ++row) {
            const double real = base_shift + 0.09 * (vec + 1) + 0.13 * (row + 1);
            const double imag = -0.04 * (vec + 1) + 0.07 * (row + 1);
            data[static_cast<size_t>(vec) * ld + row] = make_test_value<T>(real, imag);
        }
    }
}

template <typename T>
std::vector<T> dense_after_filter_expected(
    const BlockSpMat<T>& matrix,
    double threshold) {
    std::vector<T> dense = matrix.to_dense();
    matrix.for_each_local_block([&](const auto& block) {
        double norm_sq = 0.0;
        for (size_t idx = 0; idx < block.size; ++idx) {
            norm_sq += sq_norm_generic(block.values[idx]);
        }
        if (std::sqrt(norm_sq) >= threshold) {
            return;
        }
        const int rows = dense_row_count<T>(*matrix.graph);
        const int cols = dense_col_count<T>(*matrix.graph);
        const int row_offset = matrix.graph->block_offsets[block.row];
        const int col_offset = matrix.graph->block_offsets[block.col];
        for (int r = 0; r < block.row_dim; ++r) {
            for (int c = 0; c < block.col_dim; ++c) {
                dense[static_cast<size_t>(row_offset + r) * cols + (col_offset + c)] = T(0);
            }
        }
    });
    return dense;
}

template <typename T>
std::vector<T> extract_dense_block_submatrix(
    const BlockSpMat<T>& matrix,
    const std::vector<int>& global_indices) {
    const auto dense = matrix.to_dense();
    const DistGraph& graph = *matrix.graph;
    int sub_rows = 0;
    int sub_cols = 0;
    for (int global_idx : global_indices) {
        const int local_idx = graph.global_to_local.at(global_idx);
        sub_rows += graph.block_sizes[local_idx];
        sub_cols += graph.block_sizes[local_idx];
    }
    std::vector<T> sub_dense(static_cast<size_t>(sub_rows) * sub_cols, T(0));
    int out_row_offset = 0;
    for (int global_row : global_indices) {
        const int local_row = graph.global_to_local.at(global_row);
        const int row_dim = graph.block_sizes[local_row];
        const int src_row_offset = graph.block_offsets[local_row];
        int out_col_offset = 0;
        for (int global_col : global_indices) {
            const int local_col = graph.global_to_local.at(global_col);
            const int col_dim = graph.block_sizes[local_col];
            const int src_col_offset = graph.block_offsets[local_col];
            for (int r = 0; r < row_dim; ++r) {
                for (int c = 0; c < col_dim; ++c) {
                    sub_dense[static_cast<size_t>(out_row_offset + r) * sub_cols + (out_col_offset + c)] =
                        dense[static_cast<size_t>(src_row_offset + r) * dense_col_count<T>(graph) + (src_col_offset + c)];
                }
            }
            out_col_offset += col_dim;
        }
        out_row_offset += row_dim;
    }
    return sub_dense;
}

template <typename T>
std::vector<T> with_inserted_dense_block_submatrix(
    const BlockSpMat<T>& matrix,
    const std::vector<int>& global_indices,
    const std::vector<T>& sub_dense) {
    std::vector<T> dense = matrix.to_dense();
    const DistGraph& graph = *matrix.graph;
    const int cols = dense_col_count<T>(graph);
    int sub_cols = 0;
    for (int global_idx : global_indices) {
        sub_cols += graph.block_sizes[graph.global_to_local.at(global_idx)];
    }

    int in_row_offset = 0;
    for (int global_row : global_indices) {
        const int local_row = graph.global_to_local.at(global_row);
        const int row_dim = graph.block_sizes[local_row];
        const int dest_row_offset = graph.block_offsets[local_row];
        int in_col_offset = 0;
        for (int global_col : global_indices) {
            const int local_col = graph.global_to_local.at(global_col);
            const int col_dim = graph.block_sizes[local_col];
            const int dest_col_offset = graph.block_offsets[local_col];
            for (int r = 0; r < row_dim; ++r) {
                for (int c = 0; c < col_dim; ++c) {
                    dense[static_cast<size_t>(dest_row_offset + r) * cols + (dest_col_offset + c)] =
                        sub_dense[static_cast<size_t>(in_row_offset + r) * sub_cols + (in_col_offset + c)];
                }
            }
            in_col_offset += col_dim;
        }
        in_row_offset += row_dim;
    }
    return dense;
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
    Y.assemble();
    
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
    assert(Y.col_ind().empty());
    
    // Now Y is empty. X has (0,0).
    BlockSpMat<double> X(&graph); // X has (0,0)
    X.add_block(0, 0, d00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    X.assemble();
    
    // Y = 1.0 * X + 1.0 * Y
    // Y should become X
    Y.axpby(1.0, X, 1.0);
    
    if (Y.graph->owned_global_indices.size() > 0) {
        assert(Y.col_ind().size() == 1);
        assert(Y.col_ind()[0] == 0);
        
        // Check values
        const double* ptr = Y.block_data(0);
        assert(ptr[0] == 1.0);
    } else {
        assert(Y.col_ind().empty());
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
        assert(mat.col_ind().size() == 1);
    } else {
        assert(mat.col_ind().empty());
    }
    std::cout << "PASSED" << std::endl;
}

void test_const_logical_views_and_family_rejection() {
    std::cout << "Testing Const Logical Views And Family Rejection..." << std::endl;

    DistGraph csr_graph(MPI_COMM_SELF);
    csr_graph.construct_serial(2, {1, 1}, {{0, 1}, {1}});
    BlockSpMat<double> csr(&csr_graph);
    double a00[] = {2.0};
    double a01[] = {3.0};
    double a11[] = {5.0};
    csr.add_block(0, 0, a00, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    csr.add_block(0, 1, a01, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    csr.add_block(1, 1, a11, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    csr.assemble();

    assert(csr.row_ptr() == csr.row_ptr());
    assert(csr.col_ind() == csr.col_ind());

    BlockSpMat<double> csr_t = csr.transpose();
    assert(csr_t.row_ptr() == csr_t.row_ptr());
    assert(csr_t.col_ind() == csr_t.col_ind());

    DistGraph bsr_graph(MPI_COMM_SELF);
    bsr_graph.construct_serial(2, {2, 2}, {{0, 1}, {1}});
    BlockSpMat<double> bsr(&bsr_graph);
    double b00[] = {1.0, 0.0, 0.0, 1.0};
    double b01[] = {1.0, 2.0, 3.0, 4.0};
    double b11[] = {5.0, 6.0, 7.0, 8.0};
    bsr.add_block(0, 0, b00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    bsr.add_block(0, 1, b01, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    bsr.add_block(1, 1, b11, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    bsr.assemble();

    bool threw = false;
    try {
        csr.axpy(1.0, bsr);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        (void)csr.add(bsr);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        (void)csr.spmm(bsr, 0.0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    DistGraph vb_graph_a(MPI_COMM_SELF);
    DistGraph vb_graph_b(MPI_COMM_SELF);
    vb_graph_a.construct_serial(2, {2, 3}, {{0}, {1}});
    vb_graph_b.construct_serial(2, {3, 2}, {{0}, {1}});
    BlockSpMat<double> vb_a(&vb_graph_a);
    BlockSpMat<double> vb_b(&vb_graph_b);
    assert(vb_a.matrix_kind() == MatrixKind::VBCSR);
    assert(vb_b.matrix_kind() == MatrixKind::VBCSR);

    threw = false;
    try {
        vb_a.axpy(1.0, vb_b);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);

    std::cout << "PASSED" << std::endl;
}

void test_paged_storage_contracts() {
    std::cout << "Testing Paged Storage Contracts..." << std::endl;

    detail::PagedBuffer<double> lhs(4);
    detail::PagedBuffer<double> rhs(4);
    lhs.reserve(10);
    assert(lhs.capacity() == 12);
    assert(lhs.size() == 0);
    lhs.resize(10);
    rhs.resize(10);
    for (uint64_t i = 0; i < 10; ++i) {
        lhs[i] = static_cast<double>(i + 1);
        rhs[i] = static_cast<double>((i + 1) * 10);
    }
    assert(lhs.page_count() == 3);

    std::vector<double> sliced;
    lhs.for_each_range(2, 9, [&](auto slice) {
        for (uint32_t idx = 0; idx < slice.count; ++idx) {
            sliced.push_back(slice.data[idx]);
        }
    });
    assert((sliced == std::vector<double>({3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0})));

    std::vector<double> zipped;
    lhs.for_each_zipped_range(rhs, 1, 6, [&](auto lhs_slice, auto rhs_slice) {
        for (uint32_t idx = 0; idx < lhs_slice.count; ++idx) {
            zipped.push_back(lhs_slice.data[idx] + rhs_slice.data[idx]);
        }
    });
    assert((zipped == std::vector<double>({22.0, 33.0, 44.0, 55.0, 66.0})));

    lhs.resize(6);
    lhs.resize(10);
    for (uint64_t idx = 6; idx < 10; ++idx) {
        assert(lhs[idx] == 0.0);
    }

    detail::PagedBuffer<double> copied(4);
    copied.copy_prefix_from(rhs, 6);
    assert(copied.size() == 6);
    for (uint64_t idx = 0; idx < 6; ++idx) {
        assert(copied[idx] == rhs[idx]);
    }

    DistGraph csr_graph(MPI_COMM_SELF);
    csr_graph.construct_serial(3, {1, 1, 1}, {{0, 1, 2}, {1}, {2}});
    detail::CSRMatrixBackend<double> csr_backend;
    csr_backend.initialize_structure(csr_graph.adj_ind.size(), 2);
    *csr_backend.value_ptr(0) = 2.0;
    *csr_backend.value_ptr(1) = 3.0;
    *csr_backend.value_ptr(2) = 5.0;
    *csr_backend.value_ptr(3) = 7.0;
    *csr_backend.value_ptr(4) = 11.0;
    const double extra_scalar = 4.0;
    *csr_backend.value_ptr(1) += 0.5 * extra_scalar;
    assert(csr_backend.values.size() == 5);
    assert(*csr_backend.value_ptr(1) == 5.0);
    assert(csr_backend.page(csr_graph.adj_ind, 0).nnz_count == 2);
    assert(csr_backend.page(csr_graph.adj_ind, 1).nnz_count == 2);
    assert(csr_backend.page(csr_graph.adj_ind, 2).nnz_count == 1);
    std::vector<int> csr_cols;
    std::vector<double> csr_vals;
    std::vector<uint32_t> csr_chunks;
    csr_backend.for_each_row_slice(csr_graph.adj_ptr, csr_graph.adj_ind, 0, [&](auto slice) {
        csr_chunks.push_back(slice.nnz_count);
        for (uint32_t idx = 0; idx < slice.nnz_count; ++idx) {
            csr_cols.push_back(slice.cols[idx]);
            csr_vals.push_back(slice.values[idx]);
        }
    });
    assert((csr_chunks == std::vector<uint32_t>{2u, 1u}));
    assert((csr_cols == std::vector<int>{0, 1, 2}));
    assert((csr_vals == std::vector<double>{2.0, 5.0, 5.0}));

    DistGraph bsr_graph(MPI_COMM_SELF);
    bsr_graph.construct_serial(3, {2, 2, 2}, {{0, 1}, {2}, {}});
    detail::BSRMatrixBackend<double> bsr_backend;
    bsr_backend.initialize_structure(bsr_graph.adj_ind.size(), 2, 1);
    {
        double* slot0 = bsr_backend.block_ptr(0);
        double* slot1 = bsr_backend.block_ptr(1);
        double* slot2 = bsr_backend.block_ptr(2);
        for (int i = 0; i < 4; ++i) {
            slot0[i] = static_cast<double>(i + 1);
            slot1[i] = static_cast<double>(i + 5);
            slot2[i] = static_cast<double>(i + 9);
        }
    }
    std::vector<double> add_block = {1.0, 1.0, 1.0, 1.0};
    for (int i = 0; i < 4; ++i) {
        bsr_backend.block_ptr(1)[i] += 2.0 * add_block[static_cast<size_t>(i)];
    }
    assert(bsr_backend.block_size == 2);
    assert(bsr_backend.values.size() == 12);
    assert(bsr_backend.block_ptr(1)[0] == 7.0);
    assert(bsr_backend.page(bsr_graph.adj_ind, 0).block_count == 1);
    assert(bsr_backend.page(bsr_graph.adj_ind, 1).block_count == 1);
    assert(bsr_backend.page(bsr_graph.adj_ind, 2).block_count == 1);
    const auto& bsr_plan = bsr_backend.ensure_apply_plan(bsr_graph.adj_ptr, bsr_graph.adj_ind);
    assert(bsr_backend.active_blocks_per_page() == 1);
    assert(bsr_plan.batches.size() == 3);
    assert(bsr_plan.batches[0].batch.page_index == 0);
    assert(bsr_plan.batches[0].batch.first_block == 0);
    assert(bsr_plan.batches[0].batch.block_count == 1);
    assert(bsr_plan.batches[0].batch.row_begin == 0);
    assert(bsr_plan.batches[0].batch.row_end == 1);
    assert((bsr_plan.batches[0].row_block_offsets_storage == std::vector<int>{0, 1}));
    assert(bsr_plan.batches[1].batch.page_index == 1);
    assert(bsr_plan.batches[1].batch.first_block == 1);
    assert(bsr_plan.batches[1].batch.block_count == 1);
    assert(bsr_plan.batches[1].batch.row_begin == 0);
    assert(bsr_plan.batches[1].batch.row_end == 1);
    assert((bsr_plan.batches[1].row_block_offsets_storage == std::vector<int>{0, 1}));
    assert(bsr_plan.batches[2].batch.page_index == 2);
    assert(bsr_plan.batches[2].batch.first_block == 2);
    assert(bsr_plan.batches[2].batch.block_count == 1);
    assert(bsr_plan.batches[2].batch.row_begin == 1);
    assert(bsr_plan.batches[2].batch.row_end == 2);
    assert((bsr_plan.batches[2].row_block_offsets_storage == std::vector<int>{0, 1}));
    const auto& batch0 = bsr_plan.batches[0].batch;
    const auto& batch1 = bsr_plan.batches[1].batch;
    const auto& batch2 = bsr_plan.batches[2].batch;
    assert(batch0.row_count() == 1);
    assert(batch0.row_block_start(0) == 0);
    assert(batch0.row_block_end(0) == 1);
    assert(batch1.row_count() == 1);
    assert(batch1.row_block_start(0) == 0);
    assert(batch1.row_block_end(0) == 1);
    assert(batch2.row_count() == 1);
    assert(batch2.row_block_start(1) == 0);
    assert(batch2.row_block_end(1) == 1);
    std::vector<int> bsr_cols;
    std::vector<double> bsr_first_entries;
    std::vector<uint32_t> bsr_chunks;
    for (const auto* batch : {&batch0, &batch1}) {
        bsr_chunks.push_back(batch->block_count);
        for (uint32_t idx = 0; idx < batch->block_count; ++idx) {
            bsr_cols.push_back(batch->cols[idx]);
            bsr_first_entries.push_back(batch->values[idx * batch->block_value_count]);
        }
    }
    assert((bsr_chunks == std::vector<uint32_t>{1u, 1u}));
    assert((bsr_cols == std::vector<int>{0, 1}));
    assert((bsr_first_entries == std::vector<double>{1.0, 7.0}));

    std::cout << "PASSED" << std::endl;
}

void test_vbcsr_batch_views() {
    std::cout << "Testing VBCSR Batch Views..." << std::endl;

    std::vector<int> block_sizes = {2, 2, 3};
    std::vector<std::vector<int>> adj = {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(3, block_sizes, adj);

    BlockSpMat<double> mat(&graph);
    assert(mat.matrix_kind() == MatrixKind::VBCSR);

    size_t total_blocks = 0;
    mat.for_each_shape_batch([&](const auto& batch) {
        assert(batch.block_count() > 0);
        assert(batch.block_ptr(0) != nullptr);
        for (uint32_t idx = 0; idx < batch.block_count(); ++idx) {
            const int graph_block_index = batch.graph_block_index(idx);
            assert(batch.block_ptr(idx) == mat.block_data(graph_block_index));
            assert(batch.row_block_index(idx) == mat.block_row_from_slot(graph_block_index));
            assert(batch.col_block_index(idx) == mat.block_col_from_slot(graph_block_index));
        }
        total_blocks += batch.block_count();
    });
    assert(total_blocks == mat.local_block_nnz());

    std::cout << "PASSED" << std::endl;
}

void test_batched_blas_capability_flags() {
    std::cout << "Testing Batched BLAS Capability Flags..." << std::endl;

#ifdef VBCSR_USE_MKL
    assert(BLASKernel::supports_strided_gemm());
    assert(BLASKernel::supports_strided_gemv());
#elif defined(VBCSR_USE_OPENBLAS)
#ifdef VBCSR_BLAS_HAS_BATCH_GEMM
    assert(BLASKernel::supports_strided_gemm());
#else
    assert(!BLASKernel::supports_strided_gemm());
#endif
    assert(!BLASKernel::supports_strided_gemv());
#else
    assert(!BLASKernel::supports_strided_gemm());
    assert(!BLASKernel::supports_strided_gemv());
#endif

    std::cout << "PASSED" << std::endl;
}

void test_contiguous_api_compatibility() {
    std::cout << "Testing Contiguous API Compatibility..." << std::endl;

    {
        std::vector<int> block_sizes = {1, 1};
        std::vector<std::vector<int>> adj = {{0}, {1}};
        DistGraph graph(MPI_COMM_SELF);
        graph.construct_serial(2, block_sizes, adj);
        BlockSpMat<double> csr(&graph);
        double d00[] = {2.0};
        double d11[] = {3.0};
        csr.add_block(0, 0, d00, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        csr.add_block(1, 1, d11, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        csr.assemble();
        assert(csr.matrix_kind() == MatrixKind::CSR);
        assert(csr.has_contiguous_layout());
        csr.pack_contiguous();
        assert(csr.has_contiguous_layout());
    }

    {
        std::vector<int> block_sizes = {2, 2};
        std::vector<std::vector<int>> adj = {{0}, {1}};
        DistGraph graph(MPI_COMM_SELF);
        graph.construct_serial(2, block_sizes, adj);
        BlockSpMat<double> bsr(&graph);
        double d00[] = {1.0, 0.0, 0.0, 1.0};
        double d11[] = {2.0, 0.0, 0.0, 2.0};
        bsr.add_block(0, 0, d00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        bsr.add_block(1, 1, d11, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        bsr.assemble();
        assert(bsr.matrix_kind() == MatrixKind::BSR);
        assert(bsr.has_contiguous_layout());
        bsr.pack_contiguous();
        assert(bsr.has_contiguous_layout());
    }

    std::vector<int> block_sizes = {2, 3};
    std::vector<std::vector<int>> adj = {{0, 1}, {0, 1}};
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, block_sizes, adj);

    BlockSpMat<double> mat(&graph);
    assert(mat.matrix_kind() == MatrixKind::VBCSR);
    assert(mat.has_contiguous_layout());

    double a00[] = {1.0, 2.0, 3.0, 4.0};
    double a01[] = {5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double a10[] = {11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    double a11[] = {17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0};
    mat.add_block(0, 0, a00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(0, 1, a01, 2, 3, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(1, 0, a10, 3, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(1, 1, a11, 3, 3, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.assemble();
    assert(mat.has_contiguous_layout());

    const std::vector<std::vector<double>> before_blocks = {
        mat.get_block(0, 0),
        mat.get_block(0, 1),
        mat.get_block(1, 0),
        mat.get_block(1, 1)
    };

    size_t total_live_blocks = 0;
    mat.for_each_shape_batch([&](const auto& batch) {
        assert(batch.block_ptr(0) != nullptr);
        assert(batch.block_capacity() >= batch.block_count());
        for (uint32_t idx = 0; idx < batch.block_count(); ++idx) {
            const int graph_block_index = batch.graph_block_index(idx);
            assert(graph_block_index >= 0);
            assert(batch.block_ptr(idx) == mat.block_data(graph_block_index));
            assert(batch.row_block_index(idx) == mat.block_row_from_slot(graph_block_index));
            assert(batch.col_block_index(idx) == mat.block_col_from_slot(graph_block_index));
            if (idx + 1 < batch.block_count()) {
                assert(
                    batch.block_ptr(idx + 1) - batch.block_ptr(idx) ==
                    static_cast<std::ptrdiff_t>(batch.values_per_block()));
            }
        }
        total_live_blocks += batch.block_count();
    });
    assert(total_live_blocks == mat.local_block_nnz());

    assert(mat.get_block(0, 0) == before_blocks[0]);
    assert(mat.get_block(0, 1) == before_blocks[1]);
    assert(mat.get_block(1, 0) == before_blocks[2]);
    assert(mat.get_block(1, 1) == before_blocks[3]);

    double overwrite01[] = {1.0, 0.0, 0.0, 1.0, 2.0, 3.0};
    mat.add_block(0, 1, overwrite01, 2, 3, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    assert(mat.has_contiguous_layout());

    mat.assemble();
    assert(mat.has_contiguous_layout());

    BlockSpMat<double> trans = mat.transpose();
    assert(trans.matrix_kind() == MatrixKind::VBCSR);
    assert(trans.has_contiguous_layout());

    BlockSpMat<double> prod = mat.spmm_self(0.0);
    assert(prod.matrix_kind() == MatrixKind::VBCSR);
    assert(prod.has_contiguous_layout());

    mat.filter_blocks(1e9);
    assert(mat.matrix_kind() == MatrixKind::VBCSR);
    assert(mat.has_contiguous_layout());

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
    assert(mat_T.row_ptr() == mat.row_ptr());
    assert(mat_T.col_ind() == mat.col_ind());

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
    assert(real.row_ptr() == mat.row_ptr());
    assert(imag.col_ind() == mat.col_ind());

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
    assert(mat.row_ptr() == std::vector<int>({0, 2, 4}));
    assert(mat.col_ind() == std::vector<int>({0, 1, 0, 1}));

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
    assert(Y.row_ptr() == std::vector<int>({0, 2, 4}));
    assert(Y.col_ind() == std::vector<int>({0, 1, 0, 1}));
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
    assert(mat.row_ptr() == mat.row_ptr());
    assert(mat.col_ind() == mat.col_ind());

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
    assert(mat_t.row_ptr() == mat_t.row_ptr());
    assert(mat_t.col_ind() == mat_t.col_ind());

    if (size == 1) {
        assert(mat_t.row_ptr() == std::vector<int>({0, 2, 3}));
        assert(mat_t.col_ind() == std::vector<int>({0, 1, 0}));
        assert(mat_t.get_block(0, 0) == std::vector<double>({1.0, 3.0, 2.0, 4.0}));
        assert(mat_t.get_block(0, 1) == std::vector<double>({9.0, 11.0, 10.0, 12.0}));
        assert(mat_t.get_block(1, 0) == std::vector<double>({5.0, 7.0, 6.0, 8.0}));
    } else if (rank == 0) {
        assert(mat_t.row_ptr() == std::vector<int>({0, 2}));
        const int col0 = mat_t.graph->global_to_local.at(0);
        const int col1 = mat_t.graph->global_to_local.at(1);
        assert(mat_t.get_block(0, col0) == std::vector<double>({1.0, 3.0, 2.0, 4.0}));
        assert(mat_t.get_block(0, col1) == std::vector<double>({9.0, 11.0, 10.0, 12.0}));
    } else if (rank == 1) {
        assert(mat_t.row_ptr() == std::vector<int>({0, 1}));
        const int col0 = mat_t.graph->global_to_local.at(0);
        assert(mat_t.get_block(0, col0) == std::vector<double>({5.0, 7.0, 6.0, 8.0}));
    } else {
        assert(mat_t.row_ptr() == std::vector<int>({0}));
        assert(mat_t.col_ind().empty());
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
    assert(C.row_ptr() == C.row_ptr());
    assert(C.col_ind() == C.col_ind());

    if (size == 1) {
        assert(C.row_ptr() == std::vector<int>({0, 2, 4}));
        assert(C.col_ind() == std::vector<int>({0, 1, 0, 1}));
        assert(C.get_block(0, 0) == std::vector<double>({14.0, 0.0, 0.0, 14.0}));
        assert(C.get_block(0, 1) == std::vector<double>({12.0, 0.0, 0.0, 12.0}));
        assert(C.get_block(1, 0) == std::vector<double>({15.0, 0.0, 0.0, 15.0}));
        assert(C.get_block(1, 1) == std::vector<double>({18.0, 0.0, 0.0, 18.0}));
    } else if (rank == 0) {
        assert(C.row_ptr() == std::vector<int>({0, 2}));
        const int col0 = C.graph->global_to_local.at(0);
        const int col1 = C.graph->global_to_local.at(1);
        assert(C.get_block(0, col0) == std::vector<double>({14.0, 0.0, 0.0, 14.0}));
        assert(C.get_block(0, col1) == std::vector<double>({12.0, 0.0, 0.0, 12.0}));
    } else if (rank == 1) {
        assert(C.row_ptr() == std::vector<int>({0, 2}));
        const int col0 = C.graph->global_to_local.at(0);
        const int col1 = C.graph->global_to_local.at(1);
        assert(C.get_block(0, col0) == std::vector<double>({15.0, 0.0, 0.0, 15.0}));
        assert(C.get_block(0, col1) == std::vector<double>({18.0, 0.0, 0.0, 18.0}));
    } else {
        assert(C.row_ptr() == std::vector<int>({0}));
        assert(C.col_ind().empty());
    }

    std::cout << "PASSED" << std::endl;
}

void test_bsr_spmm_distributed_unsorted_global_column_regression() {
    std::cout << "Testing BSR SpMM Distributed Unsorted Global Column Regression..." << std::endl;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size == 1) {
        std::cout << "PASSED" << std::endl;
        return;
    }

    std::vector<int> owned;
    std::vector<std::vector<int>> adj_a;
    std::vector<std::vector<int>> adj_b;
    if (rank == 0) {
        owned = {0};
        adj_a = {{}};
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

    const std::vector<double> I = {1.0, 0.0, 0.0, 1.0};
    const std::vector<double> FOUR_I = {4.0, 0.0, 0.0, 4.0};
    const std::vector<double> FIVE_I = {5.0, 0.0, 0.0, 5.0};
    const std::vector<double> SIX_I = {6.0, 0.0, 0.0, 6.0};

    if (rank == 0) {
        B.add_block(0, 0, FOUR_I.data(), 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    } else if (rank == 1) {
        A.add_block(1, 1, I.data(), 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        B.add_block(1, 0, FIVE_I.data(), 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        B.add_block(1, 1, SIX_I.data(), 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    }
    A.assemble();
    B.assemble();

    if (rank == 1) {
        assert(B.row_ptr() == std::vector<int>({0, 2}));
        std::vector<int> global_cols;
        for (int slot = B.row_ptr()[0]; slot < B.row_ptr()[1]; ++slot) {
            global_cols.push_back(B.graph->get_global_index(B.col_ind()[slot]));
        }
        assert((global_cols == std::vector<int>{1, 0}));
    }

    BlockSpMat<double> C = A.spmm(B, 0.0);
    assert(C.matrix_kind() == MatrixKind::BSR);

    if (rank == 1) {
        assert(C.row_ptr() == std::vector<int>({0, 2}));
        const int col0 = C.graph->global_to_local.at(0);
        const int col1 = C.graph->global_to_local.at(1);
        assert(C.get_block(0, col0) == std::vector<double>({5.0, 0.0, 0.0, 5.0}));
        assert(C.get_block(0, col1) == std::vector<double>({6.0, 0.0, 0.0, 6.0}));
    } else {
        if (owned.empty()) {
            assert(C.row_ptr() == std::vector<int>({0}));
        } else {
            assert(C.row_ptr() == std::vector<int>({0, 0}));
        }
        assert(C.col_ind().empty());
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

void test_vbcsr_backend_shape_registry_and_family() {
    std::cout << "Testing VBCSR Backend Family..." << std::endl;

    std::vector<int> block_sizes = {2, 3};
    std::vector<std::vector<int>> adj = {{0, 1}, {0, 1}};
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, block_sizes, adj);

    BlockSpMat<double> mat(&graph);
    assert(mat.matrix_kind() == MatrixKind::VBCSR);
    assert(mat.shape_class_count() == 4);

    std::set<std::pair<int, int>> shapes;
    size_t slot_count = 0;
    mat.for_each_shape_class([&](int shape_id, int row_dim, int col_dim, const std::vector<int>& slots) {
        (void)shape_id;
        shapes.insert({row_dim, col_dim});
        slot_count += slots.size();
    });
    const std::set<std::pair<int, int>> expected_shapes = {{2, 2}, {2, 3}, {3, 2}, {3, 3}};
    assert(shapes == expected_shapes);
    assert(slot_count == 4);

    mat.for_each_local_block([&](const auto& block) {
        assert(block.size == static_cast<size_t>(block.row_dim) * static_cast<size_t>(block.col_dim));
    });

    double a00[] = {1.0, 2.0, 3.0, 4.0};
    double a01[] = {5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    double a10[] = {11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    double a11[] = {
        17.0, 18.0, 19.0,
        20.0, 21.0, 22.0,
        23.0, 24.0, 25.0};
    mat.add_block(0, 0, a00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(0, 1, a01, 2, 3, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(1, 0, a10, 3, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(1, 1, a11, 3, 3, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.assemble();

    BlockSpMat<double> dup = mat.duplicate();
    assert(dup.matrix_kind() == MatrixKind::VBCSR);
    assert(dup.shape_class_count() == 4);
    dup.for_each_local_block([&](const auto& block) {
        assert(block.size == static_cast<size_t>(block.row_dim) * static_cast<size_t>(block.col_dim));
    });

    BlockSpMat<double> mat_t = mat.transpose();
    assert(mat_t.matrix_kind() == MatrixKind::VBCSR);
    assert(mat_t.shape_class_count() == 4);
    mat_t.for_each_local_block([&](const auto& block) {
        assert(block.size == static_cast<size_t>(block.row_dim) * static_cast<size_t>(block.col_dim));
    });

    BlockSpMat<double> prod = mat.spmm_self(0.0);
    assert(prod.matrix_kind() == MatrixKind::VBCSR);
    assert(prod.shape_class_count() > 0);

    BlockSpMat<double> filtered = mat.duplicate();
    filtered.filter_blocks(1e12);
    assert(filtered.matrix_kind() == MatrixKind::VBCSR);
    assert(filtered.shape_class_count() == 0);
    assert(filtered.local_scalar_nnz() == 0);
    assert(filtered.row_ptr() == std::vector<int>({0, 0, 0}));
    assert(filtered.col_ind().empty());

    std::cout << "PASSED" << std::endl;
}

void test_vbcsr_axpby_structure_change() {
    std::cout << "Testing VBCSR AXPBY (Structure Change Native)..." << std::endl;

    std::vector<int> block_sizes = {2, 3};
    DistGraph graph_y(MPI_COMM_SELF);
    DistGraph graph_x(MPI_COMM_SELF);
    graph_y.construct_serial(2, block_sizes, {{0}, {1}});
    graph_x.construct_serial(2, block_sizes, {{1}, {0}});

    BlockSpMat<double> Y(&graph_y);
    BlockSpMat<double> X(&graph_x);
    assert(Y.matrix_kind() == MatrixKind::VBCSR);
    assert(X.matrix_kind() == MatrixKind::VBCSR);

    double y00[] = {1.0, 2.0, 3.0, 4.0};
    double y11[] = {
        5.0, 6.0, 7.0,
        8.0, 9.0, 10.0,
        11.0, 12.0, 13.0};
    double x01[] = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
    double x10[] = {1.0, 3.0, 5.0, 7.0, 9.0, 11.0};

    Y.add_block(0, 0, y00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    Y.add_block(1, 1, y11, 3, 3, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    X.add_block(0, 1, x01, 2, 3, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    X.add_block(1, 0, x10, 3, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    Y.assemble();
    X.assemble();

    Y.axpby(2.0, X, 3.0);

    assert(Y.matrix_kind() == MatrixKind::VBCSR);
    assert(Y.shape_class_count() == 4);
    assert(Y.row_ptr() == std::vector<int>({0, 2, 4}));
    assert(Y.col_ind() == std::vector<int>({0, 1, 0, 1}));

    assert(Y.get_block(0, 0) == std::vector<double>({3.0, 6.0, 9.0, 12.0}));
    assert(Y.get_block(0, 1) == std::vector<double>({4.0, 8.0, 12.0, 16.0, 20.0, 24.0}));
    assert(Y.get_block(1, 0) == std::vector<double>({2.0, 6.0, 10.0, 14.0, 18.0, 22.0}));
    assert(Y.get_block(1, 1) == std::vector<double>({
        15.0, 18.0, 21.0,
        24.0, 27.0, 30.0,
        33.0, 36.0, 39.0}));

    std::cout << "PASSED" << std::endl;
}

void test_vbcsr_shape_batched_apply_kernels() {
    std::cout << "Testing VBCSR Shape-Batched Apply Kernels..." << std::endl;

    std::vector<int> block_sizes = {2, 2, 3};
    std::vector<std::vector<int>> adj = {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(3, block_sizes, adj);

    BlockSpMat<double> mat(&graph);
    assert(mat.matrix_kind() == MatrixKind::VBCSR);

    std::map<std::pair<int, int>, size_t> batch_sizes;
    size_t batched_blocks = 0;
    mat.for_each_shape_batch([&](const auto& batch) {
        (void)batch.shape_id;
        (void)batch.page_id;
        batch_sizes[{batch.row_dim, batch.col_dim}] += batch.block_count();
        batched_blocks += batch.block_count();
    });
    assert(batched_blocks == mat.local_block_nnz());
    assert(batch_sizes[std::make_pair(2, 2)] == 4);
    assert(batch_sizes[std::make_pair(2, 3)] == 2);
    assert(batch_sizes[std::make_pair(3, 2)] == 2);
    assert(batch_sizes[std::make_pair(3, 3)] == 1);

    for (int row = 0; row < 3; ++row) {
        for (int col : adj[row]) {
            const int r_dim = block_sizes[row];
            const int c_dim = block_sizes[col];
            std::vector<double> block(static_cast<size_t>(r_dim) * c_dim);
            const double base = 1.0 + 10.0 * row + 3.0 * col;
            for (int r = 0; r < r_dim; ++r) {
                for (int c = 0; c < c_dim; ++c) {
                    block[static_cast<size_t>(r) * c_dim + c] = base + 0.5 * r + 0.25 * c;
                }
            }
            mat.add_block(row, col, block.data(), r_dim, c_dim, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        }
    }
    mat.assemble();

    const std::vector<double> dense = mat.to_dense();
    const int rows = graph.block_offsets[graph.owned_global_indices.size()];
    const int cols = graph.block_offsets.back();
    auto total_batched_apply_count = [&](const auto& matrix) {
        std::set<int> seen_shapes;
        size_t total = 0;
        matrix.for_each_shape_batch([&](const auto& batch) {
            if (seen_shapes.insert(batch.shape_id).second) {
                total += static_cast<size_t>(batch.batched_apply_batch_count());
            }
        });
        return total;
    };

    DistVector<double> x(&graph);
    for (int i = 0; i < cols; ++i) {
        x.local_data()[i] = 1.0 + i;
    }
    DistVector<double> y(&graph);
    mat.mult(x, y);
    assert_close(std::vector<double>(y.data.begin(), y.data.begin() + rows), dense_matvec(dense, rows, cols, x.data));
    if (SmartKernel<double>::supports_batched_gemv()) {
        assert(total_batched_apply_count(mat) > 0);
    }

    DistMultiVector<double> X(&graph, 2);
    for (int vec = 0; vec < 2; ++vec) {
        for (int i = 0; i < cols; ++i) {
            X(i, vec) = 0.5 * (vec + 1) + i;
        }
    }
    DistMultiVector<double> Y(&graph, 2);
    mat.mult_dense(X, Y);
    assert_close(Y.data, dense_matmat(dense, rows, cols, X.data, cols, 2));
    if (SmartKernel<double>::supports_batched_gemm()) {
        assert(total_batched_apply_count(mat) > 0);
    }

    DistVector<double> x_adj(&graph);
    for (int i = 0; i < rows; ++i) {
        x_adj.local_data()[i] = 2.0 + 0.5 * i;
    }
    DistVector<double> y_adj(&graph);
    mat.mult_adjoint(x_adj, y_adj);
    assert_close(std::vector<double>(y_adj.data.begin(), y_adj.data.begin() + cols), dense_matvec_transpose(dense, rows, cols, x_adj.data));
    if (SmartKernel<double>::supports_batched_gemv()) {
        assert(total_batched_apply_count(mat) > 0);
    }

    DistMultiVector<double> X_adj(&graph, 2);
    for (int vec = 0; vec < 2; ++vec) {
        for (int i = 0; i < rows; ++i) {
            X_adj(i, vec) = 1.25 * (vec + 1) + 0.75 * i;
        }
    }
    DistMultiVector<double> Y_adj(&graph, 2);
    mat.mult_dense_adjoint(X_adj, Y_adj);
    assert_close(Y_adj.data, dense_matmat_transpose(dense, rows, cols, X_adj.data, rows, 2));
    if (SmartKernel<double>::supports_batched_gemm()) {
        assert(total_batched_apply_count(mat) > 0);
    }

    double overwrite01[] = {
        1.0, 3.0,
        2.0, 4.0
    };
    mat.add_block(0, 1, overwrite01, 2, 2, AssemblyMode::INSERT, MatrixLayout::ColMajor);
    mat.assemble();
    assert(mat.has_contiguous_layout());
    const std::vector<double> dense_after_update = mat.to_dense();
    DistVector<double> y_after_update(&graph);
    mat.mult(x, y_after_update);
    assert_close(
        std::vector<double>(y_after_update.data.begin(), y_after_update.data.begin() + rows),
        dense_matvec(dense_after_update, rows, cols, x.data));
    if (SmartKernel<double>::supports_batched_gemv()) {
        assert(total_batched_apply_count(mat) > 0);
    }

    std::cout << "PASSED" << std::endl;
}

void test_vbcsr_shape_batched_apply_kernels_complex() {
    std::cout << "Testing VBCSR Shape-Batched Apply Kernels (Complex)..." << std::endl;

    using T = std::complex<double>;
    std::vector<int> block_sizes = {2, 2, 3};
    std::vector<std::vector<int>> adj = {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(3, block_sizes, adj);

    BlockSpMat<T> mat(&graph);
    for (int row = 0; row < 3; ++row) {
        for (int col : adj[row]) {
            const int r_dim = block_sizes[row];
            const int c_dim = block_sizes[col];
            std::vector<T> block(static_cast<size_t>(r_dim) * c_dim);
            const double real_base = 1.0 + 3.0 * row + 1.5 * col;
            const double imag_base = -0.5 + 0.75 * row - 0.25 * col;
            for (int r = 0; r < r_dim; ++r) {
                for (int c = 0; c < c_dim; ++c) {
                    block[static_cast<size_t>(r) * c_dim + c] = T(
                        real_base + 0.2 * r + 0.1 * c,
                        imag_base + 0.15 * r - 0.05 * c);
                }
            }
            mat.add_block(row, col, block.data(), r_dim, c_dim, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        }
    }
    mat.assemble();

    const std::vector<T> dense = mat.to_dense();
    const int rows = graph.block_offsets[graph.owned_global_indices.size()];
    const int cols = graph.block_offsets.back();
    auto total_batched_apply_count = [&](const auto& matrix) {
        std::set<int> seen_shapes;
        size_t total = 0;
        matrix.for_each_shape_batch([&](const auto& batch) {
            if (seen_shapes.insert(batch.shape_id).second) {
                total += static_cast<size_t>(batch.batched_apply_batch_count());
            }
        });
        return total;
    };

    DistVector<T> x(&graph);
    for (int i = 0; i < cols; ++i) {
        x.local_data()[i] = T(1.0 + 0.25 * i, -0.5 + 0.1 * i);
    }
    DistVector<T> y(&graph);
    mat.mult(x, y);
    assert_close_generic(
        std::vector<T>(y.data.begin(), y.data.begin() + rows),
        dense_matvec_generic(dense, rows, cols, x.data));
    if (SmartKernel<T>::supports_batched_gemv()) {
        assert(total_batched_apply_count(mat) > 0);
    }

    DistMultiVector<T> X(&graph, 2);
    for (int vec = 0; vec < 2; ++vec) {
        for (int i = 0; i < cols; ++i) {
            X(i, vec) = T(0.5 * (vec + 1) + i, 0.25 * vec - 0.2 * i);
        }
    }
    DistMultiVector<T> Y(&graph, 2);
    mat.mult_dense(X, Y);
    assert_close_generic(Y.data, dense_matmat_generic(dense, rows, cols, X.data, cols, 2));
    if (SmartKernel<T>::supports_batched_gemm()) {
        assert(total_batched_apply_count(mat) > 0);
    }

    DistVector<T> x_adj(&graph);
    for (int i = 0; i < rows; ++i) {
        x_adj.local_data()[i] = T(2.0 + 0.1 * i, -1.0 + 0.2 * i);
    }
    DistVector<T> y_adj(&graph);
    mat.mult_adjoint(x_adj, y_adj);
    assert_close_generic(
        std::vector<T>(y_adj.data.begin(), y_adj.data.begin() + cols),
        dense_matvec_adjoint_generic(dense, rows, cols, x_adj.data));
    if (SmartKernel<T>::supports_batched_gemv()) {
        assert(total_batched_apply_count(mat) > 0);
    }

    DistMultiVector<T> X_adj(&graph, 2);
    for (int vec = 0; vec < 2; ++vec) {
        for (int i = 0; i < rows; ++i) {
            X_adj(i, vec) = T(1.0 + 0.3 * i, 0.5 * vec - 0.1 * i);
        }
    }
    DistMultiVector<T> Y_adj(&graph, 2);
    mat.mult_dense_adjoint(X_adj, Y_adj);
    assert_close_generic(Y_adj.data, dense_matmat_adjoint_generic(dense, rows, cols, X_adj.data, rows, 2));
    if (SmartKernel<T>::supports_batched_gemm()) {
        assert(total_batched_apply_count(mat) > 0);
    }

    std::cout << "PASSED" << std::endl;
}

void test_vbcsr_shape_batched_spmm() {
    std::cout << "Testing VBCSR Shape-Batched SpMM..." << std::endl;

    std::vector<int> block_sizes = {2, 2, 3};
    std::vector<std::vector<int>> adj = {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(3, block_sizes, adj);

    BlockSpMat<double> mat(&graph);
    assert(mat.matrix_kind() == MatrixKind::VBCSR);

    for (int row = 0; row < 3; ++row) {
        for (int col : adj[row]) {
            const int r_dim = block_sizes[row];
            const int c_dim = block_sizes[col];
            std::vector<double> block(static_cast<size_t>(r_dim) * c_dim);
            const double base = 0.75 + 5.0 * row + 2.0 * col;
            for (int r = 0; r < r_dim; ++r) {
                for (int c = 0; c < c_dim; ++c) {
                    block[static_cast<size_t>(r) * c_dim + c] = base + 0.2 * r - 0.1 * c;
                }
            }
            mat.add_block(row, col, block.data(), r_dim, c_dim, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        }
    }
    mat.assemble();

    const std::vector<double> dense = mat.to_dense();
    const int rows = graph.block_offsets[graph.owned_global_indices.size()];
    const int cols = graph.block_offsets.back();

    BlockSpMat<double> prod = mat.spmm_self(0.0);
    assert(prod.matrix_kind() == MatrixKind::VBCSR);
    assert(prod.shape_class_count() > 0);
    assert(prod.has_contiguous_layout());

    const std::vector<double> dense_prod = prod.to_dense();
    assert_close(dense_prod, dense_matmul(dense, rows, cols, dense, cols));

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

void test_csr_backend_dispatch_kernels_complex() {
    std::cout << "Testing CSR Backend Kernels (Complex)..." << std::endl;

    std::vector<int> block_sizes = {1, 1};
    std::vector<std::vector<int>> adj = {{0, 1}, {0, 1}};
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, block_sizes, adj);

    using T = std::complex<double>;
    BlockSpMat<T> mat(&graph);
    T a00[] = {T(2.0, 1.0)};
    T a01[] = {T(3.0, -2.0)};
    T a10[] = {T(4.0, 1.0)};
    T a11[] = {T(5.0, 0.0)};
    mat.add_block(0, 0, a00, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(0, 1, a01, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(1, 0, a10, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(1, 1, a11, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.assemble();

    auto assert_close = [](T got, T expected) {
        assert(std::abs(got - expected) < 1e-12);
    };

    DistVector<T> x(&graph);
    DistVector<T> y(&graph);
    x.local_data()[0] = T(1.0, 2.0);
    x.local_data()[1] = T(-1.0, 1.0);
    mat.mult(x, y);
    assert_close(y.local_data()[0], T(-1.0, 10.0));
    assert_close(y.local_data()[1], T(-3.0, 14.0));

    DistMultiVector<T> X(&graph, 2);
    DistMultiVector<T> Y(&graph, 2);
    X(0, 0) = T(1.0, 0.0);
    X(1, 0) = T(2.0, 0.0);
    X(0, 1) = T(0.0, 1.0);
    X(1, 1) = T(1.0, -1.0);
    mat.mult_dense(X, Y);
    assert_close(Y(0, 0), T(8.0, -3.0));
    assert_close(Y(1, 0), T(14.0, 1.0));
    assert_close(Y(0, 1), T(0.0, -3.0));
    assert_close(Y(1, 1), T(4.0, -1.0));

    DistVector<T> x_adj(&graph);
    DistVector<T> y_adj(&graph);
    x_adj.local_data()[0] = T(2.0, -1.0);
    x_adj.local_data()[1] = T(1.0, 3.0);
    mat.mult_adjoint(x_adj, y_adj);
    assert_close(y_adj.local_data()[0], T(10.0, 7.0));
    assert_close(y_adj.local_data()[1], T(13.0, 16.0));

    DistMultiVector<T> X_adj(&graph, 2);
    DistMultiVector<T> Y_adj(&graph, 2);
    X_adj(0, 0) = T(1.0, 1.0);
    X_adj(1, 0) = T(2.0, 0.0);
    X_adj(0, 1) = T(0.0, -1.0);
    X_adj(1, 1) = T(3.0, 1.0);
    mat.mult_dense_adjoint(X_adj, Y_adj);
    assert_close(Y_adj(0, 0), T(11.0, -1.0));
    assert_close(Y_adj(1, 0), T(11.0, 5.0));
    assert_close(Y_adj(0, 1), T(12.0, -1.0));
    assert_close(Y_adj(1, 1), T(17.0, 2.0));

    std::cout << "PASSED" << std::endl;
}

void test_csr_page_cap_policy() {
    std::cout << "Testing CSR Page-Cap Policy..." << std::endl;

    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(3, {1, 1, 1}, {{0, 2}, {0, 1, 2}, {1}});

    detail::CSRMatrixBackend<double> backend;
    assert(backend.configured_page_size() == detail::CSRMatrixBackend<double>::max_page_size());

    backend.initialize_structure(graph.adj_ind.size());
    assert(backend.active_page_size() == graph.adj_ind.size());
    assert(backend.page_count() == 1);

    backend.initialize_structure(graph.adj_ind.size(), 2);
    assert(backend.configured_page_size() == 2);
    assert(backend.active_page_size() == 2);
    assert(backend.page_count() == 3);

    backend.initialize_structure(
        graph.adj_ind.size(),
        detail::CSRMatrixBackend<double>::max_page_size());
    assert(backend.configured_page_size() == detail::CSRMatrixBackend<double>::max_page_size());
    assert(backend.active_page_size() == graph.adj_ind.size());
    assert(backend.page_count() == 1);

    std::cout << "PASSED" << std::endl;
}

void test_blockspmat_csr_page_cap_repack_and_propagation() {
    std::cout << "Testing BlockSpMat CSR Page-Cap Repack And Propagation..." << std::endl;

    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, {1, 1}, {{0, 1}, {0, 1}});

    BlockSpMat<double> mat(&graph);
    assert(mat.matrix_kind() == MatrixKind::CSR);
    assert(mat.configured_page_size() == detail::CSRMatrixBackend<double>::max_page_size());

    double a00[] = {2.0};
    double a01[] = {3.0};
    double a10[] = {4.0};
    double a11[] = {5.0};
    mat.add_block(0, 0, a00, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(0, 1, a01, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(1, 0, a10, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(1, 1, a11, 1, 1, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.assemble();

    assert(mat.page_size() == mat.local_block_nnz());

    DistVector<double> x(&graph);
    DistVector<double> y_before(&graph);
    DistVector<double> y_after(&graph);
    x.local_data()[0] = 7.0;
    x.local_data()[1] = 11.0;
    mat.mult(x, y_before);
    assert(std::abs(y_before.local_data()[0] - 47.0) < 1e-12);
    assert(std::abs(y_before.local_data()[1] - 83.0) < 1e-12);

    mat.set_page_size(1);
    assert(mat.configured_page_size() == 1);
    assert(mat.page_size() == 1);
    assert(std::abs(*mat.block_data(0) - 2.0) < 1e-12);
    assert(std::abs(*mat.block_data(1) - 3.0) < 1e-12);
    assert(std::abs(*mat.block_data(2) - 4.0) < 1e-12);
    assert(std::abs(*mat.block_data(3) - 5.0) < 1e-12);

    mat.mult(x, y_after);
    assert(std::abs(y_after.local_data()[0] - 47.0) < 1e-12);
    assert(std::abs(y_after.local_data()[1] - 83.0) < 1e-12);

    BlockSpMat<double> duplicate = mat.duplicate();
    assert(duplicate.configured_page_size() == 1);
    assert(duplicate.page_size() == 1);

    BlockSpMat<double> transpose = mat.transpose();
    assert(transpose.configured_page_size() == 1);
    assert(transpose.page_size() == 1);

    BlockSpMat<double> product = mat.spmm_self(0.0);
    assert(product.configured_page_size() == 1);
    assert(product.page_size() == 1);

    std::cout << "PASSED" << std::endl;
}

void test_blockspmat_bsr_page_settings_repack_and_propagation() {
    std::cout << "Testing BlockSpMat BSR Page Settings Repack And Propagation..." << std::endl;

    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, {2, 2}, {{0, 1}, {0, 1}});

    BlockSpMat<double> mat(&graph);
    assert(mat.matrix_kind() == MatrixKind::BSR);

    double a00[] = {1.0, 3.0, 2.0, 4.0};
    double a01[] = {5.0, 7.0, 6.0, 8.0};
    double a10[] = {9.0, 11.0, 10.0, 12.0};
    double a11[] = {13.0, 15.0, 14.0, 16.0};
    mat.add_block(0, 0, a00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(0, 1, a01, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(1, 0, a10, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.add_block(1, 1, a11, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    mat.assemble();

    assert(mat.page_size() == mat.local_block_nnz());

    DistVector<double> x(&graph);
    DistVector<double> y_before(&graph);
    DistVector<double> y_after(&graph);
    for (int i = 0; i < 4; ++i) {
        x.local_data()[i] = 1.0 + i;
    }
    mat.mult(x, y_before);

    mat.set_page_size(1);
    assert(mat.configured_page_size() == 1);
    assert(mat.page_size() == 1);

    mat.mult(x, y_after);
    assert(std::abs(y_after.local_data()[0] - y_before.local_data()[0]) < 1e-12);
    assert(std::abs(y_after.local_data()[1] - y_before.local_data()[1]) < 1e-12);
    assert(std::abs(y_after.local_data()[2] - y_before.local_data()[2]) < 1e-12);
    assert(std::abs(y_after.local_data()[3] - y_before.local_data()[3]) < 1e-12);

    BlockSpMat<double> duplicate = mat.duplicate();
    assert(duplicate.configured_page_size() == 1);
    assert(duplicate.page_size() == 1);

    BlockSpMat<double> transpose = mat.transpose();
    assert(transpose.configured_page_size() == 1);
    assert(transpose.page_size() == 1);

    BlockSpMat<double> product = mat.spmm_self(0.0);
    assert(product.configured_page_size() == 1);
    assert(product.page_size() == 1);

    std::cout << "PASSED" << std::endl;
}

void test_blockspmat_vbcsr_page_settings_repack_and_propagation() {
    std::cout << "Testing BlockSpMat VBCSR Page Settings Repack And Propagation..." << std::endl;

    std::vector<int> block_sizes = {2, 2, 3};
    std::vector<std::vector<int>> adj = {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(3, block_sizes, adj);

    BlockSpMat<double> mat(&graph);
    assert(mat.matrix_kind() == MatrixKind::VBCSR);

    for (int row = 0; row < 3; ++row) {
        for (int col : adj[row]) {
            const int r_dim = block_sizes[row];
            const int c_dim = block_sizes[col];
            std::vector<double> block(static_cast<size_t>(r_dim) * c_dim);
            const double base = 1.0 + 10.0 * row + 3.0 * col;
            for (int r = 0; r < r_dim; ++r) {
                for (int c = 0; c < c_dim; ++c) {
                    block[static_cast<size_t>(r) * c_dim + c] = base + 0.5 * r + 0.25 * c;
                }
            }
            mat.add_block(row, col, block.data(), r_dim, c_dim, AssemblyMode::INSERT, MatrixLayout::RowMajor);
        }
    }
    mat.assemble();

    size_t default_batches = 0;
    mat.for_each_shape_batch([&](const auto& batch) {
        ++default_batches;
        assert(batch.block_capacity() == batch.block_count());
    });
    assert(default_batches == 4);

    const std::vector<double> dense_before = mat.to_dense();

    mat.set_page_size(1);
    assert(mat.configured_page_size() == 1);

    size_t repacked_batches = 0;
    mat.for_each_shape_batch([&](const auto& batch) {
        ++repacked_batches;
        assert(batch.block_capacity() == 1);
        assert(batch.block_count() == 1);
    });
    assert(repacked_batches == mat.local_block_nnz());
    assert_close(mat.to_dense(), dense_before);

    BlockSpMat<double> duplicate = mat.duplicate();
    assert(duplicate.configured_page_size() == 1);

    BlockSpMat<double> transpose = mat.transpose();
    assert(transpose.configured_page_size() == 1);

    BlockSpMat<double> product = mat.spmm_self(0.0);
    assert(product.configured_page_size() == 1);

    std::cout << "PASSED" << std::endl;
}

void test_csr_vendor_page_cache_metadata() {
    std::cout << "Testing CSR Vendor Page Cache Metadata..." << std::endl;

    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(3, {1, 1, 1}, {{0, 2}, {0, 1, 2}, {1}});

    detail::CSRMatrixBackend<double> backend;
    backend.initialize_structure(graph.adj_ind.size(), 3);
    for (int slot = 0; slot < static_cast<int>(graph.adj_ind.size()); ++slot) {
        *backend.value_ptr(slot) = 1.0 + slot;
    }

    const auto& cache = backend.ensure_vendor_cache(
        graph.adj_ptr,
        graph.adj_ind,
        static_cast<int>(graph.block_sizes.size()));
    assert(cache.pages.size() == 2);

    const auto& page0 = cache.pages[0];
    assert(page0.batch.page_index == 0);
    assert(page0.batch.first_nnz == 0);
    assert(page0.batch.nnz_count == 3);
    assert(page0.batch.row_begin == 0);
    assert(page0.batch.row_end == 2);
    assert(page0.row_offsets_storage == std::vector<int>({0, 2, 3}));

    const auto& page1 = cache.pages[1];
    assert(page1.batch.page_index == 1);
    assert(page1.batch.first_nnz == 3);
    assert(page1.batch.nnz_count == 3);
    assert(page1.batch.row_begin == 1);
    assert(page1.batch.row_end == 3);
    assert(page1.row_offsets_storage == std::vector<int>({0, 2, 3}));

#ifdef VBCSR_HAVE_MKL_SPARSE
    assert(cache.kind == detail::CSRVendorBackendKind::MKL);
    assert(backend.vendor_backend_name() == "mkl");
#elif defined(VBCSR_HAVE_AOCL_SPARSE)
    assert(cache.kind == detail::CSRVendorBackendKind::AOCL);
    assert(backend.vendor_backend_name() == "aocl");
#else
    assert(cache.kind == detail::CSRVendorBackendKind::None);
    assert(backend.vendor_backend_name() == "none");
#endif

    std::cout << "PASSED" << std::endl;
}

void test_bsr_vendor_batch_cache_metadata() {
    std::cout << "Testing BSR Vendor Batch Cache Metadata..." << std::endl;

    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, {2, 2}, {{0, 1}, {0, 1}});

    detail::BSRMatrixBackend<double> backend;
    backend.initialize_structure(graph.adj_ind.size(), 2, 1);
    const double blocks[][4] = {
        {1.0, 3.0, 2.0, 4.0},
        {5.0, 7.0, 6.0, 8.0},
        {2.0, 1.0, 0.0, 2.0},
        {0.0, 4.0, 1.0, 3.0},
    };
    for (int slot = 0; slot < 4; ++slot) {
        std::memcpy(backend.block_ptr(slot), blocks[slot], sizeof(blocks[slot]));
    }

    const auto& cache = backend.ensure_vendor_cache(
        graph.adj_ptr,
        graph.adj_ind,
        static_cast<int>(graph.block_sizes.size()));
    assert(cache.batches.size() == 4);

    const auto& batch0 = cache.batches[0];
    assert(batch0.batch.page_index == 0);
    assert(batch0.batch.first_block == 0);
    assert(batch0.batch.block_count == 1);
    assert(batch0.batch.row_begin == 0);
    assert(batch0.batch.row_end == 1);
    assert(batch0.row_block_offsets_storage == std::vector<int>({0, 1}));

    const auto& batch1 = cache.batches[1];
    assert(batch1.batch.page_index == 1);
    assert(batch1.batch.first_block == 1);
    assert(batch1.batch.block_count == 1);
    assert(batch1.batch.row_begin == 0);
    assert(batch1.batch.row_end == 1);
    assert(batch1.row_block_offsets_storage == std::vector<int>({0, 1}));

    const auto& batch2 = cache.batches[2];
    assert(batch2.batch.page_index == 2);
    assert(batch2.batch.first_block == 2);
    assert(batch2.batch.block_count == 1);
    assert(batch2.batch.row_begin == 1);
    assert(batch2.batch.row_end == 2);
    assert(batch2.row_block_offsets_storage == std::vector<int>({0, 1}));

    const auto& batch3 = cache.batches[3];
    assert(batch3.batch.page_index == 3);
    assert(batch3.batch.first_block == 3);
    assert(batch3.batch.block_count == 1);
    assert(batch3.batch.row_begin == 1);
    assert(batch3.batch.row_end == 2);
    assert(batch3.row_block_offsets_storage == std::vector<int>({0, 1}));

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
    assert(cache.kind == detail::BSRVendorBackendKind::MKL);
    assert(backend.vendor_backend_name() == "mkl");
    assert((batch0.mm_rows_start_one == std::vector<MKL_INT>{1}));
    assert((batch0.mm_rows_end_one == std::vector<MKL_INT>{2}));
    assert((batch0.mm_cols_one == std::vector<MKL_INT>{1}));
    assert((batch3.mm_rows_start_one == std::vector<MKL_INT>{1}));
    assert((batch3.mm_rows_end_one == std::vector<MKL_INT>{2}));
    assert((batch3.mm_cols_one == std::vector<MKL_INT>{2}));
#else
    assert(cache.kind == detail::BSRVendorBackendKind::None);
    assert(backend.vendor_backend_name() == "none");
#endif

    std::cout << "PASSED" << std::endl;
}

void test_bsr_vendor_dispatch_selection() {
    std::cout << "Testing BSR Vendor Dispatch Selection..." << std::endl;

    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, {2, 2}, {{0, 1}, {0, 1}});

    detail::BSRMatrixBackend<double> backend;
    backend.initialize_structure(graph.adj_ind.size(), 2, 1);
    const double blocks[][4] = {
        {1.0, 3.0, 2.0, 4.0},
        {5.0, 7.0, 6.0, 8.0},
        {2.0, 1.0, 0.0, 2.0},
        {0.0, 4.0, 1.0, 3.0},
    };
    for (int slot = 0; slot < 4; ++slot) {
        std::memcpy(backend.block_ptr(slot), blocks[slot], sizeof(blocks[slot]));
    }

    const std::vector<double> dense = {
        1.0, 2.0, 5.0, 6.0,
        3.0, 4.0, 7.0, 8.0,
        2.0, 0.0, 0.0, 1.0,
        1.0, 2.0, 4.0, 3.0,
    };

    backend.reset_vendor_launch_count();

    DistVector<double> x(&graph);
    DistVector<double> y(&graph);
    x.local_data()[0] = 1.0;
    x.local_data()[1] = 2.0;
    x.local_data()[2] = 3.0;
    x.local_data()[3] = 4.0;
    detail::bsr_mult(&graph, backend, x, y);
    assert_close(
        std::vector<double>(y.local_data(), y.local_data() + 4),
        dense_matvec(dense, 4, 4, std::vector<double>(x.local_data(), x.local_data() + 4)));

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
    detail::bsr_mult_dense(&graph, backend, X, Y);
    const auto expected_dense = dense_matmat_generic(dense, 4, 4, X.data, 4, 2);
    assert_close(
        std::vector<double>(Y.data.begin(), Y.data.begin() + 8),
        expected_dense);

    DistVector<double> x_adj(&graph);
    DistVector<double> y_adj(&graph);
    x_adj.local_data()[0] = 2.0;
    x_adj.local_data()[1] = -1.0;
    x_adj.local_data()[2] = 4.0;
    x_adj.local_data()[3] = 3.0;
    detail::bsr_mult_adjoint(&graph, backend, x_adj, y_adj);
    assert_close_generic(
        std::vector<double>(y_adj.local_data(), y_adj.local_data() + 4),
        dense_matvec_adjoint_generic(dense, 4, 4, std::vector<double>(x_adj.local_data(), x_adj.local_data() + 4)));

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
    detail::bsr_mult_dense_adjoint(&graph, backend, X_adj, Y_adj);
    const auto expected_dense_adj = dense_matmat_adjoint_generic(dense, 4, 4, X_adj.data, 4, 2);
    assert_close_generic(
        std::vector<double>(Y_adj.data.begin(), Y_adj.data.begin() + 8),
        expected_dense_adj);

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
    assert(backend.get_vendor_launch_count() > 0);
#else
    assert(backend.get_vendor_launch_count() == 0);
#endif

    std::cout << "PASSED" << std::endl;
}

void test_bsr_vendor_cache_reuse_repeated_apply() {
    std::cout << "Testing BSR Vendor Cache Reuse Across Repeated Apply..." << std::endl;

    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, {2, 2}, {{0, 1}, {0, 1}});

    detail::BSRMatrixBackend<double> backend;
    backend.initialize_structure(graph.adj_ind.size(), 2, 1);
    const double blocks[][4] = {
        {1.0, 0.0, 0.0, 1.0},
        {2.0, 0.0, 0.0, 2.0},
        {3.0, 0.0, 0.0, 3.0},
        {4.0, 0.0, 0.0, 4.0},
    };
    for (int slot = 0; slot < 4; ++slot) {
        std::memcpy(backend.block_ptr(slot), blocks[slot], sizeof(blocks[slot]));
    }

    assert(backend.vendor_cache_identity() == nullptr);
    backend.reset_vendor_launch_count();

    DistVector<double> x(&graph);
    DistVector<double> y(&graph);
    x.local_data()[0] = 1.0;
    x.local_data()[1] = 2.0;
    x.local_data()[2] = 3.0;
    x.local_data()[3] = 4.0;
    detail::bsr_mult(&graph, backend, x, y);

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
    const void* cache_after_first_apply = backend.vendor_cache_identity();
    assert(cache_after_first_apply != nullptr);
    detail::bsr_mult(&graph, backend, x, y);
    assert(backend.vendor_cache_identity() == cache_after_first_apply);

    DistMultiVector<double> X2(&graph, 2);
    DistMultiVector<double> Y2(&graph, 2);
    X2(0, 0) = 1.0;
    X2(1, 0) = 2.0;
    X2(2, 0) = 3.0;
    X2(3, 0) = 4.0;
    X2(0, 1) = 5.0;
    X2(1, 1) = 6.0;
    X2(2, 1) = 7.0;
    X2(3, 1) = 8.0;
    backend.reset_vendor_launch_count();
    detail::bsr_mult_dense(&graph, backend, X2, Y2);
    const auto& cache_after_dense = backend.ensure_vendor_cache(
        graph.adj_ptr,
        graph.adj_ind,
        static_cast<int>(graph.block_sizes.size()));
    assert(backend.get_vendor_launch_count() > 0);
    assert(cache_after_dense.batches[0].mkl.mm_variants.size() == 1);
    detail::bsr_mult_dense(&graph, backend, X2, Y2);
    assert(cache_after_dense.batches[0].mkl.mm_variants.size() == 1);

    DistMultiVector<double> X3(&graph, 3);
    DistMultiVector<double> Y3(&graph, 3);
    for (int vec = 0; vec < 3; ++vec) {
        for (int row = 0; row < 4; ++row) {
            X3(row, vec) = static_cast<double>(vec * 10 + row + 1);
        }
    }
    detail::bsr_mult_dense(&graph, backend, X3, Y3);
    assert(cache_after_dense.batches[0].mkl.mm_variants.size() == 2);
    assert(backend.vendor_cache_identity() == cache_after_first_apply);
    assert(backend.get_vendor_launch_count() > 0);
#else
    assert(backend.vendor_cache_identity() == nullptr);
    assert(backend.get_vendor_launch_count() == 0);
#endif

    std::cout << "PASSED" << std::endl;
}

void test_bsr_vendor_cache_invalidation_on_structure_change() {
    std::cout << "Testing BSR Vendor Cache Invalidation On Structure Change..." << std::endl;

    DistGraph graph_a(MPI_COMM_SELF);
    graph_a.construct_serial(2, {2, 2}, {{0, 1}, {1}});

    detail::BSRMatrixBackend<double> backend;
    backend.initialize_structure(graph_a.adj_ind.size(), 2, 1);
    const double blocks_a[][4] = {
        {1.0, 0.0, 0.0, 1.0},
        {2.0, 0.0, 0.0, 2.0},
        {3.0, 0.0, 0.0, 3.0},
    };
    for (int slot = 0; slot < 3; ++slot) {
        std::memcpy(backend.block_ptr(slot), blocks_a[slot], sizeof(blocks_a[slot]));
    }

    DistVector<double> x_a(&graph_a);
    DistVector<double> y_a(&graph_a);
    x_a.local_data()[0] = 1.0;
    x_a.local_data()[1] = 2.0;
    x_a.local_data()[2] = 3.0;
    x_a.local_data()[3] = 4.0;
    detail::bsr_mult(&graph_a, backend, x_a, y_a);

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
    const void* cache_a = backend.vendor_cache_identity();
    assert(cache_a != nullptr);
#else
    assert(backend.vendor_cache_identity() == nullptr);
#endif

    DistGraph graph_b(MPI_COMM_SELF);
    graph_b.construct_serial(2, {2, 2}, {{0}, {0, 1}});

    backend.initialize_structure(graph_b.adj_ind.size(), 2, 1);
    assert(backend.vendor_cache_identity() == nullptr);
    const double blocks_b[][4] = {
        {4.0, 0.0, 0.0, 4.0},
        {5.0, 0.0, 0.0, 5.0},
        {6.0, 0.0, 0.0, 6.0},
    };
    for (int slot = 0; slot < 3; ++slot) {
        std::memcpy(backend.block_ptr(slot), blocks_b[slot], sizeof(blocks_b[slot]));
    }

    DistVector<double> x_b(&graph_b);
    DistVector<double> y_b(&graph_b);
    x_b.local_data()[0] = 2.0;
    x_b.local_data()[1] = 1.0;
    x_b.local_data()[2] = -1.0;
    x_b.local_data()[3] = 3.0;
    detail::bsr_mult(&graph_b, backend, x_b, y_b);

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
    const void* cache_b = backend.vendor_cache_identity();
    assert(cache_b != nullptr);
    assert(cache_b != cache_a);
    const auto& rebuilt_cache = backend.ensure_vendor_cache(
        graph_b.adj_ptr,
        graph_b.adj_ind,
        static_cast<int>(graph_b.block_sizes.size()));
    assert(rebuilt_cache.batches.size() == 3);
#else
    assert(backend.vendor_cache_identity() == nullptr);
#endif

    std::cout << "PASSED" << std::endl;
}

#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
void test_bsr_vendor_complex_dispatch_selection() {
    std::cout << "Testing BSR Vendor Complex Dispatch Selection..." << std::endl;

    using C = std::complex<double>;

    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, {2, 2}, {{0, 1}, {0, 1}});

    detail::BSRMatrixBackend<C> backend;
    backend.initialize_structure(graph.adj_ind.size(), 2, 1);
    const C blocks[][4] = {
        {C(1.0, 1.0), C(0.0, 0.0), C(0.0, 0.0), C(1.0, 1.0)},
        {C(2.0, -1.0), C(0.0, 0.0), C(0.0, 0.0), C(2.0, -1.0)},
        {C(-1.0, 0.5), C(0.0, 0.0), C(0.0, 0.0), C(-1.0, 0.5)},
        {C(0.0, 2.0), C(0.0, 0.0), C(0.0, 0.0), C(0.0, 2.0)},
    };
    for (int slot = 0; slot < 4; ++slot) {
        std::memcpy(backend.block_ptr(slot), blocks[slot], sizeof(blocks[slot]));
    }

    const std::vector<C> dense = {
        C(1.0, 1.0), C(0.0, 0.0), C(2.0, -1.0), C(0.0, 0.0),
        C(0.0, 0.0), C(1.0, 1.0), C(0.0, 0.0), C(2.0, -1.0),
        C(-1.0, 0.5), C(0.0, 0.0), C(0.0, 2.0), C(0.0, 0.0),
        C(0.0, 0.0), C(-1.0, 0.5), C(0.0, 0.0), C(0.0, 2.0),
    };

    DistVector<C> x(&graph);
    DistVector<C> y(&graph);
    x.local_data()[0] = C(1.0, -1.0);
    x.local_data()[1] = C(2.0, 0.5);
    x.local_data()[2] = C(-3.0, 1.0);
    x.local_data()[3] = C(4.0, -2.0);
    detail::bsr_mult(&graph, backend, x, y);
    assert_close_generic(
        std::vector<C>(y.local_data(), y.local_data() + 4),
        dense_matvec_generic(dense, 4, 4, std::vector<C>(x.local_data(), x.local_data() + 4)));

    DistMultiVector<C> X(&graph, 2);
    DistMultiVector<C> Y(&graph, 2);
    X(0, 0) = C(1.0, 0.0);
    X(1, 0) = C(0.0, 1.0);
    X(2, 0) = C(2.0, -1.0);
    X(3, 0) = C(-3.0, 0.5);
    X(0, 1) = C(4.0, -2.0);
    X(1, 1) = C(1.0, 3.0);
    X(2, 1) = C(-1.0, 2.0);
    X(3, 1) = C(0.5, -0.5);
    detail::bsr_mult_dense(&graph, backend, X, Y);
    assert_close_generic(
        std::vector<C>(Y.data.begin(), Y.data.end()),
        dense_matmat_generic(dense, 4, 4, X.data, 4, 2));

    DistVector<C> x_adj(&graph);
    DistVector<C> y_adj(&graph);
    x_adj.local_data()[0] = C(2.0, 1.0);
    x_adj.local_data()[1] = C(-1.0, 0.5);
    x_adj.local_data()[2] = C(0.0, -2.0);
    x_adj.local_data()[3] = C(3.0, 4.0);
    detail::bsr_mult_adjoint(&graph, backend, x_adj, y_adj);
    assert_close_generic(
        std::vector<C>(y_adj.local_data(), y_adj.local_data() + 4),
        dense_matvec_adjoint_generic(dense, 4, 4, std::vector<C>(x_adj.local_data(), x_adj.local_data() + 4)));

    DistMultiVector<C> X_adj(&graph, 2);
    DistMultiVector<C> Y_adj(&graph, 2);
    X_adj(0, 0) = C(2.0, 1.0);
    X_adj(1, 0) = C(-1.0, 0.5);
    X_adj(2, 0) = C(0.0, -2.0);
    X_adj(3, 0) = C(3.0, 4.0);
    X_adj(0, 1) = C(1.0, -3.0);
    X_adj(1, 1) = C(0.5, 0.25);
    X_adj(2, 1) = C(-4.0, 2.0);
    X_adj(3, 1) = C(2.5, -1.5);
    detail::bsr_mult_dense_adjoint(&graph, backend, X_adj, Y_adj);
    assert_close_generic(
        std::vector<C>(Y_adj.data.begin(), Y_adj.data.end()),
        dense_matmat_adjoint_generic(dense, 4, 4, X_adj.data, 4, 2));

    assert(backend.get_vendor_launch_count() > 0);
    std::cout << "PASSED" << std::endl;
}
#endif

void test_csr_vendor_dispatch_selection() {
    std::cout << "Testing CSR Vendor Dispatch Selection..." << std::endl;

    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, {1, 1}, {{0, 1}, {0, 1}});

    detail::CSRMatrixBackend<double> backend;
    backend.initialize_structure(graph.adj_ind.size(), 1);
    const double vals[] = {2.0, 3.0, 4.0, 5.0};
    for (int slot = 0; slot < 4; ++slot) {
        *backend.value_ptr(slot) = vals[slot];
    }

    DistVector<double> x(&graph);
    DistVector<double> y(&graph);
    x.local_data()[0] = 7.0;
    x.local_data()[1] = 11.0;

    backend.reset_vendor_launch_count();
    detail::csr_mult(&graph, backend, x, y);
    assert(std::abs(y.local_data()[0] - 47.0) < 1e-12);
    assert(std::abs(y.local_data()[1] - 83.0) < 1e-12);

    DistMultiVector<double> X(&graph, 2);
    DistMultiVector<double> Y(&graph, 2);
    X(0, 0) = 1.0;
    X(1, 0) = 2.0;
    X(0, 1) = 3.0;
    X(1, 1) = 4.0;
    detail::csr_mult_dense(&graph, backend, X, Y);
    assert(std::abs(Y(0, 0) - 8.0) < 1e-12);
    assert(std::abs(Y(1, 0) - 14.0) < 1e-12);

    DistVector<double> x_adj(&graph);
    DistVector<double> y_adj(&graph);
    x_adj.local_data()[0] = 13.0;
    x_adj.local_data()[1] = 17.0;
    detail::csr_mult_adjoint(&graph, backend, x_adj, y_adj);
    assert(std::abs(y_adj.local_data()[0] - 94.0) < 1e-12);
    assert(std::abs(y_adj.local_data()[1] - 124.0) < 1e-12);

    DistMultiVector<double> X_adj(&graph, 2);
    DistMultiVector<double> Y_adj(&graph, 2);
    X_adj(0, 0) = 5.0;
    X_adj(1, 0) = 7.0;
    X_adj(0, 1) = 11.0;
    X_adj(1, 1) = 13.0;
    detail::csr_mult_dense_adjoint(&graph, backend, X_adj, Y_adj);
    assert(std::abs(Y_adj(0, 0) - 38.0) < 1e-12);
    assert(std::abs(Y_adj(1, 0) - 50.0) < 1e-12);

#if defined(VBCSR_HAVE_MKL_SPARSE) || defined(VBCSR_HAVE_AOCL_SPARSE)
    assert(backend.get_vendor_launch_count() > 0);
#else
    assert(backend.get_vendor_launch_count() == 0);
#endif

    std::cout << "PASSED" << std::endl;
}

void test_csr_vendor_cache_reuse_repeated_apply() {
    std::cout << "Testing CSR Vendor Cache Reuse Across Repeated Apply..." << std::endl;

    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, {1, 1}, {{0, 1}, {0, 1}});

    detail::CSRMatrixBackend<double> backend;
    backend.initialize_structure(graph.adj_ind.size(), 1);
    const double vals[] = {2.0, 3.0, 4.0, 5.0};
    for (int slot = 0; slot < 4; ++slot) {
        *backend.value_ptr(slot) = vals[slot];
    }

    assert(backend.vendor_cache_identity() == nullptr);
    backend.reset_vendor_launch_count();

    DistVector<double> x1(&graph);
    DistVector<double> y1(&graph);
    x1.local_data()[0] = 7.0;
    x1.local_data()[1] = 11.0;
    detail::csr_mult(&graph, backend, x1, y1);
    assert(std::abs(y1.local_data()[0] - 47.0) < 1e-12);
    assert(std::abs(y1.local_data()[1] - 83.0) < 1e-12);

    const void* cache_after_first_apply = backend.vendor_cache_identity();
    assert(cache_after_first_apply != nullptr);

    DistVector<double> y1_repeat(&graph);
    detail::csr_mult(&graph, backend, x1, y1_repeat);
    assert(std::abs(y1_repeat.local_data()[0] - 47.0) < 1e-12);
    assert(std::abs(y1_repeat.local_data()[1] - 83.0) < 1e-12);
    assert(backend.vendor_cache_identity() == cache_after_first_apply);

    DistVector<double> x2(&graph);
    DistVector<double> y2(&graph);
    x2.local_data()[0] = -1.0;
    x2.local_data()[1] = 2.0;
    detail::csr_mult(&graph, backend, x2, y2);
    assert(std::abs(y2.local_data()[0] - 4.0) < 1e-12);
    assert(std::abs(y2.local_data()[1] - 6.0) < 1e-12);
    assert(backend.vendor_cache_identity() == cache_after_first_apply);

    DistMultiVector<double> X1(&graph, 2);
    DistMultiVector<double> Y1(&graph, 2);
    X1(0, 0) = 1.0;
    X1(1, 0) = 2.0;
    X1(0, 1) = 3.0;
    X1(1, 1) = 4.0;
    detail::csr_mult_dense(&graph, backend, X1, Y1);
    assert(std::abs(Y1(0, 0) - 8.0) < 1e-12);
    assert(std::abs(Y1(1, 0) - 14.0) < 1e-12);
    assert(std::abs(Y1(0, 1) - 18.0) < 1e-12);
    assert(std::abs(Y1(1, 1) - 32.0) < 1e-12);
    assert(backend.vendor_cache_identity() == cache_after_first_apply);

    DistMultiVector<double> Y1_repeat(&graph, 2);
    detail::csr_mult_dense(&graph, backend, X1, Y1_repeat);
    assert(std::abs(Y1_repeat(0, 0) - 8.0) < 1e-12);
    assert(std::abs(Y1_repeat(1, 0) - 14.0) < 1e-12);
    assert(std::abs(Y1_repeat(0, 1) - 18.0) < 1e-12);
    assert(std::abs(Y1_repeat(1, 1) - 32.0) < 1e-12);
    assert(backend.vendor_cache_identity() == cache_after_first_apply);

    DistMultiVector<double> X2(&graph, 2);
    DistMultiVector<double> Y2(&graph, 2);
    X2(0, 0) = -2.0;
    X2(1, 0) = 1.0;
    X2(0, 1) = 0.5;
    X2(1, 1) = -1.5;
    detail::csr_mult_dense(&graph, backend, X2, Y2);
    assert(std::abs(Y2(0, 0) - (-1.0)) < 1e-12);
    assert(std::abs(Y2(1, 0) - (-3.0)) < 1e-12);
    assert(std::abs(Y2(0, 1) - (-3.5)) < 1e-12);
    assert(std::abs(Y2(1, 1) - (-5.5)) < 1e-12);
    assert(backend.vendor_cache_identity() == cache_after_first_apply);

    DistVector<double> x_adj1(&graph);
    DistVector<double> y_adj1(&graph);
    x_adj1.local_data()[0] = 13.0;
    x_adj1.local_data()[1] = 17.0;
    detail::csr_mult_adjoint(&graph, backend, x_adj1, y_adj1);
    assert(std::abs(y_adj1.local_data()[0] - 94.0) < 1e-12);
    assert(std::abs(y_adj1.local_data()[1] - 124.0) < 1e-12);
    assert(backend.vendor_cache_identity() == cache_after_first_apply);

    DistVector<double> y_adj1_repeat(&graph);
    detail::csr_mult_adjoint(&graph, backend, x_adj1, y_adj1_repeat);
    assert(std::abs(y_adj1_repeat.local_data()[0] - 94.0) < 1e-12);
    assert(std::abs(y_adj1_repeat.local_data()[1] - 124.0) < 1e-12);
    assert(backend.vendor_cache_identity() == cache_after_first_apply);

    DistVector<double> x_adj2(&graph);
    DistVector<double> y_adj2(&graph);
    x_adj2.local_data()[0] = -2.0;
    x_adj2.local_data()[1] = 1.0;
    detail::csr_mult_adjoint(&graph, backend, x_adj2, y_adj2);
    assert(std::abs(y_adj2.local_data()[0] - 0.0) < 1e-12);
    assert(std::abs(y_adj2.local_data()[1] - (-1.0)) < 1e-12);
    assert(backend.vendor_cache_identity() == cache_after_first_apply);

    DistMultiVector<double> X_adj1(&graph, 2);
    DistMultiVector<double> Y_adj1(&graph, 2);
    X_adj1(0, 0) = 5.0;
    X_adj1(1, 0) = 7.0;
    X_adj1(0, 1) = 11.0;
    X_adj1(1, 1) = 13.0;
    detail::csr_mult_dense_adjoint(&graph, backend, X_adj1, Y_adj1);
    assert(std::abs(Y_adj1(0, 0) - 38.0) < 1e-12);
    assert(std::abs(Y_adj1(1, 0) - 50.0) < 1e-12);
    assert(std::abs(Y_adj1(0, 1) - 74.0) < 1e-12);
    assert(std::abs(Y_adj1(1, 1) - 98.0) < 1e-12);
    assert(backend.vendor_cache_identity() == cache_after_first_apply);

    DistMultiVector<double> Y_adj1_repeat(&graph, 2);
    detail::csr_mult_dense_adjoint(&graph, backend, X_adj1, Y_adj1_repeat);
    assert(std::abs(Y_adj1_repeat(0, 0) - 38.0) < 1e-12);
    assert(std::abs(Y_adj1_repeat(1, 0) - 50.0) < 1e-12);
    assert(std::abs(Y_adj1_repeat(0, 1) - 74.0) < 1e-12);
    assert(std::abs(Y_adj1_repeat(1, 1) - 98.0) < 1e-12);
    assert(backend.vendor_cache_identity() == cache_after_first_apply);

    DistMultiVector<double> X_adj2(&graph, 2);
    DistMultiVector<double> Y_adj2(&graph, 2);
    X_adj2(0, 0) = 1.0;
    X_adj2(1, 0) = -3.0;
    X_adj2(0, 1) = -2.0;
    X_adj2(1, 1) = 4.0;
    detail::csr_mult_dense_adjoint(&graph, backend, X_adj2, Y_adj2);
    assert(std::abs(Y_adj2(0, 0) - (-10.0)) < 1e-12);
    assert(std::abs(Y_adj2(1, 0) - (-12.0)) < 1e-12);
    assert(std::abs(Y_adj2(0, 1) - 12.0) < 1e-12);
    assert(std::abs(Y_adj2(1, 1) - 14.0) < 1e-12);
    assert(backend.vendor_cache_identity() == cache_after_first_apply);

#if defined(VBCSR_HAVE_MKL_SPARSE) || defined(VBCSR_HAVE_AOCL_SPARSE)
    assert(backend.get_vendor_launch_count() > 0);
#endif

    std::cout << "PASSED" << std::endl;
}

void test_csr_vendor_cache_invalidation_on_structure_change() {
    std::cout << "Testing CSR Vendor Cache Invalidation On Structure Change..." << std::endl;

    DistGraph graph_a(MPI_COMM_SELF);
    graph_a.construct_serial(2, {1, 1}, {{0, 1}, {1}});

    detail::CSRMatrixBackend<double> backend;
    backend.initialize_structure(graph_a.adj_ind.size(), 2);
    const double vals_a[] = {2.0, 3.0, 5.0};
    for (int slot = 0; slot < 3; ++slot) {
        *backend.value_ptr(slot) = vals_a[slot];
    }

    DistVector<double> x_a(&graph_a);
    DistVector<double> y_a(&graph_a);
    x_a.local_data()[0] = 1.0;
    x_a.local_data()[1] = 2.0;
    detail::csr_mult(&graph_a, backend, x_a, y_a);
    assert(std::abs(y_a.local_data()[0] - 8.0) < 1e-12);
    assert(std::abs(y_a.local_data()[1] - 10.0) < 1e-12);

    const void* cache_a = backend.vendor_cache_identity();
    assert(cache_a != nullptr);

    DistGraph graph_b(MPI_COMM_SELF);
    graph_b.construct_serial(2, {1, 1}, {{0}, {0, 1}});

    backend.initialize_structure(graph_b.adj_ind.size(), 2);
    assert(backend.vendor_cache_identity() == nullptr);
    const double vals_b[] = {7.0, 11.0, 13.0};
    for (int slot = 0; slot < 3; ++slot) {
        *backend.value_ptr(slot) = vals_b[slot];
    }

    DistVector<double> x_b(&graph_b);
    DistVector<double> y_b(&graph_b);
    x_b.local_data()[0] = 2.0;
    x_b.local_data()[1] = 3.0;
    detail::csr_mult(&graph_b, backend, x_b, y_b);
    assert(std::abs(y_b.local_data()[0] - 14.0) < 1e-12);
    assert(std::abs(y_b.local_data()[1] - 61.0) < 1e-12);

    const void* cache_b = backend.vendor_cache_identity();
    assert(cache_b != nullptr);
    assert(cache_b != cache_a);

    const auto& rebuilt_cache = backend.ensure_vendor_cache(
        graph_b.adj_ptr,
        graph_b.adj_ind,
        static_cast<int>(graph_b.block_sizes.size()));
    assert(rebuilt_cache.pages.size() == 2);

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
    assert(mat_t.row_ptr() == std::vector<int>({0, 2, 3, 4}));
    assert(mat_t.col_ind() == std::vector<int>({0, 1, 2, 0}));
    assert(mat_t.row_ptr() == std::vector<int>({0, 2, 3, 4}));
    assert(mat_t.col_ind() == std::vector<int>({0, 1, 2, 0}));
    assert(mat_t.row_ptr() == mat_t.row_ptr());
    assert(mat_t.col_ind() == mat_t.col_ind());
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
    assert(mat_t.row_ptr() == mat_t.row_ptr());
    assert(mat_t.col_ind() == mat_t.col_ind());

    if (size == 1) {
        assert(mat_t.row_ptr() == std::vector<int>({0, 2, 3}));
        assert(mat_t.col_ind() == std::vector<int>({0, 1, 0}));
        assert(mat_t.get_block(0, 0) == std::vector<double>({2.0}));
        assert(mat_t.get_block(0, 1) == std::vector<double>({4.0}));
        assert(mat_t.get_block(1, 0) == std::vector<double>({3.0}));
    } else if (rank == 0) {
        assert(mat_t.row_ptr() == std::vector<int>({0, 2}));
        assert(mat_t.col_ind().size() == 2);
        const int col0 = mat_t.graph->global_to_local.at(0);
        const int col1 = mat_t.graph->global_to_local.at(1);
        assert(mat_t.get_block(0, col0) == std::vector<double>({2.0}));
        assert(mat_t.get_block(0, col1) == std::vector<double>({4.0}));
    } else if (rank == 1) {
        assert(mat_t.row_ptr() == std::vector<int>({0, 1}));
        const int col0 = mat_t.graph->global_to_local.at(0);
        assert(mat_t.get_block(0, col0) == std::vector<double>({3.0}));
    } else {
        assert(mat_t.row_ptr() == std::vector<int>({0}));
        assert(mat_t.col_ind().empty());
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
    assert(Y.row_ptr() == std::vector<int>({0, 2, 4}));
    assert(Y.col_ind() == std::vector<int>({0, 1, 0, 1}));
    assert(Y.row_ptr() == Y.row_ptr());
    assert(Y.col_ind() == Y.col_ind());
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
    assert(C.row_ptr() == C.row_ptr());
    assert(C.col_ind() == C.col_ind());

    if (size == 1) {
        assert(C.row_ptr() == std::vector<int>({0, 2, 4}));
        assert(C.col_ind() == std::vector<int>({0, 1, 0, 1}));
        assert(C.get_block(0, 0) == std::vector<double>({47.0}));
        assert(C.get_block(0, 1) == std::vector<double>({39.0}));
        assert(C.get_block(1, 0) == std::vector<double>({55.0}));
        assert(C.get_block(1, 1) == std::vector<double>({65.0}));
    } else if (rank == 0) {
        assert(C.row_ptr() == std::vector<int>({0, 2}));
        const int col0 = C.graph->global_to_local.at(0);
        const int col1 = C.graph->global_to_local.at(1);
        assert(C.get_block(0, col0) == std::vector<double>({47.0}));
        assert(C.get_block(0, col1) == std::vector<double>({39.0}));
    } else if (rank == 1) {
        assert(C.row_ptr() == std::vector<int>({0, 2}));
        const int col0 = C.graph->global_to_local.at(0);
        const int col1 = C.graph->global_to_local.at(1);
        assert(C.get_block(0, col0) == std::vector<double>({55.0}));
        assert(C.get_block(0, col1) == std::vector<double>({65.0}));
    } else {
        assert(C.row_ptr() == std::vector<int>({0}));
        assert(C.col_ind().empty());
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
    
    if (C.col_ind().size() != owned.size()) {
        std::cout << "Rank " << rank << " C size: " << C.col_ind().size() 
                  << " Expected: " << owned.size() << std::endl;
    }
    assert(C.col_ind().size() == owned.size());
    
    for(size_t k=0; k<C.col_ind().size(); ++k) {
        // Check diagonal values
        const double* d = C.block_data(static_cast<int>(k));
        assert(d[0] == 1.0 && d[3] == 1.0); // Diagonal
        assert(d[1] == 0.0 && d[2] == 0.0); // Off-diagonal
    }
    
    std::cout << "PASSED" << std::endl;
}

std::vector<int> family_block_sizes(MatrixKind kind) {
    switch (kind) {
    case MatrixKind::CSR:
        return {1, 1, 1};
    case MatrixKind::BSR:
        return {2, 2, 2};
    case MatrixKind::VBCSR:
        return {2, 3, 2};
    }
    throw std::runtime_error("Unsupported matrix kind in family_block_sizes");
}

std::vector<std::vector<int>> full_block_adjacency(int n_blocks) {
    std::vector<std::vector<int>> adj(n_blocks);
    for (int row = 0; row < n_blocks; ++row) {
        adj[row].resize(n_blocks);
        for (int col = 0; col < n_blocks; ++col) {
            adj[row][col] = col;
        }
    }
    return adj;
}

std::vector<std::vector<int>> diagonal_block_adjacency(int n_blocks) {
    std::vector<std::vector<int>> adj(n_blocks);
    for (int row = 0; row < n_blocks; ++row) {
        adj[row] = {row};
    }
    return adj;
}

std::vector<std::vector<int>> offdiagonal_block_adjacency(int n_blocks) {
    std::vector<std::vector<int>> adj(n_blocks);
    for (int row = 0; row < n_blocks; ++row) {
        for (int col = 0; col < n_blocks; ++col) {
            if (col != row) {
                adj[row].push_back(col);
            }
        }
    }
    return adj;
}

template <typename T>
void run_matrix_numerical_soundness_suite(
    const std::string& case_label,
    const BlockSpMat<T>& A_base,
    const BlockSpMat<T>& B_base) {
    (void)case_label;
    assert(A_base.matrix_kind() == B_base.matrix_kind());
    const DistGraph& graph = *A_base.graph;
    const int rows = dense_row_count<T>(graph);
    const int cols = dense_col_count<T>(graph);
    assert(rows == cols);

    const std::vector<T> expected_A = generated_dense_matrix<T>(graph, 0);
    const std::vector<T> expected_B = generated_dense_matrix<T>(graph, 1);
    assert_close_generic(A_base.to_dense(), expected_A);
    assert_close_generic(B_base.to_dense(), expected_B);
    BlockSpMat<T> A_ops = A_base.duplicate();
    BlockSpMat<T> B_ops = B_base.duplicate();

    BlockSpMat<T> duplicate = A_base.duplicate();
    assert_close_generic(duplicate.to_dense(), expected_A);

    BlockSpMat<T> roundtrip = A_base.duplicate();
    roundtrip.fill(T(0));
    roundtrip.from_dense(expected_B);
    assert_close_generic(roundtrip.to_dense(), expected_B);

    BlockSpMat<T> copied = A_base.duplicate();
    copied.fill(T(0));
    copied.copy_from(B_base);
    assert_close_generic(copied.to_dense(), expected_B);

    BlockSpMat<T> filled = A_base.duplicate();
    const T fill_value = make_test_value<T>(-0.35, 0.2);
    filled.fill(fill_value);
    assert_close_generic(
        filled.to_dense(),
        std::vector<T>(static_cast<size_t>(rows) * cols, fill_value));

    BlockSpMat<T> scaled = A_base.duplicate();
    const T scale_alpha = make_test_value<T>(1.75, -0.5);
    scaled.scale(scale_alpha);
    std::vector<T> expected_scaled = expected_A;
    for (auto& value : expected_scaled) {
        value *= scale_alpha;
    }
    assert_close_generic(scaled.to_dense(), expected_scaled);

    BlockSpMat<T> conjugated = A_base.duplicate();
    conjugated.conjugate();
    assert_close_generic(conjugated.to_dense(), dense_conjugated(expected_A));

    auto real_part = A_base.get_real();
    auto imag_part = A_base.get_imag();
    assert_close_generic(real_part.to_dense(), dense_real_part(expected_A));
    assert_close_generic(imag_part.to_dense(), dense_imag_part(expected_A));

    DistVector<T> x(A_base.graph);
    fill_generated_vector(x.data, 0.4);
    DistVector<T> y(A_base.graph);
    A_ops.mult(x, y);
    assert_close_generic(
        std::vector<T>(y.data.begin(), y.data.begin() + rows),
        dense_matvec_generic(expected_A, rows, cols, x.data));

    DistMultiVector<T> X(A_base.graph, 3);
    DistMultiVector<T> Y(A_base.graph, 3);
    fill_generated_multivector(X.data, X.local_rows + X.ghost_rows, X.num_vectors, 0.3);
    A_ops.mult_dense(X, Y);
    assert_close_generic(
        Y.data,
        dense_matmat_generic(
            expected_A,
            rows,
            cols,
            X.data,
            X.local_rows + X.ghost_rows,
            X.num_vectors));

    DistVector<T> x_adj(A_base.graph);
    fill_generated_vector(x_adj.data, -0.2);
    DistVector<T> y_adj(A_base.graph);
    A_ops.mult_adjoint(x_adj, y_adj);
    assert_close_generic(
        std::vector<T>(y_adj.data.begin(), y_adj.data.begin() + cols),
        dense_matvec_adjoint_generic(expected_A, rows, cols, x_adj.data));

    DistMultiVector<T> X_adj(A_base.graph, 2);
    DistMultiVector<T> Y_adj(A_base.graph, 2);
    fill_generated_multivector(X_adj.data, X_adj.local_rows + X_adj.ghost_rows, X_adj.num_vectors, -0.5);
    A_ops.mult_dense_adjoint(X_adj, Y_adj);
    assert_close_generic(
        Y_adj.data,
        dense_matmat_adjoint_generic(
            expected_A,
            rows,
            cols,
            X_adj.data,
            X_adj.local_rows + X_adj.ghost_rows,
            X_adj.num_vectors));

    BlockSpMat<T> transpose = A_base.transpose();
    assert_close_generic(
        transpose.to_dense(),
        dense_transpose_conjugate_generic(expected_A, rows, cols));

    BlockSpMat<T> product = A_ops.spmm(B_ops, 0.0);
    assert_close_generic(
        product.to_dense(),
        dense_matmul_generic(expected_A, rows, cols, expected_B, cols));

    BlockSpMat<T> self_product = A_base.duplicate();
    BlockSpMat<T> self_result = self_product.spmm_self(0.0);
    assert_close_generic(
        self_result.to_dense(),
        dense_matmul_generic(expected_A, rows, cols, expected_A, cols));

    const T alpha = make_test_value<T>(0.8, -0.25);
    const T beta = make_test_value<T>(-0.4, 0.1);
    BlockSpMat<T> axpby_matrix = A_base.duplicate();
    axpby_matrix.axpby(alpha, B_base, beta);
    std::vector<T> expected_axpby(expected_A.size(), T(0));
    for (size_t idx = 0; idx < expected_axpby.size(); ++idx) {
        expected_axpby[idx] = alpha * expected_B[idx] + beta * expected_A[idx];
    }
    assert_close_generic(axpby_matrix.to_dense(), expected_axpby);

    const T gamma = make_test_value<T>(-0.3, 0.2);
    BlockSpMat<T> axpy_matrix = A_base.duplicate();
    axpy_matrix.axpy(gamma, B_base);
    std::vector<T> expected_axpy = expected_A;
    for (size_t idx = 0; idx < expected_axpy.size(); ++idx) {
        expected_axpy[idx] += gamma * expected_B[idx];
    }
    assert_close_generic(axpy_matrix.to_dense(), expected_axpy);

    BlockSpMat<T> added_source = A_base.duplicate();
    BlockSpMat<T> added = added_source.add(B_base, 1.25, -0.5);
    std::vector<T> expected_added(expected_A.size(), T(0));
    for (size_t idx = 0; idx < expected_added.size(); ++idx) {
        expected_added[idx] = 1.25 * expected_A[idx] - 0.5 * expected_B[idx];
    }
    assert_close_generic(added.to_dense(), expected_added);

    BlockSpMat<T> shifted = A_base.duplicate();
    const T shift_alpha = make_test_value<T>(0.6, -0.15);
    shifted.shift(shift_alpha);
    std::vector<T> expected_shifted = expected_A;
    for (int idx = 0; idx < rows; ++idx) {
        expected_shifted[static_cast<size_t>(idx) * cols + idx] += shift_alpha;
    }
    assert_close_generic(shifted.to_dense(), expected_shifted);

    DistVector<T> diag(A_base.graph);
    fill_generated_vector(diag.data, 0.9);
    BlockSpMat<T> with_diag = A_base.duplicate();
    with_diag.add_diagonal(diag);
    std::vector<T> expected_with_diag = expected_A;
    for (int idx = 0; idx < rows; ++idx) {
        expected_with_diag[static_cast<size_t>(idx) * cols + idx] += diag.data[idx];
    }
    assert_close_generic(with_diag.to_dense(), expected_with_diag);

    DistVector<T> comm_diag(A_base.graph);
    fill_generated_vector(comm_diag.data, -0.7);
    BlockSpMat<T> commutator = A_base.duplicate();
    commutator.fill(T(0));
    A_ops.commutator_diagonal(comm_diag, commutator);
    assert_close_generic(
        commutator.to_dense(),
        dense_commutator_diagonal(expected_A, rows, cols, comm_diag.data));

    BlockSpMat<T> filtered = A_base.duplicate();
    const double filter_threshold = 0.1;
    const std::vector<T> expected_filtered = dense_after_filter_expected(filtered, filter_threshold);
    filtered.filter_blocks(filter_threshold);
    assert_close_generic(filtered.to_dense(), expected_filtered);

    const std::vector<int> subset = {0, 2};
    BlockSpMat<T> submatrix = A_ops.extract_submatrix(subset);
    assert_close_generic(
        submatrix.to_dense(),
        extract_dense_block_submatrix(A_base, subset));

    const std::vector<std::vector<int>> batched_subsets = {{0, 2}, {1, 2}};
    auto extracted_batch = A_ops.extract_submatrix_batched(batched_subsets);
    assert(extracted_batch.size() == batched_subsets.size());
    for (size_t idx = 0; idx < batched_subsets.size(); ++idx) {
        assert_close_generic(
            extracted_batch[idx].to_dense(),
            extract_dense_block_submatrix(A_base, batched_subsets[idx]));
    }

    BlockSpMat<T> insert_target = A_base.duplicate();
    BlockSpMat<T> patch = insert_target.extract_submatrix(subset);
    const T patch_scale = make_test_value<T>(-1.2, 0.35);
    patch.scale(patch_scale);
    const std::vector<T> expected_inserted = with_inserted_dense_block_submatrix(
        A_base,
        subset,
        [&]() {
            std::vector<T> sub_dense = extract_dense_block_submatrix(A_base, subset);
            for (auto& value : sub_dense) {
                value *= patch_scale;
            }
            return sub_dense;
        }());
    insert_target.insert_submatrix(patch, subset);
    assert_close_generic(insert_target.to_dense(), expected_inserted);
}

template <typename T>
void run_structure_change_axpby_suite(
    const std::string& case_label,
    MatrixKind kind,
    bool use_backend_construction) {
    (void)case_label;
    const uint32_t page_size = 2;
    const std::vector<int> block_sizes = family_block_sizes(kind);

    DistGraph graph_y(MPI_COMM_SELF);
    DistGraph graph_x(MPI_COMM_SELF);
    graph_y.construct_serial(3, block_sizes, diagonal_block_adjacency(3));
    graph_x.construct_serial(3, block_sizes, offdiagonal_block_adjacency(3));

    BlockSpMat<T> Y = use_backend_construction
        ? build_generated_matrix_via_backend<T>(&graph_y, kind, 0, page_size)
        : build_generated_matrix_via_api<T>(&graph_y, 0, page_size);
    BlockSpMat<T> X = use_backend_construction
        ? build_generated_matrix_via_backend<T>(&graph_x, kind, 1, page_size)
        : build_generated_matrix_via_api<T>(&graph_x, 1, page_size);

    const std::vector<T> dense_y = Y.to_dense();
    const std::vector<T> dense_x = X.to_dense();
    const T alpha = make_test_value<T>(1.1, -0.2);
    const T beta = make_test_value<T>(-0.6, 0.3);

    Y.axpby(alpha, X, beta);

    std::vector<T> expected(dense_y.size(), T(0));
    for (size_t idx = 0; idx < expected.size(); ++idx) {
        expected[idx] = alpha * dense_x[idx] + beta * dense_y[idx];
    }
    assert_close_generic(Y.to_dense(), expected);
}

template <typename T>
void test_matrix_family_numerical_soundness_impl(const char* scalar_label) {
    std::cout << "Testing Matrix Family Numerical Soundness (" << scalar_label << ")..." << std::endl;

    const uint32_t page_size = 2;
    for (MatrixKind kind : {MatrixKind::CSR, MatrixKind::BSR, MatrixKind::VBCSR}) {
        DistGraph graph(MPI_COMM_SELF);
        graph.construct_serial(3, family_block_sizes(kind), full_block_adjacency(3));

        BlockSpMat<T> api_A = build_generated_matrix_via_api<T>(&graph, 0, page_size);
        BlockSpMat<T> api_B = build_generated_matrix_via_api<T>(&graph, 1, page_size);
        run_matrix_numerical_soundness_suite<T>(
            std::string(matrix_kind_name(kind)) + " direct",
            api_A,
            api_B);
        run_structure_change_axpby_suite<T>(
            std::string(matrix_kind_name(kind)) + " direct structure change",
            kind,
            false);

        BlockSpMat<T> backend_A = build_generated_matrix_via_backend<T>(&graph, kind, 0, page_size);
        BlockSpMat<T> backend_B = build_generated_matrix_via_backend<T>(&graph, kind, 1, page_size);
        run_matrix_numerical_soundness_suite<T>(
            std::string(matrix_kind_name(kind)) + " backend",
            backend_A,
            backend_B);
        run_structure_change_axpby_suite<T>(
            std::string(matrix_kind_name(kind)) + " backend structure change",
            kind,
            true);
    }

    std::cout << "PASSED" << std::endl;
}

void test_matrix_family_numerical_soundness_real() {
    test_matrix_family_numerical_soundness_impl<double>("Real");
}

void test_matrix_family_numerical_soundness_complex() {
    test_matrix_family_numerical_soundness_impl<std::complex<double>>("Complex");
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
    test_const_logical_views_and_family_rejection();
    test_paged_storage_contracts();
    test_vbcsr_batch_views();
    test_batched_blas_capability_flags();
    test_contiguous_api_compatibility();
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
    test_bsr_spmm_distributed_unsorted_global_column_regression();
    test_csr_backend_family_preservation();
    test_vbcsr_backend_shape_registry_and_family();
    test_vbcsr_axpby_structure_change();
    test_vbcsr_shape_batched_apply_kernels();
    test_vbcsr_shape_batched_apply_kernels_complex();
    test_vbcsr_shape_batched_spmm();
    test_csr_backend_dispatch_kernels();
    test_csr_backend_dispatch_kernels_complex();
    test_csr_page_cap_policy();
    test_blockspmat_csr_page_cap_repack_and_propagation();
    test_blockspmat_bsr_page_settings_repack_and_propagation();
    test_blockspmat_vbcsr_page_settings_repack_and_propagation();
    test_bsr_vendor_batch_cache_metadata();
    test_bsr_vendor_dispatch_selection();
    test_bsr_vendor_cache_reuse_repeated_apply();
    test_bsr_vendor_cache_invalidation_on_structure_change();
#ifdef VBCSR_HAVE_MKL_BSR_SPARSE
    test_bsr_vendor_complex_dispatch_selection();
#endif
    test_csr_vendor_page_cache_metadata();
    test_csr_vendor_dispatch_selection();
    test_csr_vendor_cache_reuse_repeated_apply();
    test_csr_vendor_cache_invalidation_on_structure_change();
    test_csr_transpose_native_serial();
    test_csr_transpose_native_distributed();
    test_csr_axpby_same_structure_native();
    test_csr_axpby_structure_change_native();
    test_csr_spmm_native_distributed();
    test_spmm();
    test_matrix_family_numerical_soundness_real();
    test_matrix_family_numerical_soundness_complex();

    MPI_Finalize();
    return 0;
}

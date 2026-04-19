#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <mpi.h>

#include <cstring>

#include "dist_graph.hpp"
#include "block_csr.hpp"
#include "dist_vector.hpp"
#include "dist_multivector.hpp"
#include "graphmf.hpp"

namespace py = pybind11;
using namespace vbcsr;

#include "pybind_common.hpp"

template <typename T>
py::array_t<T> make_owned_array_1d(const T* data, py::ssize_t count) {
    auto* heap_data = new T[static_cast<size_t>(count)];
    std::memcpy(heap_data, data, static_cast<size_t>(count) * sizeof(T));
    py::capsule owner(heap_data, [](void* ptr) {
        delete[] static_cast<T*>(ptr);
    });
    return py::array_t<T>({count}, {static_cast<py::ssize_t>(sizeof(T))}, heap_data, owner);
}

template <typename T>
py::array_t<T> make_owned_array_1d(const std::vector<T>& data) {
    return make_owned_array_1d(data.data(), static_cast<py::ssize_t>(data.size()));
}

template <typename T>
py::array_t<T> make_owned_array_2d_row_major(const std::vector<T>& data, py::ssize_t rows, py::ssize_t cols) {
    auto* heap_data = new T[data.size()];
    std::memcpy(heap_data, data.data(), data.size() * sizeof(T));
    py::capsule owner(heap_data, [](void* ptr) {
        delete[] static_cast<T*>(ptr);
    });
    return py::array_t<T>(
        {rows, cols},
        {cols * static_cast<py::ssize_t>(sizeof(T)), static_cast<py::ssize_t>(sizeof(T))},
        heap_data,
        owner);
}

// Forward declaration for atomic module binding
void bind_atomic_module(py::module& m);

template<typename T>
void bind_dist_vector(py::module& m, const std::string& name) {
    py::class_<DistVector<T>>(m, name.c_str(), py::buffer_protocol())
        .def(py::init<DistGraph*>(), py::keep_alive<1, 2>())
        .def("sync_ghosts", &DistVector<T>::sync_ghosts)
        .def("reduce_ghosts", &DistVector<T>::reduce_ghosts)
        .def("set_constant", &DistVector<T>::set_constant)
        .def("scale", &DistVector<T>::scale)
        .def("axpy", &DistVector<T>::axpy)
        .def("axpby", &DistVector<T>::axpby)
        .def("pointwise_mult", &DistVector<T>::pointwise_mult)
        .def("dot", &DistVector<T>::dot)
        .def("duplicate", &DistVector<T>::duplicate)
        .def("copy_from", &DistVector<T>::copy_from)
        .def("set_random_normal", &DistVector<T>::set_random_normal)
        .def_property_readonly("local_size", [](const DistVector<T>& v) { return v.local_size; })
        .def_property_readonly("ghost_size", [](const DistVector<T>& v) { return v.ghost_size; })
        .def_property_readonly("full_size", &DistVector<T>::full_size)
        // Buffer protocol
        .def_buffer([](DistVector<T>& v) -> py::buffer_info {
            return py::buffer_info(
                v.data.data(),                               /* Pointer to buffer */
                sizeof(T),                                   /* Size of one scalar */
                py::format_descriptor<T>::format(),          /* Python struct-style format descriptor */
                1,                                           /* Number of dimensions */
                { (size_t)v.data.size() },                   /* Buffer dimensions */
                { sizeof(T) }                                /* Strides (in bytes) */
            );
        });
}

template<typename T>
void bind_dist_multivector(py::module& m, const std::string& name) {
    py::class_<DistMultiVector<T>>(m, name.c_str(), py::buffer_protocol())
        .def(py::init<DistGraph*, int>(), py::keep_alive<1, 2>())
        .def("sync_ghosts", &DistMultiVector<T>::sync_ghosts)
        .def("reduce_ghosts", &DistMultiVector<T>::reduce_ghosts)
        .def("set_constant", &DistMultiVector<T>::set_constant)
        .def("scale", &DistMultiVector<T>::scale)
        .def("axpy", &DistMultiVector<T>::axpy)
        .def("axpby", &DistMultiVector<T>::axpby)
        .def("pointwise_mult", py::overload_cast<const DistMultiVector<T>&>(&DistMultiVector<T>::pointwise_mult))
        .def("pointwise_mult_vec", py::overload_cast<const DistVector<T>&>(&DistMultiVector<T>::pointwise_mult))
        .def("bdot", &DistMultiVector<T>::bdot)
        .def("duplicate", &DistMultiVector<T>::duplicate)
        .def("copy_from", &DistMultiVector<T>::copy_from)
        .def("set_random_normal", &DistMultiVector<T>::set_random_normal)
        .def_property_readonly("local_rows", [](const DistMultiVector<T>& v) { return v.local_rows; })
        .def_property_readonly("ghost_rows", [](const DistMultiVector<T>& v) { return v.ghost_rows; })
        .def_property_readonly("num_vectors", [](const DistMultiVector<T>& v) { return v.num_vectors; })
        .def_buffer([](DistMultiVector<T>& v) -> py::buffer_info {
            return py::buffer_info(
                v.data.data(),
                sizeof(T),
                py::format_descriptor<T>::format(),
                2,
                { (size_t)(v.local_rows + v.ghost_rows), (size_t)v.num_vectors }, // Shape: (rows, cols)
                { sizeof(T), sizeof(T) * (v.local_rows + v.ghost_rows) }          // Strides: (row_stride, col_stride)
            );
        });
}

template<typename T>
BlockSpMat<T> py_graph_matrix_function(BlockSpMat<T>& self, const std::string& func_name, const std::string& method, bool verbose) {
    std::function<T(T)> func;
    if (func_name == "inv") {
        func = [](T x) { return T(1.0 / (x + 1e-10)); };
    } else if (func_name == "sqrt") {
        func = [](T x) { return T(std::sqrt(x)); };
    } else if (func_name == "isqrt") {
        func = [](T x) { return T(1.0 / (std::sqrt(x)+1e-10)); };
    } else if (func_name == "exp") {
        func = [](T x) { return T(std::exp(x)); };
    } else {
        throw std::runtime_error("Unsupported function: " + func_name);
    }

    BlockSpMat<T> res = self.duplicate();
    graph_matrix_function<T>(self, &res, func, method, verbose);
    return res;
}

template<typename T>
void bind_block_spmat(py::module& m, const std::string& name) {
    py::class_<BlockSpMat<T>>(m, name.c_str())
        .def(py::init<DistGraph*>(), py::keep_alive<1, 2>())
        .def_property_readonly(
            "graph",
            [](BlockSpMat<T>& self) { return self.graph; },
            py::return_value_policy::reference_internal)
        .def("add_block", [](BlockSpMat<T>& mat, int g_row, int g_col, py::array_t<T> data, AssemblyMode mode) {
            py::buffer_info info = data.request();
            if (info.ndim != 2) throw std::runtime_error("Data must be 2D");
            int rows = info.shape[0];
            int cols = info.shape[1];
            // Check layout
            MatrixLayout layout = MatrixLayout::RowMajor;
            if (info.strides[0] == sizeof(T) && info.strides[1] == sizeof(T) * rows) {
                layout = MatrixLayout::ColMajor;
            }
            mat.add_block(g_row, g_col, static_cast<T*>(info.ptr), rows, cols, mode, layout);
        }, py::arg("g_row"), py::arg("g_col"), py::arg("data"), py::arg("mode") = AssemblyMode::ADD)
        .def("assemble", &BlockSpMat<T>::assemble)
        .def("mult", &BlockSpMat<T>::mult)
        .def("mult_dense", &BlockSpMat<T>::mult_dense)
        .def("mult_adjoint", &BlockSpMat<T>::mult_adjoint)
        .def("mult_dense_adjoint", &BlockSpMat<T>::mult_dense_adjoint)
        .def("scale", &BlockSpMat<T>::scale)
        .def("conjugate", &BlockSpMat<T>::conjugate)
        .def("real", &BlockSpMat<T>::get_real)
        .def("imag", &BlockSpMat<T>::get_imag)
        .def("shift", &BlockSpMat<T>::shift)
        .def("add_diagonal", &BlockSpMat<T>::add_diagonal)
        .def("axpy", &BlockSpMat<T>::axpy)
        .def("duplicate", &BlockSpMat<T>::duplicate, py::arg("independent_graph") = true)
        .def("save_matrix_market", &BlockSpMat<T>::save_matrix_market)
        .def("spmm", &BlockSpMat<T>::spmm, py::arg("B"), py::arg("threshold"), py::arg("transA") = false, py::arg("transB") = false)
        .def("spmm_self", &BlockSpMat<T>::spmm_self, py::arg("threshold"), py::arg("transA") = false)
        .def("add", &BlockSpMat<T>::add, py::arg("B"), py::arg("alpha") = 1.0, py::arg("beta") = 1.0)
        .def("transpose", &BlockSpMat<T>::transpose)
        .def("extract_submatrix", &BlockSpMat<T>::extract_submatrix, py::arg("global_indices"))
        .def("insert_submatrix", &BlockSpMat<T>::insert_submatrix, py::arg("submat"), py::arg("global_indices"))
        .def("get_block", [](const BlockSpMat<T>& self, int row, int col) {
            std::vector<T> vec = self.get_block(row, col);
            if (vec.empty()) return py::array_t<T>(); // Return empty array or None? Empty array is safer.
            
            int r_dim = self.block_row_dim(row);
            int c_dim = self.block_col_dim(col);
            return make_owned_array_2d_row_major(vec, r_dim, c_dim);
        }, py::arg("row"), py::arg("col"))
        .def("get_values", [](const BlockSpMat<T>& self) {
            return make_owned_array_1d(self.get_values(MatrixLayout::RowMajor));
        })
        .def("get_block_density", &BlockSpMat<T>::get_block_density)
        .def("filter_blocks", &BlockSpMat<T>::filter_blocks)
        .def_property_readonly("matrix_kind", [](const BlockSpMat<T>& self) {
            return self.matrix_kind_string();
        })
        .def_property_readonly("local_nnz", [](const BlockSpMat<T>& self) {
            return self.local_scalar_nnz();
        })
        .def_property_readonly("global_nnz", [](const BlockSpMat<T>& self) {
            long long local_nnz = static_cast<long long>(self.local_scalar_nnz());
            long long global_nnz = local_nnz;
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (initialized && self.graph != nullptr && self.graph->comm != MPI_COMM_NULL && self.graph->size > 1) {
                MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_LONG_LONG, MPI_SUM, self.graph->comm);
            }
            return global_nnz;
        })
        .def("to_dense", [](const BlockSpMat<T>& self) {
            // Return 2D numpy array
            std::vector<T> vec = self.to_dense();
            
            // Calculate dimensions (same logic as in C++ to_dense)
            int n_owned = self.graph->owned_global_indices.size();
            int my_rows = self.graph->block_offsets[n_owned];
            int my_cols = self.graph->block_offsets.back();
            return make_owned_array_2d_row_major(vec, my_rows, my_cols);
        })
        .def("from_dense", [](BlockSpMat<T>& self, py::array_t<T> array) {
            // Check dimensions
            if (array.ndim() != 2) throw std::runtime_error("from_dense: array must be 2D");
            
            // Convert to flat vector (RowMajor)
            std::vector<T> vec(array.size());
            
            // Copy logic using buffer info
            py::buffer_info info = array.request();
            
            // Check if contiguous and RowMajor for fast copy
            bool is_contiguous = (info.strides[1] == sizeof(T) && info.strides[0] == sizeof(T) * info.shape[1]);
            
            if (is_contiguous) {
                std::memcpy(vec.data(), info.ptr, sizeof(T) * array.size());
            } else {
                // Slow copy for non-contiguous
                T* ptr = static_cast<T*>(info.ptr);
                // We need to iterate carefully if strides are weird, but let's assume standard numpy access
                // Actually, let's just use unchecked access for simplicity if we wanted, but memcpy is preferred.
                // If not contiguous, we can let numpy copy it to a contiguous buffer first?
                // Or just loop.
                auto r = array.template unchecked<2>();
                for (py::ssize_t i = 0; i < info.shape[0]; i++) {
                    for (py::ssize_t j = 0; j < info.shape[1]; j++) {
                        vec[i * info.shape[1] + j] = r(i, j);
                    }
                }
            }
            
            self.from_dense(vec);
        }, py::arg("array"))
        .def("spmf", &py_graph_matrix_function<T>, py::arg("func_name"), py::arg("method") = "lanczos", py::arg("verbose") = false)
        .def_property_readonly("row_ptr", [](const BlockSpMat<T>& self) {
            return make_owned_array_1d(self.row_ptr());
        })
        .def_property_readonly("col_ind", [](const BlockSpMat<T>& self) {
            return make_owned_array_1d(self.col_ind());
        })
        ;
}

PYBIND11_MODULE(vbcsr_core, m) {
    m.doc() = "VBCSR C++ Core Bindings";

    py::enum_<AssemblyMode>(m, "AssemblyMode")
        .value("INSERT", AssemblyMode::INSERT)
        .value("ADD", AssemblyMode::ADD)
        .export_values();

    m.def("finalize_mpi", &finalize_mpi, "Finalize MPI if initialized");

    py::class_<DistGraph>(m, "DistGraph")
        .def(py::init([](py::object comm_obj) {
            return new DistGraph(get_mpi_comm(comm_obj));
        }), py::arg("comm") = py::none())
        .def("construct_serial", &DistGraph::construct_serial)
        .def("construct_distributed", &DistGraph::construct_distributed)
        .def_readonly("owned_global_indices", &DistGraph::owned_global_indices)
        .def_readonly("ghost_global_indices", &DistGraph::ghost_global_indices)
        .def_readonly("block_sizes", &DistGraph::block_sizes)
        .def_property_readonly("owned_scalar_rows", [](const DistGraph& self) {
            const size_t owned_blocks = self.owned_global_indices.size();
            if (self.block_offsets.size() <= owned_blocks) {
                return 0;
            }
            return self.block_offsets[owned_blocks];
        })
        .def_property_readonly("local_scalar_cols", [](const DistGraph& self) {
            return self.block_offsets.empty() ? 0 : self.block_offsets.back();
        })
        .def_property_readonly("global_scalar_rows", [](const DistGraph& self) {
            int local_rows = 0;
            const size_t owned_blocks = self.owned_global_indices.size();
            if (self.block_offsets.size() > owned_blocks) {
                local_rows = self.block_offsets[owned_blocks];
            }

            int global_rows = local_rows;
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (initialized && self.comm != MPI_COMM_NULL && self.size > 1) {
                MPI_Allreduce(&local_rows, &global_rows, 1, MPI_INT, MPI_SUM, self.comm);
            }
            return global_rows;
        })
        .def("get_local_index", [](const DistGraph& self, int gid) {
            auto it = self.global_to_local.find(gid);
            if (it == self.global_to_local.end()) return -1;
            return it->second;
        })
        .def_readonly("rank", &DistGraph::rank)
        .def_readonly("size", &DistGraph::size);

    bind_dist_vector<double>(m, "DistVector_Double");
    bind_dist_vector<std::complex<double>>(m, "DistVector_Complex");

    bind_dist_multivector<double>(m, "DistMultiVector_Double");
    bind_dist_multivector<std::complex<double>>(m, "DistMultiVector_Complex");

    bind_block_spmat<double>(m, "BlockSpMat_Double");
    bind_block_spmat<std::complex<double>>(m, "BlockSpMat_Complex");

    bind_atomic_module(m);
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <mpi.h>

#include "dist_graph.hpp"
#include "block_csr.hpp"
#include "dist_vector.hpp"
#include "dist_multivector.hpp"

namespace py = pybind11;
using namespace vbcsr;

// Helper to get MPI_Comm from python object (mpi4py)
// For now, we assume the user passes an integer (MPI_Comm_c2f) or we use a capsule.
// mpi4py passes MPI_Comm as an integer (on some platforms) or a PyObject.
// A robust way requires mpi4py headers, but for simplicity we can accept an integer (intptr_t).
MPI_Comm get_mpi_comm(py::object comm_obj) {
    if (comm_obj.is_none()) return MPI_COMM_WORLD;
    // Try to get 'py2f' method if it's an mpi4py communicator
    if (py::hasattr(comm_obj, "py2f")) {
        MPI_Fint f_handle = (MPI_Fint)comm_obj.attr("py2f")().cast<intptr_t>();
        return MPI_Comm_f2c(f_handle);
    }
    // Assume it's an integer handle (legacy fallback, maybe also needs f2c if it's a Fortran handle)
    // Given mpi4py context, it's likely F handle.
    MPI_Fint f_handle = (MPI_Fint)comm_obj.cast<intptr_t>();
    return MPI_Comm_f2c(f_handle);
}

template<typename T>
void bind_dist_vector(py::module& m, const std::string& name) {
    py::class_<DistVector<T>>(m, name.c_str(), py::buffer_protocol())
        .def(py::init<DistGraph*>())
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
        .def(py::init<DistGraph*, int>())
        .def("sync_ghosts", &DistMultiVector<T>::sync_ghosts)
        .def("reduce_ghosts", &DistMultiVector<T>::reduce_ghosts)
        .def("set_constant", &DistMultiVector<T>::set_constant)
        .def("scale", &DistMultiVector<T>::scale)
        .def("axpy", &DistMultiVector<T>::axpy)
        .def("axpby", &DistMultiVector<T>::axpby)
        .def("pointwise_mult", py::overload_cast<const DistMultiVector<T>&>(&DistMultiVector<T>::pointwise_mult))
        .def("pointwise_mult_vec", py::overload_cast<const DistVector<T>&>(&DistMultiVector<T>::pointwise_mult))
        .def("bdot", &DistMultiVector<T>::bdot)
        .def("duplicate", [](const DistMultiVector<T>& v) {
            // DistMultiVector doesn't have a duplicate method in C++, implement a simple one here or add to C++
            // Assuming C++ side doesn't have it, we can create new and copy.
            // But wait, DistMultiVector copy constructor/assignment might be deleted or not efficient?
            // Let's rely on copy_from.
            DistMultiVector<T> new_v(v.graph, v.num_vectors);
            new_v.copy_from(v);
            return new_v;
        })
        .def("copy_from", &DistMultiVector<T>::copy_from)
        .def_property_readonly("local_rows", [](const DistMultiVector<T>& v) { return v.local_rows; })
        .def_property_readonly("ghost_rows", [](const DistMultiVector<T>& v) { return v.ghost_rows; })
        .def_property_readonly("num_vectors", [](const DistMultiVector<T>& v) { return v.num_vectors; })
        // Buffer protocol (Column-Major storage in C++)
        // Exposed as (rows, cols) 2D array
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
void bind_block_spmat(py::module& m, const std::string& name) {
    py::class_<BlockSpMat<T>>(m, name.c_str())
        .def(py::init<DistGraph*>())
        .def_readonly("graph", &BlockSpMat<T>::graph)
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
        .def("shift", &BlockSpMat<T>::shift)
        .def("add_diagonal", &BlockSpMat<T>::add_diagonal)
        .def("axpy", &BlockSpMat<T>::axpy)
        .def("duplicate", &BlockSpMat<T>::duplicate)
        .def("save_matrix_market", &BlockSpMat<T>::save_matrix_market)
        .def("spmm", &BlockSpMat<T>::spmm, py::arg("B"), py::arg("threshold"), py::arg("transA") = false, py::arg("transB") = false)
        .def("spmm_self", &BlockSpMat<T>::spmm_self, py::arg("threshold"), py::arg("transA") = false)
        .def("add", &BlockSpMat<T>::add, py::arg("B"), py::arg("alpha") = 1.0, py::arg("beta") = 1.0)
        .def("transpose", &BlockSpMat<T>::transpose)
        .def("extract_submatrix", &BlockSpMat<T>::extract_submatrix, py::arg("global_indices"))
        .def("insert_submatrix", &BlockSpMat<T>::insert_submatrix, py::arg("submat"), py::arg("global_indices"))
        .def("to_dense", [](const BlockSpMat<T>& self) {
            // Return 2D numpy array
            std::vector<T> vec = self.to_dense();
            
            // Calculate dimensions (same logic as in C++ to_dense)
            int n_owned = self.graph->owned_global_indices.size();
            int my_rows = self.graph->block_offsets[n_owned];
            int my_cols = self.graph->block_offsets.back();
            
            // Create numpy array with shape (rows, cols)
            // We let pybind11 copy the data to manage lifetime safely and simply
            return py::array_t<T>(
                {my_rows, my_cols}, // shape
                {my_cols * sizeof(T), sizeof(T)}, // strides (RowMajor)
                vec.data() // data pointer
            );
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
        }, py::arg("array"));
}

PYBIND11_MODULE(vbcsr_core, m) {
    m.doc() = "VBCSR C++ Core Bindings";

    py::enum_<AssemblyMode>(m, "AssemblyMode")
        .value("INSERT", AssemblyMode::INSERT)
        .value("ADD", AssemblyMode::ADD)
        .export_values();

    py::class_<DistGraph>(m, "DistGraph")
        .def(py::init([](py::object comm_obj) {
            return new DistGraph(get_mpi_comm(comm_obj));
        }), py::arg("comm") = py::none())
        .def("construct_serial", &DistGraph::construct_serial)
        .def("construct_distributed", &DistGraph::construct_distributed)
        .def_readonly("owned_global_indices", &DistGraph::owned_global_indices)
        .def_readonly("block_sizes", &DistGraph::block_sizes)
        .def_readonly("rank", &DistGraph::rank)
        .def_readonly("size", &DistGraph::size);

    bind_dist_vector<double>(m, "DistVector_Double");
    bind_dist_vector<std::complex<double>>(m, "DistVector_Complex");

    bind_dist_multivector<double>(m, "DistMultiVector_Double");
    bind_dist_multivector<std::complex<double>>(m, "DistMultiVector_Complex");

    bind_block_spmat<double>(m, "BlockSpMat_Double");
    bind_block_spmat<std::complex<double>>(m, "BlockSpMat_Complex");
}

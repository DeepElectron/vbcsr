#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <mpi.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "dist_graph.hpp"
#include "block_csr.hpp"
#include "dist_vector.hpp"
#include "dist_multivector.hpp"
#include "detail/ops/spmf/graph_function.hpp"

namespace py = pybind11;
using namespace vbcsr;

#include "pybind_common.hpp"

// Chunked parallel element copy: matrix-scale exports (values, col_ind) are
// GB-sized, and one serial memcpy at export was measured at the same cost as
// the SpGEMM copy-out it mirrors. No Python API is touched inside the region.
template <typename T>
void copy_elements_parallel(T* dst, const T* src, size_t count) {
    const int64_t element_count = static_cast<int64_t>(count);
    #pragma omp parallel for schedule(static)
    for (int64_t idx = 0; idx < element_count; ++idx) {
        dst[idx] = src[idx];
    }
}

// Wraps a heap-allocated vector in a numpy array without copying: the capsule
// owns the vector, numpy views its storage.
template <typename T>
py::array_t<T> adopt_vector_array(std::vector<T>&& data,
                                  std::vector<py::ssize_t> shape,
                                  std::vector<py::ssize_t> strides) {
    auto* holder = new std::vector<T>(std::move(data));
    py::capsule owner(holder, [](void* ptr) {
        delete static_cast<std::vector<T>*>(ptr);
    });
    return py::array_t<T>(std::move(shape), std::move(strides), holder->data(), owner);
}

template <typename T>
py::array_t<T> make_owned_array_1d(const T* data, py::ssize_t count) {
    auto* heap_data = new T[static_cast<size_t>(count)];
    copy_elements_parallel(heap_data, data, static_cast<size_t>(count));
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
py::array_t<T> make_owned_array_1d(const vbcsr::detail::NumaVector<T>& data) {
    return make_owned_array_1d(data.data(), static_cast<py::ssize_t>(data.size()));
}

// Rvalue overload: adopt the vector's storage instead of copying it (the
// caller's temporary — e.g. get_values() — already owns a fresh buffer).
template <typename T>
py::array_t<T> make_owned_array_1d(std::vector<T>&& data) {
    const py::ssize_t count = static_cast<py::ssize_t>(data.size());
    return adopt_vector_array(
        std::move(data), {count}, {static_cast<py::ssize_t>(sizeof(T))});
}

template <typename T>
py::array_t<T> make_owned_array_2d_row_major(const std::vector<T>& data, py::ssize_t rows, py::ssize_t cols) {
    auto* heap_data = new T[data.size()];
    copy_elements_parallel(heap_data, data.data(), data.size());
    py::capsule owner(heap_data, [](void* ptr) {
        delete[] static_cast<T*>(ptr);
    });
    return py::array_t<T>(
        {rows, cols},
        {cols * static_cast<py::ssize_t>(sizeof(T)), static_cast<py::ssize_t>(sizeof(T))},
        heap_data,
        owner);
}

template <typename T>
py::array_t<T> make_owned_array_2d_row_major(std::vector<T>&& data, py::ssize_t rows, py::ssize_t cols) {
    return adopt_vector_array(
        std::move(data),
        {rows, cols},
        {cols * static_cast<py::ssize_t>(sizeof(T)), static_cast<py::ssize_t>(sizeof(T))});
}

template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast> as_row_major_2d(
    py::array_t<T> array,
    const char* context) {
    py::array_t<T, py::array::c_style | py::array::forcecast> contiguous(array);
    if (contiguous.ndim() != 2) {
        throw std::runtime_error(std::string(context) + ": array must be 2D");
    }
    return contiguous;
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
        .def("reset_comm_stats", &DistVector<T>::reset_comm_stats)
        .def_property_readonly("comm_seconds", [](const DistVector<T>& v) { return v.comm_seconds; })
        .def_property_readonly("comm_calls", [](const DistVector<T>& v) { return v.comm_calls; })
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
        .def("reset_comm_stats", &DistMultiVector<T>::reset_comm_stats)
        .def_property_readonly("comm_seconds", [](const DistMultiVector<T>& v) { return v.comm_seconds; })
        .def_property_readonly("comm_calls", [](const DistMultiVector<T>& v) { return v.comm_calls; })
        .def_property_readonly("local_rows", [](const DistMultiVector<T>& v) { return v.local_rows; })
        .def_property_readonly("ghost_rows", [](const DistMultiVector<T>& v) { return v.ghost_rows; })
        .def_property_readonly("num_vectors", [](const DistMultiVector<T>& v) { return v.num_vectors; })
        .def_buffer([](DistMultiVector<T>& v) -> py::buffer_info {
            // Row-major storage with a padded leading dimension: the exposed
            // numpy shape is unchanged, the array is now C-ordered (strided
            // over the padding lanes).
            return py::buffer_info(
                v.data.data(),
                sizeof(T),
                py::format_descriptor<T>::format(),
                2,
                { (size_t)(v.local_rows + v.ghost_rows), (size_t)v.num_vectors }, // Shape: (rows, cols)
                { sizeof(T) * (size_t)v.ld, sizeof(T) }                           // Strides: (row_stride, col_stride)
            );
        });
}

template<typename T>
BlockSpMat<T> py_graph_matrix_function(BlockSpMat<T>& self, const std::string& func_name, bool verbose) {
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
    graph_matrix_function<T>(self, &res, func, verbose);
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
            auto contiguous = as_row_major_2d<T>(data, "add_block");
            py::buffer_info info = contiguous.request();
            const int rows = static_cast<int>(info.shape[0]);
            const int cols = static_cast<int>(info.shape[1]);
            mat.add_block(
                g_row,
                g_col,
                static_cast<T*>(info.ptr),
                rows,
                cols,
                mode,
                MatrixLayout::RowMajor);
        }, py::arg("g_row"), py::arg("g_col"), py::arg("data"), py::arg("mode") = AssemblyMode::ADD)
        .def("assemble", &BlockSpMat<T>::assemble)
        .def("redistribute",
             static_cast<BlockSpMat<T> (BlockSpMat<T>::*)(DistGraph*, AssemblyMode) const>(
                 &BlockSpMat<T>::redistribute),
             py::arg("target_graph"), py::arg("mode") = AssemblyMode::INSERT,
             // the returned matrix holds a DistGraph* into target_graph: pin it.
             py::keep_alive<0, 2>(),
             "Redistribute to a different partition of the same global structure "
             "(same comm; doc/design/35). mode INSERT=repartition, ADD=reduce.")
        .def("redistribute_cross",
             [](const BlockSpMat<T>& self, DistGraph* target_graph, RedistOp op,
                py::object common_comm) {
                 return self.redistribute(target_graph, op, get_mpi_comm(common_comm));
             },
             py::arg("target_graph"), py::arg("op"), py::arg("common_comm"),
             // the returned matrix holds a DistGraph* into target_graph: pin it.
             py::keep_alive<0, 2>(),
             "Cross-comm redistribute (doc/design/35 incr2): move blocks from this "
             "matrix's partition to target_graph's partition, transporting on "
             "common_comm. op Copy=broadcast/send-down, Sum=reduce-up.")
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
        .def("axpby", &BlockSpMat<T>::axpby)
        .def("copy_from", &BlockSpMat<T>::copy_from)
        .def("fill", &BlockSpMat<T>::fill)
        .def("fill_random", &BlockSpMat<T>::fill_random)
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
            return make_owned_array_2d_row_major(std::move(vec), r_dim, c_dim);
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
        .def_property_readonly("local_block_nnz", [](const BlockSpMat<T>& self) {
            return self.local_block_nnz();
        })
        .def_property_readonly("configured_page_size", [](const BlockSpMat<T>& self) {
            return self.configured_page_size();
        })
        .def_property_readonly("vendor_backend_name", [](const BlockSpMat<T>& self) {
            return self.vendor_backend_name();
        })
        .def_property_readonly("vendor_launch_count", [](const BlockSpMat<T>& self) {
            return self.vendor_launch_count();
        })
        .def("reset_vendor_launch_count", &BlockSpMat<T>::reset_vendor_launch_count)
        .def_property(
            "page_size",
            [](const BlockSpMat<T>& self) {
                return self.page_size();
            },
            [](BlockSpMat<T>& self, uint32_t page_size) {
                self.set_page_size(page_size);
            })
        .def("set_page_size", &BlockSpMat<T>::set_page_size, py::arg("page_size"))
        .def_property_readonly("shape_class_count", [](const BlockSpMat<T>& self) {
            return self.shape_class_count();
        })
        .def_property_readonly("has_contiguous_layout", [](const BlockSpMat<T>& self) {
            return self.has_contiguous_layout();
        })
        .def("pack_contiguous", &BlockSpMat<T>::pack_contiguous)
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
            return make_owned_array_2d_row_major(std::move(vec), my_rows, my_cols);
        })
        .def("from_dense", [](BlockSpMat<T>& self, py::array_t<T> array) {
            auto contiguous = as_row_major_2d<T>(array, "from_dense");
            py::buffer_info info = contiguous.request();

            const size_t owned_blocks = self.graph->owned_global_indices.size();
            const py::ssize_t expected_rows =
                self.graph->block_offsets[owned_blocks];
            const py::ssize_t expected_cols =
                self.graph->block_offsets.empty() ? 0 : self.graph->block_offsets.back();
            if (info.shape[0] != expected_rows || info.shape[1] != expected_cols) {
                throw std::runtime_error(
                    "from_dense: expected shape (" +
                    std::to_string(expected_rows) + ", " +
                    std::to_string(expected_cols) + ")");
            }

            std::vector<T> vec(static_cast<size_t>(contiguous.size()));
            std::memcpy(vec.data(), info.ptr, sizeof(T) * vec.size());
            self.from_dense(vec);
        }, py::arg("array"))
        .def("spmf", &py_graph_matrix_function<T>, py::arg("func_name"), py::arg("verbose") = false)
        .def_property_readonly("row_ptr", [](const BlockSpMat<T>& self) {
            return make_owned_array_1d(self.row_ptr());
        })
        .def_property_readonly("col_ind", [](const BlockSpMat<T>& self) {
            return make_owned_array_1d(self.col_ind());
        })
        ;
}

// The dense LCAO eigensolver (ELPA + ScaLAPACK + 2D block-cyclic) was relocated
// out of VBCSR into rescu++ _core (rescupp._core.lcao_eig); see
// doc/design/31-dense-linalg-relocation.md. VBCSR is a sparse library and no
// longer links ELPA/ScaLAPACK or hosts the solver.

PYBIND11_MODULE(vbcsr_core, m) {
    m.doc() = "VBCSR C++ Core Bindings";

    py::enum_<AssemblyMode>(m, "AssemblyMode")
        .value("INSERT", AssemblyMode::INSERT)
        .value("ADD", AssemblyMode::ADD)
        .export_values();

    py::enum_<RedistOp>(m, "RedistOp")
        .value("Copy", RedistOp::Copy)
        .value("Sum", RedistOp::Sum)
        .export_values();

    m.def("finalize_mpi", &finalize_mpi, "Finalize MPI if initialized");

    py::class_<DistGraph>(m, "DistGraph")
        .def(py::init([](py::object comm_obj) {
            return new DistGraph(get_mpi_comm(comm_obj));
        }), py::arg("comm") = py::none())
        .def("construct_serial", &DistGraph::construct_serial)
        .def("construct_distributed", &DistGraph::construct_distributed)
        .def("construct_distributed_flat",
            // Array-based construction: identical semantics to
            // construct_distributed, but adjacency arrives as flat CSR-style
            // numpy arrays (adj_ind holds GLOBAL block ids), avoiding the
            // per-element list conversion that dominates large graphs.
            [](DistGraph& self,
               py::array_t<int, py::array::c_style | py::array::forcecast> owned_indices,
               py::array_t<int, py::array::c_style | py::array::forcecast> block_sizes,
               py::array_t<int64_t, py::array::c_style | py::array::forcecast> adj_ptr,
               py::array_t<int, py::array::c_style | py::array::forcecast> adj_ind) {
                if (owned_indices.ndim() != 1 || block_sizes.ndim() != 1 ||
                    adj_ptr.ndim() != 1 || adj_ind.ndim() != 1) {
                    throw std::runtime_error("construct_distributed_flat expects 1-D arrays");
                }
                const py::ssize_t n_owned = owned_indices.shape(0);
                if (block_sizes.shape(0) != n_owned || adj_ptr.shape(0) != n_owned + 1) {
                    throw std::runtime_error(
                        "construct_distributed_flat: block_sizes must match owned_indices and "
                        "adj_ptr must have one more entry than owned_indices");
                }
                std::vector<int> owned_vec(
                    owned_indices.data(), owned_indices.data() + n_owned);
                std::vector<int> sizes_vec(
                    block_sizes.data(), block_sizes.data() + n_owned);
                self.construct_distributed_flat(
                    owned_vec, sizes_vec, adj_ptr.data(), adj_ind.data(),
                    static_cast<int64_t>(adj_ind.shape(0)));
            },
            py::arg("owned_indices"), py::arg("block_sizes"),
            py::arg("adj_ptr"), py::arg("adj_ind"))
        .def_readonly("owned_global_indices", &DistGraph::owned_global_indices)
        .def_readonly("ghost_global_indices", &DistGraph::ghost_global_indices)
        .def_readonly("block_sizes", &DistGraph::block_sizes)
        .def_readonly("adj_ptr", &DistGraph::adj_ptr)
        .def_property_readonly("adj_ind", [](const DistGraph& self) {
            return make_owned_array_1d(self.adj_ind);
        })
        .def_readonly("send_counts", &DistGraph::send_counts)
        .def_readonly("recv_counts", &DistGraph::recv_counts)
        .def_readonly("send_counts_scalar", &DistGraph::send_counts_scalar)
        .def_readonly("recv_counts_scalar", &DistGraph::recv_counts_scalar)
        .def_readonly("send_ranks", &DistGraph::send_ranks)
        .def_readonly("recv_ranks", &DistGraph::recv_ranks)
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
        .def("get_global_index", &DistGraph::get_global_index)
        .def_readonly("ghost_global_indices", &DistGraph::ghost_global_indices)
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

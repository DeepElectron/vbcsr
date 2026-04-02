#include "pybind_common.hpp"
#include "core/atomic/atomic_data.hpp"
#include "core/atomic/image_container.hpp"
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

namespace py = pybind11;
using namespace vbcsr;
using namespace vbcsr::atomic;

// Helper to convert numpy array to std::vector
template<typename T>
std::vector<T> numpy_to_vector(py::array_t<T> array) {
    py::buffer_info info = array.request();
    std::vector<T> vec(info.size);
    if (array.size() > 0) {
        py::array_t<T, py::array::c_style | py::array::forcecast> contiguous = array;
        std::memcpy(vec.data(), contiguous.data(), contiguous.size() * sizeof(T));
    }
    return vec;
}

template<typename T>
py::array_t<T> copy_1d_array(const T* data, py::ssize_t size) {
    py::array_t<T> result(py::array::ShapeContainer{size});
    if (size > 0 && data != nullptr) {
        auto out = result.template mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < size; ++i) {
            out(i) = data[i];
        }
    }
    return result;
}

template<typename T>
py::array_t<T> copy_2d_array(py::ssize_t rows, py::ssize_t cols, const T* data) {
    py::array_t<T> result({rows, cols});
    if (rows > 0 && cols > 0 && data != nullptr) {
        auto out = result.template mutable_unchecked<2>();
        for (py::ssize_t row = 0; row < rows; ++row) {
            for (py::ssize_t col = 0; col < cols; ++col) {
                out(row, col) = data[row * cols + col];
            }
        }
    }
    return result;
}

inline std::vector<bool> parse_pbc(py::object pbc_obj) {
    std::vector<bool> vec_pbc(3, false);
    if (pbc_obj.is_none()) {
        return vec_pbc;
    }

    if (py::isinstance<py::sequence>(pbc_obj)) {
        auto seq = pbc_obj.cast<py::sequence>();
        if (seq.size() == 3) {
            for (int i = 0; i < 3; ++i) vec_pbc[i] = seq[i].cast<bool>();
        } else if (seq.size() == 1) {
            bool val = seq[0].cast<bool>();
            std::fill(vec_pbc.begin(), vec_pbc.end(), val);
        } else {
            throw std::runtime_error("pbc must be a bool or a sequence of length 1 or 3");
        }
    } else if (py::isinstance<py::bool_>(pbc_obj)) {
        bool val = pbc_obj.cast<bool>();
        std::fill(vec_pbc.begin(), vec_pbc.end(), val);
    } else {
        throw std::runtime_error("pbc must be a bool or a sequence of length 1 or 3");
    }

    return vec_pbc;
}

template<typename T>
void bind_image_container(py::module& m, const std::string& name) {
    py::class_<ImageContainer<T>>(m, name.c_str())
        .def(py::init<AtomicData*>())
        .def("assemble", &ImageContainer<T>::assemble)
        .def("add_block", [](ImageContainer<T>& self, std::vector<int> R, int g_row, int g_col, py::array_t<T> data, AssemblyMode mode) {
            
            if (R.size() != 3) throw std::runtime_error("R must be a list/tuple of 3 integers");

            py::buffer_info info = data.request();
            if (info.ndim != 2) throw std::runtime_error("Data must be 2D");
            
            int rows = info.shape[0];
            int cols = info.shape[1];
            
            MatrixLayout layout = MatrixLayout::RowMajor;
            if (info.strides[0] == sizeof(T) && info.strides[1] == sizeof(T) * rows) {
                layout = MatrixLayout::ColMajor;
            }
            
            self.add_block(R, g_row, g_col, static_cast<T*>(info.ptr), rows, cols, mode, layout);
            
        }, py::arg("R"), py::arg("g_row"), py::arg("g_col"), py::arg("data"), py::arg("mode") = AssemblyMode::ADD)
        
        .def("add_blocks", [](ImageContainer<T>& self, 
                               std::vector<std::vector<int>> R_list,
                               std::vector<int> g_rows,
                               std::vector<int> g_cols,
                               py::list data_list,
                               AssemblyMode mode) {
            
            size_t n = R_list.size();
            if (g_rows.size() != n || g_cols.size() != n || (size_t)py::len(data_list) != n) {
                throw std::runtime_error("add_blocks: all input lists must have the same length");
            }
            
            // Pre-extract pointers, rows, cols, and layouts from numpy arrays
            std::vector<T*> ptrs(n);
            std::vector<int> rows(n), cols(n);
            std::vector<MatrixLayout> layouts(n);
            
            for (size_t i = 0; i < n; ++i) {
                py::array_t<T> arr = data_list[i].cast<py::array_t<T>>();
                py::buffer_info info = arr.request();
                if (info.ndim != 2) throw std::runtime_error("Each data element must be 2D");
                
                ptrs[i] = static_cast<T*>(info.ptr);
                rows[i] = info.shape[0];
                cols[i] = info.shape[1];
                
                layouts[i] = MatrixLayout::RowMajor;
                if (info.strides[0] == sizeof(T) && info.strides[1] == (long)(sizeof(T) * rows[i])) {
                    layouts[i] = MatrixLayout::ColMajor;
                }
            }
            
            // Call in parallel — but we can't pass per-element layouts to the C++ batch
            // function which takes a single layout. So we call add_block individually
            // inside our own parallel loop to respect per-array layouts.
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) {
                self.add_block(R_list[i], g_rows[i], g_cols[i], ptrs[i], rows[i], cols[i], mode, layouts[i]);
            }
            
        }, py::arg("R"), py::arg("g_row"), py::arg("g_col"), py::arg("data"), py::arg("mode") = AssemblyMode::ADD)
        
        .def("sample_k", [](ImageContainer<T>& self, py::array_t<double> k_points, PhaseConvention convention) {
             auto r_k = k_points.unchecked<1>();
             if (k_points.ndim() == 1 && k_points.shape(0) == 3) {
                 std::vector<double> k = {r_k(0), r_k(1), r_k(2)};
                 return self.sample_k(k, convention);
             }
             throw std::runtime_error("Only single k-point (3,) supported");
        }, py::arg("k_point"), py::arg("convention") = PhaseConvention::R_ONLY);
}

void bind_atomic_module(py::module& m) {
    auto owned_positions = [](const AtomicData& self) {
        py::array_t<double> positions({static_cast<py::ssize_t>(self.n_atom), static_cast<py::ssize_t>(3)});
        auto out = positions.mutable_unchecked<2>();
        for (int i = 0; i < self.n_atom; ++i) {
            out(i, 0) = self.x[i];
            out(i, 1) = self.y[i];
            out(i, 2) = self.z[i];
        }
        return positions;
    };

    auto owned_atomic_numbers = [](const AtomicData& self, const char* context) {
        self.ensure_owned_atomic_numbers(context);
        return copy_1d_array(self.atomic_numbers.data(), self.n_atom);
    };
    
    py::class_<AtomicData>(m, "AtomicData")
        .def(py::init([](py::object comm_obj) {
            return new AtomicData(get_mpi_comm(comm_obj));
        }), py::arg("comm") = py::none())
        
        // Static Constructor - simplified to take vectors
        // Expects: r_max and type_norb as vectors aligned covering necessary types
        .def_static("from_points", [](py::array_t<double> pos, 
                                      py::array_t<int> z, 
                                      py::array_t<double> cell, 
                                      py::object pbc_obj, 
                                      py::object r_max_obj, 
                                      py::object type_norb_obj,
                                      py::object comm_obj) {
            
            MPI_Comm comm = get_mpi_comm(comm_obj);
            
            auto r_pos = pos.unchecked<2>();
            if (r_pos.ndim() != 2 || r_pos.shape(1) != 3) throw std::runtime_error("pos must be (N, 3)");
            
            std::vector<double> vec_pos = numpy_to_vector(pos);
            std::vector<int> vec_z = numpy_to_vector(z);
            std::vector<double> vec_cell_flat = numpy_to_vector(cell);
            std::vector<bool> vec_pbc = parse_pbc(pbc_obj);
            
            // Assume r_max and type_norb are convertible to vectors
            // Python wrapper will handle logic to build them correctly
            std::vector<double> r_max_vec;
            std::vector<int> type_norb_vec;
            
            if (py::isinstance<py::list>(r_max_obj)) {
                 r_max_vec = numpy_to_vector<double>(py::array(r_max_obj));
            } else if (py::isinstance<py::array>(r_max_obj)) {
                 r_max_vec = numpy_to_vector<double>(r_max_obj);
            } else {
                 throw std::runtime_error("from_points (C++): r_max must be list or array of floats");
            }
            
            if (py::isinstance<py::list>(type_norb_obj)) {
                 type_norb_vec = numpy_to_vector<int>(py::array(type_norb_obj));
            } else if (py::isinstance<py::array>(type_norb_obj)) {
                 type_norb_vec = numpy_to_vector<int>(type_norb_obj);
            } else {
                 throw std::runtime_error("from_points (C++): type_norb must be list or array of ints");
            }
            
            return AtomicData::from_points(vec_pos, vec_z, vec_cell_flat, vec_pbc, r_max_vec, type_norb_vec, comm);

        }, py::arg("pos"), py::arg("z"), py::arg("cell"), py::arg("pbc"), 
           py::arg("r_max"), py::arg("type_norb"), py::arg("comm") = py::none())
        
        // Static Constructor - from pre-computed distributed graph data
        // Mirrors the C++ constructor: AtomicData(n_atom, N_atom, atom_offset, n_edge, N_edge,
        //   atom_index, atom_type, edge_index, type_norb, edge_shift, cell, pos, comm)
        .def_static("from_distributed", [](
                int n_atom, int N_atom, int atom_offset, int n_edge, int N_edge,
                py::array_t<int> atom_index,
                py::array_t<int> atom_type,
                py::array_t<int> edge_index,
                py::array_t<int> type_norb,
                py::array_t<int> edge_shift,
                py::array_t<double> cell,
                py::array_t<double> pos,
                py::object atomic_numbers_obj,
                py::object comm_obj) {
            
            MPI_Comm comm = get_mpi_comm(comm_obj);
            
            // Validate shapes
            if (atom_index.size() != n_atom)
                throw std::runtime_error("atom_index size must equal n_atom");
            if (atom_type.size() != n_atom)
                throw std::runtime_error("atom_type size must equal n_atom");
            if (edge_index.size() != n_edge * 2)
                throw std::runtime_error("edge_index must have shape (n_edge, 2)");
            if (edge_shift.size() != n_edge * 3)
                throw std::runtime_error("edge_shift must have shape (n_edge, 3)");
            if (cell.size() != 9)
                throw std::runtime_error("cell must have 9 elements (3x3)");
            if (pos.size() != n_atom * 3)
                throw std::runtime_error("pos must have shape (n_atom, 3)");
            
            // Ensure contiguous C-order arrays
            auto c_atom_index = py::array_t<int, py::array::c_style | py::array::forcecast>(atom_index);
            auto c_atom_type = py::array_t<int, py::array::c_style | py::array::forcecast>(atom_type);
            auto c_edge_index = py::array_t<int, py::array::c_style | py::array::forcecast>(edge_index);
            auto c_type_norb = py::array_t<int, py::array::c_style | py::array::forcecast>(type_norb);
            auto c_edge_shift = py::array_t<int, py::array::c_style | py::array::forcecast>(edge_shift);
            auto c_cell = py::array_t<double, py::array::c_style | py::array::forcecast>(cell);
            auto c_pos = py::array_t<double, py::array::c_style | py::array::forcecast>(pos);

            const int* atomic_numbers_ptr = nullptr;
            py::array_t<int, py::array::c_style | py::array::forcecast> c_atomic_numbers;
            if (!atomic_numbers_obj.is_none()) {
                c_atomic_numbers = py::array_t<int, py::array::c_style | py::array::forcecast>(
                    atomic_numbers_obj.cast<py::array_t<int>>());
                if (c_atomic_numbers.size() != n_atom) {
                    throw std::runtime_error("atomic_numbers size must equal n_atom");
                }
                atomic_numbers_ptr = c_atomic_numbers.data();
            }
            
            return new AtomicData(
                (size_t)n_atom, (size_t)N_atom, (size_t)atom_offset, (size_t)n_edge, (size_t)N_edge,
                c_atom_index.data(), c_atom_type.data(), c_edge_index.data(),
                c_type_norb.data(), c_edge_shift.data(),
                c_cell.data(), c_pos.data(), atomic_numbers_ptr,
                comm
            );
        }, py::arg("n_atom"), py::arg("N_atom"), py::arg("atom_offset"),
           py::arg("n_edge"), py::arg("N_edge"),
           py::arg("atom_index"), py::arg("atom_type"), py::arg("edge_index"),
           py::arg("type_norb"), py::arg("edge_shift"),
           py::arg("cell"), py::arg("pos"), py::arg("atomic_numbers") = py::none(),
           py::arg("comm") = py::none())
        
        // Properties
        .def_property_readonly("pos", owned_positions)
        .def_property_readonly("positions", owned_positions)
        .def_property_readonly("atom_indices", [](const AtomicData& self) { 
             return copy_1d_array(self.atom_index.data(), self.n_atom);
        })
        .def_property_readonly("indices", [](const AtomicData& self) { 
             return copy_1d_array(self.atom_index.data(), self.n_atom);
        })
        .def_property_readonly("atom_types", [](const AtomicData& self) { 
             return copy_1d_array(self.atom_type.data(), self.n_atom);
        })
        .def_property_readonly("z", [owned_atomic_numbers](const AtomicData& self) {
             return owned_atomic_numbers(self, "AtomicData.z");
        })
        .def_property_readonly("atomic_numbers", [owned_atomic_numbers](const AtomicData& self) {
             return owned_atomic_numbers(self, "AtomicData.atomic_numbers");
        })
        .def_property_readonly("cell", [](const AtomicData& self) {
             const double zero_cell[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
             const double* cell_ptr = self.cell.size() == 9 ? self.cell.data() : zero_cell;
             return copy_2d_array<double>(3, 3, cell_ptr);
        })
        .def("norb", &AtomicData::norb)
        .def_property_readonly("pbc", [](const AtomicData& self) {
             return std::vector<bool>(self.pbc.begin(), self.pbc.end());
        })
        
        .def_property_readonly("edge_index", [](const AtomicData& self) {
             py::array_t<int> edge_index({static_cast<py::ssize_t>(self.n_edge), static_cast<py::ssize_t>(2)});
             auto ptr = edge_index.mutable_unchecked<2>();

             for(int i = 0; i < self.n_edge; ++i) {
                 ptr(i, 0) = self.edges[i].src;
                 ptr(i, 1) = self.edges[i].dst;
             }
             return edge_index;
        })
        .def_property_readonly("edge_shift", [](const AtomicData& self) {
             py::array_t<int> shifts({static_cast<py::ssize_t>(self.n_edge), static_cast<py::ssize_t>(3)});
             auto ptr = shifts.mutable_unchecked<2>();
             
             for(int i = 0; i < self.n_edge; ++i) {
                  ptr(i, 0) = self.edges[i].rx;
                  ptr(i, 1) = self.edges[i].ry;
                  ptr(i, 2) = self.edges[i].rz;
             }
             return shifts;
        })
        .def_readonly("graph", &AtomicData::graph, py::return_value_policy::reference)
        
        // Keep to_ase for convenience of C++ side data export
        .def("to_ase", [owned_positions](const AtomicData& self) {
            self.ensure_owned_atomic_numbers("AtomicData.to_ase");
            py::object ase = py::module::import("ase");
            py::object Atoms = ase.attr("Atoms");
            const double zero_cell[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            const double* cell_ptr = self.cell.size() == 9 ? self.cell.data() : zero_cell;

            py::array_t<double> pos_arr = owned_positions(self);
            py::array_t<int> numbers = copy_1d_array(self.atomic_numbers.data(), self.n_atom);
            py::array_t<double> cell = copy_2d_array<double>(3, 3, cell_ptr);
            
            return Atoms(py::arg("numbers")=numbers, 
                         py::arg("positions")=pos_arr, 
                         py::arg("cell")=cell, 
                         py::arg("pbc")=std::vector<bool>(self.pbc.begin(), self.pbc.end())); 
        })
        ;
        
    py::enum_<PhaseConvention>(m, "PhaseConvention")
        .value("R_ONLY", PhaseConvention::R_ONLY)
        .value("R_AND_POSITION", PhaseConvention::R_AND_POSITION)
        .export_values();

    bind_image_container<double>(m, "ImageContainer");
    bind_image_container<std::complex<double>>(m, "ImageContainer_Complex");
}

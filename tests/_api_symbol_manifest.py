from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SurfaceSymbol:
    canonical: str
    aliases: tuple[str, ...]
    native_owner: str
    pybind_checks: tuple[str, ...]
    python_checks: tuple[str, ...]
    docs_checks: tuple[str, ...]
    tests: tuple[str, ...]


@dataclass(frozen=True)
class SurfaceClass:
    package_export: str
    wrapper_file: str | None
    wrapper_class: str | None
    pybind_file: str
    native_owner: str
    docs_checks: tuple[str, ...]
    symbols: tuple[SurfaceSymbol, ...]


SURFACE_MANIFEST: dict[str, SurfaceClass] = {
    "VBCSR": SurfaceClass(
        package_export="VBCSR",
        wrapper_file="vbcsr/matrix.py",
        wrapper_class="VBCSR",
        pybind_file="vbcsr/pybind_vbcsr.cpp",
        native_owner="vbcsr/core/block_csr.hpp::BlockSpMat",
        docs_checks=("## VBCSR (Matrix)",),
        symbols=(
            SurfaceSymbol("ndim", (), "BlockSpMat::shape facade", (), ("ndim",), ("- **`ndim`**",), ("tests/test_api_serial.py", "tests/test_api_mpi.py")),
            SurfaceSymbol("nnz", (), "BlockSpMat::local_scalar_nnz/global_nnz", ('.def_property_readonly("local_nnz"', '.def_property_readonly("global_nnz"'), ("nnz",), ("- **`nnz`**",), ("tests/test_api_serial.py", "tests/test_wrapper_contracts.py")),
            SurfaceSymbol("matrix_kind", (), "BlockSpMat::matrix_kind_string", ('.def_property_readonly("matrix_kind"',), ("matrix_kind",), ("- **`matrix_kind`**",), ("tests/test_matrix_kind.py", "tests/test_wrapper_contracts.py")),
            SurfaceSymbol("transpose", ("T",), "BlockSpMat::transpose", ('.def("transpose"',), ("transpose", "T"), ("#### `transpose`", "- **`T`**"), ("tests/test_api_serial.py", "tests/test_wrapper_contracts.py", "vbcsr/core/test/test_migration_contract.cpp")),
            SurfaceSymbol("transpose_", (), "BlockSpMat::transpose", ('.def("transpose"',), ("transpose_",), ("#### `transpose_`",), ("tests/test_api_serial.py", "tests/test_wrapper_contracts.py")),
            SurfaceSymbol("conjugate", ("conj",), "BlockSpMat::conjugate", ('.def("conjugate"',), ("conjugate", "conj"), ("#### `conj` / `conjugate`",), ("tests/test_api_serial.py", "tests/test_wrapper_contracts.py", "vbcsr/core/test/test_migration_contract.cpp")),
            SurfaceSymbol("conj_", (), "BlockSpMat::conjugate", ('.def("conjugate"',), ("conj_",), ("#### `conj_`",), ("tests/test_api_serial.py", "tests/test_wrapper_contracts.py")),
            SurfaceSymbol("real", (), "BlockSpMat::get_real", ('.def("real"',), ("real",), ("- **`real`**",), ("tests/test_wrapper_contracts.py", "vbcsr/core/test/test_migration_contract.cpp")),
            SurfaceSymbol("imag", (), "BlockSpMat::get_imag", ('.def("imag"',), ("imag",), ("- **`imag`**",), ("tests/test_wrapper_contracts.py", "vbcsr/core/test/test_migration_contract.cpp")),
            SurfaceSymbol("__getitem__", (), "VBCSR wrapper contract", (), ("__getitem__",), ("#### `__getitem__`", "Scalar and slicing indexing are currently unsupported."), ("tests/test_wrapper_contracts.py",)),
            SurfaceSymbol("create_serial", (), "DistGraph::construct_serial + BlockSpMat ctor", (), ("create_serial",), ("#### `create_serial`",), ("tests/test_api_serial.py", "tests/test_binding.py")),
            SurfaceSymbol("create_distributed", (), "DistGraph::construct_distributed + BlockSpMat ctor", (), ("create_distributed",), ("#### `create_distributed`",), ("tests/test_api_mpi.py", "tests/test_binding.py")),
            SurfaceSymbol("create_random", (), "VBCSR wrapper helper", (), ("create_random",), ("#### `create_random`",), ("tests/test_api_serial.py", "tests/test_api_mpi.py")),
            SurfaceSymbol("from_scipy", (), "VBCSR wrapper helper", (), ("from_scipy",), ("#### `from_scipy`", "root: int = 0"), ("tests/test_scipy_adapter.py", "tests/test_from_scipy_collective.py")),
            SurfaceSymbol("create_vector", (), "DistVector ctor", (), ("create_vector",), ("#### `create_vector`",), ("tests/test_api_serial.py", "tests/test_api_mpi.py")),
            SurfaceSymbol("create_multivector", (), "DistMultiVector ctor", (), ("create_multivector",), ("#### `create_multivector`",), ("tests/test_api_serial.py", "tests/test_api_mpi.py")),
            SurfaceSymbol("add_block", (), "BlockSpMat::add_block", ('.def("add_block"',), ("add_block",), ("#### `add_block`",), ("tests/test_binding.py", "tests/test_wrapper_contracts.py")),
            SurfaceSymbol("get_block", (), "BlockSpMat::get_block", ('.def("get_block"',), ("get_block",), ("#### `get_block`",), ("tests/test_wrapper_contracts.py",)),
            SurfaceSymbol("assemble", (), "BlockSpMat::assemble", ('.def("assemble"',), ("assemble",), ("#### `assemble`",), ("tests/test_binding.py", "tests/test_wrapper_contracts.py")),
            SurfaceSymbol("mult", ("dot", "__matmul__"), "BlockSpMat::mult / spmm", ('.def("mult"', '.def("spmm"'), ("mult", "dot", "__matmul__"), ("#### `mult`", "#### `dot` / `@`"), ("tests/test_binding.py", "tests/test_wrapper_contracts.py", "tests/test_spmm.py")),
            SurfaceSymbol("duplicate", ("copy",), "BlockSpMat::duplicate", ('.def("duplicate"',), ("duplicate", "copy"), ("#### `copy` / `duplicate`",), ("tests/test_api_serial.py", "tests/test_wrapper_contracts.py", "vbcsr/core/test/test_migration_contract.cpp")),
            SurfaceSymbol("scale", (), "BlockSpMat::scale", ('.def("scale"',), ("scale",), ("#### `scale`",), ("tests/test_binding.py", "tests/test_wrapper_contracts.py")),
            SurfaceSymbol("shift", (), "BlockSpMat::shift", ('.def("shift"',), ("shift",), ("#### `shift`",), ("tests/test_wrapper_contracts.py", "vbcsr/core/test/test_migration_contract.cpp")),
            SurfaceSymbol("add_diagonal", (), "BlockSpMat::add_diagonal", ('.def("add_diagonal"',), ("add_diagonal",), ("#### `add_diagonal`",), ("tests/test_wrapper_contracts.py", "vbcsr/core/test/test_migration_contract.cpp")),
            SurfaceSymbol("spmm", ("spmm_self",), "BlockSpMat::spmm", ('.def("spmm"', '.def("spmm_self"'), ("spmm", "spmm_self"), ("#### `spmm`", "#### `spmm_self`"), ("tests/test_spmm.py", "vbcsr/core/test/test_migration_contract.cpp")),
            SurfaceSymbol("add", (), "BlockSpMat::add", ('.def("add"',), ("add",), ("#### `add`",), ("tests/test_spmm.py", "vbcsr/core/test/test_migration_contract.cpp")),
            SurfaceSymbol("filter_blocks", (), "BlockSpMat::filter_blocks", ('.def("filter_blocks"',), ("filter_blocks",), ("#### `filter_blocks`",), ("tests/test_wrapper_contracts.py",)),
            SurfaceSymbol("get_block_density", (), "BlockSpMat::get_block_density", ('.def("get_block_density"',), ("get_block_density",), ("#### `get_block_density`",), ("tests/test_wrapper_contracts.py",)),
            SurfaceSymbol("extract_submatrix", (), "BlockSpMat::extract_submatrix", ('.def("extract_submatrix"',), ("extract_submatrix",), ("#### `extract_submatrix`",), ("tests/test_wrapper_contracts.py", "vbcsr/core/test/test_migration_contract.cpp")),
            SurfaceSymbol("insert_submatrix", (), "BlockSpMat::insert_submatrix", ('.def("insert_submatrix"',), ("insert_submatrix",), ("#### `insert_submatrix`",), ("tests/test_wrapper_contracts.py", "vbcsr/core/test/test_migration_contract.cpp")),
            SurfaceSymbol("spmf", (), "graph_matrix_function", ('.def("spmf"',), ("spmf",), ("#### `spmf`",), ("tests/test_spmf.py", "tests/test_kpm.py")),
            SurfaceSymbol("to_dense", (), "BlockSpMat::to_dense", ('.def("to_dense"',), ("to_dense",), ("#### `to_dense`",), ("tests/test_wrapper_contracts.py", "vbcsr/core/test/test_migration_contract.cpp")),
            SurfaceSymbol("from_dense", (), "BlockSpMat::from_dense", ('.def("from_dense"',), ("from_dense",), ("#### `from_dense`",), ("tests/test_wrapper_contracts.py", "vbcsr/core/test/test_migration_contract.cpp")),
            SurfaceSymbol("row_ptr", (), "BlockSpMat::logical_row_ptr", ('.def_property_readonly("row_ptr"',), ("row_ptr",), ("#### `row_ptr`",), ("tests/test_wrapper_contracts.py",)),
            SurfaceSymbol("col_ind", (), "BlockSpMat::logical_col_ind", ('.def_property_readonly("col_ind"',), ("col_ind",), ("#### `col_ind`",), ("tests/test_wrapper_contracts.py",)),
            SurfaceSymbol("get_values", (), "BlockSpMat::get_values", ('.def("get_values"',), ("get_values",), ("#### `get_values`",), ("tests/test_wrapper_contracts.py", "vbcsr/core/test/test_migration_contract.cpp")),
            SurfaceSymbol("to_scipy", (), "VBCSR wrapper helper", (), ("to_scipy",), ("#### `to_scipy`",), ("tests/test_scipy_adapter.py",)),
            SurfaceSymbol("save_matrix_market", (), "BlockSpMat::save_matrix_market", ('.def("save_matrix_market"',), ("save_matrix_market",), ("#### `save_matrix_market`", "Distributed matrices raise an error."), ("tests/test_wrapper_contracts.py",)),
        ),
    ),
    "DistVector": SurfaceClass(
        package_export="DistVector",
        wrapper_file="vbcsr/vector.py",
        wrapper_class="DistVector",
        pybind_file="vbcsr/pybind_vbcsr.cpp",
        native_owner="vbcsr/core/dist_vector.hpp::DistVector",
        docs_checks=("## DistVector",),
        symbols=(
            SurfaceSymbol("local_size", (), "DistVector::local_size", ('.def_property_readonly("local_size"',), ("local_size",), ("- **`local_size`**",), ("tests/test_api_serial.py",)),
            SurfaceSymbol("ghost_size", (), "DistVector::ghost_size", ('.def_property_readonly("ghost_size"',), ("ghost_size",), ("- **`ghost_size`**",), ("tests/test_api_serial.py",)),
            SurfaceSymbol("full_size", (), "DistVector::full_size", ('.def_property_readonly("full_size"',), ("full_size",), ("- **`full_size`**",), ("tests/test_api_serial.py",)),
            SurfaceSymbol("shape", (), "DistVector wrapper shape", (), ("shape",), ("- **`shape`**",), ("tests/test_api_serial.py",)),
            SurfaceSymbol("size", (), "DistVector wrapper size", (), ("size",), ("- **`size`**",), ("tests/test_api_serial.py",)),
            SurfaceSymbol("duplicate", ("copy",), "DistVector::duplicate", ('.def("duplicate"',), ("duplicate", "copy"), ("#### `copy` / `duplicate`",), ("tests/test_api_serial.py", "tests/test_wrapper_contracts.py")),
            SurfaceSymbol("to_numpy", (), "DistVector buffer view", (), ("to_numpy",), ("#### `to_numpy`",), ("tests/test_binding.py",)),
            SurfaceSymbol("from_numpy", (), "DistVector buffer view", (), ("from_numpy",), ("#### `from_numpy`",), ("tests/test_binding.py",)),
            SurfaceSymbol("set_constant", (), "DistVector::set_constant", ('.def("set_constant"',), ("set_constant",), ("#### `set_constant`",), ("tests/test_binding.py",)),
            SurfaceSymbol("set_random_normal", (), "DistVector::set_random_normal", ('.def("set_random_normal"',), ("set_random_normal",), ("#### `set_random_normal`",), ("tests/test_vbcsr.py",)),
            SurfaceSymbol("scale", (), "DistVector::scale", ('.def("scale"',), ("scale",), ("#### `scale`",), ("tests/test_binding.py",)),
            SurfaceSymbol("axpy", (), "DistVector::axpy", ('.def("axpy"',), ("axpy",), ("#### `axpy`",), ("tests/test_vbcsr.py",)),
            SurfaceSymbol("axpby", (), "DistVector::axpby", ('.def("axpby"',), ("axpby",), ("#### `axpby`",), ("tests/test_vbcsr.py",)),
            SurfaceSymbol("pointwise_mult", (), "DistVector::pointwise_mult", ('.def("pointwise_mult"',), ("pointwise_mult",), ("#### `pointwise_mult`",), ("tests/test_vbcsr.py",)),
            SurfaceSymbol("dot", ("__matmul__",), "DistVector::dot", ('.def("dot"',), ("dot", "__matmul__"), ("#### `dot`",), ("tests/test_binding.py",)),
            SurfaceSymbol("sync_ghosts", (), "DistVector::sync_ghosts", ('.def("sync_ghosts"',), ("sync_ghosts",), ("#### `sync_ghosts`",), ("tests/test_binding.py",)),
            SurfaceSymbol("reduce_ghosts", (), "DistVector::reduce_ghosts", ('.def("reduce_ghosts"',), ("reduce_ghosts",), ("#### `reduce_ghosts`",), ("tests/test_binding.py",)),
        ),
    ),
    "DistMultiVector": SurfaceClass(
        package_export="DistMultiVector",
        wrapper_file="vbcsr/multivector.py",
        wrapper_class="DistMultiVector",
        pybind_file="vbcsr/pybind_vbcsr.cpp",
        native_owner="vbcsr/core/dist_multivector.hpp::DistMultiVector",
        docs_checks=("## DistMultiVector",),
        symbols=(
            SurfaceSymbol("local_rows", (), "DistMultiVector::local_rows", ('.def_property_readonly("local_rows"',), ("local_rows",), ("- **`local_rows`**",), ("tests/test_api_serial.py",)),
            SurfaceSymbol("ghost_rows", (), "DistMultiVector::ghost_rows", ('.def_property_readonly("ghost_rows"',), ("ghost_rows",), ("- **`ghost_rows`**",), ("tests/test_wrapper_contracts.py",)),
            SurfaceSymbol("num_vectors", (), "DistMultiVector::num_vectors", ('.def_property_readonly("num_vectors"',), ("num_vectors",), ("- **`num_vectors`**",), ("tests/test_api_serial.py",)),
            SurfaceSymbol("shape", (), "DistMultiVector wrapper shape", (), ("shape",), ("- **`shape`**",), ("tests/test_api_serial.py",)),
            SurfaceSymbol("size", (), "DistMultiVector wrapper size", (), ("size",), ("- **`size`**",), ("tests/test_api_serial.py",)),
            SurfaceSymbol("duplicate", ("copy",), "DistMultiVector::duplicate", ('.def("duplicate"',), ("duplicate", "copy"), ("#### `copy` / `duplicate`",), ("tests/test_wrapper_contracts.py", "vbcsr/core/test/test_migration_contract.cpp")),
            SurfaceSymbol("to_numpy", (), "DistMultiVector buffer view", (), ("to_numpy",), ("#### `to_numpy`",), ("tests/test_wrapper_contracts.py",)),
            SurfaceSymbol("from_numpy", (), "DistMultiVector buffer view", (), ("from_numpy",), ("#### `from_numpy`",), ("tests/test_wrapper_contracts.py",)),
            SurfaceSymbol("set_constant", (), "DistMultiVector::set_constant", ('.def("set_constant"',), ("set_constant",), ("#### `set_constant`",), ("tests/test_wrapper_contracts.py",)),
            SurfaceSymbol("set_random_normal", (), "DistMultiVector::set_random_normal", ('.def("set_random_normal"',), ("set_random_normal",), ("#### `set_random_normal`",), ("tests/test_wrapper_contracts.py",)),
            SurfaceSymbol("scale", (), "DistMultiVector::scale", ('.def("scale"',), ("scale",), ("#### `scale`",), ("tests/test_wrapper_contracts.py",)),
            SurfaceSymbol("axpy", (), "DistMultiVector::axpy", ('.def("axpy"',), ("axpy",), ("#### `axpy`",), ("tests/test_wrapper_contracts.py",)),
            SurfaceSymbol("axpby", (), "DistMultiVector::axpby", ('.def("axpby"',), ("axpby",), ("#### `axpby`",), ("tests/test_wrapper_contracts.py",)),
            SurfaceSymbol("pointwise_mult", (), "DistMultiVector::pointwise_mult", ('.def("pointwise_mult"', '.def("pointwise_mult_vec"'), ("pointwise_mult",), ("#### `pointwise_mult`",), ("tests/test_wrapper_contracts.py",)),
            SurfaceSymbol("bdot", (), "DistMultiVector::bdot", ('.def("bdot"',), ("bdot",), ("#### `bdot`",), ("tests/test_wrapper_contracts.py",)),
            SurfaceSymbol("sync_ghosts", (), "DistMultiVector::sync_ghosts", ('.def("sync_ghosts"',), ("sync_ghosts",), ("#### `sync_ghosts`",), ("tests/test_wrapper_contracts.py",)),
            SurfaceSymbol("reduce_ghosts", (), "DistMultiVector::reduce_ghosts", ('.def("reduce_ghosts"',), ("reduce_ghosts",), ("#### `reduce_ghosts`",), ("tests/test_wrapper_contracts.py",)),
        ),
    ),
    "DistGraph": SurfaceClass(
        package_export="DistGraph",
        wrapper_file=None,
        wrapper_class=None,
        pybind_file="vbcsr/pybind_vbcsr.cpp",
        native_owner="vbcsr/core/dist_graph.hpp::DistGraph",
        docs_checks=("## DistGraph",),
        symbols=(
            SurfaceSymbol("construct_serial", (), "DistGraph::construct_serial", ('.def("construct_serial"',), (), ("#### `construct_serial`", "construct_serial(self, global_blocks: int, block_sizes: list[int], adjacency: list[list[int]]) -> None"), ("tests/test_api_serial.py", "tests/test_api_mpi.py", "vbcsr/core/test/test_migration_contract.cpp")),
            SurfaceSymbol("construct_distributed", (), "DistGraph::construct_distributed", ('.def("construct_distributed"',), (), ("#### `construct_distributed`",), ("tests/test_api_serial.py", "tests/test_api_mpi.py")),
            SurfaceSymbol("owned_global_indices", (), "DistGraph::owned_global_indices", ('.def_readonly("owned_global_indices"',), (), ("- **`owned_global_indices`**",), ("tests/test_api_serial.py", "tests/test_api_mpi.py")),
            SurfaceSymbol("ghost_global_indices", (), "DistGraph::ghost_global_indices", ('.def_readonly("ghost_global_indices"',), (), ("- **`ghost_global_indices`**",), ("tests/test_api_serial.py", "tests/test_api_mpi.py")),
            SurfaceSymbol("block_sizes", (), "DistGraph::block_sizes", ('.def_readonly("block_sizes"',), (), ("- **`block_sizes`**",), ("tests/test_api_serial.py", "tests/test_api_mpi.py")),
            SurfaceSymbol("owned_scalar_rows", (), "DistGraph::owned_scalar_rows", ('.def_property_readonly("owned_scalar_rows"',), (), ("- **`owned_scalar_rows`**",), ("tests/test_api_serial.py", "tests/test_api_mpi.py")),
            SurfaceSymbol("local_scalar_cols", (), "DistGraph::local_scalar_cols", ('.def_property_readonly("local_scalar_cols"',), (), ("- **`local_scalar_cols`**",), ("tests/test_api_serial.py", "tests/test_api_mpi.py")),
            SurfaceSymbol("global_scalar_rows", (), "DistGraph::global_scalar_rows", ('.def_property_readonly("global_scalar_rows"',), (), ("- **`global_scalar_rows`**",), ("tests/test_api_serial.py", "tests/test_api_mpi.py")),
            SurfaceSymbol("get_local_index", (), "DistGraph::global_to_local", ('.def("get_local_index"',), (), ("#### `get_local_index`",), ("tests/test_api_serial.py", "tests/test_api_mpi.py")),
            SurfaceSymbol("rank", (), "DistGraph::rank", ('.def_readonly("rank"',), (), ("- **`rank`** / **`size`**",), ("tests/test_api_serial.py", "tests/test_api_mpi.py")),
            SurfaceSymbol("size", (), "DistGraph::size", ('.def_readonly("size"',), (), ("- **`rank`** / **`size`**",), ("tests/test_api_serial.py", "tests/test_api_mpi.py")),
        ),
    ),
    "AtomicData": SurfaceClass(
        package_export="AtomicData",
        wrapper_file="vbcsr/atomic_data.py",
        wrapper_class="AtomicData",
        pybind_file="vbcsr/pybind_atomic.cpp",
        native_owner="vbcsr/core/atomic/atomic_data.hpp::AtomicData",
        docs_checks=("## AtomicData", "`atomic_numbers` / `z` expose true atomic numbers",),
        symbols=(
            SurfaceSymbol("from_points", (), "AtomicData::from_points", ('.def_static("from_points"',), ("from_points",), ("#### `from_points`",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("from_distributed", (), "AtomicData ctor from distributed parts", ('.def_static("from_distributed"',), ("from_distributed",), ("#### `from_distributed`",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("from_ase", (), "AtomicData wrapper helper", (), ("from_ase",), ("#### `from_ase` / `from_file`",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("from_file", (), "AtomicData wrapper helper", (), ("from_file",), ("#### `from_ase` / `from_file`",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("positions", ("pos",), "AtomicData positions", ('.def_property_readonly("pos"', '.def_property_readonly("positions"'), (), ("- **`positions` / `pos`**",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("atom_indices", ("indices",), "AtomicData atom indices", ('.def_property_readonly("atom_indices"', '.def_property_readonly("indices"'), (), ("- **`atom_indices` / `indices`**",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("atom_types", (), "AtomicData type ids", ('.def_property_readonly("atom_types"',), (), ("- **`atom_types`**",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("atomic_numbers", ("z",), "AtomicData atomic numbers", ('.def_property_readonly("z"', '.def_property_readonly("atomic_numbers"'), (), ("- **`atomic_numbers` / `z`**",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("cell", (), "AtomicData::cell", ('.def_property_readonly("cell"',), (), ("- **`cell`**",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("pbc", (), "AtomicData::pbc", ('.def_property_readonly("pbc"',), (), ("- **`pbc`**",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("edge_index", (), "AtomicData::edges", ('.def_property_readonly("edge_index"',), (), ("- **`edge_index`**",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("edge_shift", (), "AtomicData::edges", ('.def_property_readonly("edge_shift"',), (), ("- **`edge_shift`**",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("graph", (), "AtomicData::graph", ('.def_readonly("graph"',), (), ("- **`graph`**",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("norb", (), "AtomicData::norb", ('.def("norb"',), (), ("#### `norb`",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("to_ase", (), "AtomicData::to_ase", ('.def("to_ase"',), (), ("#### `to_ase`",), ("tests/test_atomic_surface.py",)),
        ),
    ),
    "ImageContainer": SurfaceClass(
        package_export="ImageContainer",
        wrapper_file="vbcsr/image_container.py",
        wrapper_class="ImageContainer",
        pybind_file="vbcsr/pybind_atomic.cpp",
        native_owner="vbcsr/core/atomic/image_container.hpp::ImageContainer",
        docs_checks=("## ImageContainer",),
        symbols=(
            SurfaceSymbol("__init__", (), "ImageContainer ctors", ('.def(py::init<AtomicData*>())',), ("__init__",), ("#### `__init__`",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("add_block", (), "ImageContainer::add_block", ('.def("add_block"',), ("add_block",), ("#### `add_block`",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("add_blocks", (), "ImageContainer::add_blocks", ('.def("add_blocks"',), ("add_blocks",), ("#### `add_blocks`",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("assemble", (), "ImageContainer::assemble", ('.def("assemble"',), ("assemble",), ("#### `assemble`",), ("tests/test_atomic_surface.py",)),
            SurfaceSymbol("sample_k", (), "ImageContainer::sample_k", ('.def("sample_k"',), ("sample_k",), ("#### `sample_k`",), ("tests/test_atomic_surface.py",)),
        ),
    ),
}


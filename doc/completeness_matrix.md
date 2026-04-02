# Repo Completeness Matrix

This matrix is the maintained cross-check between shipped surfaces, their implementation layers, docs, and active regression entrypoints.

| Surface | Native source | Python / binding surface | Docs | Active regression |
| --- | --- | --- | --- | --- |
| Matrix core (`BlockSpMat`, `VBCSR`) | `vbcsr/core/block_csr.hpp` | `vbcsr/pybind_vbcsr.cpp`, `vbcsr/matrix.py` | `doc/api_reference.md` | `tests/test_api_serial.py`, `tests/test_api_mpi.py`, `tests/test_wrapper_contracts.py`, `tests/test_binding.py`, `tests/test_spmm.py`, `tests/test_vbcsr.py`, `vbcsr/core/test/run_all_tests.py`, `vbcsr/core/test/run_cmake_registered_tests.py` |
| Distributed vectors | `vbcsr/core/dist_vector.hpp` | `vbcsr/pybind_vbcsr.cpp`, `vbcsr/vector.py` | `doc/api_reference.md` | `tests/test_wrapper_contracts.py`, `tests/test_binding.py`, `vbcsr/core/test/run_all_tests.py` |
| Distributed multivectors | `vbcsr/core/dist_multivector.hpp` | `vbcsr/pybind_vbcsr.cpp`, `vbcsr/multivector.py` | `doc/api_reference.md` | `tests/test_wrapper_contracts.py`, `vbcsr/core/test/run_all_tests.py` |
| Distributed graph | `vbcsr/core/dist_graph.hpp` | `vbcsr/pybind_vbcsr.cpp` | `doc/api_reference.md` | `tests/test_api_serial.py`, `tests/test_api_mpi.py`, `vbcsr/core/test/run_all_tests.py` |
| Atomic topology | `vbcsr/core/atomic/atomic_data.hpp` | `vbcsr/pybind_atomic.cpp`, `vbcsr/atomic_data.py` | `doc/api_reference.md` | `tests/test_atomic_surface.py`, `vbcsr/core/test/run_native_suite.py`, `vbcsr/core/test/run_cmake_registered_tests.py` |
| Image accumulation / sampling | `vbcsr/core/atomic/image_container.hpp` | `vbcsr/pybind_atomic.cpp`, `vbcsr/image_container.py` | `doc/api_reference.md` | `tests/test_atomic_surface.py`, `vbcsr/core/test/run_native_suite.py`, `vbcsr/core/test/run_cmake_registered_tests.py` |
| Python regression orchestration | n/a | `tests/run_python_suite.py`, `tests/_suite_manifest.py` | this file | `tests/test_cleanup_guards.py` |
| Native regression orchestration | native test sources | `vbcsr/core/test/run_all_tests.py`, `vbcsr/core/test/run_cmake_registered_tests.py`, `vbcsr/core/test/run_native_suite.py` | this file | `tests/test_cleanup_guards.py` |

Notes:

- `tests/_suite_manifest.py` is the authoritative Python-suite manifest. Every `tests/test_*.py` file must appear there or in its explicit exclusion list.
- `tests/_api_symbol_manifest.py` is the machine-checked symbol-level audit for the supported Python-facing package surface. `tests/test_cleanup_guards.py` validates it against wrappers, bindings, docs, and maintained tests.
- `vbcsr/core/test/run_native_suite.py` is the maintained top-level native workflow. It combines the direct core runner with the CMake-registered test runner for gtest-backed and atomic/image tests.
- `doc/api_reference.md` documents the public package surface; this file tracks the broader implementation-to-test mapping.
- README and developer-facing test instructions should point only to `tests/run_python_suite.py` and `vbcsr/core/test/run_native_suite.py` as maintained entrypoints.

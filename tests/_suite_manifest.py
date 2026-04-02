SERIAL_TESTS = [
    "test_matrix_kind.py",
    "test_scipy_adapter.py",
    "test_wrapper_contracts.py",
    "test_atomic_surface.py",
    "test_cleanup_guards.py",
    "test_api_serial.py",
    "test_binding.py",
    "test_spmf.py",
    "test_kpm.py",
    "test_spmm.py",
    "test_vbcsr.py",
]

MPI_TESTS = [
    "test_api_mpi.py",
    "test_atomic_surface.py",
    "test_binding.py",
    "test_from_scipy_collective.py",
    "test_spmm.py",
]

BASELINE_SMOKE_TESTS = [
    "test_scipy_adapter.py",
    "test_cleanup_guards.py",
    "test_api_serial.py",
    "test_api_mpi.py",
    "test_atomic_surface.py",
    "test_matrix_kind.py",
]

EXCLUDED_TESTS: list[str] = []


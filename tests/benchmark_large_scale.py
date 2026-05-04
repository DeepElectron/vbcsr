import argparse
import ctypes
import json
import os
import shlex
import subprocess
import sys
import time

import numpy as np
import scipy.sparse
import _workspace_bootstrap
import vbcsr
import vbcsr_core

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


PRESET_PROFILES = {
    "small": {"blocks": 32, "density": 0.15, "num_vecs": 4},
    "medium": {"blocks": 192, "density": 0.04, "num_vecs": 16},
}

VBCSR_SHAPES = np.array([9, 13, 15, 20], dtype=np.int32)
CANONICAL_MODES = ("mult", "mult_dense", "mult_adjoint", "mult_dense_adjoint", "spmm")
DENSE_MODES = ("mult_dense", "mult_dense_adjoint")
ADJOINT_MODES = ("mult_adjoint", "mult_dense_adjoint")


def try_get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def apply_profile_defaults(args: argparse.Namespace) -> None:
    if args.profile == "custom":
        args.blocks = 1000 if args.blocks is None else args.blocks
        args.density = 0.01 if args.density is None else args.density
        args.num_vecs = 32 if args.num_vecs is None else args.num_vecs
    else:
        defaults = PRESET_PROFILES[args.profile]
        args.blocks = defaults["blocks"] if args.blocks is None else args.blocks
        args.density = defaults["density"] if args.density is None else args.density
        args.num_vecs = defaults["num_vecs"] if args.num_vecs is None else args.num_vecs

    if args.family == "csr":
        args.min_block = 1
        args.max_block = 1
    elif args.family == "bsr":
        if args.min_block is None:
            args.min_block = 8
        if args.max_block is None:
            args.max_block = 8
    elif args.family == "vbcsr":
        args.min_block = int(VBCSR_SHAPES.min())
        args.max_block = int(VBCSR_SHAPES.max())
    elif args.family == "random":
        args.min_block = 1 if args.min_block is None else args.min_block
        args.max_block = 16 if args.max_block is None else args.max_block
    else:
        raise ValueError(f"Unknown family: {args.family}")

def generate_block_sizes(global_blocks: int, family: str, block_size_min: int, block_size_max: int, seed: int = 42) -> list[int]:
    rng = np.random.default_rng(seed)
    if family == "csr":
        return [1] * global_blocks
    if family == "bsr":
        assert block_size_min == block_size_max, "For BSR family, min_block and max_block must be equal"
        return [block_size_min] * global_blocks
    if family == "vbcsr":
        return rng.choice(VBCSR_SHAPES, size=global_blocks, replace=True).astype(int).tolist()
    if family == "random":
        return rng.integers(block_size_min, block_size_max + 1, size=global_blocks).astype(int).tolist()
    raise ValueError(f"Unknown family: {family}")


def generate_random_structure(global_blocks: int, block_sizes: list[int], density: float, seed: int = 42) -> tuple[list[int], list[list[int]]]:
    rng = np.random.default_rng(seed)
    adj = []
    for i in range(global_blocks):
        neighbors = {(i - 1) % global_blocks, i, (i + 1) % global_blocks}
        n_random = max(0, int(global_blocks * density) - 2)
        if n_random > 0:
            random_neighbors = rng.choice(global_blocks, size=min(n_random, global_blocks), replace=False)
            neighbors.update(int(idx) for idx in random_neighbors)
        adj.append(sorted(neighbors))
    return block_sizes, adj


def sort_sparse_indices(matrix):
    if hasattr(matrix, "sort_indices"):
        matrix.sort_indices()
    return matrix


def make_mkl_sparse_baseline(scalar_csr, block_sizes: list[int]):
    unique_block_sizes = sorted({int(size) for size in block_sizes})
    info = {
        "mkl_sparse_format": "csr",
        "mkl_blocksize": None,
        "mkl_format_reason": "variable_block_sizes",
    }

    if len(unique_block_sizes) == 1:
        block_size = unique_block_sizes[0]
        if block_size == 1:
            info["mkl_format_reason"] = "scalar_blocks"
            return sort_sparse_indices(scalar_csr), info

        if (
            scalar_csr.shape[0] % block_size == 0 and
            scalar_csr.shape[1] % block_size == 0
        ):
            bsr = scalar_csr.tobsr(blocksize=(block_size, block_size))
            info["mkl_sparse_format"] = "bsr"
            info["mkl_blocksize"] = [block_size, block_size]
            info["mkl_format_reason"] = "uniform_non_scalar_blocks"
            return sort_sparse_indices(bsr), info

        info["mkl_format_reason"] = "uniform_block_size_does_not_divide_shape"

    return sort_sparse_indices(scalar_csr), info


def make_scalar_csr_from_vbcsr(mat):
    """Build an aligned scalar CSR baseline from the already-filled VBCSR matrix."""
    if mat.matrix_kind in ("csr", "bsr"):
        return sort_sparse_indices(mat.to_scipy(format="bsr").tocsr())
    return sort_sparse_indices(mat.to_scipy(format="csr"))


class CachedMKLSparseOp:
    """Reusable MKL sparse handle for fair repeated SpMV/sparse-dense timing."""

    def __init__(self, sparse_matrix, rhs: np.ndarray, mode: str):
        if mode not in ("mult", "mult_dense", "mult_adjoint", "mult_dense_adjoint"):
            raise ValueError(f"Cached MKL baseline does not support mode {mode!r}")

        from sparse_dot_mkl._mkl_interface import (
            LAYOUT_CODE_C,
            MKL,
            SPARSE_OPERATION_CONJUGATE_TRANSPOSE,
            SPARSE_DIAG_NON_UNIT,
            SPARSE_FILL_MODE_FULL,
            SPARSE_MATRIX_TYPE_GENERAL,
            SPARSE_OPERATION_NON_TRANSPOSE,
            SPARSE_OPERATION_TRANSPOSE,
            _check_return_value,
            _create_mkl_sparse,
            _destroy_mkl_handle,
            _get_numpy_layout,
            _mkl_scalar,
            _output_dtypes,
            matrix_descr,
            sparse_matrix_t,
        )

        self._check_return_value = _check_return_value
        self._destroy_mkl_handle = _destroy_mkl_handle
        self._get_numpy_layout = _get_numpy_layout
        self._sparse_matrix_t = sparse_matrix_t
        self._MKL = MKL
        self._handle = None
        self.mode = mode
        self.info = {
            "mkl_runner": "cached_sparse_handle",
            "mkl_handle_reused": True,
            "mkl_call_binding": "raw_ctypes_pointers",
        }

        self._handle, self._double_precision, self._complex_type = _create_mkl_sparse(sparse_matrix)
        self._descr = matrix_descr(
            SPARSE_MATRIX_TYPE_GENERAL,
            SPARSE_FILL_MODE_FULL,
            SPARSE_DIAG_NON_UNIT,
        )
        self._adjoint = mode in ADJOINT_MODES
        if self._adjoint:
            self._operation = (
                SPARSE_OPERATION_CONJUGATE_TRANSPOSE
                if self._complex_type
                else SPARSE_OPERATION_TRANSPOSE
            )
            output_rows = sparse_matrix.shape[1]
        else:
            self._operation = SPARSE_OPERATION_NON_TRANSPOSE
            output_rows = sparse_matrix.shape[0]
        self._alpha = _mkl_scalar(1.0, self._complex_type, self._double_precision)
        self._beta = _mkl_scalar(0.0, self._complex_type, self._double_precision)
        self._scalar_ctype = (
            type(self._alpha)
            if self._complex_type
            else (ctypes.c_double if self._double_precision else ctypes.c_float)
        )
        self.output = None

        if mode in ("mult", "mult_adjoint"):
            self._rhs = np.asarray(rhs).ravel()
            self.output = np.empty(
                output_rows,
                dtype=_output_dtypes[(self._double_precision, self._complex_type)],
            )
            self._func = self._mv_function()
            self._bind_mv_signature()
            self._rhs_ptr = ctypes.c_void_p(self._rhs.ctypes.data)
            self._out_ptr = ctypes.c_void_p(self.output.ctypes.data)
            self._apply_hints(kind="mv", columns=1)
        else:
            rhs_arr = np.asarray(rhs)
            if not (rhs_arr.flags.c_contiguous or rhs_arr.flags.f_contiguous):
                rhs_arr = np.asfortranarray(rhs_arr)
            self._rhs = rhs_arr
            self._layout, self._rhs_ld = self._get_numpy_layout(self._rhs)
            order = "C" if self._layout == LAYOUT_CODE_C else "F"
            self.output = np.empty(
                (output_rows, self._rhs.shape[1]),
                dtype=_output_dtypes[(self._double_precision, self._complex_type)],
                order=order,
            )
            _, self._out_ld = self._get_numpy_layout(self.output, second_arr=self._rhs)
            self._func = self._mm_function()
            self._bind_mm_signature()
            self._rhs_ptr = ctypes.c_void_p(self._rhs.ctypes.data)
            self._out_ptr = ctypes.c_void_p(self.output.ctypes.data)
            self._apply_hints(kind="mm", columns=self._rhs.shape[1], layout=self._layout)

    def _mv_function(self):
        if self._double_precision and self._complex_type:
            return self._MKL._mkl_sparse_z_mv
        if self._complex_type:
            return self._MKL._mkl_sparse_c_mv
        if self._double_precision:
            return self._MKL._mkl_sparse_d_mv
        return self._MKL._mkl_sparse_s_mv

    def _mm_function(self):
        if self._double_precision and self._complex_type:
            return self._MKL._mkl_sparse_z_mm
        if self._complex_type:
            return self._MKL._mkl_sparse_c_mm
        if self._double_precision:
            return self._MKL._mkl_sparse_d_mm
        return self._MKL._mkl_sparse_s_mm

    def _bind_mv_signature(self) -> None:
        self._func.argtypes = [
            ctypes.c_int,
            self._scalar_ctype,
            self._sparse_matrix_t,
            type(self._descr),
            ctypes.c_void_p,
            self._scalar_ctype,
            ctypes.c_void_p,
        ]
        self._func.restype = ctypes.c_int

    def _bind_mm_signature(self) -> None:
        self._func.argtypes = [
            ctypes.c_int,
            self._scalar_ctype,
            self._sparse_matrix_t,
            type(self._descr),
            ctypes.c_int,
            ctypes.c_void_p,
            self._MKL.MKL_INT,
            self._MKL.MKL_INT,
            self._scalar_ctype,
            ctypes.c_void_p,
            self._MKL.MKL_INT,
        ]
        self._func.restype = ctypes.c_int

    def _apply_hints(self, kind: str, columns: int, layout=None) -> None:
        try:
            from sparse_dot_mkl._mkl_interface._load_library import mkl_library

            libmkl = mkl_library()
            optimize = getattr(libmkl, "mkl_sparse_optimize")
            optimize.argtypes = [self._sparse_matrix_t]
            optimize.restype = ctypes.c_int

            if kind == "mv":
                set_hint = getattr(libmkl, "mkl_sparse_set_mv_hint")
                set_hint.argtypes = [
                    self._sparse_matrix_t,
                    ctypes.c_int,
                    type(self._descr),
                    self._MKL.MKL_INT,
                ]
                set_hint.restype = ctypes.c_int
                status = set_hint(self._handle, self._operation, self._descr, self._MKL.MKL_INT(1))
                self._check_return_value(status, "mkl_sparse_set_mv_hint")
            else:
                set_hint = getattr(libmkl, "mkl_sparse_set_mm_hint")
                set_hint.argtypes = [
                    self._sparse_matrix_t,
                    ctypes.c_int,
                    type(self._descr),
                    ctypes.c_int,
                    self._MKL.MKL_INT,
                    self._MKL.MKL_INT,
                ]
                set_hint.restype = ctypes.c_int
                status = set_hint(
                    self._handle,
                    self._operation,
                    self._descr,
                    ctypes.c_int(layout),
                    self._MKL.MKL_INT(columns),
                    self._MKL.MKL_INT(1),
                )
                self._check_return_value(status, "mkl_sparse_set_mm_hint")

            status = optimize(self._handle)
            self._check_return_value(status, "mkl_sparse_optimize")
            self.info["mkl_handle_hint"] = f"{kind}+optimize"
        except Exception as exc:
            self.info["mkl_handle_hint"] = "unavailable"
            self.info["mkl_handle_hint_error"] = str(exc)

    def __call__(self):
        if self.mode in ("mult", "mult_adjoint"):
            status = self._func(
                self._operation,
                self._alpha,
                self._handle,
                self._descr,
                self._rhs_ptr,
                self._beta,
                self._out_ptr,
            )
            self._check_return_value(status, self._func.__name__)
            return self.output

        status = self._func(
            self._operation,
            self._alpha,
            self._handle,
            self._descr,
            self._layout,
            self._rhs_ptr,
            self._rhs.shape[1],
            self._rhs_ld,
            self._beta,
            self._out_ptr,
            self._out_ld,
        )
        self._check_return_value(status, self._func.__name__)
        return self.output

    def close(self) -> None:
        if self._handle is not None:
            self._destroy_mkl_handle(self._handle)
            self._handle = None


def make_snapshot(args: argparse.Namespace, rank_count: int, dtype: np.dtype, mat, timings: dict[str, float], extra: dict[str, object]) -> dict[str, object]:
    snapshot = {
        "preset": args.label or f"{args.family}:{args.profile}",
        "family": args.family,
        "profile": args.profile,
        "mode": args.mode,
        "env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "git_commit": try_get_git_commit(),
        "command": shlex.join(sys.argv),
        "python": sys.executable,
        "vbcsr_core": getattr(vbcsr_core, "__file__", "unknown"),
        "rank_count": rank_count,
        "dtype": str(dtype),
        "matrix_kind": mat.matrix_kind,
        "structure": {
            "blocks": args.blocks,
            "density": args.density,
            "min_block": args.min_block,
            "max_block": args.max_block,
            "unique_block_sizes": sorted({int(size) for size in mat.graph.block_sizes}),
        },
        "timings": timings,
    }
    snapshot.update(extra)
    return snapshot


def write_snapshot(path: str, snapshot: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description="VBCSR phase 0 benchmark and baseline runner")
    parser.add_argument("--family", type=str, default="random", choices=["random", "csr", "bsr", "vbcsr"], help="Matrix family preset")
    parser.add_argument("--profile", type=str, default="custom", choices=["custom", "small", "medium"], help="Preset profile")
    parser.add_argument("--label", type=str, default=None, help="Optional snapshot label override")
    parser.add_argument("--snapshot-out", type=str, default=None, help="Write a JSON snapshot to this path")
    parser.add_argument("--blocks", type=int, default=None, help="Total number of blocks")
    parser.add_argument("--min-block", type=int, default=None, help="Min block size")
    parser.add_argument("--max-block", type=int, default=None, help="Max block size")
    parser.add_argument("--density", type=float, default=None, help="Sparsity density")
    parser.add_argument("--complex", action="store_true", help="Use complex numbers")
    parser.add_argument("--scipy", action="store_true", help="Compare with SciPy (serial only)")
    parser.add_argument("--mkl", action="store_true", help="Compare with MKL (serial only)")
    parser.add_argument(
        "--mode",
        type=str,
        default="mult",
        choices=CANONICAL_MODES,
        help="Benchmark mode (`mult`, `mult_dense`, `mult_adjoint`, `mult_dense_adjoint`, `spmm`)",
    )
    parser.add_argument("--num-vecs", type=int, default=None, help="Number of vectors for dense RHS modes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-seconds", type=float, default=2.0, help="Minimum benchmark loop duration")
    parser.add_argument("--min-iterations", type=int, default=5, help="Minimum benchmark iterations")
    args = parser.parse_args()

    apply_profile_defaults(args)
    if args.mkl:
        try:
            import sparse_dot_mkl
            mkl_thread_count = sparse_dot_mkl.mkl_get_max_threads()
        except ImportError:
            print("Error: sparse_dot_mkl not found. Please install it to use --mkl.")
            return
    else:
        sparse_dot_mkl = None
        mkl_thread_count = None

    if MPI is not None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1

    dtype = np.complex128 if args.complex else np.float64

    if rank == 0:
        print("=== VBCSR Benchmark ===")
        print(f"Preset: {args.label or f'{args.family}:{args.profile}'}")
        print(f"Mode: {args.mode}")
        print(f"Ranks: {size}")
        print(f"Blocks: {args.blocks}")
        print(f"Block Size Range: [{args.min_block}, {args.max_block}]")
        print(f"Density: {args.density}")
        print(f"Dtype: {dtype}")
        if args.mode in DENSE_MODES:
            print(f"Num Vecs: {args.num_vecs}")
        print("Generating structure...", flush=True)

    block_sizes = generate_block_sizes(args.blocks, args.family, args.min_block, args.max_block, seed=args.seed)
    _, adj = generate_random_structure(args.blocks, block_sizes, args.density, seed=args.seed)

    blocks_per_rank = args.blocks // size
    remainder = args.blocks % size
    start_block = rank * blocks_per_rank + min(rank, remainder)
    my_count = blocks_per_rank + (1 if rank < remainder else 0)
    end_block = start_block + my_count
    owned_indices = list(range(start_block, end_block))
    my_block_sizes = block_sizes[start_block:end_block]
    my_adj = adj[start_block:end_block]

    if rank == 0:
        print("Building VBCSR...", flush=True)

    if comm is not None:
        comm.Barrier()
    t0 = time.perf_counter()
    mat = vbcsr.VBCSR.create_distributed(
        owned_indices=owned_indices,
        block_sizes=my_block_sizes,
        adjacency=my_adj,
        dtype=dtype,
        comm=comm,
    )
    mat.fill_random()
    mat.assemble()
    if comm is not None:
        comm.Barrier()
        t_gen = comm.allreduce(time.perf_counter() - t0, op=MPI.MAX)
    else:
        t_gen = time.perf_counter() - t0

    if rank == 0:
        print(f"VBCSR Generation Time: {t_gen:.4f} s")
        print(f"Matrix Shape: {mat.shape}")
        print(f"Matrix Kind: {mat.matrix_kind}")

    rng = np.random.default_rng(args.seed + rank)

    x_vbcsr = None
    y_vbcsr = None
    x_np = None
    if args.mode in ("mult", "mult_adjoint"):
        x_vbcsr = mat.create_vector()
        y_vbcsr = mat.create_vector()
        x_vbcsr.set_constant(1.0)
        x_np = x_vbcsr.to_numpy() if size == 1 else None
    elif args.mode in DENSE_MODES:
        x_vbcsr = mat.create_multivector(args.num_vecs)
        y_vbcsr = mat.create_multivector(args.num_vecs)
        x_vbcsr.set_constant(1.0)
        if dtype == np.float64:
            x_local_data = rng.random((x_vbcsr.local_rows, args.num_vecs))
        else:
            x_local_data = rng.random((x_vbcsr.local_rows, args.num_vecs)) + 1j * rng.random((x_vbcsr.local_rows, args.num_vecs))
        x_vbcsr.from_numpy(x_local_data)
        x_np = x_vbcsr.to_numpy() if size == 1 else None

    def benchmark_op(op_func, name):
        if rank == 0:
            print(f"Benchmarking {name}...", flush=True)
        if comm is not None:
            comm.Barrier()
        op_func()
        if comm is not None:
            comm.Barrier()

        t_start = time.perf_counter()
        n_iter = 0
        while True:
            op_func()
            n_iter += 1
            local_elapsed = time.perf_counter() - t_start
            keep_going = local_elapsed < args.min_seconds or n_iter < args.min_iterations
            if comm is not None:
                keep_going = bool(comm.allreduce(int(keep_going), op=MPI.MAX))
            if not keep_going:
                break
        if comm is not None:
            comm.Barrier()
            total_elapsed = comm.allreduce(time.perf_counter() - t_start, op=MPI.MAX)
        else:
            total_elapsed = time.perf_counter() - t_start
        t_avg = total_elapsed / n_iter
        if rank == 0:
            print(f"{name} Average Time: {t_avg:.6f} s ({n_iter} iterations)")
        return t_avg

    timings: dict[str, float] = {"generation": t_gen}
    comparisons: dict[str, object] = {}

    if args.mode == "mult":
        t_vbcsr = benchmark_op(lambda: mat.mult(x_vbcsr, y_vbcsr), "VBCSR Mult")
    elif args.mode == "mult_dense":
        t_vbcsr = benchmark_op(lambda: mat.mult(x_vbcsr, y_vbcsr), "VBCSR MultDense")
    elif args.mode == "mult_adjoint":
        t_vbcsr = benchmark_op(lambda: mat.mult_adjoint(x_vbcsr, y_vbcsr), "VBCSR MultAdjoint")
    elif args.mode == "mult_dense_adjoint":
        t_vbcsr = benchmark_op(lambda: mat.mult_adjoint(x_vbcsr, y_vbcsr), "VBCSR MultDenseAdjoint")
    else:
        t_vbcsr = benchmark_op(lambda: mat.spmm(mat), "VBCSR SpMM")
    timings["vbcsr"] = t_vbcsr

    sp_mat = None
    sp_mat_adjoint = None
    mkl_sp_mat = None
    baseline_info: dict[str, object] = {}
    if (args.scipy or args.mkl) and size == 1:
        if rank == 0:
            print("Building scalar CSR baseline...", flush=True)
        t0 = time.perf_counter()
        sp_mat = make_scalar_csr_from_vbcsr(mat)
        timings["scipy_build"] = time.perf_counter() - t0
        if rank == 0:
            print(f"Scalar CSR Generation Time: {timings['scipy_build']:.4f} s")

        if args.scipy and args.mode in ADJOINT_MODES:
            if rank == 0:
                print("Building scalar CSR adjoint baseline...", flush=True)
            t0 = time.perf_counter()
            sp_mat_adjoint = sort_sparse_indices(sp_mat.getH().tocsr())
            timings["scipy_adjoint_build"] = time.perf_counter() - t0
            if rank == 0:
                print(f"Scalar CSR adjoint build time: {timings['scipy_adjoint_build']:.4f} s")

        if args.mkl:
            if rank == 0:
                print("Selecting MKL sparse baseline format...", flush=True)
            t0 = time.perf_counter()
            mkl_sp_mat, baseline_info = make_mkl_sparse_baseline(sp_mat, block_sizes)
            timings["mkl_sparse_build"] = time.perf_counter() - t0
            if rank == 0:
                blocksize = baseline_info.get("mkl_blocksize")
                blocksize_text = f", blocksize={blocksize}" if blocksize is not None else ""
                print(
                    "MKL sparse baseline: "
                    f"{baseline_info['mkl_sparse_format']}"
                    f"{blocksize_text} "
                    f"({baseline_info['mkl_format_reason']})"
                )
                print(f"MKL sparse format build time: {timings['mkl_sparse_build']:.4f} s")

    if args.scipy and size == 1 and sp_mat is not None:
        if args.mode == "mult":
            t_scipy = benchmark_op(lambda: sp_mat.dot(x_np), "SciPy Mult")
        elif args.mode == "mult_dense":
            t_scipy = benchmark_op(lambda: sp_mat.dot(x_np), "SciPy MultDense")
        elif args.mode == "mult_adjoint":
            t_scipy = benchmark_op(lambda: sp_mat_adjoint.dot(x_np), "SciPy MultAdjoint")
        elif args.mode == "mult_dense_adjoint":
            t_scipy = benchmark_op(lambda: sp_mat_adjoint.dot(x_np), "SciPy MultDenseAdjoint")
        else:
            t_scipy = benchmark_op(lambda: sp_mat.dot(sp_mat), "SciPy SpMM")
        timings["scipy"] = t_scipy
        comparisons["scipy_speedup"] = t_scipy / t_vbcsr
        if rank == 0:
            print(f"Speedup (SciPy / VBCSR): {comparisons['scipy_speedup']:.2f}x")

    if args.mkl and size == 1 and mkl_sp_mat is not None and sparse_dot_mkl is not None:
        sparse_dot_mkl.mkl_set_num_threads(mkl_thread_count)
        baseline_info["mkl_threads"] = mkl_thread_count
        cached_mkl_op = None
        op_mkl = None
        mkl_name = f"MKL {args.mode}"

        if args.mode != "spmm":
            try:
                t0 = time.perf_counter()
                cached_mkl_op = CachedMKLSparseOp(mkl_sp_mat, x_np, args.mode)
                timings["mkl_handle_build"] = time.perf_counter() - t0
                baseline_info.update(cached_mkl_op.info)
                op_mkl = cached_mkl_op
                mkl_name = f"MKL cached {args.mode}"
                if rank == 0:
                    print(f"MKL sparse handle build time: {timings['mkl_handle_build']:.4f} s")
                    print(
                        "MKL sparse runner: cached sparse handle "
                        f"({baseline_info.get('mkl_handle_hint', 'no hint')})"
                    )
            except Exception as exc:
                comparisons["mkl_cached_error"] = str(exc)
                baseline_info["mkl_runner"] = "sparse_dot_mkl_wrapper"
                if rank == 0:
                    print(f"Cached MKL setup failed, falling back to sparse_dot_mkl wrapper: {exc}")

        if op_mkl is None:
            baseline_info["mkl_runner"] = "sparse_dot_mkl_wrapper"
            if args.mode == "mult":
                op_mkl = lambda: sparse_dot_mkl.dot_product_mkl(mkl_sp_mat, x_np)
            elif args.mode == "mult_dense":
                # Fallback wrapper path creates and destroys an MKL sparse
                # handle per call. Keep its previously fastest RHS layout.
                x_np_mkl = np.ascontiguousarray(x_np)
                op_mkl = lambda: sparse_dot_mkl.dot_product_mkl(mkl_sp_mat, x_np_mkl)
            elif args.mode == "mult_adjoint":
                mkl_sp_mat_adjoint = sort_sparse_indices(mkl_sp_mat.getH().tocsr())
                op_mkl = lambda: sparse_dot_mkl.dot_product_mkl(mkl_sp_mat_adjoint, x_np)
            elif args.mode == "mult_dense_adjoint":
                mkl_sp_mat_adjoint = sort_sparse_indices(mkl_sp_mat.getH().tocsr())
                x_np_mkl = np.ascontiguousarray(x_np)
                op_mkl = lambda: sparse_dot_mkl.dot_product_mkl(mkl_sp_mat_adjoint, x_np_mkl)
            else:
                op_mkl = lambda: sparse_dot_mkl.dot_product_mkl(mkl_sp_mat, mkl_sp_mat)

        try:
            t_mkl = benchmark_op(op_mkl, mkl_name)
            timings["mkl"] = t_mkl
            comparisons["mkl_speedup"] = t_mkl / t_vbcsr
            if rank == 0:
                print(f"Speedup (MKL / VBCSR): {comparisons['mkl_speedup']:.2f}x")
        except Exception as exc:
            comparisons["mkl_error"] = str(exc)
            if rank == 0:
                print(f"MKL benchmark failed: {exc}")
        finally:
            if cached_mkl_op is not None:
                cached_mkl_op.close()

    if args.scipy and size == 1 and sp_mat is not None:
        if rank == 0:
            print("Verifying correctness...", flush=True)
        if args.mode == "mult":
            mat.mult(x_vbcsr, y_vbcsr)
            res_vbcsr = y_vbcsr.to_numpy()
            res_scipy = sp_mat.dot(x_np)
        elif args.mode == "mult_dense":
            mat.mult(x_vbcsr, y_vbcsr)
            res_vbcsr = y_vbcsr.to_numpy()
            res_scipy = sp_mat.dot(x_np)
        elif args.mode == "mult_adjoint":
            mat.mult_adjoint(x_vbcsr, y_vbcsr)
            res_vbcsr = y_vbcsr.to_numpy()
            res_scipy = sp_mat_adjoint.dot(x_np)
        elif args.mode == "mult_dense_adjoint":
            mat.mult_adjoint(x_vbcsr, y_vbcsr)
            res_vbcsr = y_vbcsr.to_numpy()
            res_scipy = sp_mat_adjoint.dot(x_np)
        else:
            res_vbcsr = mat.spmm(mat).to_scipy().toarray()
            res_scipy = sp_mat.dot(sp_mat).toarray()

        diff = np.linalg.norm(res_vbcsr - res_scipy) / (np.linalg.norm(res_scipy) + 1e-10)
        comparisons["relative_difference"] = float(diff)
        if rank == 0:
            print(f"Relative Difference (SciPy): {diff:.2e}")

    if rank == 0 and args.snapshot_out:
        snapshot = make_snapshot(
            args,
            size,
            np.dtype(dtype),
            mat,
            timings,
            {
                "comparisons": comparisons,
                "baseline": baseline_info,
            },
        )
        write_snapshot(args.snapshot_out, snapshot)
        print(f"Wrote snapshot to {args.snapshot_out}")


if __name__ == "__main__":
    main()

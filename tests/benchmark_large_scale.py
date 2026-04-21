import argparse
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
CANONICAL_MODES = ("mult", "mult_dense", "spmm")


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
        args.min_block = 8
        args.max_block = 8
    elif args.family == "vbcsr":
        args.min_block = int(VBCSR_SHAPES.min())
        args.max_block = int(VBCSR_SHAPES.max())
    else:
        args.min_block = 10 if args.min_block is None else args.min_block
        args.max_block = 50 if args.max_block is None else args.max_block

def generate_block_sizes(global_blocks: int, family: str, block_size_min: int, block_size_max: int, seed: int = 42) -> list[int]:
    rng = np.random.default_rng(seed)
    if family == "csr":
        return [1] * global_blocks
    if family == "bsr":
        return [8] * global_blocks
    if family == "vbcsr":
        return rng.choice(VBCSR_SHAPES, size=global_blocks, replace=True).astype(int).tolist()
    return rng.integers(block_size_min, block_size_max + 1, size=global_blocks).astype(int).tolist()


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
        help="Benchmark mode (`mult`, `mult_dense`, `spmm`)",
    )
    parser.add_argument("--num-vecs", type=int, default=None, help="Number of vectors for mult_dense")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-seconds", type=float, default=1.0, help="Minimum benchmark loop duration")
    parser.add_argument("--min-iterations", type=int, default=5, help="Minimum benchmark iterations")
    args = parser.parse_args()

    apply_profile_defaults(args)
    if args.mkl:
        try:
            import sparse_dot_mkl
        except ImportError:
            print("Error: sparse_dot_mkl not found. Please install it to use --mkl.")
            return
    else:
        sparse_dot_mkl = None

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
        if args.mode == "mult_dense":
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

    t0 = time.perf_counter()
    mat = vbcsr.VBCSR.create_distributed(
        owned_indices=owned_indices,
        block_sizes=my_block_sizes,
        adjacency=my_adj,
        dtype=dtype,
        comm=comm,
    )

    scipy_rows = []
    scipy_cols = []
    scipy_data = []

    if (args.scipy or args.mkl) and rank == 0:
        row_offsets = np.zeros(args.blocks + 1, dtype=int)
        np.cumsum(block_sizes, out=row_offsets[1:])

    rng = np.random.default_rng(args.seed + rank)
    for local_i, global_i in enumerate(owned_indices):
        r_dim = my_block_sizes[local_i]
        neighbors = my_adj[local_i]
        r_start = row_offsets[global_i] if (args.scipy or args.mkl) and size == 1 else 0

        for global_j in neighbors:
            c_dim = block_sizes[global_j]
            if dtype == np.float64:
                data = rng.random((r_dim, c_dim))
            else:
                data = rng.random((r_dim, c_dim)) + 1j * rng.random((r_dim, c_dim))

            mat.add_block(global_i, global_j, data)

            if (args.scipy or args.mkl) and size == 1:
                c_start = row_offsets[global_j]
                r_idx, c_idx = np.indices((r_dim, c_dim))
                scipy_rows.append((r_idx + r_start).ravel())
                scipy_cols.append((c_idx + c_start).ravel())
                scipy_data.append(data.ravel())

    mat.assemble()
    if comm is not None:
        comm.Barrier()
    t_gen = time.perf_counter() - t0

    if rank == 0:
        print(f"VBCSR Generation Time: {t_gen:.4f} s")
        print(f"Matrix Shape: {mat.shape}")
        print(f"Matrix Kind: {mat.matrix_kind}")

    x_vbcsr = None
    y_vbcsr = None
    x_np = None
    if args.mode == "mult":
        x_vbcsr = mat.create_vector()
        y_vbcsr = mat.create_vector()
        x_vbcsr.set_constant(1.0)
        x_np = x_vbcsr.to_numpy() if size == 1 else None
    elif args.mode == "mult_dense":
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
    else:
        t_vbcsr = benchmark_op(lambda: mat.spmm(mat), "VBCSR SpMM")
    timings["vbcsr"] = t_vbcsr

    sp_mat = None
    mkl_sp_mat = None
    baseline_info: dict[str, object] = {}
    if (args.scipy or args.mkl) and size == 1:
        if rank == 0:
            print("Building scalar CSR baseline...", flush=True)
        t0 = time.perf_counter()
        if scipy_rows:
            all_rows = np.concatenate(scipy_rows)
            all_cols = np.concatenate(scipy_cols)
            all_data = np.concatenate(scipy_data)
            sp_mat = scipy.sparse.csr_matrix((all_data, (all_rows, all_cols)), shape=mat.shape)
        else:
            sp_mat = scipy.sparse.csr_matrix(mat.shape, dtype=dtype)
        sort_sparse_indices(sp_mat)
        timings["scipy_build"] = time.perf_counter() - t0
        if rank == 0:
            print(f"Scalar CSR Generation Time: {timings['scipy_build']:.4f} s")

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
        else:
            t_scipy = benchmark_op(lambda: sp_mat.dot(sp_mat), "SciPy SpMM")
        timings["scipy"] = t_scipy
        comparisons["scipy_speedup"] = t_scipy / t_vbcsr
        if rank == 0:
            print(f"Speedup (SciPy / VBCSR): {comparisons['scipy_speedup']:.2f}x")

    if args.mkl and size == 1 and mkl_sp_mat is not None and sparse_dot_mkl is not None:
        if args.mode == "mult":
            op_mkl = lambda: sparse_dot_mkl.dot_product_mkl(mkl_sp_mat, x_np)
        elif args.mode == "mult_dense":
            # sparse_dot_mkl accepts both row-major and column-major dense RHS
            # arrays, but its sparse x dense path is much faster with a C-order
            # RHS in this benchmark. DistMultiVector exposes a Fortran-order
            # view, so make the layout explicit before timing sparse_dot_mkl.
            x_np_mkl = np.ascontiguousarray(x_np)
            op_mkl = lambda: sparse_dot_mkl.dot_product_mkl(mkl_sp_mat, x_np_mkl)
        else:
            op_mkl = lambda: sparse_dot_mkl.dot_product_mkl(mkl_sp_mat, mkl_sp_mat)

        try:
            t_mkl = benchmark_op(op_mkl, f"MKL {args.mode}")
            timings["mkl"] = t_mkl
            comparisons["mkl_speedup"] = t_mkl / t_vbcsr
            if rank == 0:
                print(f"Speedup (MKL / VBCSR): {comparisons['mkl_speedup']:.2f}x")
        except Exception as exc:
            comparisons["mkl_error"] = str(exc)
            if rank == 0:
                print(f"MKL benchmark failed: {exc}")

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

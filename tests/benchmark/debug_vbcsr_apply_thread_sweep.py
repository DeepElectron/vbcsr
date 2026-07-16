#!/usr/bin/env python3
"""Thread/RHS/degree sweep for VBCSR apply diagnostics."""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import datetime as dt
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

import debug_vbcsr_apply_breakdown as breakdown
import run_benchmark as rb


SCRIPT_DIR = Path(__file__).resolve().parent


def comma_ints(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list must not be empty")
    return items


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep VBCSR apply over degree, RHS, and thread count.")
    parser.add_argument("--blocks", type=int, default=int(os.environ.get("BLOCKS", "4096")))
    parser.add_argument("--target-degrees", type=comma_ints, default=comma_ints(os.environ.get("TARGET_DEGREES", "12,25,50,100")))
    parser.add_argument("--rhs-values", type=comma_ints, default=comma_ints(os.environ.get("RHS_VALUES", "1,4,16")))
    parser.add_argument("--threads", type=comma_ints, default=comma_ints(os.environ.get("THREADS", "1,2,4,8,16,32")))
    parser.add_argument("--operation", choices=("spmv", "spmm"), default=os.environ.get("OPERATION", "spmm"))
    parser.add_argument("--dtype", choices=("real", "complex"), default=os.environ.get("DTYPE", "real"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "1729")))
    parser.add_argument("--geometry-dim", type=int, default=int(os.environ.get("GEOMETRY_DIM", "3")))
    parser.add_argument("--geometry-spacing", type=float, default=float(os.environ.get("GEOMETRY_SPACING", "1.0")))
    parser.add_argument("--geometry-jitter", type=float, default=float(os.environ.get("GEOMETRY_JITTER", "0.12")))
    parser.add_argument("--geometry-cutoff", type=float, default=None)
    parser.add_argument("--geometry-cutoff-quantile", type=float, default=float(os.environ.get("GEOMETRY_CUTOFF_QUANTILE", "0.90")))
    parser.add_argument("--magnitude-decay-length", type=float, default=float(os.environ.get("MAGNITUDE_DECAY_LENGTH", "0.5")))
    parser.add_argument("--offdiagonal-scale", type=float, default=float(os.environ.get("OFFDIAGONAL_SCALE", "1.0")))
    parser.add_argument("--diagonal-shift", type=float, default=float(os.environ.get("DIAGONAL_SHIFT", "2.0")))
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("REPEATS", "3")))
    parser.add_argument("--warmups", type=int, default=int(os.environ.get("WARMUPS", "1")))
    parser.add_argument("--min-seconds", type=float, default=float(os.environ.get("MIN_SECONDS", "0.04")))
    parser.add_argument("--min-iterations", type=int, default=int(os.environ.get("MIN_ITERATIONS", "1")))
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--label", default=os.environ.get("LABEL"))
    return parser


def load_openmp_runtime() -> Any | None:
    candidates = [
        ctypes.util.find_library("gomp"),
        ctypes.util.find_library("omp"),
        "libgomp.so.1",
        "libomp.so",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            lib = ctypes.CDLL(candidate)
            lib.omp_set_num_threads.argtypes = [ctypes.c_int]
            lib.omp_set_num_threads.restype = None
            lib.omp_get_max_threads.argtypes = []
            lib.omp_get_max_threads.restype = ctypes.c_int
            return lib
        except Exception:
            continue
    return None


def set_thread_count(omp_runtime: Any | None, sparse_dot_mkl: Any, threads: int) -> dict[str, Any]:
    info: dict[str, Any] = {"requested": int(threads)}
    if omp_runtime is not None:
        omp_runtime.omp_set_num_threads(int(threads))
        info["omp_max_threads"] = int(omp_runtime.omp_get_max_threads())
    else:
        info["omp_runtime_available"] = False
    info["sparse_dot_mkl"] = rb.apply_sparse_dot_mkl_threading(
        sparse_dot_mkl,
        int(threads),
        "debug_vbcsr_apply_thread_sweep",
        "set_reference_mkl_threads",
    )
    return info


def make_spec(args: argparse.Namespace, degree: int, rhs: int) -> rb.BenchmarkSpec:
    return rb.BenchmarkSpec(
        suite="vbcsr-apply-thread-sweep",
        domain="vbcsr",
        operation=str(args.operation),
        blocks=int(args.blocks),
        target_degree=int(degree),
        rhs=int(rhs),
        dtype=rb.parse_dtype(args.dtype),
        seed=int(args.seed),
        bsr_block_size=8,
        spgemm_threshold=0.0,
        spgemm_audit_limit=0,
        geometry_dim=int(args.geometry_dim),
        geometry_spacing=float(args.geometry_spacing),
        geometry_jitter=float(args.geometry_jitter),
        geometry_cutoff=args.geometry_cutoff,
        geometry_cutoff_quantile=float(args.geometry_cutoff_quantile),
        magnitude_decay_length=float(args.magnitude_decay_length),
        offdiagonal_scale=float(args.offdiagonal_scale),
        diagonal_shift=float(args.diagonal_shift),
    )


def benchmark(op: Callable[[], Any], args: argparse.Namespace) -> dict[str, Any]:
    return rb.benchmark_repeated(
        op,
        comm=None,
        repeats=int(args.repeats),
        min_seconds=float(args.min_seconds),
        min_iterations=int(args.min_iterations),
        warmups=int(args.warmups),
    )


def make_vbcsr_op(matrix: Any, inputs: dict[str, Any], operation: str) -> Callable[[], Any]:
    if operation == "spmv":
        return lambda: matrix.mult(inputs["x_vector"], inputs["y_vector"])
    return lambda: matrix.mult(inputs["x_multivector"], inputs["y_multivector"])


def make_mkl_op(scalar: Any, inputs: dict[str, Any], operation: str, sparse_dot_mkl: Any) -> Callable[[], Any]:
    if operation == "spmv":
        rhs = inputs["x_vector"].to_numpy().copy()
        return lambda: sparse_dot_mkl.dot_product_mkl(scalar, rhs)
    rhs = np.ascontiguousarray(inputs["x_multivector"].to_numpy())
    return lambda: sparse_dot_mkl.dot_product_mkl(scalar, rhs)


def run(args: argparse.Namespace) -> dict[str, Any]:
    sparse_dot_mkl = importlib.import_module("sparse_dot_mkl")
    omp_runtime = load_openmp_runtime()
    results: list[dict[str, Any]] = []
    builds: list[dict[str, Any]] = []

    for degree in args.target_degrees:
        base_spec = make_spec(args, int(degree), int(args.rhs_values[0]))
        block_sizes = rb.make_block_sizes(base_spec)
        adjacency, adjacency_info, positions, box_lengths = rb.make_geometric_adjacency(base_spec)
        matrix, build_timings = rb.build_matrix(base_spec, block_sizes, adjacency, positions, box_lengths, None, 0, 1)
        scalar = rb.scipy_baseline_matrix(matrix)
        builds.append(
            {
                "degree": int(degree),
                "adjacency": adjacency_info,
                "build_timings": build_timings,
                "matrix": {
                    "shape": list(matrix.shape),
                    "matrix_kind": matrix.matrix_kind,
                    "local_block_nnz": int(matrix.local_block_nnz),
                    "local_scalar_nnz": int(matrix.local_nnz),
                    "shape_class_count": int(matrix.shape_class_count),
                    "scalar_csr_nnz": int(scalar.nnz),
                },
            }
        )

        for rhs in args.rhs_values:
            spec = make_spec(args, int(degree), int(rhs))
            inputs = rb.make_inputs(matrix, spec, rank=0)
            schedule = breakdown.shape_schedule_statistics(block_sizes, adjacency, matrix, int(rhs), spec.dtype)
            for threads in args.threads:
                thread_info = set_thread_count(omp_runtime, sparse_dot_mkl, int(threads))
                scalar_rows = int(schedule["scalar_rows"])
                itemsize = int(spec.dtype.itemsize)
                if args.operation == "spmm":
                    shape_thread_accumulator_bytes = int(threads) * scalar_rows * int(rhs) * itemsize
                else:
                    shape_thread_accumulator_bytes = int(threads) * scalar_rows * itemsize
                vbcsr_timing = benchmark(make_vbcsr_op(matrix, inputs, args.operation), args)
                mkl_timing = benchmark(make_mkl_op(scalar, inputs, args.operation, sparse_dot_mkl), args)
                results.append(
                    {
                        "operation": args.operation,
                        "degree": int(degree),
                        "rhs": int(rhs),
                        "threads": int(threads),
                        "omp_max_threads": thread_info.get("omp_max_threads"),
                        "block_nnz": int(matrix.local_block_nnz),
                        "scalar_nnz": int(matrix.local_nnz),
                        "shape_classes": int(schedule["shape_class_count"]),
                        "tasks_estimate": int(
                            schedule["spmm_task_count_estimate"]
                            if args.operation == "spmm"
                            else schedule["spmv_task_count_estimate"]
                        ),
                        "shape_thread_accumulator_bytes_if_used": shape_thread_accumulator_bytes,
                        "packed_rhs_bytes": int(
                            schedule["spmm_packed_rhs_bytes_per_call"]
                            if args.operation == "spmm"
                            else schedule["spmv_packed_rhs_bytes_per_call"]
                        ),
                        "packed_output_bytes": int(
                            schedule["spmm_packed_output_bytes_per_call"]
                            if args.operation == "spmm"
                            else schedule["spmv_packed_output_bytes_per_call"]
                        ),
                        "vbcsr_median_seconds": float(vbcsr_timing["median_seconds"]),
                        "mkl_csr_median_seconds": float(mkl_timing["median_seconds"]),
                        "vbcsr_over_mkl_time_ratio": float(vbcsr_timing["median_seconds"] / mkl_timing["median_seconds"]),
                        "vbcsr_iterations": list(vbcsr_timing["iterations"]),
                        "mkl_iterations": list(mkl_timing["iterations"]),
                    }
                )
                print(
                    f"degree={degree} rhs={rhs} threads={threads} "
                    f"vbcsr={vbcsr_timing['median_seconds']:.6g}s "
                    f"mkl={mkl_timing['median_seconds']:.6g}s "
                    f"ratio={vbcsr_timing['median_seconds'] / mkl_timing['median_seconds']:.3g}",
                    flush=True,
                )
    return {
        "schema": "debug-vbcsr-apply-thread-sweep-v1",
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "argv": sys.argv,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OMP_DYNAMIC",
                "MKL_NUM_THREADS",
                "MKL_DYNAMIC",
                "SPARSE_DOT_MKL_NUM_THREADS",
                "LD_LIBRARY_PATH",
            )
        },
        "builds": builds,
        "results": results,
    }


def write_outputs(payload: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or f"vbcsr_apply_thread_sweep_n{args.blocks}_{args.operation}"
    json_path = args.output_dir / f"{label}.json"
    csv_path = args.output_dir / f"{label}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = [
        "operation",
        "degree",
        "rhs",
        "threads",
        "omp_max_threads",
        "block_nnz",
        "scalar_nnz",
        "shape_classes",
        "tasks_estimate",
        "shape_thread_accumulator_bytes_if_used",
        "packed_rhs_bytes",
        "packed_output_bytes",
        "vbcsr_median_seconds",
        "mkl_csr_median_seconds",
        "vbcsr_over_mkl_time_ratio",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["results"]:
            writer.writerow({field: row.get(field) for field in fields})
    return csv_path, json_path


def main() -> int:
    args = make_parser().parse_args()
    payload = run(args)
    csv_path, json_path = write_outputs(payload, args)
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

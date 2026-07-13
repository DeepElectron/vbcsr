#!/usr/bin/env python3
"""Diagnose variable-block VBCSR apply performance against scalar MKL CSR.

This script intentionally uses the publication benchmark generator, but only
for the apply cases.  It separates the variable-block kernel from the fixed
BSR vendor path and records the structural reasons that VBCSR apply may cost
more than a scalar CSR reference.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

import run_benchmark as rb


SCRIPT_DIR = Path(__file__).resolve().parent


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Break down VBCSR SpMV/SpMM apply costs.")
    parser.add_argument("--blocks", type=int, default=int(os.environ.get("BLOCKS", "4096")))
    parser.add_argument("--target-degree", type=int, default=int(os.environ.get("TARGET_DEGREE", "100")))
    parser.add_argument("--rhs", type=int, default=int(os.environ.get("RHS", "16")))
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
    parser.add_argument("--operations", default=os.environ.get("OPERATIONS", "spmv,spmm"))
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("REPEATS", "5")))
    parser.add_argument("--warmups", type=int, default=int(os.environ.get("WARMUPS", "2")))
    parser.add_argument("--min-seconds", type=float, default=float(os.environ.get("MIN_SECONDS", "0.2")))
    parser.add_argument("--min-iterations", type=int, default=int(os.environ.get("MIN_ITERATIONS", "3")))
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--label", default=os.environ.get("LABEL"))
    return parser


def parse_operations(value: str) -> list[str]:
    operations = [item.strip().lower() for item in value.replace(";", ",").split(",") if item.strip()]
    bad = [item for item in operations if item not in {"spmv", "spmm"}]
    if bad:
        raise ValueError(f"unsupported operations {bad}; allowed spmv,spmm")
    if not operations:
        raise ValueError("at least one operation is required")
    return operations


def make_spec(args: argparse.Namespace, operation: str) -> rb.BenchmarkSpec:
    return rb.BenchmarkSpec(
        suite="vbcsr-apply-breakdown",
        domain="vbcsr",
        operation=operation,
        blocks=int(args.blocks),
        target_degree=int(args.target_degree),
        rhs=int(args.rhs),
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


def timed_once(op: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    result = op()
    return time.perf_counter() - start, result


def benchmark(op: Callable[[], Any], args: argparse.Namespace) -> dict[str, Any]:
    return rb.benchmark_repeated(
        op,
        comm=None,
        repeats=int(args.repeats),
        min_seconds=float(args.min_seconds),
        min_iterations=int(args.min_iterations),
        warmups=int(args.warmups),
    )


def operation_callable(matrix: Any, inputs: dict[str, Any], operation: str) -> Callable[[], Any]:
    if operation == "spmv":
        return lambda: matrix.mult(inputs["x_vector"], inputs["y_vector"])
    if operation == "spmm":
        return lambda: matrix.mult(inputs["x_multivector"], inputs["y_multivector"])
    raise ValueError(operation)


def mkl_callable(scalar: Any, inputs: dict[str, Any], operation: str, sparse_dot_mkl: Any, rhs_order: str) -> Callable[[], Any]:
    if operation == "spmv":
        rhs = inputs["x_vector"].to_numpy().copy()
        return lambda: sparse_dot_mkl.dot_product_mkl(scalar, rhs)
    if operation == "spmm":
        rhs_view = inputs["x_multivector"].to_numpy()
        if rhs_order == "C":
            rhs = np.ascontiguousarray(rhs_view)
        elif rhs_order == "F":
            rhs = np.asfortranarray(rhs_view)
        else:
            raise ValueError(rhs_order)
        return lambda: sparse_dot_mkl.dot_product_mkl(scalar, rhs)
    raise ValueError(operation)


def effective_blocks_per_page(elements_per_block: int, shape_count: int, configured_page_size: int) -> int:
    hard_safe_blocks = (1 << 24) - 1
    payload_limit = max(1, ((1 << 32) - 1) // max(elements_per_block, 1))
    return max(1, min(shape_count, configured_page_size, hard_safe_blocks, payload_limit))


def shape_schedule_statistics(
    block_sizes: list[int],
    adjacency: list[list[int]],
    matrix: Any,
    rhs: int,
    dtype: np.dtype,
) -> dict[str, Any]:
    shape_counts: Counter[tuple[int, int]] = Counter()
    for row, cols in enumerate(adjacency):
        row_dim = int(block_sizes[row])
        for col in cols:
            shape_counts[(row_dim, int(block_sizes[col]))] += 1

    configured = int(matrix.configured_page_size)
    page_rows: list[dict[str, Any]] = []
    page_count = 0
    dense_task_count = 0
    vector_task_count = 0
    dense_packed_rhs_bytes = 0
    dense_packed_out_bytes = 0
    vector_packed_rhs_bytes = 0
    vector_packed_out_bytes = 0
    itemsize = int(dtype.itemsize)
    target_elems = max(1, (1 << 20) // itemsize)

    for (row_dim, col_dim), count in sorted(shape_counts.items()):
        elems = row_dim * col_dim
        blocks_per_page = effective_blocks_per_page(elems, count, configured)
        pages = int(math.ceil(count / blocks_per_page))
        dense_per_block_scratch = rhs * (row_dim + col_dim)
        vector_per_block_scratch = row_dim + col_dim
        dense_chunk = max(1, min(count, target_elems // max(dense_per_block_scratch, 1)))
        vector_chunk = max(1, min(count, target_elems // max(vector_per_block_scratch, 1)))
        dense_tasks = 0
        vector_tasks = 0
        for page in range(pages):
            first = page * blocks_per_page
            page_blocks = min(blocks_per_page, count - first)
            dense_tasks += int(math.ceil(page_blocks / dense_chunk))
            vector_tasks += int(math.ceil(page_blocks / vector_chunk))

        dense_task_count += dense_tasks
        vector_task_count += vector_tasks
        dense_packed_rhs_bytes += count * col_dim * rhs * itemsize
        dense_packed_out_bytes += count * row_dim * rhs * itemsize
        vector_packed_rhs_bytes += count * col_dim * itemsize
        vector_packed_out_bytes += count * row_dim * itemsize
        page_count += pages
        page_rows.append(
            {
                "shape": [row_dim, col_dim],
                "blocks": int(count),
                "pages": pages,
                "blocks_per_page": int(blocks_per_page),
                "spmv_chunk_blocks": int(vector_chunk),
                "spmv_tasks": int(vector_tasks),
                "spmm_chunk_blocks": int(dense_chunk),
                "spmm_tasks": int(dense_tasks),
            }
        )

    scalar_rows = int(sum(block_sizes))
    threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
    vector_accumulator_bytes = threads * scalar_rows * itemsize
    dense_accumulator_bytes = threads * scalar_rows * rhs * itemsize
    return {
        "shape_class_count": len(shape_counts),
        "page_count": page_count,
        "configured_page_size": configured,
        "scalar_rows": scalar_rows,
        "openmp_threads_from_env": threads,
        "spmv_task_count_estimate": vector_task_count,
        "spmm_task_count_estimate": dense_task_count,
        "spmv_packed_rhs_bytes_per_call": int(vector_packed_rhs_bytes),
        "spmv_packed_output_bytes_per_call": int(vector_packed_out_bytes),
        "spmm_packed_rhs_bytes_per_call": int(dense_packed_rhs_bytes),
        "spmm_packed_output_bytes_per_call": int(dense_packed_out_bytes),
        "spmv_thread_accumulator_bytes_per_call": int(vector_accumulator_bytes),
        "spmm_thread_accumulator_bytes_per_call": int(dense_accumulator_bytes),
        "shape_rows": page_rows,
    }


def data_motion_microbench(scalar_rows: int, rhs: int, dtype: np.dtype, threads: int, repeats: int) -> dict[str, Any]:
    repeats = max(1, repeats)
    local = np.empty((threads, scalar_rows, rhs), dtype=dtype, order="F")
    out = np.empty((scalar_rows, rhs), dtype=dtype, order="F")

    def zero_full() -> None:
        local.fill(0)

    def reduce_full() -> None:
        np.sum(local, axis=0, out=out)

    zero_samples = []
    reduce_samples = []
    for _ in range(repeats):
        zero_samples.append(timed_once(zero_full)[0])
        reduce_samples.append(timed_once(reduce_full)[0])
    return {
        "numpy_zero_thread_accumulators_median_seconds": float(np.median(zero_samples)),
        "numpy_reduce_thread_accumulators_median_seconds": float(np.median(reduce_samples)),
        "buffer_bytes": int(local.nbytes),
        "note": "Python/numpy lower-level memory-touch reference, not the exact C++ loop implementation.",
    }


def run_case(operation: str, args: argparse.Namespace) -> dict[str, Any]:
    spec = make_spec(args, operation)
    block_sizes = rb.make_block_sizes(spec)
    adjacency, adjacency_info, positions, box_lengths = rb.make_geometric_adjacency(spec)
    matrix, build_timings = rb.build_matrix(spec, block_sizes, adjacency, positions, box_lengths, None, 0, 1)
    inputs = rb.make_inputs(matrix, spec, rank=0)

    op = operation_callable(matrix, inputs, operation)
    cold_seconds, _ = timed_once(op)
    timing = benchmark(op, args)

    scalar = rb.scipy_baseline_matrix(matrix)
    sparse_dot_mkl = importlib.import_module("sparse_dot_mkl")
    threading = rb.configure_sparse_dot_mkl_threading(sparse_dot_mkl)
    try:
        mkl_csr_timing = benchmark(mkl_callable(scalar, inputs, operation, sparse_dot_mkl, "C"), args)
        mkl_f_timing = None
        if operation == "spmm":
            mkl_f_timing = benchmark(mkl_callable(scalar, inputs, operation, sparse_dot_mkl, "F"), args)
    finally:
        threading_restore = rb.restore_sparse_dot_mkl_threading(sparse_dot_mkl, threading)

    shape_stats = shape_schedule_statistics(block_sizes, adjacency, matrix, int(args.rhs), spec.dtype)
    microbench = None
    if operation == "spmm":
        microbench = data_motion_microbench(
            int(shape_stats["scalar_rows"]),
            int(args.rhs),
            spec.dtype,
            max(1, int(shape_stats["openmp_threads_from_env"])),
            repeats=min(3, int(args.repeats)),
        )

    result = {
        "label": args.label or f"vbcsr_apply_breakdown_n{args.blocks}_deg{args.target_degree}_{operation}",
        "operation": operation,
        "parameters": {
            "blocks": int(args.blocks),
            "target_degree": int(args.target_degree),
            "rhs": int(args.rhs),
            "dtype": str(spec.dtype),
            "seed": int(args.seed),
            "vbcsr_block_sizes": list(rb.VBCSR_BLOCK_SIZES),
        },
        "adjacency": adjacency_info,
        "build_timings": build_timings,
        "matrix": {
            "matrix_kind": matrix.matrix_kind,
            "shape": list(matrix.shape),
            "local_block_nnz": int(matrix.local_block_nnz),
            "local_scalar_nnz": int(matrix.local_nnz),
            "shape_class_count": int(matrix.shape_class_count),
            "configured_page_size": int(matrix.configured_page_size),
            "page_size": int(matrix.page_size),
            "scalar_csr_nnz": int(scalar.nnz),
        },
        "shape_schedule": shape_stats,
        "timings": {
            "vbcsr_cold_first_call_seconds": float(cold_seconds),
            "vbcsr": timing,
            "mkl_csr_c_rhs": mkl_csr_timing,
            "mkl_csr_f_rhs": mkl_f_timing,
        },
        "speedups": {
            "mkl_csr_c_over_vbcsr": float(mkl_csr_timing["median_seconds"] / timing["median_seconds"]),
            "vbcsr_over_mkl_csr_c_time_ratio": float(timing["median_seconds"] / mkl_csr_timing["median_seconds"]),
        },
        "threading": {"sparse_dot_mkl": threading, "restore": threading_restore},
        "data_motion_microbench": microbench,
    }
    return result


def write_outputs(results: list[dict[str, Any]], args: argparse.Namespace) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or f"vbcsr_apply_breakdown_n{args.blocks}_deg{args.target_degree}"
    csv_path = args.output_dir / f"{label}.csv"
    json_path = args.output_dir / f"{label}.json"

    payload = {
        "schema": "debug-vbcsr-apply-breakdown-v1",
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "argv": sys.argv,
        "environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS",
            "OMP_DYNAMIC",
            "MKL_NUM_THREADS",
            "MKL_DYNAMIC",
            "SPARSE_DOT_MKL_NUM_THREADS",
            "LD_LIBRARY_PATH",
        )},
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fields = [
        "operation",
        "blocks",
        "target_degree",
        "rhs",
        "scalar_rows",
        "block_nnz",
        "scalar_nnz",
        "shape_classes",
        "shape_pages",
        "tasks_estimate",
        "thread_accumulator_mb",
        "packed_rhs_mb",
        "packed_output_mb",
        "vbcsr_median_seconds",
        "vbcsr_cold_first_call_seconds",
        "mkl_csr_c_median_seconds",
        "mkl_csr_f_median_seconds",
        "vbcsr_over_mkl_csr_c_time_ratio",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for result in results:
            operation = result["operation"]
            schedule = result["shape_schedule"]
            timings = result["timings"]
            task_key = "spmm_task_count_estimate" if operation == "spmm" else "spmv_task_count_estimate"
            acc_key = "spmm_thread_accumulator_bytes_per_call" if operation == "spmm" else "spmv_thread_accumulator_bytes_per_call"
            rhs_key = "spmm_packed_rhs_bytes_per_call" if operation == "spmm" else "spmv_packed_rhs_bytes_per_call"
            out_key = "spmm_packed_output_bytes_per_call" if operation == "spmm" else "spmv_packed_output_bytes_per_call"
            writer.writerow({
                "operation": operation,
                "blocks": result["parameters"]["blocks"],
                "target_degree": result["parameters"]["target_degree"],
                "rhs": result["parameters"]["rhs"],
                "scalar_rows": schedule["scalar_rows"],
                "block_nnz": result["matrix"]["local_block_nnz"],
                "scalar_nnz": result["matrix"]["local_scalar_nnz"],
                "shape_classes": schedule["shape_class_count"],
                "shape_pages": schedule["page_count"],
                "tasks_estimate": schedule[task_key],
                "thread_accumulator_mb": schedule[acc_key] / 1e6,
                "packed_rhs_mb": schedule[rhs_key] / 1e6,
                "packed_output_mb": schedule[out_key] / 1e6,
                "vbcsr_median_seconds": timings["vbcsr"]["median_seconds"],
                "vbcsr_cold_first_call_seconds": timings["vbcsr_cold_first_call_seconds"],
                "mkl_csr_c_median_seconds": timings["mkl_csr_c_rhs"]["median_seconds"],
                "mkl_csr_f_median_seconds": (
                    timings["mkl_csr_f_rhs"]["median_seconds"]
                    if timings["mkl_csr_f_rhs"] is not None else ""
                ),
                "vbcsr_over_mkl_csr_c_time_ratio": result["speedups"]["vbcsr_over_mkl_csr_c_time_ratio"],
            })
    return csv_path, json_path


def main() -> int:
    args = make_parser().parse_args()
    operations = parse_operations(args.operations)
    results = [run_case(operation, args) for operation in operations]
    csv_path, json_path = write_outputs(results, args)
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    for result in results:
        print(
            result["operation"],
            "vbcsr",
            result["timings"]["vbcsr"]["median_seconds"],
            "mkl_csr_c",
            result["timings"]["mkl_csr_c_rhs"]["median_seconds"],
            "ratio",
            result["speedups"]["vbcsr_over_mkl_csr_c_time_ratio"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

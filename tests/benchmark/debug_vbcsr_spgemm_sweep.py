#!/usr/bin/env python3
"""Focused VBCSR sparse-sparse multiply sweep.

This is a diagnostic companion to run_benchmark.py. It uses the same
publication matrix generator, but sweeps only VBCSR SpGEMM so performance
questions can be answered without running the full CSR/BSR/VBCSR suite.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import datetime as dt
import gc
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

import run_benchmark as rb


SCRIPT_DIR = Path(__file__).resolve().parent


def comma_ints(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list must not be empty")
    return items


def comma_floats(value: str) -> list[float]:
    items = [float(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list must not be empty")
    return items


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep VBCSR SpGEMM over size, degree, threshold, and thread count.")
    parser.add_argument("--blocks-values", type=comma_ints, default=comma_ints(os.environ.get("BLOCKS_VALUES", "512")))
    parser.add_argument("--target-degrees", type=comma_ints, default=comma_ints(os.environ.get("TARGET_DEGREES", "12,25,50,100")))
    parser.add_argument("--thresholds", type=comma_floats, default=comma_floats(os.environ.get("SPGEMM_THRESHOLDS", "0.0")))
    parser.add_argument("--threads", type=comma_ints, default=comma_ints(os.environ.get("THREADS", "1,8,32")))
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
    parser.add_argument("--spgemm-audit-limit", type=int, default=int(os.environ.get("SPGEMM_AUDIT_LIMIT", "200000")))
    parser.add_argument("--vbcsr-nested-mkl-threads", type=int, default=int(os.environ.get("VBCSR_NESTED_MKL_NUM_THREADS", "1")))
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("REPEATS", "3")))
    parser.add_argument("--warmups", type=int, default=int(os.environ.get("WARMUPS", "1")))
    parser.add_argument("--min-seconds", type=float, default=float(os.environ.get("MIN_SECONDS", "0.05")))
    parser.add_argument("--min-iterations", type=int, default=int(os.environ.get("MIN_ITERATIONS", "1")))
    parser.add_argument("--require-mkl", action="store_true")
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


def set_omp_threads(omp_runtime: Any | None, threads: int) -> dict[str, Any]:
    info: dict[str, Any] = {"requested": int(threads)}
    if omp_runtime is None:
        info["available"] = False
        return info
    omp_runtime.omp_set_num_threads(int(threads))
    info["available"] = True
    info["max_threads"] = int(omp_runtime.omp_get_max_threads())
    return info


def set_mkl_threads(sparse_dot_mkl: Any, threads: int, policy: str) -> dict[str, Any]:
    return rb.apply_sparse_dot_mkl_threading(
        sparse_dot_mkl,
        int(threads),
        "debug_vbcsr_spgemm_sweep",
        policy,
    )


def make_spec(args: argparse.Namespace, blocks: int, degree: int, threshold: float) -> rb.BenchmarkSpec:
    return rb.BenchmarkSpec(
        suite="vbcsr-spgemm-sweep",
        domain="vbcsr",
        operation="spgemm",
        blocks=int(blocks),
        target_degree=int(degree),
        rhs=1,
        dtype=rb.parse_dtype(args.dtype),
        seed=int(args.seed),
        bsr_block_size=8,
        spgemm_threshold=float(threshold),
        spgemm_audit_limit=int(args.spgemm_audit_limit),
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


def exact_validation(reference: Any, observed: Any, threshold: float, dtype: np.dtype) -> dict[str, Any]:
    error = rb.relative_error(reference, observed)
    tolerance = 1e-9 if dtype == np.dtype(np.complex128) else 1e-10
    exact_required = float(threshold) == 0.0
    return {
        "relative_error": error,
        "tolerance": tolerance,
        "passed": bool((not exact_required) or error <= tolerance),
        "exact_validation_required": exact_required,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    sparse_dot_mkl = importlib.import_module("sparse_dot_mkl")
    omp_runtime = load_openmp_runtime()
    results: list[dict[str, Any]] = []
    builds: list[dict[str, Any]] = []

    for blocks in args.blocks_values:
        for degree in args.target_degrees:
            base_spec = make_spec(args, int(blocks), int(degree), float(args.thresholds[0]))
            block_sizes = rb.make_block_sizes(base_spec)
            adjacency, adjacency_info, positions, box_lengths = rb.make_geometric_adjacency(base_spec)
            value_model = rb.matrix_value_model_statistics(adjacency, positions, box_lengths, base_spec)
            matrix, build_timings = rb.build_matrix(base_spec, block_sizes, adjacency, positions, box_lengths, None, 0, 1)
            scalar = rb.scipy_baseline_matrix(matrix)
            candidate_count = rb.spgemm_candidate_count(adjacency)
            matrix_info = {
                "blocks": int(blocks),
                "degree": int(degree),
                "adjacency": adjacency_info,
                "value_model": value_model,
                "build_timings": build_timings,
                "matrix": {
                    "shape": list(matrix.shape),
                    "matrix_kind": matrix.matrix_kind,
                    "local_block_nnz": int(matrix.local_block_nnz),
                    "local_scalar_nnz": int(matrix.local_nnz),
                    "shape_class_count": int(matrix.shape_class_count),
                    "scalar_csr_nnz": int(scalar.nnz),
                },
                "candidate_block_products": int(candidate_count),
            }
            builds.append(matrix_info)

            for threads in args.threads:
                omp_info = set_omp_threads(omp_runtime, int(threads))
                vbcsr_mkl_info = set_mkl_threads(
                    sparse_dot_mkl,
                    int(args.vbcsr_nested_mkl_threads),
                    "pin_nested_blas_during_vbcsr_spgemm",
                )
                for threshold in args.thresholds:
                    spec = make_spec(args, int(blocks), int(degree), float(threshold))
                    vbcsr_timing = benchmark(lambda: matrix.spmm(matrix, float(threshold)), args)
                    output = matrix.spmm(matrix, float(threshold))
                    observed = output.to_scipy(format="csr")
                    reference = scalar.dot(scalar)
                    validation = exact_validation(reference, observed, float(threshold), spec.dtype)
                    output_block_nnz = int(output.local_block_nnz)
                    output_scalar_nnz = int(output.local_nnz)
                    del output, observed
                    gc.collect()

                    scipy_timing = benchmark(lambda: scalar.dot(scalar), args)
                    mkl_thread_info = set_mkl_threads(
                        sparse_dot_mkl,
                        int(threads),
                        "set_reference_mkl_sparse_threads",
                    )
                    try:
                        mkl_timing = benchmark(lambda: sparse_dot_mkl.dot_product_mkl(scalar, scalar), args)
                        mkl_available = True
                        mkl_error = None
                    except Exception as exc:
                        if args.require_mkl:
                            raise
                        mkl_timing = {}
                        mkl_available = False
                        mkl_error = f"{type(exc).__name__}: {exc}"
                    finally:
                        set_mkl_threads(
                            sparse_dot_mkl,
                            int(args.vbcsr_nested_mkl_threads),
                            "restore_nested_blas_after_mkl_reference",
                        )

                    row = {
                        "blocks": int(blocks),
                        "degree": int(degree),
                        "threshold": float(threshold),
                        "threads": int(threads),
                        "omp": omp_info,
                        "vbcsr_nested_mkl": vbcsr_mkl_info,
                        "mkl_reference_threading": mkl_thread_info,
                        "block_nnz": int(matrix.local_block_nnz),
                        "scalar_nnz": int(matrix.local_nnz),
                        "candidate_block_products": int(candidate_count),
                        "output_block_nnz": output_block_nnz,
                        "output_scalar_nnz": output_scalar_nnz,
                        "fill_ratio_scalar": float(output_scalar_nnz / max(int(matrix.local_nnz), 1)),
                        "vbcsr_median_seconds": float(vbcsr_timing["median_seconds"]),
                        "scipy_median_seconds": float(scipy_timing["median_seconds"]),
                        "mkl_median_seconds": float(mkl_timing["median_seconds"]) if mkl_available else None,
                        "vbcsr_vs_scipy_speedup": float(scipy_timing["median_seconds"] / vbcsr_timing["median_seconds"]),
                        "vbcsr_vs_mkl_speedup": (
                            float(mkl_timing["median_seconds"] / vbcsr_timing["median_seconds"]) if mkl_available else None
                        ),
                        "mkl_available": mkl_available,
                        "mkl_error": mkl_error,
                        "validation": validation,
                        "vbcsr_iterations": list(vbcsr_timing["iterations"]),
                        "scipy_iterations": list(scipy_timing["iterations"]),
                        "mkl_iterations": list(mkl_timing.get("iterations", [])),
                    }
                    results.append(row)
                    mkl_text = f"{row['mkl_median_seconds']:.6g}s" if mkl_available else "unavailable"
                    speedup_text = f"{row['vbcsr_vs_mkl_speedup']:.3g}" if mkl_available else "NA"
                    print(
                        f"blocks={blocks} degree={degree} threshold={threshold:g} threads={threads} "
                        f"vbcsr={row['vbcsr_median_seconds']:.6g}s scipy={row['scipy_median_seconds']:.6g}s "
                        f"mkl={mkl_text} vbcsr_vs_mkl={speedup_text} "
                        f"out_blocks={output_block_nnz} fill={row['fill_ratio_scalar']:.3g} "
                        f"relerr={validation['relative_error']:.3g}",
                        flush=True,
                    )
    return {
        "schema": "debug-vbcsr-spgemm-sweep-v1",
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
                "VBCSR_NESTED_MKL_NUM_THREADS",
                "LD_LIBRARY_PATH",
            )
        },
        "builds": builds,
        "results": results,
    }


def write_outputs(payload: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or "vbcsr_spgemm_sweep"
    json_path = args.output_dir / f"{label}.json"
    csv_path = args.output_dir / f"{label}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = [
        "blocks",
        "degree",
        "threshold",
        "threads",
        "block_nnz",
        "scalar_nnz",
        "candidate_block_products",
        "output_block_nnz",
        "output_scalar_nnz",
        "fill_ratio_scalar",
        "vbcsr_median_seconds",
        "scipy_median_seconds",
        "mkl_median_seconds",
        "vbcsr_vs_scipy_speedup",
        "vbcsr_vs_mkl_speedup",
        "mkl_available",
        "validation_relative_error",
        "validation_passed",
        "validation_exact_required",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["results"]:
            validation = row["validation"]
            writer.writerow(
                {
                    "blocks": row["blocks"],
                    "degree": row["degree"],
                    "threshold": row["threshold"],
                    "threads": row["threads"],
                    "block_nnz": row["block_nnz"],
                    "scalar_nnz": row["scalar_nnz"],
                    "candidate_block_products": row["candidate_block_products"],
                    "output_block_nnz": row["output_block_nnz"],
                    "output_scalar_nnz": row["output_scalar_nnz"],
                    "fill_ratio_scalar": row["fill_ratio_scalar"],
                    "vbcsr_median_seconds": row["vbcsr_median_seconds"],
                    "scipy_median_seconds": row["scipy_median_seconds"],
                    "mkl_median_seconds": row["mkl_median_seconds"],
                    "vbcsr_vs_scipy_speedup": row["vbcsr_vs_scipy_speedup"],
                    "vbcsr_vs_mkl_speedup": row["vbcsr_vs_mkl_speedup"],
                    "mkl_available": row["mkl_available"],
                    "validation_relative_error": validation["relative_error"],
                    "validation_passed": validation["passed"],
                    "validation_exact_required": validation["exact_validation_required"],
                }
            )
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

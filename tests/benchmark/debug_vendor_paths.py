#!/usr/bin/env python3
"""Diagnose whether CSR/BSR apply paths actually use vendor sparse kernels.

This script intentionally reuses the publication benchmark matrix generator so
the debug data is comparable with run_benchmark.py.  It records the internal
VBCSR vendor launch counters, sparse-dot-mkl reference timings, and RHS memory
layout sensitivity for SpMM.
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
from pathlib import Path
from typing import Any, Callable

import numpy as np

import run_benchmark as rb


SCRIPT_DIR = Path(__file__).resolve().parent


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug CSR/BSR MKL vendor dispatch and SpMM layout costs.")
    parser.add_argument("--blocks", type=int, default=int(os.environ.get("BLOCKS", "512")))
    parser.add_argument("--target-degree", type=int, default=int(os.environ.get("TARGET_DEGREE", "64")))
    parser.add_argument("--rhs", type=int, default=int(os.environ.get("RHS", "16")))
    parser.add_argument("--bsr-block-size", type=int, default=int(os.environ.get("BSR_BLOCK_SIZE", "8")))
    parser.add_argument("--domains", default=os.environ.get("DOMAINS", "csr,bsr"))
    parser.add_argument("--operations", default=os.environ.get("OPERATIONS", "spmv,spmm"))
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
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("REPEATS", "5")))
    parser.add_argument("--min-seconds", type=float, default=float(os.environ.get("MIN_SECONDS", "0.2")))
    parser.add_argument("--min-iterations", type=int, default=int(os.environ.get("MIN_ITERATIONS", "3")))
    parser.add_argument("--warmups", type=int, default=int(os.environ.get("WARMUPS", "2")))
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--label", default=os.environ.get("LABEL"))
    return parser


def parse_list(value: str, allowed: tuple[str, ...], name: str) -> list[str]:
    items = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    bad = [item for item in items if item not in allowed]
    if bad:
        raise ValueError(f"unsupported {name}: {bad}; allowed={allowed}")
    if not items:
        raise ValueError(f"{name} must not be empty")
    return items


def make_spec(args: argparse.Namespace, domain: str, operation: str) -> rb.BenchmarkSpec:
    return rb.BenchmarkSpec(
        suite="vendor-debug",
        domain=domain,
        operation=operation,
        blocks=int(args.blocks),
        target_degree=int(args.target_degree),
        rhs=int(args.rhs),
        dtype=rb.parse_dtype(args.dtype),
        seed=int(args.seed),
        bsr_block_size=int(args.bsr_block_size),
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


def median_seconds(timing: dict[str, Any] | None) -> float | None:
    if not timing:
        return None
    value = timing.get("median_seconds")
    return float(value) if value is not None else None


def make_mkl_op(matrix: Any, inputs: dict[str, Any], operation: str, sparse_dot_mkl: Any, rhs_order: str) -> Callable[[], Any]:
    if operation == "spmv":
        rhs = inputs["x_vector"].to_numpy().copy()
        return lambda: sparse_dot_mkl.dot_product_mkl(matrix, rhs)
    if operation == "spmm":
        rhs_view = inputs["x_multivector"].to_numpy()
        if rhs_order == "C":
            rhs = np.ascontiguousarray(rhs_view)
        elif rhs_order == "F":
            rhs = np.asfortranarray(rhs_view)
        else:
            rhs = rhs_view.copy()
        return lambda: sparse_dot_mkl.dot_product_mkl(matrix, rhs)
    raise ValueError(operation)


def benchmark_vbcsr_with_launches(matrix: Any, inputs: dict[str, Any], spec: rb.BenchmarkSpec, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    missing = [name for name in ("vendor_backend_name", "vendor_launch_count", "reset_vendor_launch_count") if not hasattr(matrix, name)]
    if missing:
        raise RuntimeError(
            "installed vbcsr_core does not expose vendor debug hooks "
            f"{missing}; rebuild/reinstall after applying this patch"
        )

    matrix.reset_vendor_launch_count()
    before = int(matrix.vendor_launch_count)
    timing = rb.benchmark_repeated(
        rb.vbcsr_op(matrix, inputs, spec),
        comm=None,
        repeats=int(args.repeats),
        min_seconds=float(args.min_seconds),
        min_iterations=int(args.min_iterations),
        warmups=int(args.warmups),
    )
    after = int(matrix.vendor_launch_count)
    total_calls = int(sum(int(item) for item in timing.get("iterations", []))) + int(args.repeats) * int(args.warmups)
    launch_delta = after - before
    launch_per_call = float(launch_delta / total_calls) if total_calls > 0 else math.nan
    diagnostic = {
        "vendor_backend_name": matrix.vendor_backend_name,
        "vendor_launch_count_before": before,
        "vendor_launch_count_after": after,
        "vendor_launch_delta": launch_delta,
        "total_operation_calls_including_warmups": total_calls,
        "vendor_launches_per_call": launch_per_call,
        "page_size": int(matrix.page_size),
        "configured_page_size": int(matrix.configured_page_size),
        "matrix_kind": matrix.matrix_kind,
    }
    return timing, diagnostic


def benchmark_mkl_references(scalar: Any, inputs: dict[str, Any], spec: rb.BenchmarkSpec, args: argparse.Namespace) -> dict[str, Any]:
    sparse_dot_mkl = importlib.import_module("sparse_dot_mkl")
    threading = rb.configure_sparse_dot_mkl_threading(sparse_dot_mkl)
    if not threading.get("thread_control_ok", False):
        raise RuntimeError(f"sparse_dot_mkl thread control failed: {threading}")

    references: dict[str, Any] = {"threading": threading, "timings": {}}
    try:
        mkl_matrix, mkl_info = rb.mkl_baseline_matrix(scalar, spec)
        references["primary_matrix"] = {
            **mkl_info,
            "shape": list(mkl_matrix.shape),
            "nnz": int(mkl_matrix.nnz),
            "storage_bytes": rb.sparse_storage_bytes(mkl_matrix),
        }

        rhs_orders = ["native"]
        if spec.operation == "spmm":
            rhs_orders = ["C", "F"]

        for rhs_order in rhs_orders:
            timing = rb.benchmark_repeated(
                make_mkl_op(mkl_matrix, inputs, spec.operation, sparse_dot_mkl, rhs_order),
                comm=None,
                repeats=int(args.repeats),
                min_seconds=float(args.min_seconds),
                min_iterations=int(args.min_iterations),
                warmups=int(args.warmups),
            )
            references["timings"][f"primary_{rhs_order}"] = timing

        if spec.domain == "bsr":
            csr_matrix = scalar.tocsr()
            references["scalar_csr_matrix"] = {
                "format": "csr",
                "shape": list(csr_matrix.shape),
                "nnz": int(csr_matrix.nnz),
                "storage_bytes": rb.sparse_storage_bytes(csr_matrix),
            }
            for rhs_order in rhs_orders:
                timing = rb.benchmark_repeated(
                    make_mkl_op(csr_matrix, inputs, spec.operation, sparse_dot_mkl, rhs_order),
                    comm=None,
                    repeats=int(args.repeats),
                    min_seconds=float(args.min_seconds),
                    min_iterations=int(args.min_iterations),
                    warmups=int(args.warmups),
                )
                references["timings"][f"scalar_csr_{rhs_order}"] = timing
    finally:
        references["threading_restore"] = rb.restore_sparse_dot_mkl_threading(sparse_dot_mkl, threading)
    return references


def run_case(spec: rb.BenchmarkSpec, args: argparse.Namespace) -> dict[str, Any]:
    block_sizes = rb.make_block_sizes(spec)
    adjacency, adjacency_info, positions, box_lengths = rb.make_geometric_adjacency(spec)
    matrix, build_timings = rb.build_matrix(
        spec,
        block_sizes,
        adjacency,
        positions,
        box_lengths,
        comm=None,
        rank=0,
        size=1,
    )
    inputs = rb.make_inputs(matrix, spec, rank=0)
    vbcsr_timing, vendor = benchmark_vbcsr_with_launches(matrix, inputs, spec, args)

    scalar = rb.scipy_baseline_matrix(matrix)
    mkl_references = benchmark_mkl_references(scalar, inputs, spec, args)
    validation = rb.validate_against_scipy(matrix, scalar, inputs, spec)

    matrix_stats = rb.matrix_statistics(matrix, block_sizes, adjacency, spec, comm=None, rank=0, size=1)
    return {
        "label": spec.label,
        "domain": spec.domain,
        "operation": spec.operation,
        "parameters": {
            "blocks": spec.blocks,
            "target_degree": spec.target_degree,
            "rhs": spec.rhs,
            "dtype": str(spec.dtype),
            "bsr_block_size": spec.bsr_block_size,
            "magnitude_decay_length": spec.magnitude_decay_length,
        },
        "adjacency": adjacency_info,
        "matrix": matrix_stats,
        "build_timings": build_timings,
        "validation": validation,
        "vbcsr": {
            "timing": vbcsr_timing,
            "vendor": vendor,
        },
        "mkl_references": mkl_references,
    }


def flatten_case(case: dict[str, Any]) -> dict[str, Any]:
    vbcsr_seconds = median_seconds(case["vbcsr"]["timing"])
    timings = case["mkl_references"]["timings"]
    primary_c = median_seconds(timings.get("primary_C") or timings.get("primary_native"))
    primary_f = median_seconds(timings.get("primary_F"))
    scalar_csr_c = median_seconds(timings.get("scalar_csr_C") or timings.get("scalar_csr_native"))
    scalar_csr_f = median_seconds(timings.get("scalar_csr_F"))

    def ratio(ref: float | None) -> float | None:
        if vbcsr_seconds is None or ref is None or ref == 0.0:
            return None
        return vbcsr_seconds / ref

    vendor = case["vbcsr"]["vendor"]
    matrix = case["matrix"]
    return {
        "label": case["label"],
        "domain": case["domain"],
        "operation": case["operation"],
        "blocks": case["parameters"]["blocks"],
        "degree_mean": case["adjacency"]["degree_mean"],
        "rhs": case["parameters"]["rhs"],
        "matrix_kind": vendor["matrix_kind"],
        "vendor_backend": vendor["vendor_backend_name"],
        "vendor_launch_delta": vendor["vendor_launch_delta"],
        "total_calls_including_warmups": vendor["total_operation_calls_including_warmups"],
        "vendor_launches_per_call": vendor["vendor_launches_per_call"],
        "page_size": vendor["page_size"],
        "configured_page_size": vendor["configured_page_size"],
        "scalar_rows": matrix["global_scalar_rows"],
        "block_nnz": matrix["global_block_nnz"],
        "scalar_nnz": matrix["global_scalar_nnz"],
        "vbcsr_median_seconds": vbcsr_seconds,
        "mkl_primary_c_or_native_seconds": primary_c,
        "mkl_primary_f_seconds": primary_f,
        "mkl_scalar_csr_c_or_native_seconds": scalar_csr_c,
        "mkl_scalar_csr_f_seconds": scalar_csr_f,
        "vbcsr_over_mkl_primary_c_or_native": ratio(primary_c),
        "vbcsr_over_mkl_primary_f": ratio(primary_f),
        "vbcsr_over_mkl_scalar_csr_c_or_native": ratio(scalar_csr_c),
        "vbcsr_over_mkl_scalar_csr_f": ratio(scalar_csr_f),
        "validation_passed": case["validation"].get("passed"),
        "validation_relative_error": case["validation"].get("relative_error"),
    }


def write_outputs(payload: dict[str, Any], output_dir: Path, label: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{label}.json"
    csv_path = output_dir / f"{label}.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = [flatten_case(case) for case in payload["cases"]]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> int:
    args = make_parser().parse_args()
    domains = parse_list(args.domains, ("csr", "bsr"), "domains")
    operations = parse_list(args.operations, ("spmv", "spmm"), "operations")
    cases = []
    for domain in domains:
        for operation in operations:
            spec = make_spec(args, domain, operation)
            print(f"[{spec.label}] running", flush=True)
            cases.append(run_case(spec, args))
            flat = flatten_case(cases[-1])
            print(
                f"[{spec.label}] VBCSR={flat['vbcsr_median_seconds']:.6g}s "
                f"vendor={flat['vendor_backend']} launches/call={flat['vendor_launches_per_call']:.3g} "
                f"MKL(primary)={flat['mkl_primary_c_or_native_seconds']:.6g}s",
                flush=True,
            )

    label = args.label.strip() if isinstance(args.label, str) and args.label.strip() else None
    if label is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"vendor_path_debug_n{args.blocks}_deg{args.target_degree}_rhs{args.rhs}_{stamp}"
    payload = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "cases": cases,
    }
    json_path, csv_path = write_outputs(payload, args.output_dir, label)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

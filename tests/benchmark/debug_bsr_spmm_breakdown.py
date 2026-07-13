#!/usr/bin/env python3
"""Full BSR SpMM performance decomposition for the publication benchmark case."""

from __future__ import annotations

import argparse
import ctypes
import csv
import datetime as dt
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

import debug_mkl_bsr_layout as raw_bsr
import run_benchmark as rb

from sparse_dot_mkl._mkl_interface import MKL
from sparse_dot_mkl import mkl_get_max_threads, mkl_set_num_threads
try:
    from sparse_dot_mkl import mkl_set_num_threads_local
except ImportError:  # pragma: no cover
    mkl_set_num_threads_local = None
from sparse_dot_mkl._mkl_interface._constants import (
    LAYOUT_CODE_C,
    LAYOUT_CODE_F,
    SPARSE_INDEX_BASE_ONE,
    SPARSE_INDEX_BASE_ZERO,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decompose BSR SpMM slowdown into wrapper, layout, and data-movement costs.")
    parser.add_argument("--blocks", type=int, default=int(os.environ.get("BLOCKS", "4096")))
    parser.add_argument("--target-degree", type=int, default=int(os.environ.get("TARGET_DEGREE", "100")))
    parser.add_argument("--rhs", type=int, default=int(os.environ.get("RHS", "16")))
    parser.add_argument("--bsr-block-size", type=int, default=int(os.environ.get("BSR_BLOCK_SIZE", "8")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "1729")))
    parser.add_argument("--geometry-dim", type=int, default=int(os.environ.get("GEOMETRY_DIM", "3")))
    parser.add_argument("--geometry-spacing", type=float, default=float(os.environ.get("GEOMETRY_SPACING", "1.0")))
    parser.add_argument("--geometry-jitter", type=float, default=float(os.environ.get("GEOMETRY_JITTER", "0.12")))
    parser.add_argument("--geometry-cutoff", type=float, default=None)
    parser.add_argument("--geometry-cutoff-quantile", type=float, default=float(os.environ.get("GEOMETRY_CUTOFF_QUANTILE", "0.90")))
    parser.add_argument("--magnitude-decay-length", type=float, default=float(os.environ.get("MAGNITUDE_DECAY_LENGTH", "0.5")))
    parser.add_argument("--offdiagonal-scale", type=float, default=float(os.environ.get("OFFDIAGONAL_SCALE", "1.0")))
    parser.add_argument("--diagonal-shift", type=float, default=float(os.environ.get("DIAGONAL_SHIFT", "2.0")))
    parser.add_argument("--mkl-threads", type=int, default=int(os.environ.get("SPARSE_DOT_MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "1"))))
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("REPEATS", "5")))
    parser.add_argument("--warmups", type=int, default=int(os.environ.get("WARMUPS", "2")))
    parser.add_argument("--min-seconds", type=float, default=float(os.environ.get("MIN_SECONDS", "0.2")))
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--label", default=os.environ.get("LABEL"))
    return parser


def make_spec(args: argparse.Namespace) -> rb.BenchmarkSpec:
    return rb.BenchmarkSpec(
        suite="bsr-spmm-breakdown",
        domain="bsr",
        operation="spmm",
        blocks=int(args.blocks),
        target_degree=int(args.target_degree),
        rhs=int(args.rhs),
        dtype=np.dtype(np.float64),
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


def timed_once(op: Callable[[], Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    result = op()
    return time.perf_counter() - start, result


def benchmark(op: Callable[[], Any], repeats: int, warmups: int, min_seconds: float) -> dict[str, Any]:
    for _ in range(warmups):
        op()
    samples: list[float] = []
    iterations: list[int] = []
    for _ in range(repeats):
        count = 0
        start = time.perf_counter()
        while True:
            op()
            count += 1
            elapsed = time.perf_counter() - start
            if elapsed >= min_seconds and count >= 1:
                break
        samples.append(elapsed / count)
        iterations.append(count)
    return summarize(samples, iterations)


def summarize(samples: list[float], iterations: list[int] | None = None) -> dict[str, Any]:
    samples_arr = np.asarray(samples, dtype=np.float64)
    return {
        "samples_seconds": [float(item) for item in samples],
        "iterations": iterations or [1 for _ in samples],
        "median_seconds": float(np.median(samples_arr)),
        "min_seconds": float(np.min(samples_arr)),
        "max_seconds": float(np.max(samples_arr)),
        "mean_seconds": float(np.mean(samples_arr)),
        "std_seconds": float(np.std(samples_arr, ddof=1)) if len(samples) > 1 else 0.0,
    }


def relative_error(reference: np.ndarray, observed: np.ndarray) -> float:
    return float(np.linalg.norm(observed - reference) / max(np.linalg.norm(reference), 1e-30))


def make_raw_values(bsr: Any) -> tuple[np.ndarray, np.ndarray]:
    row_major = np.ascontiguousarray(bsr.data)
    col_major = np.ascontiguousarray(
        np.stack([block.ravel(order="F") for block in bsr.data], axis=0).reshape(
            bsr.data.shape[0],
            bsr.blocksize[0],
            bsr.blocksize[1],
        )
    )
    return row_major, col_major


def time_raw_mkl_case(
    bsr: Any,
    values: np.ndarray,
    block_layout: int,
    index_base: int,
    dense_layout: int,
    rhs_array: np.ndarray,
    repeats: int,
    warmups: int,
    min_seconds: float,
    reference: np.ndarray,
    use_mm_hints: bool = True,
) -> dict[str, Any]:
    raw_bsr.configure_raw_mkl_signatures()
    rhs = int(rhs_array.shape[1])
    creation_seconds, handle_buffers = timed_once(
        lambda: raw_bsr.create_bsr_handle(
            bsr,
            values,
            block_layout,
            index_base,
            dense_layout,
            rhs,
            use_mm_hints,
        )
    )
    handle, _buffers = handle_buffers
    y = np.zeros_like(rhs_array, order="C" if dense_layout == LAYOUT_CODE_C else "F")
    try:
        raw_bsr.run_mm(handle, rhs_array, y, dense_layout)
        timing = benchmark(
            lambda: raw_bsr.run_mm(handle, rhs_array, y, dense_layout),
            repeats=repeats,
            warmups=warmups,
            min_seconds=min_seconds,
        )
        status = "ok"
        message = ""
        error = relative_error(reference, np.asarray(y))
    except Exception as exc:
        timing = {}
        status = "unsupported"
        message = f"{type(exc).__name__}: {exc}"
        error = None
    finally:
        raw_bsr.destroy_handle(handle)
    return {
        "status": status,
        "message": message,
        "handle_create_seconds": creation_seconds,
        "timing": timing,
        "relative_error": error,
    }


def main() -> int:
    args = make_parser().parse_args()
    previous_threads = mkl_set_num_threads(int(args.mkl_threads))
    previous_local = None
    if mkl_set_num_threads_local is not None:
        previous_local = mkl_set_num_threads_local(int(args.mkl_threads))

    spec = make_spec(args)
    block_sizes = rb.make_block_sizes(spec)
    adjacency, adjacency_info, positions, box_lengths = rb.make_geometric_adjacency(spec)

    matrix, build_timings = rb.build_matrix(spec, block_sizes, adjacency, positions, box_lengths, None, 0, 1)
    inputs = rb.make_inputs(matrix, spec, rank=0)
    x_view_f = inputs["x_multivector"].to_numpy()
    y_view_f = inputs["y_multivector"].to_numpy()

    # Cold VBCSR call includes vendor cache/MM-handle setup.
    matrix.reset_vendor_launch_count()
    cold_seconds, _ = timed_once(lambda: matrix.mult(inputs["x_multivector"], inputs["y_multivector"]))
    cold_launches = int(matrix.vendor_launch_count)

    matrix.reset_vendor_launch_count()
    vbcsr_timing = benchmark(
        lambda: matrix.mult(inputs["x_multivector"], inputs["y_multivector"]),
        repeats=int(args.repeats),
        warmups=int(args.warmups),
        min_seconds=float(args.min_seconds),
    )
    warm_launches = int(matrix.vendor_launch_count)
    warm_calls = int(sum(vbcsr_timing["iterations"]) + int(args.warmups))
    vbcsr_output = np.asarray(inputs["y_multivector"].to_numpy()).copy()

    scalar = rb.scipy_baseline_matrix(matrix)
    bsr = scalar.tobsr(blocksize=(spec.bsr_block_size, spec.bsr_block_size))
    bsr.sort_indices()
    csr = scalar.tocsr()
    row_values, col_values = make_raw_values(bsr)

    x_c = np.ascontiguousarray(x_view_f)
    x_f = np.asfortranarray(x_view_f)
    reference = bsr.dot(x_c)

    raw_row = time_raw_mkl_case(
        bsr,
        row_values,
        LAYOUT_CODE_C,
        SPARSE_INDEX_BASE_ZERO,
        LAYOUT_CODE_C,
        x_c,
        int(args.repeats),
        int(args.warmups),
        float(args.min_seconds),
        reference,
        use_mm_hints=True,
    )
    raw_vbcsr = time_raw_mkl_case(
        bsr,
        col_values,
        LAYOUT_CODE_F,
        SPARSE_INDEX_BASE_ONE,
        LAYOUT_CODE_F,
        x_f,
        int(args.repeats),
        int(args.warmups),
        float(args.min_seconds),
        reference,
        use_mm_hints=True,
    )

    # The raw layout probe temporarily relaxes ctypes signatures for
    # mkl_sparse_d_create_bsr so it can pass VBCSR-style block-major
    # column-major payloads. Restore sparse_dot_mkl's normal ndarray signatures
    # before timing the package wrapper.
    MKL._set_int_type(ctypes.c_int, np.int32)

    sparse_dot_mkl = importlib.import_module("sparse_dot_mkl")
    mkl_threading = rb.configure_sparse_dot_mkl_threading(sparse_dot_mkl)
    bsr_c_timing = benchmark(
        lambda: sparse_dot_mkl.dot_product_mkl(bsr, x_c),
        repeats=int(args.repeats),
        warmups=int(args.warmups),
        min_seconds=float(args.min_seconds),
    )
    bsr_f_timing = benchmark(
        lambda: sparse_dot_mkl.dot_product_mkl(bsr, x_f),
        repeats=int(args.repeats),
        warmups=int(args.warmups),
        min_seconds=float(args.min_seconds),
    )
    csr_c_timing = benchmark(
        lambda: sparse_dot_mkl.dot_product_mkl(csr, x_c),
        repeats=int(args.repeats),
        warmups=int(args.warmups),
        min_seconds=float(args.min_seconds),
    )
    csr_f_timing = benchmark(
        lambda: sparse_dot_mkl.dot_product_mkl(csr, x_f),
        repeats=int(args.repeats),
        warmups=int(args.warmups),
        min_seconds=float(args.min_seconds),
    )
    threading_restore = rb.restore_sparse_dot_mkl_threading(sparse_dot_mkl, mkl_threading)

    y_c = np.empty_like(x_c)
    movement = {
        "rhs_f_to_c_copy": benchmark(lambda: np.ascontiguousarray(x_view_f), int(args.repeats), int(args.warmups), float(args.min_seconds)),
        "rhs_f_to_f_view_or_copy": benchmark(lambda: np.asfortranarray(x_view_f), int(args.repeats), int(args.warmups), float(args.min_seconds)),
        "output_c_zero": benchmark(lambda: y_c.fill(0.0), int(args.repeats), int(args.warmups), float(args.min_seconds)),
        "output_f_zero_numpy": benchmark(lambda: y_view_f.fill(0.0), int(args.repeats), int(args.warmups), float(args.min_seconds)),
        "output_f_zero_core": benchmark(lambda: inputs["y_multivector"].set_constant(0.0), int(args.repeats), int(args.warmups), float(args.min_seconds)),
        "output_c_to_f_copy": benchmark(lambda: np.copyto(y_view_f, y_c), int(args.repeats), int(args.warmups), float(args.min_seconds)),
        "x_sync_ghosts": benchmark(lambda: inputs["x_multivector"].sync_ghosts(), int(args.repeats), int(args.warmups), float(args.min_seconds)),
    }

    validation = {
        "vbcsr_vs_scipy_relative_error": relative_error(reference, vbcsr_output),
        "raw_row_vs_scipy_relative_error": raw_row["relative_error"],
        "raw_vbcsr_vs_scipy_relative_error": raw_vbcsr["relative_error"],
    }

    rows = [
        {
            "component": "vbcsr_cold_first_call",
            "median_seconds": cold_seconds,
            "status": "ok",
            "notes": f"vendor_launches={cold_launches}",
        },
        {
            "component": "vbcsr_warm_full_wrapper",
            "median_seconds": vbcsr_timing["median_seconds"],
            "status": "ok",
            "notes": f"vendor_launches_per_call={warm_launches / max(warm_calls, 1):.6g}",
        },
        {
            "component": "raw_mkl_sparse_dot_mkl_convention",
            "median_seconds": raw_row.get("timing", {}).get("median_seconds"),
            "status": raw_row["status"],
            "notes": "row-major blocks, base0, row-major dense, mm hints",
        },
        {
            "component": "raw_mkl_vbcsr_convention",
            "median_seconds": raw_vbcsr.get("timing", {}).get("median_seconds"),
            "status": raw_vbcsr["status"],
            "notes": "column-major blocks, base1, column-major dense, mm hints",
        },
        {
            "component": "sparse_dot_mkl_bsr_c_rhs",
            "median_seconds": bsr_c_timing["median_seconds"],
            "status": "ok",
            "notes": "SciPy BSR wrapper, C dense RHS",
        },
        {
            "component": "sparse_dot_mkl_bsr_f_rhs",
            "median_seconds": bsr_f_timing["median_seconds"],
            "status": "ok",
            "notes": "SciPy BSR wrapper, F dense RHS",
        },
        {
            "component": "sparse_dot_mkl_csr_c_rhs",
            "median_seconds": csr_c_timing["median_seconds"],
            "status": "ok",
            "notes": "scalar CSR wrapper, C dense RHS",
        },
        {
            "component": "sparse_dot_mkl_csr_f_rhs",
            "median_seconds": csr_f_timing["median_seconds"],
            "status": "ok",
            "notes": "scalar CSR wrapper, F dense RHS",
        },
    ]
    for name, timing in movement.items():
        rows.append(
            {
                "component": name,
                "median_seconds": timing["median_seconds"],
                "status": "ok",
                "notes": "standalone data-movement probe",
            }
        )

    label = args.label.strip() if isinstance(args.label, str) and args.label.strip() else None
    if label is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"bsr_spmm_breakdown_n{args.blocks}_deg{args.target_degree}_rhs{args.rhs}_{stamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{label}.json"
    csv_path = args.output_dir / f"{label}.csv"

    payload = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "parameters": {
            "blocks": args.blocks,
            "target_degree": args.target_degree,
            "rhs": args.rhs,
            "bsr_block_size": args.bsr_block_size,
            "mkl_threads": args.mkl_threads,
            "mkl_get_max_threads": int(mkl_get_max_threads()),
        },
        "matrix": {
            "shape": list(bsr.shape),
            "block_nnz": int(bsr.indices.size),
            "scalar_nnz": int(bsr.nnz),
            "degree_mean": adjacency_info["degree_mean"],
            "storage_bytes_bsr": rb.sparse_storage_bytes(bsr),
            "storage_bytes_csr": rb.sparse_storage_bytes(csr),
        },
        "build_timings": build_timings,
        "vbcsr_internal": {
            "vendor_backend": matrix.vendor_backend_name,
            "page_size": matrix.page_size,
            "configured_page_size": matrix.configured_page_size,
            "cold_vendor_launches": cold_launches,
            "warm_vendor_launches": warm_launches,
            "warm_calls_including_warmups": warm_calls,
        },
        "mkl_threading": {
            "configure": mkl_threading,
            "restore": threading_restore,
        },
        "validation": validation,
        "raw_mkl": {
            "sparse_dot_mkl_convention": raw_row,
            "vbcsr_convention": raw_vbcsr,
        },
        "timings": {
            "vbcsr_warm": vbcsr_timing,
            "sparse_dot_mkl_bsr_c": bsr_c_timing,
            "sparse_dot_mkl_bsr_f": bsr_f_timing,
            "sparse_dot_mkl_csr_c": csr_c_timing,
            "sparse_dot_mkl_csr_f": csr_f_timing,
            "movement": movement,
        },
        "summary_rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["component", "median_seconds", "status", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    baseline = raw_row.get("timing", {}).get("median_seconds")
    for row in rows:
        value = row["median_seconds"]
        ratio = ""
        if baseline and value:
            ratio = f", vs_raw_row={value / baseline:.3g}"
        print(f"{row['component']}: {value}{ratio} [{row['status']}] {row['notes']}", flush=True)
    print(f"validation: {validation}", flush=True)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")

    if previous_local is not None and mkl_set_num_threads_local is not None:
        mkl_set_num_threads_local(previous_local)
    if previous_threads is not None:
        mkl_set_num_threads(previous_threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

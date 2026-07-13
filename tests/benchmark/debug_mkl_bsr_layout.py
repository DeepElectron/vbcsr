#!/usr/bin/env python3
"""Isolate MKL BSR block-layout and dense-RHS layout costs.

This script does not call VBCSR kernels.  It builds the same geometric BSR
matrix used by the publication benchmark, then calls MKL directly through the
sparse_dot_mkl-loaded ctypes library with four layout combinations:

  row-major BSR blocks    x row-major dense RHS
  row-major BSR blocks    x column-major dense RHS
  column-major BSR blocks x row-major dense RHS
  column-major BSR blocks x column-major dense RHS

The last case matches VBCSR's current BSR MKL convention.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import sparse as sp

import run_benchmark as rb

from sparse_dot_mkl import mkl_get_max_threads, mkl_set_num_threads
try:
    from sparse_dot_mkl import mkl_set_num_threads_local
except ImportError:  # pragma: no cover
    mkl_set_num_threads_local = None
from sparse_dot_mkl._mkl_interface import MKL
from sparse_dot_mkl._mkl_interface import _cfunctions as mkl_cfuncs
from sparse_dot_mkl._mkl_interface._constants import (
    LAYOUT_CODE_C,
    LAYOUT_CODE_F,
    SPARSE_INDEX_BASE_ONE,
    SPARSE_INDEX_BASE_ZERO,
    SPARSE_OPERATION_NON_TRANSPOSE,
)
from sparse_dot_mkl._mkl_interface._structs import matrix_descr, sparse_matrix_t


SCRIPT_DIR = Path(__file__).resolve().parent


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark raw MKL BSR block and dense layout combinations.")
    parser.add_argument("--blocks", type=int, default=int(os.environ.get("BLOCKS", "512")))
    parser.add_argument("--target-degree", type=int, default=int(os.environ.get("TARGET_DEGREE", "64")))
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
    parser.add_argument("--use-mm-hints", action="store_true", help="Apply mkl_sparse_set_mm_hint/optimize before timing")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--label", default=os.environ.get("LABEL"))
    return parser


def make_spec(args: argparse.Namespace) -> rb.BenchmarkSpec:
    return rb.BenchmarkSpec(
        suite="mkl-layout-debug",
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


def build_scipy_bsr(spec: rb.BenchmarkSpec) -> tuple[sp.bsr_matrix, dict[str, Any]]:
    block_sizes = rb.make_block_sizes(spec)
    adjacency, adjacency_info, positions, box_lengths = rb.make_geometric_adjacency(spec)
    block_size = int(spec.bsr_block_size)
    indptr = np.zeros(spec.blocks + 1, dtype=np.int32)
    indices: list[int] = []
    data = np.empty((sum(len(row) for row in adjacency), block_size, block_size), dtype=np.float64)
    slot = 0
    for row, cols in enumerate(adjacency):
        indptr[row] = slot
        for col in cols:
            indices.append(int(col))
            data[slot, :, :] = rb.make_block_data(
                row,
                col,
                block_size,
                block_size,
                spec,
                positions,
                box_lengths,
            )
            slot += 1
    indptr[spec.blocks] = slot
    indices_array = np.asarray(indices, dtype=np.int32)
    shape = (spec.blocks * block_size, spec.blocks * block_size)
    matrix = sp.bsr_matrix((data, indices_array, indptr), shape=shape, blocksize=(block_size, block_size))
    matrix.sort_indices()
    return matrix, {
        "adjacency": adjacency_info,
        "block_nnz": int(slot),
        "scalar_nnz": int(slot * block_size * block_size),
        "shape": list(shape),
        "block_size": block_size,
    }


def configure_raw_mkl_signatures() -> None:
    int_array = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    double_ptr = ctypes.POINTER(ctypes.c_double)
    mkl_cfuncs._libmkl.mkl_sparse_d_create_bsr.argtypes = [
        ctypes.POINTER(sparse_matrix_t),
        ctypes.c_int,
        ctypes.c_int,
        MKL.MKL_INT,
        MKL.MKL_INT,
        MKL.MKL_INT,
        int_array,
        int_array,
        int_array,
        double_ptr,
    ]
    mkl_cfuncs._libmkl.mkl_sparse_d_create_bsr.restype = ctypes.c_int
    mkl_cfuncs._libmkl.mkl_sparse_d_mm.argtypes = MKL._mkl_sparse_d_mm.argtypes
    mkl_cfuncs._libmkl.mkl_sparse_d_mm.restype = ctypes.c_int
    mkl_cfuncs._libmkl.mkl_sparse_destroy.argtypes = [sparse_matrix_t]
    mkl_cfuncs._libmkl.mkl_sparse_destroy.restype = ctypes.c_int
    if hasattr(mkl_cfuncs._libmkl, "mkl_sparse_set_mm_hint"):
        mkl_cfuncs._libmkl.mkl_sparse_set_mm_hint.argtypes = [
            sparse_matrix_t,
            ctypes.c_int,
            matrix_descr,
            ctypes.c_int,
            MKL.MKL_INT,
            MKL.MKL_INT,
        ]
        mkl_cfuncs._libmkl.mkl_sparse_set_mm_hint.restype = ctypes.c_int
    if hasattr(mkl_cfuncs._libmkl, "mkl_sparse_optimize"):
        mkl_cfuncs._libmkl.mkl_sparse_optimize.argtypes = [sparse_matrix_t]
        mkl_cfuncs._libmkl.mkl_sparse_optimize.restype = ctypes.c_int


def create_bsr_handle(
    bsr: sp.bsr_matrix,
    block_values: np.ndarray,
    block_layout: int,
    index_base: int,
    dense_layout: int,
    rhs: int,
    use_mm_hints: bool,
) -> tuple[sparse_matrix_t, list[np.ndarray]]:
    ref = sparse_matrix_t()
    block_size = int(bsr.blocksize[0])
    rows = int(bsr.shape[0] // block_size)
    cols = int(bsr.shape[1] // block_size)
    values = np.ascontiguousarray(block_values, dtype=np.float64)
    base_shift = 1 if index_base == SPARSE_INDEX_BASE_ONE else 0
    row_start = np.ascontiguousarray(bsr.indptr[:-1] + base_shift, dtype=np.int32)
    row_end = np.ascontiguousarray(bsr.indptr[1:] + base_shift, dtype=np.int32)
    col_ind = np.ascontiguousarray(bsr.indices + base_shift, dtype=np.int32)
    status = mkl_cfuncs._libmkl.mkl_sparse_d_create_bsr(
        ctypes.byref(ref),
        ctypes.c_int(index_base),
        ctypes.c_int(block_layout),
        MKL.MKL_INT(rows),
        MKL.MKL_INT(cols),
        MKL.MKL_INT(block_size),
        row_start,
        row_end,
        col_ind,
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if status != 0:
        raise RuntimeError(f"mkl_sparse_d_create_bsr failed with status {status}")
    if use_mm_hints and hasattr(mkl_cfuncs._libmkl, "mkl_sparse_set_mm_hint"):
        mkl_cfuncs._libmkl.mkl_sparse_set_mm_hint(
            ref,
            ctypes.c_int(SPARSE_OPERATION_NON_TRANSPOSE),
            matrix_descr(),
            ctypes.c_int(dense_layout),
            MKL.MKL_INT(rhs),
            MKL.MKL_INT(1),
        )
    if use_mm_hints and hasattr(mkl_cfuncs._libmkl, "mkl_sparse_optimize"):
        mkl_cfuncs._libmkl.mkl_sparse_optimize(ref)
    return ref, [row_start, row_end, col_ind, values]


def destroy_handle(ref: sparse_matrix_t) -> None:
    if ref:
        mkl_cfuncs._libmkl.mkl_sparse_destroy(ref)


def run_mm(handle: sparse_matrix_t, x: np.ndarray, y: np.ndarray, dense_layout: int) -> None:
    rows, rhs = x.shape
    if dense_layout == LAYOUT_CODE_C:
        ldb = rhs
        ldc = rhs
    else:
        ldb = rows
        ldc = rows
    status = mkl_cfuncs._libmkl.mkl_sparse_d_mm(
        ctypes.c_int(SPARSE_OPERATION_NON_TRANSPOSE),
        ctypes.c_double(1.0),
        handle,
        matrix_descr(),
        ctypes.c_int(dense_layout),
        x,
        ctypes.c_int(rhs),
        ctypes.c_int(ldb),
        ctypes.c_double(0.0),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(ldc),
    )
    if status != 0:
        raise RuntimeError(f"mkl_sparse_d_mm failed with status {status}")


def benchmark(op: Callable[[], None], repeats: int, warmups: int, min_seconds: float) -> dict[str, Any]:
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
    sorted_samples = sorted(samples)
    return {
        "samples_seconds": samples,
        "iterations": iterations,
        "median_seconds": float(np.median(samples)),
        "min_seconds": float(sorted_samples[0]),
        "max_seconds": float(sorted_samples[-1]),
        "mean_seconds": float(np.mean(samples)),
        "std_seconds": float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0,
    }


def relative_error(reference: np.ndarray, observed: np.ndarray) -> float:
    return float(np.linalg.norm(observed - reference) / max(np.linalg.norm(reference), 1e-30))


def main() -> int:
    args = make_parser().parse_args()
    if args.mkl_threads <= 0:
        raise ValueError("--mkl-threads must be positive")
    previous_threads = mkl_set_num_threads(int(args.mkl_threads))
    previous_local = None
    if mkl_set_num_threads_local is not None:
        previous_local = mkl_set_num_threads_local(int(args.mkl_threads))

    configure_raw_mkl_signatures()
    spec = make_spec(args)
    bsr, info = build_scipy_bsr(spec)
    n = int(bsr.shape[0])
    rhs = int(args.rhs)
    rng = np.random.default_rng(rb.stable_seed(spec.seed, 911, spec.blocks, rhs))
    x_c = np.ascontiguousarray(rng.standard_normal((n, rhs)))
    x_f = np.asfortranarray(x_c)
    ref = bsr.dot(x_c)

    row_major_values = np.ascontiguousarray(bsr.data)
    col_major_values = np.ascontiguousarray(
        np.stack([block.ravel(order="F") for block in bsr.data], axis=0).reshape(
            bsr.data.shape[0],
            bsr.blocksize[0],
            bsr.blocksize[1],
        )
    )

    cases = [
        ("block_row_base0_dense_row", LAYOUT_CODE_C, SPARSE_INDEX_BASE_ZERO, row_major_values, LAYOUT_CODE_C, x_c),
        ("block_row_base0_dense_col", LAYOUT_CODE_C, SPARSE_INDEX_BASE_ZERO, row_major_values, LAYOUT_CODE_F, x_f),
        ("block_row_base1_dense_row", LAYOUT_CODE_C, SPARSE_INDEX_BASE_ONE, row_major_values, LAYOUT_CODE_C, x_c),
        ("block_row_base1_dense_col", LAYOUT_CODE_C, SPARSE_INDEX_BASE_ONE, row_major_values, LAYOUT_CODE_F, x_f),
        ("block_col_base0_dense_row", LAYOUT_CODE_F, SPARSE_INDEX_BASE_ZERO, col_major_values, LAYOUT_CODE_C, x_c),
        ("block_col_base0_dense_col", LAYOUT_CODE_F, SPARSE_INDEX_BASE_ZERO, col_major_values, LAYOUT_CODE_F, x_f),
        ("block_col_base1_dense_row", LAYOUT_CODE_F, SPARSE_INDEX_BASE_ONE, col_major_values, LAYOUT_CODE_C, x_c),
        ("block_col_base1_dense_col", LAYOUT_CODE_F, SPARSE_INDEX_BASE_ONE, col_major_values, LAYOUT_CODE_F, x_f),
    ]

    rows: list[dict[str, Any]] = []
    handles: list[sparse_matrix_t] = []
    keepalive: list[list[np.ndarray]] = []
    try:
        for name, block_layout, index_base, values, dense_layout, x in cases:
            handle, buffers = create_bsr_handle(bsr, values, block_layout, index_base, dense_layout, rhs, bool(args.use_mm_hints))
            handles.append(handle)
            keepalive.append(buffers)
            y = np.zeros_like(x, order="C" if dense_layout == LAYOUT_CODE_C else "F")
            print(f"testing {name}", flush=True)
            try:
                run_mm(handle, x, y, dense_layout)
                error = relative_error(ref, np.asarray(y))
                timing = benchmark(
                    lambda handle=handle, x=x, y=y, dense_layout=dense_layout: run_mm(handle, x, y, dense_layout),
                    repeats=int(args.repeats),
                    warmups=int(args.warmups),
                    min_seconds=float(args.min_seconds),
                )
                status = "ok"
                message = ""
            except Exception as exc:
                error = None
                timing = {
                    "median_seconds": None,
                    "min_seconds": None,
                    "max_seconds": None,
                    "std_seconds": None,
                    "samples_seconds": [],
                    "iterations": [],
                }
                status = "unsupported"
                message = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "case": name,
                    "block_layout": "row_major" if block_layout == LAYOUT_CODE_C else "column_major",
                    "index_base": "zero" if index_base == SPARSE_INDEX_BASE_ZERO else "one",
                    "dense_layout": "row_major" if dense_layout == LAYOUT_CODE_C else "column_major",
                    "status": status,
                    "message": message,
                    "median_seconds": timing["median_seconds"],
                    "min_seconds": timing["min_seconds"],
                    "max_seconds": timing["max_seconds"],
                    "std_seconds": timing["std_seconds"],
                    "relative_error": error,
                    "samples_seconds": timing["samples_seconds"],
                    "iterations": timing["iterations"],
                }
            )
    finally:
        for handle in handles:
            destroy_handle(handle)
        if previous_local is not None and mkl_set_num_threads_local is not None:
            mkl_set_num_threads_local(previous_local)
        if previous_threads is not None:
            mkl_set_num_threads(previous_threads)

    label = args.label.strip() if isinstance(args.label, str) and args.label.strip() else None
    if label is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"mkl_bsr_layout_n{args.blocks}_deg{args.target_degree}_rhs{args.rhs}_{stamp}"
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
        "matrix": info,
        "cases": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "case",
            "block_layout",
            "index_base",
            "dense_layout",
            "status",
            "message",
            "median_seconds",
            "min_seconds",
            "max_seconds",
            "std_seconds",
            "relative_error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    baseline = next(row["median_seconds"] for row in rows if row["case"] == "block_row_base0_dense_row")
    for row in rows:
        if row["status"] == "ok":
            print(
                f"{row['case']}: {row['median_seconds']:.6g} s, "
                f"vs row/row={row['median_seconds'] / baseline:.3g}, "
                f"relerr={row['relative_error']:.3e}",
                flush=True,
            )
        else:
            print(f"{row['case']}: {row['status']} ({row['message']})", flush=True)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

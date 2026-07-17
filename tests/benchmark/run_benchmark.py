#!/usr/bin/env python3
"""Publication benchmark driver for the VBCSR manuscript.

This script is intentionally narrow. It generates the data required by
doc/main.tex:

1. kernel efficiency for SpMV, SpMM, and SpGEMM across CSR, BSR, and VBCSR
   structural domains;
2. distributed strong/weak scaling for the same operations and domains;
3. reproducibility metadata needed to rerun the figures.

All matrices use an atom-like geometric finite-cutoff graph. There is no random
Erdos-Renyi or compatibility benchmark mode here; this script is for the paper.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib
import io
import json
import math
import os
import platform
import resource
import shutil
import shlex
import socket
import subprocess
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
TESTS_DIR = SCRIPT_DIR.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from _workspace_bootstrap import REPO_ROOT  # noqa: E402


def add_local_build_paths() -> None:
    candidates: list[Path] = []
    if os.environ.get("VBCSR_BUILD_DIR"):
        candidates.append(Path(os.environ["VBCSR_BUILD_DIR"]))
    candidates.extend([REPO_ROOT / "build", REPO_ROOT / "build_dbg"])
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("vbcsr_core*.so")):
            text = str(candidate)
            if text not in sys.path:
                sys.path.insert(0, text)


add_local_build_paths()

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

import vbcsr  # noqa: E402

try:  # noqa: E402
    from mpi4py import MPI
except Exception:  # pragma: no cover - depends on the benchmark environment.
    MPI = None


SCHEMA_VERSION = 6
DOMAINS = ("csr", "bsr", "vbcsr")
OPERATIONS = ("spmv", "spmm", "spgemm")
VBCSR_BLOCK_SIZES = (9, 13, 15, 20)
MPI_FALLBACK_STATUS: dict[str, Any] = {}


@dataclass(frozen=True)
class BenchmarkSpec:
    suite: str
    domain: str
    operation: str
    blocks: int
    target_degree: int
    rhs: int
    dtype: np.dtype
    seed: int
    bsr_block_size: int
    spgemm_threshold: float
    spgemm_audit_limit: int
    geometry_dim: int
    geometry_spacing: float
    geometry_jitter: float
    geometry_cutoff: float | None
    geometry_cutoff_quantile: float
    magnitude_decay_length: float
    offdiagonal_scale: float
    diagonal_shift: float
    weak_blocks_per_rank: int | None = None

    @property
    def label(self) -> str:
        dtype_label = "complex" if self.dtype == np.dtype(np.complex128) else "real"
        threshold_label = ""
        if self.operation == "spgemm":
            threshold_label = f"_thr{format_float_label(self.spgemm_threshold)}"
        return (
            f"{self.suite}_{self.domain}_{self.operation}_"
            f"geom_n{self.blocks}_deg{self.target_degree}_rhs{self.rhs}_{dtype_label}"
            f"{threshold_label}"
        )


@dataclass(frozen=True)
class TimingSummary:
    samples: list[float]
    iterations: list[int]
    total_seconds: list[float]
    warmups: int

    def as_dict(self) -> dict[str, Any]:
        sorted_samples = sorted(self.samples)
        count = len(sorted_samples)
        result: dict[str, Any] = {
            "samples_seconds": self.samples,
            "iterations": self.iterations,
            "total_seconds": self.total_seconds,
            "warmups": self.warmups,
            "repeat_count": count,
        }
        if count == 0:
            return result
        result.update(
            {
                "min_seconds": sorted_samples[0],
                "median_seconds": median(sorted_samples),
                "mean_seconds": mean(sorted_samples),
                "max_seconds": sorted_samples[-1],
                "std_seconds": float(np.std(sorted_samples, ddof=1)) if count > 1 else 0.0,
            }
        )
        return result


def mpi_context() -> tuple[Any, int, int]:
    if MPI is not None:
        comm = MPI.COMM_WORLD
        MPI_FALLBACK_STATUS.update({"mpi4py_available": True})
        return comm, comm.Get_rank(), comm.Get_size()

    env_rank = None
    env_size = None
    for rank_name, size_name in (
        ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
        ("PMI_RANK", "PMI_SIZE"),
        ("PMIX_RANK", "PMIX_SIZE"),
        ("SLURM_PROCID", "SLURM_NTASKS"),
    ):
        if rank_name in os.environ and size_name in os.environ:
            env_rank = int(os.environ[rank_name])
            env_size = int(os.environ[size_name])
            break

    try:
        vbcsr_core = importlib.import_module("vbcsr_core")
        graph = vbcsr_core.DistGraph(None)
        native_rank = int(graph.rank)
        native_size = int(graph.size)
    except Exception as exc:
        native_rank = 0
        native_size = 1
        MPI_FALLBACK_STATUS["native_error"] = f"{type(exc).__name__}: {exc}"

    MPI_FALLBACK_STATUS.update(
        {
            "mpi4py_available": False,
            "environment_rank": env_rank,
            "environment_size": env_size,
            "native_vbcsr_rank": native_rank,
            "native_vbcsr_size": native_size,
        }
    )
    if native_size > 1:
        return None, native_rank, native_size
    if env_size is not None and env_size > 1:
        return None, int(env_rank or 0), int(env_size)
    return None, native_rank, native_size


def barrier(comm: Any) -> None:
    if comm is not None:
        comm.Barrier()


def reduce_value(comm: Any, value: int | float, op_name: str) -> int | float:
    if comm is None:
        return value
    op = {"sum": MPI.SUM, "min": MPI.MIN, "max": MPI.MAX}[op_name]
    return comm.allreduce(value, op=op)


def parse_dtype(value: str) -> np.dtype:
    if value == "real":
        return np.dtype(np.float64)
    if value == "complex":
        return np.dtype(np.complex128)
    raise argparse.ArgumentTypeError("dtype must be 'real' or 'complex'")


def parse_thresholds(value: str | None, fallback: float) -> list[float]:
    if value is None or value.strip() == "":
        return [float(fallback)]
    thresholds = [float(item) for item in value.replace(",", " ").split()]
    if not thresholds:
        raise argparse.ArgumentTypeError("--spgemm-thresholds cannot be empty")
    return thresholds


def format_float_label(value: float) -> str:
    text = f"{value:.3e}"
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def stable_seed(seed: int, *items: int) -> int:
    value = np.uint64(seed) ^ np.uint64(0x9E3779B97F4A7C15)
    for item in items:
        value ^= np.uint64(item + 0x9E3779B9) + (value << np.uint64(6)) + (value >> np.uint64(2))
    return int(value & np.uint64(0x7FFFFFFFFFFFFFFF))


def partition_range(n_items: int, size: int, rank: int) -> tuple[int, int]:
    base = n_items // size
    rem = n_items % size
    start = rank * base + min(rank, rem)
    return start, start + base + (1 if rank < rem else 0)


def owner_of_block(gid: int, blocks: int, size: int) -> int:
    for rank in range(size):
        start, end = partition_range(blocks, size, rank)
        if start <= gid < end:
            return rank
    raise ValueError(f"block id {gid} outside [0, {blocks})")


def geometric_grid_shape(blocks: int, dim: int) -> tuple[int, ...]:
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    if dim <= 0:
        raise ValueError("geometry dimension must be positive")
    side = max(1, int(math.ceil(blocks ** (1.0 / dim))))
    return tuple([side] * dim)


def make_geometric_positions(spec: BenchmarkSpec) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    shape = geometric_grid_shape(spec.blocks, spec.geometry_dim)
    grid_size = math.prod(shape)
    flat = (np.arange(spec.blocks, dtype=np.int64) * grid_size) // spec.blocks
    grid = np.array(np.unravel_index(flat, shape), dtype=np.float64).T
    positions = grid * spec.geometry_spacing
    box_lengths = np.array(shape, dtype=np.float64) * spec.geometry_spacing
    if spec.geometry_jitter > 0.0:
        rng = np.random.default_rng(stable_seed(spec.seed, 503, spec.blocks, spec.geometry_dim))
        jitter = rng.uniform(-spec.geometry_jitter, spec.geometry_jitter, size=positions.shape)
        positions = positions + jitter * spec.geometry_spacing
    return np.mod(positions, box_lengths), box_lengths, shape


def calibrate_cutoff(positions: np.ndarray, box_lengths: np.ndarray, spec: BenchmarkSpec) -> tuple[float, str]:
    if spec.geometry_cutoff is not None:
        if spec.geometry_cutoff <= 0.0:
            raise ValueError("--geometry-cutoff must be positive")
        return float(spec.geometry_cutoff), "explicit"
    if spec.blocks <= 1:
        return np.finfo(float).eps, "single_block"
    if not (0.0 < spec.geometry_cutoff_quantile <= 1.0):
        raise ValueError("--geometry-cutoff-quantile must be in (0, 1]")
    tree = cKDTree(positions, boxsize=box_lengths)
    kth = min(max(spec.target_degree, 2), spec.blocks)
    distances, _ = tree.query(positions, k=kth)
    kth_distances = np.asarray(distances)[:, kth - 1]
    cutoff = float(np.quantile(kth_distances[np.isfinite(kth_distances)], spec.geometry_cutoff_quantile))
    return max(cutoff * 1.000001, np.finfo(float).eps), "calibrated_from_target_degree"


def periodic_distance(row: int, col: int, positions: np.ndarray, box_lengths: np.ndarray) -> float:
    delta = positions[row] - positions[col]
    delta = delta - box_lengths * np.round(delta / box_lengths)
    return float(np.linalg.norm(delta))


def block_magnitude_scale(row: int, col: int, positions: np.ndarray, box_lengths: np.ndarray, spec: BenchmarkSpec) -> float:
    if row == col:
        return 1.0
    if spec.magnitude_decay_length <= 0.0:
        raise ValueError("--magnitude-decay-length must be positive")
    distance = periodic_distance(row, col, positions, box_lengths)
    return float(spec.offdiagonal_scale * math.exp(-distance / spec.magnitude_decay_length))


def matrix_value_model_statistics(
    adjacency: list[list[int]],
    positions: np.ndarray,
    box_lengths: np.ndarray,
    spec: BenchmarkSpec,
) -> dict[str, Any]:
    offdiag_distances: list[float] = []
    offdiag_scales: list[float] = []
    for row, cols in enumerate(adjacency):
        for col in cols:
            if row == col:
                continue
            distance = periodic_distance(row, col, positions, box_lengths)
            offdiag_distances.append(distance)
            offdiag_scales.append(float(spec.offdiagonal_scale * math.exp(-distance / spec.magnitude_decay_length)))

    distances = np.asarray(offdiag_distances, dtype=np.float64)
    scales = np.asarray(offdiag_scales, dtype=np.float64)
    result: dict[str, Any] = {
        "model": "exponential_distance_decay_with_onsite_shift",
        "distance_metric": "periodic_minimum_image",
        "magnitude_decay_length": spec.magnitude_decay_length,
        "offdiagonal_scale": spec.offdiagonal_scale,
        "diagonal_shift": spec.diagonal_shift,
        "diagonal_random_scale": 1.0,
        "block_random_normalization": "1/sqrt(row_block_size * column_block_size)",
    }
    if distances.size:
        result.update(
            {
                "offdiagonal_distance_min": float(distances.min()),
                "offdiagonal_distance_mean": float(distances.mean()),
                "offdiagonal_distance_max": float(distances.max()),
                "offdiagonal_magnitude_scale_min": float(scales.min()),
                "offdiagonal_magnitude_scale_mean": float(scales.mean()),
                "offdiagonal_magnitude_scale_max": float(scales.max()),
            }
        )
    return result


def make_geometric_adjacency(
    spec: BenchmarkSpec,
) -> tuple[list[list[int]], dict[str, Any], np.ndarray, np.ndarray]:
    if spec.geometry_spacing <= 0.0:
        raise ValueError("--geometry-spacing must be positive")
    if spec.geometry_jitter < 0.0:
        raise ValueError("--geometry-jitter must be non-negative")
    positions, box_lengths, grid_shape = make_geometric_positions(spec)
    cutoff, cutoff_source = calibrate_cutoff(positions, box_lengths, spec)
    tree = cKDTree(positions, boxsize=box_lengths)
    raw = tree.query_ball_point(positions, r=cutoff)
    adjacency = [sorted({int(col) for col in cols} | {row}) for row, cols in enumerate(raw)]
    degrees = np.array([len(row) for row in adjacency], dtype=np.int64)
    reciprocal = 0
    adjacency_sets = [set(row) for row in adjacency]
    for row, cols in enumerate(adjacency_sets):
        reciprocal += sum(1 for col in cols if row in adjacency_sets[col])
    directed_edges = int(degrees.sum())
    return adjacency, {
        "model": "geometric_finite_cutoff",
        "dimension": spec.geometry_dim,
        "grid_shape": list(grid_shape),
        "periodic": True,
        "spacing": spec.geometry_spacing,
        "jitter_fraction_of_spacing": spec.geometry_jitter,
        "cutoff": cutoff,
        "cutoff_source": cutoff_source,
        "cutoff_quantile": spec.geometry_cutoff_quantile,
        "degree_min": int(degrees.min()) if degrees.size else 0,
        "degree_max": int(degrees.max()) if degrees.size else 0,
        "degree_mean": float(degrees.mean()) if degrees.size else 0.0,
        "degree_std": float(degrees.std()) if degrees.size else 0.0,
        "directed_block_edges": directed_edges,
        "reciprocal_directed_edge_fraction": float(reciprocal / max(directed_edges, 1)),
    }, positions, box_lengths


def make_block_sizes(spec: BenchmarkSpec) -> list[int]:
    if spec.domain == "csr":
        return [1] * spec.blocks
    if spec.domain == "bsr":
        return [spec.bsr_block_size] * spec.blocks
    if spec.domain == "vbcsr":
        rng = np.random.default_rng(stable_seed(spec.seed, 101, spec.blocks))
        return rng.choice(np.array(VBCSR_BLOCK_SIZES, dtype=np.int32), size=spec.blocks).astype(int).tolist()
    raise ValueError(f"unknown domain {spec.domain!r}")


def make_block_data(
    row: int,
    col: int,
    row_dim: int,
    col_dim: int,
    spec: BenchmarkSpec,
    positions: np.ndarray,
    box_lengths: np.ndarray,
) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(spec.seed, 307, row, col, row_dim, col_dim))
    scale = 1.0 / math.sqrt(max(row_dim * col_dim, 1))
    magnitude = block_magnitude_scale(row, col, positions, box_lengths, spec)
    real = rng.standard_normal((row_dim, col_dim)) * scale * magnitude
    data = real
    if spec.dtype == np.dtype(np.complex128):
        data = real + 1j * rng.standard_normal((row_dim, col_dim)) * scale * magnitude
    if row == col and row_dim == col_dim:
        data = data + np.eye(row_dim, dtype=spec.dtype) * (spec.diagonal_shift + 0.01 * (row % 17))
    return np.ascontiguousarray(data, dtype=spec.dtype)


def build_matrix(
    spec: BenchmarkSpec,
    block_sizes: list[int],
    adjacency: list[list[int]],
    positions: np.ndarray,
    box_lengths: np.ndarray,
    comm: Any,
    rank: int,
    size: int,
) -> tuple[vbcsr.VBCSR, dict[str, float]]:
    start, end = partition_range(spec.blocks, size, rank)
    owned = list(range(start, end))
    local_sizes = [block_sizes[idx] for idx in owned]
    local_adj = [adjacency[idx] for idx in owned]

    barrier(comm)
    t0 = time.perf_counter()
    matrix = vbcsr.VBCSR.create_distributed(
        owned_indices=owned,
        block_sizes=local_sizes,
        adjacency=local_adj,
        dtype=spec.dtype,
        comm=comm,
    )
    barrier(comm)
    graph_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    for row in owned:
        row_dim = block_sizes[row]
        for col in adjacency[row]:
            matrix.add_block(row, col, make_block_data(row, col, row_dim, block_sizes[col], spec, positions, box_lengths))
    matrix.assemble()
    barrier(comm)
    assembly_seconds = time.perf_counter() - t1

    return matrix, {
        "graph_construction_seconds": float(reduce_value(comm, graph_seconds, "max")),
        "assembly_seconds": float(reduce_value(comm, assembly_seconds, "max")),
    }


def atomic_numbers_and_type_norb(block_sizes: list[int], spec: BenchmarkSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if spec.domain == "csr":
        z_by_size = {1: 1}
    elif spec.domain == "bsr":
        z_by_size = {spec.bsr_block_size: 6}
    else:
        z_by_size = {9: 1, 13: 6, 15: 8, 20: 14}

    atomic_numbers = np.array([z_by_size[int(size)] for size in block_sizes], dtype=np.int32)
    unique_z = np.array(sorted(set(int(z) for z in atomic_numbers)), dtype=np.int32)
    type_norb = np.array(
        [next(int(size) for size, z in z_by_size.items() if z == int(atomic_number)) for atomic_number in unique_z],
        dtype=np.int32,
    )
    return atomic_numbers, unique_z, type_norb


def benchmark_atomistic_conversion(
    spec: BenchmarkSpec,
    block_sizes: list[int],
    adjacency: list[list[int]],
    positions: np.ndarray,
    box_lengths: np.ndarray,
    cutoff: float,
    cutoff_source: str,
    comm: Any,
    rank: int,
    size: int,
) -> dict[str, Any]:
    atomic_numbers, unique_z, type_norb = atomic_numbers_and_type_norb(block_sizes, spec)
    cell = np.diag(box_lengths).astype(np.float64)

    start, end = partition_range(spec.blocks, size, rank)
    local_edge_index = np.array(
        [[row, col] for row in range(start, end) for col in adjacency[row]],
        dtype=np.int32,
    ).reshape(-1, 2)
    local_edge_shift = np.zeros((local_edge_index.shape[0], 3), dtype=np.int32)
    z_to_type = {int(atomic_number): idx for idx, atomic_number in enumerate(unique_z)}
    local_atom_type = np.array([z_to_type[int(z)] for z in atomic_numbers[start:end]], dtype=np.int32)
    global_edge_count = sum(len(row) for row in adjacency)

    barrier(comm)
    t0 = time.perf_counter()
    atoms = vbcsr.AtomicData.from_graph_arrays(
        end - start,
        spec.blocks,
        start,
        local_edge_index.shape[0],
        global_edge_count,
        np.arange(start, end, dtype=np.int32),
        local_atom_type,
        local_edge_index,
        type_norb,
        local_edge_shift,
        cell,
        positions[start:end],
        atomic_numbers=atomic_numbers[start:end],
        comm=comm,
    )
    barrier(comm)
    atomic_seconds = float(reduce_value(comm, time.perf_counter() - t0, "max"))

    barrier(comm)
    t1 = time.perf_counter()
    matrix = vbcsr.VBCSR(atoms.graph, dtype=spec.dtype, comm=comm)
    barrier(comm)
    matrix_facade_seconds = float(reduce_value(comm, time.perf_counter() - t1, "max"))

    local_edges = int(atoms.edge_index.shape[0])
    return {
        "atomic_data_construction_seconds": atomic_seconds,
        "atomic_graph_to_matrix_facade_seconds": matrix_facade_seconds,
        "cutoff": cutoff,
        "cutoff_source": cutoff_source,
        "unique_atomic_numbers": unique_z.astype(int).tolist(),
        "type_norb": type_norb.astype(int).tolist(),
        "local_atom_count_sum": int(reduce_value(comm, int(atoms.n_atom), "sum")),
        "local_edge_count_sum": int(reduce_value(comm, local_edges, "sum")),
        "matrix_kind_from_atomic_graph": matrix.matrix_kind,
        "note": "Measures AtomicData construction from the benchmark graph arrays and VBCSR facade creation from AtomicData.graph.",
    }


def make_inputs(matrix: vbcsr.VBCSR, spec: BenchmarkSpec, rank: int) -> dict[str, Any]:
    rng = np.random.default_rng(stable_seed(spec.seed, 401, rank, spec.blocks))
    vector = matrix.create_vector()
    x = rng.standard_normal(vector.local_size)
    if spec.dtype == np.dtype(np.complex128):
        x = x + 1j * rng.standard_normal(vector.local_size)
    vector.from_numpy(np.asarray(x, dtype=spec.dtype))

    multivector = matrix.create_multivector(spec.rhs)
    x_dense = rng.standard_normal((multivector.local_rows, spec.rhs))
    if spec.dtype == np.dtype(np.complex128):
        x_dense = x_dense + 1j * rng.standard_normal((multivector.local_rows, spec.rhs))
    multivector.from_numpy(np.asarray(x_dense, dtype=spec.dtype))

    return {
        "x_vector": vector,
        "y_vector": matrix.create_vector(),
        "x_multivector": multivector,
        "y_multivector": matrix.create_multivector(spec.rhs),
    }


def benchmark_once(
    op: Callable[[], Any],
    *,
    comm: Any,
    min_seconds: float,
    min_iterations: int,
    warmups: int,
) -> tuple[float, int, float]:
    barrier(comm)
    for _ in range(warmups):
        op()
    barrier(comm)
    start = time.perf_counter()
    iterations = 0
    while True:
        op()
        iterations += 1
        elapsed = time.perf_counter() - start
        keep_going = elapsed < min_seconds or iterations < min_iterations
        if comm is not None:
            keep_going = bool(comm.allreduce(int(keep_going), op=MPI.MAX))
        if not keep_going:
            break
    barrier(comm)
    total = float(reduce_value(comm, time.perf_counter() - start, "max"))
    return total / max(iterations, 1), iterations, total


def benchmark_repeated(
    op: Callable[[], Any],
    *,
    comm: Any,
    repeats: int,
    min_seconds: float,
    min_iterations: int,
    warmups: int,
) -> dict[str, Any]:
    samples: list[float] = []
    iterations: list[int] = []
    totals: list[float] = []
    for _ in range(repeats):
        seconds, n_iter, total = benchmark_once(
            op,
            comm=comm,
            min_seconds=min_seconds,
            min_iterations=min_iterations,
            warmups=warmups,
        )
        samples.append(seconds)
        iterations.append(n_iter)
        totals.append(total)
    return TimingSummary(samples, iterations, totals, warmups).as_dict()


def sparse_storage_bytes(matrix: sp.spmatrix) -> int:
    total = int(matrix.data.nbytes)
    for attr in ("indices", "indptr"):
        value = getattr(matrix, attr, None)
        if value is not None:
            total += int(value.nbytes)
    return total


def scipy_baseline_matrix(matrix: vbcsr.VBCSR) -> sp.csr_matrix:
    scalar = matrix.to_scipy(format="csr")
    scalar.sort_indices()
    return scalar


def mkl_baseline_matrix(scalar_csr: sp.csr_matrix, spec: BenchmarkSpec) -> tuple[sp.spmatrix, dict[str, Any]]:
    info: dict[str, Any] = {"format": "csr", "blocksize": None}
    if spec.domain == "bsr" and scalar_csr.shape[0] % spec.bsr_block_size == 0:
        info["format"] = "bsr"
        info["blocksize"] = [spec.bsr_block_size, spec.bsr_block_size]
        return scalar_csr.tobsr(blocksize=(spec.bsr_block_size, spec.bsr_block_size)), info
    return scalar_csr, info


def positive_thread_count_from_env(names: tuple[str, ...], default: int, default_source: str) -> tuple[int, str]:
    for name in names:
        value = os.environ.get(name)
        if value is None or value.strip() == "":
            continue
        try:
            threads = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
        if threads <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
        return threads, name
    return default, default_source


def sparse_dot_mkl_thread_request() -> tuple[int, str]:
    return positive_thread_count_from_env(("SPARSE_DOT_MKL_NUM_THREADS", "MKL_REFERENCE_NUM_THREADS", "MKL_NUM_THREADS"), 1, "benchmark_default")


def nested_blas_mkl_thread_request() -> tuple[int, str]:
    return positive_thread_count_from_env(("VBCSR_NESTED_MKL_NUM_THREADS", "MKL_NUM_THREADS"), 1, "benchmark_default")


def apply_sparse_dot_mkl_threading(sparse_dot_mkl: Any, threads: int, source: str, policy: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "policy": policy,
        "requested_threads": threads,
        "request_source": source,
        "omp_num_threads_ignored_for_mkl_baseline": True,
    }

    get_max_threads = getattr(sparse_dot_mkl, "mkl_get_max_threads", None)
    if get_max_threads is not None:
        try:
            info["max_threads_before"] = int(get_max_threads())
        except Exception as exc:
            info["max_threads_before_error"] = f"{type(exc).__name__}: {exc}"

    for setter_name in ("mkl_set_num_threads", "mkl_set_num_threads_local"):
        setter = getattr(sparse_dot_mkl, setter_name, None)
        key = setter_name.replace("mkl_", "")
        if setter is None:
            info[key] = {"available": False}
            continue
        try:
            previous = setter(threads)
            info[key] = {"available": True, "ok": True, "return_value": previous}
        except Exception as exc:
            info[key] = {"available": True, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if get_max_threads is not None:
        try:
            info["max_threads_after"] = int(get_max_threads())
        except Exception as exc:
            info["max_threads_after_error"] = f"{type(exc).__name__}: {exc}"
    global_setter = info.get("set_num_threads", {})
    reported_after = info.get("max_threads_after")
    info["effective_thread_count_verified"] = reported_after is not None
    info["thread_control_ok"] = bool(global_setter.get("ok") and (reported_after is None or reported_after == threads))
    return info


def configure_sparse_dot_mkl_threading(sparse_dot_mkl: Any) -> dict[str, Any]:
    threads, source = sparse_dot_mkl_thread_request()
    return apply_sparse_dot_mkl_threading(sparse_dot_mkl, threads, source, "explicit_mkl_sparse_reference_thread_control")


def restore_sparse_dot_mkl_threading(sparse_dot_mkl: Any, configured: dict[str, Any]) -> dict[str, Any]:
    threads, source = nested_blas_mkl_thread_request()
    restored = apply_sparse_dot_mkl_threading(sparse_dot_mkl, threads, source, "restore_nested_blas_mkl_thread_control")
    previous_local = configured.get("set_num_threads_local", {}).get("return_value")
    if isinstance(previous_local, int):
        setter = getattr(sparse_dot_mkl, "mkl_set_num_threads_local", None)
        if setter is not None:
            try:
                setter(previous_local)
                restored["restored_previous_local_thread_setting"] = previous_local
            except Exception as exc:
                restored["restore_previous_local_thread_setting_error"] = f"{type(exc).__name__}: {exc}"
    return restored


def vbcsr_op(matrix: vbcsr.VBCSR, inputs: dict[str, Any], spec: BenchmarkSpec) -> Callable[[], Any]:
    if spec.operation == "spmv":
        return lambda: matrix.mult(inputs["x_vector"], inputs["y_vector"])
    if spec.operation == "spmm":
        return lambda: matrix.mult(inputs["x_multivector"], inputs["y_multivector"])
    if spec.operation == "spgemm":
        return lambda: matrix.spmm(matrix, spec.spgemm_threshold)
    raise ValueError(spec.operation)


def benchmark_vbcsr_with_internal_diagnostics(
    matrix: vbcsr.VBCSR,
    inputs: dict[str, Any],
    spec: BenchmarkSpec,
    *,
    comm: Any,
    repeats: int,
    min_seconds: float,
    min_iterations: int,
    warmups: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    diagnostic: dict[str, Any] = {
        "matrix_kind": matrix.matrix_kind,
        "page_size": int(matrix.page_size),
        "configured_page_size": int(matrix.configured_page_size),
        "vendor_debug_available": False,
    }

    try:
        diagnostic["vendor_backend_name"] = matrix.vendor_backend_name
        matrix.reset_vendor_launch_count()
        before = int(matrix.vendor_launch_count)
        diagnostic["vendor_debug_available"] = True
    except Exception as exc:
        diagnostic["vendor_debug_error"] = f"{type(exc).__name__}: {exc}"
        timing = benchmark_repeated(
            vbcsr_op(matrix, inputs, spec),
            comm=comm,
            repeats=repeats,
            min_seconds=min_seconds,
            min_iterations=min_iterations,
            warmups=warmups,
        )
        return timing, diagnostic

    vbcsr_threading: dict[str, Any] | None = None
    sparse_dot_mkl_for_threading: Any | None = None
    if (
        spec.domain == "csr"
        and spec.operation == "spgemm"
        and diagnostic.get("vendor_backend_name") == "mkl"
        and comm is None
    ):
        try:
            sparse_dot_mkl_for_threading = importlib.import_module("sparse_dot_mkl")
            vbcsr_threading = configure_sparse_dot_mkl_threading(sparse_dot_mkl_for_threading)
            diagnostic["vendor_threading"] = vbcsr_threading
        except Exception as exc:
            diagnostic["vendor_threading_error"] = f"{type(exc).__name__}: {exc}"

    try:
        timing = benchmark_repeated(
            vbcsr_op(matrix, inputs, spec),
            comm=comm,
            repeats=repeats,
            min_seconds=min_seconds,
            min_iterations=min_iterations,
            warmups=warmups,
        )
    finally:
        if sparse_dot_mkl_for_threading is not None and vbcsr_threading is not None:
            diagnostic["vendor_threading_restore"] = restore_sparse_dot_mkl_threading(
                sparse_dot_mkl_for_threading,
                vbcsr_threading,
            )

    after = int(matrix.vendor_launch_count)
    local_calls = int(sum(int(item) for item in timing.get("iterations", []))) + int(repeats) * int(warmups)
    local_delta = after - before
    global_calls = int(reduce_value(comm, local_calls, "sum"))
    global_delta = int(reduce_value(comm, local_delta, "sum"))
    diagnostic.update(
        {
            "vendor_launch_count_before": before,
            "vendor_launch_count_after": after,
            "vendor_launch_delta_local": local_delta,
            "operation_calls_including_warmups_local": local_calls,
            "vendor_launch_delta_sum": global_delta,
            "operation_calls_including_warmups_sum": global_calls,
            "vendor_launches_per_call": float(global_delta / global_calls) if global_calls > 0 else None,
        }
    )
    return timing, diagnostic


def scipy_op(scalar: sp.csr_matrix, inputs: dict[str, Any], spec: BenchmarkSpec) -> Callable[[], Any]:
    if spec.operation == "spmv":
        rhs = inputs["x_vector"].to_numpy().copy()
        return lambda: scalar.dot(rhs)
    if spec.operation == "spmm":
        rhs = inputs["x_multivector"].to_numpy().copy()
        return lambda: scalar.dot(rhs)
    if spec.operation == "spgemm":
        return lambda: scalar.dot(scalar)
    raise ValueError(spec.operation)


def mkl_op(matrix: sp.spmatrix, inputs: dict[str, Any], spec: BenchmarkSpec, sparse_dot_mkl: Any) -> Callable[[], Any]:
    if spec.operation == "spmv":
        rhs = inputs["x_vector"].to_numpy().copy()
        return lambda: sparse_dot_mkl.dot_product_mkl(matrix, rhs)
    if spec.operation == "spmm":
        rhs = np.ascontiguousarray(inputs["x_multivector"].to_numpy().copy())
        return lambda: sparse_dot_mkl.dot_product_mkl(matrix, rhs)
    if spec.operation == "spgemm":
        return lambda: sparse_dot_mkl.dot_product_mkl(matrix, matrix)
    raise ValueError(spec.operation)


def relative_error(reference: Any, observed: Any) -> float:
    if sp.issparse(reference) or sp.issparse(observed):
        ref = reference.tocsr()
        obs = observed.tocsr()
        diff = (obs - ref).tocsr()
        return float(np.linalg.norm(diff.data) / max(np.linalg.norm(ref.data), 1e-30))
    ref_arr = np.asarray(reference)
    obs_arr = np.asarray(observed)
    return float(np.linalg.norm(obs_arr - ref_arr) / max(np.linalg.norm(ref_arr), 1e-30))


def validate_against_scipy(matrix: vbcsr.VBCSR, scalar: sp.csr_matrix, inputs: dict[str, Any], spec: BenchmarkSpec) -> dict[str, Any]:
    if spec.operation == "spmv":
        matrix.mult(inputs["x_vector"], inputs["y_vector"])
        observed = inputs["y_vector"].to_numpy().copy()
        reference = scalar.dot(inputs["x_vector"].to_numpy().copy())
    elif spec.operation == "spmm":
        matrix.mult(inputs["x_multivector"], inputs["y_multivector"])
        observed = inputs["y_multivector"].to_numpy().copy()
        reference = scalar.dot(inputs["x_multivector"].to_numpy().copy())
    else:
        observed = matrix.spmm(matrix, spec.spgemm_threshold).to_scipy(format="csr")
        reference = scalar.dot(scalar)
    error = relative_error(reference, observed)
    tolerance = 1e-9 if spec.dtype == np.dtype(np.complex128) else 1e-10
    exact_required = not (spec.operation == "spgemm" and spec.spgemm_threshold > 0.0)
    return {
        "reference": "scipy_csr_exact_product",
        "relative_error": error,
        "tolerance": tolerance,
        "passed": bool(error <= tolerance),
        "exact_validation_required": exact_required,
        "role": "correctness_check" if exact_required else "threshold_accuracy_measurement",
    }


def get_current_rss_bytes() -> int:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def get_ru_maxrss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def matrix_statistics(
    matrix: vbcsr.VBCSR,
    block_sizes: list[int],
    adjacency: list[list[int]],
    spec: BenchmarkSpec,
    comm: Any,
    rank: int,
    size: int,
) -> dict[str, Any]:
    vector = matrix.create_vector()
    multivector = matrix.create_multivector(spec.rhs)
    local_storage = int(matrix.get_values().nbytes + matrix.row_ptr.nbytes + matrix.col_ind.nbytes)
    graph = matrix.graph
    recv_counts_scalar = getattr(graph, "recv_counts_scalar", None)
    send_counts_scalar = getattr(graph, "send_counts_scalar", None)
    if recv_counts_scalar is not None and send_counts_scalar is not None:
        local_recv_scalar = int(sum(int(item) for item in recv_counts_scalar))
        local_send_scalar = int(sum(int(item) for item in send_counts_scalar))
        communication_source = "DistGraph send_counts_scalar/recv_counts_scalar"
    else:
        local_recv_scalar = int(vector.ghost_size)
        local_send_scalar = int(vector.ghost_size)
        communication_source = "DistVector ghost_size fallback"
    local_cross_rank_edges = 0
    for row, cols in enumerate(adjacency):
        if owner_of_block(row, spec.blocks, size) != rank:
            continue
        row_owner = owner_of_block(row, spec.blocks, size)
        local_cross_rank_edges += sum(1 for col in cols if owner_of_block(col, spec.blocks, size) != row_owner)

    block_nnz = sum(len(row) for row in adjacency)
    scalar_nnz = sum(block_sizes[row] * sum(block_sizes[col] for col in cols) for row, cols in enumerate(adjacency))
    return {
        "global_shape": list(matrix.shape),
        "matrix_kind": matrix.matrix_kind,
        "global_blocks": spec.blocks,
        "global_scalar_rows": int(sum(block_sizes)),
        "global_block_nnz": int(block_nnz),
        "global_scalar_nnz": int(scalar_nnz),
        "block_density": float(block_nnz / max(spec.blocks * spec.blocks, 1)),
        "block_sizes": {"min": min(block_sizes), "max": max(block_sizes), "unique": sorted(set(block_sizes))},
        "local": {
            "reduction_scope": "mpi4py_allreduce" if comm is not None else ("single_rank" if size == 1 else "local_rank_only_no_mpi4py"),
            "owned_scalar_rows_min": int(reduce_value(comm, vector.local_size, "min")),
            "owned_scalar_rows_max": int(reduce_value(comm, vector.local_size, "max")),
            "ghost_scalar_rows_sum": int(reduce_value(comm, vector.ghost_size, "sum")),
            "ghost_scalar_rows_max": int(reduce_value(comm, vector.ghost_size, "max")),
            "ghost_multivector_rows_sum": int(reduce_value(comm, multivector.ghost_rows, "sum")),
            "cross_rank_block_edges_sum": int(reduce_value(comm, local_cross_rank_edges, "sum")),
            "estimated_vbcsr_storage_bytes_sum": int(reduce_value(comm, local_storage, "sum")),
            "estimated_vbcsr_storage_bytes_max": int(reduce_value(comm, local_storage, "max")),
            "rss_bytes_max": int(reduce_value(comm, get_current_rss_bytes(), "max")),
            "ru_maxrss_bytes_max": int(reduce_value(comm, get_ru_maxrss_bytes(), "max")),
        },
        "communication_estimate": {
            "spmv_send_bytes_sum": int(reduce_value(comm, local_send_scalar * spec.dtype.itemsize, "sum")),
            "spmv_recv_bytes_sum": int(reduce_value(comm, local_recv_scalar * spec.dtype.itemsize, "sum")),
            "spmm_send_bytes_sum": int(reduce_value(comm, local_send_scalar * spec.rhs * spec.dtype.itemsize, "sum")),
            "spmm_recv_bytes_sum": int(reduce_value(comm, local_recv_scalar * spec.rhs * spec.dtype.itemsize, "sum")),
            "source": communication_source,
        },
    }


def spgemm_candidate_count(adjacency: list[list[int]]) -> int:
    return int(sum(len(adjacency[inner]) for row in adjacency for inner in row))


def spgemm_threshold_audit(
    adjacency: list[list[int]],
    block_sizes: list[int],
    spec: BenchmarkSpec,
    positions: np.ndarray,
    box_lengths: np.ndarray,
) -> dict[str, Any]:
    candidate_count = spgemm_candidate_count(adjacency)
    if candidate_count > spec.spgemm_audit_limit:
        return {"audited": False, "candidate_block_products": candidate_count, "audit_limit": spec.spgemm_audit_limit}
    exact_product_below_threshold = 0
    row_scaled_product_skips = 0
    for row, inners in enumerate(adjacency):
        a_row_dim = block_sizes[row]
        row_eps = spec.spgemm_threshold / max(1, len(inners))
        for inner in inners:
            a = make_block_data(row, inner, a_row_dim, block_sizes[inner], spec, positions, box_lengths)
            norm_a = float(np.linalg.norm(a))
            for col in adjacency[inner]:
                b = make_block_data(inner, col, block_sizes[inner], block_sizes[col], spec, positions, box_lengths)
                norm_b = float(np.linalg.norm(b))
                if norm_a * norm_b < row_eps:
                    row_scaled_product_skips += 1
                if float(np.linalg.norm(a @ b)) <= spec.spgemm_threshold:
                    exact_product_below_threshold += 1
    return {
        "audited": True,
        "candidate_block_products": candidate_count,
        "row_scaled_product_skip_count": row_scaled_product_skips,
        "exact_product_norm_below_threshold_count": exact_product_below_threshold,
        "audit_limit": spec.spgemm_audit_limit,
        "native_rule": "norm(A_ik) * norm(B_kj) < threshold / max(1, row_block_count)",
        "native_skip_counter_exposed": False,
    }


def run_case(
    spec: BenchmarkSpec,
    *,
    comm: Any,
    rank: int,
    size: int,
    repeats: int,
    min_seconds: float,
    min_iterations: int,
    warmups: int,
    require_mkl: bool,
) -> dict[str, Any]:
    block_sizes = make_block_sizes(spec)
    adjacency, adjacency_info, positions, box_lengths = make_geometric_adjacency(spec)
    value_model = matrix_value_model_statistics(adjacency, positions, box_lengths, spec)
    matrix, build_timings = build_matrix(spec, block_sizes, adjacency, positions, box_lengths, comm, rank, size)
    atomistic_conversion = benchmark_atomistic_conversion(
        spec,
        block_sizes,
        adjacency,
        positions,
        box_lengths,
        float(adjacency_info["cutoff"]),
        str(adjacency_info["cutoff_source"]),
        comm,
        rank,
        size,
    )
    inputs = make_inputs(matrix, spec, rank)

    result: dict[str, Any] = {
        "label": spec.label,
        "suite": spec.suite,
        "rank_count": size,
        "domain": spec.domain,
        "operation": spec.operation,
        "parameters": {
            "blocks": spec.blocks,
            "weak_blocks_per_rank": spec.weak_blocks_per_rank,
            "target_degree": spec.target_degree,
            "rhs": spec.rhs,
            "dtype": str(spec.dtype),
            "seed": spec.seed,
            "bsr_block_size": spec.bsr_block_size,
            "vbcsr_block_sizes": list(VBCSR_BLOCK_SIZES),
            "spgemm_threshold": spec.spgemm_threshold,
            "geometry_dim": spec.geometry_dim,
            "geometry_spacing": spec.geometry_spacing,
            "geometry_jitter": spec.geometry_jitter,
            "geometry_cutoff": spec.geometry_cutoff,
            "geometry_cutoff_quantile": spec.geometry_cutoff_quantile,
            "magnitude_decay_length": spec.magnitude_decay_length,
            "offdiagonal_scale": spec.offdiagonal_scale,
            "diagonal_shift": spec.diagonal_shift,
        },
        "adjacency": adjacency_info,
        "matrix_value_model": value_model,
        "atomistic_conversion": atomistic_conversion,
        "timings": dict(build_timings),
        "matrix": matrix_statistics(matrix, block_sizes, adjacency, spec, comm, rank, size),
        "baselines": {},
        "validation": {"checked": False},
        "speedups": {},
    }

    result["timings"]["vbcsr"], result["vbcsr_internal"] = benchmark_vbcsr_with_internal_diagnostics(
        matrix,
        inputs,
        spec,
        comm=comm,
        repeats=repeats,
        min_seconds=min_seconds,
        min_iterations=min_iterations,
        warmups=warmups,
    )

    scalar: sp.csr_matrix | None = None
    if size == 1:
        t0 = time.perf_counter()
        scalar = scipy_baseline_matrix(matrix)
        result["timings"]["scipy_csr_build_seconds"] = time.perf_counter() - t0
        result["baselines"]["scipy_csr"] = {
            "available": True,
            "shape": list(scalar.shape),
            "nnz": int(scalar.nnz),
            "storage_bytes": sparse_storage_bytes(scalar),
        }
        result["validation"] = validate_against_scipy(matrix, scalar, inputs, spec)
        result["timings"]["scipy"] = benchmark_repeated(
            scipy_op(scalar, inputs, spec),
            comm=None,
            repeats=repeats,
            min_seconds=min_seconds,
            min_iterations=min_iterations,
            warmups=warmups,
        )
        try:
            sparse_dot_mkl = importlib.import_module("sparse_dot_mkl")
            mkl_threading = configure_sparse_dot_mkl_threading(sparse_dot_mkl)
            if not mkl_threading.get("thread_control_ok", False):
                raise RuntimeError(f"sparse_dot_mkl thread control failed: {mkl_threading}")

            mkl_matrix, mkl_info = mkl_baseline_matrix(scalar, spec)
            mkl_info.update(
                {
                    "available": True,
                    "nnz": int(mkl_matrix.nnz),
                    "storage_bytes": sparse_storage_bytes(mkl_matrix),
                    "threading": mkl_threading,
                }
            )
            result["baselines"]["mkl_sparse"] = mkl_info
            try:
                result["timings"]["mkl"] = benchmark_repeated(
                    mkl_op(mkl_matrix, inputs, spec, sparse_dot_mkl),
                    comm=None,
                    repeats=repeats,
                    min_seconds=min_seconds,
                    min_iterations=min_iterations,
                    warmups=warmups,
                )
            finally:
                mkl_info["threading_restore"] = restore_sparse_dot_mkl_threading(sparse_dot_mkl, mkl_threading)
        except Exception as exc:
            result["baselines"]["mkl_sparse"] = {
                "available": False,
                "reason": type(exc).__name__,
                "message": str(exc),
                "required_for_publication": True,
            }
            if require_mkl:
                raise
    else:
        result["baselines"]["scipy_csr"] = {"available": False, "reason": "serial_efficiency_only"}
        result["baselines"]["mkl_sparse"] = {"available": False, "reason": "serial_efficiency_only"}

    if spec.operation == "spgemm":
        output = matrix.spmm(matrix, spec.spgemm_threshold)
        result["spgemm"] = {
            "threshold": spec.spgemm_threshold,
            "output_matrix_kind": output.matrix_kind,
            "output_block_nnz_sum": int(reduce_value(comm, output.local_block_nnz, "sum")),
            "output_scalar_nnz_sum": int(reduce_value(comm, output.local_nnz, "sum")),
            "fill_ratio_scalar": float(reduce_value(comm, output.local_nnz, "sum") / max(result["matrix"]["global_scalar_nnz"], 1)),
            "threshold_audit": spgemm_threshold_audit(adjacency, block_sizes, spec, positions, box_lengths),
        }

    vbcsr_median = result["timings"]["vbcsr"]["median_seconds"]
    baseline_times = []
    if "scipy" in result["timings"]:
        scipy_median = result["timings"]["scipy"]["median_seconds"]
        result["speedups"]["vbcsr_vs_scipy"] = scipy_median / vbcsr_median
        baseline_times.append(scipy_median)
    if "mkl" in result["timings"]:
        mkl_median = result["timings"]["mkl"]["median_seconds"]
        result["speedups"]["vbcsr_vs_mkl"] = mkl_median / vbcsr_median
        baseline_times.append(mkl_median)
    if baseline_times:
        result["speedups"]["vbcsr_vs_best_scalar_baseline"] = min(baseline_times) / vbcsr_median
        result["speedups"]["best_scalar_baseline_seconds"] = min(baseline_times)
    return result


def git_metadata() -> dict[str, Any]:
    def git(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = git(["status", "--short"])
    return {"commit": git(["rev-parse", "HEAD"]), "branch": git(["rev-parse", "--abbrev-ref", "HEAD"]), "dirty": bool(status), "status_short": status}


def cmake_metadata() -> dict[str, Any]:
    paths = []
    if os.environ.get("VBCSR_BUILD_DIR"):
        paths.append(Path(os.environ["VBCSR_BUILD_DIR"]) / "CMakeCache.txt")
    paths.extend([REPO_ROOT / "build" / "CMakeCache.txt", REPO_ROOT / "build_dbg" / "CMakeCache.txt"])
    prefixes = ("CMAKE_BUILD_TYPE", "CMAKE_C_COMPILER", "CMAKE_CXX_COMPILER", "BLAS_", "LAPACK_", "MPI_", "OpenMP_", "VBCSR_")
    for path in paths:
        if not path.exists():
            continue
        entries = {}
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line or line.startswith(("//", "#")) or "=" not in line:
                continue
            key_type, value = line.split("=", 1)
            key = key_type.split(":", 1)[0]
            if key.startswith(prefixes):
                entries[key] = value
        return {"cache": str(path), "entries": entries}
    return {"cache": None, "entries": {}}


def numpy_blas_metadata() -> dict[str, str]:
    stream = io.StringIO()
    with redirect_stdout(stream):
        np.__config__.show()
    return {"numpy_config": stream.getvalue()}


def flexiblas_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "executable": shutil.which("flexiblas"),
        "FLEXIBLAS": os.environ.get("FLEXIBLAS"),
        "FLEXIBLAS64": os.environ.get("FLEXIBLAS64"),
        "FLEXIBLAS_CONFIG": os.environ.get("FLEXIBLAS_CONFIG"),
    }
    if result["executable"] is None:
        return result
    for name, args in (("list", ["list"]), ("current", ["current"])):
        try:
            result[name] = subprocess.check_output(
                [str(result["executable"]), *args],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=10,
            ).strip()
        except Exception as exc:
            result[name] = f"{type(exc).__name__}: {exc}"
    return result


def package_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "vbcsr": getattr(vbcsr, "__version__", "unknown"),
        "vbcsr_file": getattr(vbcsr, "__file__", None),
    }
    for name in ("numpy", "scipy", "mpi4py", "sparse_dot_mkl", "vbcsr_core"):
        try:
            module = importlib.import_module(name)
            result[name] = {"available": True, "version": getattr(module, "__version__", "unknown"), "file": getattr(module, "__file__", None)}
        except Exception as exc:
            result[name] = {"available": False, "reason": type(exc).__name__, "message": str(exc)}
    return result


def cpu_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {"hostname": socket.gethostname(), "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count()}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        model_name = None
        logical = 0
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("processor"):
                logical += 1
            elif line.startswith("model name") and model_name is None:
                model_name = line.split(":", 1)[1].strip()
        result.update({"linux_logical_cpu_count": logical, "linux_model_name": model_name})
    return result


def mpi_metadata(comm: Any, rank: int, size: int) -> dict[str, Any]:
    result = {"rank_count": size, "mpi4py_available": MPI is not None, "python_reductions_available": comm is not None}
    if MPI is not None:
        result["mpi_library_version"] = MPI.Get_library_version()
        result["processor_name"] = MPI.Get_processor_name()
    if MPI_FALLBACK_STATUS:
        result["fallback_status"] = dict(MPI_FALLBACK_STATUS)
    gathered = comm.gather({"rank": rank, "hostname": socket.gethostname()}, root=0) if comm is not None else None
    if rank == 0:
        result["rank_hosts"] = gathered or [{"rank": rank, "hostname": socket.gethostname()}]
    return result


def environment_metadata(comm: Any, rank: int, size: int) -> dict[str, Any]:
    thread_vars = (
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "SPARSE_DOT_MKL_NUM_THREADS",
        "MKL_NUM_THREADS",
        "MKL_DYNAMIC",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    slurm_vars = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_ACCOUNT",
        "SLURM_JOB_PARTITION",
        "SLURM_NNODES",
        "SLURM_NTASKS",
        "SLURM_TASKS_PER_NODE",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_SUBMIT_DIR",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "cwd": str(Path.cwd()),
        "git": git_metadata(),
        "packages": package_metadata(),
        "cmake": cmake_metadata(),
        "blas": numpy_blas_metadata(),
        "flexiblas": flexiblas_metadata(),
        "cpu": cpu_metadata(),
        "mpi": mpi_metadata(comm, rank, size),
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
            "python_venv": os.environ.get("PYTHON_VENV"),
            "VBCSR_BUILD_DIR": os.environ.get("VBCSR_BUILD_DIR"),
            "loaded_modules": os.environ.get("LOADEDMODULES"),
            "lmod_family_mpi": os.environ.get("LMOD_FAMILY_MPI"),
            "lmod_family_compiler": os.environ.get("LMOD_FAMILY_COMPILER"),
            "flexiblas": {
                "FLEXIBLAS": os.environ.get("FLEXIBLAS"),
                "FLEXIBLAS64": os.environ.get("FLEXIBLAS64"),
                "FLEXIBLAS_CONFIG": os.environ.get("FLEXIBLAS_CONFIG"),
            },
            "slurm": {name: os.environ.get(name) for name in slurm_vars},
            "threads": {name: os.environ.get(name) for name in thread_vars},
        },
    }


def build_specs(args: argparse.Namespace, rank_count: int) -> list[BenchmarkSpec]:
    dtype = parse_dtype(args.dtype)
    spgemm_thresholds = parse_thresholds(args.spgemm_thresholds, args.spgemm_threshold)
    if args.magnitude_decay_length <= 0.0:
        raise ValueError("--magnitude-decay-length must be positive")
    if args.offdiagonal_scale <= 0.0:
        raise ValueError("--offdiagonal-scale must be positive")
    if args.suite == "distributed-weak":
        blocks = int(args.weak_blocks_per_rank) * rank_count
        weak_blocks = int(args.weak_blocks_per_rank)
    else:
        blocks = int(args.blocks)
        weak_blocks = None

    specs: list[BenchmarkSpec] = []
    for domain in DOMAINS:
        for operation in OPERATIONS:
            thresholds = spgemm_thresholds if operation == "spgemm" else [float(args.spgemm_threshold)]
            for threshold in thresholds:
                specs.append(
                    BenchmarkSpec(
                        suite=args.suite,
                        domain=domain,
                        operation=operation,
                        blocks=blocks,
                        target_degree=int(args.target_degree),
                        rhs=int(args.rhs),
                        dtype=dtype,
                        seed=int(args.seed),
                        bsr_block_size=int(args.bsr_block_size),
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
                        weak_blocks_per_rank=weak_blocks,
                    )
                )
    return specs


def write_outputs(payload: dict[str, Any], output_dir: Path, label: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{label}.json"
    csv_path = output_dir / f"{label}.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fields = [
        "label",
        "suite",
        "rank_count",
        "domain",
        "operation",
        "matrix_kind",
        "blocks",
        "scalar_rows",
        "block_nnz",
        "scalar_nnz",
        "degree_mean",
        "degree_min",
        "degree_max",
        "rhs",
        "spgemm_threshold",
        "magnitude_decay_length",
        "offdiagonal_magnitude_scale_min",
        "offdiagonal_magnitude_scale_mean",
        "offdiagonal_magnitude_scale_max",
        "vbcsr_median_seconds",
        "vbcsr_min_seconds",
        "vbcsr_max_seconds",
        "vbcsr_std_seconds",
        "scipy_median_seconds",
        "scipy_min_seconds",
        "scipy_max_seconds",
        "scipy_std_seconds",
        "mkl_median_seconds",
        "mkl_min_seconds",
        "mkl_max_seconds",
        "mkl_std_seconds",
        "vbcsr_vs_best_scalar_baseline",
        "vbcsr_vendor_backend",
        "vbcsr_vendor_launches_per_call",
        "vbcsr_vendor_launch_delta_sum",
        "vbcsr_operation_calls_including_warmups_sum",
        "validation_passed",
        "validation_exact_required",
        "validation_relative_error",
        "graph_construction_seconds",
        "assembly_seconds",
        "atomic_data_construction_seconds",
        "atomic_graph_to_matrix_facade_seconds",
        "atomic_edge_count_sum",
        "ghost_spmv_send_bytes_sum",
        "ghost_spmv_recv_bytes_sum",
        "ghost_spmm_send_bytes_sum",
        "ghost_spmm_recv_bytes_sum",
        "estimated_vbcsr_storage_bytes_sum",
        "spgemm_candidate_block_products",
        "spgemm_row_scaled_product_skip_count",
        "spgemm_output_scalar_nnz_sum",
        "spgemm_fill_ratio_scalar",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in payload["cases"]:
            timings = case["timings"]
            writer.writerow(
                {
                    "label": case["label"],
                    "suite": case["suite"],
                    "rank_count": case["rank_count"],
                    "domain": case["domain"],
                    "operation": case["operation"],
                    "matrix_kind": case["matrix"]["matrix_kind"],
                    "blocks": case["parameters"]["blocks"],
                    "scalar_rows": case["matrix"]["global_scalar_rows"],
                    "block_nnz": case["matrix"]["global_block_nnz"],
                    "scalar_nnz": case["matrix"]["global_scalar_nnz"],
                    "degree_mean": case["adjacency"]["degree_mean"],
                    "degree_min": case["adjacency"]["degree_min"],
                    "degree_max": case["adjacency"]["degree_max"],
                    "rhs": case["parameters"]["rhs"],
                    "spgemm_threshold": case["parameters"]["spgemm_threshold"] if case["operation"] == "spgemm" else None,
                    "magnitude_decay_length": case["parameters"]["magnitude_decay_length"],
                    "offdiagonal_magnitude_scale_min": case["matrix_value_model"].get("offdiagonal_magnitude_scale_min"),
                    "offdiagonal_magnitude_scale_mean": case["matrix_value_model"].get("offdiagonal_magnitude_scale_mean"),
                    "offdiagonal_magnitude_scale_max": case["matrix_value_model"].get("offdiagonal_magnitude_scale_max"),
                    "vbcsr_median_seconds": timings["vbcsr"]["median_seconds"],
                    "vbcsr_min_seconds": timings["vbcsr"]["min_seconds"],
                    "vbcsr_max_seconds": timings["vbcsr"]["max_seconds"],
                    "vbcsr_std_seconds": timings["vbcsr"]["std_seconds"],
                    "scipy_median_seconds": timings.get("scipy", {}).get("median_seconds"),
                    "scipy_min_seconds": timings.get("scipy", {}).get("min_seconds"),
                    "scipy_max_seconds": timings.get("scipy", {}).get("max_seconds"),
                    "scipy_std_seconds": timings.get("scipy", {}).get("std_seconds"),
                    "mkl_median_seconds": timings.get("mkl", {}).get("median_seconds"),
                    "mkl_min_seconds": timings.get("mkl", {}).get("min_seconds"),
                    "mkl_max_seconds": timings.get("mkl", {}).get("max_seconds"),
                    "mkl_std_seconds": timings.get("mkl", {}).get("std_seconds"),
                    "vbcsr_vs_best_scalar_baseline": case["speedups"].get("vbcsr_vs_best_scalar_baseline"),
                    "vbcsr_vendor_backend": case.get("vbcsr_internal", {}).get("vendor_backend_name"),
                    "vbcsr_vendor_launches_per_call": case.get("vbcsr_internal", {}).get("vendor_launches_per_call"),
                    "vbcsr_vendor_launch_delta_sum": case.get("vbcsr_internal", {}).get("vendor_launch_delta_sum"),
                    "vbcsr_operation_calls_including_warmups_sum": case.get("vbcsr_internal", {}).get(
                        "operation_calls_including_warmups_sum"
                    ),
                    "validation_passed": case["validation"].get("passed"),
                    "validation_exact_required": case["validation"].get("exact_validation_required"),
                    "validation_relative_error": case["validation"].get("relative_error"),
                    "graph_construction_seconds": timings["graph_construction_seconds"],
                    "assembly_seconds": timings["assembly_seconds"],
                    "atomic_data_construction_seconds": case["atomistic_conversion"]["atomic_data_construction_seconds"],
                    "atomic_graph_to_matrix_facade_seconds": case["atomistic_conversion"]["atomic_graph_to_matrix_facade_seconds"],
                    "atomic_edge_count_sum": case["atomistic_conversion"]["local_edge_count_sum"],
                    "ghost_spmv_send_bytes_sum": case["matrix"]["communication_estimate"]["spmv_send_bytes_sum"],
                    "ghost_spmv_recv_bytes_sum": case["matrix"]["communication_estimate"]["spmv_recv_bytes_sum"],
                    "ghost_spmm_send_bytes_sum": case["matrix"]["communication_estimate"]["spmm_send_bytes_sum"],
                    "ghost_spmm_recv_bytes_sum": case["matrix"]["communication_estimate"]["spmm_recv_bytes_sum"],
                    "estimated_vbcsr_storage_bytes_sum": case["matrix"]["local"]["estimated_vbcsr_storage_bytes_sum"],
                    "spgemm_candidate_block_products": case.get("spgemm", {}).get("threshold_audit", {}).get("candidate_block_products"),
                    "spgemm_row_scaled_product_skip_count": case.get("spgemm", {}).get("threshold_audit", {}).get("row_scaled_product_skip_count"),
                    "spgemm_output_scalar_nnz_sum": case.get("spgemm", {}).get("output_scalar_nnz_sum"),
                    "spgemm_fill_ratio_scalar": case.get("spgemm", {}).get("fill_ratio_scalar"),
                }
            )
    return json_path, csv_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run publication VBCSR benchmark data generation.")
    parser.add_argument("--suite", choices=("efficiency", "distributed-strong", "distributed-weak"), default="efficiency")
    parser.add_argument("--blocks", type=int, default=4096, help="Global block count for efficiency and strong scaling")
    parser.add_argument("--weak-blocks-per-rank", type=int, default=4096)
    parser.add_argument("--target-degree", type=int, default=12)
    parser.add_argument("--rhs", type=int, default=16)
    parser.add_argument("--dtype", choices=("real", "complex"), default="real")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--bsr-block-size", type=int, default=8)
    parser.add_argument("--spgemm-threshold", type=float, default=0.0)
    parser.add_argument(
        "--spgemm-thresholds",
        default=None,
        help="Comma or space separated SpGEMM threshold sweep. SpMV/SpMM are still run once.",
    )
    parser.add_argument("--spgemm-audit-limit", type=int, default=200000)
    parser.add_argument("--geometry-dim", type=int, default=3)
    parser.add_argument("--geometry-spacing", type=float, default=1.0)
    parser.add_argument("--geometry-jitter", type=float, default=0.12)
    parser.add_argument("--geometry-cutoff", type=float, default=None)
    parser.add_argument("--geometry-cutoff-quantile", type=float, default=0.90)
    parser.add_argument("--magnitude-decay-length", type=float, default=0.5)
    parser.add_argument("--offdiagonal-scale", type=float, default=1.0)
    parser.add_argument("--diagonal-shift", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--min-iterations", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--require-mkl", action="store_true", help="Fail if sparse_dot_mkl cannot be used for serial efficiency data")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--label", default=None)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    comm, rank, size = mpi_context()

    if args.suite != "efficiency" and comm is None:
        if rank == 0:
            if int(MPI_FALLBACK_STATUS.get("environment_size") or 1) > 1:
                detail = (
                    "launcher reports "
                    f"size={MPI_FALLBACK_STATUS.get('environment_size')}, but vbcsr_core reports "
                    f"native MPI size={MPI_FALLBACK_STATUS.get('native_vbcsr_size')}"
                )
            else:
                detail = "mpi4py is not available"
            print(
                "Distributed publication data requires mpi4py so timing and metadata "
                f"can be reduced correctly across ranks ({detail}).",
                file=sys.stderr,
            )
        return 2
    if args.suite == "efficiency" and size != 1:
        if rank == 0:
            print("Efficiency data must be generated with one rank.", file=sys.stderr)
        return 2

    specs = build_specs(args, size)
    cases: list[dict[str, Any]] = []
    for spec in specs:
        if rank == 0:
            print(f"[{spec.label}] running", flush=True)
        case_start = time.perf_counter()
        case = run_case(
            spec,
            comm=comm,
            rank=rank,
            size=size,
            repeats=int(args.repeats),
            min_seconds=float(args.min_seconds),
            min_iterations=int(args.min_iterations),
            warmups=int(args.warmups),
            require_mkl=bool(args.require_mkl),
        )
        if rank == 0:
            cases.append(case)
            elapsed = time.perf_counter() - case_start
            vbcsr_median = case.get("timings", {}).get("vbcsr", {}).get("median_seconds")
            print(
                f"[{spec.label}] done in {elapsed:.2f} s; "
                f"VBCSR median={vbcsr_median:.6g} s",
                flush=True,
            )
            validation = case.get("validation", {})
            if validation.get("exact_validation_required", True) and validation.get("passed") is False:
                print(f"Validation failed for {case['label']}: {validation}", file=sys.stderr)
                return 1

    if rank == 0:
        label = args.label.strip() if isinstance(args.label, str) else args.label
        if label is None or label == "":
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            label = f"vbcsr_publication_{args.suite}_np{size}_{stamp}"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "metadata": environment_metadata(comm, rank, size),
            "publication_coverage": {
                "domains": list(DOMAINS),
                "operations": list(OPERATIONS),
                "adjacency_model": "geometric_finite_cutoff",
                "matrix_value_model": "exponential_distance_decay_with_onsite_shift",
                "efficiency_baselines": ["scipy_csr", "mkl_sparse_if_available"],
                "distributed_metrics": [
                    "operation_time",
                    "graph_construction_seconds",
                    "assembly_seconds",
                    "memory_estimates",
                    "ghost_exchange_bytes",
                    "atomistic_conversion_seconds",
                    "spgemm_fill_ratio",
                    "spgemm_threshold_audit",
                ],
            },
            "cases": cases,
        }
        json_path, csv_path = write_outputs(payload, args.output_dir, label)
        print(f"Wrote {json_path}")
        print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

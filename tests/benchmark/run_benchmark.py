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
import hashlib
import importlib
import io
import itertools
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
from collections.abc import Sequence
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
# At most one entry: the matrix shared by the operations of the current domain.
_CASE_MATRIX_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


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
    geometry_ordering: str = "bisection"
    weak_blocks_per_rank: int | None = None
    # "physical": per-block values from the geometric decay model (needed for
    # accuracy validation and thresholded-SpGEMM realism). "random": one
    # parallel C++ fill (VBCSR.fill_random) -- same structure and flop/byte
    # pattern, no per-block Python assembly; assembly cost drops to ~zero.
    # Timing-only runs (scaling sweeps at threshold 0) should use "random".
    value_fill: str = "physical"

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


def parse_operations(value: str | None) -> list[str]:
    if value is None or value.strip() == "":
        return list(OPERATIONS)
    operations = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [operation for operation in operations if operation not in OPERATIONS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown operation(s) {unknown}; choose from {list(OPERATIONS)}")
    if not operations:
        raise argparse.ArgumentTypeError("--operations selected nothing")
    return operations


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


def geometric_grid_shape(blocks: int, dim: int) -> tuple[int, ...]:
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    if dim <= 0:
        raise ValueError("geometry dimension must be positive")
    side = max(1, int(math.ceil(blocks ** (1.0 / dim))))
    return tuple([side] * dim)


def bisection_order(coords: np.ndarray) -> np.ndarray:
    """Permutation ordering points by orthogonal recursive bisection.

    Ranks are assigned contiguous global id ranges (`partition_range`), so the
    numbering decides the shape of each rank's subdomain. Under lexicographic
    numbering a contiguous range is a *slab* of the box: at fixed volume per
    rank its halo grows as P^(2/3), which is what made communication grow with
    rank count. Recursively bisecting along the longest axis instead makes any
    contiguous range a compact box, so the halo stays proportional to the
    subdomain surface.

    Splitting on the longest axis reproduces the lexicographic cut whenever
    that is already optimal (e.g. a single planar cut at P = 2), so this never
    costs communication relative to the old ordering.
    """
    n_points = coords.shape[0]
    if n_points == 0:
        return np.zeros(0, dtype=np.int64)
    # Bounded leaf size: ordering inside a leaf is irrelevant as long as leaves
    # are far smaller than a rank's share, and it keeps the split count linear.
    leaf_size = max(8, n_points // 100_000)
    order = np.empty(n_points, dtype=np.int64)
    filled = 0
    # Reverse-order stack so the traversal emits points left-to-right.
    pending = [np.arange(n_points, dtype=np.int64)]
    while pending:
        selection = pending.pop()
        if selection.size <= leaf_size:
            order[filled:filled + selection.size] = selection
            filled += selection.size
            continue
        points = coords[selection]
        axis = int(np.argmax(points.max(axis=0) - points.min(axis=0)))
        middle = selection.size // 2
        split = np.argpartition(points[:, axis], middle)
        pending.append(selection[split[middle:]])
        pending.append(selection[split[:middle]])
    return order


def make_geometric_positions(spec: BenchmarkSpec) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    shape = geometric_grid_shape(spec.blocks, spec.geometry_dim)
    grid_size = math.prod(shape)
    flat = (np.arange(spec.blocks, dtype=np.int64) * grid_size) // spec.blocks
    grid_int = np.array(np.unravel_index(flat, shape), dtype=np.int64).T
    if spec.geometry_ordering == "bisection":
        grid_int = grid_int[bisection_order(grid_int)]
    elif spec.geometry_ordering != "lexicographic":
        raise ValueError(f"unknown geometry ordering {spec.geometry_ordering!r}")
    grid = grid_int.astype(np.float64)
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
    # Calibrate from a bounded, evenly spread sample. The k-th neighbour
    # distance is a property of the lattice, not of the block count, so a
    # sample fixes the cutoff just as well as querying every point -- and it
    # keeps the O(blocks * k) distance array from dominating memory on the
    # large runs, where every rank performs this calibration.
    sample_limit = 50000
    if positions.shape[0] > sample_limit:
        stride = positions.shape[0] // sample_limit
        sample = positions[::stride]
    else:
        sample = positions
    distances, _ = tree.query(sample, k=kth)
    kth_distances = np.asarray(distances)[:, kth - 1]
    cutoff = float(np.quantile(kth_distances[np.isfinite(kth_distances)], spec.geometry_cutoff_quantile))
    return max(cutoff * 1.000001, np.finfo(float).eps), "calibrated_from_target_degree"


def flat_adjacency(adjacency: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Adjacency as flat CSR-style arrays: (per-row counts, indptr, indices).

    The scaling sizes reach tens of millions of block edges, where per-edge
    Python loops over the list-of-lists dominate the whole run. Every O(nnz)
    statistic is computed from these arrays instead.
    """
    counts = np.fromiter((len(cols) for cols in adjacency), dtype=np.int64, count=len(adjacency))
    indptr = np.zeros(len(adjacency) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = np.fromiter(
        itertools.chain.from_iterable(adjacency), dtype=np.int64, count=int(indptr[-1])
    )
    return counts, indptr, indices


def block_owners(blocks: int, size: int) -> np.ndarray:
    """Owning rank of every global block id, as a lookup table."""
    bounds = np.array([partition_range(blocks, size, r)[1] for r in range(size)], dtype=np.int64)
    return np.searchsorted(bounds, np.arange(blocks, dtype=np.int64), side="right")


def row_periodic_distances(
    row: int,
    cols: Sequence[int],
    positions: np.ndarray,
    box_lengths: np.ndarray,
) -> np.ndarray:
    """Minimum-image distances from one block-row to all of its neighbours."""
    delta = positions[np.asarray(cols, dtype=np.int64)] - positions[row]
    delta = delta - box_lengths * np.round(delta / box_lengths)
    return np.linalg.norm(delta, axis=1)


def row_magnitude_scales(
    row: int,
    cols: Sequence[int],
    positions: np.ndarray,
    box_lengths: np.ndarray,
    spec: BenchmarkSpec,
) -> np.ndarray:
    """Off-diagonal magnitude envelope for one block-row's neighbours.

    Exponential decay in minimum-image distance; the diagonal block is left at
    unit scale.
    """
    if spec.magnitude_decay_length <= 0.0:
        raise ValueError("--magnitude-decay-length must be positive")
    columns = np.asarray(cols, dtype=np.int64)
    distances = row_periodic_distances(row, columns, positions, box_lengths)
    scales = spec.offdiagonal_scale * np.exp(-distances / spec.magnitude_decay_length)
    scales[columns == row] = 1.0
    return scales


def matrix_value_model_statistics(
    adjacency: list[list[int]],
    positions: np.ndarray,
    box_lengths: np.ndarray,
    spec: BenchmarkSpec,
    row_offset: int = 0,
    comm: Any = None,
) -> dict[str, Any]:
    # Accumulated per row rather than materialized: at the sizes used for
    # scaling runs the off-diagonal edge list has tens of millions of entries.
    count = 0
    distance_sum = 0.0
    distance_min = math.inf
    distance_max = -math.inf
    scale_sum = 0.0
    scale_min = math.inf
    scale_max = -math.inf
    for offset, cols in enumerate(adjacency):
        row = row_offset + offset
        columns = np.asarray(cols, dtype=np.int64)
        columns = columns[columns != row]
        if columns.size == 0:
            continue
        distances = row_periodic_distances(row, columns, positions, box_lengths)
        scales = spec.offdiagonal_scale * np.exp(-distances / spec.magnitude_decay_length)
        count += int(columns.size)
        distance_sum += float(distances.sum())
        distance_min = min(distance_min, float(distances.min()))
        distance_max = max(distance_max, float(distances.max()))
        scale_sum += float(scales.sum())
        scale_min = min(scale_min, float(scales.min()))
        scale_max = max(scale_max, float(scales.max()))

    # Rows are rank-local; fold the partial accumulators into global values.
    count = int(reduce_value(comm, count, "sum"))
    distance_sum = float(reduce_value(comm, distance_sum, "sum"))
    scale_sum = float(reduce_value(comm, scale_sum, "sum"))
    if count:
        distance_min = float(reduce_value(comm, distance_min, "min"))
        distance_max = float(reduce_value(comm, distance_max, "max"))
        scale_min = float(reduce_value(comm, scale_min, "min"))
        scale_max = float(reduce_value(comm, scale_max, "max"))

    result: dict[str, Any] = {
        "model": "exponential_distance_decay_with_onsite_shift",
        "distance_metric": "periodic_minimum_image",
        "magnitude_decay_length": spec.magnitude_decay_length,
        "offdiagonal_scale": spec.offdiagonal_scale,
        "diagonal_shift": spec.diagonal_shift,
        "diagonal_random_scale": 1.0,
        "block_random_normalization": "1/sqrt(row_block_size * column_block_size)",
    }
    if count:
        result.update(
            {
                "offdiagonal_distance_min": distance_min,
                "offdiagonal_distance_mean": distance_sum / count,
                "offdiagonal_distance_max": distance_max,
                "offdiagonal_magnitude_scale_min": scale_min,
                "offdiagonal_magnitude_scale_mean": scale_sum / count,
                "offdiagonal_magnitude_scale_max": scale_max,
            }
        )
    return result


def make_geometric_adjacency(
    spec: BenchmarkSpec,
    owned_range: tuple[int, int] | None = None,
) -> tuple[list[list[int]], dict[str, Any], np.ndarray, np.ndarray]:
    """Cutoff-graph rows owned by this rank, with global column ids.

    `owned_range` restricts the neighbour query to the rank's own rows. The
    KD-tree still spans the whole box (neighbours may live anywhere), but the
    returned neighbour lists do not: materializing the global adjacency as a
    Python list-of-lists on every rank costs ~3 GB at the sizes used for
    scaling runs, which does not fit 48 times over. Row `i` of the result is
    global block `owned_range[0] + i`.

    Degree statistics are local to the rank; `run_case` reduces them.
    """
    if spec.geometry_spacing <= 0.0:
        raise ValueError("--geometry-spacing must be positive")
    if spec.geometry_jitter < 0.0:
        raise ValueError("--geometry-jitter must be non-negative")
    positions, box_lengths, grid_shape = make_geometric_positions(spec)
    cutoff, cutoff_source = calibrate_cutoff(positions, box_lengths, spec)
    tree = cKDTree(positions, boxsize=box_lengths)
    start, end = owned_range if owned_range is not None else (0, spec.blocks)
    raw = tree.query_ball_point(positions[start:end], r=cutoff)
    adjacency = [
        sorted({int(col) for col in cols} | {start + offset})
        for offset, cols in enumerate(raw)
    ]
    degrees = np.array([len(row) for row in adjacency], dtype=np.int64)
    return adjacency, {
        "model": "geometric_finite_cutoff",
        "dimension": spec.geometry_dim,
        "ordering": spec.geometry_ordering,
        "grid_shape": list(grid_shape),
        "periodic": True,
        "spacing": spec.geometry_spacing,
        "jitter_fraction_of_spacing": spec.geometry_jitter,
        "cutoff": cutoff,
        "cutoff_source": cutoff_source,
        "cutoff_quantile": spec.geometry_cutoff_quantile,
        "local_degree_min": int(degrees.min()) if degrees.size else 0,
        "local_degree_max": int(degrees.max()) if degrees.size else 0,
        "local_degree_sum": int(degrees.sum()),
        "local_row_count": int(degrees.size),
    }, positions, box_lengths


# Structure disk cache: the geometric graph is a pure function of the spec
# fields below, yet every process (each thread count of a sweep, each rank
# count of a panel) regenerated it serially -- minutes of single-core work per
# point, dwarfing the measurements themselves. Rank 0 generates the GLOBAL
# flat adjacency + positions once and stores them as .npy files; every later
# process mmap-loads and slices its owned range in seconds. Files are keyed by
# a hash of the generating parameters and validated against the stored meta.
GEOMETRY_CACHE_VERSION = 1
_GEOMETRY_CACHE_DIR: Path | None = None


def geometry_cache_entry(spec: BenchmarkSpec, cache_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    key_fields = {
        "version": GEOMETRY_CACHE_VERSION,
        "blocks": int(spec.blocks),
        "dim": int(spec.geometry_dim),
        "spacing": float(spec.geometry_spacing),
        "jitter": float(spec.geometry_jitter),
        "cutoff": None if spec.geometry_cutoff is None else float(spec.geometry_cutoff),
        "cutoff_quantile": float(spec.geometry_cutoff_quantile),
        "ordering": str(spec.geometry_ordering),
        "target_degree": int(spec.target_degree),
        "seed": int(spec.seed),
    }
    digest = hashlib.sha1(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()[:16]
    base = cache_dir / f"geom_{digest}"
    paths = {
        "meta": base.with_suffix(".meta.json"),
        "adj_ptr": base.with_suffix(".adj_ptr.npy"),
        "adj_ind": base.with_suffix(".adj_ind.npy"),
        "positions": base.with_suffix(".positions.npy"),
    }
    return key_fields, paths


def _geometry_cache_valid(key_fields: dict[str, Any], paths: dict[str, Path]) -> bool:
    if not all(path.exists() for path in paths.values()):
        return False
    try:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("key_fields") == key_fields


def _atomic_np_save(path: Path, array: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.save(tmp, array)
    # np.save appends .npy to names without it
    tmp_actual = tmp if tmp.suffix == ".npy" else Path(str(tmp) + ".npy")
    os.replace(tmp_actual, path)


def cached_geometric_adjacency(
    spec: BenchmarkSpec,
    owned_range: tuple[int, int] | None,
    comm: Any,
    rank: int,
    cache_dir: Path,
) -> tuple[list[np.ndarray], dict[str, Any], np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """make_geometric_adjacency semantics, backed by the on-disk global graph.

    Adjacency rows come back as numpy index arrays instead of Python lists;
    every consumer only iterates them. Positions are a read-only mmap. The
    fifth element is the rank-local flat CSR view (rebased adj_ptr, GLOBAL
    column ids) consumed by VBCSR.create_distributed_flat.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_fields, paths = geometry_cache_entry(spec, cache_dir)
    if rank == 0 and not _geometry_cache_valid(key_fields, paths):
        adjacency, info, positions, box_lengths = make_geometric_adjacency(spec, None)
        _, adj_ptr, adj_ind = flat_adjacency(adjacency)
        del adjacency
        _atomic_np_save(paths["adj_ptr"], adj_ptr)
        _atomic_np_save(paths["adj_ind"], adj_ind.astype(np.int32))
        _atomic_np_save(paths["positions"], positions)
        info.pop("local_degree_min", None)
        info.pop("local_degree_max", None)
        info.pop("local_degree_sum", None)
        info.pop("local_row_count", None)
        meta = {"key_fields": key_fields, "info": info, "box_lengths": list(map(float, box_lengths))}
        tmp = paths["meta"].with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, paths["meta"])
    barrier(comm)

    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    adj_ptr = np.load(paths["adj_ptr"], mmap_mode="r")
    adj_ind = np.load(paths["adj_ind"], mmap_mode="r")
    positions = np.load(paths["positions"], mmap_mode="r")
    box_lengths = np.array(meta["box_lengths"], dtype=np.float64)

    start, end = owned_range if owned_range is not None else (0, spec.blocks)
    local_ptr = np.asarray(adj_ptr[start:end + 1], dtype=np.int64)
    base_offset = int(local_ptr[0]) if local_ptr.size else 0
    local_ind = np.asarray(adj_ind[base_offset:int(local_ptr[-1])], dtype=np.int64) if local_ptr.size else np.empty(0, np.int64)
    rows = np.split(local_ind, (local_ptr - base_offset)[1:-1]) if local_ptr.size else []
    degrees = np.diff(local_ptr)
    info = dict(meta["info"])
    info["local_degree_min"] = int(degrees.min()) if degrees.size else 0
    info["local_degree_max"] = int(degrees.max()) if degrees.size else 0
    info["local_degree_sum"] = int(degrees.sum())
    info["local_row_count"] = int(degrees.size)
    # Rank-local flat CSR view (row pointers rebased to 0, GLOBAL column ids):
    # the zero-conversion input for VBCSR.create_distributed_flat.
    flat_local = (local_ptr - base_offset, local_ind)
    return rows, info, positions, box_lengths, flat_local


def reduce_degree_statistics(adjacency_info: dict[str, Any], comm: Any) -> dict[str, Any]:
    """Fold the rank-local degree accumulators into global degree statistics."""
    row_count = int(reduce_value(comm, adjacency_info.pop("local_row_count"), "sum"))
    degree_sum = int(reduce_value(comm, adjacency_info.pop("local_degree_sum"), "sum"))
    degree_min = int(reduce_value(comm, adjacency_info.pop("local_degree_min"), "min"))
    degree_max = int(reduce_value(comm, adjacency_info.pop("local_degree_max"), "max"))
    adjacency_info["degree_min"] = degree_min
    adjacency_info["degree_max"] = degree_max
    adjacency_info["degree_mean"] = float(degree_sum / row_count) if row_count else 0.0
    adjacency_info["directed_block_edges"] = degree_sum
    return adjacency_info


def make_block_sizes(spec: BenchmarkSpec) -> list[int]:
    if spec.domain == "csr":
        return [1] * spec.blocks
    if spec.domain == "bsr":
        return [spec.bsr_block_size] * spec.blocks
    if spec.domain == "vbcsr":
        rng = np.random.default_rng(stable_seed(spec.seed, 101, spec.blocks))
        return rng.choice(np.array(VBCSR_BLOCK_SIZES, dtype=np.int32), size=spec.blocks).astype(int).tolist()
    raise ValueError(f"unknown domain {spec.domain!r}")


def make_row_block_data(
    row: int,
    cols: Sequence[int],
    block_sizes: Sequence[int],
    spec: BenchmarkSpec,
    positions: np.ndarray,
    box_lengths: np.ndarray,
) -> list[np.ndarray]:
    """Every block of one block-row, drawn in a single batch.

    The generator is seeded per row rather than per block, and `cols` is that
    row's sorted adjacency. Assembly visits each owned row exactly once, so
    values stay reproducible and independent of how rows are spread over ranks.
    Batching is what makes large runs affordable: seeding a fresh generator and
    computing a periodic distance per *block* cost roughly 48 us per nonzero,
    which dominated everything else in the harness.
    """
    rng = np.random.default_rng(stable_seed(spec.seed, 307, row))
    row_dim = block_sizes[row]
    col_dims = [block_sizes[col] for col in cols]
    magnitudes = row_magnitude_scales(row, cols, positions, box_lengths, spec)
    counts = [row_dim * col_dim for col_dim in col_dims]
    samples = rng.standard_normal(sum(counts))
    if spec.dtype == np.dtype(np.complex128):
        samples = samples + 1j * rng.standard_normal(sum(counts))

    blocks: list[np.ndarray] = []
    offset = 0
    for index, col in enumerate(cols):
        col_dim = col_dims[index]
        count = counts[index]
        scale = float(magnitudes[index]) / math.sqrt(max(count, 1))
        data = (samples[offset:offset + count] * scale).reshape(row_dim, col_dim)
        offset += count
        if row == col and row_dim == col_dim:
            data = data + np.eye(row_dim, dtype=spec.dtype) * (spec.diagonal_shift + 0.01 * (row % 17))
        blocks.append(np.ascontiguousarray(data, dtype=spec.dtype))
    return blocks


def build_matrix(
    spec: BenchmarkSpec,
    block_sizes: list[int],
    adjacency: list[list[int]],
    positions: np.ndarray,
    box_lengths: np.ndarray,
    comm: Any,
    rank: int,
    size: int,
    flat_local: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[vbcsr.VBCSR, dict[str, float]]:
    start, end = partition_range(spec.blocks, size, rank)
    owned = list(range(start, end))
    local_sizes = [block_sizes[idx] for idx in owned]

    barrier(comm)
    t0 = time.perf_counter()
    if flat_local is not None:
        # Flat CSR arrays go to C++ via the buffer protocol — no per-edge
        # Python conversion (the list-of-lists path boxes every column id).
        local_ptr, local_ind = flat_local
        matrix = vbcsr.VBCSR.create_distributed_flat(
            owned_indices=np.arange(start, end, dtype=np.int32),
            block_sizes=np.asarray(local_sizes, dtype=np.int32),
            adj_ptr=local_ptr,
            adj_ind=local_ind,
            dtype=spec.dtype,
            comm=comm,
        )
    else:
        matrix = vbcsr.VBCSR.create_distributed(
            owned_indices=owned,
            block_sizes=local_sizes,
            adjacency=adjacency,
            dtype=spec.dtype,
            comm=comm,
        )
    barrier(comm)
    graph_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    if spec.value_fill == "random":
        # One parallel C++ pass over the already-allocated (and NUMA
        # first-touched) storage; no per-block Python assembly.
        matrix.fill_random()
    else:
        for offset, row in enumerate(owned):
            cols = adjacency[offset]
            row_blocks = make_row_block_data(row, cols, block_sizes, spec, positions, box_lengths)
            for col, data in zip(cols, row_blocks):
                matrix.add_block(row, col, data)
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
    counts, _, indices = flat_adjacency(adjacency)
    local_edge_index = np.empty((indices.size, 2), dtype=np.int32)
    local_edge_index[:, 0] = np.repeat(np.arange(start, end, dtype=np.int32), counts)
    local_edge_index[:, 1] = indices
    local_edge_shift = np.zeros((local_edge_index.shape[0], 3), dtype=np.int32)
    z_to_type = {int(atomic_number): idx for idx, atomic_number in enumerate(unique_z)}
    local_atom_type = np.array([z_to_type[int(z)] for z in atomic_numbers[start:end]], dtype=np.int32)
    global_edge_count = int(reduce_value(comm, int(indices.size), "sum"))

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

    # Do NOT configure MKL threading around the VBCSR timing.
    # mkl_set_num_threads_local() (what sparse_dot_mkl's helper applies) is a
    # thread-local override that beats the global mkl_set_num_threads the
    # library issues at each vendor entry, silently pinning VBCSR's SpGEMM to
    # the sparse_dot_mkl reference thread count. OMP_NUM_THREADS is the
    # library's single source of truth (doc/developer_guide.md §Threading
    # Model); reference-baseline threading is configured only around the
    # sparse_dot_mkl calls themselves.
    try:
        vbcsr_core = importlib.import_module("vbcsr_core")
        threading_diagnostics = getattr(vbcsr_core, "threading_diagnostics", None)
        if threading_diagnostics is not None:
            diagnostic["threading_before_vendor_config"] = threading_diagnostics(False)
            diagnostic["threading_after_vendor_config"] = threading_diagnostics(True)
    except Exception as exc:
        diagnostic["threading_diagnostics_error"] = f"{type(exc).__name__}: {exc}"

    # Ghost-exchange accounting for the apply ops: the X operand's core
    # object accumulates pack+MPI+unpack seconds per sync (multi-node runs
    # read the comm fraction from this — a weak-scaling curve without it is
    # a pass/fail black box).
    comm_source = None
    if spec.operation == "spmv":
        comm_source = inputs.get("x_vector")
    elif spec.operation == "spmm":
        comm_source = inputs.get("x_multivector")
    if comm_source is not None and not hasattr(comm_source, "reset_comm_stats"):
        comm_source = None
    if comm_source is not None:
        comm_source.reset_comm_stats()

    timing = benchmark_repeated(
        vbcsr_op(matrix, inputs, spec),
        comm=comm,
        repeats=repeats,
        min_seconds=min_seconds,
        min_iterations=min_iterations,
        warmups=warmups,
    )
    try:
        vbcsr_core = importlib.import_module("vbcsr_core")
        threading_diagnostics = getattr(vbcsr_core, "threading_diagnostics", None)
        if threading_diagnostics is not None:
            diagnostic["threading_after_timing"] = threading_diagnostics(False)
    except Exception as exc:
        diagnostic["threading_after_timing_error"] = f"{type(exc).__name__}: {exc}"

    if comm_source is not None:
        local_calls_comm = int(comm_source.comm_calls)
        per_call = (
            float(comm_source.comm_seconds) / local_calls_comm
            if local_calls_comm
            else 0.0
        )
        # The slowest rank's exchange bounds the collective op.
        per_call_max = float(reduce_value(comm, per_call, "max"))
        median = float(timing.get("median_seconds") or 0.0)
        diagnostic["ghost_comm_seconds_per_call_max"] = per_call_max
        diagnostic["ghost_comm_calls_local"] = local_calls_comm
        if median > 0.0:
            diagnostic["ghost_comm_fraction_of_median"] = per_call_max / median

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
    # `adjacency` holds only this rank's rows, so every edge statistic below is
    # a partial sum that has to be reduced across ranks.
    _, indptr, indices = flat_adjacency(adjacency)
    owners = block_owners(spec.blocks, size)
    local_cross_rank_edges = int((owners[indices] != rank).sum())

    local_block_nnz = int(indptr[-1])
    sizes = np.asarray(block_sizes, dtype=np.int64)
    column_size_cumsum = np.zeros(local_block_nnz + 1, dtype=np.int64)
    np.cumsum(sizes[indices], out=column_size_cumsum[1:])
    row_column_scalars = column_size_cumsum[indptr[1:]] - column_size_cumsum[indptr[:-1]]
    owned_start, _ = partition_range(spec.blocks, size, rank)
    local_scalar_nnz = int((sizes[owned_start:owned_start + len(adjacency)] * row_column_scalars).sum())

    block_nnz = int(reduce_value(comm, local_block_nnz, "sum"))
    scalar_nnz = int(reduce_value(comm, local_scalar_nnz, "sum"))
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
    """Number of candidate A_ik * B_kj products. Single-rank only.

    Indexes per-row degrees by *column* id, so it needs `adjacency` to cover
    every global row -- true only when one rank owns everything. Callers must
    keep the single-rank guard in `spgemm_threshold_audit`.
    """
    counts, _, indices = flat_adjacency(adjacency)
    return int(counts[indices].sum())


def spgemm_threshold_audit(
    adjacency: list[list[int]],
    block_sizes: list[int],
    spec: BenchmarkSpec,
    positions: np.ndarray,
    box_lengths: np.ndarray,
    size: int = 1,
) -> dict[str, Any]:
    # The audit walks A_ik * B_kj, so it needs the rows of *inner* blocks that
    # this rank does not own. Adjacency is rank-local, so it is only available
    # on a single rank.
    if size > 1:
        return {"audited": False, "reason": "adjacency is rank-local; audit requires a single rank"}
    candidate_count = spgemm_candidate_count(adjacency)
    if candidate_count > spec.spgemm_audit_limit:
        return {"audited": False, "candidate_block_products": candidate_count, "audit_limit": spec.spgemm_audit_limit}
    exact_product_below_threshold = 0
    row_scaled_product_skips = 0
    for row, inners in enumerate(adjacency):
        row_eps = spec.spgemm_threshold / max(1, len(inners))
        a_blocks = make_row_block_data(row, inners, block_sizes, spec, positions, box_lengths)
        for inner, a in zip(inners, a_blocks):
            norm_a = float(np.linalg.norm(a))
            inner_cols = adjacency[inner]
            b_blocks = make_row_block_data(inner, inner_cols, block_sizes, spec, positions, box_lengths)
            for col, b in zip(inner_cols, b_blocks):
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


def build_case_matrix(
    spec: BenchmarkSpec,
    comm: Any,
    rank: int,
    size: int,
) -> dict[str, Any]:
    """Graph, values and matrix for one (domain, size) point.

    The operations of a domain all run against the same matrix, and building it
    dominates a large run -- so the result is memoized on everything it depends
    on, and the three operations of a domain share one build instead of
    repeating it. The graph/assembly/atomistic timings therefore become a
    per-domain measurement rather than a per-operation one.
    """
    key = (spec.domain, spec.blocks, str(spec.dtype), spec.bsr_block_size, spec.seed,
           spec.target_degree, spec.geometry_ordering, size)
    cached = _CASE_MATRIX_CACHE.get(key)
    if cached is not None:
        return cached
    # Evict first: never hold two large matrices at once.
    _CASE_MATRIX_CACHE.clear()

    block_sizes = make_block_sizes(spec)
    owned_range = partition_range(spec.blocks, size, rank)
    flat_local = None
    if _GEOMETRY_CACHE_DIR is not None:
        adjacency, adjacency_info, positions, box_lengths, flat_local = cached_geometric_adjacency(
            spec, owned_range, comm, rank, _GEOMETRY_CACHE_DIR)
    else:
        adjacency, adjacency_info, positions, box_lengths = make_geometric_adjacency(spec, owned_range)
    adjacency_info = reduce_degree_statistics(adjacency_info, comm)
    if spec.value_fill == "random":
        # fill_random values are not drawn from the distance-decay model, so
        # its statistics would be metadata about values the run does not use —
        # and the per-row Python loop below costs ~310 s per point on the
        # 6.5M-row scalar-CSR domain (untimed, once per worker count).
        value_model = {
            "model": "uniform_random",
            "block_random_normalization": "1/sqrt(row_block_size * column_block_size)",
            "note": "decay-model statistics skipped under --value-fill random",
        }
    else:
        value_model = matrix_value_model_statistics(
            adjacency, positions, box_lengths, spec, owned_range[0], comm
        )
    matrix, build_timings = build_matrix(
        spec, block_sizes, adjacency, positions, box_lengths, comm, rank, size,
        flat_local=flat_local)
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
    built = {
        "block_sizes": block_sizes,
        "adjacency": adjacency,
        "adjacency_info": adjacency_info,
        "positions": positions,
        "box_lengths": box_lengths,
        "value_model": value_model,
        "matrix": matrix,
        "build_timings": build_timings,
        "atomistic_conversion": atomistic_conversion,
        "matrix_statistics": matrix_statistics(matrix, block_sizes, adjacency, spec, comm, rank, size),
    }
    _CASE_MATRIX_CACHE[key] = built
    return built


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
    no_baselines: bool = False,
) -> dict[str, Any]:
    built = build_case_matrix(spec, comm, rank, size)
    block_sizes = built["block_sizes"]
    adjacency = built["adjacency"]
    adjacency_info = built["adjacency_info"]
    positions = built["positions"]
    box_lengths = built["box_lengths"]
    value_model = built["value_model"]
    matrix = built["matrix"]
    build_timings = built["build_timings"]
    atomistic_conversion = built["atomistic_conversion"]
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
            "geometry_ordering": spec.geometry_ordering,
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
        "matrix": built["matrix_statistics"],
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
    if no_baselines:
        # VBCSR-only timing (scaling studies): skip the scipy/MKL baselines
        # and the serial validation multiply. Correctness is covered by the
        # gated test suite; here we measure only the library op.
        result["baselines"]["scipy_csr"] = {"available": False, "reason": "no_baselines"}
        result["baselines"]["mkl_sparse"] = {"available": False, "reason": "no_baselines"}
    elif size == 1:
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

    if spec.operation == "spgemm" and not no_baselines:
        output = matrix.spmm(matrix, spec.spgemm_threshold)
        result["spgemm"] = {
            "threshold": spec.spgemm_threshold,
            "output_matrix_kind": output.matrix_kind,
            "output_block_nnz_sum": int(reduce_value(comm, output.local_block_nnz, "sum")),
            "output_scalar_nnz_sum": int(reduce_value(comm, output.local_nnz, "sum")),
            "fill_ratio_scalar": float(reduce_value(comm, output.local_nnz, "sum") / max(result["matrix"]["global_scalar_nnz"], 1)),
            "threshold_audit": spgemm_threshold_audit(adjacency, block_sizes, spec, positions, box_lengths, size),
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


def gather_rank_hosts(comm: Any, rank: int) -> list[dict[str, Any]] | None:
    # Collective: every rank must call it. Kept out of the rank-0-only
    # finalization block so it cannot deadlock (root waiting on a gather the
    # other ranks never enter).
    if comm is None:
        return [{"rank": rank, "hostname": socket.gethostname()}]
    gathered = comm.gather({"rank": rank, "hostname": socket.gethostname()}, root=0)
    return gathered if rank == 0 else None


def mpi_metadata(comm: Any, rank: int, size: int, rank_hosts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    result = {"rank_count": size, "mpi4py_available": MPI is not None, "python_reductions_available": comm is not None}
    if MPI is not None:
        result["mpi_library_version"] = MPI.Get_library_version()
        result["processor_name"] = MPI.Get_processor_name()
    if MPI_FALLBACK_STATUS:
        result["fallback_status"] = dict(MPI_FALLBACK_STATUS)
    if rank == 0:
        result["rank_hosts"] = rank_hosts or [{"rank": rank, "hostname": socket.gethostname()}]
    return result


def environment_metadata(comm: Any, rank: int, size: int, rank_hosts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    thread_vars = (
        "OMP_NUM_THREADS",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "OMP_DYNAMIC",
        "SPARSE_DOT_MKL_NUM_THREADS",
        "MKL_NUM_THREADS",
        "MKL_THREADING_LAYER",
        "MKL_DYNAMIC",
        "KMP_AFFINITY",
        "GOMP_CPU_AFFINITY",
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
        "mpi": mpi_metadata(comm, rank, size, rank_hosts),
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


def mean_block_area(domain: str, args: argparse.Namespace) -> float:
    """Expected scalar elements per stored block, for the domain's size model."""
    if domain == "csr":
        return 1.0
    if domain == "bsr":
        return float(args.bsr_block_size) ** 2
    if domain == "vbcsr":
        # Row and column sizes are drawn independently, so E[r*c] = E[r]*E[c].
        return float(np.mean(VBCSR_BLOCK_SIZES)) ** 2
    raise ValueError(f"unknown domain {domain!r}")


def calibrate_mean_degree(args: argparse.Namespace, dtype: np.dtype) -> float:
    """Mean block degree of the cutoff graph, measured on a small probe.

    The cutoff is calibrated to `--target-degree` at a quantile, so the mean
    degree is essentially independent of the block count. Probing once on a
    cheap graph avoids sizing the real runs from a guessed degree.
    """
    probe = BenchmarkSpec(
        suite=args.suite,
        domain="csr",
        operation="spmv",
        blocks=min(20000, max(1000, int(args.blocks))),
        target_degree=int(args.target_degree),
        rhs=int(args.rhs),
        dtype=dtype,
        seed=int(args.seed),
        bsr_block_size=int(args.bsr_block_size),
        spgemm_threshold=0.0,
        spgemm_audit_limit=0,
        geometry_dim=int(args.geometry_dim),
        geometry_spacing=float(args.geometry_spacing),
        geometry_jitter=float(args.geometry_jitter),
        geometry_ordering=str(args.geometry_ordering),
        geometry_cutoff=args.geometry_cutoff,
        geometry_cutoff_quantile=float(args.geometry_cutoff_quantile),
        magnitude_decay_length=float(args.magnitude_decay_length),
        offdiagonal_scale=float(args.offdiagonal_scale),
        diagonal_shift=float(args.diagonal_shift),
    )
    _, info, _, _ = make_geometric_adjacency(probe)
    return max(float(info["local_degree_sum"] / max(info["local_row_count"], 1)), 1.0)


def blocks_for_target_storage(domain: str, target_bytes: int, mean_degree: float, args: argparse.Namespace, dtype: np.dtype) -> int:
    """Block count putting `domain` at roughly `target_bytes` of matrix storage.

    Comparing domains at a fixed *block count* compares wildly different
    problems: with 8x8 BSR blocks and ~14x14 VBCSR blocks, the same block count
    spans a 130x range in footprint, and scalar CSR stays entirely inside L3
    while the others are far into DRAM. Sizing by footprint puts all three at
    the same point on the memory hierarchy.
    """
    # Matches what `matrix_statistics` reports: values plus the CSR index
    # arrays (int32 column index per block, int32 row pointer per block row).
    # The indices are noise for blocked domains but ~50% of a scalar CSR
    # matrix, so leaving them out would oversize CSR by half.
    bytes_per_block = mean_degree * (mean_block_area(domain, args) * dtype.itemsize + 4) + 4
    return max(1, int(round(target_bytes / bytes_per_block)))


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

    selected_domains = [d.strip() for d in str(args.domains).split(",") if d.strip()]
    unknown = [d for d in selected_domains if d not in DOMAINS]
    if unknown:
        raise ValueError(f"unknown domain(s) {unknown}; choose from {list(DOMAINS)}")
    if not selected_domains:
        raise ValueError("--domains selected nothing")

    target_bytes = int(args.target_storage_bytes) if args.target_storage_bytes else 0
    blocks_by_domain: dict[str, int] = {}
    if target_bytes > 0:
        if args.suite == "distributed-weak":
            raise ValueError("--target-storage-bytes sizes a fixed global problem; use --weak-blocks-per-rank for the weak suite")
        mean_degree = calibrate_mean_degree(args, dtype)
        blocks_by_domain = {
            domain: blocks_for_target_storage(domain, target_bytes, mean_degree, args, dtype)
            for domain in selected_domains
        }

    specs: list[BenchmarkSpec] = []
    selected_operations = parse_operations(args.operations)

    for domain in selected_domains:
        domain_blocks = blocks_by_domain.get(domain, blocks)
        for operation in selected_operations:
            thresholds = spgemm_thresholds if operation == "spgemm" else [float(args.spgemm_threshold)]
            for threshold in thresholds:
                specs.append(
                    BenchmarkSpec(
                        suite=args.suite,
                        domain=domain,
                        operation=operation,
                        blocks=domain_blocks,
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
                        geometry_ordering=str(args.geometry_ordering),
                        geometry_cutoff=args.geometry_cutoff,
                        geometry_cutoff_quantile=float(args.geometry_cutoff_quantile),
                        magnitude_decay_length=float(args.magnitude_decay_length),
                        offdiagonal_scale=float(args.offdiagonal_scale),
                        diagonal_shift=float(args.diagonal_shift),
                        weak_blocks_per_rank=weak_blocks,
                        value_fill=str(args.value_fill),
                    )
                )
    return specs


def write_outputs(payload: dict[str, Any], output_dir: Path, label: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{label}.json"
    csv_path = output_dir / f"{label}.csv"
    json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    csv_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    json_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(json_tmp, json_path)

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
    with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
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
    os.replace(csv_tmp, csv_path)
    return json_path, csv_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run publication VBCSR benchmark data generation.")
    parser.add_argument("--suite", choices=("efficiency", "distributed-strong", "distributed-weak"), default="efficiency")
    parser.add_argument("--blocks", type=int, default=4096, help="Global block count for efficiency and strong scaling")
    parser.add_argument("--weak-blocks-per-rank", type=int, default=4096)
    parser.add_argument(
        "--domains",
        default=",".join(DOMAINS),
        help=("comma-separated subset of %s. The weak suite takes a single "
              "--weak-blocks-per-rank for whatever it runs, and a fixed block count is a "
              "very different footprint per domain, so size a weak study by running one "
              "domain per job." % (",".join(DOMAINS),)),
    )
    parser.add_argument("--target-degree", type=int, default=12)
    parser.add_argument(
        "--operations",
        default=",".join(OPERATIONS),
        help="comma-separated subset of spmv,spmm,spgemm",
    )
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
    parser.add_argument(
        "--target-storage-bytes",
        type=int,
        default=0,
        help="size each domain to this many bytes of stored block values instead of using --blocks",
    )
    parser.add_argument("--geometry-ordering", choices=("bisection", "lexicographic"), default="bisection")
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
    parser.add_argument("--no-baselines", action="store_true", help="Time only VBCSR (skip scipy/MKL baselines and validation). Used by scaling studies.")
    parser.add_argument(
        "--geometry-cache-dir",
        type=Path,
        default=SCRIPT_DIR / "results" / "geom_cache",
        help="Directory for the on-disk geometric-structure cache (global adjacency + "
        "positions, keyed by the generating parameters). The same structure is reused "
        "across every thread/rank count of a sweep instead of being regenerated.",
    )
    parser.add_argument(
        "--no-geometry-cache",
        action="store_true",
        help="Regenerate the geometric structure in-process instead of using the on-disk cache.",
    )
    parser.add_argument(
        "--value-fill",
        choices=("physical", "random"),
        default="physical",
        help="Matrix value source: 'physical' assembles decay-model blocks from Python "
        "(required for validation / thresholded SpGEMM); 'random' fills values in one "
        "parallel C++ pass, eliminating assembly cost (timing-only scaling runs).",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--label", default=None)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    comm, rank, size = mpi_context()

    global _GEOMETRY_CACHE_DIR
    _GEOMETRY_CACHE_DIR = None if args.no_geometry_cache else Path(args.geometry_cache_dir)

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
    if args.suite == "efficiency" and comm is not None:
        # Single-rank efficiency timing must not depend on whether mpi4py is
        # installed: with a live communicator the timing loop does a real
        # Allreduce per iteration (inflates microsecond-scale ops) and the
        # csr/spgemm vendor-threading special case (gated on comm is None)
        # is skipped. Match the historical comm=None behavior exactly.
        comm = None

    specs = build_specs(args, size)
    label = args.label.strip() if isinstance(args.label, str) else args.label
    if label is None or label == "":
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"vbcsr_publication_{args.suite}_np{size}_{stamp}"

    # Collective: all ranks participate before rank 0 builds the payload.
    # The rank-0 payload is written after every completed case, so a later
    # memory kill in a heavier operation does not discard earlier timings.
    rank_hosts = gather_rank_hosts(comm, rank)
    payload: dict[str, Any] | None = None
    if rank == 0:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "metadata": environment_metadata(comm, rank, size, rank_hosts),
            "publication_coverage": {
                "domains": [spec.domain for spec in specs],
                "operations": [spec.operation for spec in specs],
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
                "partial_write_after_each_case": True,
            },
            "cases": [],
        }

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
            no_baselines=bool(args.no_baselines),
        )
        if rank == 0:
            cases.append(case)
            assert payload is not None
            payload["cases"] = cases
            json_path, csv_path = write_outputs(payload, args.output_dir, label)
            elapsed = time.perf_counter() - case_start
            vbcsr_median = case.get("timings", {}).get("vbcsr", {}).get("median_seconds")
            print(
                f"[{spec.label}] done in {elapsed:.2f} s; "
                f"VBCSR median={vbcsr_median:.6g} s",
                flush=True,
            )
            print(f"[{spec.label}] checkpointed {json_path} and {csv_path}", flush=True)
            validation = case.get("validation", {})
            if validation.get("exact_validation_required", True) and validation.get("passed") is False:
                print(f"Validation failed for {case['label']}: {validation}", file=sys.stderr)
                return 1

    if rank == 0:
        assert payload is not None
        json_path, csv_path = write_outputs(payload, args.output_dir, label)
        print(f"Wrote {json_path}")
        print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

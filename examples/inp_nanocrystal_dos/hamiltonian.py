"""Distributed VBCSR assembly of the finite InP tight-binding Hamiltonian."""

from dataclasses import asdict, dataclass
from typing import Iterator, Tuple

import numpy as np

from .parameters import (
    ATOMIC_NUMBERS,
    H,
    IN,
    NORB,
    NORB_TO_SYMBOL,
    P,
    InPParameters,
    PAPER_PARAMETERS,
    atomwise_cutoff_radii,
)
from .slater_koster import onsite_block, pair_block


@dataclass(frozen=True)
class AssemblyStatistics:
    local_atoms: int
    local_graph_edges: int
    local_hopping_edges: int
    local_orbitals: int
    spectral_lower_ev: float
    spectral_upper_ev: float

    def to_dict(self):
        return asdict(self)


@dataclass
class HamiltonianAssembly:
    atomic_data: object
    images: object
    matrix: object
    spectral_bounds_ev: Tuple[float, float]
    statistics: AssemblyStatistics


def build_atomic_data(geometry, comm=None, parameters: InPParameters = PAPER_PARAMETERS):
    """Distribute a root-held cluster and build its finite neighbour graph."""
    import vbcsr

    rank = comm.Get_rank() if comm is not None else 0
    if rank == 0:
        positions = geometry.positions
        atomic_numbers = geometry.atomic_numbers
        cell = geometry.cell
    else:
        positions = np.empty((0, 3), dtype=np.float64)
        atomic_numbers = np.empty(0, dtype=np.int32)
        cell = np.empty((3, 3), dtype=np.float64)
    if comm is not None:
        cell = comm.bcast(cell, root=0)

    cutoffs = atomwise_cutoff_radii(parameters)
    return vbcsr.AtomicData.from_points(
        positions,
        atomic_numbers,
        cell,
        [False, False, False],
        {ATOMIC_NUMBERS[symbol]: radius for symbol, radius in cutoffs.items()},
        {ATOMIC_NUMBERS[symbol]: NORB[symbol] for symbol in (H, P, IN)},
        comm=comm,
    )


def _accepted_pair(
    row_symbol: str,
    col_symbol: str,
    distance: float,
    parameters: InPParameters,
) -> bool:
    pair = frozenset((row_symbol, col_symbol))
    tolerance = 5.0e-3
    if pair == frozenset((IN, P)):
        target = parameters.lattice_constant * np.sqrt(3.0) / 4.0
    elif row_symbol == col_symbol and row_symbol in (IN, P):
        target = parameters.lattice_constant / np.sqrt(2.0)
    elif pair == frozenset((IN, H)):
        target = parameters.in_h_bond_length
    else:
        return False
    return abs(distance - target) <= tolerance


def _bond_groups(
    edge_index: np.ndarray,
    edge_vectors: np.ndarray,
    block_sizes: np.ndarray,
    parameters: InPParameters,
) -> Iterator[Tuple[np.ndarray, int, int, Tuple[int, int, int]]]:
    """Classify one local edge chunk without a Python loop over its edges.

    Zinc-blende InP has only tetrahedral first-neighbour directions and
    face-diagonal second-neighbour directions.  Encoding their normalized
    components as {-1, 0, 1} creates a small block codebook: SK rotations are
    evaluated once per class, while NumPy performs the O(n_edge) work.
    """
    if edge_index.shape[0] == 0:
        return

    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    row_sizes = block_sizes[src]
    col_sizes = block_sizes[dst]
    distance = np.sqrt(np.einsum("ij,ij->i", edge_vectors, edge_vectors))

    is_in_p = ((row_sizes == NORB[IN]) & (col_sizes == NORB[P])) | (
        (row_sizes == NORB[P]) & (col_sizes == NORB[IN])
    )
    is_second = ((row_sizes == NORB[IN]) & (col_sizes == NORB[IN])) | (
        (row_sizes == NORB[P]) & (col_sizes == NORB[P])
    )
    is_in_h = ((row_sizes == NORB[IN]) & (col_sizes == NORB[H])) | (
        (row_sizes == NORB[H]) & (col_sizes == NORB[IN])
    )

    target = np.zeros_like(distance)
    target[is_in_p] = parameters.lattice_constant * np.sqrt(3.0) / 4.0
    target[is_second] = parameters.lattice_constant / np.sqrt(2.0)
    target[is_in_h] = parameters.in_h_bond_length
    accepted = (is_in_p | is_second | is_in_h) & (np.abs(distance - target) <= 5.0e-3)
    accepted_ids = np.flatnonzero(accepted)
    if accepted_ids.size == 0:
        return

    accepted_vectors = edge_vectors[accepted_ids]
    accepted_distance = distance[accepted_ids]
    direction_scale = np.where(is_second[accepted_ids], np.sqrt(2.0), np.sqrt(3.0))
    directions = np.rint(
        accepted_vectors / accepted_distance[:, None] * direction_scale[:, None]
    ).astype(np.int8)
    if np.any(np.abs(directions) > 1) or np.any(np.all(directions == 0, axis=1)):
        raise RuntimeError("failed to encode a zinc-blende bond direction")

    # Pair identity plus a base-3 direction code. The range is compact, so a
    # stable integer sort groups all equal blocks without object arrays/dicts.
    direction_code = (
        (directions[:, 0].astype(np.int32) + 1) * 9
        + (directions[:, 1].astype(np.int32) + 1) * 3
        + directions[:, 2].astype(np.int32)
        + 1
    )
    pair_code = (
        row_sizes[accepted_ids].astype(np.int32) * 16
        + col_sizes[accepted_ids].astype(np.int32)
    )
    code = pair_code * 27 + direction_code
    order = np.argsort(code, kind="stable")
    sorted_ids = accepted_ids[order]
    sorted_code = code[order]
    boundaries = np.concatenate(
        (
            np.asarray((0,), dtype=np.int64),
            np.flatnonzero(sorted_code[1:] != sorted_code[:-1]) + 1,
            np.asarray((sorted_ids.size,), dtype=np.int64),
        )
    )
    sorted_directions = directions[order]
    for group in range(boundaries.size - 1):
        begin = int(boundaries[group])
        end = int(boundaries[group + 1])
        representative = sorted_ids[begin]
        yield (
            sorted_ids[begin:end],
            int(row_sizes[representative]),
            int(col_sizes[representative]),
            tuple(int(value) for value in sorted_directions[begin]),
        )


def _global_minmax(comm, lower: float, upper: float) -> Tuple[float, float]:
    if comm is None:
        return lower, upper
    from mpi4py import MPI

    return comm.allreduce(lower, op=MPI.MIN), comm.allreduce(upper, op=MPI.MAX)


def build_hamiltonian(
    atomic_data,
    parameters: InPParameters = PAPER_PARAMETERS,
    chunk_size: int = 100_000,
    comm=None,
) -> HamiltonianAssembly:
    """Assemble onsite, nearest-neighbour, and next-nearest-neighbour blocks.

    The public batched ``ImageContainer.add_blocks`` path performs insertion;
    edge-vector evaluation and Slater--Koster rotation remain in Python so the
    example is readable and parameter changes do not require recompilation.
    """
    import vbcsr

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    graph = atomic_data.graph
    owned_gids = np.asarray(graph.owned_global_indices, dtype=np.int64)
    ghost_gids = np.asarray(graph.ghost_global_indices, dtype=np.int64)
    local_gids = np.concatenate((owned_gids, ghost_gids))
    block_sizes = np.asarray(graph.block_sizes, dtype=np.int64)
    n_owned = owned_gids.size
    owned_sizes = block_sizes[:n_owned]
    scalar_offsets = np.concatenate(([0], np.cumsum(owned_sizes, dtype=np.int64)))

    images = vbcsr.ImageContainer(atomic_data, dtype=np.float64)
    centers = np.empty(int(scalar_offsets[-1]), dtype=np.float64)
    for size in np.unique(owned_sizes):
        symbol = NORB_TO_SYMBOL[int(size)]
        atom_ids = np.flatnonzero(owned_sizes == size)
        block = onsite_block(symbol, parameters)
        scalar_rows = scalar_offsets[atom_ids, None] + np.arange(int(size), dtype=np.int64)
        centers[scalar_rows] = parameters.onsite[symbol]
        for begin in range(0, atom_ids.size, chunk_size):
            batch = atom_ids[begin : begin + chunk_size]
            gids = owned_gids[batch].tolist()
            images.add_blocks(gids, gids, [block] * batch.size, mode="insert")
    radii = np.zeros_like(centers)

    edge_index = np.asarray(atomic_data.edge_index, dtype=np.int64)
    edge_vectors = np.asarray(atomic_data.edge_vectors, dtype=np.float64)
    accepted_total = 0
    block_cache = {}

    for begin in range(0, edge_index.shape[0], chunk_size):
        end = min(begin + chunk_size, edge_index.shape[0])
        chunk_edges = edge_index[begin:end]
        if chunk_edges.size and np.any(chunk_edges[:, 0] >= n_owned):
            raise RuntimeError("AtomicData edge rows must be locally owned")
        for local_ids, row_size, col_size, direction in _bond_groups(
            chunk_edges, edge_vectors[begin:end], block_sizes, parameters
        ):
            src = chunk_edges[local_ids, 0]
            dst = chunk_edges[local_ids, 1]
            row_symbol = NORB_TO_SYMBOL[row_size]
            col_symbol = NORB_TO_SYMBOL[col_size]
            block_key = (row_symbol, col_symbol, direction)
            cached = block_cache.get(block_key)
            if cached is None:
                block = pair_block(row_symbol, col_symbol, direction, parameters)
                cached = (block, np.abs(block).sum(axis=1))
                block_cache[block_key] = cached
            block, row_abs_sums = cached
            images.add_blocks(
                local_gids[src].tolist(),
                local_gids[dst].tolist(),
                [block] * local_ids.size,
                mode="insert",
            )

            scalar_rows = scalar_offsets[src, None] + np.arange(block.shape[0], dtype=np.int64)
            np.add.at(radii, scalar_rows.ravel(), np.tile(row_abs_sums, local_ids.size))
            accepted_total += int(local_ids.size)

    images.assemble()

    local_lower = float(np.min(centers - radii)) if centers.size else np.inf
    local_upper = float(np.max(centers + radii)) if centers.size else -np.inf
    lower, upper = _global_minmax(comm, local_lower, local_upper)
    # Keep all rescaled eigenvalues strictly inside (-1, 1), where the DOS
    # reconstruction denominator is finite.
    half_width = 0.5 * (upper - lower)
    padding = max(0.01 * half_width, 1.0e-6)
    lower -= padding
    upper += padding

    matrix = images.sample_k([0.0, 0.0, 0.0], convention="R", symm=False)
    stats = AssemblyStatistics(
        local_atoms=int(n_owned),
        local_graph_edges=int(edge_index.shape[0]),
        local_hopping_edges=accepted_total,
        local_orbitals=int(scalar_offsets[-1]),
        spectral_lower_ev=lower,
        spectral_upper_ev=upper,
    )
    return HamiltonianAssembly(atomic_data, images, matrix, (lower, upper), stats)

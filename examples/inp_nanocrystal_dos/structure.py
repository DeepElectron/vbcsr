"""Exact shell-by-shell zinc-blende InP nanocrystals.

The paper starts from a central In atom and adds alternating tetrahedral
coordination shells.  An even shell ``2*K`` is In terminated.  The formulas
and generator below reproduce shell 22 exactly: 5,083 In + 4,444 P atoms and
60,328 core orbitals before passivation.
"""

from dataclasses import asdict, dataclass
from typing import Iterator, Tuple

import numpy as np

from .parameters import ATOMIC_NUMBERS, H, IN, NORB, P, InPParameters, PAPER_PARAMETERS


TETRAHEDRAL_DIRECTIONS = np.asarray(
    ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)), dtype=np.int32
)


@dataclass(frozen=True)
class ClusterCounts:
    shells: int
    in_atoms: int
    p_atoms: int
    h_atoms: int
    core_atoms: int
    total_atoms: int
    core_orbitals: int
    total_orbitals: int
    effective_diameter_angstrom: float

    def to_dict(self):
        return asdict(self)


@dataclass
class ClusterGeometry:
    positions: np.ndarray
    atomic_numbers: np.ndarray
    input_indices: np.ndarray
    cell: np.ndarray
    counts: ClusterCounts


def _validate_shells(shells: int) -> int:
    shells = int(shells)
    if shells < 0 or shells % 2:
        raise ValueError("shells must be a non-negative even integer")
    return shells // 2


def cluster_counts(
    shells: int,
    passivate: bool = True,
    parameters: InPParameters = PAPER_PARAMETERS,
) -> ClusterCounts:
    """Return exact counts without constructing coordinates."""
    k = _validate_shells(shells)
    n_in = 1 + 10 * k * (k + 1) * (2 * k + 1) // 6 + 2 * k
    n_p = 10 * k * (k + 1) * (k - 1) // 3 + 4 * k
    n_h = (20 * k * k + 12 * k + 4) if passivate and k > 0 else 0
    n_core = n_in + n_p
    n_total = n_core + n_h
    core_orbitals = NORB[IN] * n_in + NORB[P] * n_p
    total_orbitals = core_orbitals + NORB[H] * n_h
    diameter = parameters.lattice_constant * (3.0 * n_core / (4.0 * np.pi)) ** (1.0 / 3.0)
    return ClusterCounts(
        shells=int(shells),
        in_atoms=n_in,
        p_atoms=n_p,
        h_atoms=n_h,
        core_atoms=n_core,
        total_atoms=n_total,
        core_orbitals=core_orbitals,
        total_orbitals=total_orbitals,
        effective_diameter_angstrom=float(diameter),
    )


def shell_for_target_atoms(target_core_atoms: int) -> int:
    """Smallest even shell with at least ``target_core_atoms`` core atoms."""
    target = int(target_core_atoms)
    if target <= 1:
        return 0
    lo, hi = 0, 1
    while cluster_counts(2 * hi, passivate=False).core_atoms < target:
        hi *= 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if cluster_counts(2 * mid, passivate=False).core_atoms < target:
            lo = mid
        else:
            hi = mid
    return 2 * hi


def _fcc_distance(u: Tuple[int, int, int]) -> int:
    ax, ay, az = abs(u[0]), abs(u[1]), abs(u[2])
    return max(ax, ay, az, (ax + ay + az) // 2)


def _in_sites(k: int) -> Iterator[Tuple[int, int, int]]:
    for x in range(-k, k + 1):
        for y in range(-k, k + 1):
            for z in range(-k, k + 1):
                u = (x, y, z)
                if (x + y + z) % 2 == 0 and _fcc_distance(u) <= k:
                    yield u


def _p_distance_from_representative(u: Tuple[int, int, int]) -> int:
    # P position p = 2*u + (1,1,1), in units a/4.  Its four In neighbours
    # have fcc coordinates u, u+(0,1,1), u+(1,0,1), u+(1,1,0).
    x, y, z = u
    return min(
        _fcc_distance((x, y, z)),
        _fcc_distance((x, y + 1, z + 1)),
        _fcc_distance((x + 1, y, z + 1)),
        _fcc_distance((x + 1, y + 1, z)),
    )


def _p_sites(k: int) -> Iterator[Tuple[int, int, int]]:
    if k == 0:
        return
    # Representative fcc coordinates for p = 2*u + (1,1,1).
    for x in range(-k - 1, k + 1):
        for y in range(-k - 1, k + 1):
            for z in range(-k - 1, k + 1):
                if (x + y + z) % 2:
                    continue
                u = (x, y, z)
                if _p_distance_from_representative(u) <= k - 1:
                    yield u


def _missing_p_directions(u: Tuple[int, int, int], k: int) -> Iterator[int]:
    x, y, z = u
    representative_shifts = ((0, 0, 0), (0, -1, -1), (-1, 0, -1), (-1, -1, 0))
    for direction_index, shift in enumerate(representative_shifts):
        rep = (x + shift[0], y + shift[1], z + shift[2])
        if _p_distance_from_representative(rep) > k - 1:
            yield direction_index


def generate_cluster(
    shells: int,
    passivate: bool = True,
    parameters: InPParameters = PAPER_PARAMETERS,
) -> ClusterGeometry:
    """Generate the exact finite cluster on one process.

    The returned positions are translated into a non-periodic orthorhombic box.
    ``AtomicData.from_points`` subsequently distributes and partitions them.
    """
    k = _validate_shells(shells)
    counts = cluster_counts(shells, passivate=passivate, parameters=parameters)
    a = parameters.lattice_constant

    # Preallocation matters for the million-atom demonstrations: retaining a
    # Python tuple and integer object per coordinate would otherwise dominate
    # the root process's memory before VBCSR has a chance to distribute it.
    pos = np.empty((counts.total_atoms, 3), dtype=np.float64)
    z = np.empty(counts.total_atoms, dtype=np.int32)
    in_sites = np.empty((counts.in_atoms, 3), dtype=np.int32)
    cursor = 0
    for u in _in_sites(k):
        in_sites[cursor] = u
        pos[cursor] = 0.5 * a * np.asarray(u)
        z[cursor] = ATOMIC_NUMBERS[IN]
        cursor += 1
    if cursor != counts.in_atoms:
        raise RuntimeError(f"generated {cursor} In sites, expected {counts.in_atoms}")

    for u in _p_sites(k):
        pos[cursor] = 0.25 * a * (2.0 * np.asarray(u) + 1.0)
        z[cursor] = ATOMIC_NUMBERS[P]
        cursor += 1
    if cursor != counts.in_atoms + counts.p_atoms:
        raise RuntimeError(
            f"generated {cursor - counts.in_atoms} P sites, expected {counts.p_atoms}"
        )

    if passivate and k > 0:
        unit_directions = TETRAHEDRAL_DIRECTIONS / np.sqrt(3.0)
        for u_array in in_sites:
            u = tuple(int(value) for value in u_array)
            if _fcc_distance(u) != k:
                continue
            parent = 0.5 * a * np.asarray(u, dtype=np.float64)
            for direction_index in _missing_p_directions(u, k):
                pos[cursor] = (
                    parent + parameters.in_h_bond_length * unit_directions[direction_index]
                )
                z[cursor] = ATOMIC_NUMBERS[H]
                cursor += 1

    if cursor != counts.total_atoms:
        raise RuntimeError(
            "generated atom count does not match the closed-form count: "
            f"{cursor} != {counts.total_atoms}"
        )

    margin = max(5.0, parameters.lattice_constant)
    if pos.size:
        lower = pos.min(axis=0)
        upper = pos.max(axis=0)
        pos -= lower - margin
        lengths = upper - lower + 2.0 * margin
    else:
        lengths = np.full(3, 2.0 * margin)
    cell = np.diag(lengths)
    indices = np.arange(pos.shape[0], dtype=np.int32)
    return ClusterGeometry(pos, z, indices, cell, counts)


def write_extxyz(filename: str, geometry: ClusterGeometry) -> None:
    """Write the generated geometry for inspection; intended for modest sizes."""
    from ase import Atoms

    atoms = Atoms(
        numbers=geometry.atomic_numbers,
        positions=geometry.positions,
        cell=geometry.cell,
        pbc=False,
    )
    atoms.info.update(geometry.counts.to_dict())
    atoms.write(filename, format="extxyz")


def as_vasp_atoms(geometry: ClusterGeometry, vacuum: float = 10.0):
    """Return a centered periodic ASE cell suitable for a finite-dot POSCAR."""
    from ase import Atoms

    vacuum = float(vacuum)
    if vacuum <= 0.0:
        raise ValueError("VASP vacuum padding must be positive")
    positions = np.asarray(geometry.positions, dtype=np.float64).copy()
    lower = positions.min(axis=0)
    upper = positions.max(axis=0)
    positions += vacuum - lower
    cell = np.diag(upper - lower + 2.0 * vacuum)
    atoms = Atoms(
        numbers=geometry.atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=True,
    )
    atoms.info.update(geometry.counts.to_dict())
    atoms.info["vacuum_angstrom"] = vacuum
    return atoms


def write_poscar(filename: str, geometry: ClusterGeometry, vacuum: float = 10.0) -> None:
    """Write a VASP-5 POSCAR with Cartesian coordinates and explicit species."""
    atoms = as_vasp_atoms(geometry, vacuum=vacuum)
    atoms.write(filename, format="vasp", direct=False, sort=False, vasp5=True)

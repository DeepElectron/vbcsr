"""Exact small-cluster validation for the InP nanocrystal DOS example.

This complements the scalable KPM driver.  It diagonalizes several modest
members of the same shell family, checks the bonding/antibonding gap, and
reports how the size trend compares with the fit printed by Sapra et al.
The fit comparison is diagnostic rather than an exact regression target: the
paper plots its finite-cluster points and does not uniquely specify the In-H
block.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import eigh

from .hamiltonian import build_atomic_data, build_hamiltonian
from .structure import ClusterCounts, generate_cluster


PAPER_BULK_GAP_EV = 1.4
# Direct Gamma gap obtained from the rounded parameters printed in Table I.
# The paper's fit is a gap *shift*, so comparing absolute gaps without first
# removing this table-rounding offset would introduce a spurious 0.189 eV.
ROUNDED_TABLE_BULK_GAP_EV = 1.589033817101524
INP_DIELECTRIC_CONSTANT = 12.4
COULOMB_EV_NM = 1.43996454784255


@dataclass(frozen=True)
class SizeValidation:
    shells: int
    diameter_nm: float
    core_atoms: int
    total_atoms: int
    total_orbitals: int
    occupied_bonding_states: int
    homo_ev: float
    lumo_ev: float
    gap_ev: float
    homo_h_weight: float
    lumo_h_weight: float
    exciton_binding_ev: float
    optical_gap_ev: float
    optical_confinement_shift_ev: float
    paper_fit_confinement_shift_ev: float
    paper_fit_optical_gap_ev: float
    confinement_shift_residual_ev: float

    def to_dict(self):
        return asdict(self)


def occupied_bonding_states(counts: ClusterCounts) -> int:
    """Return the number of filled spinless bonding levels.

    Every fourfold-coordinated P contributes four In-P bonds and every surface
    H terminates one otherwise missing bond.  With spin degeneracy suppressed
    in the Hamiltonian, this gives one occupied bonding level per bond.
    """
    occupied = 4 * counts.p_atoms + counts.h_atoms
    if occupied <= 0 or occupied >= counts.total_orbitals:
        raise ValueError("the requested cluster has no finite bonding/antibonding gap")
    return occupied


def paper_gap_shift_ev(diameter_nm: float) -> float:
    """Sapra et al.'s printed fit for the optical gap shift from bulk."""
    diameter_nm = float(diameter_nm)
    if diameter_nm <= 0.0:
        raise ValueError("diameter must be positive")
    return 100.0 / (
        5.8 * diameter_nm * diameter_nm + 27.2 * diameter_nm + 10.4
    )


def exciton_binding_ev(diameter_nm: float) -> float:
    """Exciton correction used in the paper, with distance expressed in nm."""
    diameter_nm = float(diameter_nm)
    if diameter_nm <= 0.0:
        raise ValueError("diameter must be positive")
    return 3.572 * COULOMB_EV_NM / (INP_DIELECTRIC_CONSTANT * diameter_nm)


def exact_size_validation(shells: int, max_dense_states: int = 5000) -> SizeValidation:
    """Assemble and exactly extract HOMO/LUMO for one serial cluster."""
    geometry = generate_cluster(shells)
    counts = geometry.counts
    if counts.total_orbitals > int(max_dense_states):
        raise ValueError(
            f"shell {shells} has {counts.total_orbitals} states, exceeding "
            f"--max-dense-states={max_dense_states}"
        )

    atoms = build_atomic_data(geometry)
    assembly = build_hamiltonian(atoms)
    matrix = assembly.matrix.to_dense().real
    if not np.allclose(matrix, matrix.T, atol=1.0e-12):
        raise RuntimeError(f"shell {shells} Hamiltonian is not Hermitian")

    occupied = occupied_bonding_states(counts)
    eigenvalues, eigenvectors = eigh(
        matrix,
        subset_by_index=(occupied - 1, occupied),
        driver="evr",
        check_finite=False,
    )

    sizes = np.asarray(atoms.graph.block_sizes, dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
    h_mask = np.zeros(counts.total_orbitals, dtype=bool)
    h_atoms = np.flatnonzero(sizes == 1)
    for atom in h_atoms:
        h_mask[offsets[atom] : offsets[atom + 1]] = True
    h_weights = np.sum(np.abs(eigenvectors[h_mask]) ** 2, axis=0)

    homo, lumo = (float(value) for value in eigenvalues)
    gap = lumo - homo
    diameter_nm = counts.effective_diameter_angstrom / 10.0
    binding = exciton_binding_ev(diameter_nm)
    optical_gap = gap - binding
    confinement_shift = optical_gap - ROUNDED_TABLE_BULK_GAP_EV
    fit_shift = paper_gap_shift_ev(diameter_nm)
    fit_gap = PAPER_BULK_GAP_EV + fit_shift
    return SizeValidation(
        shells=int(shells),
        diameter_nm=diameter_nm,
        core_atoms=counts.core_atoms,
        total_atoms=counts.total_atoms,
        total_orbitals=counts.total_orbitals,
        occupied_bonding_states=occupied,
        homo_ev=homo,
        lumo_ev=lumo,
        gap_ev=gap,
        homo_h_weight=float(h_weights[0]),
        lumo_h_weight=float(h_weights[1]),
        exciton_binding_ev=binding,
        optical_gap_ev=optical_gap,
        optical_confinement_shift_ev=confinement_shift,
        paper_fit_confinement_shift_ev=fit_shift,
        paper_fit_optical_gap_ev=fit_gap,
        confinement_shift_residual_ev=confinement_shift - fit_shift,
    )


def validate_series(
    shells: Iterable[int], max_dense_states: int = 5000
) -> Sequence[SizeValidation]:
    """Validate a strictly increasing, even-shell size series."""
    selected = tuple(int(value) for value in shells)
    if len(selected) < 2:
        raise ValueError("a size-series validation requires at least two clusters")
    if any(value <= 0 or value % 2 for value in selected):
        raise ValueError("validation shells must be positive even integers")
    if any(right <= left for left, right in zip(selected, selected[1:])):
        raise ValueError("validation shells must be strictly increasing")

    results = tuple(
        exact_size_validation(value, max_dense_states=max_dense_states)
        for value in selected
    )
    if any(right.gap_ev >= left.gap_ev for left, right in zip(results, results[1:])):
        raise RuntimeError("the finite-cluster gap does not decrease monotonically with size")
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact multi-size validation for the Sapra InP nanocrystal example"
    )
    parser.add_argument("--shells", type=int, nargs="+", default=(2, 4, 6, 8))
    parser.add_argument("--max-dense-states", type=int, default=5000)
    parser.add_argument("--json", type=Path, help="optional machine-readable result file")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    results = validate_series(args.shells, max_dense_states=args.max_dense_states)
    print(
        "shell D_nm states occupied gap_eV H_HOMO H_LUMO "
        "optical_gap_eV shift_eV paper_shift_eV shift_residual_eV"
    )
    for result in results:
        print(
            f"{result.shells:5d} {result.diameter_nm:8.5f} "
            f"{result.total_orbitals:6d} {result.occupied_bonding_states:8d} "
            f"{result.gap_ev:9.6f} {result.homo_h_weight:7.4f} "
            f"{result.lumo_h_weight:7.4f} {result.optical_gap_ev:14.6f} "
            f"{result.optical_confinement_shift_ev:8.6f} "
            f"{result.paper_fit_confinement_shift_ev:14.6f} "
            f"{result.confinement_shift_residual_ev:+17.6f}"
        )

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump([result.to_dict() for result in results], handle, indent=2)
            handle.write("\n")
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Export passivated InP nanocrystals as checked VASP POSCAR files."""

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read
from scipy.spatial import cKDTree

from .parameters import PAPER_PARAMETERS
from .structure import generate_cluster, write_poscar


def _nearest_cross_distances(positions, row_mask, col_mask):
    row = positions[row_mask]
    col = positions[col_mask]
    if row.shape[0] == 0 or col.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    return np.asarray(cKDTree(col).query(row, k=1)[0], dtype=np.float64)


def _nearest_same_distances(positions, mask):
    selected = positions[mask]
    if selected.shape[0] < 2:
        return np.empty(0, dtype=np.float64)
    return np.asarray(cKDTree(selected).query(selected, k=2)[0][:, 1], dtype=np.float64)


def _roundtrip_checks(filename: Path, expected_counts, vacuum: float):
    atoms = read(filename, format="vasp")
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    lengths = atoms.cell.lengths()
    lower_clearance = positions.min(axis=0)
    upper_clearance = lengths - positions.max(axis=0)

    actual = {
        "In": int(np.count_nonzero(numbers == 49)),
        "P": int(np.count_nonzero(numbers == 15)),
        "H": int(np.count_nonzero(numbers == 1)),
    }
    expected = {
        "In": expected_counts.in_atoms,
        "P": expected_counts.p_atoms,
        "H": expected_counts.h_atoms,
    }
    if actual != expected:
        raise RuntimeError(f"POSCAR species counts differ: {actual} != {expected}")
    if not np.all(atoms.pbc):
        raise RuntimeError("ASE did not round-trip the POSCAR as a periodic cell")
    if not np.allclose(lower_clearance, vacuum, atol=2.0e-10) or not np.allclose(
        upper_clearance, vacuum, atol=2.0e-10
    ):
        raise RuntimeError("POSCAR vacuum clearance changed during round-trip")

    is_in = numbers == 49
    is_p = numbers == 15
    is_h = numbers == 1
    in_p = _nearest_cross_distances(positions, is_p, is_in)
    in_in = _nearest_same_distances(positions, is_in)
    p_p = _nearest_same_distances(positions, is_p)
    in_h = _nearest_cross_distances(positions, is_h, is_in)

    expected_in_p = PAPER_PARAMETERS.lattice_constant * np.sqrt(3.0) / 4.0
    expected_second = PAPER_PARAMETERS.lattice_constant / np.sqrt(2.0)
    if in_p.size and not np.allclose(in_p, expected_in_p, atol=2.0e-10):
        raise RuntimeError("POSCAR contains a non-zinc-blende nearest In-P distance")
    if in_in.size and not np.allclose(in_in, expected_second, atol=2.0e-10):
        raise RuntimeError("POSCAR contains a non-zinc-blende nearest In-In distance")
    if p_p.size and not np.allclose(p_p, expected_second, atol=2.0e-10):
        raise RuntimeError("POSCAR contains a non-zinc-blende nearest P-P distance")
    if in_h.size and not np.allclose(
        in_h, PAPER_PARAMETERS.in_h_bond_length, atol=2.0e-10
    ):
        raise RuntimeError("POSCAR contains an unexpected nearest In-H distance")

    def distance_range(values):
        if values.size == 0:
            return None
        return [float(values.min()), float(values.max())]

    return {
        "species_counts": actual,
        "cell_lengths_angstrom": [float(value) for value in lengths],
        "lower_clearance_angstrom": [float(value) for value in lower_clearance],
        "upper_clearance_angstrom": [float(value) for value in upper_clearance],
        "nearest_distances_angstrom": {
            "In-P": distance_range(in_p),
            "In-In": distance_range(in_in),
            "P-P": distance_range(p_p),
            "In-H": distance_range(in_h),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write checked VASP POSCARs for the InP nanocrystal shell family"
    )
    parser.add_argument("--shells", type=int, nargs="+", default=(2, 4, 6, 8, 22))
    parser.add_argument("--vacuum", type=float, default=10.0, help="vacuum to each cell face in A")
    parser.add_argument("--output-dir", type=Path, default=Path("inp_vasp_structures"))
    parser.add_argument("--no-passivation", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.vacuum <= 0.0:
        raise ValueError("--vacuum must be positive")
    selected = tuple(int(value) for value in args.shells)
    if len(set(selected)) != len(selected) or any(value <= 0 or value % 2 for value in selected):
        raise ValueError("--shells must be distinct positive even integers")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "VASP 5 POSCAR, Cartesian coordinates",
        "vacuum_angstrom_to_each_face": float(args.vacuum),
        "passivated": not args.no_passivation,
        "lattice_constant_angstrom": PAPER_PARAMETERS.lattice_constant,
        "nominal_in_h_bond_length_angstrom": PAPER_PARAMETERS.in_h_bond_length,
        "structures": [],
    }
    for shells in selected:
        geometry = generate_cluster(shells, passivate=not args.no_passivation)
        directory = args.output_dir / f"shell_{shells:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / "POSCAR"
        write_poscar(str(filename), geometry, vacuum=args.vacuum)
        checks = _roundtrip_checks(filename, geometry.counts, args.vacuum)
        record = {
            "shells": shells,
            "poscar": str(filename.relative_to(args.output_dir)),
            "cluster": geometry.counts.to_dict(),
            "roundtrip_checks": checks,
        }
        manifest["structures"].append(record)
        print(
            f"shell={shells:2d} atoms={geometry.counts.total_atoms:6d} "
            f"orbitals={geometry.counts.total_orbitals:6d} "
            f"cell={checks['cell_lengths_angstrom']} -> {filename}"
        )

    manifest_file = args.output_dir / "manifest.json"
    with manifest_file.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {manifest_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

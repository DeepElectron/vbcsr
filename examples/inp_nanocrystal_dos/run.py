"""Command-line driver for the distributed InP nanocrystal DOS example."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples.inp_nanocrystal_dos.hamiltonian import build_atomic_data, build_hamiltonian
    from examples.inp_nanocrystal_dos.kpm import compute_dos
    from examples.inp_nanocrystal_dos.parameters import BASIS, PAPER_PARAMETERS
    from examples.inp_nanocrystal_dos.structure import (
        cluster_counts,
        generate_cluster,
        shell_for_target_atoms,
        write_extxyz,
    )
else:
    from .hamiltonian import build_atomic_data, build_hamiltonian
    from .kpm import compute_dos
    from .parameters import BASIS, PAPER_PARAMETERS
    from .structure import cluster_counts, generate_cluster, shell_for_target_atoms, write_extxyz


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MPI/VBCSR KPM DOS for the Sapra et al. InP nanocrystal model"
    )
    size = parser.add_mutually_exclusive_group()
    size.add_argument(
        "--shells", type=int, help="even coordination-shell count (default: paper shell 22)"
    )
    size.add_argument(
        "--target-core-atoms",
        type=int,
        help="choose the smallest even shell containing at least this many In+P atoms",
    )
    parser.add_argument("--no-passivation", action="store_true", help="omit surface pseudo-H atoms")
    parser.add_argument("--moments", type=int, default=1024, help="Chebyshev moment count")
    parser.add_argument("--random-vectors", type=int, default=16, help="Hutchinson trace vectors")
    parser.add_argument("--batch-size", type=int, default=8, help="simultaneous trace vectors")
    parser.add_argument("--energy-points", type=int, default=2000, help="DOS output grid size")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--assembly-chunk", type=int, default=100_000)
    parser.add_argument("--output-dir", type=Path, default=Path("inp_nanocrystal_dos_output"))
    parser.add_argument(
        "--write-xyz", action="store_true", help="write structure.extxyz (not advised for millions)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report the selected size without allocating coordinates"
    )
    parser.add_argument(
        "--list-sizes", action="store_true", help="report paper, 10^5, 10^6, and 3x10^6 cases"
    )
    return parser


def _selected_shells(args) -> int:
    if args.target_core_atoms is not None:
        return shell_for_target_atoms(args.target_core_atoms)
    return 22 if args.shells is None else args.shells


def _print_count(counts) -> None:
    print(
        f"shells={counts.shells:3d} core_atoms={counts.core_atoms:,} "
        f"(In={counts.in_atoms:,}, P={counts.p_atoms:,}) H={counts.h_atoms:,} "
        f"orbitals={counts.total_orbitals:,} diameter={counts.effective_diameter_angstrom:.1f} A"
    )


def _max_time(comm, elapsed: float) -> float:
    from mpi4py import MPI

    return float(comm.allreduce(elapsed, op=MPI.MAX))


def _parameters_metadata():
    parameters = PAPER_PARAMETERS
    channels = {
        f"{row}-{col}": {
            f"l{row_l}-l{col_l}-m{m_abs}": value
            for (row_l, col_l, m_abs), value in pair.items()
        }
        for (row, col), pair in parameters.pair_channels.items()
    }
    return {
        "source": "Sapra, Viswanatha, and Sarma, J. Phys. D 36, 1595 (2003)",
        "source_url": "https://arxiv.org/abs/cond-mat/0308038",
        "lattice_constant_angstrom": parameters.lattice_constant,
        "nominal_in_h_bond_length_angstrom": parameters.in_h_bond_length,
        "basis_order": {symbol: list(labels) for symbol, labels in BASIS.items()},
        "onsite_ev": {symbol: values.tolist() for symbol, values in parameters.onsite.items()},
        "builder_radial_channels_ev": channels,
        "slater_koster_sign_note": (
            "The real-harmonic builder applies odd-parity direction signs. In particular, "
            "its +1.63 eV p(In)-s(P) radial channel produces the -1.63 eV aligned "
            "matrix element printed in Table I."
        ),
        "paper_in_h_values_ev": list(parameters.paper_in_h_values),
        "passivation_note": parameters.passivation_note,
    }


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    shells = _selected_shells(args)
    passivate = not args.no_passivation

    if args.list_sizes:
        if rank == 0:
            _print_count(cluster_counts(22, passivate=passivate))
            for target in (100_000, 1_000_000, 3_000_000):
                _print_count(
                    cluster_counts(shell_for_target_atoms(target), passivate=passivate)
                )
        return 0

    counts = cluster_counts(shells, passivate=passivate)
    if rank == 0:
        _print_count(counts)
    if args.dry_run:
        return 0

    start = time.perf_counter()
    geometry = generate_cluster(shells, passivate=passivate) if rank == 0 else None
    generation_seconds = _max_time(comm, time.perf_counter() - start)

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.write_xyz:
            write_extxyz(str(args.output_dir / "structure.extxyz"), geometry)
    comm.Barrier()

    start = time.perf_counter()
    atomic_data = build_atomic_data(geometry, comm=comm)
    graph_seconds = _max_time(comm, time.perf_counter() - start)

    start = time.perf_counter()
    assembly = build_hamiltonian(
        atomic_data,
        chunk_size=args.assembly_chunk,
        comm=comm,
    )
    assembly_seconds = _max_time(comm, time.perf_counter() - start)
    if rank == 0:
        print(
            "Hamiltonian assembled; Gershgorin bounds "
            f"[{assembly.spectral_bounds_ev[0]:.6f}, {assembly.spectral_bounds_ev[1]:.6f}] eV"
        )

    start = time.perf_counter()

    def progress(done, total):
        if rank == 0:
            print(f"KPM trace vectors: {done}/{total}", flush=True)

    result = compute_dos(
        assembly.matrix,
        assembly.spectral_bounds_ev,
        num_moments=args.moments,
        num_random_vectors=args.random_vectors,
        batch_size=args.batch_size,
        num_energy_points=args.energy_points,
        seed=args.seed,
        comm=comm,
        progress=progress,
    )
    kpm_seconds = _max_time(comm, time.perf_counter() - start)

    local_stats = assembly.statistics.to_dict()
    totals = {
        key: int(comm.allreduce(local_stats[key], op=MPI.SUM))
        for key in (
            "local_atoms",
            "local_graph_edges",
            "local_hopping_edges",
            "local_orbitals",
        )
    }
    if rank == 0:
        table = np.column_stack(
            (
                result.energy_ev,
                result.dos_states_per_ev,
                2.0 * result.dos_states_per_ev,
                result.dos_per_orbital_per_ev,
                result.integration_weights_ev,
            )
        )
        np.savetxt(
            args.output_dir / "dos.dat",
            table,
            header=(
                "energy_eV DOS_single_spin_states_per_eV "
                "DOS_spin_degenerate_states_per_eV DOS_per_orbital_per_eV "
                "gauss_chebyshev_integration_weight_eV"
            ),
        )
        np.savez(
            args.output_dir / "moments.npz",
            moments=result.moments,
            jackson_weights=result.jackson_weights,
            spectral_bounds_ev=np.asarray(result.spectral_bounds_ev),
        )
        metadata = {
            "cluster": counts.to_dict(),
            "parameters": _parameters_metadata(),
            "assembly": {
                "chunk_size": args.assembly_chunk,
            },
            "kpm": {
                "moments": args.moments,
                "random_vectors": args.random_vectors,
                "batch_size": args.batch_size,
                "energy_points": args.energy_points,
                "seed": args.seed,
                "resolution_ev_approx": result.resolution_ev,
                "dos_is_single_spin": True,
                "energy_grid": "ascending Gauss-Chebyshev nodes",
            },
            "distributed_totals": totals,
            "mpi_ranks": comm.Get_size(),
            "timings_seconds_max_rank": {
                "coordinate_generation": generation_seconds,
                "graph_construction": graph_seconds,
                "hamiltonian_assembly": assembly_seconds,
                "kpm": kpm_seconds,
            },
        }
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"Wrote {args.output_dir / 'dos.dat'}; approximate Jackson resolution "
            f"{result.resolution_ev:.4f} eV"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

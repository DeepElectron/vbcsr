import copy
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from ase.io import read

import _workspace_bootstrap

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from examples.inp_nanocrystal_dos.hamiltonian import (
    _accepted_pair,
    build_atomic_data,
    build_hamiltonian,
)
from examples.inp_nanocrystal_dos.kpm import compute_dos
from examples.inp_nanocrystal_dos.parameters import IN, P, NORB_TO_SYMBOL, PAPER_PARAMETERS
from examples.inp_nanocrystal_dos.slater_koster import onsite_block, pair_block
from examples.inp_nanocrystal_dos.structure import (
    cluster_counts,
    generate_cluster,
    shell_for_target_atoms,
    write_poscar,
)
from examples.inp_nanocrystal_dos.validate import (
    ROUNDED_TABLE_BULK_GAP_EV,
    validate_series,
)
from examples.inp_nanocrystal_dos.cluster.collect_campaign import (
    collect_campaign,
    validate_result,
)
from examples.inp_nanocrystal_dos.cluster.submit_campaign import (
    DEFAULT_CAMPAIGN,
    load_campaign,
    main as submit_campaign,
    select_runs,
)


class TestInPNanocrystalExample(unittest.TestCase):
    @staticmethod
    def _comm_self():
        return MPI.COMM_SELF if MPI is not None else None

    def test_exact_paper_shell_22_geometry(self):
        counts = cluster_counts(22)
        self.assertEqual(counts.in_atoms, 5083)
        self.assertEqual(counts.p_atoms, 4444)
        self.assertEqual(counts.core_atoms, 9527)
        self.assertEqual(counts.core_orbitals, 60328)
        self.assertEqual(counts.h_atoms, 2556)
        self.assertAlmostEqual(counts.effective_diameter_angstrom, 77.1, places=1)

        geometry = generate_cluster(22)
        self.assertEqual(geometry.positions.shape, (12083, 3))
        self.assertEqual(np.count_nonzero(geometry.atomic_numbers == 49), 5083)
        self.assertEqual(np.count_nonzero(geometry.atomic_numbers == 15), 4444)
        self.assertEqual(np.count_nonzero(geometry.atomic_numbers == 1), 2556)

    def test_scaled_shell_targets_reach_a_few_million_atoms(self):
        shell_1m = shell_for_target_atoms(1_000_000)
        shell_3m = shell_for_target_atoms(3_000_000)
        self.assertEqual(shell_1m, 106)
        self.assertEqual(shell_3m, 154)
        self.assertGreaterEqual(cluster_counts(shell_1m).core_atoms, 1_000_000)
        self.assertGreaterEqual(cluster_counts(shell_3m).core_atoms, 3_000_000)

    def test_pair_blocks_obey_real_hermiticity(self):
        directions = ((1, 1, 1), (0, 1, -1), (1, 2, 3))
        for row, col in (("In", "P"), ("In", "In"), ("P", "P"), ("In", "H")):
            for direction in directions:
                forward = pair_block(row, col, direction)
                reverse = pair_block(col, row, -np.asarray(direction))
                np.testing.assert_allclose(forward, reverse.T, atol=1.0e-13)

    def test_in_p_axial_block_matches_tabulated_channels(self):
        block = pair_block("In", "P", (0, 0, 1))
        expected = np.zeros((4, 9))
        expected[0, 0] = -1.43
        expected[0, 2] = 2.19
        expected[0, 6] = -2.72
        expected[1, 1] = -0.66
        expected[1, 5] = 3.35
        expected[2, 0] = -1.63
        expected[2, 2] = 3.35
        expected[2, 6] = -3.38
        expected[3, 3] = -0.66
        expected[3, 7] = 3.35
        np.testing.assert_allclose(block, expected, atol=1.0e-13)

    def test_in_h_axial_block_uses_host_s_and_p_passivation_channels(self):
        block = pair_block("In", "H", (0, 0, 1))
        expected = np.zeros((4, 1))
        expected[0, 0] = -2.944
        expected[2, 0] = -2.76
        np.testing.assert_allclose(block, expected, atol=1.0e-13)

    def test_rounded_table_parameters_give_a_direct_bulk_gap(self):
        a = PAPER_PARAMETERS.lattice_constant
        first = 0.25 * a * np.asarray(
            ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)),
            dtype=np.float64,
        )
        second = 0.5 * a * np.asarray(
            [(0, j, k) for j in (-1, 1) for k in (-1, 1)]
            + [(i, 0, k) for i in (-1, 1) for k in (-1, 1)]
            + [(i, j, 0) for i in (-1, 1) for j in (-1, 1)],
            dtype=np.float64,
        )

        def bulk_bands(kpoint):
            matrix = np.zeros((13, 13), dtype=np.complex128)
            matrix[:4, :4] = onsite_block(IN)
            matrix[4:, 4:] = onsite_block(P)
            for displacement in second:
                phase = np.exp(1j * np.dot(kpoint, displacement))
                matrix[:4, :4] += pair_block(IN, IN, displacement) * phase
                matrix[4:, 4:] += pair_block(P, P, displacement) * phase
            for displacement in first:
                matrix[:4, 4:] += pair_block(IN, P, displacement) * np.exp(
                    1j * np.dot(kpoint, displacement)
                )
            matrix[4:, :4] = matrix[:4, 4:].conj().T
            return np.linalg.eigvalsh(matrix)

        gamma = bulk_bands(np.zeros(3))
        l_point = bulk_bands(np.full(3, np.pi / a))
        self.assertAlmostEqual(
            gamma[4] - gamma[3], ROUNDED_TABLE_BULK_GAP_EV, places=11
        )
        self.assertGreater(l_point[4], gamma[4])

        delta_k = 2.0e-4
        conduction = bulk_bands(np.asarray((delta_k, 0.0, 0.0)))[4]
        curvature = (conduction - gamma[4]) / delta_k**2
        electron_mass = 3.80998211615486 / curvature
        self.assertAlmostEqual(electron_mass, 0.10223949, places=7)

    def test_exact_gap_decreases_across_multiple_nanocrystal_sizes(self):
        results = validate_series((2, 4, 6), max_dense_states=2000)
        self.assertEqual([result.total_orbitals for result in results], [124, 580, 1636])
        np.testing.assert_allclose(
            [result.gap_ev for result in results],
            [3.690097803426, 3.215601495817, 2.754226605607],
            atol=2.0e-11,
            rtol=0.0,
        )
        self.assertTrue(
            all(right.gap_ev < left.gap_ev for left, right in zip(results, results[1:]))
        )
        # Shell 2 is surface dominated; from shell 4 onward the bulk-aligned
        # confinement shift rapidly approaches the paper's printed fit.
        self.assertGreater(results[0].homo_h_weight, 0.8)
        self.assertLess(abs(results[1].confinement_shift_residual_ev), 0.11)
        self.assertLess(abs(results[2].confinement_shift_residual_ev), 0.01)

    def test_kpm_integrates_to_state_count_for_multiple_finite_sizes(self):
        for shells in (2, 4):
            geometry = generate_cluster(shells)
            comm = self._comm_self()
            atoms = build_atomic_data(geometry, comm=comm)
            assembly = build_hamiltonian(atoms, comm=comm)
            result = compute_dos(
                assembly.matrix,
                assembly.spectral_bounds_ev,
                num_moments=24,
                num_random_vectors=2,
                batch_size=2,
                num_energy_points=100,
                comm=comm,
            )
            integrated = np.sum(result.dos_states_per_ev * result.integration_weights_ev)
            self.assertAlmostEqual(integrated, geometry.counts.total_orbitals, places=10)

    def test_vasp_poscar_roundtrip_preserves_geometry_and_vacuum(self):
        geometry = generate_cluster(2)
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "POSCAR"
            write_poscar(str(filename), geometry, vacuum=10.0)
            atoms = read(filename, format="vasp")

        numbers = atoms.get_atomic_numbers()
        self.assertEqual(np.count_nonzero(numbers == 49), geometry.counts.in_atoms)
        self.assertEqual(np.count_nonzero(numbers == 15), geometry.counts.p_atoms)
        self.assertEqual(np.count_nonzero(numbers == 1), geometry.counts.h_atoms)
        self.assertTrue(np.all(atoms.pbc))
        positions = atoms.get_positions()
        lengths = atoms.cell.lengths()
        np.testing.assert_allclose(positions.min(axis=0), 10.0, atol=2.0e-10)
        np.testing.assert_allclose(lengths - positions.max(axis=0), 10.0, atol=2.0e-10)

    def test_small_assembled_hamiltonian_is_hermitian(self):
        geometry = generate_cluster(2)
        comm = self._comm_self()
        atoms = build_atomic_data(geometry, comm=comm)
        assembly = build_hamiltonian(atoms, chunk_size=32, comm=comm)
        dense = assembly.matrix.to_dense()
        self.assertEqual(dense.shape, (124, 124))
        np.testing.assert_allclose(dense, dense.conj().T, atol=1.0e-12)
        self.assertEqual(assembly.statistics.local_hopping_edges, 188)

        # Independent small-system reference: one straightforward Python loop
        # over edges verifies that vectorized bond coding drops or merges none.
        sizes = np.asarray(atoms.graph.block_sizes, dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(sizes)))
        reference = np.zeros_like(dense)
        for atom, size in enumerate(sizes):
            symbol = NORB_TO_SYMBOL[int(size)]
            begin, end = offsets[atom : atom + 2]
            reference[begin:end, begin:end] = onsite_block(symbol)
        for (src, dst), direction in zip(atoms.edge_index, atoms.edge_vectors):
            row_symbol = NORB_TO_SYMBOL[int(sizes[src])]
            col_symbol = NORB_TO_SYMBOL[int(sizes[dst])]
            if not _accepted_pair(row_symbol, col_symbol, np.linalg.norm(direction), PAPER_PARAMETERS):
                continue
            row_begin, row_end = offsets[src : src + 2]
            col_begin, col_end = offsets[dst : dst + 2]
            reference[row_begin:row_end, col_begin:col_end] = pair_block(
                row_symbol, col_symbol, direction
            )
        np.testing.assert_allclose(dense, reference, atol=1.0e-12)

    def test_kpm_moments_for_onsite_only_cluster(self):
        geometry = generate_cluster(0)
        comm = self._comm_self()
        atoms = build_atomic_data(geometry, comm=comm)
        assembly = build_hamiltonian(atoms, comm=comm)
        result = compute_dos(
            assembly.matrix,
            assembly.spectral_bounds_ev,
            num_moments=12,
            num_random_vectors=2,
            batch_size=2,
            num_energy_points=100,
            comm=comm,
        )
        eigenvalues = PAPER_PARAMETERS.onsite["In"]
        lower, upper = assembly.spectral_bounds_ev
        scaled = (eigenvalues - 0.5 * (lower + upper)) / (0.5 * (upper - lower))
        expected = np.asarray(
            [np.sum(np.cos(n * np.arccos(scaled))) for n in range(12)]
        )
        np.testing.assert_allclose(result.moments.real, expected, atol=2.0e-12)
        np.testing.assert_allclose(result.moments.imag, 0.0, atol=2.0e-12)
        self.assertEqual(result.num_orbitals, 4)
        self.assertAlmostEqual(
            np.sum(result.dos_states_per_ev * result.integration_weights_ev), 4.0, places=12
        )

    def test_cluster_campaign_covers_physics_and_million_atom_scaling(self):
        campaign = load_campaign(DEFAULT_CAMPAIGN)
        physics = select_runs(campaign, "physics", names=None)
        scaling = select_runs(campaign, "scaling", names=None)
        self.assertEqual(
            [run["name"] for run in physics],
            ["paper_shell22", "size_100k", "size_1m_n8"],
        )
        self.assertEqual(
            [run["resources"]["nodes"] for run in scaling], [8, 4, 16]
        )
        self.assertTrue(
            all(run["target_core_atoms"] == 1_000_000 for run in scaling)
        )

    def test_cluster_campaign_rejects_invalid_numerics_and_resources(self):
        campaign = load_campaign(DEFAULT_CAMPAIGN)
        for field, value, message in (
            ("moments", 1, "moments must be at least 2"),
            ("energy_points", 1, "energy_points must be at least 2"),
        ):
            invalid = copy.deepcopy(campaign)
            invalid["runs"][0][field] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "campaign.json"
                path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_campaign(path)

        invalid = copy.deepcopy(campaign)
        invalid["runs"][0]["resources"]["memory_per_cpu"] = "many gigabytes"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.json"
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "memory_per_cpu is invalid"):
                load_campaign(path)

    def test_cluster_result_validator_checks_dos_normalization(self):
        metadata = {
            "cluster": {"shells": 0, "core_atoms": 1, "total_atoms": 1, "total_orbitals": 4},
            "assembly": {"chunk_size": 32},
            "kpm": {
                "moments": 4,
                "random_vectors": 2,
                "batch_size": 2,
                "energy_points": 2,
                "seed": 2026,
                "resolution_ev_approx": 0.5,
            },
            "distributed_totals": {"local_orbitals": 4},
            "mpi_ranks": 1,
            "timings_seconds_max_rank": {
                "coordinate_generation": 0.01,
                "graph_construction": 0.02,
                "hamiltonian_assembly": 0.03,
                "kpm": 0.04,
            },
        }
        campaign_run = {
            "INP_RUN_NAME": "synthetic",
            "INP_TARGET_CORE_ATOMS": "1",
            "INP_MOMENTS": "4",
            "INP_RANDOM_VECTORS": "2",
            "INP_BATCH_SIZE": "2",
            "INP_ENERGY_POINTS": "2",
            "INP_SEED": "2026",
            "INP_ASSEMBLY_CHUNK": "32",
            "slurm_job_id": "local",
            "slurm_ntasks": 1,
        }
        dos = np.asarray(
            [
                [0.0, 4.0, 8.0, 1.0, 0.5],
                [1.0, 4.0, 8.0, 1.0, 0.5],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.savetxt(path / "dos.dat", dos)
            np.savez(
                path / "moments.npz",
                moments=np.asarray((4.0, 1.0, 1.0, 1.0)),
                jackson_weights=np.ones(4),
                spectral_bounds_ev=np.asarray((-1.0, 1.0)),
            )
            (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (path / "campaign_run.json").write_text(
                json.dumps(campaign_run), encoding="utf-8"
            )
            summary = validate_result(path)

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["orbitals"], 4)
        self.assertAlmostEqual(summary["dos_integral"], 4.0)
        self.assertEqual(set(summary["checksums_sha256"]), {
            "dos.dat", "moments.npz", "metadata.json", "campaign_run.json"
        })

    def test_cluster_submitter_journals_once_and_avoids_duplicate_jobs(self):
        completed = mock.Mock(stdout="12345;cluster\n", stderr="")
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "examples.inp_nanocrystal_dos.cluster.submit_campaign.subprocess.run",
            return_value=completed,
        ) as run_sbatch, contextlib.redirect_stdout(io.StringIO()):
            arguments = [
                "--output-root",
                directory,
                "--account",
                "def-test",
                "--run",
                "paper_shell22",
                "--submit",
            ]
            self.assertEqual(submit_campaign(arguments), 0)
            self.assertEqual(submit_campaign(arguments), 0)
            journal = Path(directory) / "submissions.jsonl"
            records = [json.loads(line) for line in journal.read_text().splitlines()]

        run_sbatch.assert_called_once()
        self.assertEqual(records[0]["job_id"], "12345")
        self.assertEqual(records[0]["run_name"], "paper_shell22")

    def test_cluster_worker_validates_and_atomically_publishes_result(self):
        repo_root = Path(__file__).resolve().parent.parent
        worker = (
            repo_root
            / "examples"
            / "inp_nanocrystal_dos"
            / "cluster"
            / "run_alliance.slurm"
        )
        bash = shutil.which("bash")
        self.assertIsNotNone(bash)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_driver = root / "fake_driver.py"
            fake_driver.write_text(
                textwrap.dedent(
                    """
                    import argparse
                    import json
                    from pathlib import Path

                    import numpy as np

                    parser = argparse.ArgumentParser()
                    parser.add_argument("--target-core-atoms", type=int)
                    parser.add_argument("--moments", type=int)
                    parser.add_argument("--random-vectors", type=int)
                    parser.add_argument("--batch-size", type=int)
                    parser.add_argument("--energy-points", type=int)
                    parser.add_argument("--seed", type=int)
                    parser.add_argument("--assembly-chunk", type=int)
                    parser.add_argument("--output-dir", type=Path)
                    args = parser.parse_args()
                    args.output_dir.mkdir(parents=True, exist_ok=True)
                    dos = np.asarray(((0.0, 4.0, 8.0, 1.0, 0.5),
                                      (1.0, 4.0, 8.0, 1.0, 0.5)))
                    np.savetxt(args.output_dir / "dos.dat", dos)
                    np.savez(
                        args.output_dir / "moments.npz",
                        moments=np.asarray((4.0, 1.0, 1.0, 1.0)),
                        jackson_weights=np.ones(4),
                        spectral_bounds_ev=np.asarray((-1.0, 1.0)),
                    )
                    metadata = {
                        "cluster": {
                            "shells": 0, "core_atoms": 1,
                            "total_atoms": 1, "total_orbitals": 4,
                        },
                        "assembly": {"chunk_size": args.assembly_chunk},
                        "kpm": {
                            "moments": args.moments,
                            "random_vectors": args.random_vectors,
                            "batch_size": args.batch_size,
                            "energy_points": args.energy_points,
                            "seed": args.seed,
                            "resolution_ev_approx": 0.5,
                        },
                        "distributed_totals": {"local_orbitals": 4},
                        "mpi_ranks": 1,
                        "timings_seconds_max_rank": {
                            "coordinate_generation": 0.01,
                            "graph_construction": 0.02,
                            "hamiltonian_assembly": 0.03,
                            "kpm": 0.04,
                        },
                    }
                    (args.output_dir / "metadata.json").write_text(
                        json.dumps(metadata), encoding="utf-8"
                    )
                    """
                ),
                encoding="utf-8",
            )
            fake_python = bin_dir / "python-for-test"
            fake_python.write_text(
                textwrap.dedent(
                    """\
                    #!/bin/bash
                    set -e
                    if [[ "${1:-}" == "-c" ]]; then
                        echo "synthetic package preflight"
                        exit 0
                    fi
                    if [[ "${1:-}" == "-m" && "${2:-}" == "examples.inp_nanocrystal_dos.run" ]]; then
                        shift 2
                        exec "__PYTHON__" "__DRIVER__" "$@"
                    fi
                    exec "__PYTHON__" "$@"
                    """
                )
                .replace("__PYTHON__", sys.executable)
                .replace("__DRIVER__", str(fake_driver)),
                encoding="utf-8",
            )
            fake_mpirun = bin_dir / "mpirun"
            fake_mpirun.write_text(
                "#!/bin/bash\nset -e\n[[ \"${1:-}\" == -np ]] && shift 2\nexec \"$@\"\n",
                encoding="utf-8",
            )
            os.chmod(fake_python, 0o755)
            os.chmod(fake_mpirun, 0o755)

            campaign_root = root / "campaign"
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": str(bin_dir) + os.pathsep + environment["PATH"],
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "INP_CAMPAIGN_ROOT": str(campaign_root),
                    "INP_REPO_ROOT": str(repo_root),
                    "INP_RUN_NAME": "synthetic",
                    "INP_TARGET_CORE_ATOMS": "1",
                    "INP_MOMENTS": "4",
                    "INP_RANDOM_VECTORS": "2",
                    "INP_BATCH_SIZE": "2",
                    "INP_ENERGY_POINTS": "2",
                    "INP_SEED": "2026",
                    "INP_ASSEMBLY_CHUNK": "32",
                    "INP_SETUP_SCRIPT": "",
                    "INP_PYTHON": str(fake_python),
                    "INP_LAUNCHER": "mpirun",
                    "SLURM_JOB_ID": "9001",
                    "SLURM_JOB_NODELIST": "test-node",
                    "SLURM_NTASKS": "1",
                    "SLURM_CPUS_PER_TASK": "1",
                }
            )
            first = subprocess.run(
                [bash, str(worker)],
                cwd=repo_root,
                env=environment,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if first.returncode:
                self.fail(
                    "worker failed with {}\nstdout:\n{}\nstderr:\n{}".format(
                        first.returncode, first.stdout, first.stderr
                    )
                )
            result_dir = campaign_root / "results" / "synthetic"
            self.assertTrue((result_dir / "COMPLETE").is_file())
            self.assertTrue((result_dir / "validation.json").is_file())
            self.assertIn("Published validated result", first.stdout)
            self.assertEqual(validate_result(result_dir)["dos_integral"], 4.0)

            campaign = {
                "schema_version": 1,
                "description": "synthetic worker test",
                "runs": [
                    {
                        "name": "synthetic",
                        "groups": ["physics"],
                        "target_core_atoms": 1,
                        "moments": 4,
                        "random_vectors": 2,
                        "batch_size": 2,
                        "energy_points": 2,
                        "seed": 2026,
                        "assembly_chunk": 32,
                        "resources": {
                            "nodes": 1,
                            "tasks_per_node": 1,
                            "cpus_per_task": 1,
                            "memory_per_cpu": "1G",
                            "time": "00:05:00",
                        },
                    }
                ],
            }
            campaign_file = campaign_root / "campaign_snapshot.json"
            campaign_file.write_text(json.dumps(campaign), encoding="utf-8")
            submissions = (
                {"run_name": "synthetic", "job_id": "8999"},
                {"run_name": "synthetic", "job_id": "9001"},
                {"run_name": "synthetic", "job_id": "9002"},
            )
            (campaign_root / "submissions.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in submissions),
                encoding="utf-8",
            )
            collected = collect_campaign(campaign_file, campaign_root)
            self.assertEqual(collected["runs"][0]["submitted_job_id"], "9001")
            self.assertEqual(
                collected["runs"][0]["submitted_job_ids"],
                ["8999", "9001", "9002"],
            )

            second = subprocess.run(
                [bash, str(worker)],
                cwd=repo_root,
                env=environment,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertIn("already complete", second.stdout)

            bad_environment = environment.copy()
            bad_environment.update(
                {
                    "INP_RUN_NAME": "synthetic_bad",
                    "INP_MOMENTS": "5",
                    "SLURM_JOB_ID": "9003",
                }
            )
            bad = subprocess.run(
                [bash, str(worker)],
                cwd=repo_root,
                env=bad_environment,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(bad.returncode, 0)
            self.assertFalse(
                (campaign_root / "results" / "synthetic_bad").exists()
            )
            failed_markers = list(
                (campaign_root / "attempts" / "synthetic_bad").glob("*/FAILED")
            )
            self.assertEqual(len(failed_markers), 1)


if __name__ == "__main__":
    unittest.main()

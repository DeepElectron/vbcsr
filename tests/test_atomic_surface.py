import unittest
import tempfile
from pathlib import Path

import numpy as np
from ase import Atoms

import _workspace_bootstrap

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import vbcsr
import vbcsr_core


def comm_self():
    return MPI.COMM_SELF if MPI else None


class TestAtomicSurface(unittest.TestCase):
    def test_from_points_preserves_atomic_numbers_and_matches_core(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float64)
        atomic_numbers = np.array([1, 8], dtype=np.int32)
        cell = np.eye(3, dtype=np.float64) * 10.0
        pbc = [False, False, False]

        wrapped = vbcsr.AtomicData.from_points(
            pos,
            atomic_numbers,
            cell,
            pbc,
            {1: 2.0, 8: 2.0},
            {1: 1, 8: 4},
            comm=comm_self(),
        )
        core = vbcsr_core.AtomicData.from_points(
            pos,
            atomic_numbers,
            cell,
            pbc,
            np.array([2.0, 2.0], dtype=np.float64),
            np.array([1, 4], dtype=np.int32),
            comm_self(),
        )

        np.testing.assert_allclose(wrapped.pos, pos)
        np.testing.assert_allclose(wrapped.positions, pos)
        np.testing.assert_array_equal(wrapped.atomic_numbers, atomic_numbers)
        np.testing.assert_array_equal(wrapped.z, atomic_numbers)
        np.testing.assert_array_equal(wrapped.atom_types, np.array([0, 1], dtype=np.int32))
        np.testing.assert_array_equal(wrapped.atom_indices, np.array([0, 1], dtype=np.int32))
        np.testing.assert_array_equal(wrapped.edge_index, np.array([[0, 1], [1, 0]], dtype=np.int32))
        np.testing.assert_array_equal(wrapped.edge_shift, np.zeros((2, 3), dtype=np.int32))
        np.testing.assert_array_equal(core.atomic_numbers, wrapped.atomic_numbers)
        np.testing.assert_array_equal(core.atom_types, wrapped.atom_types)

    def test_image_container_wrapper_matches_core_sample(self):
        pos = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float64)
        atomic_numbers = np.array([1, 1], dtype=np.int32)
        cell = np.eye(3, dtype=np.float64) * 10.0
        pbc = [False, False, False]

        wrapped_atoms = vbcsr.AtomicData.from_points(
            pos,
            atomic_numbers,
            cell,
            pbc,
            2.0,
            1,
            comm=comm_self(),
        )
        core_atoms = vbcsr_core.AtomicData.from_points(
            pos,
            atomic_numbers,
            cell,
            pbc,
            np.array([2.0], dtype=np.float64),
            np.array([1], dtype=np.int32),
            comm_self(),
        )

        wrapped_image = vbcsr.ImageContainer(wrapped_atoms, dtype=np.float64)
        wrapped_image.add_block(0, 1, np.array([[2.0]], dtype=np.float64), mode="insert")
        wrapped_image.assemble()
        wrapped_result = wrapped_image.sample_k([0.0, 0.0, 0.0])

        core_image = vbcsr_core.ImageContainer(core_atoms)
        core_image.add_block(
            [0, 0, 0],
            0,
            1,
            np.array([[2.0]], dtype=np.float64),
            vbcsr.AssemblyMode.INSERT,
        )
        core_image.assemble()
        core_result = core_image.sample_k(np.array([0.0, 0.0, 0.0], dtype=np.float64))

        self.assertEqual(wrapped_result.shape, (2, 2))
        np.testing.assert_allclose(wrapped_result.to_dense(), core_result.to_dense())

    def test_atomic_file_and_ase_round_trip(self):
        atoms = Atoms(
            numbers=[6, 8],
            positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
            cell=np.eye(3) * 8.0,
            pbc=[False, False, False],
        )

        wrapped = vbcsr.AtomicData.from_ase(atoms, r_max=2.0, type_norb=1, comm=comm_self())
        round_trip = wrapped.to_ase()

        np.testing.assert_array_equal(wrapped.atomic_numbers, np.array([6, 8], dtype=np.int32))
        np.testing.assert_array_equal(round_trip.get_atomic_numbers(), np.array([6, 8], dtype=np.int32))
        np.testing.assert_allclose(round_trip.get_positions(), atoms.get_positions())

        handle = tempfile.NamedTemporaryFile(suffix=".xyz", delete=False)
        handle.close()
        try:
            atoms.write(handle.name, format="extxyz")
            loaded = vbcsr.AtomicData.from_file(
                handle.name,
                r_max=2.0,
                type_norb=1,
                comm=comm_self(),
                format="extxyz",
            )
        finally:
            Path(handle.name).unlink(missing_ok=True)

        np.testing.assert_array_equal(loaded.atomic_numbers, np.array([6, 8], dtype=np.int32))
        np.testing.assert_allclose(loaded.positions, atoms.get_positions())

    def test_distributed_constructor_requires_real_atomic_numbers(self):
        if MPI is None:
            self.skipTest("mpi4py is not installed")

        comm = MPI.COMM_WORLD
        if comm.Get_size() != 2:
            self.skipTest("Run this test with exactly 2 MPI ranks")

        rank = comm.Get_rank()
        atom_index = np.array([rank], dtype=np.int32)
        atom_type = np.array([rank], dtype=np.int32)
        edge_index = np.array([[rank, 1 - rank]], dtype=np.int32)
        edge_shift = np.zeros((1, 3), dtype=np.int32)
        cell = np.eye(3, dtype=np.float64) * 10.0
        pos = np.array([[float(rank), 0.0, 0.0]], dtype=np.float64)
        atomic_numbers = np.array([6 + 2 * rank], dtype=np.int32)
        type_norb = np.array([1, 1], dtype=np.int32)

        atoms = vbcsr.AtomicData.from_distributed(
            1,
            2,
            rank,
            1,
            2,
            atom_index,
            atom_type,
            edge_index,
            type_norb,
            edge_shift,
            cell,
            pos,
            atomic_numbers=atomic_numbers,
            comm=comm,
        )

        np.testing.assert_array_equal(atoms.atom_types, atom_type)
        np.testing.assert_array_equal(atoms.atomic_numbers, atomic_numbers)
        np.testing.assert_array_equal(atoms.z, atomic_numbers)
        np.testing.assert_array_equal(atoms.edge_index, np.array([[0, 1]], dtype=np.int32))
        np.testing.assert_array_equal(atoms.edge_shift, edge_shift)

        missing_numbers = vbcsr.AtomicData.from_distributed(
            1,
            2,
            rank,
            1,
            2,
            atom_index,
            atom_type,
            edge_index,
            type_norb,
            edge_shift,
            cell,
            pos,
            atomic_numbers=None,
            comm=comm,
        )
        with self.assertRaises(RuntimeError):
            _ = missing_numbers.atomic_numbers


if __name__ == "__main__":
    unittest.main()

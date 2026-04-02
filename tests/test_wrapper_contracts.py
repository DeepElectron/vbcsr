import os
import tempfile
import unittest

import numpy as np
import scipy.sparse as sp

import _workspace_bootstrap

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from vbcsr import VBCSR


def comm_self():
    return MPI.COMM_SELF if MPI else None


def build_sample_matrix(dtype=np.float64) -> VBCSR:
    mat = VBCSR.create_serial(2, [2, 3], [[0, 1], [0, 1]], dtype=dtype, comm=comm_self())
    mat.add_block(0, 0, np.array([[1, 2], [3, 4]], dtype=dtype))
    mat.add_block(0, 1, np.array([[5, 6, 7], [8, 9, 10]], dtype=dtype))
    mat.add_block(1, 0, np.array([[11, 12], [13, 14], [15, 16]], dtype=dtype))
    mat.add_block(1, 1, np.array([[17, 18, 19], [20, 21, 22], [23, 24, 25]], dtype=dtype))
    mat.assemble()
    return mat


class TestWrapperContracts(unittest.TestCase):
    def test_matrix_aliases_and_duplicate_policy(self):
        comm = comm_self()
        mat = VBCSR.create_serial(1, [2], [[0]], dtype=np.complex128, comm=comm)
        mat.add_block(0, 0, np.array([[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]]))
        mat.assemble()

        dup_copy = mat.copy()
        dup_independent = mat.duplicate()
        dup_shared_graph = mat.duplicate(independent_graph=False)

        np.testing.assert_array_equal(dup_copy.get_values(), dup_independent.get_values())
        np.testing.assert_array_equal(dup_copy.get_values(), dup_shared_graph.get_values())

        dup_shared_graph.shift(1.0)
        np.testing.assert_array_equal(
            mat.get_block(0, 0),
            np.array([[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]]),
        )

        np.testing.assert_array_equal(mat.conj().get_values(), mat.conjugate().get_values())

    def test_duplicate_wrapper_accessors_and_numeric_independence(self):
        comm = comm_self()
        mat = VBCSR.create_serial(2, [2, 2], [[0, 1], [1]], comm=comm)

        mat.add_block(0, 0, np.array([[1.0, 2.0], [3.0, 4.0]]))
        mat.add_block(0, 1, np.array([[5.0, 6.0], [7.0, 8.0]]))
        mat.add_block(1, 1, np.array([[9.0, 10.0], [11.0, 12.0]]))
        mat.assemble()

        dup = mat.duplicate()

        self.assertEqual(dup.shape, mat.shape)
        self.assertEqual(dup.matrix_kind, "bsr")
        np.testing.assert_array_equal(dup.row_ptr, np.array([0, 2, 3], dtype=np.int32))
        np.testing.assert_array_equal(dup.col_ind, np.array([0, 1, 1], dtype=np.int32))
        np.testing.assert_array_equal(dup.get_values(), mat.get_values())

        dup.shift(1.0)

        np.testing.assert_array_equal(mat.get_block(0, 0), np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_array_equal(dup.get_block(0, 0), np.array([[2.0, 2.0], [3.0, 5.0]]))

    def test_extract_submatrix_has_shape_and_dense_export(self):
        comm = comm_self()
        mat = VBCSR.create_serial(2, [2, 3], [[0, 1], [0, 1]], comm=comm)

        mat.add_block(0, 0, np.array([[1.0, 2.0], [3.0, 4.0]]))
        mat.add_block(0, 1, np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]))
        mat.add_block(1, 0, np.array([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]))
        mat.add_block(1, 1, np.array([[17.0, 18.0, 19.0], [20.0, 21.0, 22.0], [23.0, 24.0, 25.0]]))
        mat.assemble()

        sub = mat.extract_submatrix([0, 1])

        self.assertEqual(sub.shape, (5, 5))
        self.assertEqual(sub.matrix_kind, "vbcsr")
        np.testing.assert_array_equal(sub.to_dense(), mat.to_dense())

    def test_save_matrix_market_wrapper(self):
        comm = comm_self()
        mat = VBCSR.create_serial(1, [1], [[0]], comm=comm)
        mat.add_block(0, 0, np.array([[2.0]]))
        mat.assemble()

        handle = tempfile.NamedTemporaryFile("r", suffix=".mtx", delete=False)
        handle.close()
        try:
            mat.save_matrix_market(handle.name)
            with open(handle.name, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines()]
        finally:
            os.remove(handle.name)

        self.assertTrue(lines[0].startswith("%%MatrixMarket matrix coordinate real general"))
        self.assertEqual(lines[1], "1 1 1")
        self.assertIn("1 1", lines[2])

    def test_multivector_duplicate_wrapper(self):
        comm = comm_self()
        mat = VBCSR.create_serial(1, [2], [[0]], comm=comm)
        mat.add_block(0, 0, np.eye(2))
        mat.assemble()

        mv = mat.create_multivector(2)
        mv.set_constant(1.0)
        mv_dup = mv.duplicate()
        self.assertEqual(mv.ghost_rows, 0)
        mv.reduce_ghosts()

        mv += 2.0

        np.testing.assert_allclose(mv_dup.to_numpy(), np.ones((2, 2)))
        np.testing.assert_allclose(mv.to_numpy(), np.full((2, 2), 3.0))

    def test_matrix_wrapper_matches_core_utility_surface(self):
        mat = build_sample_matrix()
        core = mat._core.duplicate()

        self.assertEqual(mat.shape, (5, 5))
        self.assertEqual(mat.matrix_kind, core.matrix_kind)
        self.assertEqual(mat.nnz, core.global_nnz)

        np.testing.assert_allclose(mat.transpose().to_dense(), core.transpose().to_dense())
        np.testing.assert_allclose(mat.get_block(0, 1), core.get_block(0, 1))
        np.testing.assert_allclose(mat.get_values(), core.get_values())
        np.testing.assert_array_equal(mat.row_ptr, core.row_ptr)
        np.testing.assert_array_equal(mat.col_ind, core.col_ind)

        shifted = mat.duplicate()
        shifted_core = core.duplicate()
        shifted.scale(2.0)
        shifted_core.scale(2.0)
        shifted.shift(1.5)
        shifted_core.shift(1.5)

        diag = shifted.create_vector()
        diag.from_numpy(np.arange(1, shifted.shape[0] + 1, dtype=np.float64))
        shifted.add_diagonal(diag)
        shifted_core.add_diagonal(diag._core)
        np.testing.assert_allclose(shifted.to_dense(), shifted_core.to_dense())

        dense_update = np.arange(25, dtype=np.float64).reshape(5, 5)
        dense_wrapper = mat.duplicate()
        dense_core = core.duplicate()
        dense_wrapper.from_dense(dense_update)
        dense_core.from_dense(dense_update)
        np.testing.assert_allclose(dense_wrapper.to_dense(), dense_core.to_dense())

        sub = mat.extract_submatrix([0, 1])
        sub_core = core.extract_submatrix([0, 1])
        self.assertEqual(sub.shape, (5, 5))
        self.assertEqual(sub.matrix_kind, sub_core.matrix_kind)
        self.assertEqual(sub.nnz, sub_core.global_nnz)
        np.testing.assert_allclose(sub.to_dense(), sub_core.to_dense())

        patched_wrapper = mat.duplicate()
        patched_core = core.duplicate()
        sub_scaled = sub.duplicate()
        sub_scaled.scale(0.5)
        sub_core_scaled = sub_core.duplicate()
        sub_core_scaled.scale(0.5)
        patched_wrapper.insert_submatrix(sub_scaled, [0, 1])
        patched_core.insert_submatrix(sub_core_scaled, [0, 1])
        np.testing.assert_allclose(patched_wrapper.to_dense(), patched_core.to_dense())

    def test_complex_real_imag_and_conjugate_match_core(self):
        mat = build_sample_matrix(dtype=np.complex128)
        mat.shift(1.0j)
        core = mat._core.duplicate()

        core_conjugated = core.duplicate()
        core_conjugated.conjugate()
        np.testing.assert_allclose(mat.conjugate().to_dense(), core_conjugated.to_dense())
        np.testing.assert_allclose(mat.real.to_dense(), core.real().to_dense())
        np.testing.assert_allclose(mat.imag.to_dense(), core.imag().to_dense())

    def test_negative_wrapper_contracts(self):
        mat = build_sample_matrix()

        with self.assertRaises(NotImplementedError):
            _ = mat[0, 0]

        if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            dist = VBCSR.create_distributed([rank], [1], [[rank]], comm=comm)
            dist.add_block(rank, rank, np.array([[1.0]]))
            dist.assemble()

            with tempfile.NamedTemporaryFile("r", suffix=".mtx", delete=False) as handle:
                filename = handle.name
            try:
                with self.assertRaises(RuntimeError):
                    dist.save_matrix_market(filename)
            finally:
                os.remove(filename)

    def test_to_scipy_export_contracts(self):
        mat = build_sample_matrix()

        csr = mat.to_scipy("csr")
        self.assertEqual(csr.shape, (5, 5))
        np.testing.assert_allclose(csr.toarray(), mat.to_dense())

        bsr = VBCSR.from_scipy(sp.bsr_matrix(np.eye(4), blocksize=(2, 2)))
        self.assertEqual(bsr.to_scipy("bsr").blocksize, (2, 2))
        with self.assertRaises(ValueError):
            _ = mat.to_scipy("bsr")


if __name__ == "__main__":
    unittest.main()

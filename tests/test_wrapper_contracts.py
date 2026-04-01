import os
import tempfile
import unittest

import numpy as np

import _workspace_bootstrap

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from vbcsr import VBCSR


class TestWrapperContracts(unittest.TestCase):
    def test_duplicate_wrapper_accessors_and_numeric_independence(self):
        comm = MPI.COMM_SELF if MPI else None
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
        comm = MPI.COMM_SELF if MPI else None
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
        comm = MPI.COMM_SELF if MPI else None
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


if __name__ == "__main__":
    unittest.main()

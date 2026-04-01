import unittest

import numpy as np
import scipy.sparse as sp
import _workspace_bootstrap

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from vbcsr import VBCSR


def partition_owned(global_blocks: int, size: int, rank: int) -> list[int]:
    blocks_per_rank = global_blocks // size
    remainder = global_blocks % size
    start = rank * blocks_per_rank + min(rank, remainder)
    count = blocks_per_rank + (1 if rank < remainder else 0)
    return list(range(start, start + count))


class TestMatrixKind(unittest.TestCase):
    def test_serial_facade_accessors_and_kind(self):
        comm = MPI.COMM_SELF if MPI else None
        block_sizes = [2, 2]
        adj = [[0, 1], [1]]
        mat = VBCSR.create_serial(2, block_sizes, adj, comm=comm)

        mat.add_block(0, 0, np.array([[1.0, 2.0], [3.0, 4.0]]))
        mat.add_block(0, 1, np.array([[5.0, 6.0], [7.0, 8.0]]))
        mat.add_block(1, 1, np.array([[9.0, 10.0], [11.0, 12.0]]))
        mat.assemble()

        self.assertEqual(mat.matrix_kind, "bsr")
        np.testing.assert_array_equal(mat.row_ptr, np.array([0, 2, 3], dtype=np.int32))
        np.testing.assert_array_equal(mat.col_ind, np.array([0, 1, 1], dtype=np.int32))
        np.testing.assert_array_equal(mat._core.row_ptr, np.array([0, 2, 3], dtype=np.int32))
        np.testing.assert_array_equal(mat._core.col_ind, np.array([0, 1, 1], dtype=np.int32))
        self.assertEqual(mat._core.local_nnz, 12)

        block = mat.get_block(0, 1)
        np.testing.assert_array_almost_equal(block, np.array([[5.0, 6.0], [7.0, 8.0]]))

        values = mat.get_values()
        self.assertEqual(values.size, 12)
        self.assertTrue(sp.isspmatrix_bsr(mat.to_scipy()))
        self.assertEqual(mat.transpose().matrix_kind, "bsr")

    def test_serial_matrix_kind_classification(self):
        comm = MPI.COMM_SELF if MPI else None
        mats = [
            (VBCSR.create_serial(3, [1, 1, 1], [[0], [1], [2]], comm=comm), "csr"),
            (VBCSR.create_serial(3, [8, 8, 8], [[0], [1], [2]], comm=comm), "bsr"),
            (VBCSR.create_serial(3, [9, 13, 15], [[0], [1], [2]], comm=comm), "vbcsr"),
        ]
        for mat, expected in mats:
            self.assertEqual(mat.matrix_kind, expected)

    def test_serial_vbcsr_transpose_preserves_kind(self):
        comm = MPI.COMM_SELF if MPI else None
        mat = VBCSR.create_serial(2, [2, 3], [[0, 1], [0, 1]], comm=comm)

        mat.add_block(0, 0, np.array([[1.0, 2.0], [3.0, 4.0]]))
        mat.add_block(0, 1, np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]))
        mat.add_block(1, 0, np.array([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]))
        mat.add_block(1, 1, np.array([[17.0, 18.0, 19.0], [20.0, 21.0, 22.0], [23.0, 24.0, 25.0]]))
        mat.assemble()

        self.assertEqual(mat.matrix_kind, "vbcsr")
        transposed = mat.transpose()
        self.assertEqual(transposed.matrix_kind, "vbcsr")
        np.testing.assert_array_equal(transposed._core.row_ptr, np.array([0, 2, 4], dtype=np.int32))
        np.testing.assert_array_equal(transposed._core.col_ind, np.array([0, 1, 0, 1], dtype=np.int32))

    def test_serial_csr_to_scipy_and_transpose(self):
        comm = MPI.COMM_SELF if MPI else None
        mat = VBCSR.create_serial(3, [1, 1, 1], [[0, 1], [1, 2], [2]], comm=comm)

        mat.add_block(0, 0, np.array([[2.0]]))
        mat.add_block(0, 1, np.array([[3.0]]))
        mat.add_block(1, 1, np.array([[5.0]]))
        mat.add_block(1, 2, np.array([[7.0]]))
        mat.add_block(2, 2, np.array([[11.0]]))
        mat.assemble()

        self.assertEqual(mat.matrix_kind, "csr")
        auto_mat = mat.to_scipy()
        self.assertTrue(sp.isspmatrix_bsr(auto_mat))
        self.assertEqual(auto_mat.blocksize, (1, 1))

        scipy_mat = mat.to_scipy(format="csr")
        self.assertTrue(sp.isspmatrix_csr(scipy_mat))
        np.testing.assert_array_equal(scipy_mat.indptr, np.array([0, 2, 4, 5], dtype=np.int32))
        np.testing.assert_array_equal(scipy_mat.indices, np.array([0, 1, 1, 2, 2], dtype=np.int32))
        np.testing.assert_array_equal(scipy_mat.data, np.array([2.0, 3.0, 5.0, 7.0, 11.0]))
        self.assertEqual(mat.transpose().matrix_kind, "csr")

    def test_distributed_matrix_kind_classification(self):
        if MPI is not None:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
        else:
            comm = None
            rank = 0
            size = 1

        global_blocks = max(4, size * 2)
        families = {
            "csr": [1] * global_blocks,
            "bsr": [8] * global_blocks,
            "vbcsr": [9, 13, 15, 20] * ((global_blocks + 3) // 4),
        }

        for expected, sizes in families.items():
            block_sizes = sizes[:global_blocks]
            owned = partition_owned(global_blocks, size, rank)
            local_block_sizes = [block_sizes[idx] for idx in owned]
            local_adj = []
            for gid in owned:
                local_adj.append(sorted({gid, (gid + 1) % global_blocks}))

            mat = VBCSR.create_distributed(owned, local_block_sizes, local_adj, comm=comm)
            self.assertEqual(mat.matrix_kind, expected)


if __name__ == "__main__":
    unittest.main()

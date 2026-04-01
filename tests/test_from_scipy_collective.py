import unittest

import numpy as np
import scipy.sparse as sp

import _workspace_bootstrap

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from vbcsr import VBCSR


class TestFromScipyCollective(unittest.TestCase):
    def test_requires_explicit_comm_under_mpi(self):
        if MPI is None:
            self.skipTest("mpi4py is not available")

        comm = MPI.COMM_WORLD
        if comm.Get_size() == 1:
            self.skipTest("requires multiple MPI ranks")

        with self.assertRaises(ValueError):
            VBCSR.from_scipy(sp.eye(4, format="csr", dtype=np.float64))

    def test_root_only_collective_csr_import(self):
        if MPI is None:
            self.skipTest("mpi4py is not available")

        comm = MPI.COMM_WORLD
        if comm.Get_size() == 1:
            self.skipTest("requires multiple MPI ranks")

        n = max(comm.Get_size() * 2, 4)
        spmat = sp.eye(n, format="csr", dtype=np.float64) if comm.Get_rank() == 0 else None

        mat = VBCSR.from_scipy(spmat, comm=comm, root=0)

        self.assertEqual(mat.shape, (n, n))
        self.assertEqual(mat.matrix_kind, "csr")

        x = mat.create_vector()
        x.set_constant(1.0)
        y = mat.mult(x).to_numpy()

        np.testing.assert_allclose(y, np.ones_like(y))


if __name__ == "__main__":
    unittest.main()

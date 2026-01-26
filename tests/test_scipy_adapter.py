import unittest
import numpy as np
import scipy.sparse as sp
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import vbcsr
from vbcsr import VBCSR

class TestScipyAdapter(unittest.TestCase):
    def setUp(self):
        self.comm = MPI.COMM_WORLD if MPI else None
        if self.comm:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            try:
                import vbcsr_core
                g = vbcsr_core.DistGraph(None)
                self.rank = g.rank
                self.size = g.size
            except ImportError:
                self.rank = 0
                self.size = 1
        
        # Ensure identical random numbers on all ranks
        np.random.seed(42)
        
        if self.size > 1:
            self.skipTest("SciPy adapter tests are designed for serial execution only.")

    def test_from_scipy_bsr(self):
        # Create a BSR matrix on all ranks
        # 4 blocks of 2x2
        # Block structure:
        # [B1, 0 ]
        # [0,  B2]
        data = np.array([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ])
        indptr = [0, 1, 2]
        indices = [0, 1]
        bsr = sp.bsr_matrix((data, indices, indptr), shape=(4, 4))
            
        mat = VBCSR.from_scipy(bsr)
        
        # Verify structure
        # We can check via to_dense
        dense = mat.to_dense()
        
        # Compare with original
        np.testing.assert_array_almost_equal(dense, bsr.toarray())
            
        # Test round trip
        back_bsr = mat.to_scipy(format='bsr')
        self.assertTrue(sp.isspmatrix_bsr(back_bsr))
        np.testing.assert_array_almost_equal(back_bsr.toarray(), bsr.toarray())
        
        back_csr = mat.to_scipy(format='csr')
        self.assertTrue(sp.isspmatrix_csr(back_csr))
        np.testing.assert_array_almost_equal(back_csr.toarray(), bsr.toarray())

    def test_from_scipy_csr(self):
        # Random CSR on all ranks
        csr = sp.random(10, 10, density=0.2, format='csr', dtype=np.float64)
            
        mat = VBCSR.from_scipy(csr)
        
        dense = mat.to_dense()
        np.testing.assert_array_almost_equal(dense, csr.toarray())
        
        # Round trip (should default to CSR because blocks are 1x1 which is uniform, 
        # so it might default to BSR with blocksize 1? Yes.
        # Let's force CSR first.
        back_csr = mat.to_scipy(format='csr')
        self.assertTrue(sp.isspmatrix_csr(back_csr))
        np.testing.assert_array_almost_equal(back_csr.toarray(), csr.toarray())
        
        # Default (might be BSR 1x1)
        back_auto = mat.to_scipy()
        # 1x1 blocks are uniform, so it should be BSR
        self.assertTrue(sp.isspmatrix_bsr(back_auto))
        self.assertEqual(back_auto.blocksize, (1, 1))
        np.testing.assert_array_almost_equal(back_auto.toarray(), csr.toarray())

    def test_non_uniform_to_scipy(self):
        # Create a non-uniform VBCSR manually on all ranks independently
        # Block sizes: [2, 3]
        # 0 -> 0 (2x2)
        # 1 -> 1 (3x3)
        
        global_blocks = 2
        block_sizes = [2, 3]
        adj = [[0], [1]] # Diagonal
        
        # Use COMM_SELF for independent serial matrices
        comm = MPI.COMM_SELF if MPI else None
        mat = VBCSR.create_serial(global_blocks, block_sizes, adj, comm=comm)
        
        # Add data (everyone adds, since everyone is Rank 0 of their COMM_SELF)
        mat.add_block(0, 0, np.full((2, 2), 1.0))
        mat.add_block(1, 1, np.full((3, 3), 2.0))
            
        mat.assemble()
        
        # Try converting to BSR -> Should fail
        with self.assertRaises(ValueError):
            mat.to_scipy(format='bsr')
            
        # Convert to CSR -> Should work
        csr = mat.to_scipy(format='csr')
        self.assertTrue(sp.isspmatrix_csr(csr))
        
        expected = np.zeros((5, 5))
        expected[0:2, 0:2] = 1.0
        expected[2:5, 2:5] = 2.0
        
        np.testing.assert_array_almost_equal(csr.toarray(), expected)

    def test_behavioral_equivalence(self):
        # Create a random matrix via create_serial
        # Use COMM_SELF for independent tests
        comm = MPI.COMM_SELF if MPI else None
        
        global_blocks = 4
        block_sizes = [2, 2, 2, 2]
        adj = [[0, 1], [0, 1], [2, 3], [2, 3]]
        
        mat_native = VBCSR.create_serial(global_blocks, block_sizes, adj, comm=comm)
        
        # Fill with data (everyone fills)
        data_map = {}
        for i in range(global_blocks):
            for j in adj[i]:
                val = np.random.rand(2, 2)
                mat_native.add_block(i, j, val)
                data_map[(i, j)] = val
        
        mat_native.assemble()
        
        # Create equivalent SciPy matrix
        data_list = []
        indices_list = []
        indptr_list = [0]
        for i in range(global_blocks):
            for j in sorted(adj[i]): # SciPy expects sorted indices
                data_list.append(data_map[(i, j)])
                indices_list.append(j)
            indptr_list.append(len(data_list))
        
        bsr = sp.bsr_matrix((data_list, indices_list, indptr_list), shape=(8, 8))
            
        # Create VBCSR from SciPy
        mat_scipy = VBCSR.from_scipy(bsr)
        
        # Verify Mult
        x_np = np.random.rand(8)
        
        # Native Mult
        y_native = mat_native.mult(x_np).to_numpy()
        
        # Scipy-derived Mult
        y_scipy_wrapper = mat_scipy.mult(x_np).to_numpy()
        
        # SciPy Direct Mult (Ground Truth)
        y_truth = bsr.dot(x_np)
            
        # Check consistency
        np.testing.assert_array_almost_equal(y_native, y_truth, err_msg="Native VBCSR != SciPy Truth")
        np.testing.assert_array_almost_equal(y_scipy_wrapper, y_truth, err_msg="From-SciPy VBCSR != SciPy Truth")
        np.testing.assert_array_almost_equal(y_native, y_scipy_wrapper, err_msg="Native VBCSR != From-SciPy VBCSR")

    def test_numerical_accuracy_roundtrip(self):
        # Test with a larger random matrix on all ranks
        # 100x100 matrix, 10x10 blocks
        N = 100
        B = 10
        n_blocks = N // B
        
        # Create random BSR (seed ensures identical)
        # Create CSR then convert to BSR with blocksize
        csr = sp.random(N, N, density=0.05, format='csr', dtype=np.float64)
        bsr = csr.tobsr(blocksize=(B, B))
            
        # 1. SciPy -> VBCSR
        mat = VBCSR.from_scipy(bsr)
        
        # 2. VBCSR -> SciPy (Roundtrip)
        local_scipy = mat.to_scipy()
        
        # Single rank logic for everyone
        diff = abs(bsr - local_scipy).max()
        self.assertLess(diff, 1e-14, "Roundtrip accuracy failure")
        
        # Check Matvec
        x = np.random.rand(100)
        y_vbcsr = mat.mult(x).to_numpy()
        y_scipy = bsr.dot(x)
        
        np.testing.assert_array_almost_equal(y_vbcsr, y_scipy, decimal=14)

if __name__ == '__main__':
    unittest.main()

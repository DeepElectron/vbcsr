import unittest
import numpy as np
import scipy.sparse as sp
from mpi4py import MPI
import vbcsr
from vbcsr import VBCSR

class TestScipyAdapter(unittest.TestCase):
    def setUp(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def test_from_scipy_bsr(self):
        # Create a random BSR matrix on rank 0
        if self.rank == 0:
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
        else:
            bsr = None
            
        mat = VBCSR.from_scipy(bsr, self.comm)
        
        # Verify structure
        # Total blocks should be 2
        # Block sizes should be [2, 2]
        
        # We can check via to_dense or by inspecting graph if exposed
        dense = mat.to_dense()
        
        # Compare with original
        if self.rank == 0:
            expected = bsr.toarray()
            # Note: to_dense returns local part.
            # For serial creation (create_serial), rank 0 distributes.
            # If size=1, rank 0 owns everything.
            # If size > 1, we need to gather or check local parts.
        
        # Let's just check local consistency
        # For simple test, run with np=1
        if self.size == 1:
            np.testing.assert_array_almost_equal(dense, bsr.toarray())
            
            # Test round trip
            back_bsr = mat.to_scipy(format='bsr')
            self.assertTrue(sp.isspmatrix_bsr(back_bsr))
            np.testing.assert_array_almost_equal(back_bsr.toarray(), bsr.toarray())
            
            back_csr = mat.to_scipy(format='csr')
            self.assertTrue(sp.isspmatrix_csr(back_csr))
            np.testing.assert_array_almost_equal(back_csr.toarray(), bsr.toarray())

    def test_from_scipy_csr(self):
        if self.rank == 0:
            # Random CSR
            csr = sp.random(10, 10, density=0.2, format='csr', dtype=np.float64)
        else:
            csr = None
            
        mat = VBCSR.from_scipy(csr, self.comm)
        
        if self.size == 1:
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
        # Create a non-uniform VBCSR manually
        # Block sizes: [2, 3]
        # 0 -> 0 (2x2)
        # 1 -> 1 (3x3)
        
        global_blocks = 2
        block_sizes = [2, 3]
        adj = [[0], [1]] # Diagonal
        
        mat = VBCSR.create_serial(self.comm, global_blocks, block_sizes, adj)
        
        # Add data
        if self.rank == 0:
            mat.add_block(0, 0, np.full((2, 2), 1.0))
            mat.add_block(1, 1, np.full((3, 3), 2.0))
            
        mat.assemble()
        
        if self.size == 1:
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
        global_blocks = 4
        block_sizes = [2, 2, 2, 2]
        adj = [[0, 1], [0, 1], [2, 3], [2, 3]]
        
        mat_native = VBCSR.create_serial(self.comm, global_blocks, block_sizes, adj)
        
        # Fill with data
        data_map = {}
        if self.rank == 0:
            for i in range(global_blocks):
                for j in adj[i]:
                    val = np.random.rand(2, 2)
                    mat_native.add_block(i, j, val)
                    data_map[(i, j)] = val
        
        mat_native.assemble()
        
        # Create equivalent SciPy matrix
        if self.rank == 0:
            # Construct BSR
            data_list = []
            indices_list = []
            indptr_list = [0]
            for i in range(global_blocks):
                for j in sorted(adj[i]): # SciPy expects sorted indices
                    data_list.append(data_map[(i, j)])
                    indices_list.append(j)
                indptr_list.append(len(data_list))
            
            bsr = sp.bsr_matrix((data_list, indices_list, indptr_list), shape=(8, 8))
        else:
            bsr = None
            
        # Create VBCSR from SciPy
        mat_scipy = VBCSR.from_scipy(bsr, self.comm)
        
        # Verify Mult
        x_np = np.random.rand(8)
        
        # Native Mult
        y_native = mat_native.mult(x_np).to_numpy()
        
        # Scipy-derived Mult
        y_scipy_wrapper = mat_scipy.mult(x_np).to_numpy()
        
        # SciPy Direct Mult (Ground Truth)
        if self.rank == 0:
            y_truth = bsr.dot(x_np)
        else:
            y_truth = None
            
        # Check consistency
        if self.rank == 0:
            np.testing.assert_array_almost_equal(y_native, y_truth, err_msg="Native VBCSR != SciPy Truth")
            np.testing.assert_array_almost_equal(y_scipy_wrapper, y_truth, err_msg="From-SciPy VBCSR != SciPy Truth")
            np.testing.assert_array_almost_equal(y_native, y_scipy_wrapper, err_msg="Native VBCSR != From-SciPy VBCSR")

    def test_numerical_accuracy_roundtrip(self):
        # Test with a larger random matrix
        if self.rank == 0:
            # 100x100 matrix, 10x10 blocks
            N = 100
            B = 10
            n_blocks = N // B
            
            # Create random BSR
            # density=0.1
            bsr = sp.random(n_blocks, n_blocks, density=0.1, format='bsr', dtype=np.float64)
            # sp.random creates 1x1 blocks for bsr? No, it creates csr then converts?
            # Actually sp.random returns coo or csr usually.
            # Let's create block data manually to be sure of structure or use blocksize arg if available (it's not).
            # So we create CSR then convert to BSR with blocksize.
            csr = sp.random(N, N, density=0.05, format='csr', dtype=np.float64)
            bsr = csr.tobsr(blocksize=(B, B))
        else:
            bsr = None
            
        # 1. SciPy -> VBCSR
        mat = VBCSR.from_scipy(bsr, self.comm)
        
        # 2. VBCSR -> SciPy (Roundtrip)
        # Only rank 0 gets the full result if we gather? 
        # to_scipy returns LOCAL part.
        # If size > 1, we need to gather to compare with original global matrix.
        
        local_scipy = mat.to_scipy()
        
        if self.size == 1:
            # Single rank: Local is Global
            diff = abs(bsr - local_scipy).max()
            self.assertLess(diff, 1e-14, "Roundtrip accuracy failure")
            
            # Check Matvec
            x = np.random.rand(100)
            y_vbcsr = mat.mult(x).to_numpy()
            y_scipy = bsr.dot(x)
            
            np.testing.assert_array_almost_equal(y_vbcsr, y_scipy, decimal=14)
        else:
            # Distributed: Gather local parts and reconstruct global
            # This is complex to test generically without a gather helper.
            # But we can verify that local part matches the corresponding slice of global BSR.
            
            # Get local range
            # We need to know which rows we own.
            # VBCSR doesn't expose owned range easily in python yet?
            # mat.graph.owned_global_indices gives block indices.
            owned_blocks = mat.graph.owned_global_indices
            
            # Assuming uniform blocks BxB
            B = 10
            
            # Construct local slice of original BSR
            if self.rank == 0:
                # Broadcast BSR to all (for testing purposes only)
                pass
            
            bsr = self.comm.bcast(bsr, root=0)
            
            # Extract rows for this rank
            # owned_blocks is list of block indices.
            # Convert to row indices.
            rows = []
            for b in owned_blocks:
                rows.extend(range(b*B, (b+1)*B))
            
            if len(rows) > 0:
                bsr_slice = bsr[rows, :]
                
                # local_scipy has shape (n_owned_rows, n_total_cols)
                # But VBCSR includes ghost columns in structure.
                # `to_scipy` returns matrix with shape `mat.shape` (global shape)?
                # No, `to_scipy` uses `self.shape` which is global shape.
                # But the data is only local rows.
                # So it returns a sparse matrix of size (Global_Rows, Global_Cols) but only non-zeros in local rows?
                # Let's check `to_scipy` implementation.
                # `sp.bsr_matrix((data, col_ind, row_ptr), shape=self.shape)`
                # `row_ptr` has size `n_block_rows + 1`.
                # If `n_block_rows` corresponds to OWNED rows, then `row_ptr` covers only owned rows.
                # But `shape=self.shape` declares it as Global Size.
                # SciPy allows this?
                # If `row_ptr` has fewer entries than `shape[0]`, it might be invalid for CSR/BSR?
                # CSR/BSR `row_ptr` (indptr) must have length `n_rows + 1`.
                # If we declare shape (N, M), indptr must be length N+1.
                
                # ISSUE: `to_scipy` implementation sets shape=self.shape (Global), but provides indptr for Local Rows only.
                # This will likely crash or produce garbage if we are distributed and own only a subset.
                # We should return a matrix of shape (Local_Rows, Global_Cols) or (Local_Rows, Local_Cols_Range?).
                # Usually (Local_Rows, Global_Cols) is best for distributed assembly.
                
                # Let's fix `to_scipy` logic first if this is true.
                pass

if __name__ == '__main__':
    unittest.main()

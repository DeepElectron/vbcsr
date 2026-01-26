import unittest
import numpy as np
import vbcsr
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import sys

# the code still have some logical problem when mpi4py is not available.
# TODO: fix it.

class TestSpMM(unittest.TestCase):
    def setUp(self):
        self.comm = MPI.COMM_WORLD if MPI else None
        if self.comm:
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            # Fallback: use VBCSR to detect MPI environment
            # This handles the case where mpi4py is missing but mpirun is used
            try:
                import vbcsr_core
                temp_graph = vbcsr_core.DistGraph(None)
                self.rank = temp_graph.rank
                self.size = temp_graph.size
            except ImportError:
                self.rank = 0
                self.size = 1
        
        # Determine mode
        import os
        mode = os.environ.get("VBCSR_TEST_MODE", "auto")
        
        if mode == "serial":
            self.use_distributed = False
            # Force serial behavior even if MPI is present
            self.comm = None
            self.rank = 0
            self.size = 1
        elif mode == "distributed":
            self.use_distributed = True
        else:
            # Auto: Distributed if size > 1, else Serial
            self.use_distributed = (self.size > 1)
            


        # Define a system
        # 4 global blocks. 2x2 each.
        self.global_blocks = 4
        self.block_sizes = [2, 2, 2, 2]
        
        # Adjacency: Ring + Diagonal
        # 0: 0, 1
        # 1: 1, 2
        # 2: 2, 3
        # 3: 3, 0
        self.adj = [[0, 1], [1, 2], [2, 3], [3, 0]]
        
        if not self.use_distributed:
            # Serial Construction
            self.owned = [0, 1, 2, 3]
            self.local_sizes = [2, 2, 2, 2]
            self.local_adj = self.adj
            self.mat = vbcsr.VBCSR.create_serial(self.global_blocks, self.block_sizes, self.adj, comm=self.comm)
        else:
            # Distributed Construction
            # Generic partitioning for any size
            blocks_per_rank = self.global_blocks // self.size
            remainder = self.global_blocks % self.size
            
            start = self.rank * blocks_per_rank + min(self.rank, remainder)
            count = blocks_per_rank + (1 if self.rank < remainder else 0)
            self.owned = list(range(start, start + count))
            
            self.local_sizes = [self.block_sizes[i] for i in self.owned]
            self.local_adj = [self.adj[i] for i in self.owned]
            
            self.mat = vbcsr.VBCSR.create_distributed(self.owned, self.local_sizes, self.local_adj, comm=self.comm)
            
        # Fill A
        # Diagonal = Identity
        # Off-diagonal = 0.5 * Identity
        for i, row in enumerate(self.owned):
            for col in self.local_adj[i]:
                if row == col:
                    self.mat.add_block(row, col, np.eye(2))
                else:
                    self.mat.add_block(row, col, np.eye(2) * 0.5)
        self.mat.assemble()
        
        # Create B = A
        self.matB = self.mat.duplicate()

    def test_spmm_identity(self):
        # C = A * I
        # Create I
        if not self.use_distributed:
            adj_I = [[0], [1], [2], [3]]
            matI = vbcsr.VBCSR.create_serial(self.global_blocks, self.block_sizes, adj_I, comm=self.comm)
        else:
            # Distributed Identity
            local_adj_I = [[row] for row in self.owned]
            matI = vbcsr.VBCSR.create_distributed(self.owned, self.local_sizes, local_adj_I, comm=self.comm)
            
        for row in self.owned:
            matI.add_block(row, row, np.eye(2))
        matI.assemble()
        
        C = self.mat.spmm(matI, 0.0)
        
        # C should be equal to A
        # Check diagonal blocks
        # We can't easily check equality without a method.
        # But we can check norms or multiply by vector.
        
        v = self.mat.create_vector()
        v.set_constant(1.0)
        
        resA = self.mat.mult(v)
        resC = C.mult(v)
        
        if not np.allclose(resA.to_numpy(), resC.to_numpy()):
            print(f"Rank {self.rank} resA: {resA.to_numpy()}")
            print(f"Rank {self.rank} resC: {resC.to_numpy()}")
        self.assertTrue(np.allclose(resA.to_numpy(), resC.to_numpy()))

    def test_spmm_structure(self):
        # A has ring structure (0-1, 1-2, 2-3, 3-0)
        # A^2 should have (0-2, 1-3, 2-0, 3-1) + diagonals
        # 0 connects to 1. 1 connects to 2. So 0->2.
        # 0 connects to 0. 0 connects to 1. So 0->1.
        
        C = self.mat.spmm(self.matB, 0.0)
        
        # Check result
        # Row 0 should have cols: 0, 1, 2
        # 0->0 (I*I + 0.5*0 = I)
        # 0->1 (I*0.5 + 0.5*I = I)
        # 0->2 (0.5*0.5 = 0.25)
        
        # Let's check values by multiplying vector
        v = self.mat.create_vector()
        v.set_constant(1.0)
        
        # A * v = [1.5, 1.5, 1.5, 1.5]
        # C * v = A * (A * v) = A * 1.5 = 2.25
        
        resC = C.mult(v)
        self.assertTrue(np.allclose(resC.to_numpy(), 2.25))

    def test_threshold(self):
        # A has 0.5 off-diagonal.
        # A*A has 0.25 at distance 2 (0->2).
        # If we set threshold > 0.25, we should lose 0->2 connection.
        # Norm of 0.25*I (2x2) is sqrt(0.25^2 * 2) = sqrt(0.0625 * 2) = sqrt(0.125) approx 0.35.
        # Wait, Frobenius norm of 0.25*I_2 is sqrt(0.0625 + 0.0625) = sqrt(0.125) ~ 0.353.
        # So threshold 0.4 should kill it.
        
        C = self.mat.spmm(self.matB, 0.6)
        
        # C should NOT have 0->2 connection.
        # Row 0 should only have 0, 1.
        # 0->0 norm: I + 0.25? No.
        # 0->0 comes from 0->0->0 (I*I) + 0->1->0 (0.5*0). 
        # Wait, A is not symmetric in structure?
        # 0: 0, 1. (0->1 is 0.5).
        # 1: 1, 2. (1->0 is 0? No, 1->0 is not in adj).
        # Ah, my adj is directed ring: 0->1, 1->2, 2->3, 3->0.
        # Plus diagonal.
        
        # 0->0 path: 0->0->0 (I*I) = I. Norm 1.414. Kept.
        # 0->1 path: 0->0->1 (I*0.5) + 0->1->1 (0.5*I) = 0.5 + 0.5 = I. Norm 1.414. Kept.
        # 0->2 path: 0->1->2 (0.5*0.5) = 0.25. Norm 0.353. Killed by 0.4.
        
        # Verify C * v
        # C_00 = I. C_01 = I. C_02 = 0.
        # Row 0 sum = 2.0.
        # Without threshold: Row 0 sum = 2.25.
        
        v = self.mat.create_vector()
        v.set_constant(1.0)
        resC = C.mult(v)
        
        self.assertTrue(np.allclose(resC.to_numpy(), 2.0))

    def test_add(self):
        C = self.mat.add(self.matB, 1.0, 1.0) # A + A = 2A
        v = self.mat.create_vector()
        v.set_constant(1.0)
        res = C.mult(v)
        # A*v = 1.5. 2A*v = 3.0.
        self.assertTrue(np.allclose(res.to_numpy(), 3.0))

class TestSpMMDiverse(unittest.TestCase):
    def setUp(self):
        self.comm = MPI.COMM_WORLD if MPI else None
        self.rank = self.comm.Get_rank() if self.comm else 0
        self.size = self.comm.Get_size() if self.comm else 1
        
        # Determine mode
        import os
        mode = os.environ.get("VBCSR_TEST_MODE", "auto")
        
        if mode == "serial":
            self.use_distributed = False
            self.comm = None
            self.rank = 0
            self.size = 1
        elif mode == "distributed":
            self.use_distributed = True
        else:
            self.use_distributed = (self.size > 1)

        # Larger system for diverse connectivity
        self.n_blocks = 20
        self.block_size = 4
        self.block_sizes = [self.block_size] * self.n_blocks
        
        # Generate diverse adjacency (Hubs and Spokes)
        # Random importance for each block
        np.random.seed(42)
        importance = np.random.uniform(0.1, 1.0, self.n_blocks)
        
        if not self.use_distributed:
            # Serial Construction
            self.owned = list(range(self.n_blocks))
            self.count = self.n_blocks
            self.start = 0
        else:
            # Distributed Partitioning
            blocks_per_rank = self.n_blocks // self.size
            remainder = self.n_blocks % self.size
            self.start = self.rank * blocks_per_rank + min(self.rank, remainder)
            self.count = blocks_per_rank + (1 if self.rank < remainder else 0)
            self.owned = list(range(self.start, self.start + self.count))
        
        # Adjacency
        self.local_adj = []
        for i in self.owned:
            row_adj = []
            for j in range(self.n_blocks):
                # Probability depends on importance of both row and column
                prob = 0.2 * importance[i] * importance[j] * 4.0
                if np.random.random() < prob or i == j:
                    row_adj.append(j)
            self.local_adj.append(row_adj)
            
        if not self.use_distributed:
            self.mat = vbcsr.VBCSR.create_serial(self.n_blocks, self.block_sizes, self.local_adj, comm=self.comm)
        else:
            self.mat = vbcsr.VBCSR.create_distributed(self.owned, [self.block_size]*self.count, self.local_adj, comm=self.comm)
        
        # Fill with random values
        for i, row in enumerate(self.owned):
            for col in self.local_adj[i]:
                self.mat.add_block(row, col, np.random.uniform(-1, 1, (self.block_size, self.block_size)))
        self.mat.assemble()

    def test_diverse_connectivity(self):
        # C = A * A
        C = self.mat.spmm(self.mat, 0.0)
        
        # Verify via vector multiplication
        v = self.mat.create_vector()
        v.set_constant(1.0)
        
        # (A * A) * v = A * (A * v)
        res_direct = C.mult(v).to_numpy()
        res_sequential = self.mat.mult(self.mat.mult(v)).to_numpy()
        
        self.assertTrue(np.allclose(res_direct, res_sequential, atol=1e-10))

class TestSpMMDiverseComplex(unittest.TestCase):
    def setUp(self):
        self.comm = MPI.COMM_WORLD if MPI else None
        self.rank = self.comm.Get_rank() if self.comm else 0
        self.size = self.comm.Get_size() if self.comm else 1
        
        # Determine mode
        import os
        mode = os.environ.get("VBCSR_TEST_MODE", "auto")
        
        if mode == "serial":
            self.use_distributed = False
            self.comm = None
            self.rank = 0
            self.size = 1
        elif mode == "distributed":
            self.use_distributed = True
        else:
            self.use_distributed = (self.size > 1)
        
        # System for diverse connectivity
        self.n_blocks = 15
        self.block_size = 3
        self.block_sizes = [self.block_size] * self.n_blocks
        
        # Generate diverse adjacency
        np.random.seed(123)
        importance = np.random.uniform(0.1, 1.0, self.n_blocks)
        
        if not self.use_distributed:
            # Serial Construction
            self.owned = list(range(self.n_blocks))
            self.count = self.n_blocks
            self.start = 0
        else:
            # Distributed Partitioning
            blocks_per_rank = self.n_blocks // self.size
            remainder = self.n_blocks % self.size
            self.start = self.rank * blocks_per_rank + min(self.rank, remainder)
            self.count = blocks_per_rank + (1 if self.rank < remainder else 0)
            self.owned = list(range(self.start, self.start + self.count))
        
        # Adjacency
        self.local_adj = []
        for i in self.owned:
            row_adj = []
            for j in range(self.n_blocks):
                prob = 0.25 * importance[i] * importance[j] * 4.0
                if np.random.random() < prob or i == j:
                    row_adj.append(j)
            self.local_adj.append(row_adj)
            
        if not self.use_distributed:
            self.mat = vbcsr.VBCSR.create_serial(self.n_blocks, self.block_sizes, self.local_adj, dtype=np.complex128, comm=self.comm)
        else:
            self.mat = vbcsr.VBCSR.create_distributed(self.owned, [self.block_size]*self.count, self.local_adj, dtype=np.complex128, comm=self.comm)
        
        # Fill with random complex values
        for i, row in enumerate(self.owned):
            for col in self.local_adj[i]:
                data = np.random.uniform(-1, 1, (self.block_size, self.block_size)) + \
                       1j * np.random.uniform(-1, 1, (self.block_size, self.block_size))
                self.mat.add_block(row, col, data)
        self.mat.assemble()

    def test_diverse_connectivity_complex(self):
        # C = A * A
        C = self.mat.spmm(self.mat, 0.0)
        
        # Verify via vector multiplication
        v = self.mat.create_vector()
        v.set_constant(1.0 + 0.5j)
        
        res_direct = C.mult(v).to_numpy()
        res_sequential = self.mat.mult(self.mat.mult(v)).to_numpy()
        
        self.assertTrue(np.allclose(res_direct, res_sequential, atol=1e-10))

if __name__ == '__main__':
    unittest.main()

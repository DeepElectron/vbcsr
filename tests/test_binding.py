import numpy as np
import vbcsr
from mpi4py import MPI
import scipy.sparse.linalg
import sys

def test_serial():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size > 1:
        print("Skipping serial test in parallel run")
        return

    print("Testing Serial Construction...")
    
    # 2 blocks, size 2 each. Total 4x4 matrix.
    # Block 0: [0, 1] -> size 2
    # Block 1: [2, 3] -> size 2
    # Adj: 0->0, 0->1; 1->0, 1->1 (Dense block structure)
    
    global_blocks = 2
    block_sizes = [2, 2]
    adj = [[0, 1], [0, 1]]
    
    mat = vbcsr.VBCSR.create_serial(comm, global_blocks, block_sizes, adj, dtype=np.float64)
    
    # Fill diagonal blocks with identity, off-diagonal with 0.5
    # Block 0,0
    b00 = np.eye(2)
    mat.add_block(0, 0, b00, vbcsr.AssemblyMode.INSERT)
    
    # Block 1,1
    b11 = np.eye(2)
    mat.add_block(1, 1, b11, vbcsr.AssemblyMode.INSERT)
    
    # Block 0,1
    b01 = np.full((2,2), 0.5)
    mat.add_block(0, 1, b01, vbcsr.AssemblyMode.INSERT)
    
    # Block 1,0
    b10 = np.full((2,2), 0.5)
    mat.add_block(1, 0, b10, vbcsr.AssemblyMode.INSERT)
    
    mat.assemble()
    
    # Create vector
    v = mat.create_vector()
    v_np = np.array([1.0, 2.0, 3.0, 4.0])
    v.from_numpy(v_np)
    
    # Mult
    res = mat.mult(v)
    res_np = res.to_numpy()
    
    # Expected:
    # Block 0,1 (0.5) * [3,4] = [3.5, 3.5]
    # Block 0,0 (I) * [1,2] = [1, 2]
    # Row 0: 1 + 3.5 = 4.5
    # Row 1: 2 + 3.5 = 5.5
    
    # Block 1,0 (0.5) * [1,2] = [1.5, 1.5]
    # Block 1,1 (I) * [3,4] = [3, 4]
    # Row 2: 1.5 + 3 = 4.5
    # Row 3: 1.5 + 4 = 5.5
    
    expected = np.array([4.5, 5.5, 4.5, 5.5])
    
    if np.allclose(res_np, expected):
        print("Serial Mult Passed")
    else:
        print(f"Serial Mult FAILED. Got {res_np}, expected {expected}")
        sys.exit(1)

    # Test Operators
    v2 = v * 2.0
    if np.allclose(v2.to_numpy(), v_np * 2):
        print("Vector Scale Passed")
    else:
        print("Vector Scale FAILED")
        sys.exit(1)
        
    v3 = v + v2
    if np.allclose(v3.to_numpy(), v_np * 3):
        print("Vector Add Passed")
    else:
        print("Vector Add FAILED")
        sys.exit(1)

    # Matrix Scale
    mat2 = mat * 2.0
    res2 = mat2.mult(v)
    if np.allclose(res2.to_numpy(), expected * 2):
        print("Matrix Scale Passed")
    else:
        print("Matrix Scale FAILED")
        sys.exit(1)

def test_parallel():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 2:
        print("Skipping parallel test (run with -np 2)")
        return

    if rank == 0:
        print("Testing Parallel Construction...")

    # 2 Ranks. Rank 0 owns Block 0. Rank 1 owns Block 1.
    # Same 4x4 system.
    
    if rank == 0:
        owned_indices = [0]
        my_block_sizes = [2]
        my_adj = [[0, 1]] # Block 0 connects to 0 and 1
    else:
        owned_indices = [1]
        my_block_sizes = [2]
        my_adj = [[0, 1]] # Block 1 connects to 0 and 1
        
    mat = vbcsr.VBCSR.create_distributed(comm, owned_indices, my_block_sizes, my_adj, dtype=np.float64)
    
    # Add blocks
    # Everyone adds their own blocks (or any blocks, really, but let's do local)
    if rank == 0:
        mat.add_block(0, 0, np.eye(2), vbcsr.AssemblyMode.INSERT)
        mat.add_block(0, 1, np.full((2,2), 0.5), vbcsr.AssemblyMode.INSERT)
    else:
        mat.add_block(1, 1, np.eye(2), vbcsr.AssemblyMode.INSERT)
        mat.add_block(1, 0, np.full((2,2), 0.5), vbcsr.AssemblyMode.INSERT)
        
    mat.assemble()
    
    # Vector
    v = mat.create_vector()
    # Global vector is [1,2, 3,4]
    # Rank 0 has [1,2], Rank 1 has [3,4]
    if rank == 0:
        v.from_numpy(np.array([1.0, 2.0]))
    else:
        v.from_numpy(np.array([3.0, 4.0]))
        
    # Mult
    res = mat.mult(v)
    res_np = res.to_numpy()
    
    if rank == 0:
        expected = np.array([4.5, 5.5])
    else:
        expected = np.array([4.5, 5.5])
        
    if np.allclose(res_np, expected):
        print(f"Rank {rank} Parallel Mult Passed")
    else:
        print(f"Rank {rank} Parallel Mult FAILED. Got {res_np}, expected {expected}")
        sys.exit(1)
        
    # SciPy Solver (GMRES)
    # Solve A x = b. We know x=[1,2,3,4], b=[2.5, 4.0, 3.5, 5.0]
    # Let's try to solve for x given b.
    
    # Construct RHS vector
    b = mat.create_vector()
    b.from_numpy(expected)
    
    # Initial guess
    x0 = mat.create_vector()
    x0.set_constant(0.0)
    
    # Callback to monitor
    def callback(pr_norm):
        pass
        # if rank == 0: print(f"Resid: {pr_norm}")

    # Run GMRES
    # Note: scipy.sparse.linalg.gmres might not support distributed LinearOperator perfectly 
    # out of the box unless we are careful with the 'b' argument.
    # SciPy expects 'b' to be a numpy array.
    # If we pass a distributed numpy array (local part), SciPy doesn't know it's distributed.
    # However, since our LinearOperator._matvec takes a local numpy array and returns a local numpy array,
    # and the inner product in GMRES (if any) is done by SciPy... wait.
    # GMRES inside SciPy computes dot products of vectors. 
    # If vectors are distributed, SciPy's dot product (numpy.dot) will only sum local parts!
    # This is a problem. SciPy solvers are NOT MPI-aware.
    
    # To use SciPy solvers with MPI, we usually need a custom inner product or use a package that supports it.
    # OR, we gather everything to rank 0 (not scalable).
    
    # BUT, for this test, let's just verify that _matvec works as expected by SciPy interface.
    # We can check if mat * v_np works.
    
    v_np_local = v.to_numpy()
    res_scipy = mat._matvec(v_np_local) # Call _matvec directly to bypass LinearOperator shape check in distributed mode
    
    if np.allclose(res_scipy, expected):
        print(f"Rank {rank} SciPy Interface (_matvec) Passed")
    else:
        print(f"Rank {rank} SciPy Interface FAILED")
        sys.exit(1)

if __name__ == "__main__":
    test_serial()
    test_parallel()

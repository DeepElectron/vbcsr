import numpy as np
import vbcsr
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def main():
    comm = MPI.COMM_WORLD if MPI else None
    if comm and comm.Get_size() > 1:
        print("This example is meant for serial execution (np=1).")
        return

    print("=== Example 1: Basic Serial Usage ===")

    # 1. Create a VBCSR matrix
    # System: 2 blocks of size 2. Total 4x4.
    # Structure:
    # [ B00  B01 ]
    # [ B10  B11 ]
    
    global_blocks = 2
    block_sizes = [2, 2]
    adjacency = [[0, 1], [0, 1]] # Block 0 connects to 0,1; Block 1 connects to 0,1
    
    mat = vbcsr.VBCSR.create_serial(global_blocks, block_sizes, adjacency)
    
    # 2. Fill matrix
    # B00 = Identity
    mat.add_block(0, 0, np.eye(2))
    # B11 = Identity * 2
    mat.add_block(1, 1, np.eye(2) * 2)
    # Off-diagonals = 0.5
    mat.add_block(0, 1, np.full((2,2), 0.5))
    mat.add_block(1, 0, np.full((2,2), 0.5))
    
    mat.assemble()
    
    # 3. Create vector and fill with numpy array
    v_np = np.array([1.0, 1.0, 1.0, 1.0])
    
    # 4. Multiply using numpy array directly (auto-converted)
    res = mat.mult(v_np)
    print(f"Result (numpy input): {res.to_numpy()}")
    
    # 5. Multiply using DistVector
    v = mat.create_vector()
    v.from_numpy(v_np)
    res2 = mat.mult(v)
    print(f"Result (DistVector input): {res2.to_numpy()}")
    
    # 6. Matrix Operations
    # Add diagonal shift
    mat.shift(1.0) # Add 1.0 to diagonal elements
    res3 = mat.mult(v)
    print(f"Result after shift(1.0): {res3.to_numpy()}")
    
    # Add vector to diagonal
    diag_shift = np.array([0.1, 0.2, 0.3, 0.4])
    mat.add_diagonal(diag_shift)
    res4 = mat.mult(v)
    print(f"Result after add_diagonal: {res4.to_numpy()}")

if __name__ == "__main__":
    main()

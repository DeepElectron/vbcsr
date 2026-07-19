import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import vbcsr
from vbcsr import VBCSR
import sys

def test_spmf_diagonal():
    try:
        import vbcsr_core
        g = vbcsr_core.DistGraph(None)
        rank = g.rank
        size = g.size
    except ImportError:
        rank = 0
        size = 1

    if rank == 0:
        print("Testing VBCSR.spmf with diagonal matrix...")

    # 1. Setup Graph (1D Chain)
    n_blocks = 20
    block_sizes = [1] * n_blocks
    adj = []
    for i in range(n_blocks):
        neighbors = [i]
        if i > 0: neighbors.append(i-1)
        if i < n_blocks - 1: neighbors.append(i+1)
        adj.append(neighbors)
    
    # Create matrix
    mat = VBCSR.create_serial(n_blocks, block_sizes, adj, dtype=np.float64, comm=None)
    
    # Fill with diagonal values: S = diag(1, 2, ..., n_blocks)
    # Each rank fills its owned blocks
    owned_indices = mat.graph.owned_global_indices
    for gid in owned_indices:
        val = float(gid + 1)
        mat.add_block(gid, gid, np.array([[val]]))
    
    mat.assemble()

    # 2. Compute exp(S)
    if rank == 0:
        print("  Computing exp(S)...")
    mat_exp = mat.spmf("exp", verbose=False)
    
    # 3. Verify
    local_dense = mat_exp.to_dense()
    # local_dense shape is (owned_rows, all_local_cols)
    # Since it's diagonal, we only care about diagonal elements
    
    max_err = 0.0
    for gid in owned_indices:
        block = mat_exp.get_block(gid, gid)
        if block is not None:
            val = block[0, 0]
            expected = np.exp(float(gid + 1))
            err = abs(val - expected)
            max_err = max(max_err, err)
        else:
            # If diagonal not found, it's 0.
            expected = np.exp(float(gid + 1))
            max_err = max(max_err, expected)

    global_max_err = max_err
    
    if rank == 0:
        print(f"  Max Error: {global_max_err}")
        if global_max_err < 1e-10:
            print("  PASSED")
        else:
            print("  FAILED")
            sys.exit(1)

if __name__ == "__main__":
    test_spmf_diagonal()

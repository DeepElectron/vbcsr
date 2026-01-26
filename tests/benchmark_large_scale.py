import time
import numpy as np
import vbcsr
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import argparse
import scipy.sparse

def generate_random_structure(global_blocks, block_size_min, block_size_max, density, seed=42):
    np.random.seed(seed)
    block_sizes = np.random.randint(block_size_min, block_size_max + 1, size=global_blocks).tolist()
    
    adj = []
    for i in range(global_blocks):
        neighbors = set()
        neighbors.add((i - 1) % global_blocks)
        neighbors.add((i + 1) % global_blocks)
        neighbors.add(i)
        
        n_random = max(0, int(global_blocks * density) - 2)
        if n_random > 0:
            random_neighbors = np.random.choice(global_blocks, size=n_random, replace=False)
            neighbors.update(random_neighbors)
        adj.append(sorted(list(neighbors)))
    return block_sizes, adj

def main():
    parser = argparse.ArgumentParser(description="VBCSR Large Scale Benchmark")
    parser.add_argument("--blocks", type=int, default=1000, help="Total number of blocks")
    parser.add_argument("--min-block", type=int, default=10, help="Min block size")
    parser.add_argument("--max-block", type=int, default=50, help="Max block size")
    parser.add_argument("--density", type=float, default=0.01, help="Sparsity density")
    parser.add_argument("--complex", action="store_true", help="Use complex numbers")
    parser.add_argument("--scipy", action="store_true", help="Compare with SciPy (Serial only)")
    parser.add_argument("--mkl", action="store_true", help="Compare with MKL (Serial only)")
    args = parser.parse_args()

    if args.mkl:
        try:
            import sparse_dot_mkl
        except ImportError:
            print("Error: sparse_dot_mkl not found. Please install it to use --mkl.")
            return

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dtype = np.complex128 if args.complex else np.float64
    
    if rank == 0:
        print(f"=== VBCSR Benchmark ===")
        print(f"Ranks: {size}")
        print(f"Blocks: {args.blocks}")
        print(f"Block Size Range: [{args.min_block}, {args.max_block}]")
        print(f"Density: {args.density}")
        print(f"Dtype: {dtype}")
        print("Generating structure...", flush=True)

    # Generate structure on rank 0 and broadcast (or just replicate for simplicity)
    # For true large scale, we should distribute generation, but for comparison we need exact match.
    # Replicating generation is easiest for now.
    block_sizes, adj = generate_random_structure(args.blocks, args.min_block, args.max_block, args.density)
    
    # Partition for VBCSR
    blocks_per_rank = args.blocks // size
    remainder = args.blocks % size
    start_block = rank * blocks_per_rank + min(rank, remainder)
    my_count = blocks_per_rank + (1 if rank < remainder else 0)
    end_block = start_block + my_count
    owned_indices = list(range(start_block, end_block))
    my_block_sizes = block_sizes[start_block:end_block]
    my_adj = adj[start_block:end_block]

    # Build VBCSR
    if rank == 0:
        print("Building VBCSR...", flush=True)
    
    t0 = time.time()
    mat = vbcsr.VBCSR.create_distributed(comm, owned_indices, my_block_sizes, my_adj, dtype)
    
    # Generate data and fill
    # We also collect data for SciPy/MKL if needed
    scipy_rows = []
    scipy_cols = []
    scipy_data = []
    
    # Pre-calculate offsets for SciPy/MKL
    if (args.scipy or args.mkl) and rank == 0:
        row_offsets = np.zeros(args.blocks + 1, dtype=int)
        np.cumsum(block_sizes, out=row_offsets[1:])
    
    np.random.seed(42 + rank) # Different seed per rank for data
    
    for local_i, global_i in enumerate(owned_indices):
        r_dim = my_block_sizes[local_i]
        neighbors = my_adj[local_i]
        
        # Row offset for this block
        r_start = 0
        if args.scipy or args.mkl:
            # This is slow for distributed, but we assume --scipy/--mkl is used mostly in serial
            # or we only collect on rank 0. 
            # Actually, constructing SciPy matrix in parallel is hard.
            # We will only support SciPy/MKL comparison in serial (size=1).
            if size == 1:
                r_start = row_offsets[global_i]

        for global_j in neighbors:
            c_dim = block_sizes[global_j]
            if dtype == np.float64:
                data = np.random.rand(r_dim, c_dim)
            else:
                data = np.random.rand(r_dim, c_dim) + 1j * np.random.rand(r_dim, c_dim)
            
            mat.add_block(global_i, global_j, data)
            
            if (args.scipy or args.mkl) and size == 1:
                c_start = row_offsets[global_j]
                # Expand block to COO
                # This is heavy loop in python, might be slow for generation
                # Optimize: create meshgrid
                r_idx, c_idx = np.indices((r_dim, c_dim))
                scipy_rows.append((r_idx + r_start).flatten())
                scipy_cols.append((c_idx + c_start).flatten())
                scipy_data.append(data.flatten())

    mat.assemble()
    comm.Barrier()
    t_gen = time.time() - t0
    
    if rank == 0:
        print(f"VBCSR Generation Time: {t_gen:.4f} s")
        print(f"Matrix Shape: {mat.shape}")

    # VBCSR Benchmark
    v = mat.create_vector()
    v.set_constant(1.0)
    mat.mult(v) # Warmup
    
    if rank == 0:
        print("Benchmarking VBCSR SpMV...", flush=True)
        
    comm.Barrier()
    t_start = time.perf_counter()
    n_iter = 0
    while time.perf_counter() - t_start < 1.0 or n_iter < 10:
        mat.mult(v)
        n_iter += 1
    comm.Barrier()
    t_vbcsr = (time.perf_counter() - t_start) / n_iter
    
    if rank == 0:
        print(f"VBCSR Average SpMV Time: {t_vbcsr:.6f} s ({n_iter} iterations)")

    # SciPy/MKL Matrix Construction
    sp_mat = None
    v_np = None
    if (args.scipy or args.mkl) and size == 1:
        print("Building SciPy CSR...", flush=True)
        t0 = time.perf_counter()
        # Flatten lists
        if scipy_rows:
            all_rows = np.concatenate(scipy_rows)
            all_cols = np.concatenate(scipy_cols)
            all_data = np.concatenate(scipy_data)
            sp_mat = scipy.sparse.csr_matrix((all_data, (all_rows, all_cols)), shape=mat.shape)
        else:
            sp_mat = scipy.sparse.csr_matrix(mat.shape, dtype=dtype)
        t_sp_gen = time.perf_counter() - t0
        print(f"SciPy Generation Time: {t_sp_gen:.4f} s")
        
        v_np = v.to_numpy()

    # SciPy Comparison
    if args.scipy and size == 1:
        print("Benchmarking SciPy SpMV...", flush=True)
        t_start = time.perf_counter()
        n_iter_sp = 0
        while time.perf_counter() - t_start < 1.0 or n_iter_sp < 10:
            sp_mat.dot(v_np)
            n_iter_sp += 1
        t_scipy = (time.perf_counter() - t_start) / n_iter_sp
        print(f"SciPy Average SpMV Time: {t_scipy:.6f} s ({n_iter_sp} iterations)")
        
        print(f"Speedup (SciPy / VBCSR): {t_scipy / t_vbcsr:.2f}x")
        
        # Verify correctness
        res = mat.mult(v)
        res_sp = sp_mat.dot(v_np)
        diff = np.linalg.norm(res.to_numpy() - res_sp) / np.linalg.norm(res_sp)
        print(f"Relative Difference (SciPy): {diff:.2e}")

    # MKL Comparison
    if args.mkl and size == 1:
        print("Benchmarking MKL SpMV...", flush=True)
        
        # Ensure matrix is in CSR format and sorted indices for MKL if needed, 
        # but sparse_dot_mkl handles CSR.
        # sparse_dot_mkl might need float32 or float64 specifically, check support.
        # It supports float32, float64, complex64, complex128.
        
        t_start = time.perf_counter()
        n_iter_mkl = 0
        while time.perf_counter() - t_start < 1.0 or n_iter_mkl < 10:
            sparse_dot_mkl.dot_product_mkl(sp_mat, v_np)
            n_iter_mkl += 1
        t_mkl = (time.perf_counter() - t_start) / n_iter_mkl
        print(f"MKL Average SpMV Time: {t_mkl:.6f} s ({n_iter_mkl} iterations)")
        
        print(f"Speedup (MKL / VBCSR): {t_mkl / t_vbcsr:.2f}x")
        
        # Verify correctness
        res = mat.mult(v)
        res_mkl = sparse_dot_mkl.dot_product_mkl(sp_mat, v_np)
        diff = np.linalg.norm(res.to_numpy() - res_mkl) / np.linalg.norm(res_mkl)
        print(f"Relative Difference (MKL): {diff:.2e}")

if __name__ == "__main__":
    main()
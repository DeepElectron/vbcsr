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
    parser.add_argument("--mode", type=str, default="spmv", choices=["spmv", "spmm", "spgemm"], help="Benchmark mode")
    parser.add_argument("--num-vecs", type=int, default=32, help="Number of vectors for SpMM (K)")
    args = parser.parse_args()

    if args.mkl:
        try:
            import sparse_dot_mkl
        except ImportError:
            print("Error: sparse_dot_mkl not found. Please install it to use --mkl.")
            return

    if MPI is not None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1
    
    dtype = np.complex128 if args.complex else np.float64
    
    if rank == 0:
        print(f"=== VBCSR Benchmark ===")
        print(f"Mode: {args.mode}")
        print(f"Ranks: {size}")
        print(f"Blocks: {args.blocks}")
        print(f"Block Size Range: [{args.min_block}, {args.max_block}]")
        print(f"Density: {args.density}")
        print(f"Dtype: {dtype}")
        if args.mode == "spmm":
            print(f"Num Vecs: {args.num_vecs}")
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
    mat = vbcsr.VBCSR.create_distributed(
        owned_indices=owned_indices, 
        block_sizes=my_block_sizes, 
        adjacency=my_adj, 
        dtype=dtype, 
        comm=comm
    )
    
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

    # Prepare Inputs
    x_vbcsr = None
    x_np = None
    
    if args.mode == "spmv":
        x_vbcsr = mat.create_vector()
        x_vbcsr.set_constant(1.0)
        x_np = x_vbcsr.to_numpy() if size == 1 else None
    elif args.mode == "spmm":
        x_vbcsr = mat.create_multivector(args.num_vecs)
        x_vbcsr.set_constant(1.0) # Actually sets all to 1.0
        # Randomize content
        if dtype == np.float64:
            x_local_data = np.random.rand(x_vbcsr.local_rows, args.num_vecs)
        else:
            x_local_data = np.random.rand(x_vbcsr.local_rows, args.num_vecs) + 1j * np.random.rand(x_vbcsr.local_rows, args.num_vecs)
        x_vbcsr.from_numpy(x_local_data)
        x_np = x_vbcsr.to_numpy() if size == 1 else None # Full dense matrix
    elif args.mode == "spgemm":
        # For SpGEMM, let's just square the matrix: A * A
        # Or A * A.T
        x_vbcsr = mat # B = A
        # For SciPy, we need the matrix itself
        pass

    # Function to run benchmark loop
    def benchmark_op(op_func, name):
        if rank == 0:
            print(f"Benchmarking {name}...", flush=True)
            
        comm.Barrier()
        # Warmup
        op_func()
        comm.Barrier()
        
        t_start = time.perf_counter()
        n_iter = 0
        while time.perf_counter() - t_start < 1.0 or n_iter < 5:
            op_func()
            n_iter += 1
        comm.Barrier()
        t_avg = (time.perf_counter() - t_start) / n_iter
        
        if rank == 0:
            print(f"{name} Average Time: {t_avg:.6f} s ({n_iter} iterations)")
        return t_avg

    # 1. VBCSR Benchmark
    t_vbcsr = 0.0
    if args.mode == "spmv":
        t_vbcsr = benchmark_op(lambda: mat.mult(x_vbcsr), "VBCSR SpMV")
    elif args.mode == "spmm":
        t_vbcsr = benchmark_op(lambda: mat.mult(x_vbcsr), "VBCSR SpMM")
    elif args.mode == "spgemm":
        # spmm_self computes A * A if transA=False. 
        # Actually spmm_self might compute A * A^T or A^T * A. Check matrix.py
        # matrix.py: spmm_self(threshold, transA) -> C
        # If transA=False, it computes A * A? (Need to check doc/implementation)
        # Using general spmm: A.spmm(A)
        t_vbcsr = benchmark_op(lambda: mat.spmm(mat), "VBCSR SpGEMM")

    # 2. SciPy/MKL Comparison (Serial Only)
    sp_mat = None
    if (args.scipy or args.mkl) and size == 1:
        print("Building SciPy CSR...", flush=True)
        t0 = time.perf_counter()
        if scipy_rows:
            all_rows = np.concatenate(scipy_rows)
            all_cols = np.concatenate(scipy_cols)
            all_data = np.concatenate(scipy_data)
            sp_mat = scipy.sparse.csr_matrix((all_data, (all_rows, all_cols)), shape=mat.shape)
        else:
            sp_mat = scipy.sparse.csr_matrix(mat.shape, dtype=dtype)
        print(f"SciPy Generation Time: {time.perf_counter() - t0:.4f} s")

    if args.scipy and size == 1:
        if args.mode == "spmv":
            t_scipy = benchmark_op(lambda: sp_mat.dot(x_np), "SciPy SpMV")
        elif args.mode == "spmm":
             t_scipy = benchmark_op(lambda: sp_mat.dot(x_np), "SciPy SpMM")
        elif args.mode == "spgemm":
             t_scipy = benchmark_op(lambda: sp_mat.dot(sp_mat), "SciPy SpGEMM")
        
        print(f"Speedup (SciPy / VBCSR): {t_scipy / t_vbcsr:.2f}x")

    if args.mkl and size == 1:
        # Check MKL support for SpMM/SpGEMM
        # sparse_dot_mkl mainly supports SpMV (dot_product_mkl) and SpGEMM (gram_matrix_mkl for A^T*A, or dot_product_mkl with sparse B?)
        # dot_product_mkl(matrix_a, matrix_b, ...)
        # If matrix_b is dense, it's SpMM.
        # If matrix_b is sparse, it's SpGEMM.
        
        op_mkl = None
        if args.mode == "spmv":
            op_mkl = lambda: sparse_dot_mkl.dot_product_mkl(sp_mat, x_np)
            t_mkl = benchmark_op(op_mkl, "MKL SpMV")
        elif args.mode == "spmm":
            # Check if sparse_dot_mkl supports dense matrix B
            # It usually does.
            try:
                # sparse_dot_mkl might need F-contiguous dense matrix or specific layout
                # Force F-contiguous to avoid MKL looping over columns
                x_np_mkl = np.asfortranarray(x_np)
                op_mkl = lambda: sparse_dot_mkl.dot_product_mkl(sp_mat, x_np_mkl)
                t_mkl = benchmark_op(op_mkl, "MKL SpMM")
            except Exception as e:
                print(f"MKL SpMM failed or not supported: {e}")
                t_mkl = float('inf')
        elif args.mode == "spgemm":
            try:
                op_mkl = lambda: sparse_dot_mkl.dot_product_mkl(sp_mat, sp_mat)
                t_mkl = benchmark_op(op_mkl, "MKL SpGEMM")
            except Exception as e:
                print(f"MKL SpGEMM failed: {e}")
                t_mkl = float('inf')

        if t_mkl != float('inf'):
            print(f"Speedup (MKL / VBCSR): {t_mkl / t_vbcsr:.2f}x")
            if args.mode == 'spmm' and (t_mkl / t_vbcsr > 10.0):
                 print(f"Note: High speedup might indicate MKL bottleneck (e.g. layout or looping).")
            
    # Correctness Check
    if (args.scipy) and size == 1:
        print("Verifying correctness...", flush=True)
        res_vbcsr = None
        res_scipy = None
        
        if args.mode == "spmv":
            res_vbcsr = mat.mult(x_vbcsr).to_numpy()
            res_scipy = sp_mat.dot(x_np)
        elif args.mode == "spmm":
            res_vbcsr = mat.mult(x_vbcsr).to_numpy()
            res_scipy = sp_mat.dot(x_np)
        elif args.mode == "spgemm":
            res_vbcsr = mat.spmm(mat).to_scipy().toarray() # Convert VBCSR to SciPy/Array
            res_scipy = sp_mat.dot(sp_mat).toarray()
            
        if res_vbcsr is not None and res_scipy is not None:
            # For SpGEMM, struct might be slightly different zero pattern if logic differs, 
            # but comparing dense should work.
            diff = np.linalg.norm(res_vbcsr - res_scipy) / (np.linalg.norm(res_scipy) + 1e-10)
            print(f"Relative Difference (SciPy): {diff:.2e}")

if __name__ == "__main__":
    main()
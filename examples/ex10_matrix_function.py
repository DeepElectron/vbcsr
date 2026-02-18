"""
Example 10: Graph-based Matrix Function Precomputation

This example demonstrates:
1.  Constructing a matrix with a k-hop sparsity pattern but values from the original matrix.
    -   This allows `spmf` to utilize a wider sparsity pattern for better approximation.
2.  Computing the approximate inverse square root ($P \\approx A^{-1/2}$) using `VBCSR.spmf`.
    -   Using `method="dense"` which performs dense matrix functions on the subgraphs.
3.  Using the computed $P$ to form a preconditioner $M \\approx A^{-1}$ for Conjugate Gradient (CG).
4.  Benchmarking the convergence rate improvement.

Run with:
    python ex10_matrix_function.py
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import vbcsr
from vbcsr import VBCSR, MPI, HAS_MPI
import time
import matplotlib.pyplot as plt
import sys

# Set style for publication quality
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'sans-serif',
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'savefig.bbox': 'tight'
})

class MatrixFunctionBenchmark:
    def __init__(self, nx, ny, t=-1.0):
        self.nx = nx
        self.ny = ny
        self.t = t
        self.comm = MPI.COMM_WORLD if HAS_MPI else None
        self.rank = self.comm.Get_rank() if self.comm else 0
        self.size = self.comm.Get_size() if self.comm else 1
        
        # Create Laplacian-like matrix (Positive Definite for CG)
        # Shifted Laplacian: A = 4I - Laplacian  (or similar to ensure SPD)
        # Actually standard 2D Laplacian is A = 4I - Adj.
        # Eigenvalues are in [0, 8]. If we add shift, say +0.1, it's strictly PD.
        self.H = self._create_poisson_matrix()

    def _get_idx(self, ix, iy):
        return (ix % self.nx) * self.ny + (iy % self.ny)

    def _create_poisson_matrix(self):
        """Creates a 2D Poisson/Laplacian matrix (5-point stencil)."""
        n_dofs = self.nx * self.ny
        
        if self.rank == 0:
            print(f"Creating 2D Laplacian for {self.nx}x{self.ny} grid ({n_dofs} DOFs)...")

        # Partition
        dofs_per_rank = (n_dofs + self.size - 1) // self.size
        start_dof = self.rank * dofs_per_rank
        end_dof = min((self.rank + 1) * dofs_per_rank, n_dofs)
        
        owned_indices = list(range(start_dof, end_dof))
        block_sizes = [1] * len(owned_indices) # Scalar blocks
        
        # Adjacency
        adj = []
        for i in owned_indices:
            ix = i // self.ny
            iy = i % self.ny
            
            neighbors = []
            # 4 neighbors (periodic bounds)
            neighbors.append(self._get_idx(ix + 1, iy))
            neighbors.append(self._get_idx(ix - 1, iy))
            neighbors.append(self._get_idx(ix, iy + 1))
            neighbors.append(self._get_idx(ix, iy - 1))
            neighbors.append(i) # Diagonal
            
            adj.append(sorted(list(set(neighbors))))
            
        # Create VBCSR
        H = VBCSR.create_distributed(owned_indices, block_sizes, adj, comm=self.comm)
        
        # Fill Matrix: 4 on diagonal, -1 on off-diagonal (Negative Laplacian)
        # Plus shift to ensure good condition number for testing?
        # Let's use 4.1 on diagonal to be strictly PD.
        diag_val = 4.1
        off_val = -1.0
        
        for idx, i in enumerate(owned_indices):
            ix = i // self.ny
            iy = i % self.ny
            
            H.add_block(i, i, np.array([[diag_val]]))
            
            neighbors = [
                self._get_idx(ix + 1, iy),
                self._get_idx(ix - 1, iy),
                self._get_idx(ix, iy + 1),
                self._get_idx(ix, iy - 1)
            ]
            
            for nb in neighbors:
                H.add_block(i, nb, np.array([[off_val]]))
                
        H.assemble()
        return H

    def create_k_hop_structure(self, k: int):
        """
        Creates a matrix with k-hop sparsity pattern but values from self.H (where available).
        Zeros elsewhere.
        """
        if self.rank == 0:
            print(f"  Constructing {k}-hop structure...")
            
        if k == 1:
            return self.H.copy()
            
        # 1. Generate Structure using SpMM
        # S_1 = H
        # S_k requires multiplying S_{k-1} * H (symbolically or numerically)
        # We perform numerical SpMM but will ignore values later.
        
        S = self.H.copy()
        for _ in range(k - 1):
             # S accumulates neighbors. 
             # Note: SpMM might drop zeros if we are unlucky with cancellation, 
             # but with all negative off-diagonals and positive diagonal, 
             # A^k usually fills in.
             S = S.spmm(self.H)
             
        # 2. Zero out all values to keep only structure
        S.scale(0.0)
        
        # 3. Add original H values
        # This assumes S has a superset of H's structure.
        # And assume VBCSR.add handles adding into existing structure.
        S = S + self.H
        
        return S

    def run_benchmark(self):
        # Problem setup
        n = self.H.shape[0]
        b_vec = self.H.create_vector()
        # Random RHS
        np.random.seed(42)
        local_b = np.random.rand(b_vec.local_size)
        b_vec.from_numpy(local_b)
        
        # For callback tracking
        residuals = []
        def callback(xk):
            # Calculate residual norm |Ax - b|
            # Note: Scipy passes the current vector xk (numpy array)
            # But in distributed mode, scipy's callback might receive the gathered vector?
            # Or if we use proper LinearOperator...
            # For VBCSR as LinearOperator, Scipy sees it as (N, N).
            # If standard scipy.sparse.linalg.cg is used, everything on Rank 0 usually?
            # VBCSR supports distributed, but scipy's cg logic is serial unless we use a distributed solver.
            # VBCSR wraps MPI, but scipy's generic cg loop runs on rank 0 (usually) acting on global vectors?
            # WAIT. vbcsr.matrix.py inherits LinearOperator.
            # But the logic inside _matvec assumes local computation?
            # If we run scipy.sparse.linalg.cg on Rank 0, we pass the GLOBAL size operator.
            # But _matvec must handle the distribution.
            # Currently VBCSR.mult works distributedly. 
            pass

        # Since integrating distributed VBCSR with serial Scipy CG is tricky (Scipy expects full vector on one rank),
        # we will implement a simple Distributed CG solver here or assume we run small enough to gather.
        # Or simpler: Just implement PCG in python using VBCSR primitives.
        
        def distributed_pcg(A, b, M=None, tol=1e-8, max_iter=500):
            """
            Preconditioned Conjugate Gradient for distributed VBCSR.
            A: VBCSR matrix
            b: DistVector (RHS)
            M: Preconditioner (VBCSR or linear operator returning DistVector)
            """
            x = A.create_vector()
            x.set_constant(0.0)
            
            # r = b - A x = b (since x=0)
            r = b.duplicate()
            r.scale(1.0) # copy b
            # (Strictly r = b - A*x. A*0=0)
            
            # z = M^{-1} r. Here M is approx A^{-1}. So z = M r.
            if M is None:
                z = r.duplicate()
                z.axpy(0.0, r) # copy r
                # Or just z = r.copy()
            else:
                # Apply preconditioner M
                # If M is VBCSR, result is DistVector
                if isinstance(M, VBCSR):
                    z = M.mult(r)
                else:
                    # Assume M(r) -> DistVector
                    z = M(r)
            
            # p = z
            p = z.duplicate()
            
            # rho = r . z
            rho = r.dot(z)
            
            res_history = []
            init_res = np.sqrt(r.dot(r))
            res_history.append(init_res)
            
            if self.rank == 0:
                print(f"    CG Init Residual: {init_res:.4e}")
                
            for k in range(max_iter):
                # q = A p
                q = A.mult(p)
                
                # alpha = rho / (p . q)
                pq = p.dot(q)
                if pq == 0: break
                alpha = rho / pq
                
                # x = x + alpha p
                x.axpy(alpha, p)
                
                # r = r - alpha q
                r.axpy(-alpha, q)
                
                # Check convergence
                res_norm = np.sqrt(r.dot(r))
                res_history.append(res_norm)
                
                if res_norm < tol * init_res or res_norm < 1e-12:
                    if self.rank == 0:
                        print(f"    CG Converged at iter {k+1}, res={res_norm:.4e}")
                    break
                
                # z_new = M r
                if M is None:
                    z_new = r
                else:
                    if isinstance(M, VBCSR):
                        z_new = M.mult(r)
                    else:
                        z_new = M(r)
                
                # rho_new = r . z_new
                # Note: if M is None, z_new is r, so rho_new = r.r
                rho_new = r.dot(z_new)
                
                # beta = rho_new / rho
                beta = rho_new / rho
                
                # p = z_new + beta p
                # careful with overwriting p
                # p = z_new + beta * p
                p.scale(beta)
                p.axpy(1.0, z_new)
                
                rho = rho_new
                
            return x, res_history

        results = {}
        
        # 1. Baseline (No Preconditioner)
        if self.rank == 0:
            print("\nRunning Baseline CG (No Preconditioner)...")
        t0 = time.time()
        _, hist_base = distributed_pcg(self.H, b_vec, M=None)
        results['Baseline'] = hist_base
        if self.rank == 0:
            print(f"  Baseline finished in {time.time()-t0:.2f}s ({len(hist_base)} iters)")

        # 2. Preconditioned with k-hop ISQRT
        k_hops = [1, 2, 3] # Test different overlaps
        for k in k_hops:
            if self.rank == 0:
                print(f"\nConstructing Preconditioner k={k}...")
            
            # A_k: values of A, structure of A^k
            A_k = self.create_k_hop_structure(k)
            
            if self.rank == 0:
                 print(f"  Computing spmf('isqrt', method='dense') on {k}-hop matrix...")
            
            # P = A_k^{-1/2} approx
            # Using dense method on local blocks/structure
            t_spmf = time.time()
            P = A_k.spmf("isqrt", method="dense")
            if self.rank == 0:
                print(f"  spmf finished in {time.time()-t_spmf:.2f}s")
            
            # M = P * P approx A^{-1}
            # Instead of forming M explicitly (which might be dense), 
            # we define an operator to apply P then P.
            # But wait, P is VBCSR.
            # DistributedPCG expects M.mult(r) or M(r).
            # Let's define a callable class.
            
            class PrecondOp:
                def __init__(self, P_mat):
                    self.P = P_mat
                def __call__(self, r):
                    # z = P * (P * r)
                    # Use a temporary vector
                    tmp = self.P.mult(r)
                    return self.P.mult(tmp)

            M_op = PrecondOp(P)
            
            if self.rank == 0:
                print(f"Running Preconditioned CG (k={k})...")
            
            t0 = time.time()
            _, hist_pcg = distributed_pcg(self.H, b_vec, M=M_op)
            results[f'k={k}'] = hist_pcg
            if self.rank == 0:
                print(f"  PCG (k={k}) finished in {time.time()-t0:.2f}s ({len(hist_pcg)} iters)")
                
        # Plotting
        if self.rank == 0:
            print("\nGenerating plot...")
            plt.figure()
            
            # Plot Baseline
            plt.semilogy(results['Baseline'], label='Baseline (No Precond)', 
                        color='black', linestyle='--', linewidth=2)
            
            colors = ['#E63946', '#457B9D', '#2A9D8F', '#F4A261']
            for i, k in enumerate(k_hops):
                lbl = f'Precond (k={k})'
                if f'k={k}' in results:
                    plt.semilogy(results[f'k={k}'], label=lbl, 
                                color=colors[i % len(colors)], linewidth=2)
            
            plt.xlabel('Iteration')
            plt.ylabel('Residual Norm $|Ax-b|$')
            plt.title(f'CG Convergence with Graph-Based Matrix Function Preconditioner')
            plt.legend()
            plt.grid(True, which="both", ls="--")
            plt.savefig("cg_convergence.pdf")
            print("Saved cg_convergence.pdf")

if __name__ == "__main__":
    # Use larger grid to see convergence differences
    nx, ny = 60, 60 
    bench = MatrixFunctionBenchmark(nx, ny)
    bench.run_benchmark()

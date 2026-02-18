"""
Example 07: Graphene Density Matrix via McWeeny Purification

This example demonstrates:
1. Generating a tight-binding Hamiltonian for a graphene lattice.
2. Computing the density matrix using the McWeeny purification algorithm.
3. Benchmarking VBCSR performance and accuracy with different filter thresholds (epsilon).
4. Generating publication-quality plots for NNZ evolution and error scaling.

Run with:
    mpirun -np 4 python ex07_graphene_purification.py
"""

import numpy as np
import scipy.sparse as sp
import vbcsr
from vbcsr import VBCSR, MPI, HAS_MPI
import time
import matplotlib.pyplot as plt
import os

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

class GraphenePurification:
    def __init__(self, nx, ny, t=-2.7):
        self.nx = nx
        self.ny = ny
        self.t = t
        self.comm = MPI.COMM_WORLD if HAS_MPI else None
        self.rank = self.comm.Get_rank() if self.comm else 0
        self.size = self.comm.Get_size() if self.comm else 1
        self.n_atoms = 2 * nx * ny
        self.H = None
        self.H_dense_full = None
        self.P_exact_full = None

    def get_idx(self, ix, iy, sub):
        """Global index mapping: (ix, iy, sub) -> global_idx"""
        return (ix % self.nx) * self.ny * 2 + (iy % self.ny) * 2 + sub

    def create_hamiltonian(self):
        """Creates the distributed VBCSR Hamiltonian."""
        if self.rank == 0:
            print(f"Creating Hamiltonian for {self.nx}x{self.ny} graphene ({self.n_atoms} atoms)...")

        # Partition atoms among ranks
        atoms_per_rank = (self.n_atoms + self.size - 1) // self.size
        start_atom = self.rank * atoms_per_rank
        end_atom = min((self.rank + 1) * atoms_per_rank, self.n_atoms)
        
        owned_indices = list(range(start_atom, end_atom))
        block_sizes = [1] * len(owned_indices)
        
        # Build Adjacency
        adj = []
        for i in owned_indices:
            ix = i // (self.ny * 2)
            iy = (i % (self.ny * 2)) // 2
            sub = i % 2
            
            neighbors = []
            if sub == 0: # A atom
                neighbors.append(self.get_idx(ix, iy, 1))
                neighbors.append(self.get_idx(ix - 1, iy, 1))
                neighbors.append(self.get_idx(ix, iy - 1, 1))
            else: # B atom
                neighbors.append(self.get_idx(ix, iy, 0))
                neighbors.append(self.get_idx(ix + 1, iy, 0))
                neighbors.append(self.get_idx(ix, iy + 1, 0))
            
            neighbors.append(i) # Diagonal
            adj.append(list(set(neighbors)))

        # Create VBCSR matrix
        self.H = VBCSR.create_distributed(owned_indices, block_sizes, adj, comm=self.comm)
        
        # Fill Matrix
        for idx, i in enumerate(owned_indices):
            ix = i // (self.ny * 2)
            iy = (i % (self.ny * 2)) // 2
            sub = i % 2
            
            self.H.add_block(i, i, np.zeros((1, 1))) # Diagonal
            
            # Hopping terms
            t_mat = np.array([[self.t]])
            if sub == 0:
                self.H.add_block(i, self.get_idx(ix, iy, 1), t_mat)
                self.H.add_block(i, self.get_idx(ix - 1, iy, 1), t_mat)
                self.H.add_block(i, self.get_idx(ix, iy - 1, 1), t_mat)
            else:
                self.H.add_block(i, self.get_idx(ix, iy, 0), t_mat)
                self.H.add_block(i, self.get_idx(ix + 1, iy, 0), t_mat)
                self.H.add_block(i, self.get_idx(ix, iy + 1, 0), t_mat)
                
        self.H.assemble()
        return self.H

    def compute_exact_density_matrix(self):
        """Computes exact density matrix via dense diagonalization on Rank 0."""
        # Gather full matrix
        local_csr = self.H.to_scipy(format='csr')
        full_csr = None
        
        if self.comm:
            all_mats = self.comm.gather(local_csr, root=0)
            if self.rank == 0:
                full_csr = sp.vstack(all_mats)
        else:
            full_csr = local_csr
            
        if self.rank == 0:
            print("Computing exact diagonalization (reference)...")
            self.H_dense_full = full_csr.toarray()
            evals, evecs = np.linalg.eigh(self.H_dense_full)
            n_occ = len(evals) // 2 # Half-filling
            self.P_exact_full = evecs[:, :n_occ] @ evecs[:, :n_occ].T.conj()
            
    def purify(self, epsilon, max_iter=30):
        """
        Runs McWeeny purification: P_{n+1} = 3P_n^2 - 2P_n^3.
        Returns P_final, nnz_history (list of (iter, nnz)).
        """
        # Scale H to [0, 1]
        # Spectal bounds: [-3|t|, 3|t|]
        t_val = abs(self.t)
        E_min, E_max = -3.1 * t_val, 3.1 * t_val
        
        P = self.H.copy()
        P.scale(-1.0 / (E_max - E_min))
        P.shift(E_max / (E_max - E_min))
        
        nnz_history = []
        if self.rank == 0:
            print(f"  Purification (eps={epsilon:.1e}): Iter 0, NNZ={P.get_block_density()*self.n_atoms**2:.0f}") # Approx NNZ for tracking
        
        for i in range(max_iter):
            # P2 = P * P
            P2 = P.spmm(P, threshold=epsilon)
            
            # P3 = P2 * P
            P3 = P2.spmm(P, threshold=epsilon)
            
            # P_new = 3P^2 - 2P^3
            P = 3.0 * P2 - 2.0 * P3
            P.filter_blocks(epsilon)
            
            # Track NNZ
            # get_block_density returns density. Multiply by N^2 to get nnz approx or exact if block_size=1
            # Assuming block size 1 here for simplicity of NNZ definition
            current_nnz = P.get_block_density() * (self.n_atoms**2) 
            nnz_history.append(current_nnz)
            
            if self.rank == 0 and (i+1) % 5 == 0:
                 print(f"    Iter {i+1}: NNZ ~ {current_nnz:.0f}")

        return P, nnz_history

    def compute_error(self, P_dist):
        """Computes Max Abs Error against exact P on Rank 0."""
        # Gather P
        local_csr = P_dist.to_scipy(format='csr')
        full_P = None
        
        if self.comm:
            all_mats = self.comm.gather(local_csr, root=0)
            if self.rank == 0:
                full_P = sp.vstack(all_mats).toarray()
        else:
            full_P = local_csr.toarray()
            
        err = 0.0
        if self.rank == 0:
            diff = full_P - self.P_exact_full
            err = np.max(np.abs(diff))
        
        return err

def run_experiment():
    nx, ny = 20, 20
    sim = GraphenePurification(nx, ny)
    sim.create_hamiltonian()
    
    # Compute reference (only feasible for small systems)
    sim.compute_exact_density_matrix()
    
    eps_list = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    results = {
        'eps': eps_list,
        'final_nnz': [],
        'error': [],
        'nnz_histories': {}
    }
    
    if sim.rank == 0:
        print("\n--- Starting Parameter Sweep ---")

    for eps in eps_list:
        P, hist = sim.purify(eps)
        err = sim.compute_error(P)
        final_nnz = hist[-1] if hist else 0
        
        if sim.rank == 0:
            print(f"  Result eps={eps:.1e}: Error={err:.2e}, Final NNZ={final_nnz:.0f}")
            results['final_nnz'].append(final_nnz)
            results['error'].append(err)
            results['nnz_histories'][eps] = hist

    if sim.rank == 0:
        # Save plots
        print("\nGenerating plots...")
        
        n_atoms = sim.n_atoms
        total_nnz_dense = n_atoms**2
        
        # 1. NNZ vs Iteration
        plt.figure()
        for eps in eps_list:
            hist = results['nnz_histories'][eps]
            plt.plot(range(1, len(hist)+1), hist, label=f'eps={eps:.0e}', marker='o', markersize=4)
        
        plt.axhline(y=total_nnz_dense, color='k', linestyle='--', alpha=0.5, label='Fully Dense')
        plt.xlabel('Iteration')
        plt.ylabel('Number of Non-Zeros (NNZ)')
        plt.title(f'NNZ Evolution during Purification\nGraphene {n_atoms} atoms')
        plt.legend()
        plt.yscale('log')
        plt.savefig('purification_nnz.pdf')
        print("Saved purification_nnz.pdf")

        # 2. Final Error vs Epsilon
        plt.figure()
        plt.plot(eps_list, results['error'], 'o-', color='#E63946')
        plt.xlabel(r'Filter Threshold ($\epsilon$)')
        plt.ylabel('Max Absolute Error vs Exact')
        plt.title('Accuracy vs Filter Threshold')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.savefig('purification_error.pdf')
        print("Saved purification_error.pdf")
        
        # 3. Final NNZ vs Epsilon (Tradeoff)
        plt.figure()
        plt.plot(eps_list, results['final_nnz'], 's-', color='#2A9D8F', label='Purified Matrix')
        plt.axhline(y=total_nnz_dense, color='k', linestyle='--', alpha=0.5, label='Fully Dense')
        plt.xlabel(r'Filter Threshold ($\epsilon$)')
        plt.ylabel('Final NNZ')
        plt.title('Sparsity vs Filter Threshold')
        plt.xscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.savefig('purification_tradeoff.pdf')
        print("Saved purification_tradeoff.pdf")

if __name__ == "__main__":
    run_experiment()

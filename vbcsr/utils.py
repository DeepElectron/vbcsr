import numpy as np
from vbcsr import VBCSR, MPI, HAS_MPI
from scipy.linalg import eigh_tridiagonal

def jackson_kernel(N):
    """
    Computes the Jackson damping kernel coefficients g_n for n = 0..N-1.
    This kernel removes Gibbs oscillations (ripples) in the approximation.
    """
    n = np.arange(N)
    # Formula for Jackson kernel
    # g_n = [ (N-n+1)*cos(pi*n/(N+1)) + sin(pi*n/(N+1))*cot(pi/(N+1)) ] / (N+1)
    
    numerator = (N - n + 1) * np.cos(np.pi * n / (N + 1)) + \
                np.sin(np.pi * n / (N + 1)) / np.tan(np.pi / (N + 1))
    
    return numerator / (N + 1)


def fermi_dirac(E, beta, mu):
    """
    Fermi-Dirac distribution function.
    """
    # Use expit for numerical stability to avoid overflow in exp
    from scipy.special import expit
    return expit(-beta * (E - mu))

def gaussian(E, mu):
    """
    Fermi-Dirac distribution function.
    """
    # Use expit for numerical stability to avoid overflow in exp
    return np.exp(-(E - mu)**2)


def compute_kpm_coeffs(n_kpm, func):
    """
    Generates Chebyshev moments for the Fermi-Dirac function using FFT.
    
    Args:
        n_kpm (int): Number of moments (Chebyshev degree).
        temp (float): Temperature in Kelvin.
        mu (float): Chemical potential in eV.
        scale (float): Half-width of the spectrum (eV).
        center (float): Center of the spectrum (eV).
        
    Returns:
        coeffs (ndarray): The first n_kpm Chebyshev coefficients.
    """

    # 1. Sample the function on the Chebyshev grid
    # We need 2*N points for the FFT to extract N moments correctly via orthogonality
    k = np.arange(2 * n_kpm)
    theta_k = np.pi * k / n_kpm  # dE * i where dE = pi / n_kpm in your loop implies 2*n_kpm range
    x_k = np.cos(theta_k)
    
    # Calculate Fermi-Dirac on the scaled grid
    # f_vals = fermi_dirac(x_k, beta_re, mu_re)
    f_vals = func(x_k)

    # 2. Compute Coefficients using FFT
    fft_vals = np.fft.fft(f_vals).real
    
    # Normalize: The 1/N factor comes from the discrete orthogonality condition
    coeffs = fft_vals[:n_kpm] / n_kpm
    
    # 3. Apply Jackson Kernel
    kernel = jackson_kernel(n_kpm)
    damped_coeffs = coeffs * kernel
    
    # Common convention: coeffs[0] /= 2.0 
    damped_coeffs[0] /= 2.0 

    return damped_coeffs


def kpm_apply(matrix, vector, coeffs, scale, center):
    """
    Applies a KPM-approximated matrix function to a vector or multivector.
    
    Args:
        matrix (VBCSR): The matrix H.
        vector (DistVector or DistMultiVector): The input vector or multivector v.
        coeffs (ndarray): Chebyshev coefficients.
        scale (float): Half-width of the spectrum.
        center (float): Center of the spectrum.
        
    Returns:
        The result y = f(H) v.
    """
    # Initialize recurrence:
    # v0 = v
    # v1 = (H v0 - center * v0) / scale
    # v_{n+1} = 2 * (H v_n - center * v_n) / scale - v_{n-1}
    
    n_kpm = len(coeffs)
    if n_kpm == 0:
        res = vector.duplicate()
        res.set_constant(0.0)
        return res

    # y = c0 * v0
    y = vector.duplicate()
    y.scale(coeffs[0])
    
    if n_kpm == 1:
        return y
        
    # v0 = vector
    v0 = vector.duplicate()
    
    # v1 = (H * v0 - center * v0) / scale
    v1 = matrix.mult(v0)
    v1.axpby(-center/scale, v0, 1.0/scale)
    
    # y += c1 * v1
    y.axpy(coeffs[1], v1)
    
    for n in range(2, n_kpm):
        # v_next = 2 * (H * v1 - center * v1) / scale - v0
        v_next = matrix.mult(v1)
        v_next.axpby(-2.0*center/scale, v1, 2.0/scale)
        v_next.axpy(-1.0, v0)
        
        # y += coeffs[n] * v_next
        y.axpy(coeffs[n], v_next)
        
        # Update for next step
        v0 = v1
        v1 = v_next
        
    return y

def ozaki_residues(M_cut:int=1000):
    """
    It computes the poles and residues of the Ozaki formulism.

    Parameters
    ----------
    M_cut (int (optional)): The cutoff, i.e. 2 * M_cut is dimension of the Ozaki matrix.

    Returns
    -------
    poles: The positive half of poles, in ascending order.
    res: The residues of positive half of poles.
    ref:  Karrasch, C., V. Meden, and K. Schönhammer. "Finite-temperature linear conductance from the Matsubara Greens function without analytic continuation to the real axis." Physical Review B 82.12 (2010): 125114.
    """
    if not isinstance(M_cut, int):
        M_cut = int(M_cut)
    # diagonal part of Ozaki matrix

    N_curt = int(2 * M_cut)
    diag = np.zeros(N_curt)
    # off-diagonal part of Ozaki matrix
    off_diag = np.array([.5 / np.sqrt((2. * n - 1) * (2. * n + 1)) for n in range(1, N_curt)])
    # The reciprocal of poles (eigenvalues) are in numerically ascending order, we just need the positive half.
    evals, evecs = eigh_tridiagonal(d=diag, e=off_diag, select='i', select_range=(N_curt // 2, N_curt - 1))
    # return poles in ascending order
    poles = np.flip(1. / evals)
    # compute residues
    res = np.flip(np.abs(evecs[0, :]) ** 2 / (4. * evals ** 2)) # eq.12

    return poles, res

def create_graphene_hamiltonian(nx, ny, t=-2.7):
    """Create a graphene Hamiltonian (honeycomb lattice)."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if hasattr(comm, 'Get_rank') else 0
    size = comm.Get_size() if hasattr(comm, 'Get_size') else 1
    
    n_atoms = 2 * nx * ny
    atoms_per_rank = (n_atoms + size - 1) // size
    start_atom = rank * atoms_per_rank
    end_atom = min((rank + 1) * atoms_per_rank, n_atoms)
    
    owned_indices = list(range(start_atom, end_atom))
    block_sizes = [1] * len(owned_indices)
    
    def get_idx(ix, iy, sub):
        return (ix % nx) * ny * 2 + (iy % ny) * 2 + sub

    adj = []
    for i in owned_indices:
        ix = i // (ny * 2)
        iy = (i % (ny * 2)) // 2
        sub = i % 2
        neighbors = []
        if sub == 0: # A atom
            neighbors.extend([get_idx(ix, iy, 1), get_idx(ix - 1, iy, 1), get_idx(ix, iy - 1, 1)])
        else: # B atom
            neighbors.extend([get_idx(ix, iy, 0), get_idx(ix + 1, iy, 0), get_idx(ix, iy + 1, 0)])
        neighbors.append(i)
        adj.append(list(set(neighbors)))

    H = VBCSR.create_distributed(owned_indices, block_sizes, adj, dtype=np.complex128, comm=comm)
    for i in owned_indices:
        ix = i // (ny * 2)
        iy = (i % (ny * 2)) // 2
        sub = i % 2
        if sub == 0:
            H.add_block(i, get_idx(ix, iy, 1), np.array([[t]], dtype=np.complex128))
            H.add_block(i, get_idx(ix - 1, iy, 1), np.array([[t]], dtype=np.complex128))
            H.add_block(i, get_idx(ix, iy - 1, 1), np.array([[t]], dtype=np.complex128))
        else:
            H.add_block(i, get_idx(ix, iy, 0), np.array([[t]], dtype=np.complex128))
            H.add_block(i, get_idx(ix + 1, iy, 0), np.array([[t]], dtype=np.complex128))
            H.add_block(i, get_idx(ix, iy + 1, 0), np.array([[t]], dtype=np.complex128))
    H.assemble()
    return H
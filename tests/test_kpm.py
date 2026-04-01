import numpy as np
import vbcsr
from vbcsr import VBCSR
from vbcsr.utils import compute_kpm_coeffs, kpm_apply
import sys

def test_kpm_diagonal():
    print("Testing KPM application with diagonal matrix...")
    
    # 1. Setup Matrix (Diagonal)
    n_blocks = 50
    block_sizes = [1] * n_blocks
    adj = [[i] for i in range(n_blocks)]
    
    mat = VBCSR.create_serial(n_blocks, block_sizes, adj, dtype=np.float64)
    
    # H = diag(0.1, 0.2, ..., 5.0)
    for i in range(n_blocks):
        val = (i + 1) * 0.1
        mat.add_block(i, i, np.array([[val]]))
    mat.assemble()
    
    # 2. Setup KPM Coeffs for f(x) = exp(x)
    # Spectrum of H is [0.1, 5.0]
    # We need to scale to [-1, 1]
    center = 2.55
    scale = 2.5
    # (x - 2.55) / 2.5 maps [0.05, 5.05] to [-1, 1]
    
    def func_scaled(x):
        # x is in [-1, 1]
        # E = x * scale + center
        return np.exp(x * scale + center)
    
    n_kpm = 200
    coeffs = compute_kpm_coeffs(n_kpm, func_scaled)
    
    # 3. Apply KPM to a vector of ones
    v = mat.create_vector()
    v.set_constant(1.0)
    
    y_kpm = kpm_apply(mat, v, coeffs, scale, center)
    y_kpm_np = y_kpm.to_numpy()
    
    # 4. Exact result
    # y_exact = exp(H) * 1
    y_exact = np.exp(np.arange(1, n_blocks + 1) * 0.1)
    
    # 5. Compare
    max_err = np.max(np.abs(y_kpm_np - y_exact))
    print(f"  Max Error: {max_err}")
    
    # With Jackson kernel, error is expected to be around 0.04 for n_kpm=200
    if max_err < 0.05:
        print("  PASSED")
    else:
        print("  FAILED")
        sys.exit(1)

if __name__ == "__main__":
    test_kpm_diagonal()

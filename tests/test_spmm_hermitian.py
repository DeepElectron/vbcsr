"""spmm_hermitian against the full product, on products that are Hermitian.

The contract: for C = op(A) B known Hermitian, only the upper block triangle
is computed and the lower is mirrored as its conjugate transpose. Correctness
means (1) it matches the full spmm up to the drop threshold, and (2) the
result is exactly Hermitian -- which the full product is not once the
threshold drops blocks asymmetrically.
"""

import sys
from unittest.mock import patch

import numpy as np
import scipy.sparse  # noqa: F401  -- imported before the extension: importing
# it afterwards trips numpy's one-load-per-process guard in this stack.

import _workspace_bootstrap  # noqa: F401

with patch.dict(sys.modules, {"mpi4py": None}):
    from vbcsr import VBCSR


def _build(n_blocks, bs, rng, hermitian):
    """A serial block matrix with a symmetric pattern, optionally Hermitian."""
    adj = [[j for j in range(n_blocks)] for _ in range(n_blocks)]
    mat = VBCSR.create_serial(n_blocks, [bs] * n_blocks, adj, dtype=np.complex128, comm=None)
    blocks = {}
    for i in range(n_blocks):
        for j in range(n_blocks):
            if j < i:
                continue
            block = rng.standard_normal((bs, bs)) + 1j * rng.standard_normal((bs, bs))
            if hermitian and i == j:
                block = 0.5 * (block + block.conj().T)
            blocks[(i, j)] = block
    for (i, j), block in blocks.items():
        mat.add_block(i, j, block)
        if i != j:
            mat.add_block(j, i, block.conj().T if hermitian else
                          rng.standard_normal((bs, bs)) + 1j * rng.standard_normal((bs, bs)))
    mat.assemble()
    return mat


def test_congruence_matches_full_product():
    rng = np.random.default_rng(7)
    n_blocks, bs = 6, 3
    S = _build(n_blocks, bs, rng, hermitian=True)
    Z = _build(n_blocks, bs, rng, hermitian=False)

    SZ = S.spmm(Z, 0.0)
    full = Z.spmm(SZ, 0.0, True).to_scipy().toarray()      # Z^H (S Z)
    half = Z.spmm_hermitian(SZ, 0.0, True).to_scipy().toarray()

    scale = np.abs(full).max()
    assert np.abs(full - half).max() < 1e-12 * scale
    assert np.abs(half - half.conj().T).max() == 0.0        # exactly Hermitian


def test_square_of_hermitian_matches():
    rng = np.random.default_rng(11)
    X = _build(5, 4, rng, hermitian=True)
    full = X.spmm(X, 0.0).to_scipy().toarray()
    half = X.spmm_hermitian(X, 0.0).to_scipy().toarray()
    scale = np.abs(full).max()
    assert np.abs(full - half).max() < 1e-12 * scale
    assert np.abs(half - half.conj().T).max() == 0.0


if __name__ == "__main__":
    test_congruence_matches_full_product()
    test_square_of_hermitian_matches()
    print("PASSED")

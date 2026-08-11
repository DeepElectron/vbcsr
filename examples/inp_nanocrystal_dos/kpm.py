"""MPI-parallel kernel polynomial DOS using VBCSR multivectors."""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class KPMResult:
    energy_ev: np.ndarray
    integration_weights_ev: np.ndarray
    dos_states_per_ev: np.ndarray
    dos_per_orbital_per_ev: np.ndarray
    moments: np.ndarray
    jackson_weights: np.ndarray
    spectral_bounds_ev: Tuple[float, float]
    resolution_ev: float
    num_orbitals: int
    num_random_vectors: int


def jackson_kernel(num_moments: int) -> np.ndarray:
    """Jackson damping factors in the convention used by RSATB."""
    if num_moments <= 0:
        raise ValueError("num_moments must be positive")
    n = np.arange(num_moments, dtype=np.float64)
    denominator = float(num_moments + 1)
    theta = np.pi / denominator
    return (
        (denominator - n) * np.cos(theta * n)
        + np.sin(theta * n) / np.tan(theta)
    ) / denominator


def _apply_rescaled(matrix, source, target, center: float, half_width: float):
    matrix.mult(source, y=target)
    target.axpy(-center, source)
    target.scale(1.0 / half_width)


def _mean_bdot(left, right) -> complex:
    return sum(left.bdot(right)) / float(left.num_vectors)


def _batch_moments(matrix, random_vectors, num_moments: int, center: float, half_width: float):
    """Two-moments-per-matvec Hermitian recurrence from RSATB's DOS path."""
    moments = np.zeros(num_moments, dtype=np.complex128)
    t0 = random_vectors.copy()
    moments[0] = _mean_bdot(t0, t0)
    if num_moments == 1:
        return moments

    t1 = matrix.create_multivector(random_vectors.num_vectors)
    t2 = matrix.create_multivector(random_vectors.num_vectors)
    _apply_rescaled(matrix, t0, t1, center, half_width)
    moments[1] = _mean_bdot(t0, t1)

    n = 1
    while True:
        odd = 2 * n - 1
        even = 2 * n
        if odd > 1 and odd < num_moments:
            moments[odd] = 2.0 * _mean_bdot(t1, t0) - moments[1]
        if even < num_moments:
            moments[even] = 2.0 * _mean_bdot(t1, t1) - moments[0]
        if 2 * n + 1 >= num_moments:
            break
        _apply_rescaled(matrix, t1, t2, center, half_width)
        t2.axpby(-1.0, t0, 2.0)
        t0, t1, t2 = t1, t2, t0
        n += 1
    return moments


def compute_dos(
    matrix,
    spectral_bounds_ev: Tuple[float, float],
    num_moments: int = 1024,
    num_random_vectors: int = 16,
    batch_size: int = 8,
    num_energy_points: int = 2000,
    seed: int = 2026,
    comm=None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> KPMResult:
    """Estimate the total single-spin DOS of a finite Hermitian matrix.

    Random Rademacher vectors estimate ``Tr[T_n(H)]``.  Columns are propagated
    in batches, while VBCSR performs every sparse multivector multiply and
    ghost exchange collectively on the matrix communicator.
    """
    if num_moments <= 1:
        raise ValueError("num_moments must be at least 2")
    if num_random_vectors <= 0 or batch_size <= 0:
        raise ValueError("num_random_vectors and batch_size must be positive")
    if num_energy_points <= 1:
        raise ValueError("num_energy_points must be at least 2")

    lower, upper = (float(value) for value in spectral_bounds_ev)
    if not np.isfinite(lower + upper) or upper <= lower:
        raise ValueError("spectral bounds must be finite and increasing")
    center = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    rank = comm.Get_rank() if comm is not None else 0

    moments = np.zeros(num_moments, dtype=np.complex128)
    completed = 0
    while completed < num_random_vectors:
        current_batch = min(batch_size, num_random_vectors - completed)
        random_vectors = matrix.create_multivector(current_batch)
        # A rank-local stream avoids constructing the global dense random
        # vectors while remaining reproducible for a fixed MPI decomposition.
        rank_seed = np.random.SeedSequence((int(seed), int(rank), int(completed)))
        rng = np.random.default_rng(rank_seed)
        signs = 2 * rng.integers(
            0, 2, size=(random_vectors.local_rows, current_batch), dtype=np.int8
        ) - 1
        random_vectors.from_numpy(signs.astype(matrix.dtype, copy=False))
        moments += current_batch * _batch_moments(
            matrix, random_vectors, num_moments, center, half_width
        )
        completed += current_batch
        if progress is not None:
            progress(completed, num_random_vectors)
    moments /= float(num_random_vectors)

    # The stochastic zeroth moment is exactly N for unnormalised Rademacher
    # vectors, up to floating-point reduction rounding.
    num_orbitals = int(round(moments[0].real))
    weights = jackson_kernel(num_moments)
    # Ascending Gauss--Chebyshev nodes resolve the integrable endpoint weight
    # far better than a uniform E grid.  The returned quadrature weights make
    # sum(DOS * weight) reproduce the number of states directly.
    theta = np.pi * (np.arange(num_energy_points) + 0.5) / num_energy_points
    x = -np.cos(theta)
    energy = center + half_width * x
    integration_weights = (
        np.pi * half_width * np.sqrt(1.0 - x * x) / float(num_energy_points)
    )

    t0 = np.ones_like(x)
    t1 = x.copy()
    density_x = weights[0] * moments[0].real * t0
    density_x += 2.0 * weights[1] * moments[1].real * t1
    for n in range(2, num_moments):
        t2 = 2.0 * x * t1 - t0
        density_x += 2.0 * weights[n] * moments[n].real * t2
        t0, t1 = t1, t2
    density_x /= np.pi * np.sqrt(np.maximum(1.0e-15, 1.0 - x * x))
    dos = density_x / half_width

    return KPMResult(
        energy_ev=energy,
        integration_weights_ev=integration_weights,
        dos_states_per_ev=dos,
        dos_per_orbital_per_ev=dos / float(num_orbitals),
        moments=moments,
        jackson_weights=weights,
        spectral_bounds_ev=(lower, upper),
        resolution_ev=float(np.pi * half_width / num_moments),
        num_orbitals=num_orbitals,
        num_random_vectors=num_random_vectors,
    )

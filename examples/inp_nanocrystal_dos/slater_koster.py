"""Real-harmonic Slater--Koster block construction for the InP model."""

from functools import lru_cache
from typing import Dict, Mapping, Tuple

import numpy as np

from .parameters import BASIS, NORB, InPParameters, PAPER_PARAMETERS


def _angular_momentum(label: str) -> int:
    if label == "s":
        return 0
    if label.startswith("p"):
        return 1
    if label.startswith("d"):
        return 2
    raise ValueError(f"unsupported orbital label: {label}")


_SHELL_SLICES: Mapping[str, Mapping[int, slice]] = {
    symbol: {
        l: slice(next(i for i, label in enumerate(labels) if _angular_momentum(label) == l),
                 1 + max(i for i, label in enumerate(labels) if _angular_momentum(label) == l))
        for l in sorted({_angular_momentum(label) for label in labels})
    }
    for symbol, labels in BASIS.items()
}


def _local_frame(direction: np.ndarray) -> np.ndarray:
    ez = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(ez)
    if norm <= 1.0e-14:
        raise ValueError("a Slater--Koster bond direction cannot be zero")
    ez = ez / norm
    if abs(ez[2]) < 1.0 - 1.0e-12:
        ex = np.cross((0.0, 0.0, 1.0), ez)
        ex /= np.linalg.norm(ex)
    else:
        ex = np.asarray((1.0, 0.0, 0.0))
    ey = np.cross(ez, ex)
    return np.column_stack((ex, ey, ez))


def _p_rotation(frame: np.ndarray) -> np.ndarray:
    # m-index order (py, pz, px), matching RSATB's real-harmonic SK builder.
    axes = (1, 2, 0)
    return np.asarray(
        [[frame[global_axis, local_axis] for global_axis in axes] for local_axis in axes],
        dtype=np.float64,
    )


def _canonical_d_matrices() -> np.ndarray:
    sqrt2 = np.sqrt(2.0)
    sqrt6 = np.sqrt(6.0)
    return np.asarray(
        (
            ((0, 1 / sqrt2, 0), (1 / sqrt2, 0, 0), (0, 0, 0)),
            ((0, 0, 0), (0, 0, 1 / sqrt2), (0, 1 / sqrt2, 0)),
            ((-1 / sqrt6, 0, 0), (0, -1 / sqrt6, 0), (0, 0, 2 / sqrt6)),
            ((0, 0, 1 / sqrt2), (0, 0, 0), (1 / sqrt2, 0, 0)),
            ((1 / sqrt2, 0, 0), (0, -1 / sqrt2, 0), (0, 0, 0)),
        ),
        dtype=np.float64,
    )


def _d_rotation(frame: np.ndarray) -> np.ndarray:
    canonical = _canonical_d_matrices()
    rotated = np.einsum("ai,lij,bj->lab", frame, canonical, frame)
    return np.einsum("gab,lab->lg", canonical, rotated)


def _shell_rotation(l: int, frame: np.ndarray) -> np.ndarray:
    if l == 0:
        return np.ones((1, 1), dtype=np.float64)
    if l == 1:
        return _p_rotation(frame)
    if l == 2:
        return _d_rotation(frame)
    raise ValueError("only s, p, and d orbitals are supported")


def _aligned_shell_block(row_l: int, col_l: int, m_abs: int, value: float) -> np.ndarray:
    if m_abs < 0 or m_abs > min(row_l, col_l):
        raise ValueError(f"invalid |m|={m_abs} for l=({row_l}, {col_l})")
    block = np.zeros((2 * row_l + 1, 2 * col_l + 1), dtype=np.float64)
    signed = -value if row_l > col_l and (row_l + col_l) % 2 else value
    if m_abs == 0:
        block[row_l, col_l] = signed
    else:
        block[row_l - m_abs, col_l - m_abs] = signed
        block[row_l + m_abs, col_l + m_abs] = signed
    return block


def _build_declared_pair_block(
    row_symbol: str,
    col_symbol: str,
    direction: np.ndarray,
    channels: Mapping[Tuple[int, int, int], float],
) -> np.ndarray:
    frame = _local_frame(direction)
    block = np.zeros((NORB[row_symbol], NORB[col_symbol]), dtype=np.float64)
    rotations: Dict[int, np.ndarray] = {}
    for (row_l, col_l, m_abs), value in channels.items():
        if row_l not in _SHELL_SLICES[row_symbol] or col_l not in _SHELL_SLICES[col_symbol]:
            raise ValueError(
                f"channel {(row_l, col_l, m_abs)} is incompatible with "
                f"{row_symbol}-{col_symbol} basis"
            )
        rotations.setdefault(row_l, _shell_rotation(row_l, frame))
        rotations.setdefault(col_l, _shell_rotation(col_l, frame))
        aligned = _aligned_shell_block(row_l, col_l, m_abs, value)
        rotated = rotations[row_l].T @ aligned @ rotations[col_l]
        block[_SHELL_SLICES[row_symbol][row_l], _SHELL_SLICES[col_symbol][col_l]] += rotated
    return block


def pair_block(
    row_symbol: str,
    col_symbol: str,
    direction,
    parameters: InPParameters = PAPER_PARAMETERS,
) -> np.ndarray:
    """Return ``H[row_symbol, col_symbol]`` for a row-to-column bond vector."""
    direction = np.asarray(direction, dtype=np.float64)
    direct = parameters.pair_channels.get((row_symbol, col_symbol))
    if direct is not None:
        return _build_declared_pair_block(row_symbol, col_symbol, direction, direct)
    reverse = parameters.pair_channels.get((col_symbol, row_symbol))
    if reverse is None:
        raise KeyError(f"no Slater--Koster parameters for {row_symbol}-{col_symbol}")
    return _build_declared_pair_block(col_symbol, row_symbol, -direction, reverse).T.copy()


def onsite_block(symbol: str, parameters: InPParameters = PAPER_PARAMETERS) -> np.ndarray:
    """Return the diagonal onsite block for one atom."""
    return np.diag(parameters.onsite[symbol]).astype(np.float64, copy=False)


@lru_cache(maxsize=128)
def tetrahedral_pair_block(
    row_symbol: str,
    col_symbol: str,
    direction_key: Tuple[int, int, int],
) -> np.ndarray:
    """Cached helper for the integer zinc-blende bond directions."""
    return pair_block(row_symbol, col_symbol, direction_key)

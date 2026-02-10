from . import vbcsr_core
from vbcsr_core import AssemblyMode, PhaseConvention
import numpy as np

_MODE_MAP = {
    "add": AssemblyMode.ADD,
    "insert": AssemblyMode.INSERT,
    "set": AssemblyMode.INSERT,
}

_CONVENTION_MAP = {
    "R": PhaseConvention.R_ONLY,
    "r": PhaseConvention.R_ONLY,
    "R+tau": PhaseConvention.R_AND_POSITION,
    "r+tau": PhaseConvention.R_AND_POSITION,
}


def _resolve_mode(mode):
    if isinstance(mode, str):
        key = mode.lower()
        if key not in _MODE_MAP:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(_MODE_MAP.keys())}")
        return _MODE_MAP[key]
    return mode


def _resolve_convention(convention):
    if isinstance(convention, str):
        if convention not in _CONVENTION_MAP:
            raise ValueError(
                f"Unknown convention '{convention}'. Choose from: {list(_CONVENTION_MAP.keys())}"
            )
        return _CONVENTION_MAP[convention]
    return convention


class ImageContainer(vbcsr_core.ImageContainer):
    """
    Pythonic wrapper for vbcsr_core.ImageContainer.

    Provides:
      - ``add_block``  - single block, optional *R*, string *mode*
      - ``add_blocks`` - batched parallel insertion
      - ``assemble``   - finalize assembly (exchange remote blocks)
      - ``sample_k``   - Fourier transform to k-space
    """

    def __init__(self, atomic_data):
        super().__init__(atomic_data)

    # ------------------------------------------------------------------ #
    #  add_block
    # ------------------------------------------------------------------ #
    def add_block(self, g_row, g_col, data, R=None, mode="add"):
        """
        Add a single dense block.

        Args:
            g_row (int): Global row index (atom index).
            g_col (int): Global column index (atom index).
            data (array-like): 2-D block (n_orb_row x n_orb_col).
            R (list[int], optional): Lattice vector [rx, ry, rz]. Default [0,0,0].
            mode (str): ``"add"`` | ``"insert"`` | ``"set"``.
        """
        if R is None:
            R = [0, 0, 0]
        else:
            R = [int(x) for x in R]
            if len(R) != 3:
                raise ValueError("R must have exactly 3 elements")

        super().add_block(R, g_row, g_col,
                          np.asarray(data, dtype=np.float64),
                          _resolve_mode(mode))

    # ------------------------------------------------------------------ #
    #  add_blocks  (parallel batch)
    # ------------------------------------------------------------------ #
    def add_blocks(self, g_rows, g_cols, data_list, R_list=None, mode="add"):
        """
        Add multiple blocks in parallel (OpenMP on the C++ side).

        Args:
            g_rows (list[int]): Global row indices.
            g_cols (list[int]): Global column indices.
            data_list (list[ndarray]): List of 2-D block arrays.
            R_list (list[list[int]], optional):
                Lattice vectors per block.  If *None*, every block
                uses [0, 0, 0].
            mode (str): ``"add"`` | ``"insert"`` | ``"set"``.
        """
        n = len(g_rows)
        if R_list is None:
            R_list = [[0, 0, 0]] * n
        else:
            R_list = [[int(x) for x in r] for r in R_list]

        data_list = [np.asarray(d, dtype=np.float64) for d in data_list]

        super().add_blocks(R_list, list(g_rows), list(g_cols),
                           data_list, _resolve_mode(mode))

    # ------------------------------------------------------------------ #
    #  assemble
    # ------------------------------------------------------------------ #
    def assemble(self):
        """Finalize assembly - exchange remote blocks between MPI ranks."""
        super().assemble()

    # ------------------------------------------------------------------ #
    #  sample_k
    # ------------------------------------------------------------------ #
    def sample_k(self, k_point, convention="R"):
        """
        Fourier-transform real-space blocks to a single k-point.

        Args:
            k_point (array-like): Length-3 fractional k-point.
            convention (str): ``"R"`` | ``"R+tau"``.

        Returns:
            VBCSR: Complex-valued block-sparse matrix at this k-point.
        """
        from .matrix import VBCSR

        k_point = np.asarray(k_point, dtype=np.float64).ravel()
        if k_point.size != 3:
            raise ValueError("k_point must have exactly 3 elements")

        core_result = super().sample_k(k_point, _resolve_convention(convention))

        # Wrap in VBCSR (same pattern as spmm / transpose in matrix.py)
        obj = VBCSR.__new__(VBCSR)
        obj.graph = core_result.graph
        obj.dtype = np.complex128
        obj._core = core_result
        obj.comm = None
        obj.shape = (None, None)
        obj._global_nnz = None
        return obj


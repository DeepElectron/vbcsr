from . import vbcsr_core
from vbcsr_core import AssemblyMode, PhaseConvention
import numpy as np
from .matrix import VBCSR

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


class ImageContainer:
    """
    Pythonic wrapper for vbcsr_core.ImageContainer (Double or Complex).

    Provides:
      - ``add_block``  - single block, optional *R*, string *mode*
      - ``add_blocks`` - batched parallel insertion
      - ``assemble``   - finalize assembly (exchange remote blocks)
      - ``sample_k``   - Fourier transform to k-space
    """

    def __init__(self, atomic_data, dtype=np.float64):
        self.atomic_data = atomic_data
        self.dtype = np.dtype(dtype)
        
        if self.dtype == np.float64:
             # Depending on how pybind exposes it, atomic_data might need to be passed as pointer
             # But pybind handles object conversion if `AtomicData` is bound.
             self._core = vbcsr_core.ImageContainer(atomic_data)
        elif self.dtype == np.complex128:
             self._core = vbcsr_core.ImageContainer_Complex(atomic_data)
        else:
             raise ValueError("Unsupported dtype. Use float64 or complex128.")

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

        # Cast data to correct dtype
        data_arr = np.asarray(data, dtype=self.dtype)
        
        self._core.add_block(R, g_row, g_col, data_arr, _resolve_mode(mode))

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

        # Cast all data blocks
        data_list = [np.asarray(d, dtype=self.dtype) for d in data_list]

        self._core.add_blocks(R_list, list(g_rows), list(g_cols),
                              data_list, _resolve_mode(mode))

    def assemble(self):
        """Finalize assembly - exchange remote blocks between MPI ranks."""
        self._core.assemble()

    def sample_k(self, k_point, convention="R", symm=False):
        """
        Fourier-transform real-space blocks to a single k-point.

        Args:
            k_point (array-like): Length-3 fractional k-point.
            convention (str): ``"R"`` | ``"R+tau"``.

        Returns:
            VBCSR: Complex-valued block-sparse matrix at this k-point.
        """
        k_point = np.asarray(k_point, dtype=np.float64).ravel()
        if k_point.size != 3:
            raise ValueError("k_point must have exactly 3 elements")

        core_result = self._core.sample_k(k_point, _resolve_convention(convention))

        # Wrap in VBCSR (same pattern as spmm / transpose in matrix.py)
        obj = VBCSR.__new__(VBCSR)
        obj.graph = core_result.graph
        obj.dtype = np.complex128
        obj._core = core_result
        obj.comm = self.atomic_data.comm if hasattr(self.atomic_data, 'comm') else None
        norb = self.atomic_data.norb()
        obj.shape = (norb, norb)
        obj._global_nnz = None

        if symm:
            obj = obj + obj.T
            obj /= 2.0
        return obj

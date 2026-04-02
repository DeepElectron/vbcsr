import numpy as np
from typing import Union, Any

from ._wrapper_utils import core_buffer, duplicate_wrapper, infer_dtype_from_core, reduced_extent
from .vector import DistVector

class DistMultiVector:
    """
    Distributed multi-vector class wrapping the C++ DistMultiVector.
    
    Represents a collection of vectors (columns) distributed across MPI ranks.
    Stored in column-major format.
    """
    
    def __init__(self, core_obj: Any, comm: Any = None):
        """
        Initialize the DistMultiVector.
        
        Args:
            core_obj: The underlying C++ DistMultiVector object.
            comm: MPI communicator.
        """
        self._core = core_obj
        self.dtype = infer_dtype_from_core(core_obj)
        self.comm = comm
        self._global_rows = reduced_extent(self.comm, self.local_rows)

    def _local_buffer(self) -> np.ndarray:
        return core_buffer(self._core)[:self.local_rows, :]

    @property
    def local_rows(self) -> int:
        """Returns the number of locally owned rows."""
        return self._core.local_rows

    @property
    def ghost_rows(self) -> int:
        """Returns the number of ghost rows cached locally."""
        return self._core.ghost_rows

    @property
    def num_vectors(self) -> int:
        """Returns the number of vectors (columns)."""
        return self._core.num_vectors

    @property
    def ndim(self) -> int:
        return 2

    @property
    def shape(self):
        if self._global_rows is not None:
            return (self._global_rows, self.num_vectors)
        return (None, self.num_vectors)

    @property
    def size(self) -> int:
        s = self.shape
        if s[0] is not None:
            return s[0] * s[1]
        return 0

    def copy(self) -> 'DistMultiVector':
        return duplicate_wrapper(self)

    def __len__(self) -> int:
        s = self.shape[0]
        return s if s is not None else 0

    def __repr__(self):
        s = self.shape
        shape_str = f"({s[0]}, {s[1]})" if s[0] is not None else f"(Unknown, {s[1]})"
        return f"<DistMultiVector of shape {shape_str}, dtype={self.dtype}>"

    def to_numpy(self) -> np.ndarray:
        """
        Convert the locally owned part to a NumPy array.
        
        Returns:
            np.ndarray: A 2D array of shape (local_rows, num_vectors).
        """
        # Buffer is (rows, cols) F-contiguous
        return self._local_buffer()

    def from_numpy(self, arr: np.ndarray) -> None:
        """
        Update the locally owned part from a NumPy array.
        
        Args:
            arr (np.ndarray): Input array of shape (local_rows, num_vectors).
            
        Raises:
            ValueError: If shape mismatch.
        """
        if arr.shape != (self.local_rows, self.num_vectors):
             raise ValueError(f"Array shape {arr.shape} mismatch. Expected ({self.local_rows}, {self.num_vectors})")
        self._local_buffer()[:] = arr

    def __array__(self) -> np.ndarray:
        """Support for np.array(vec)."""
        return self.to_numpy()

    def sync_ghosts(self) -> None:
        """Synchronize ghost elements."""
        self._core.sync_ghosts()

    def reduce_ghosts(self) -> None:
        """Reduce ghost contributions back to their owners."""
        self._core.reduce_ghosts()
    
    def duplicate(self) -> 'DistMultiVector':
        """
        Create a deep copy.
        
        Returns:
            DistMultiVector: A new multivector with same structure and data.
        """
        return DistMultiVector(self._core.duplicate(), self.comm)

    def set_constant(self, val: Union[float, complex, int]) -> None:
        """Set all elements to a constant value."""
        self._core.set_constant(val)

    def set_random_normal(self, normalize: bool = False) -> None:
        """Set all elements to random normal values."""
        self._core.set_random_normal(normalize)

    def scale(self, alpha: Union[float, complex, int]) -> None:
        """Scale all elements by a scalar."""
        self._core.scale(alpha)

    def axpy(self, alpha: Union[float, complex, int], x: 'DistMultiVector') -> None:
        """Compute y = alpha * x + y (in-place)."""
        if isinstance(x, DistMultiVector):
            self._core.axpy(alpha, x._core)
        else:
            raise TypeError("axpy expects DistMultiVector")

    def axpby(self, alpha: Union[float, complex, int], x: 'DistMultiVector', beta: Union[float, complex, int]) -> None:
        """Compute y = alpha * x + beta * y (in-place)."""
        if isinstance(x, DistMultiVector):
            self._core.axpby(alpha, x._core, beta)
        else:
            raise TypeError("axpby expects DistMultiVector")

    def pointwise_mult(self, other: Union['DistMultiVector', DistVector]) -> None:
        """
        Element-wise multiplication.
        
        Args:
            other: Can be DistMultiVector (same shape) or DistVector (broadcast across columns).
        """
        if isinstance(other, DistMultiVector):
            self._core.pointwise_mult(other._core)
        elif isinstance(other, DistVector):
            self._core.pointwise_mult_vec(other._core)
        else:
            raise TypeError("pointwise_mult expects DistMultiVector or DistVector")

    def bdot(self, other: 'DistMultiVector') -> list:
        """
        Compute batch dot product (column-wise dot).
        
        Args:
            other (DistMultiVector): The other multivector.
            
        Returns:
            list: A list of scalar dot products, one for each column.
        """
        if isinstance(other, DistMultiVector):
            return self._core.bdot(other._core)
        else:
            raise TypeError("bdot expects DistMultiVector")

    # Operators

    def __neg__(self) -> 'DistMultiVector':
        res = self.duplicate()
        res._core.scale(-1.0)
        return res

    def __pos__(self) -> 'DistMultiVector':
        return self
    
    def __add__(self, other: Union['DistMultiVector', float, complex, int, np.ndarray]) -> 'DistMultiVector':
        res = self.duplicate()
        res += other
        return res

    def __iadd__(self, other: Union['DistMultiVector', float, complex, int, np.ndarray]) -> 'DistMultiVector':
        if isinstance(other, DistMultiVector):
            self._core.axpy(1.0, other._core)
        elif np.isscalar(other) or isinstance(other, np.ndarray):
            self._local_buffer()[:] += other
        else:
            return NotImplemented
        return self

    def __sub__(self, other: Union['DistMultiVector', float, complex, int, np.ndarray]) -> 'DistMultiVector':
        res = self.duplicate()
        res -= other
        return res

    def __isub__(self, other: Union['DistMultiVector', float, complex, int, np.ndarray]) -> 'DistMultiVector':
        if isinstance(other, DistMultiVector):
            self._core.axpy(-1.0, other._core)
        elif np.isscalar(other) or isinstance(other, np.ndarray):
            self._local_buffer()[:] -= other
        else:
            return NotImplemented
        return self

    def __mul__(self, other: Union['DistMultiVector', DistVector, float, complex, int, np.ndarray]) -> 'DistMultiVector':
        res = self.duplicate()
        res *= other
        return res

    def __imul__(self, other: Union['DistMultiVector', DistVector, float, complex, int, np.ndarray]) -> 'DistMultiVector':
        if np.isscalar(other):
            self._core.scale(other)
        elif isinstance(other, DistMultiVector):
            self._core.pointwise_mult(other._core)
        elif isinstance(other, DistVector):
            self._core.pointwise_mult_vec(other._core)
        elif isinstance(other, np.ndarray):
            self._local_buffer()[:] *= other
        else:
            return NotImplemented
        return self

    def __radd__(self, other: Union[float, complex, int, np.ndarray]) -> 'DistMultiVector':
        return self.__add__(other)

    def __rsub__(self, other: Union[float, complex, int, np.ndarray]) -> 'DistMultiVector':
        res = -self
        res += other
        return res

    def __rmul__(self, other: Union[float, complex, int, np.ndarray]) -> 'DistMultiVector':
        return self.__mul__(other)

    def __truediv__(self, other: Union['DistMultiVector', DistVector, float, complex, int, np.ndarray]) -> 'DistMultiVector':
        res = self.duplicate()
        res /= other
        return res

    def __itruediv__(self, other: Union['DistMultiVector', DistVector, float, complex, int, np.ndarray]) -> 'DistMultiVector':
        if np.isscalar(other):
            self._core.scale(1.0 / other)
        elif isinstance(other, DistMultiVector):
            self._local_buffer()[:] /= core_buffer(other._core)[:other.local_rows, :]
        elif isinstance(other, DistVector):
            self._local_buffer()[:] /= core_buffer(other._core)[:other.local_size]
        elif isinstance(other, np.ndarray):
            self._local_buffer()[:] /= other
        else:
            return NotImplemented
        return self

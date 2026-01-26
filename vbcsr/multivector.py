import numpy as np
import vbcsr_core
from typing import Union, Optional, Any
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
        self.dtype = np.complex128 if "Complex" in core_obj.__class__.__name__ else np.float64
        self.comm = comm
        self._global_rows = None
        if self.comm:
            try:
                self._global_rows = self.comm.allreduce(self.local_rows)
            except:
                pass
        else:
            self._global_rows = self.local_rows

    @property
    def local_rows(self) -> int:
        """Returns the number of locally owned rows."""
        return self._core.local_rows

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
        obj = self.duplicate()
        obj.comm = self.comm
        return obj

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
        buf = np.array(self._core, copy=False)
        return buf[:self.local_rows, :]

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
        buf = np.array(self._core, copy=False)
        buf[:self.local_rows, :] = arr

    def __array__(self) -> np.ndarray:
        """Support for np.array(vec)."""
        return self.to_numpy()

    def sync_ghosts(self) -> None:
        """Synchronize ghost elements."""
        self._core.sync_ghosts()
    
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
            buf = np.array(self._core, copy=False)
            buf[:self.local_rows, :] += other
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
            buf = np.array(self._core, copy=False)
            buf[:self.local_rows, :] -= other
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
            buf = np.array(self._core, copy=False)
            buf[:self.local_rows, :] *= other
        else:
            return NotImplemented
        return self

    def __truediv__(self, other: Union['DistMultiVector', DistVector, float, complex, int, np.ndarray]) -> 'DistMultiVector':
        res = self.duplicate()
        res /= other
        return res

    def __itruediv__(self, other: Union['DistMultiVector', DistVector, float, complex, int, np.ndarray]) -> 'DistMultiVector':
        if np.isscalar(other):
            self._core.scale(1.0 / other)
        elif isinstance(other, DistMultiVector):
            buf = np.array(self._core, copy=False)
            buf[:self.local_rows, :] /= np.array(other._core, copy=False)
        elif isinstance(other, DistVector):
            buf = np.array(self._core, copy=False)
            buf[:self.local_rows, :] /= np.array(other._core, copy=False)
        elif isinstance(other, np.ndarray):
            buf = np.array(self._core, copy=False)
            buf[:self.local_rows, :] /= other
        else:
            return NotImplemented
        return self

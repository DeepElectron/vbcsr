import numpy as np
import vbcsr_core
from typing import Union, Optional, Any

class DistVector:
    """
    Distributed vector class wrapping the C++ DistVector.
    
    This class manages a distributed vector partitioned across MPI ranks.
    It supports basic arithmetic operations, synchronization of ghost elements,
    and interoperability with NumPy.
    """
    
    def __init__(self, core_obj: Any, comm: Any = None):
        """
        Initialize the DistVector.
        
        Args:
            core_obj: The underlying C++ DistVector object (DistVector_Double or DistVector_Complex).
            comm: MPI communicator.
        """
        self._core = core_obj
        self.dtype = np.complex128 if "Complex" in core_obj.__class__.__name__ else np.float64
        self.comm = comm
        self._global_size = None
        if self.comm:
            try:
                self._global_size = self.comm.allreduce(self.local_size)
            except:
                pass
        else:
            self._global_size = self.local_size

    @property
    def local_size(self) -> int:
        """Returns the number of locally owned elements."""
        return self._core.local_size

    @property
    def ghost_size(self) -> int:
        """Returns the number of ghost elements."""
        return self._core.ghost_size
    
    @property
    def full_size(self) -> int:
        """Returns the total local size (owned + ghost)."""
        return self._core.full_size

    @property
    def ndim(self) -> int:
        return 1

    @property
    def shape(self):
        if self._global_size is not None:
            return (self._global_size,)
        return (self.full_size,)

    @property
    def size(self) -> int:
        s = self.shape[0]
        return s if s is not None else 0

    @property
    def T(self) -> 'DistVector':
        return self

    def copy(self) -> 'DistVector':
        obj = self.duplicate()
        obj.comm = self.comm
        return obj

    def __matmul__(self, other):
        if isinstance(other, DistVector):
            return self.dot(other)
        return NotImplemented

    def __len__(self) -> int:
        return self.size

    def __repr__(self):
        s = self.shape[0]
        size_str = str(s) if s is not None else "Unknown"
        return f"<DistVector of size {size_str}, dtype={self.dtype}>"

    def to_numpy(self) -> np.ndarray:
        """
        Convert the locally owned part of the vector to a NumPy array.
        
        Returns:
            np.ndarray: A copy (or view) of the locally owned data.
        """
        buf = np.array(self._core, copy=False)
        return buf[:self.local_size]

    def from_numpy(self, arr: np.ndarray) -> None:
        """
        Update the locally owned part of the vector from a NumPy array.
        
        Args:
            arr (np.ndarray): Input array. Must match local_size.
            
        Raises:
            ValueError: If array size does not match local_size.
        """
        if arr.size != self.local_size:
            raise ValueError(f"Array size {arr.size} does not match vector local size {self.local_size}")
        buf = np.array(self._core, copy=False)
        buf[:self.local_size] = arr

    def __array__(self) -> np.ndarray:
        """Support for np.array(vec)."""
        return self.to_numpy()

    def sync_ghosts(self) -> None:
        """Synchronize ghost elements from their owners."""
        self._core.sync_ghosts()

    def reduce_ghosts(self) -> None:
        """Reduce (sum) ghost elements back to their owners."""
        self._core.reduce_ghosts()

    def duplicate(self) -> 'DistVector':
        """
        Create a deep copy of the vector.
        
        Returns:
            DistVector: A new vector with the same structure and data.
        """
        return DistVector(self._core.duplicate(), self.comm)

    def set_constant(self, val: Union[float, complex, int]) -> None:
        """
        Set all local elements to a constant value.
        
        Args:
            val: The value to set.
        """
        self._core.set_constant(val)

    def scale(self, alpha: Union[float, complex, int]) -> None:
        """
        Scale the vector by a scalar.
        
        Args:
            alpha: The scaling factor.
        """
        self._core.scale(alpha)

    def axpy(self, alpha: Union[float, complex, int], x: 'DistVector') -> None:
        """
        Compute y = alpha * x + y (in-place).
        
        Args:
            alpha: Scalar multiplier.
            x (DistVector): The vector to add.
            
        Raises:
            TypeError: If x is not a DistVector.
        """
        if isinstance(x, DistVector):
            self._core.axpy(alpha, x._core)
        else:
            raise TypeError("axpy expects DistVector")

    def axpby(self, alpha: Union[float, complex, int], x: 'DistVector', beta: Union[float, complex, int]) -> None:
        """
        Compute y = alpha * x + beta * y (in-place).
        
        Args:
            alpha: Scalar multiplier for x.
            x (DistVector): The vector to add.
            beta: Scalar multiplier for y (this vector).
            
        Raises:
            TypeError: If x is not a DistVector.
        """
        if isinstance(x, DistVector):
            self._core.axpby(alpha, x._core, beta)
        else:
            raise TypeError("axpby expects DistVector")

    def pointwise_mult(self, other: 'DistVector') -> None:
        """
        Element-wise multiplication: y[i] *= other[i].
        
        Args:
            other (DistVector): The vector to multiply with.
            
        Raises:
            TypeError: If other is not a DistVector.
        """
        if isinstance(other, DistVector):
            self._core.pointwise_mult(other._core)
        else:
            raise TypeError("pointwise_mult expects DistVector")
            
    def dot(self, other: 'DistVector') -> Union[float, complex]:
        """
        Compute the global dot product: sum(conj(self[i]) * other[i]).
        
        Args:
            other (DistVector): The other vector.
            
        Returns:
            The dot product (scalar).
            
        Raises:
            TypeError: If other is not a DistVector.
        """
        if isinstance(other, DistVector):
            return self._core.dot(other._core)
        else:
            raise TypeError("Dot product requires DistVector")

    # Operators
    def __add__(self, other: Union['DistVector', float, complex, int, np.ndarray]) -> 'DistVector':
        res = self.duplicate()
        res += other
        return res

    def __iadd__(self, other: Union['DistVector', float, complex, int, np.ndarray]) -> 'DistVector':
        if isinstance(other, DistVector):
            self._core.axpy(1.0, other._core)
        elif np.isscalar(other):
            buf = np.array(self._core, copy=False)
            buf[:self.local_size] += other
        elif isinstance(other, np.ndarray):
            buf = np.array(self._core, copy=False)
            buf[:self.local_size] += other
        else:
            return NotImplemented
        return self

    def __sub__(self, other: Union['DistVector', float, complex, int, np.ndarray]) -> 'DistVector':
        res = self.duplicate()
        res -= other
        return res

    def __isub__(self, other: Union['DistVector', float, complex, int, np.ndarray]) -> 'DistVector':
        if isinstance(other, DistVector):
            self._core.axpy(-1.0, other._core)
        elif np.isscalar(other):
            buf = np.array(self._core, copy=False)
            buf[:self.local_size] -= other
        elif isinstance(other, np.ndarray):
            buf = np.array(self._core, copy=False)
            buf[:self.local_size] -= other
        else:
            return NotImplemented
        return self

    def __mul__(self, other: Union['DistVector', float, complex, int, np.ndarray]) -> 'DistVector':
        res = self.duplicate()
        res *= other
        return res

    def __imul__(self, other: Union['DistVector', float, complex, int, np.ndarray]) -> 'DistVector':
        if np.isscalar(other):
            self._core.scale(other)
        elif isinstance(other, DistVector):
            self._core.pointwise_mult(other._core)
        elif isinstance(other, np.ndarray):
            buf = np.array(self._core, copy=False)
            buf[:self.local_size] *= other
        else:
            return NotImplemented
        return self

    def __rmul__(self, other: Union[float, complex, int]) -> 'DistVector':
        return self.__mul__(other)

# VBCSR API Reference

This document provides detailed documentation for the classes and methods in the `vbcsr` library.

## Table of Contents
1. [VBCSR (Matrix)](#vbcsr-matrix)
2. [DistVector](#distvector)
3. [DistMultiVector](#distmultivector)

---

## VBCSR (Matrix)

The `VBCSR` class represents a distributed block-sparse matrix. It inherits from `scipy.sparse.linalg.LinearOperator`.

### Properties

- **`shape`**: `Tuple[int, int]` - Global shape of the matrix `(rows, cols)`.
- **`ndim`**: `int` - Number of dimensions (always 2).
- **`nnz`**: `int` - Global number of non-zero elements.
- **`dtype`**: `np.dtype` - Data type of the matrix elements (`np.float64` or `np.complex128`).
- **`T`**: `VBCSR` - Transpose of the matrix (returns a new object).
- **`real`**: `VBCSR` - Real part of the matrix (returns a new object).
- **`imag`**: `VBCSR` - Imaginary part of the matrix (returns a new object).

### Factory Methods

#### `create_serial`
```python
@classmethod
create_serial(cls, comm: Any, global_blocks: int, block_sizes: List[int], adjacency: List[List[int]], dtype: type = np.float64) -> 'VBCSR'
```
Creates a matrix using serial graph construction. Rank 0 defines the entire structure, which is then distributed.
- **comm**: MPI communicator (or `None` for serial).
- **global_blocks**: Total number of blocks.
- **block_sizes**: List of block sizes.
- **adjacency**: Adjacency list (list of lists of neighbors).

#### `create_distributed`
```python
@classmethod
create_distributed(cls, comm: Any, owned_indices: List[int], block_sizes: List[int], adjacency: List[List[int]], dtype: type = np.float64) -> 'VBCSR'
```
Creates a matrix using distributed graph construction. Each rank defines only its owned blocks.
- **owned_indices**: Global indices of blocks owned by this rank.

#### `create_random`
```python
@classmethod
create_random(cls, comm: Any, global_blocks: int, block_size_min: int, block_size_max: int, density: float = 0.01, dtype: type = np.float64, seed: int = 42) -> 'VBCSR'
```
Creates a random connected matrix for benchmarking purposes.

#### `from_scipy`
```python
@classmethod
from_scipy(cls, spmat: Any, comm=None) -> 'VBCSR'
```
Creates a VBCSR matrix from a SciPy sparse matrix.
- **spmat**: SciPy sparse matrix (assumed to be on Rank 0).

### Methods

#### `add_block`
```python
add_block(self, g_row: int, g_col: int, data: np.ndarray, mode: AssemblyMode = AssemblyMode.ADD) -> None
```
Adds or inserts a dense block into the matrix.

#### `assemble`
```python
assemble(self) -> None
```
Finalizes matrix assembly. Must be called after adding blocks.

#### `transpose`
```python
transpose(self) -> 'VBCSR'
```
Returns a new transposed matrix.

#### `transpose_`
```python
transpose_(self) -> None
```
In-place transpose. More memory efficient.

#### `conj` / `conjugate`
```python
conj(self) -> 'VBCSR'
```
Returns a new conjugated matrix.

#### `conj_`
```python
conj_(self) -> None
```
In-place conjugate.

#### `copy` / `duplicate`
```python
copy(self) -> 'VBCSR'
```
Returns a deep copy of the matrix.

#### `dot` / `@`
```python
dot(self, other: Union['VBCSR', DistVector, DistMultiVector, np.ndarray]) -> Union['VBCSR', DistVector, DistMultiVector]
```
Performs matrix multiplication.
- If `other` is `VBCSR`: Sparse Matrix-Matrix Multiplication (SpMM).
- If `other` is `DistVector`/`DistMultiVector`: Matrix-Vector Multiplication (SpMV).

#### `spmm`
```python
spmm(self, B: 'VBCSR', threshold: float = 0.0, transA: bool = False, transB: bool = False) -> 'VBCSR'
```
Sparse Matrix-Matrix Multiplication with filtering options.
- **threshold**: Drop blocks with Frobenius norm less than threshold.

#### `spmm_self`
```python
spmm_self(self, threshold: float = 0.0, transA: bool = False) -> 'VBCSR'
```
Computes $A \times A$ (or $A^T \times A$) efficiently.

#### `add`
```python
add(self, B: 'VBCSR', alpha: float = 1.0, beta: float = 1.0) -> 'VBCSR'
```
Computes $C = \alpha A + \beta B$.

#### `scale`
```python
scale(self, alpha: Union[float, complex, int]) -> None
```
Scales the matrix in-place by `alpha`.

#### `shift`
```python
shift(self, alpha: Union[float, complex, int]) -> None
```
Adds `alpha` to diagonal elements.

#### `add_diagonal`
```python
add_diagonal(self, v: Union[DistVector, np.ndarray]) -> None
```
Adds vector `v` to diagonal elements ($A_{ii} += v_i$).

#### `extract_submatrix`
```python
extract_submatrix(self, global_indices: List[int]) -> 'VBCSR'
```
Extracts a submatrix corresponding to given global block indices.

#### `insert_submatrix`
```python
insert_submatrix(self, submat: 'VBCSR', global_indices: List[int]) -> None
```
Inserts a submatrix back into the matrix.

#### `to_dense`
```python
to_dense(self) -> np.ndarray
```
Converts locally owned part to dense NumPy array.

#### `from_dense`
```python
from_dense(self, data: np.ndarray) -> None
```
Updates locally owned part from dense NumPy array.

#### `to_scipy`
```python
to_scipy(self, format: Optional[str] = None) -> Any
```
Converts locally owned part to SciPy sparse matrix (`bsr` or `csr`).

#### `create_vector`
```python
create_vector(self) -> DistVector
```
Creates a compatible `DistVector`.

#### `create_multivector`
```python
create_multivector(self, k: int) -> DistMultiVector
```
Creates a compatible `DistMultiVector` with `k` columns.

#### `__getitem__`
```python
__getitem__(self, key: Tuple[int, int]) -> Scalar
```
Access individual elements (e.g., `A[0, 0]`). Note: Slow, for debugging.

### Operators
- `+`, `-`: Matrix addition/subtraction.
- `*`: Scalar multiplication.
- `@`: Matrix multiplication.
- `+=`, `-=`, `*=`: In-place operators.
- `-A`: Negation.

---

## DistVector

Represents a distributed 1D vector.

### Properties
- **`shape`**: `Tuple[int]` - Global shape `(size,)`.
- **`ndim`**: `int` - Number of dimensions (1).
- **`size`**: `int` - Global size.
- **`local_size`**: `int` - Number of locally owned elements.
- **`ghost_size`**: `int` - Number of ghost elements.
- **`full_size`**: `int` - Total local size (owned + ghost).
- **`T`**: `DistVector` - Returns self.

### Methods

#### `copy` / `duplicate`
```python
copy(self) -> 'DistVector'
```
Returns a deep copy.

#### `to_numpy`
```python
to_numpy(self) -> np.ndarray
```
Returns locally owned part as NumPy array.

#### `from_numpy`
```python
from_numpy(self, arr: np.ndarray) -> None
```
Updates locally owned part from NumPy array.

#### `set_constant`
```python
set_constant(self, val: Union[float, complex, int]) -> None
```
Sets all local elements to `val`.

#### `scale`
```python
scale(self, alpha: Union[float, complex, int]) -> None
```
Scales vector in-place.

#### `axpy`
```python
axpy(self, alpha: Scalar, x: 'DistVector') -> None
```
Computes $y = \alpha x + y$ in-place.

#### `axpby`
```python
axpby(self, alpha: Scalar, x: 'DistVector', beta: Scalar) -> None
```
Computes $y = \alpha x + \beta y$ in-place.

#### `pointwise_mult`
```python
pointwise_mult(self, other: 'DistVector') -> None
```
Element-wise multiplication $y_i *= x_i$.

#### `dot`
```python
dot(self, other: 'DistVector') -> Scalar
```
Computes global dot product $\sum \bar{x}_i y_i$.

#### `sync_ghosts`
```python
sync_ghosts(self) -> None
```
Synchronizes ghost elements from their owners.

#### `reduce_ghosts`
```python
reduce_ghosts(self) -> None
```
Reduces (sums) ghost elements back to their owners.

### Operators
- `+`, `-`: Vector addition/subtraction (supports scalar broadcast).
- `*`: Element-wise multiplication (supports scalar).
- `@`: Dot product.
- `+=`, `-=`, `*=`: In-place operators.

---

## DistMultiVector

Represents a distributed collection of vectors (2D, column-major).

### Properties
- **`shape`**: `Tuple[int, int]` - Global shape `(rows, cols)`.
- **`ndim`**: `int` - Number of dimensions (2).
- **`size`**: `int` - Total elements.
- **`local_rows`**: `int` - Number of locally owned rows.
- **`num_vectors`**: `int` - Number of columns.

### Methods

#### `copy` / `duplicate`
```python
copy(self) -> 'DistMultiVector'
```
Returns a deep copy.

#### `to_numpy`
```python
to_numpy(self) -> np.ndarray
```
Returns locally owned part as 2D NumPy array.

#### `from_numpy`
```python
from_numpy(self, arr: np.ndarray) -> None
```
Updates locally owned part from 2D NumPy array.

#### `set_constant`
```python
set_constant(self, val: Union[float, complex, int]) -> None
```
Sets all elements to `val`.

#### `scale`
```python
scale(self, alpha: Union[float, complex, int]) -> None
```
Scales all elements in-place.

#### `axpy`
```python
axpy(self, alpha: Scalar, x: 'DistMultiVector') -> None
```
Computes $Y = \alpha X + Y$ in-place.

#### `axpby`
```python
axpby(self, alpha: Scalar, x: 'DistMultiVector', beta: Scalar) -> None
```
Computes $Y = \alpha X + \beta Y$ in-place.

#### `pointwise_mult`
```python
pointwise_mult(self, other: Union['DistMultiVector', DistVector]) -> None
```
Element-wise multiplication. Supports broadcasting if `other` is a `DistVector`.

#### `bdot`
```python
bdot(self, other: 'DistMultiVector') -> list
```
Computes batch dot products (one for each column).

#### `sync_ghosts`
```python
sync_ghosts(self) -> None
```
Synchronizes ghost elements.

### Operators
- `+`, `-`: Addition/subtraction.
- `*`: Element-wise multiplication.
- `+=`, `-=`, `*=`: In-place operators.

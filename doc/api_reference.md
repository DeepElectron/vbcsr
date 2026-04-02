# VBCSR API Reference

This document provides detailed documentation for the classes and methods in the `vbcsr` library.

## Table of Contents
1. [VBCSR (Matrix)](#vbcsr-matrix)
2. [DistVector](#distvector)
3. [DistMultiVector](#distmultivector)
4. [DistGraph](#distgraph)
5. [AtomicData](#atomicdata)
6. [ImageContainer](#imagecontainer)

---

## VBCSR (Matrix)

The `VBCSR` class represents a distributed block-sparse matrix. It inherits from `scipy.sparse.linalg.LinearOperator`.

### Properties

- **`shape`**: `Tuple[int, int]` - Global shape of the matrix `(rows, cols)`.
- **`ndim`**: `int` - Number of dimensions (always 2).
- **`nnz`**: `int` - Global number of non-zero elements.
- **`dtype`**: `np.dtype` - Data type of the matrix elements (`np.float64` or `np.complex128`).
- **`matrix_kind`**: `str` - Active backend family (`"csr"`, `"bsr"`, or `"vbcsr"`).
- **`T`**: `VBCSR` - Transpose of the matrix (returns a new object).
- **`real`**: `VBCSR` - Real part of the matrix (returns a new object).
- **`imag`**: `VBCSR` - Imaginary part of the matrix (returns a new object).

### Factory Methods

#### `create_serial`
```python
@classmethod
create_serial(cls, global_blocks: int, block_sizes: List[int], adjacency: List[List[int]], dtype: type = np.float64, comm: Any = None) -> 'VBCSR'
```
Creates a matrix using serial graph construction. Rank 0 defines the entire structure, which is then distributed.
- **comm**: MPI communicator (or `None` for serial).
- **global_blocks**: Total number of blocks.
- **block_sizes**: List of block sizes.
- **adjacency**: Adjacency list (list of lists of neighbors).

#### `create_distributed`
```python
@classmethod
create_distributed(cls, owned_indices: List[int], block_sizes: List[int], adjacency: List[List[int]], dtype: type = np.float64, comm: Any = None) -> 'VBCSR'
```
Creates a matrix using distributed graph construction. Each rank defines only its owned blocks.
- **owned_indices**: Global indices of blocks owned by this rank.

#### `create_random`
```python
@classmethod
create_random(cls, global_blocks: int, block_size_min: int, block_size_max: int, density: float = 0.01, dtype: type = np.float64, seed: int = 42, comm: Any = None) -> 'VBCSR'
```
Creates a random connected matrix for benchmarking purposes.

#### `from_scipy`
```python
@classmethod
from_scipy(cls, spmat: Any, comm=None, root: int = 0) -> 'VBCSR'
```
Creates a VBCSR matrix from a SciPy sparse matrix.
- **spmat**: SciPy sparse matrix. In MPI, provide it on `root` and pass `None` on other ranks.
- **comm**: Optional communicator. Distributed imports must pass it explicitly.
- **root**: Rank that owns the SciPy input for collective MPI imports.

### Methods

#### `add_block`
```python
add_block(self, g_row: int, g_col: int, data: np.ndarray, mode: AssemblyMode = AssemblyMode.ADD) -> None
```
Adds or inserts a dense block into the matrix.

#### `get_block`
```python
get_block(self, g_row: int, g_col: int) -> Optional[np.ndarray]
```
Returns a dense copy of one local block in row-major layout, or `None` if that block is not present locally.

#### `assemble`
```python
assemble(self) -> None
```
Finalizes matrix assembly. Must be called after adding blocks.

#### `mult`
```python
mult(self, x: Union[DistVector, DistMultiVector, np.ndarray], y=None) -> Union[DistVector, DistMultiVector]
```
Applies the matrix to a distributed vector, multivector, or compatible local NumPy array.

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
Returns a new conjugated matrix. `conjugate()` is the primary name; `conj()` is a compatibility alias.

#### `conj_`
```python
conj_(self) -> None
```
In-place conjugate.

#### `copy` / `duplicate`
```python
duplicate(self, independent_graph: bool = True) -> 'VBCSR'
```
Returns a deep copy of the matrix values. `copy()` is a compatibility alias for `duplicate()`.
- **independent_graph**: If `True`, duplicate the graph object too. If `False`, reuse the same immutable graph structure.

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

#### `filter_blocks`
```python
filter_blocks(self, threshold: float = 0.0) -> None
```
Drops locally stored blocks with Frobenius norm below `threshold` while preserving the current backend family.

#### `get_block_density`
```python
get_block_density(self) -> float
```
Returns the global block-density estimate `nnz_blocks / n_blocks^2`.

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
Extracts a submatrix corresponding to given global block indices. For distributed matrices this operation is collective.

#### `insert_submatrix`
```python
insert_submatrix(self, submat: 'VBCSR', global_indices: List[int]) -> None
```
Inserts a submatrix back into the matrix. For distributed matrices this operation is collective.

#### `spmf`
```python
spmf(self, func_name: str, method: str = "lanczos", verbose: bool = False) -> 'VBCSR'
```
Applies a supported sparse matrix function approximation and returns a matrix in the same backend family.

#### `to_dense`
```python
to_dense(self) -> np.ndarray
```
Converts the local matrix view to a dense NumPy array. This does not gather remote ranks.

#### `from_dense`
```python
from_dense(self, data: np.ndarray) -> None
```
Updates the local matrix view from a dense NumPy array.

#### `to_scipy`
```python
to_scipy(self, format: Optional[str] = None) -> Any
```
Converts the local matrix view to a SciPy sparse matrix (`bsr` or `csr`). This is a local export helper, not a distributed gather.

#### `row_ptr`
```python
row_ptr: np.ndarray
```
Returns the local logical block-row pointer array.

#### `col_ind`
```python
col_ind: np.ndarray
```
Returns the local logical block-column index array.

#### `get_values`
```python
get_values(self) -> np.ndarray
```
Returns the packed local block values in row-major order, matching `row_ptr` / `col_ind`.

#### `save_matrix_market`
```python
save_matrix_market(self, filename: Union[str, bytes]) -> None
```
Writes a Matrix Market file in serial execution. Distributed matrices raise an error.

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
Scalar and slicing indexing are currently unsupported. Use `get_block(...)` for block-level inspection.

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
duplicate(self) -> 'DistVector'
```
Returns a deep copy. `copy()` is a compatibility alias for `duplicate()`.

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

#### `set_random_normal`
```python
set_random_normal(self, normalize: bool = False) -> None
```
Fills the locally owned entries with random normal values.

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
- **`ghost_rows`**: `int` - Number of ghost rows cached locally.
- **`num_vectors`**: `int` - Number of columns.

### Methods

#### `copy` / `duplicate`
```python
duplicate(self) -> 'DistMultiVector'
```
Returns a deep copy. `copy()` is a compatibility alias for `duplicate()`.

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

#### `set_random_normal`
```python
set_random_normal(self, normalize: bool = False) -> None
```
Fills the locally owned rows with random normal values.

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

#### `reduce_ghosts`
```python
reduce_ghosts(self) -> None
```
Reduces accumulated ghost-row contributions back to their owners.

### Operators
- `+`, `-`: Addition/subtraction.
- `*`: Element-wise multiplication.
- `+=`, `-=`, `*=`: In-place operators.

---

## DistGraph

Low-level distributed graph object exposed by `vbcsr_core`. This is the canonical ownership and ghost-layout description used by matrices, vectors, and atomic/image helpers.

### Methods

#### `construct_serial`
```python
construct_serial(self, global_blocks: int, block_sizes: list[int], adjacency: list[list[int]]) -> None
```
Collectively builds the distributed graph from root-owned serial input. Root rank is fixed at rank 0.

#### `construct_distributed`
```python
construct_distributed(self, owned_indices: list[int], block_sizes: list[int], adjacency: list[list[int]]) -> None
```
Builds the graph from per-rank owned rows.

### Properties
- **`owned_global_indices`**: global block ids owned on this rank.
- **`ghost_global_indices`**: ghost block ids imported onto this rank.
- **`block_sizes`**: local block sizes for owned and ghost blocks.
- **`owned_scalar_rows`**: locally owned scalar row count.
- **`local_scalar_cols`**: local scalar column count including ghosts.
- **`global_scalar_rows`**: global scalar row count.
- **`rank`** / **`size`**: communicator rank and size.

#### `get_local_index`
```python
get_local_index(self, gid: int) -> int
```
Returns the local block index for a global block id, or `-1` if that block is not present locally.

---

## AtomicData

Distributed atomic topology and metadata used by `ImageContainer` and atom-centered workflows.

`atom_types` remains the internal type-id surface. `atomic_numbers` / `z` expose true atomic numbers.

### Constructors

#### `from_points`
```python
@classmethod
from_points(cls, pos, z, cell, pbc, r_max, type_norb=1, comm=None) -> 'AtomicData'
```
Builds atomic connectivity from positions and atomic numbers.
- `r_max` and `type_norb` may be scalars, vectors aligned to sorted unique atomic numbers, or dictionaries keyed by atomic number / chemical symbol.
- If `comm` is provided, rank 0 may hold the full input and other ranks may provide empty arrays.

#### `from_distributed`
```python
@classmethod
from_distributed(cls, n_atom, N_atom, atom_offset, n_edge, N_edge, atom_index, atom_type, edge_index, type_norb, edge_shift, cell, pos, atomic_numbers=None, comm=None) -> 'AtomicData'
```
Builds `AtomicData` from prepartitioned distributed graph data.
- `atom_types` stay as type ids in this path.
- Pass `atomic_numbers=` to preserve authoritative atomic-number metadata.
- If `atomic_numbers` is omitted, accessing `atomic_numbers`, `z`, or `to_ase()` raises instead of guessing from type ids.

#### `from_ase` / `from_file`
```python
@classmethod
from_ase(cls, atoms, r_max, type_norb=1, comm=None) -> 'AtomicData'

@classmethod
from_file(cls, filename, r_max, type_norb=1, comm=None, format=None) -> 'AtomicData'
```
Convenience constructors layered on top of `from_points`.

### Properties
- **`positions` / `pos`**: owned atomic positions as `(n_atom, 3)` NumPy arrays.
- **`atom_indices` / `indices`**: original atom ids.
- **`atom_types`**: internal type ids used for orbital lookups.
- **`atomic_numbers` / `z`**: true atomic numbers for owned atoms.
- **`cell`**: `(3, 3)` unit-cell matrix.
- **`pbc`**: periodic-boundary flags.
- **`edge_index`**: local edge list using local/ghost atom indices.
- **`edge_shift`**: lattice shifts for the local edge list.
- **`graph`**: backing `DistGraph`.

### Methods

#### `norb`
```python
norb(self) -> int
```
Returns the global orbital count implied by `atom_types` and `type_norb`.

#### `to_ase`
```python
to_ase(self)
```
Exports owned atoms to an ASE `Atoms` object. This requires explicit atomic-number metadata.

---

## ImageContainer

Collects real-space image blocks keyed by lattice shifts and samples them into k-space matrices.

### Constructor

#### `__init__`
```python
ImageContainer(atomic_data: AtomicData, dtype=np.float64)
```
Creates a real or complex image container over the supplied `AtomicData`.

### Methods

#### `add_block`
```python
add_block(self, g_row, g_col, data, R=None, mode="add") -> None
```
Adds one real-space image block.

#### `add_blocks`
```python
add_blocks(self, g_rows, g_cols, data_list, R_list=None, mode="add") -> None
```
Adds many image blocks in one call.

#### `assemble`
```python
assemble(self) -> None
```
Finalizes distributed image assembly.

#### `sample_k`
```python
sample_k(self, k_point, convention="R", symm=False) -> VBCSR
```
Samples the assembled image blocks at a single k-point and returns a complex `VBCSR`.
- `convention="R"` uses only lattice-vector phases.
- `convention="R+tau"` includes the basis-position phase term.

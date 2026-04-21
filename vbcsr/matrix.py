import numpy as np
from scipy.sparse.linalg import LinearOperator
import vbcsr_core
from vbcsr_core import AssemblyMode
from typing import Union, Optional, List, Any, Tuple
from .vector import DistVector
from .multivector import DistMultiVector


def _owned_block_count(graph: Any) -> int:
    return len(graph.owned_global_indices)


def _owned_scalar_rows(graph: Any) -> int:
    if hasattr(graph, "owned_scalar_rows"):
        return int(graph.owned_scalar_rows)
    owned_blocks = _owned_block_count(graph)
    return int(sum(graph.block_sizes[:owned_blocks]))


def _global_scalar_rows(graph: Any, comm: Any) -> int:
    if hasattr(graph, "global_scalar_rows"):
        return int(graph.global_scalar_rows)

    local_rows = _owned_scalar_rows(graph)
    if comm is not None and hasattr(comm, "allreduce"):
        try:
            return int(comm.allreduce(local_rows))
        except Exception:
            pass
    return local_rows


class VBCSR(LinearOperator):
    """
    Variable Block Compressed Sparse Row (VBCSR) Matrix.
    
    This class wraps the C++ BlockSpMat and provides a SciPy-compatible LinearOperator interface.
    It supports distributed matrix operations using MPI.
    """
    
    def __init__(self, graph: Any, dtype: type = np.float64, comm: Any = None):
        """
        Initialize VBCSR matrix from a DistGraph.
        
        Args:
            graph: The underlying C++ DistGraph object.
            dtype: Data type (np.float64 or np.complex128).
        """
        self.graph = graph
        self.dtype = np.dtype(dtype)
        self.comm = comm
        self._global_nnz = None
        self.shape = self._infer_square_shape(graph, comm)
        
        if self.dtype == np.dtype(np.float64):
            self._core = vbcsr_core.BlockSpMat_Double(graph)
        else:
            self._core = vbcsr_core.BlockSpMat_Complex(graph)

    @staticmethod
    def _infer_square_shape(graph: Any, comm: Any) -> Tuple[int, int]:
        total_rows = _global_scalar_rows(graph, comm)
        return (total_rows, total_rows)

    @classmethod
    def _wrap_core(
        cls,
        core: Any,
        dtype: type,
        comm: Any,
        shape: Optional[Tuple[int, int]] = None,
        global_nnz: Optional[int] = None,
    ) -> 'VBCSR':
        obj = cls.__new__(cls)
        obj.graph = core.graph
        obj.dtype = np.dtype(dtype)
        obj.comm = comm
        obj._core = core
        obj.shape = shape if shape is not None else cls._infer_square_shape(core.graph, comm)
        obj._global_nnz = global_nnz
        return obj

    def _invalidate_nnz(self) -> None:
        self._global_nnz = None

    @property
    def ndim(self) -> int:
        return 2

    @property
    def nnz(self) -> int:
        """Global number of non-zero elements."""
        if self._global_nnz is not None:
            return int(self._global_nnz)

        if hasattr(self._core, "global_nnz"):
            self._global_nnz = int(self._core.global_nnz)
            return self._global_nnz

        local_nnz = int(self._core.local_nnz)
        if self.comm is not None and hasattr(self.comm, "allreduce"):
            try:
                self._global_nnz = int(self.comm.allreduce(local_nnz))
                return self._global_nnz
            except Exception:
                pass

        self._global_nnz = local_nnz
        return self._global_nnz

    @property
    def matrix_kind(self) -> str:
        return self._core.matrix_kind

    @property
    def local_nnz(self) -> int:
        return int(self._core.local_nnz)

    @property
    def local_block_nnz(self) -> int:
        return int(self._core.local_block_nnz)

    @property
    def configured_page_size(self) -> int:
        return int(self._core.configured_page_size)

    @property
    def page_size(self) -> int:
        return int(self._core.page_size)

    @page_size.setter
    def page_size(self, value: int) -> None:
        self.set_page_size(value)

    def set_page_size(self, value: int) -> None:
        self._core.set_page_size(int(value))

    @property
    def shape_class_count(self) -> int:
        return int(self._core.shape_class_count)

    @property
    def has_contiguous_layout(self) -> bool:
        return bool(self._core.has_contiguous_layout)

    def pack_contiguous(self) -> None:
        self._core.pack_contiguous()

    @property
    def T(self) -> 'VBCSR':
        return self.transpose()

    def transpose(self) -> 'VBCSR':
        core_T = self._core.transpose()
        shape = None
        if self.shape[0] is not None and self.shape[1] is not None:
            shape = (self.shape[1], self.shape[0])
        return self._wrap_core(core_T, self.dtype, self.comm, shape=shape, global_nnz=self._global_nnz)

    def transpose_(self) -> None:
        """In-place transpose."""
        core_T = self._core.transpose()
        self._core = core_T
        self.graph = core_T.graph
        if self.shape[0] is not None and self.shape[1] is not None:
            self.shape = (self.shape[1], self.shape[0])
        else:
            self.shape = self._infer_square_shape(self.graph, self.comm)

    def conj_(self) -> None:
        """In-place conjugate."""
        self._core.conjugate()

    def conj(self) -> 'VBCSR':
        """Compatibility alias for conjugate()."""
        obj = self.duplicate()
        obj.conj_()
        return obj

    def conjugate(self) -> 'VBCSR':
        """Primary out-of-place conjugation API. `conj()` is a compatibility alias."""
        return self.conj()

    @property
    def real(self) -> 'VBCSR':
        """Return the real part of the matrix."""
        core_real = self._core.real()
        return self._wrap_core(core_real, np.float64, self.comm, shape=self.shape, global_nnz=self._global_nnz)

    @property
    def imag(self) -> 'VBCSR':
        """Return the imaginary part of the matrix."""
        core_imag = self._core.imag()
        return self._wrap_core(core_imag, np.float64, self.comm, shape=self.shape, global_nnz=self._global_nnz)

    def __neg__(self) -> 'VBCSR':
        return self * -1

    def __pos__(self) -> 'VBCSR':
        return self

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            if isinstance(row, int) and isinstance(col, int):
                raise NotImplementedError("Scalar indexing is not supported; use get_block(...) for block access.")
        raise NotImplementedError("Slicing is not supported on VBCSR matrices.")

    def dot(self, other: Union['VBCSR', DistVector, DistMultiVector, np.ndarray]) -> Union['VBCSR', DistVector, DistMultiVector]:
        return self @ other

    def copy(self) -> 'VBCSR':
        """Compatibility alias for duplicate()."""
        return self.duplicate()

    def __len__(self) -> int:
        if self.shape[0] is not None:
            return self.shape[0]
        return 0

    def __repr__(self):
        shape_str = f"{self.shape[0]}x{self.shape[1]}" if self.shape[0] is not None else "Unknown shape"
        return f"<{shape_str} VBCSR matrix of type {self.dtype} with {self.nnz} stored elements>"

    @classmethod
    def create_serial(cls, global_blocks: int, block_sizes: List[int], adjacency: List[List[int]], dtype: type = np.float64, comm: Any = None) -> 'VBCSR':
        """
        Create a VBCSR matrix using serial graph construction (Rank 0 distributes).
        
        Args:
            global_blocks (int): Total number of blocks.
            block_sizes (List[int]): Size of each block.
            adjacency (List[List[int]]): Adjacency list (list of neighbors for each block).
            dtype: Data type.
            comm: MPI communicator (mpi4py or integer handle).
            
        Returns:
            VBCSR: The initialized matrix.
        """
        graph = vbcsr_core.DistGraph(comm)
        graph.construct_serial(global_blocks, block_sizes, adjacency)
        return cls(graph, dtype, comm)

    @classmethod
    def create_distributed(cls, owned_indices: List[int], block_sizes: List[int], adjacency: List[List[int]], dtype: type = np.float64, comm: Any = None) -> 'VBCSR':
        """
        Create a VBCSR matrix using distributed graph construction.
        
        Args:
            owned_indices (List[int]): Global indices of blocks owned by this rank.
            block_sizes (List[int]): Sizes of owned blocks.
            adjacency (List[List[int]]): Adjacency list for owned blocks.
            dtype: Data type.
            comm: MPI communicator.
            
        Returns:
            VBCSR: The initialized matrix.
        """
        graph = vbcsr_core.DistGraph(comm)
        graph.construct_distributed(owned_indices, block_sizes, adjacency)
        return cls(graph, dtype, comm)

    @classmethod
    def create_random(cls, global_blocks: int = 100, block_size_min: int = 1, block_size_max: int = 4, density: float = 0.01, dtype: type = np.float64, seed: int = 42, comm: Any = None) -> 'VBCSR':
        """
        Create a random connected VBCSR matrix for benchmarking.
        
        Args:
            global_blocks (int): Total number of blocks.
            block_size_min (int): Minimum block size.
            block_size_max (int): Maximum block size.
            density (float): Sparsity density (approximate fraction of non-zero blocks).
            dtype: Data type.
            seed (int): Random seed.
            comm: MPI communicator.
            
        Returns:
            VBCSR: The initialized matrix with random structure and data.
        """
        dtype = np.dtype(dtype)

        if comm is None:
            rank = 0
            size = 1
        else:
            rank = comm.Get_rank()
            size = comm.Get_size()
        
        np.random.seed(seed)
        
        # 1. Generate block sizes (replicated on all ranks for simplicity in this helper)
        # In a real large-scale scenario, we would generate distributedly, but for this helper
        # we assume we can hold the structure metadata in memory.
        all_block_sizes = np.random.randint(block_size_min, block_size_max + 1, size=global_blocks).tolist()
        
        # 2. Partition blocks among ranks
        # Simple linear partition
        blocks_per_rank = global_blocks // size
        remainder = global_blocks % size
        
        start_block = rank * blocks_per_rank + min(rank, remainder)
        my_count = blocks_per_rank + (1 if rank < remainder else 0)
        end_block = start_block + my_count
        
        owned_indices = list(range(start_block, end_block))
        my_block_sizes = all_block_sizes[start_block:end_block]
        
        # 3. Generate Adjacency
        # Ensure connectivity: Ring topology + random edges
        # We generate adjacency for OWNED blocks.
        
        my_adj = []
        for i in owned_indices:
            neighbors = set()
            # Ring connections (ensure connectivity)
            neighbors.add((i - 1) % global_blocks)
            neighbors.add((i + 1) % global_blocks)
            neighbors.add(i) # Self-loop
            
            # Random edges
            # Number of extra edges based on density
            # density is fraction of TOTAL blocks.
            # n_random = int(global_blocks * density)
            # This might be too dense. Let's interpret density as prob of edge.
            # Or just fixed average degree.
            # Let's use density as probability.
            
            # Generating random edges efficiently:
            # We want approx global_blocks * density edges per row.
            n_random = max(0, int(global_blocks * density) - 2) # Subtract mandatory ones
            if n_random > 0:
                random_neighbors = np.random.choice(global_blocks, size=n_random, replace=False)
                neighbors.update(random_neighbors)
            
            my_adj.append(sorted(list(neighbors)))
            
        # 4. Create Matrix
        mat = cls.create_distributed(owned_indices, my_block_sizes, my_adj, dtype, comm)
        
        # 5. Fill with random data
        # We iterate over owned blocks and their neighbors
        for local_i, global_i in enumerate(owned_indices):
            r_dim = my_block_sizes[local_i]
            neighbors = my_adj[local_i]
            
            for global_j in neighbors:
                c_dim = all_block_sizes[global_j]
                
                # Generate random block
                if dtype == np.dtype(np.float64):
                    data = np.random.rand(r_dim, c_dim)
                else:
                    data = np.random.rand(r_dim, c_dim) + 1j * np.random.rand(r_dim, c_dim)
                
                mat.add_block(global_i, global_j, data)
                
        mat.assemble()
        return mat

    def create_vector(self) -> DistVector:
        """Create a DistVector compatible with this matrix."""
        if self.dtype == np.dtype(np.float64):
            core_vec = vbcsr_core.DistVector_Double(self.graph)
        else:
            core_vec = vbcsr_core.DistVector_Complex(self.graph)
        return DistVector(core_vec, self.comm)

    def create_multivector(self, k: int) -> DistMultiVector:
        """
        Create a DistMultiVector compatible with this matrix.
        
        Args:
            k (int): Number of vectors (columns).
        """
        if self.dtype == np.dtype(np.float64):
            core_vec = vbcsr_core.DistMultiVector_Double(self.graph, k)
        else:
            core_vec = vbcsr_core.DistMultiVector_Complex(self.graph, k)
        return DistMultiVector(core_vec, self.comm)

    def add_block(self, g_row: int, g_col: int, data: np.ndarray, mode: AssemblyMode = AssemblyMode.ADD) -> None:
        """
        Add or insert a block into the matrix.
        
        Args:
            g_row (int): Global row block index.
            g_col (int): Global column block index.
            data (np.ndarray): Block data (2D array).
            mode (AssemblyMode): INSERT or ADD.
        """
        self._invalidate_nnz()
        self._core.add_block(g_row, g_col, data, mode)

    def get_block(self, g_row: int, g_col: int) -> Optional[np.ndarray]:
        """
        Get a block from the matrix using global indices.
        
        Args:
            g_row (int): Global row block index.
            g_col (int): Global column block index.
            
        Returns:
            np.ndarray: The block data, or None if the block is not owned by this rank.
        """
        l_row = self.graph.get_local_index(g_row)
        if l_row == -1 or l_row >= len(self.graph.owned_global_indices):
            return None
            
        l_col = self.graph.get_local_index(g_col)
        if l_col == -1:
            return None
            
        # Call core get_block with local indices
        data = self._core.get_block(l_row, l_col)
        if data.size == 0:
            return None
        return data

    def assemble(self) -> None:
        """Finalize matrix assembly (exchange remote blocks)."""
        self._core.assemble()
        self._invalidate_nnz()
        _ = self.nnz

    def mult(self, x: Union[DistVector, DistMultiVector, np.ndarray], y: Optional[Union[DistVector, DistMultiVector]] = None) -> Union[DistVector, DistMultiVector]:
        """
        Perform matrix multiplication: y = A * x.
        
        Args:
            x: Input vector (DistVector), multivector (DistMultiVector), or numpy array.
               If numpy array, it is assumed to be the local part of the vector/multivector.
            y: Output vector or multivector (optional).
            
        Returns:
            The result y.
        """
        # Auto-convert numpy array
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                v = self.create_vector()
                v.from_numpy(x)
                x = v
            elif x.ndim == 2:
                k = x.shape[1]
                mv = self.create_multivector(k)
                mv.from_numpy(x)
                x = mv
            else:
                raise ValueError("Input numpy array must be 1D or 2D")

        if isinstance(x, DistVector):
            if y is None:
                y = self.create_vector()
            self._core.mult(x._core, y._core)
            return y
        elif isinstance(x, DistMultiVector):
            if y is None:
                y = self.create_multivector(x.num_vectors)
            self._core.mult_dense(x._core, y._core)
            return y
        else:
            raise TypeError("mult expects DistVector, DistMultiVector, or numpy.ndarray")

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """
        Matrix-vector multiplication for SciPy LinearOperator.
        
        Args:
            x (np.ndarray): Input array (local part).
            
        Returns:
            np.ndarray: Result array (local part).
        """
        # Reuse mult which now handles numpy
        res = self.mult(x)
        return res.to_numpy()

    def _matmat(self, X: np.ndarray) -> np.ndarray:
        """
        Matrix-matrix multiplication for SciPy LinearOperator.
        
        Args:
            X (np.ndarray): Input array (local part, shape (N, K)).
            
        Returns:
            np.ndarray: Result array.
        """
        # Reuse mult which now handles numpy
        res = self.mult(X)
        return res.to_numpy()

    def scale(self, alpha: Union[float, complex, int]) -> None:
        """Scale the matrix by a scalar."""
        self._core.scale(alpha)

    def fill(self, value: Union[float, complex, int]) -> None:
        """Fill all currently stored blocks with a scalar value."""
        self._core.fill(value)

    def copy_from(self, other: 'VBCSR') -> None:
        """Copy values from another matrix with the same logical structure."""
        if not isinstance(other, VBCSR):
            raise TypeError("copy_from expects a VBCSR matrix")
        self._core.copy_from(other._core)
        self._global_nnz = other._global_nnz

    def axpby(self, alpha: Union[float, complex, int], x: 'VBCSR', beta: Union[float, complex, int]) -> None:
        """Compute self = alpha * x + beta * self in place."""
        if not isinstance(x, VBCSR):
            raise TypeError("axpby expects a VBCSR matrix")
        self._core.axpby(alpha, x._core, beta)
        self._invalidate_nnz()

    def shift(self, alpha: Union[float, complex, int]) -> None:
        """Add a scalar to the diagonal elements."""
        self._core.shift(alpha)

    def add_diagonal(self, v: Union[DistVector, np.ndarray]) -> None:
        """
        Add a vector to the diagonal elements: A_ii += v_i.
        
        Args:
            v (DistVector or np.ndarray): The vector to add.
        """
        if isinstance(v, np.ndarray):
            dv = self.create_vector()
            dv.from_numpy(v)
            v = dv
            
        if isinstance(v, DistVector):
            self._core.add_diagonal(v._core)
        else:
            raise TypeError("add_diagonal expects DistVector or numpy.ndarray")

    def duplicate(self, independent_graph: bool = True) -> 'VBCSR':
        """Primary copy API. `copy()` is a compatibility alias."""
        return self._wrap_core(
            self._core.duplicate(independent_graph),
            self.dtype,
            self.comm,
            shape=self.shape,
            global_nnz=self._global_nnz,
        )

    # Operators
    def __add__(self, other: 'VBCSR') -> 'VBCSR':
        if isinstance(other, VBCSR):
            res = self.duplicate()
            res += other
            return res
        return NotImplemented

    def __sub__(self, other: 'VBCSR') -> 'VBCSR':
        if isinstance(other, VBCSR):
            res = self.duplicate()
            res -= other
            return res
        return NotImplemented
    
    def __isub__(self, other: 'VBCSR') -> 'VBCSR':
        if isinstance(other, VBCSR):
            self._core.axpy(-1.0, other._core)
            self._invalidate_nnz()
            return self
        return NotImplemented

    def __iadd__(self, other: 'VBCSR') -> 'VBCSR':
        if isinstance(other, VBCSR):
            self._core.axpy(1.0, other._core)
            self._invalidate_nnz()
            return self
        return NotImplemented

    def __mul__(self, other: Union[float, complex, int]) -> 'VBCSR':
        if np.isscalar(other):
            res = self.duplicate()
            res.scale(other)
            return res
        return NotImplemented

    def __imul__(self, other: Union[float, complex, int]) -> 'VBCSR':
        if np.isscalar(other):
            self.scale(other)
            return self
        return NotImplemented

    def __rmul__(self, other: Union[float, complex, int]) -> 'VBCSR':
        return self.__mul__(other)

    def __truediv__(self, other: Union[float, complex, int]) -> 'VBCSR':
        if np.isscalar(other):
            return self.__mul__(1.0 / other)
        return NotImplemented

    def __itruediv__(self, other: Union[float, complex, int]) -> 'VBCSR':
        if np.isscalar(other):
            self.scale(1.0 / other)
            return self
        return NotImplemented

    def __matmul__(self, other: Union['VBCSR', DistVector, DistMultiVector, np.ndarray]) -> Union['VBCSR', DistVector, DistMultiVector]:
        """
        Support for the @ operator.
        
        If other is VBCSR, performs SpMM (Sparse Matrix-Matrix Multiplication).
        If other is Vector/MultiVector/ndarray, performs Matrix-Vector Multiplication.
        """
        if isinstance(other, VBCSR):
            return self.spmm(other)
        elif isinstance(other, (DistVector, DistMultiVector, np.ndarray)):
            return self.mult(other)
        else:
            return NotImplemented


    def spmm(self, B: 'VBCSR', threshold: float = 0.0, transA: bool = False, transB: bool = False) -> 'VBCSR':
        """
        Sparse Matrix-Matrix Multiplication: C = op(A) * op(B).
        
        Args:
            B (VBCSR): The matrix to multiply with.
            threshold (float): Threshold for dropping small blocks.
            transA (bool): If True, use A^H.
            transB (bool): If True, use B^H.
            
        Returns:
            VBCSR: The result matrix C.
        """
        if not isinstance(B, VBCSR):
            raise TypeError("B must be a VBCSR matrix")
        if self.dtype != B.dtype:
            raise TypeError("A and B must have the same dtype")
            
        core_C = self._core.spmm(B._core, threshold, transA, transB)
        shape = None
        if self.shape[0] is not None and B.shape[1] is not None:
            shape = (self.shape[0], B.shape[1])
        return self._wrap_core(core_C, self.dtype, self.comm, shape=shape)

    def spmm_self(self, threshold: float = 0.0, transA: bool = False) -> 'VBCSR':
        core_C = self._core.spmm_self(threshold, transA)
        return self._wrap_core(core_C, self.dtype, self.comm, shape=self.shape)

    def get_block_density(self) -> float:
        return self._core.get_block_density()

    def filter_blocks(self, threshold: float = 0.0):
        self._core.filter_blocks(threshold)
        self._invalidate_nnz()

    def add(self, B: 'VBCSR', alpha: float = 1.0, beta: float = 1.0) -> 'VBCSR':
        if not isinstance(B, VBCSR):
            raise TypeError("B must be a VBCSR matrix")
        core_C = self._core.add(B._core, alpha, beta)
        return self._wrap_core(core_C, self.dtype, self.comm, shape=self.shape)

    def extract_submatrix(self, global_indices: List[int]) -> 'VBCSR':
        """
        Extract a submatrix corresponding to the given global indices.
        
        Args:
            global_indices (List[int]): List of global vertex indices to extract.
            
        Returns:
            VBCSR: A serial VBCSR matrix containing the submatrix.
        """
        core_sub = self._core.extract_submatrix(global_indices)
        return self._wrap_core(core_sub, self.dtype, None)

    def insert_submatrix(self, submat: 'VBCSR', global_indices: List[int]) -> None:
        """
        Insert a submatrix back into the distributed matrix.
        
        Args:
            submat (VBCSR): The submatrix to insert.
            global_indices (List[int]): The global indices corresponding to the submatrix rows/cols.
        """
        if not isinstance(submat, VBCSR):
            raise TypeError("submat must be a VBCSR matrix")
        
        self._core.insert_submatrix(submat._core, global_indices)
        self._invalidate_nnz()

    def spmf(self, func_name: str, method: str = "lanczos", verbose: bool = False) -> 'VBCSR':
        """
        Compute a matrix function using the graph-based method.
        
        Args:
            func_name (str): Name of the function to compute ("inv", "sqrt", "isqrt", "exp").
            method (str): Method to use ("lanczos", "dense").
            verbose (bool): Whether to print verbose output.
            
        Returns:
            VBCSR: The computed matrix function.
        """
        core_res = self._core.spmf(func_name, method, verbose)
        return self._wrap_core(core_res, self.dtype, self.comm, shape=self.shape)

    def to_dense(self) -> np.ndarray:
        """
        Convert the local portion of the matrix to a dense numpy array.
        
        Returns:
            np.ndarray: 2D array of shape (owned_rows, all_local_cols).
        """
        return self._core.to_dense()

    def from_dense(self, data: np.ndarray) -> None:
        """
        Update the local portion of the matrix from a dense numpy array.
        
        Args:
            data (np.ndarray): 2D array of shape (owned_rows, all_local_cols).
        """
        self._core.from_dense(data)
        self._invalidate_nnz()

    @classmethod
    def from_scipy(cls, spmat: Any, comm: Any = None, root: int = 0) -> 'VBCSR':
        """
        Create a VBCSR matrix from a SciPy sparse matrix.
        
        Args:
            spmat: SciPy sparse matrix (bsr_matrix, csr_matrix, etc.).
                   In MPI mode, provide it on `root` and pass `None` on other ranks.
            comm: Optional MPI communicator. If omitted, distributed usage is rejected.
            root: Root rank that owns the SciPy input when `comm` is distributed.
                   
        Returns:
            VBCSR: The initialized matrix.
        """
        import scipy.sparse as sp

        rank = 0
        size = 1
        if comm is not None:
            rank = comm.Get_rank()
            size = comm.Get_size()
        else:
            world_graph = vbcsr_core.DistGraph(None)
            if world_graph.size > 1:
                raise ValueError(
                    "from_scipy is ambiguous in MPI without an explicit communicator. "
                    "Pass `comm` and call collectively with the SciPy matrix on the root rank only."
                )

        if size > 1:
            if rank == root and spmat is None:
                raise ValueError("from_scipy requires a SciPy matrix on the root rank.")
            if rank != root and spmat is not None:
                raise ValueError(
                    "from_scipy expects `spmat=None` on non-root MPI ranks to avoid ambiguous replicated input."
                )
        elif spmat is None:
            raise ValueError("from_scipy requires a SciPy sparse matrix in serial mode.")

        root_dtype = None
        if rank == root and spmat is not None:
            if spmat.shape[0] != spmat.shape[1]:
                raise ValueError("VBCSR.from_scipy requires a square sparse matrix.")
            root_dtype = str(np.dtype(spmat.dtype))

        if comm is not None and size > 1:
            dtype = np.dtype(comm.bcast(root_dtype, root=root))
        else:
            dtype = np.dtype(root_dtype)

        if rank == root and spmat is not None:
            if sp.isspmatrix_bsr(spmat):
                R, C = spmat.blocksize
                if R != C:
                    raise ValueError("VBCSR requires square blocks (R == C) for BSR input.")
                if spmat.shape[0] % R != 0:
                    raise ValueError("BSR input dimensions must be divisible by the block size.")

                n_blocks = spmat.shape[0] // R
                block_sizes = [R] * n_blocks
                adj = []
                for i in range(n_blocks):
                    start = spmat.indptr[i]
                    end = spmat.indptr[i + 1]
                    adj.append(spmat.indices[start:end].tolist())
            else:
                spmat = spmat.tocsr()
                n_blocks = spmat.shape[0]
                block_sizes = [1] * n_blocks
                adj = []
                for i in range(n_blocks):
                    start = spmat.indptr[i]
                    end = spmat.indptr[i + 1]
                    adj.append(spmat.indices[start:end].tolist())
        else:
            n_blocks = 0
            block_sizes = []
            adj = []

        mat = cls.create_serial(n_blocks, block_sizes, adj, dtype=dtype, comm=comm)

        if rank == root and spmat is not None:
            if sp.isspmatrix_bsr(spmat):
                for i in range(n_blocks):
                    start = spmat.indptr[i]
                    end = spmat.indptr[i + 1]
                    for k in range(start, end):
                        j = spmat.indices[k]
                        mat.add_block(i, j, spmat.data[k])
            else:
                spmat_csr = spmat.tocsr()
                for i in range(n_blocks):
                    start = spmat_csr.indptr[i]
                    end = spmat_csr.indptr[i + 1]
                    for k in range(start, end):
                        j = spmat_csr.indices[k]
                        val = spmat_csr.data[k]
                        mat.add_block(i, j, np.array([[val]], dtype=dtype))

        mat.assemble()
        return mat

    @property
    def row_ptr(self) -> np.ndarray:
        return np.asarray(self._core.row_ptr, dtype=np.int32)

    @property
    def col_ind(self) -> np.ndarray:
        return np.asarray(self._core.col_ind, dtype=np.int32)

    def get_values(self) -> np.ndarray:
        return np.asarray(self._core.get_values(), dtype=self.dtype)

    def save_matrix_market(self, filename: Union[str, bytes]) -> None:
        self._core.save_matrix_market(str(filename))

    def to_scipy(self, format: Optional[str] = None) -> Any:
        """
        Convert the LOCAL portion of the VBCSR matrix to a SciPy sparse matrix.
        
        Args:
            format: 'bsr', 'csr', or None (default).
                    If None, automatically chooses 'bsr' if blocks are uniform, else 'csr'.
                    
        Returns:
            scipy.sparse.spmatrix: The local matrix.
        """
        import scipy.sparse as sp
        # 1. Get Packed Values
        # Layout: RowMajor for easy numpy/scipy compatibility
        values = np.asarray(self._core.get_values(), dtype=self.dtype) # 1D array
        
        # 2. Get Structure
        row_ptr = self.row_ptr
        col_ind = self.col_ind
        
        # 3. Check Uniformity
        block_sizes = self.graph.block_sizes
        matrix_kind = self.matrix_kind

        # Check uniformity
        is_uniform = False
        uniform_size = 0
        if len(block_sizes) > 0:
            first_size = block_sizes[0]
            if all(s == first_size for s in block_sizes):
                is_uniform = True
                uniform_size = first_size

        target_format = format
        if target_format is None:
            target_format = 'bsr' if matrix_kind in ('csr', 'bsr') else 'csr'
            
        if target_format == 'bsr':
            if not is_uniform:
                raise ValueError("Cannot convert non-uniform VBCSR to BSR format.")
            
            R = uniform_size
            C = uniform_size
            
            # Reshape values to (nnz_blocks, R, C)
            # values is flat.
            # Total size = nnz_blocks * R * C
            nnz_blocks = len(col_ind)
            if len(values) != nnz_blocks * R * C:
                raise RuntimeError(f"Data size mismatch: expected {nnz_blocks*R*C}, got {len(values)}")
            
            data = values.reshape((nnz_blocks, R, C))
            
            n_block_rows = len(row_ptr) - 1
            local_rows = n_block_rows * R
            
            total_local_cols = sum(block_sizes)
            local_shape = (local_rows, total_local_cols)
            
            return sp.bsr_matrix((data, col_ind, row_ptr), shape=local_shape)
            
        elif target_format == 'csr':
            # Expand to Scalar CSR
            
            # 1. Calculate scalar row pointers
            n_block_rows = len(row_ptr) - 1
            
            # Pre-calculate offsets for scalar rows
            scalar_row_offsets = np.zeros(n_block_rows + 1, dtype=np.int32)
            # block_sizes is list-like, convert to numpy for efficiency if needed, but it's fine.
            # We need block sizes for owned rows.
            # block_sizes contains ALL local blocks.
            # We assume row i corresponds to block i?
            # Yes, VBCSR graph assumes nodes 0..N.
            
            for i in range(n_block_rows):
                scalar_row_offsets[i+1] = scalar_row_offsets[i] + block_sizes[i]
                
            total_scalar_rows = scalar_row_offsets[-1]
            
            # 2. Pre-calculate scalar column offsets
            scalar_col_offsets = np.zeros(len(block_sizes) + 1, dtype=np.int32)
            np.cumsum(block_sizes, out=scalar_col_offsets[1:])
            
            # 3. Prepare CSR arrays
            total_nnz = len(values)
            scalar_indptr = np.zeros(total_scalar_rows + 1, dtype=np.int32)
            scalar_indices = np.zeros(total_nnz, dtype=np.int32)
            scalar_data_out = np.zeros(total_nnz, dtype=self.dtype)
            
            current_nnz = 0
            blk_value_offset = 0
            
            for i in range(n_block_rows):
                R_i = block_sizes[i]
                start_blk = row_ptr[i]
                end_blk = row_ptr[i+1]
                
                # Cache block info for this row
                row_blocks = []
                for k in range(start_blk, end_blk):
                    j = col_ind[k]
                    C_j = block_sizes[j]
                    row_blocks.append((j, C_j, blk_value_offset))
                    blk_value_offset += R_i * C_j
                
                for r in range(R_i):
                    scalar_row_idx = scalar_row_offsets[i] + r
                    scalar_indptr[scalar_row_idx] = current_nnz
                    
                    for (j, C_j, blk_start) in row_blocks:
                        # Data copy
                        src_start = blk_start + r * C_j
                        src_end = src_start + C_j
                        dst_end = current_nnz + C_j
                        
                        scalar_data_out[current_nnz:dst_end] = values[src_start:src_end]
                        
                        # Indices
                        col_start = scalar_col_offsets[j]
                        # Manual loop for indices
                        for c in range(C_j):
                            scalar_indices[current_nnz + c] = col_start + c
                            
                        current_nnz += C_j
                        
            scalar_indptr[-1] = current_nnz
            
            # Shape
            local_rows = len(scalar_indptr) - 1
            total_local_cols = sum(block_sizes)
            local_shape = (local_rows, total_local_cols)
            
            return sp.csr_matrix((scalar_data_out, scalar_indices, scalar_indptr), shape=local_shape)
            
        else:
            raise ValueError(f"Unknown format: {target_format}")

import numpy as np
from scipy.sparse.linalg import LinearOperator
import vbcsr_core
from vbcsr_core import AssemblyMode
from typing import Union, Optional, List, Any, Tuple
from .vector import DistVector
from .multivector import DistMultiVector

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
        
        # Calculate global size
        local_size = len(graph.owned_global_indices)
        # We need to sum block sizes to get matrix dimension (number of orbitals)
        # block_sizes contains sizes for owned blocks
        # Note: This assumes graph.block_sizes is accessible and correct
        # For distributed graph, block_sizes might contain ghosts too, but owned_global_indices
        # maps to the first N entries.
        
        # Compute shape placeholder. Ideally this should be set by factory methods.
        self.shape = (None, None) 
        
        if dtype == np.float64:
            self._core = vbcsr_core.BlockSpMat_Double(graph)
        else:
            self._core = vbcsr_core.BlockSpMat_Complex(graph)

    @property
    def ndim(self) -> int:
        return 2

    @property
    def nnz(self) -> int:
        """Global number of non-zero elements."""
        if self._global_nnz is not None:
            return self._global_nnz
        return self._core.local_nnz

    @property
    def T(self) -> 'VBCSR':
        return self.transpose()

    def transpose(self) -> 'VBCSR':
        core_T = self._core.transpose()
        obj = VBCSR(core_T.graph, self.dtype, self.comm)
        obj._core = core_T
        if self.shape[0] is not None and self.shape[1] is not None:
            obj.shape = (self.shape[1], self.shape[0])
        return obj

    def transpose_(self) -> None:
        """In-place transpose."""
        core_T = self._core.transpose()
        self._core = core_T
        self.graph = core_T.graph
        if self.shape[0] is not None and self.shape[1] is not None:
            self.shape = (self.shape[1], self.shape[0])

    def conj_(self) -> None:
        """In-place conjugate."""
        self._core.conjugate()

    def conj(self) -> 'VBCSR':
        obj = self.duplicate()
        obj.conj_()
        return obj

    def conjugate(self) -> 'VBCSR':
        return self.conj()

    @property
    def real(self) -> 'VBCSR':
        """Return the real part of the matrix."""
        core_real = self._core.real()
        obj = VBCSR(core_real.graph, np.float64, self.comm)
        obj._core = core_real
        obj.shape = self.shape
        return obj

    @property
    def imag(self) -> 'VBCSR':
        """Return the imaginary part of the matrix."""
        core_imag = self._core.imag()
        obj = VBCSR(core_imag.graph, np.float64, self.comm)
        obj._core = core_imag
        obj.shape = self.shape
        return obj

    def __neg__(self) -> 'VBCSR':
        return self * -1

    def __sub__(self, other: 'VBCSR') -> 'VBCSR':
        if isinstance(other, VBCSR):
            return self.add(other, alpha=1.0, beta=-1.0)
        return NotImplemented

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            if isinstance(row, int) and isinstance(col, int):
                # Scalar access
                # This is slow and requires finding the block.
                # We need to map global row/col to block row/col and offset.
                # This is hard without full metadata on Python side.
                # For now, let's raise NotImplementedError or try to implement if possible.
                # Actually, get_values returns all values.
                # We can use extract_submatrix for a single element?
                # extract_submatrix takes global indices of blocks? No, vertices.
                # Let's check extract_submatrix signature.
                # "Extract a submatrix corresponding to the given global indices."
                # It returns a VBCSR.
                # If we extract row i, we get row i.
                # But we want scalar.
                raise NotImplementedError("Scalar indexing not yet efficiently implemented.")
        raise NotImplementedError("Slicing not implemented.")

    def dot(self, other: Union['VBCSR', DistVector, DistMultiVector, np.ndarray]) -> Union['VBCSR', DistVector, DistMultiVector]:
        return self @ other

    def copy(self) -> 'VBCSR':
        return self.duplicate()

    def __len__(self) -> int:
        if self.shape[0] is not None:
            return self.shape[0]
        return 0

    def __repr__(self):
        shape_str = f"{self.shape[0]}x{self.shape[1]}" if self.shape[0] is not None else "Unknown shape"
        return f"<{shape_str} VBCSR matrix of type {self.dtype} with {self.nnz} stored elements>"

    @classmethod
    def create_serial(cls, comm: Any, global_blocks: int, block_sizes: List[int], adjacency: List[List[int]], dtype: type = np.float64) -> 'VBCSR':
        """
        Create a VBCSR matrix using serial graph construction (Rank 0 distributes).
        
        Args:
            comm: MPI communicator (mpi4py or integer handle).
            global_blocks (int): Total number of blocks.
            block_sizes (List[int]): Size of each block.
            adjacency (List[List[int]]): Adjacency list (list of neighbors for each block).
            dtype: Data type.
            
        Returns:
            VBCSR: The initialized matrix.
        """
        graph = vbcsr_core.DistGraph(comm)
        graph.construct_serial(global_blocks, block_sizes, adjacency)
        
        obj = cls(graph, dtype, comm)
        
        # Compute shape for serial (rank 0 knows all)
        total_rows = sum(block_sizes)
        obj.shape = (total_rows, total_rows)
        return obj

    @classmethod
    def create_distributed(cls, comm: Any, owned_indices: List[int], block_sizes: List[int], adjacency: List[List[int]], dtype: type = np.float64) -> 'VBCSR':
        """
        Create a VBCSR matrix using distributed graph construction.
        
        Args:
            comm: MPI communicator.
            owned_indices (List[int]): Global indices of blocks owned by this rank.
            block_sizes (List[int]): Sizes of owned blocks.
            adjacency (List[List[int]]): Adjacency list for owned blocks.
            dtype: Data type.
            
        Returns:
            VBCSR: The initialized matrix.
        """
        graph = vbcsr_core.DistGraph(comm)
        graph.construct_distributed(owned_indices, block_sizes, adjacency)
        
        obj = cls(graph, dtype, comm)
        
        # Compute shape using MPI allreduce if possible
        if comm is not None and hasattr(comm, "allreduce"):
            local_rows = sum(block_sizes)
            total_rows = comm.allreduce(local_rows)
            obj.shape = (total_rows, total_rows)
        elif comm is None:
            total_rows = sum(block_sizes)
            obj.shape = (total_rows, total_rows)
        
        return obj

    @classmethod
    def create_random(cls, comm: Any, global_blocks: int, block_size_min: int, block_size_max: int, density: float = 0.01, dtype: type = np.float64, seed: int = 42) -> 'VBCSR':
        """
        Create a random connected VBCSR matrix for benchmarking.
        
        Args:
            comm: MPI communicator.
            global_blocks (int): Total number of blocks.
            block_size_min (int): Minimum block size.
            block_size_max (int): Maximum block size.
            density (float): Sparsity density (approximate fraction of non-zero blocks).
            dtype: Data type.
            seed (int): Random seed.
            
        Returns:
            VBCSR: The initialized matrix with random structure and data.
        """
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
        mat = cls.create_distributed(comm, owned_indices, my_block_sizes, my_adj, dtype)
        
        # 5. Fill with random data
        # We iterate over owned blocks and their neighbors
        for local_i, global_i in enumerate(owned_indices):
            r_dim = my_block_sizes[local_i]
            neighbors = my_adj[local_i]
            
            for global_j in neighbors:
                c_dim = all_block_sizes[global_j]
                
                # Generate random block
                if dtype == np.float64:
                    data = np.random.rand(r_dim, c_dim)
                else:
                    data = np.random.rand(r_dim, c_dim) + 1j * np.random.rand(r_dim, c_dim)
                
                mat.add_block(global_i, global_j, data)
                
        mat.assemble()
        return mat

    def create_vector(self) -> DistVector:
        """Create a DistVector compatible with this matrix."""
        if self.dtype == np.float64:
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
        if self.dtype == np.float64:
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
        self._core.add_block(g_row, g_col, data, mode)

    def assemble(self) -> None:
        """Finalize matrix assembly (exchange remote blocks)."""
        self._core.assemble()
        local_nnz = self._core.local_nnz
        if self.comm:
            try:
                self._global_nnz = self.comm.allreduce(local_nnz)
            except:
                self._global_nnz = local_nnz
        else:
            self._global_nnz = local_nnz

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

    def duplicate(self) -> 'VBCSR':
        """Create a deep copy of the matrix."""
        new_obj = VBCSR(self.graph, self.dtype, self.comm)
        new_obj._core = self._core.duplicate(False) # Share graph
        new_obj.shape = self.shape
        new_obj._global_nnz = self._global_nnz
        return new_obj

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
            return self
        return NotImplemented

    def __iadd__(self, other: 'VBCSR') -> 'VBCSR':
        if isinstance(other, VBCSR):
            self._core.axpy(1.0, other._core)
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
        
        # Wrap result
        obj = VBCSR.__new__(VBCSR)
        obj.graph = core_C.graph
        obj.dtype = self.dtype
        obj._core = core_C
        obj.comm = self.comm
        obj.shape = (None, None)
        obj._global_nnz = None
        return obj

    def spmm_self(self, threshold: float = 0.0, transA: bool = False) -> 'VBCSR':
        core_C = self._core.spmm_self(threshold, transA)
        obj = VBCSR.__new__(VBCSR)
        obj.graph = core_C.graph
        obj.dtype = self.dtype
        obj._core = core_C
        obj.comm = self.comm
        obj.shape = (None, None)
        obj._global_nnz = None
        return obj

    def add(self, B: 'VBCSR', alpha: float = 1.0, beta: float = 1.0) -> 'VBCSR':
        if not isinstance(B, VBCSR):
            raise TypeError("B must be a VBCSR matrix")
        core_C = self._core.add(B._core, alpha, beta)
        obj = VBCSR.__new__(VBCSR)
        obj.graph = core_C.graph
        obj.dtype = self.dtype
        obj._core = core_C
        obj.comm = self.comm
        obj.shape = self.shape
        obj._global_nnz = None
        return obj

    def extract_submatrix(self, global_indices: List[int]) -> 'VBCSR':
        """
        Extract a submatrix corresponding to the given global indices.
        
        Args:
            global_indices (List[int]): List of global vertex indices to extract.
            
        Returns:
            VBCSR: A serial VBCSR matrix containing the submatrix.
        """
        core_sub = self._core.extract_submatrix(global_indices)
        
        obj = VBCSR.__new__(VBCSR)
        obj.graph = core_sub.graph
        obj.dtype = self.dtype
        obj._core = core_sub
        obj.comm = None # Extracted submatrix is serial/local
        # Shape is (M, M) where M = len(global_indices)
        # But let's let the core handle it or compute it.
        # Since it's serial, we can compute it if needed, but for now (None, None) is safe or we can query graph.
        # Actually, for serial submatrix, we might want to know the shape.
        # The core binding doesn't expose block sizes easily unless we query graph.
        # Let's leave as None for now or improve later.
        obj.shape = (None, None)
        obj._global_nnz = None
        return obj

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

    @classmethod
    def from_scipy(cls, spmat: Any, comm=None) -> 'VBCSR':
        """
        Create a VBCSR matrix from a SciPy sparse matrix.
        
        Args:
            spmat: SciPy sparse matrix (bsr_matrix, csr_matrix, etc.).
                   Assumed to be on Rank 0 (or all ranks).
                   
        Returns:
            VBCSR: The initialized matrix.
        """
        import scipy.sparse as sp
        
        rank = comm.Get_rank() if comm else 0
        
        # Ensure spmat is available (at least on rank 0)
        # If passed as None on other ranks, we handle it.
        
        # Convert to BSR or CSR
        # If it's BSR, we use its block structure.
        # If not, we convert to CSR and treat as 1x1 blocks.
        
        global_blocks = 0
        block_sizes = []
        adj = []
        
        # Data for filling
        # We need to broadcast structure if only rank 0 has it.
        # For simplicity, we assume spmat is provided on Rank 0 and we use create_serial logic
        # which expects arguments on all ranks (or at least rank 0 to distribute).
        # But create_serial implementation in python currently expects arguments on all ranks?
        # Let's check create_serial:
        # "Create a VBCSR matrix using serial graph construction (Rank 0 distributes)."
        # It calls graph.construct_serial.
        # graph.construct_serial implementation in C++:
        # "If rank 0 has data, scatter." -> It expects data on Rank 0.
        
        if rank == 0:
            if sp.isspmatrix_bsr(spmat):
                # BSR Matrix
                R, C = spmat.blocksize
                if R != C:
                    raise ValueError("VBCSR requires square blocks (R == C) for BSR input.")
                
                n_blocks = spmat.shape[0] // R
                block_sizes = [R] * n_blocks
                
                # Adjacency
                # spmat.indptr, spmat.indices
                adj = []
                for i in range(n_blocks):
                    start = spmat.indptr[i]
                    end = spmat.indptr[i+1]
                    adj.append(spmat.indices[start:end].tolist())
                    
            else:
                # Treat as CSR (1x1 blocks)
                spmat_csr = spmat.tocsr()
                n_blocks = spmat_csr.shape[0]
                block_sizes = [1] * n_blocks
                
                adj = []
                for i in range(n_blocks):
                    start = spmat_csr.indptr[i]
                    end = spmat_csr.indptr[i+1]
                    adj.append(spmat_csr.indices[start:end].tolist())
        
        # Create Matrix (Distributes structure)
        mat = cls.create_serial(comm, n_blocks, block_sizes, adj, spmat.dtype)
        
        # Fill Data
        # We iterate over blocks on Rank 0 and add them.
        # Since create_serial distributes ownership, Rank 0 might not own everything.
        # But add_block handles remote owners (if implemented with MPI).
        # However, sending every block individually is slow.
        # Ideally we should scatter data.
        # But for "adapter", correctness first.
        
        if rank == 0:
            if sp.isspmatrix_bsr(spmat):
                R, C = spmat.blocksize
                for i in range(n_blocks):
                    start = spmat.indptr[i]
                    end = spmat.indptr[i+1]
                    for k in range(start, end):
                        j = spmat.indices[k]
                        data_blk = spmat.data[k] # Shape (R, C)
                        mat.add_block(i, j, data_blk)
            else:
                spmat_csr = spmat.tocsr()
                for i in range(n_blocks):
                    start = spmat_csr.indptr[i]
                    end = spmat_csr.indptr[i+1]
                    for k in range(start, end):
                        j = spmat_csr.indices[k]
                        val = spmat_csr.data[k]
                        # 1x1 block
                        mat.add_block(i, j, np.array([[val]], dtype=spmat.dtype))
                        
        mat.assemble()
        return mat

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
        values = self._core.get_values() # 1D array
        
        # 2. Get Structure
        row_ptr = self._core.row_ptr
        col_ind = self._core.col_ind
        
        # 3. Check Uniformity
        # We need block sizes.
        block_sizes = self.graph.block_sizes
        
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
            target_format = 'bsr' if is_uniform else 'csr'
            
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



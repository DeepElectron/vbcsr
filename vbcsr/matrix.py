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
    
    def __init__(self, graph: Any, dtype: type = np.float64):
        """
        Initialize VBCSR matrix from a DistGraph.
        
        Args:
            graph: The underlying C++ DistGraph object.
            dtype: Data type (np.float64 or np.complex128).
        """
        self.graph = graph
        self.dtype = np.dtype(dtype)
        
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
        
        obj = cls(graph, dtype)
        
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
        
        obj = cls(graph, dtype)
        
        # Compute shape using MPI allreduce if possible
        if hasattr(comm, "allreduce"):
            local_rows = sum(block_sizes)
            total_rows = comm.allreduce(local_rows)
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
        return DistVector(core_vec)

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
        return DistMultiVector(core_vec)

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
        new_obj = VBCSR(self.graph, self.dtype)
        new_obj._core = self._core.duplicate(False) # Share graph
        new_obj.shape = self.shape
        return new_obj

    # Operators
    def __add__(self, other: 'VBCSR') -> 'VBCSR':
        if isinstance(other, VBCSR):
            res = self.duplicate()
            res += other
            return res
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
        obj.shape = (None, None)
        return obj

    def spmm_self(self, threshold: float = 0.0, transA: bool = False) -> 'VBCSR':
        core_C = self._core.spmm_self(threshold, transA)
        obj = VBCSR.__new__(VBCSR)
        obj.graph = core_C.graph
        obj.dtype = self.dtype
        obj._core = core_C
        obj.shape = (None, None)
        return obj

    def add(self, B: 'VBCSR', alpha: float = 1.0, beta: float = 1.0) -> 'VBCSR':
        if not isinstance(B, VBCSR):
            raise TypeError("B must be a VBCSR matrix")
        core_C = self._core.add(B._core, alpha, beta)
        obj = VBCSR.__new__(VBCSR)
        obj.graph = core_C.graph
        obj.dtype = self.dtype
        obj._core = core_C
        obj.shape = self.shape
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
        # Shape is (M, M) where M = len(global_indices)
        # But let's let the core handle it or compute it.
        # Since it's serial, we can compute it if needed, but for now (None, None) is safe or we can query graph.
        # Actually, for serial submatrix, we might want to know the shape.
        # The core binding doesn't expose block sizes easily unless we query graph.
        # Let's leave as None for now or improve later.
        obj.shape = (None, None)
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

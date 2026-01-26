# VBCSR User Guide

Welcome to the VBCSR User Guide. This document provides a comprehensive overview of the `vbcsr` library, covering installation, core concepts, and detailed usage examples.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Creating Matrices](#creating-matrices)
5. [Matrix Operations](#matrix-operations)
6. [Vector Operations](#vector-operations)
7. [Python & NumPy Interoperability](#python--numpy-interoperability)
8. [SciPy Integration](#scipy-integration)
9. [Distributed Computing with MPI](#distributed-computing-with-mpi)

---

## Introduction

**VBCSR** (Variable Block Compressed Sparse Row) is a high-performance sparse matrix library designed for scientific computing. It bridges the gap between the ease of use of Python and the raw performance of C++.

### Key Features
- **Block-Sparse Structure**: Optimized for matrices with block structures (common in physics and engineering).
- **Distributed Memory**: Built on MPI to scale across multiple nodes.
- **Hardware Acceleration**: Uses AVX/AVX2 instructions and OpenMP threading.
- **Pythonic Interface**: Supports standard Python operators (`+`, `*`, `+=`, `@`) and NumPy arrays.
- **Flexible Deployment**: Works in both distributed (MPI) and serial (no-MPI) environments.

---

## Installation

### Basic Installation
```bash
pip install .
```

### Advanced Installation
For performance tuning (linking against MKL/OpenBLAS) or troubleshooting OpenMP on macOS, please refer to the [Advanced Installation Guide](advanced_installation.md).

---

## Core Concepts

### The VBCSR Matrix
The core object is the `VBCSR` matrix. Unlike standard CSR matrices that store individual non-zero elements, VBCSR stores **dense blocks** of non-zeros. This improves cache locality and allows for efficient SIMD vectorization.

### Distributed Graph
The matrix structure is defined by a `DistGraph`, which manages:
- **Block Sizes**: The dimensions of each block row/column.
- **Connectivity**: Which blocks are non-zero (adjacency list).
- **Distribution**: Which MPI rank owns which rows.

### Vectors and MultiVectors
- **DistVector**: A distributed vector (1D).
- **DistMultiVector**: A collection of vectors (2D, column-major), useful for block solvers (e.g., solving $AX=B$).

---

## Creating Matrices

### 1. Serial Creation (Simple)
Best for small matrices or when the structure is known on a single rank. If `comm` is `None`, it runs in serial mode.

```python
import numpy as np
import vbcsr

# Serial mode (no MPI)
mat = vbcsr.VBCSR.create_serial(
    comm=None, 
    global_blocks=2, 
    block_sizes=[2, 2], 
    adjacency=[[0, 1], [0, 1]] # Full connectivity
)
```

### 2. Distributed Creation (Scalable)
For large problems, each rank defines only its owned rows.

```python
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Each rank owns 1 block of size 100
owned_indices = [rank] 
block_sizes = [100]
adjacency = [[rank, (rank+1)%size]] # Connect to self and next rank

mat = vbcsr.VBCSR.create_distributed(comm, owned_indices, block_sizes, adjacency)
```

### 3. Random Creation (Benchmarking)
Useful for testing and benchmarking.

```python
mat = vbcsr.VBCSR.create_random(comm, global_blocks=1000, block_size_min=10, block_size_max=20)
```

---

## Matrix Operations

### Assembly
After creating the matrix, you must add data blocks and call `assemble()`.

```python
# Add a 2x2 identity block at global block coordinates (0, 0)
mat.add_block(0, 0, np.eye(2))
mat.assemble() # Finalize communication
```

### Arithmetic & Properties
VBCSR supports natural Python operators and properties.

```python
# Properties
print(mat.shape)
print(mat.nnz)
print(mat.real)   # Real part
print(mat.imag)   # Imaginary part

# Arithmetic
B = mat * 2.0       # Scalar multiplication
C = mat + B         # Matrix addition
D = -mat            # Negation
E = mat - B         # Subtraction

# In-place Operations (Memory Efficient)
mat += C            # In-place addition
mat.scale(0.5)      # In-place scaling
mat.transpose_()    # In-place transpose
mat.conj_()         # In-place conjugate
```

### Scalar Access
You can access individual elements (slow, for debugging).

```python
val = mat[0, 0]
```

---

## Vector Operations

Vectors can be created from the matrix or from NumPy arrays.

```python
# Create vector compatible with matrix A
v = mat.create_vector()
v.set_constant(1.0)

# From NumPy (local part)
v_np = np.array([1.0, 2.0])
v.from_numpy(v_np)

# Arithmetic
w = mat @ v         # Matrix-Vector Multiplication (SpMV)
z = v + w           # Vector addition
dot = v.dot(w)      # Dot product
```

---

## Python & NumPy Interoperability

One of VBCSR's strongest features is its seamless integration with standard Python types and NumPy arrays. You don't need to manually wrap everything in VBCSR objects.

### Scalar Operations
You can use standard Python `int`, `float`, or `complex` scalars directly with VBCSR objects.

```python
# Matrix-Scalar
mat *= 2.5          # Scale matrix
mat += mat2         # Matrix addition

# Vector-Scalar
v = mat.create_vector()
v.set_constant(1.0)
v *= 0.5            # Scale vector
v += 1.0            # Add scalar to all elements (broadcast)
```

### NumPy Array Operations
VBCSR methods automatically handle NumPy arrays, treating them as the **local portion** of a distributed vector.

```python
import numpy as np

# Matrix-Vector Multiplication with NumPy input
x_np = np.random.rand(100)  # Local numpy array
y = mat @ x_np              # Returns a DistVector (y = A * x)

# Matrix-Matrix Multiplication (SpMM)
X_np = np.random.rand(100, 5) # 5 columns
Y = mat @ X_np                # Returns a DistMultiVector

# Vector Operations
v = mat.create_vector()
v += x_np           # Add numpy array to distributed vector
v *= x_np           # Element-wise multiplication
```

---

## SciPy Integration

VBCSR implements the `scipy.sparse.linalg.LinearOperator` interface, making it compatible with SciPy's iterative solvers.

```python
from scipy.sparse.linalg import cg, gmres

# Solve Ax = b
x, info = cg(mat, v, rtol=1e-5)

if info == 0:
    print("Converged!")
```

---

## Distributed Computing with MPI

When running with MPI (`mpirun -np N python script.py`), VBCSR automatically handles:
1. **Partitioning**: Rows are distributed among ranks.
2. **Ghost Exchange**: `assemble()` exchanges non-local matrix blocks.
3. **Communication**: Matrix-vector multiplication automatically synchronizes vector "ghost" elements (elements needed from other ranks).

**Note**: Input NumPy arrays passed to `from_numpy` or `mult` are always assumed to represent the **local** portion of the vector owned by the calling rank.

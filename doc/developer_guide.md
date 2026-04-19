# VBCSR Developer Guide

This guide documents the internal design decisions and conventions of the VBCSR library. It is intended for developers maintaining or extending the core functionality.

## Distributed Graph (`DistGraph`)

The `DistGraph` class manages the distributed adjacency structure of the block sparse matrix. A key design decision involves the ordering of ghost indices.

### Ghost Index Convention: "Sort by Owner"

In a distributed graph, local indices are assigned as follows:
1.  **Owned Indices**: $0$ to $N_{owned}-1$. These correspond to global indices owned by the local rank, sorted by global ID.
2.  **Ghost Indices**: $N_{owned}$ to $N_{total}-1$. These correspond to global indices owned by remote ranks.

**CRITICAL**: Ghost indices are sorted by **Owner Rank** first, and then by Global ID.

#### Rationale
This ordering is chosen to optimize the `DistVector::sync_ghosts` operation, which is the communication backbone of Sparse Matrix-Vector Multiplication (SpMV).
-   `MPI_Alltoallv` delivers data in rank order (data from Rank 0, then Rank 1, etc.).
-   By sorting ghost indices by owner rank, the incoming data buffer from MPI exactly matches the memory layout of the ghost elements in `DistVector`.
-   This enables a **zero-copy receive**, avoiding a costly unpacking step in the inner loop of iterative solvers.

#### Implications for Algorithms
Because ghosts are sorted by owner, **local indices do not necessarily correspond to monotonically increasing global indices**.
-   Example: Rank 0 has ghosts from Rank 2 (GID 50) and Rank 1 (GID 100).
-   Ghost order: Rank 1's ghosts (GID 100), then Rank 2's ghosts (GID 50).
-   Local Indices: $L_{owned}, \dots, L_{ghost1} \rightarrow 100, L_{ghost2} \rightarrow 50$.
-   Here, $L_{ghost1} < L_{ghost2}$ but $Global(L_{ghost1}) > Global(L_{ghost2})$.

**Developers must NOT assume that iterating over local column indices yields sorted global column indices.**
-   Operations like `axpby` (matrix addition) or `spmm` (matrix multiplication) that merge structures must handle unsorted global indices explicitly (e.g., using a "collect, sort, unique" approach instead of a linear merge).

## Block CSR (`BlockSpMat`)

`BlockSpMat` builds upon `DistGraph` to store matrix values.

-   **Storage**: Values are stored in backend-owned paged payload arrays. CSR and BSR use `PagedArray`, while true VBCSR uses shape-bucketed pages owned by the VBCSR backend.
-   **Handles**: the backend stores one graph-block handle per local nonzero block so graph traversal and shape-page storage remain decoupled.
-   **Thread Safety**: `BlockSpMat` is designed for OpenMP threading. Temporary buffers in operations like `axpby` or `spmm` should be thread-local to avoid contention and allocation overhead.

## VBCSR Apply Execution

The mixed-size VBCSR backend stores blocks in shape-bucketed pages. Each page is already physically contiguous in memory, so apply execution no longer has a separate packed-vs-unpacked layout mode.

For `mult`, `mult_dense`, `mult_adjoint`, and `mult_dense_adjoint`, the executor chooses between:

-   **Scalar path**: iterate one graph block at a time inside a same-shape page batch.
-   **Batched path**: use the same page-local block payloads plus operand scratch to launch strided batched BLAS or fixed-shape kernels.

### Shape Pages and Batch Metadata

-   every page contains blocks of exactly one shape `(row_dim, col_dim)`
-   live blocks are stored in `0..live_count-1`
-   the matrix-side block payloads are page-local contiguous storage
-   page batches carry graph-block indices and row/column block metadata so kernels do not need to reconstruct that information elsewhere

### What One Apply Batch Means

`for_each_shape_batch(...)` emits one batch per non-empty VBCSR shape page. For one batch:

-   `batch.row_dim` is the number of rows in every block in the batch
-   `batch.col_dim` is the number of columns in every block in the batch
-   `batch.block_count()` is the number of live blocks in that page

So a batch is not a matrix row batch. It is a **same-shape page batch**.

### Batched Apply Pipeline

In the batched path, the block payloads are read directly from the VBCSR page, but the dense operands are still packed into operation-local scratch:

-   `mult`: pack one `x` segment per block and a zeroed `y` scratch buffer
-   `mult_dense`: pack one dense RHS tile per block and a zeroed output scratch buffer
-   adjoint variants do the analogous transpose/adjoint packing

After the batched GEMV/GEMM call completes, each task scatters its scratch result back into the thread-local output accumulator.

This means VBCSR avoids repacking matrix blocks for apply, but still pays for operand packing because the logical destination/source rows of the RHS data are not globally strided.

### Threading Model

Apply parallelism is in the outer OpenMP loop. The intended model is:

-   one user-visible thread count for VBCSR apply
-   one BLAS thread per OpenMP worker
-   one private output accumulator per OpenMP thread, followed by a final reduction

The current code exposes a helper to pin MKL/OpenBLAS to one thread, but VBCSR callers should still treat "outer OpenMP threads, inner BLAS thread = 1" as the supported execution model.

### Why Large Page Batches May Need Splitting

VBCSR parallelizes primarily across shape pages. This works well when there are many non-empty pages, but it can underutilize threads when:

-   the matrix has only a few repeated shapes, and
-   one shape owns one or a small number of very large pages

To avoid that starvation, the apply executor can split one large page batch into multiple `ApplyTask`s. Each task is a contiguous subrange of one shape page:

-   `batch_index`: which shape page batch owns the task
-   `begin`: first live block inside that page
-   `count`: number of blocks in the task

The matrix data remain contiguous inside the page, so each task can still launch a single batched kernel on its subrange.

### `choose_chunk_size()` and `kTargetScratchBytes`

`choose_chunk_size()` is currently the key heuristic controlling both scratch size and apply task granularity.

-   `kTargetScratchBytes` is the target scratch budget per batched micro-kernel launch
-   the executor estimates scratch needed **per block**
-   chunk size is chosen so `chunk_size * per_block_scratch ~= kTargetScratchBytes`

The per-block scratch estimate depends on block shape:

-   `mult` / `mult_adjoint`: roughly `row_dim + col_dim`
-   `mult_dense` / `mult_dense_adjoint`: roughly `num_vecs * (row_dim + col_dim)`

Implications:

-   larger blocks or more RHS vectors produce smaller chunks
-   smaller blocks produce larger chunks
-   when page splitting is enabled, the same chunk size also becomes the apply task size

So `kTargetScratchBytes` is currently both:

-   the scratch-memory budget knob, and
-   an indirect parallelization-granularity knob for VBCSR apply

This is a deliberate simplification for now. A future refinement may separate "scratch budget" from "parallel task size" into different tuning parameters.

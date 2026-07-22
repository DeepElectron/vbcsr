# VBCSR Developer Guide

This guide documents the internal design decisions and conventions of the VBCSR library. It is intended for developers maintaining or extending the core functionality.

## Distributed Graph (`DistGraph`)

The `DistGraph` class manages the distributed adjacency structure of the block sparse matrix. A key design decision involves the ordering of ghost indices.

### Global-to-Local Lookup (`detail::IndexMap`)

`DistGraph::global_to_local` is a `detail::IndexMap`
(`detail/storage/index_map.hpp`), not a `std::map`: a flat, open-addressed
int → int table. It is the hottest lookup in the library (every distributed
apply, SpGEMM, transpose and assembly probes it), and building it dominated
distributed result-graph construction when it was tree-based.

It exposes a deliberate **subset** of the `std::map` API — `at`, `find`/`end`,
`count`, `operator[]`, `size`, `empty`, `clear`, `reserve`, with `find`
returning a handle exposing `->first`/`->second` — so call sites read exactly
as they did before.

It deliberately provides **no iteration**. The container has no key order, and
omitting `begin()` turns a future ordering dependency into a compile error
rather than a silent behavior change. Keys must be `>= 0` (`-1` marks a free
slot), and `operator[]` references are invalidated by a later growing insert,
the same rule `std::unordered_map` has for rehash.

Rank-keyed maps (e.g. `ghosts_by_rank`) stay `std::map`: their ordered
iteration is what makes ghost ordering deterministic across ranks.

### Distributed CSR SpGEMM: the fused wrap path

The distributed CSR SpGEMM (`CSRSpMMExecutor::run_generic`) runs a **fused
wrap** by default — the distributed generalization of the serial
`run_mkl_serial` wrap, so serial and distributed are one algorithm:

1. Build one extended operand `B_ext = [B's owned rows ; one row per A-ghost]`.
   The ghost rows are laid out **in A's ghost order** (`ghost_global_indices`),
   with an empty row where nothing was fetched. Because A's local column ids
   are owned-first then owner-sorted ghosts, A's own CSR arrays then index
   `B_ext`'s rows verbatim: **A passes to the vendor zero-copy**
   (`full_mkl_batch(A)`). `B_ext`'s columns reuse **B's local column space**
   plus a small tail for ghost-row columns that are not B blocks, so the
   B-owned portion is two `memcpy`s.
2. One `mkl_sparse_spmm(A, B_ext)` performs the owned+ghost merge internally.
3. The exported CSR is **wrapped as the result**: used ghost columns are
   compressed and merged into the (owner, gid) convention (B's ghosts are
   already so ordered; only the surface-sized extras are sorted), column ids
   are remapped in one linear pass, values copy in vendor order, and
   `DistGraph::construct_from_local_csr` adopts the arrays. Ghost block sizes
   and the partition are already known locally, so the only collective in
   result construction is `build_comm_pattern`.

**Collective-uniformity invariant.** The wrap and the fallback hash path
execute *different MPI collective sequences* while building the result graph.
The branch is therefore decided by a global vote (`MPI_Allreduce(MIN)` over
each rank's local eligibility); after a unanimous vote the wrap must not fall
back — vendor failures throw. Never add a per-rank early-return to this path.

Eligibility: MKL scalar type, single-page operands, unsorted-output mode, and
A/B sharing the ownership partition. Anything else takes the native hash path.
Opt out globally with `VBCSR_FUSED_DIST_SPGEMM=0`. Profile either path with
`VBCSR_PROFILE_CSR_DIST_SPGEMM=1`.

`threshold > 0` follows the **serial MKL semantics**: the vendor product is
exact, then entries with `|value| < threshold` are dropped in a compaction
pass before the result graph is built — so thresholded CSR results are
identical at every rank count. The hash path keeps the legacy per-product
`row_eps` drop (its results can differ from the serial/fused ones within the
threshold bound); it is only reached via opt-out or ineligibility.

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

## Data Layout Contract

Since the row-major migration (history and rationale in `doc/row_major_migration_plan.md`, §2.1 plus the per-phase execution records), the core observes one layout contract:

-   **Block storage and MPI payloads**: canonical row-major — element $(i, j)$ of an $r \times c$ block sits at `data[i * c + j]`. The single source of truth is `vbcsr::kCanonicalBlockLayout` in `vbcsr/core/block_csr.hpp`; use the constant at staging and transport call sites, never a literal enum. `MatrixLayout::ColMajor` remains only as a conversion option at the `add_block` / `get_block` / `get_values` boundary (transpose per block) and in genuinely column-major code: the CBLAS wrappers in `detail/kernels/` and the LAPACK work buffers in `detail/ops/spmf/`.
-   **`DistMultiVector`**: row-major — element $(row, vec)$ at `data[row * ld + vec]` with `ld = round_up(num_vectors * sizeof(T), 64) / sizeof(T)`, so each row starts on a cache line.
-   **Padding invariant**: lanes `[num_vectors, ld)` of every multivector row are always zero. Flat-loop ops (`scale`, `axpy`, `bdot`, norms) run over the whole padded buffer and depend on this; any code that bulk-writes the buffer must re-establish it via `zero_padding()`.

### Kernel Family

The dense block kernels live in `detail/kernels/rowmajor_kernels.hpp` and vectorize along the `num_vectors` axis, treating the matrix element as a broadcast operand:

-   `rm_gemm` / `rm_gemm_adjoint`: block times dense tile, for the `mult_dense` paths
-   `rm_gemv` / `rm_gemv_adjoint`: block times vector, for the SpMV paths
-   AVX2 fast paths cover `double` and `std::complex<double>`; a generic scalar fallback covers other types and ISAs

The old column-major kernel family (`NaiveKernel`, `TinyBlockKernel`, `FixedBlockKernel`, the `SmartKernel` switch tables) was removed, and `BlockSpMat` no longer takes a kernel template parameter. `BLASKernel` in `detail/kernels/dense_kernels.hpp` remains as the vendor BLAS wrapper plus threading configuration (`configure_native_threading` and friends).

### Runtime and Build Knobs

-   `VBCSR_BSR_VENDOR=1` (legacy alias: `VBCSR_BSR_DENSE_VENDOR`): route BSR SpMV and dense applies through the MKL vendor paths. The default is the native path, which measured faster. MKL BSR quirk: `mm` with a row-major dense operand supports exactly the (row-major blocks, zero-based indexing) and (column-major blocks, one-based indexing) pairings — the other two combinations fail at apply time.
-   `VBCSR_FUSED_DIST_SPGEMM=0`: opt out of the fused distributed CSR SpGEMM wrap (see §Distributed CSR SpGEMM) and use the native hash-accumulation path. Default is on.
-   `VBCSR_SPGEMM_SORTED=1`: make MKL-path SpGEMM emit sorted column indices per row. The default keeps the vendor export order: no library consumer needs a matrix's own adjacency sorted (audited during the migration), and `to_scipy` sorts on the scipy side.
-   Build options: `VBCSR_ARCH=native|avx2|none` selects the ISA flags; `VBCSR_SANITIZE=address|undefined` enables sanitizer builds.

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

`OMP_NUM_THREADS` is the single source of truth for parallelism. Every
operation entry point declares which of the two execution models it uses
*before* computing, via the two `BLASKernel` helpers
(`detail/kernels/dense_kernels.hpp`):

-   **Vendor-executed** (MKL sparse owns the parallelism — CSR applies, the
    CSR/BSR MKL SpGEMM multiplies, opt-in BSR vendor applies):
    `configure_vendor_sparse_threading()` sets the MKL pool to
    `omp_get_max_threads()` so the vendor kernel uses the same budget the
    OpenMP paths would.
-   **Native** (our OpenMP loop owns the parallelism; BLAS/LAPACK may be
    called from inside it — all native applies, the VBCSR SpGEMM numeric
    phase with `gemm_batched`, the `spmf` per-subgraph `dsyevd`/`gemm`):
    `configure_native_threading()` clamps the BLAS runtime to one thread so
    a pool size left behind by a previous vendor call cannot oversubscribe.

Invariants: a vendor entry configures the pool *before* building/optimizing
its handles (`mkl_sparse_optimize` records a schedule for the current pool);
opt-in vendor branches (BSR: `VBCSR_BSR_VENDOR`) configure the pool and build
handles only when the opt-in is active; native paths that call no BLAS at all
(transpose, axpby, SpGEMM generic fallbacks) need no configuration — their
OpenMP loops follow `OMP_NUM_THREADS` directly. Environment variables like
`MKL_NUM_THREADS` only determine the pool state *before the first VBCSR
call*; the entry-point configuration overrides them per operation.

On multi-socket hosts, pin threads (e.g. `OMP_PROC_BIND=close
OMP_PLACES=cores`) — native applies first-touch their output inside the
parallel compute region, so placement follows the compute threads.

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

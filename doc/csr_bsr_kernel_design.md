# CSR / BSR Kernel Design Note

This note records the recommended high-performance kernel direction for the `CSR` and `BSR` backends after the packed VBCSR apply work.

It is intentionally a design reference, not a promise that every item is already implemented. The goal is to make future backend work converge on one mature architecture instead of growing several unrelated kernel paths.

## Scope

This document covers:

- `mult`
- `mult_dense`
- `mult_adjoint`
- `mult_dense_adjoint`
- `spmm` / `spmm_self`

for the internal `CSR` and `BSR` backends used by `BlockSpMat<T, Kernel>`.

The main code entry points today are:

- `vbcsr/core/detail/csr_kernels.hpp`
- `vbcsr/core/detail/bsr_kernels.hpp`
- `vbcsr/core/detail/csr_spmm.hpp`
- `vbcsr/core/detail/bsr_spmm.hpp`
- `vbcsr/core/block_csr.hpp`

## Current Baseline

### CSR

The current CSR apply kernels are straightforward row-parallel OpenMP loops:

- `csr_mult(...)`
- `csr_mult_dense(...)`
- `csr_mult_adjoint(...)`
- `csr_mult_dense_adjoint(...)`

This is a good correctness baseline, but it is not a state-of-the-art CPU sparse apply design. In particular:

- `mult_dense` is still scalar-inner-loop code instead of a mature sparse-matrix-dense-matrix backend.
- adjoint methods allocate one full private output buffer per OpenMP thread and reduce it through a critical section.
- the implementation does not cache any inspector-style execution metadata.

### BSR

The current BSR apply kernels are stronger than CSR already:

- fixed-size native block kernels exist for hot block sizes such as `2`, `4`, `8`, and `16`
- the fallback path uses `SmartKernel`
- the outer parallelism is over block rows

This is the right general shape for BSR, but there is still room to make `mult_dense` and adjoint paths more cache-aware and more scalable.

### Important Correction About BSR

Uniform `BSR` block shape does **not** mean the dense RHS is globally contiguous.

What uniform shape gives us is:

- one fixed `(block_size, block_size)` kernel shape
- contiguous matrix payload storage inside a page
- simpler fixed-block microkernel dispatch

What it does **not** remove is:

- sparse gathers into `x` / `B`
- sparse scatters or reductions into output blocks
- irregular access driven by block column indices

So BSR is much simpler than VBCSR, but it should not be treated as a fully strided dense problem.

## Design Goals

The target design should optimize for:

- highest practical CPU throughput on real workloads
- predictable threading behavior
- minimal duplication of hard-to-maintain kernel logic
- good support for both real and complex scalar types
- clear backend-specific fast paths instead of forcing one execution model onto all formats

## Recommended High-Level Architecture

Use a hybrid design:

- `CSR`: vendor sparse backend first, strong internal transformed fallback second
- `BSR`: native fixed-block kernels first, vendor sparse optional for selected cases
- `spmm`: vendor sparse only when semantics match the public API; keep custom paths where thresholding or filtering is essential

This is intentionally different from true VBCSR:

- VBCSR benefits from shape batching and packed operand scratch
- CSR is better served by mature sparse-library apply backends or a transformed SIMD-friendly fallback
- BSR is better served by row-parallel fixed-block kernels with dense-RHS tiling

## CSR Design

### Recommended Primary Path

For CPU apply, the primary CSR fast path should be a vendor sparse backend when available.

#### Intel / oneMKL

Prefer the inspector-executor sparse interface:

- `mkl_sparse_create_csr`
- `mkl_sparse_set_mv_hint`
- `mkl_sparse_set_mm_hint`
- `mkl_sparse_optimize`
- `mkl_sparse_mv`
- `mkl_sparse_mm`

For sparse-sparse multiplication, prefer:

- `mkl_sparse_spmm`

when its semantics match the requested operation closely enough.

The implementation should cache the created sparse handle and optimized execution metadata across repeated applies of the same matrix.

#### AMD / AOCL-Sparse

Prefer AOCL-Sparse for the operations it covers well, especially:

- CSR SpMV
- CSR SpMM / sparse-dense multiply
- CSR sparse-sparse multiply where the library behavior matches our API requirements

The same design principle applies: build and cache an execution object, provide hints, and reuse it instead of rebuilding every call.

### Recommended Fallback Path

Plain CSR row loops should remain the portability baseline, but they should not be the long-term "fast" fallback for CPUs.

The recommended internal fallback is a cached `SELL-C-sigma` style execution format for apply:

- better SIMD regularity than raw CSR
- simpler engineering and maintenance cost than CSR5
- strong literature support as a practical CPU format
- a good fit for both `mult` and `mult_dense`

This transformed execution view can be built lazily from the canonical CSR structure and cached per matrix.

### Adjoint Apply

The current CSR adjoint path uses:

- thread-local full output buffers
- a critical-section reduction at the end

That is simple and correct, but it is not the mature target.

The preferred future design is:

- use the vendor sparse backend transpose/conjugate-transpose operation when available, or
- build and cache a transpose-oriented execution view, such as CSC-like row access or a transformed transpose cache

This avoids the large per-thread temporary memory footprint and the serial reduction point.

### Dense RHS (`mult_dense`)

For dense multi-vector apply:

- prefer vendor `csrmm` / sparse-dense-matrix multiply when available
- otherwise tile the RHS columns in the internal fallback

The internal fallback should never process a large dense RHS as one long scalar inner loop if a tiled kernel can improve locality.

## BSR Design

### Recommended Primary Path

For BSR apply, native fixed-block kernels should remain the primary design.

This means:

- parallelize over block rows
- dispatch to specialized microkernels for hot block sizes
- keep the generic `SmartKernel` fallback for cold or large block sizes

This is already close to the current structure in `vbcsr/core/detail/bsr_kernels.hpp`, so the path forward is evolutionary rather than a full rewrite.

### `mult`

The preferred BSR `mult` design is:

- row-parallel outer OpenMP
- fixed-block `gemv` microkernels for common block sizes
- fallback to `SmartKernel::gemv` for the rest

This should usually beat trying to force a VBCSR-style page batch abstraction onto BSR SpMV.

### `mult_dense`

The most important next improvement for BSR apply is a tiled dense-RHS kernel.

The natural compute unit is:

- one block row
- one tile of RHS columns

not:

- one whole page batch across unrelated rows

The recommended structure is:

- choose a small RHS tile width
- gather one RHS tile for each source block touched by the row
- run block GEMM-style kernels over that tile
- accumulate directly into the destination row tile

This keeps the dense-RHS reuse local to the block row and preserves the simple ownership model.

### Adjoint Apply

For BSR adjoint methods, the same issue as CSR exists today:

- full per-thread temporary output buffers
- reduction after the OpenMP loop

The preferred long-term design is either:

- a transpose-oriented execution view, or
- a row/column ownership strategy that avoids full-vector critical-section reductions

This can be introduced after the forward `mult_dense` path is improved.

### Vendor Sparse for BSR

Vendor BSR kernels are still worth considering, but they should be treated as optional accelerators rather than the only plan.

Reasons:

- BSR apply already maps naturally to efficient native fixed-block kernels
- vendor BSR interfaces may impose layout or indexing constraints that are not exact drop-ins
- internal kernels remain important for complex support, custom storage, and backend control

So the recommended priority is:

- native BSR kernels first
- optional vendor BSR path second

## Sparse-Sparse Multiply (`spmm`)

### CSR

For CSR `spmm`, vendor sparse multiplication is attractive when:

- thresholding is zero, or
- it is acceptable to build the full product first and filter afterward

When thresholding and structure filtering are fundamental to the algorithm, the current symbolic-plus-numeric custom pipeline remains the better architectural base.

### BSR

For BSR `spmm`, the same rule applies:

- use vendor sparse only when the public API semantics line up cleanly
- otherwise keep the custom symbolic filtering pipeline and optimize its numeric phase

For the internal numeric phase, BSR should continue to benefit from block GEMM kernels, but the main algorithmic structure should still be driven by the symbolic plan and threshold logic.

## Threading Model

CSR and BSR should not blindly copy the VBCSR threading strategy.

### Vendor Sparse Path

When a call is handed to oneMKL or AOCL-Sparse:

- let the vendor library own the threading for that call
- do not wrap the vendor sparse call inside another OpenMP row loop

This avoids oversubscription and respects how mature sparse libraries are tuned.

### Internal Native Path

When the library uses its own kernels:

- use outer OpenMP parallelism
- keep any nested BLAS or dense helper kernels single-threaded

This matches the intended model already used by VBCSR packed apply.

### Practical Rule

The simplest long-term policy is:

- one user-visible thread setting
- internal code decides whether that means vendor-threaded sparse execution or outer-OpenMP native execution
- no nested sparse-library threads inside our own parallel row loops

## Recommended Implementation Order

The highest-value roadmap is:

1. Add a vendor sparse backend layer for CSR apply and CSR `spmm`.
2. Add a cached execution object / inspector layer for CSR matrices.
3. Improve BSR `mult_dense` with explicit RHS tiling and keep native fixed-block kernels as the default.
4. Rework CSR and BSR adjoint methods to avoid full-thread-local output reductions.
5. Add optional vendor BSR apply or `spmm` acceleration only where it clearly improves over the native path.
6. Add a stronger transformed CSR fallback such as `SELL-C-sigma` for platforms without a strong vendor sparse backend.

## Practical Guidance For Future Work

If the goal is highest immediate speed with the least risk, prefer this order of decisions:

- for `CSR mult` and `CSR mult_dense`: vendor sparse first
- for `BSR mult`: native fixed-block kernel first
- for `BSR mult_dense`: native tiled block-row kernel first
- for `CSR/BSR spmm`: keep the custom threshold-aware pipeline unless a vendor call matches the requested semantics exactly

## External References

- Intel oneMKL sparse matrix handle and execution APIs:
  - `mkl_sparse_create_csr`
  - `mkl_sparse_create_bsr`
  - `mkl_sparse_mv`
  - `mkl_sparse_mm`
  - `mkl_sparse_spmm`
  - `mkl_sparse_optimize`
- AMD AOCL-Sparse overview and user guide
- SELL-C-sigma:
  - Kreutzer et al., "A unified sparse matrix data format for efficient general sparse matrix-vector multiply on modern processors with wide SIMD units"
- CSR5:
  - Liu and Vinter, "CSR5: An Efficient Storage Format for Cross-Platform Sparse Matrix-Vector Multiplication"

## Status

As of this note:

- VBCSR has the new packed apply path and explicit `contiguous()` preparation
- CSR and BSR still use the current native kernels in `csr_kernels.hpp`, `bsr_kernels.hpp`, `csr_spmm.hpp`, and `bsr_spmm.hpp`
- this document describes the preferred next-generation direction for CSR / BSR performance work

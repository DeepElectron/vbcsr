# VBCSR Core Architecture

The C++ core uses a shallow hybrid layout. Public headers stay in `vbcsr/core`;
implementation details live under `vbcsr/core/detail` and are grouped by the
kind of code they own.

## Reading Path

Start with `vbcsr/core/block_csr.hpp`. It is the public matrix facade and shows
which public operations dispatch into backend-specific implementations.

Then read the internals in this order:

1. `vbcsr/core/block_csr.hpp` for backend selection and storage dispatch.
2. One concrete backend, such as `detail/backend/csr_backend.hpp`,
   `detail/backend/bsr_backend.hpp`, or `detail/backend/vbcsr_backend.hpp`.
3. `detail/kernels/rowmajor_kernels.hpp` for the native dense block kernels,
   plus `detail/kernels/dense_kernels.hpp` for the `BLASKernel` vendor
   wrappers and threading configuration, and `detail/kernels/blas_api.hpp` /
   `detail/kernels/lapack_api.hpp` for external numerical library
   declarations.
4. `detail/kernels/*_apply.hpp` for sparse matrix-vector and multivector apply.
5. `detail/ops/` for larger matrix operations such as transpose, axpby, SpMM,
   and graph/subspace matrix functions.
6. `detail/distributed/` for shared communication helpers and result graph
   construction.

## Module Groups

- `detail/storage/` contains low-level paged storage utilities shared by
  backends.
- `detail/backend/` contains CSR, BSR, and VBCSR storage backends.
- `detail/kernels/` contains BLAS/LAPACK declarations, dense block kernels, and
  sparse apply implementations.
- `detail/ops/` contains operation-level algorithms. Keep shared symbolic or
  exchange logic close to the operation that uses it.
- `detail/distributed/` contains communication helpers shared across
  operations, including internal MPI utility wrappers.

Top-level public headers should expose user-facing matrix, graph, and vector
types. Internal code should include the specific detail header that owns the
needed implementation.

## Data Layout Contract

These invariants hold everywhere in the core. They were established by the
row-major migration; `doc/row_major_migration_plan.md` (§2.1 and the per-phase
execution records) has the derivation and history.

- Matrix blocks are canonically row-major: element `(i, j)` of an `r x c`
  block sits at `data[i * c + j]`. This holds for backend storage, pending
  assembly buffers, and every MPI transport payload. The single source of
  truth is `vbcsr::kCanonicalBlockLayout` in `vbcsr/core/block_csr.hpp`;
  staging and transport call sites use the constant, never a literal enum.
  `MatrixLayout::ColMajor` survives only as a boundary conversion option
  (`add_block` / `get_block` / `get_values` transpose per block) and in
  genuinely column-major territory such as the CBLAS wrappers and the LAPACK
  workspaces under `detail/ops/spmf/`.
- `DistMultiVector` is row-major with a padded leading dimension: element
  `(row, vec)` sits at `data[row * ld + vec]` with
  `ld = round_up(num_vectors * sizeof(T), 64) / sizeof(T)`. Padding lanes
  `[num_vectors, ld)` of every row are always zero. Flat loops over the whole
  buffer rely on this invariant, and any bulk write must restore it
  (`zero_padding()`).
- Dense block kernels vectorize over the vector axis:
  `detail/kernels/rowmajor_kernels.hpp` provides `rm_gemm`,
  `rm_gemm_adjoint`, `rm_gemv`, and `rm_gemv_adjoint`, with the matrix
  element as a broadcast operand, AVX2 fast paths for `double` and
  `std::complex<double>`, and a generic scalar fallback. The former
  column-major kernel family and the `BlockSpMat` kernel template parameter
  are gone; `BLASKernel` in `detail/kernels/dense_kernels.hpp` remains as
  the vendor BLAS wrapper and threading configuration.
- Python contract: `to_numpy()` returns a C-contiguous
  `(rows, num_vectors)` array, `add_block` takes a memcpy fast path for
  C-contiguous input, and `to_scipy` sorts indices on the scipy side.

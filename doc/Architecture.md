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
3. `detail/kernels/dense_kernels.hpp` for local dense block kernels, plus
   `detail/kernels/blas_api.hpp` and `detail/kernels/lapack_api.hpp` for
   external numerical library declarations.
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

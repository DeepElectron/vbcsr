# High-Performance Unified-API Refactor Plan for VBCSR

## Summary

- Keep one public Python matrix class, `VBCSR`, and one public C++ matrix class, `BlockSpMat<T, Kernel>`.
- Split the implementation into three internal backends selected from the block-size list: `CSR` for all block sizes equal to 1, `BSR` for uniform block sizes greater than 1, and true `VBCSR` for mixed block sizes.
- Keep `DistGraph`, `DistVector`, and `DistMultiVector` shared across all backend kinds; the distribution layer remains unified.
- Backend kind is fixed for a matrix family once the matrix/graph is constructed, and supported operations preserve that backend kind. Inter-backend operations are out of scope for this refactor, so a `CSR` matrix produces `CSR` results, a `BSR` matrix produces `BSR` results, and a true `VBCSR` matrix produces true `VBCSR` results.
- Prefer the smallest set of persistent abstractions that still exposes the right kernel-friendly layout. Derived execution metadata should be created lazily or on the stack unless repeated reuse clearly justifies caching it.
- Deliver the refactor as a staged migration with benchmark gates after each backend is enabled.

## Public API and Compatibility

- Preserve the current Python API shape and preserve the current high-level C++ method set on `BlockSpMat<T, Kernel>`: constructor from `DistGraph*`, `add_block`, `assemble`, `mult`, `mult_dense`, `mult_adjoint`, `mult_dense_adjoint`, `spmm`, `spmm_self`, `axpy`, `add`, `transpose`, `filter_blocks`, `duplicate`, `extract_submatrix`, `insert_submatrix`, `to_dense`, `from_dense`, `get_block`, and `get_values`.
- Keep `vbcsr/core/block_csr.hpp` as the installed top-level C++ header and make it the facade header for all backend kinds.
- Freeze the meaning of the `Kernel` template parameter early. After this refactor, `Kernel` is not the global execution policy for every backend. It should be interpreted as a dense-block microkernel policy / compatibility parameter that is relevant primarily to block backends, especially true `VBCSR`, while `CSR` may ignore it and `BSR` may use it only partially. Backend selection and high-level sparse algorithm selection remain separate from the `Kernel` template parameter.
- Add one non-breaking inspector, `matrix_kind()` or `backend_kind()`, on both C++ and Python sides for debugging, tests, and benchmarks.
- Keep `graph` plus logical block-structure views `row_ptr` and `col_ind` available from the facade so existing export/binding code still has a backend-neutral block-level structure. The canonical logical block graph should live on `DistGraph` as `adj_ptr` / `adj_ind`; facade `row_ptr` / `col_ind` should be graph-backed views rather than duplicated steady-state ownership.
- Treat raw storage members such as `blk_handles`, `blk_sizes`, and `arena` as internal-only going forward. Migrate pybind, tests, and internal helper code to facade accessors and backend-neutral iteration before enabling multiple backends.
- Add one backend-neutral C++ iterator/helper such as `for_each_local_block(...)` for internal algorithms and advanced C++ use, so backend details never leak into user code.

## Backend Storage Architecture

- Make storage separation the core architectural rule: `BlockSpMat` preserves the unified logical interface, while each backend owns the native execution-form storage needed for its hot kernels.
- Keep the facade as the single owner of `DistGraph* graph`, `MatrixKind`, and shared caches such as block norms. Logical structure views such as `row_ptr` and `col_ind` remain part of the facade interface, but their authoritative storage should come from `DistGraph::adj_ptr` / `DistGraph::adj_ind` so all backends reuse one canonical logical block graph.
- Use paged linear storage as the general memory policy for large arrays across all backends. True `VBCSR` needs paging for dynamic block allocation, and `CSR`/`BSR` need paging as well to avoid single-allocation overflow and 32-bit array-length limits that would otherwise cap problem size.
- Use one canonical dense-block memory layout across block backends, column-major, so block kernels and BLAS paths can operate directly without per-operation transpose/copy for layout matching.
- Enforce a no-convert-for-kernel rule: every operation must read the source matrix in its native storage and, when a new matrix is created, write the result directly into the same backend family. Temporary format conversion and cross-backend result construction are not allowed on the critical path.
- Split steady-state storage from mutation/build storage. Every backend gets a native immutable-or-mostly-stable storage layout plus a builder/workspace used by structure-changing operations such as `axpby`, `filter_blocks`, `spmm`, and `transpose`.
- Here "builder" means an internal construction workspace for one result matrix. It is not a new public matrix type, not a new exposed API concept, and not a replacement for the current paged arena used by true `VBCSR`.
- Helper type names in this document are illustrative unless explicitly called out as part of the stable public API. A single shared helper is preferred over multiple backend-specific helper classes when it does not compromise hot-path performance.

### Facade and Backend Object Boundary

- `BlockSpMat<T, Kernel>` remains the single public C++ matrix object and owns all backend-independent matrix identity:
  - `DistGraph* graph`
  - `bool owns_graph`
  - `MatrixKind kind`
  - logical block-structure accessors such as `row_ptr()` and `col_ind()`
  - shared cache state such as `block_norms` and `norms_valid`
  - one closed-set backend handle such as a tagged union or `std::variant`
- `Kernel` should not control backend dispatch. Backend dispatch is determined solely by `MatrixKind`. `Kernel` is carried through the public type for source compatibility and for dense-block execution policy where applicable.
- Preserve the current vector-like `row_ptr` / `col_ind` interface, but treat it as a stable facade view over graph-owned logical adjacency rather than as separate steady-state matrix storage. Builders and symbolic phases may still use temporary `row_ptr` / `col_ind` buffers before committing a new `DistGraph`.
- `BlockSpMat` should not retain steady-state backend-specific fields from the current monolithic implementation. In particular, `blk_handles`, `blk_sizes`, `arena`, and `thread_remote_blocks` should move out of the facade.
- Backend objects own only what is necessary to execute kernels and expose block values in their native layout:
  - `CSRMatrixBackend<T>` owns paged scalar `values`, its authoritative execution-form structure where needed for paired page traversal, and may ignore `Kernel`
  - `BSRMatrixBackend<T>` owns uniform `bsz`, paged packed-block `values`, and the authoritative execution-form block structure needed for paired page traversal; it may use `Kernel` only for dense block microkernel choices
  - `VBCSRMatrixBackend<T, Kernel>` owns the logical-slot-to-handle map, shape registry, page pools, and optional shape-kernel cache
- Operation-scoped structures are not part of the steady-state matrix object:
  - symbolic slot maps from `(row, col)` to logical block position
  - distributed row/payload exchange plans
  - temporary remote assembly buffers
  - thread-local hash accumulators
  - shape batch descriptor lists
- Because the backend set is closed and known at compile time, the preferred representation is one top-level closed tagged union inside `BlockSpMat`, for example `std::variant`. Public methods dispatch once at the facade boundary; inner numeric kernels stay statically typed inside each backend implementation.
- Add one internal `from_parts(...)` or equivalent factory that assembles a `BlockSpMat` from `graph`, `kind`, facade-visible logical structure, and a committed backend object. Builders commit backend storage, not whole matrices.

### Native Storage by Backend

- `CSR`: native scalar CSR whose backend owns paged scalar payloads and any derived execution metadata, while reusing graph-owned logical adjacency for row/column structure.
- `BSR`: native uniform-block storage whose backend owns one global `bsz`, paged block payloads, and any derived execution metadata, while reusing graph-owned logical adjacency for logical block slots.
- `VBCSR`: native paged block storage, evolved from the former `BlockArena` design, whose backend owns block handles, shape registry, and page-local payload organization, while the backend-neutral logical block graph continues to come from `DistGraph`.

### Cross-Backend Paged Linear Storage

- Introduce one small internal paged-array abstraction for large linear storage used by all backends, for example `PagedArray<T>` or `PagedBuffer<T>`, with fixed-capacity pages plus a logical global length.
- The primary motivation is robustness, not only mutability:
  - avoid one huge allocation for `values`
  - avoid `std::vector` or allocator length limits in 32-bit compilation modes
  - keep system size limited by total memory and index width, not by one container's maximum length
- The abstraction should support:
  - logical random access by global offset
  - efficient page-local contiguous spans for kernels
  - append/reserve by page count during builder fill
  - move-only ownership transfer at commit time
  - optional page-aligned allocation for SIMD/BLAS friendliness
- The first arrays that should be eligible for paging are:
  - CSR scalar `values`
  - BSR block `values`
  - true-`VBCSR` page payload slabs
- Extend paging to index-like arrays only if real system-size limits show that payload paging alone is not enough. Avoid paying that complexity up front.
- Performance-sensitive kernels must not pay a per-entry paging penalty. The intended execution model is page-wise traversal:
  - CSR kernels iterate row ranges but consume page-local contiguous value/index spans
  - BSR kernels iterate contiguous groups of blocks within a page
  - true `VBCSR` already works page-wise by design
- Paging is therefore a storage-level capacity mechanism, not a change to the logical sparse format. CSR stays CSR, BSR stays BSR, and true `VBCSR` stays shape-resolved variable-block storage.

### `PagedArray` Abstraction

- `PagedArray<T>` should be a small owning container with these steady-state data members:
  - `uint64_t logical_size`
  - `uint32_t page_capacity`
  - `std::vector<Page>`
- Each `Page` should carry:
  - aligned owning storage for `capacity` elements
  - `uint32_t used`
- Alignment policy and convenience caches such as `global_begin` are implementation details. They are useful when they simplify kernels, but they are not part of the required conceptual interface.
- Addressing must be defined entirely by 64-bit logical offsets:
  - `page_id = global_index / page_capacity`
  - `slot = global_index % page_capacity`
  - the last page may be partially full; all preceding pages are full in committed storage
- The minimal steady-state API should be:
  - `uint64_t size() const`
  - `uint32_t page_count() const`
  - `uint32_t page_capacity() const`
  - `const T& operator[](uint64_t i) const`
  - `T& operator[](uint64_t i)`
  - `PageSpan<T> page_span(uint32_t page_id)`
  - `void for_each_span(uint64_t begin, uint64_t end, Fn&& fn)` where each callback receives one contiguous page-local span
- Builders may construct paged arrays through either a mutable `PagedArray<T>` or a distinct `PagedArrayBuilder<T>`, but the committed matrix must end with stable page pointers and immutable logical size.
- The default page target should be byte-based and backend-tunable, for example an aligned target in the low-megabyte range. Element capacity is derived from the target bytes and rounded so page boundaries remain page-local-kernel-friendly.

### Kernel-Facing Page View API

- Kernels should not consume raw `PagedArray` internals directly. They should consume lightweight page-local view objects that are cheap to create and cheap to pass by value.
- The generic view is:
  - `PageSpan<T> { T* data; uint32_t length; uint32_t page_id; uint64_t global_begin; }`
- The generic traversal primitive is:
  - `for_each_span(begin, end, fn)` for one paged array
  - `for_each_zip_span(begin, end, fn)` for paired arrays that share the same logical partitioning
- CSR should expose a backend-specific zipped page view:
  - `CSRPageView<T> { const int* cols; const T* vals; uint32_t nnz; uint32_t page_id; uint64_t global_nnz_begin; }`
  - CSR kernels should iterate rows by `[row_ptr[i], row_ptr[i+1])`, then consume that row range through one or more `CSRPageView` callbacks if it crosses page boundaries
- BSR should expose a backend-specific zipped page view:
  - `BSRPageView<T> { const int* cols; const T* vals; uint32_t nblocks; uint32_t bsz; uint32_t block_elems; uint32_t page_id; uint64_t global_block_begin; }`
  - BSR kernels should iterate block rows by block-position ranges and consume one or more `BSRPageView` callbacks if a row crosses a page boundary
- True `VBCSR` already has a natural page-local view in its shape pages. Its kernel-facing page views should align with the same style:
  - `VBCSRShapePageView<T> { shape_id; data; live_slots; count; r_dim; c_dim; page_id; }`
- Page views are read/write views over already committed or builder-owned storage. They must not allocate, repack, or synthesize temporary numeric buffers.

### Paired-Page Policy for CSR and BSR

- CSR and BSR need aligned paging across structural and numeric arrays so zipped traversal remains simple and fast.
- CSR should page `col_ind` and `values` by the same logical unit, nonzeros:
  - choose one `nnz_per_page`
  - each CSR page stores up to `nnz_per_page` column ids and `nnz_per_page` scalar values
  - this guarantees `CSRPageView` can expose one contiguous pointer pair per callback
- BSR should page `col_ind` and `values` by the same logical unit, dense blocks:
  - choose one `blocks_per_page`
  - each BSR page stores up to `blocks_per_page` column ids and `blocks_per_page * block_elems` numeric values
  - this guarantees `BSRPageView` can expose one contiguous pointer pair per callback
- This paired-page policy is more important than using identical byte capacity across arrays. The goal is to preserve zipped kernel traversal by logical entry count.
- `row_ptr` may use its own paging policy because it is consumed differently and does not need to be zipped with payload arrays in hot inner loops.

### Page-Aware MatVec and Multi-RHS Execution

- `spmv` and multi-RHS `spmv`/`mult_dense` need an execution layer above raw page views. The storage layer should expose pages, but the execution layer should separate:
  - reduction metadata, which says how page-local work contributes back to output rows or block rows
  - compute batches, which are the real units sent to high-performance kernels
- Introduce reusable execution descriptors for page-aware vector application:
  - `CSRRowSegment { row; page_id; global_nnz_begin; local_offset; length; is_row_start; is_row_end; }`
  - `BSRRowSegment { block_row; page_id; global_block_begin; local_block_offset; block_count; is_row_start; is_row_end; }`
  - `VBCSRRowSegment { row; shape_id; page_id; segment_id; count; is_row_start; is_row_end; }`
- A row or block row may therefore be represented by one or more segments. These segments are reduction descriptors, not the primary compute granularity. One generic segment descriptor with optional backend-specific fields is acceptable if it keeps the implementation smaller.
- Build lightweight execution metadata lazily. A cached `MatVecExecutionPlan` is allowed for repeated multi-RHS use, but the first implementation should avoid committing to a large amount of persistent per-matrix scheduling state. The plan, when present, should contain:
  - row-to-segment offsets
  - per-page segment lists
  - per-shape segment lists for true `VBCSR`
  - page-batch descriptors for CSR and BSR
  - shape-batch descriptors for true `VBCSR`
  - optional precomputed gather metadata for multi-RHS paths
- The execution plan is matrix-structure-dependent and can be invalidated whenever the graph or logical sparsity changes.

### Single-RHS Rule

- For single-RHS `spmv`, page discontinuity should be handled by segmented reduction, not by repacking sparse values.
- CSR path:
  - schedule one `CSRPageView` or one CSR page-batch containing segments from many rows
  - stream contiguous page-local `cols` and `vals` through the kernel
  - use `CSRRowSegment` only to route partial sums to the correct output rows and to detect row starts/ends across page boundaries
- BSR path:
  - schedule one `BSRPageView` or one BSR page-batch containing segments from many block rows
  - stream contiguous page-local blocks through the kernel
  - use `BSRRowSegment` only to route partial dense-block sums to the correct output rows and to detect block-row starts/ends across page boundaries
- True `VBCSR` path:
  - schedule compute primarily by shape batch, not by row
  - within one shape batch, consume one or more shape pages that contain blocks from many rows
  - use `VBCSRRowSegment` only to route partial outputs back to the correct row accumulators when a row touches multiple shape pages
- This means page boundaries are a reduction concern, not a reason to give up page-local or shape-local batching.

### Multi-RHS Rule

- For multi-RHS application, the main compute unit should be `(compute batch, RHS tile)`, not `(row, RHS tile)`.
- CSR path:
  - use page batches that contain many rows from one CSR page
  - process one dense RHS tile at a time
  - choose between direct gather and packed gather for the RHS tile based on reuse inside that page batch
- BSR path:
  - use page batches that contain many block rows from one BSR page
  - process one dense RHS tile at a time
  - run page-local block GEMV/GEMM style kernels over all blocks in the batch, not one row at a time
- True `VBCSR` path:
  - use shape batches as the primary compute unit
  - each shape batch may span one or more shape pages and many rows
  - process one dense RHS tile at a time and launch the best available same-shape kernel path
- Row segments remain necessary in the multi-RHS path, but only as bookkeeping for output reduction. They must not dictate the main kernel granularity.

### VBCSR Paged Storage Design

- Keep handle-based indirection only for the true variable-block backend, because this is the backend that benefits from dynamic allocation during in-place or structure-changing operations.
- Retain the current arena idea for true `VBCSR`: paged allocation, stable handles, and decoupling between logical sparsity and physical block payload remain the core storage model.
- Replace the current `blk_sizes`-driven storage contract with shape-derived sizing from the logical structure in steady-state storage. For a block at `(row, col)`, size is computed from `graph->block_sizes[row]` and `graph->block_sizes[col]`; temporary builder/workspace code may still cache sizes or shapes during construction if that simplifies assembly.
- In the mixed-size steady state, facade accessors such as `block_data()` and `block_size_elements()` should resolve payload pointers and sizes from shape-page metadata plus slot handles, not from a persistent per-slot `blk_sizes` array.
- Organize pages into per-shape or per-shape-bin pools with fixed-size slots, free-slot lists, and stable 64-bit handles. This preserves the current "logical structure separate from physical values" strength while reducing fragmentation and removing redundant metadata.
- Keep allocation O(1) on the hot path for reused shapes by drawing from a shape-local free list first, then from the current page for that shape class.
- Support move-only ownership transfer from a `VBCSR` construction workspace into the final matrix so structure-changing operations can finish with pointer/handle swaps instead of deep copies.

### Shape-Resolved Pages and Kernel Batching

- Make block shape a first-class storage key for true `VBCSR`. Every stored block belongs to a `ShapeClass = (r_dim, c_dim)` and is allocated from the page pool of that exact shape class or a tightly bounded shape bin when exact classes are intentionally coalesced.
- Maintain a shape registry on each matrix that maps `ShapeClass` to page pool metadata, active block count, free-slot lists, and optional execution statistics such as call frequency and total flop volume.
- Ensure blocks of the same shape are not only logically discoverable but physically co-located well enough to feed batched dense kernels without a gather-copy stage. The storage design should let us materialize arrays of pointers or slot descriptors for same-shape blocks directly from page metadata.
- Add backend-native iteration APIs such as `for_each_shape_class(...)` and `for_each_shape_batch(...)` so `spmv`, multi-RHS `spmv`, adjoint paths, and `spmm` numeric phases can process one shape class at a time instead of dispatching block-by-block.
- Introduce per-shape work queues in structure-changing operations. For example, true `VBCSR` `spmm` should accumulate products into batches keyed by `(r_dim, inner_dim, c_dim)` and then launch one kernel path per batch.
- Use shape-resolved storage to support three execution tiers for true `VBCSR` kernels:
  - static fixed-shape microkernels for common hot shapes
  - batched BLAS-style kernels for medium and large repeated shapes
  - optional JIT-generated kernels for very hot irregular shapes when the runtime detects enough reuse to amortize compilation cost
- Treat JIT as an optimization layer, not a correctness dependency. The default implementation must still run fully through static kernels and BLAS fallback when JIT is disabled or unavailable.
- Avoid any batch-packing copy whose sole purpose is format matching. At most, allow lightweight pointer-list or descriptor-list assembly from the existing shape-resolved pages before launching a batched or JIT kernel.
- Preserve row-wise logical traversal semantics at the facade level, but allow the true `VBCSR` backend to schedule numeric work by shape class internally when that improves throughput.

### Shape Metadata

- Use a packed 64-bit handle for true `VBCSR` slots with logical fields `(shape_id, page_id, slot_id)`. The bit split should be chosen to avoid pointer chasing and to cover the expected scale envelope; the current `16/24/24` split is a reference target, not a hard design commitment.
- A handle must be sufficient to compute the payload pointer as `shape_registry[shape_id].pages[page_id].data + slot_id * slot_elems` with no per-block header lookup.
- The logical `row_ptr` and `col_ind` arrays remain the only source of sparsity order. True `VBCSR` payload pages do not store row or column ids per block.

### Shape Page Metadata

- Each `ShapePage` carries the minimal metadata needed for allocation, reclamation, and batch iteration:
- `shape_id`
- `page_id`
- `slot_elems = r_dim * c_dim`
- `slot_capacity`
- `live_count`
- `next_unused_slot`
- `data` pointer or owning storage object for the contiguous payload slab of size `slot_capacity * slot_elems`
- `free_slots` stack for reclaimed slot ids
- `live_slots` dense array listing currently active slot ids in iteration order
- `slot_live_pos` array of length `slot_capacity`, storing the position of each slot in `live_slots` or `UINT32_MAX` when the slot is free
- No per-slot payload header is allowed in steady-state storage. Slot state is tracked only by page metadata.
- Allocation rule: first reuse `free_slots`, otherwise consume `next_unused_slot`, otherwise allocate a new page in the same shape class.
- Free rule: remove the slot from `live_slots` by swap-remove through `slot_live_pos`, clear the payload if needed for correctness/debugging, and push the slot id onto `free_slots`.

### Shape Registry Entry

- Each `ShapeRegistryEntry` should keep the minimal persistent metadata needed to own one shape class and drive kernel dispatch:
- `shape_id`
- `r_dim`
- `c_dim`
- `slot_elems`
- `page_slot_capacity`
- `pages` vector
- `pages_with_free_slots` list or queue for O(1) allocation of non-full pages
- `active_block_count`
- `active_page_count`
- Optional tuning metadata may include allocation/free counters, hotness statistics, and preferred kernel kind, but those are optimization features rather than mandatory storage fields.
- Compiled JIT kernels themselves should not be owned solely by one matrix or one `ShapeRegistryEntry`. They should live in a process-global or otherwise shared compiled-kernel cache keyed by properties such as dtype, shape, transpose/adjoint mode, and ISA. Matrix-local state may keep only the hotness information and the preferred execution path used to query that shared cache.
- The registry entry is the place where shape-global policy lives: page sizing, batching thresholds, and preferred execution path.

### Batch View Metadata

- Batched execution metadata is temporary and must not be stored in steady-state matrix storage.
- For one numeric launch, a `VBCSRPageBatch` or equivalent function-local descriptor should contain:
- `shape_id`
- `count`
- array of `A` pointers or slot descriptors
- array of `B` pointers or slot descriptors when the kernel is binary
- array of `C` pointers or slot descriptors for output/accumulation
- leading dimensions, if the kernel path needs them
- optional scalar coefficients such as `alpha`, `beta`
- optional operation flags such as transpose/adjoint mode
- For true `VBCSR` `spmm`, the batch key must be `(r_dim, inner_dim, c_dim)`, not only `(r_dim, c_dim)`, because GEMM specialization depends on the reduction dimension too.
- Building a `VBCSRPageBatch` may assemble pointer arrays or slot-descriptor arrays, but it must not repack dense payload into a new contiguous numeric buffer just to satisfy batching.

### Builders and Mutation Path

- Add backend-native builders: `CSRBuilder<T>`, `BSRBuilder<T>`, and `VBCSRBuilder<T, Kernel>`. The implementation may call these builders, assemblers, or construction workspaces; the key requirement is backend-native result construction.
- This is an explicit generalization of what the current generic `VBCSR` code already does implicitly in structure-changing paths such as union `axpby`, `transpose`, and `spmm`: discover the output structure, allocate result storage, populate it, then replace or return the result matrix.
- For structure-preserving operations such as `scale`, `shift`, `add_diagonal`, and same-structure `axpy`, operate directly on the native storage without reallocation.
- For structure-changing operations, perform symbolic analysis first, construct the destination graph within the same backend family, then write directly into that backend family's builder. Finalization should be a move/swap into `BlockSpMat`, not a second materialization pass.
- `CSRBuilder` and `BSRBuilder` exist so the final matrix can land immediately in backend-native paged arrays, which avoids any post-build repacking before kernels run.
- `VBCSRBuilder` exists so a structure-changing result can be assembled in a separate paged arena/workspace, then committed by ownership transfer. This keeps the benefits of dynamic paged storage while avoiding mutation of the live matrix until the new structure is ready.
- `VBCSRBuilder` must populate shape-resolved page pools directly, update the destination shape registry as blocks are created, and emit the final matrix ready for shape-batched execution without a post-build reshuffle.
- `transpose`, `spmm`, `filter_blocks`, `extract_submatrix`, and structure-changing `axpby` must allocate directly in the source matrix's backend family. The refactor does not attempt to reinterpret an operation result as a different backend kind even if the resulting graph could satisfy another backend's structural rules.
- `axpby` should keep a same-structure fast path that never reallocates. When the sparsity pattern changes, use the destination builder and move-swap the result into place.
- If the operation does not create a new graph, backend kind does not change, no builder is needed, and no memory reorder is performed. Builders are only for constructing a changed result in the natural layout of a newly constructed graph within the same backend family.

### Minimal Builder API

- Builders are internal result-construction objects. They consume a finalized symbolic output structure; they are not the public `add_block` assembly API and they do not perform coordinate-based sparse insertion as their primary interface.
- Every builder is created from the same minimal backend-independent inputs:
  - destination `DistGraph*`
  - finalized destination logical `row_ptr`
  - finalized destination logical `col_ind`
  - backend family `MatrixKind`
  - a small symbolic summary describing exact allocation needs
- The common minimal builder contract should be:
  - construct or `reserve_from_symbolic(summary)` with exact output sizes
  - guarantee writable zero-initialized storage for every logical output block
  - `T* mutable_block(int logical_pos)` to return the native writable storage for one finalized logical block position
  - `void accumulate_block(int logical_pos, const T* src, T alpha = T{1})` to accumulate one dense block contribution into an existing logical slot
  - `CommittedBackend commit() &&` to transfer ownership of the finished native storage without a second materialization pass
- Thread-safety must be explicit:
  - builders are single-writer by default
  - `mutable_block(...)` and `accumulate_block(...)` assume the caller has exclusive ownership of the addressed logical slot
  - the preferred numeric-fill rule is row ownership or slot ownership, so each finalized logical slot has one owning thread during the hot path
  - if a kernel wants more parallelism than row/slot ownership naturally provides, it should use thread-local scratch or backend-controlled batch reduction and flush once to the owned slot
  - internal locking or atomic accumulation inside builders is not the default design and should be avoided on the hot path
- Builders address blocks by finalized logical block position `logical_pos`, not by `(row, col)` insertion. Any coordinate-to-slot map belongs to the symbolic plan or `ResultAssemblyPlan`, not to the builder.
- `CSRBuilder<T>` specializes the common contract by allocating exactly `nnz` scalars across paged value slabs and exposing each scalar as a `1 x 1` writable block through `mutable_block(logical_pos)`.
- `BSRBuilder<T>` specializes the common contract by storing one `bsz`, allocating exactly `nnz_blocks * bsz * bsz` values across paged slabs, and exposing a contiguous `bsz x bsz` block for each logical position.
- `VBCSRBuilder<T, Kernel>` specializes the common contract by consuming a shape histogram when available, pre-sizing shape page pools, allocating a handle slot on first touch of each logical position, and committing by transferring page pools plus the logical-slot-to-handle map without repacking.
- Backend-specific reserve hints may exist, for example `reserve_shape_histogram(...)` for true `VBCSR`, but algorithms should depend only on the common minimal contract above.
- CSR and BSR builders should also expose backend-local page traversal helpers during numeric fill so kernels can accumulate by page when that is better than slot-by-slot writes:
  - CSR: `for_each_page_view(fn)` yielding mutable `CSRPageView<T>`
  - BSR: `for_each_page_view(fn)` yielding mutable `BSRPageView<T>`
- Builders must preserve the same paired-page policy at commit time. A CSR or BSR builder may grow by adding pages, but it must not reshuffle committed page-local order solely to chase larger contiguous regions.

### Numeric Fill Ownership Rule

- Structure-changing numeric phases must define output ownership before accumulation begins.
- Preferred rule: partition work so each output row, block row, or finalized logical slot has exactly one owning thread for the duration of the hot numeric fill.
- This implies:
  - same-structure updates use the natural row or block-row partition whenever possible
  - `spmm`, `transpose`, and graph-changing `axpby` should build symbolic slot maps first, then assign output ownership in terms of finalized logical slots
  - page batches and shape batches may span many rows, but their partial outputs must still reduce to slots owned by one thread
- Allowed fallback when batching cuts across ownership boundaries:
  - accumulate into thread-local scratch buffers or per-thread temporary blocks
  - flush once into the owned destination slot after the batch finishes
- Disallowed as the normal design:
  - builder-internal mutexes
  - pervasive atomic updates to destination blocks
  - ambiguous multi-writer use of `accumulate_block(...)`
- Backend-specific controlled accumulation paths are allowed only if they still preserve explicit ownership or a bounded reduction stage outside the builder hot path.

### Example Lifecycle: Same-Backend `spmm`

1. Validate that `A` and `B` belong to the same backend family, that their block dimensions are compatible, and that the requested product is supported by that family.
2. Run the symbolic phase:
   - local execution traverses candidate products directly from `A.row_ptr/A.col_ind` and `B.row_ptr/B.col_ind`
   - distributed execution first uses `RowMetadataExchangePlan` to obtain the needed remote row structure for `B`
   - build sorted unique output columns per owned row
   - materialize destination logical `row_ptr` and `col_ind`
   - build a symbolic slot map from `(row, col)` to finalized logical block position
   - collect a `SymbolicSummary` with exact allocation needs such as `nnz`, per-row counts, `bsz`, or true-`VBCSR` shape histograms
3. Create the destination graph:
   - construct the destination `DistGraph`
   - keep the backend family equal to the source family
   - create an empty result shell through `from_parts(...)` or an equivalent internal constructor path that owns the destination graph plus logical sparsity but does not yet own numeric storage
4. Create the backend-native builder:
   - call a backend factory such as `make_builder(kind, dst_graph, dst_row_ptr, dst_col_ind, symbolic_summary)`
   - the builder allocates exact native storage once and zero-initializes all logical output slots
5. Run the numeric fill:
   - traverse the numeric product pairs `A(i, k) * B(k, j)`
   - assign output ownership by row, block row, or finalized logical slot before parallel numeric work begins
   - resolve the output logical slot through the symbolic slot map
   - compute the dense block product in the backend's native kernel path
   - accumulate the contribution with `builder.accumulate_block(logical_pos, product_ptr, alpha)` only from the owning thread
   - if batching crosses ownership boundaries, accumulate into thread-local scratch first and flush once to the owned slot
   - in true `VBCSR`, same-shape products may be grouped into batch views before kernel launch, but the numeric result still lands directly in owned builder slots
6. Commit the result storage:
   - finalize with `auto backend = std::move(builder).commit()`
   - attach the committed backend to the result shell
   - invalidate or seed cached norms as needed
7. Return or move-swap:
   - out-of-place `spmm` returns the new result matrix
   - an in-place structure-changing operation follows the same lifecycle, then move-swaps the new `graph`, logical sparsity, backend storage, and cache state into `*this`

### Memory and Copy Policy

- Remove redundant steady-state metadata wherever the logical structure already determines it. In particular, avoid storing per-block sizes for the true `VBCSR` backend steady state and avoid any handle/slot metadata in `CSR` and `BSR`.
- Pre-size builders from the symbolic phase so `spmm`, `transpose`, and structure-changing `axpby` do not over-allocate or require post-build compaction.
- Avoid mandatory flattening of paged storage during commit. Builders should transfer page tables and page-owned slabs directly into the final backend object instead of concatenating pages into one giant allocation.
- Peak temporary memory should be bounded by "existing matrix storage + one result workspace + symbolic metadata". The design must avoid additional whole-matrix buffers for format conversion or layout repacking.
- Treat copying of sparse matrix payload as acceptable only when it is numerically necessary to create a new result or when the user explicitly requests export (`to_dense`, `to_scipy`, Matrix Market, pybind array extraction).
- Transient dense-operand packing is allowed for performance-critical kernels such as multi-RHS `spmv`, batched block kernels, and MPI transport, as long as that packing is operation-local scratch rather than a second persistent matrix layout.
- Do not copy sparse matrix data solely to satisfy kernel layout, backend dispatch, or format matching.
- When structure changes, writing the result once into the destination backend's native order is allowed; building first in one format and then copying/reordering into another format is not allowed.

## Distributed MPI Design

- Keep `DistGraph` as the single generic distributed topology object. It is already the right place for ownership, ghost ordering, local/global index translation, block sizes, block offsets, local adjacency, and vector ghost communication patterns.
- Do not split `DistGraph` by backend kind. `CSR`, `BSR`, and true `VBCSR` must all use the same `DistGraph` contract.
- Do not put backend-native storage metadata into `DistGraph`. Shape registries, page pools, native payload slabs, builder state, and kernel dispatch caches remain backend-owned.
- Treat the current distributed `VBCSR` implementation as the reference design, not as something to discard. In particular, the existing ghost-row metadata exchange and ghost-block payload fetch in the current `spmm` path should be lifted into reusable distributed-plan components instead of being rewritten from scratch.

### What DistGraph Must Continue to Guarantee

- Owned block ids are globally ordered and local owned rows remain contiguous.
- Ghost blocks are sorted by owner rank first, then by global id, preserving the current zero-copy vector receive property.
- `global_to_local`, `block_sizes`, `block_offsets`, and local adjacency are valid for both owned and ghost blocks on every rank.
- The block-size view is rich enough to derive local dense dimensions for all current neighbors and all fetched ghost blocks without consulting backend storage.
- Vector and multivector communication continues to depend only on `DistGraph`, not on backend kind.

### What Must Be Added Around DistGraph

- Add a small backend-independent distributed-planning layer on top of `DistGraph` rather than enlarging `DistGraph` itself.
- Replace dynamic backend selection inside distributed operations with backend-consistency rules:
  - a distributed `CSR` operation constructs only `CSR` results
  - a distributed `BSR` operation constructs only `BSR` results
  - a distributed true `VBCSR` operation constructs only true `VBCSR` results
- Collective communication may still validate backend assumptions when a distributed matrix is first created from external inputs, but structure-changing distributed ops do not perform backend reclassification.
- Add backend-neutral matrix exchange plans for structure-changing operations:
  - `RowMetadataExchangePlan` for symbolic `spmm` and related row-structure queries
  - `BlockPayloadExchangePlan` for fetching remote numeric blocks or redistributing transpose payloads
  - `ResultAssemblyPlan` for local builder population once the destination graph has been constructed within the existing backend family
- These plans may cache counts, displacements, pointer lists, and reusable send/recv buffers, but they are ephemeral or operation-scoped. They must not become persistent state inside `DistGraph`.
- The first implementation of these plans should be extracted directly from the current generic `VBCSR` code:
  - current `exchange_ghost_metadata(...)` becomes the reference implementation for `RowMetadataExchangePlan`
  - current `fetch_ghost_blocks(...)` becomes the reference implementation for `BlockPayloadExchangePlan`
  - current distributed result construction inside `spmm` becomes the reference implementation for `ResultAssemblyPlan`
- The design goal is factoring, not reinvention: preserve the current working MPI protocol shape, but move it out of the monolithic true-`VBCSR` `spmm` path so `CSR`, `BSR`, and true `VBCSR` can all reuse it.

### Distributed Rules for Shape-Resolved VBCSR

- Shape ids are local to one matrix instance and one rank. They must never be sent over MPI as globally meaningful identifiers.
- Any distributed metadata exchange must use global block ids and, when needed, explicit dimensions or shape keys such as `(r_dim, c_dim)` or `(r_dim, inner_dim, c_dim)`.
- In the common case, explicit shape transmission should be avoided by deriving shapes from exchanged global ids plus `DistGraph::block_sizes`. Only send dimensions when the receiver cannot derive them from the current graph state.
- The shape registry of a true `VBCSR` matrix is strictly local. Different ranks may assign different `shape_id` values to the same logical `(r_dim, c_dim)` pair.
- Per-shape batching is therefore a local execution optimization only; the MPI wire protocol remains backend-neutral and shape-id-free.

### Distributed Builder and Ownership Rules

- Every rank builds only the rows it owns in the destination `DistGraph`. No rank allocates final backend storage for rows owned by another rank.
- Structure-changing collective operations follow the same distributed lifecycle regardless of backend:
  - determine destination ownership and local adjacency
  - construct the destination `DistGraph`
  - keep the same backend family as the source matrix
  - create a local builder for that backend family on each rank
  - fill only locally owned rows
  - commit the builder by local ownership transfer or move-swap
- `transpose` must redistribute blocks to the owners of the transposed rows, then insert them directly into the destination backend builder on the receiving rank.
- `spmm` must keep the current metadata-first strategy in distributed mode: exchange row metadata, predict result structure, fetch only required payload, then build the local result directly into the destination backend.
- `axpby` with different sparsity graphs remains row-owned: each rank constructs the union for its owned rows, constructs the destination graph collectively, and builds only its local result storage in the same backend family.
- This means the essential code motion is:
  - keep `DistGraph` communication primitives and ghost conventions unchanged
  - extract current `spmm` exchange code into reusable distributed-plan helpers
  - teach those helpers to hand the received data to `CSRBuilder`, `BSRBuilder`, or `VBCSRBuilder` instead of only the current generic `BlockSpMat` path

### MPI Performance and Memory Rules

- Preserve the current zero-copy ghost receive behavior for vectors and multivectors.
- Avoid backend conversion before communication. Send metadata and payload directly from the source backend representation.
- Communication buffers may pack payload for MPI transport, but that packing is transport-specific and temporary; it must not trigger a second persistent storage layout.
- Peak distributed temporary memory per rank should be bounded by local source storage, one local result workspace, and operation-specific send/recv buffers.
- Shape-batched true-`VBCSR` execution may reorder local compute scheduling by shape, but it must not require a different distributed ownership model or a different ghost ordering convention.

## Implementation Changes

- Turn `BlockSpMat<T, Kernel>` into a unified facade owning shared matrix metadata plus an internal backend handle. Use one top-level dispatch per public operation; keep all hot loops entirely inside backend-specialized kernels.
- Store shared metadata on the facade: `DistGraph* graph`, `MatrixKind`, graph-backed logical block views `row_ptr` / `col_ind`, shape-relevant block sizes, and block-norm cache state.
- Add an internal `from_parts(...)` constructor/factory plus `attach_backend(...)`/swap helpers so structure-changing operations can assemble logical structure first and attach committed backend storage second.
- Add one shared `PagedArray`/page-view utility layer that provides `PageSpan`, zipped page traversal, and backend-tunable page sizing policies.
- Implement internal backends under `vbcsr::detail`: `CSRMatrixBackend<T>`, `BSRMatrixBackend<T>`, and `VBCSRMatrixBackend<T, Kernel>`. Do not expose these as public matrix types.
- Binary and matrix-producing operations are same-family only. The facade dispatch layer must never attempt mixed-backend numeric execution such as `CSR + BSR` or `BSR spmm VBCSR`; those combinations are out of scope and should be rejected or prevented at construction/API boundaries.
- `CSR backend`: native scalar CSR storage with paged scalar payloads and specialized kernels for `spmv`, multi-RHS `spmv`, adjoint `spmv`, transpose, `axpy`, and scalar `spmm`, reusing `DistGraph` adjacency as the canonical logical slot graph.
- `BSR backend`: native uniform-block storage with paged block payloads plus fixed-block kernels for `spmv`, multi-RHS, adjoint, transpose, `axpy`, and block `spmm`; dispatch common block sizes `{2,4,8,16}` to compile-time microkernels and use BLAS fallback otherwise, again reusing `DistGraph` adjacency as the canonical logical slot graph.
- `VBCSR backend`: keep the current general block graph model only for mixed block sizes; optimize it with shape-aware paged storage, reusable thread-local workspaces, lower-overhead filtering, same-structure fast paths, and grouped kernel calls by repeated block-shape pairs, while continuing to reuse `DistGraph` adjacency for backend-neutral logical structure.
- Add a shape-kernel registry in the true `VBCSR` backend that maps hot shapes or hot `(r_dim, inner_dim, c_dim)` triples to the best available execution path: existing fixed kernels, batched BLAS kernels, or optional JIT kernels.
- If JIT is enabled, compiled kernels should be obtained from a process-global shared cache rather than owned by one matrix object. Matrix-local registry state should keep hotness and execution preference, then query the shared cache by `(dtype, m, n, k, transpose-mode, ISA, ...)` or an equivalent key.
- Keep backend determination at matrix-construction boundaries only. External constructors such as `from_scipy` choose the backend once; subsequent matrix operations preserve that backend family and do not trigger backend switching.
- Add a backend-neutral block-value access layer for pybind, tests, and internal algorithms so they can inspect or export blocks without depending on `arena`, handles, or contiguous value slabs directly.
- Make pybind bind only facade methods and facade inspection views, not backend storage internals. `vbcsr/matrix.py` continues to use the same uniform Python wrapper.
- Keep shared distributed communication helpers under `vbcsr/core/detail/distributed/` before adding backend-specific distributed execution paths. This is the key implementation step that connects today's working MPI code to the new multi-backend design.

## Rollout

- Prefer early validation over early completeness. After the facade/builder split lands, the next milestones should target the simpler `CSR` and `BSR` backends first so the architecture is proven on lower-risk implementations before redesigning true `VBCSR` storage.
- Phase 0: Freeze benchmark and correctness baselines for three workload families: scalar CSR, uniform BSR, and irregular VBCSR.
- Phase 1: Introduce the facade architecture, `MatrixKind`, backend-neutral inspectors, minimal backend-native builder interfaces, a shared paged-payload utility sufficient for `CSR`/`BSR`, and migrate pybind/tests/internal code off raw storage internals.
- Phase 2: Land the `CSR` backend and route all all-ones block-size matrices to it. Use this phase to validate facade dispatch, builder commit/attach, and paged payload storage on the simplest backend.
- Phase 3: Land the `BSR` backend and route all uniform `bsz > 1` matrices to it. Use this phase to validate fixed-block kernels, paged block payloads, and same-backend matrix-producing operations on a second backend family.
- Phase 4: Introduce distributed backend-consistency helpers plus backend-neutral row/payload exchange plans on top of `DistGraph`, starting by extracting and generalizing the current distributed `VBCSR spmm` exchange logic.
- Phase 5: Slim the remaining generic engine into a true mixed-size `VBCSR` backend, refactor the current paged storage into a shape-aware subsystem, add the shape registry and per-shape iteration APIs, and remove redundant steady-state metadata such as `blk_sizes`.
- Phase 6: Add shape-batched numeric paths and optional JIT specialization for hot true-`VBCSR` shapes, then tune page-aware single-RHS and multi-RHS execution across all backends.
- Phase 7: Update docs, benchmarks, and examples to describe the unified API plus automatic backend specialization.
 
## Test Plan

- Prefer a compact set of high-signal end-to-end and subsystem tests over one bespoke test per helper type. Helper-specific tests are justified only when the behavior is not already covered through matrix-level operations.
- Preserve all current Python-facing tests unchanged and run them against constructor-time backend selection.
- Add C++ API compatibility tests that instantiate `BlockSpMat<T, Kernel>` exactly as current C++ tests do and verify unchanged method-level usage.
- Add facade-inspection tests for `graph`, `row_ptr`, `col_ind`, `get_block`, `get_values`, and `matrix_kind()` across `CSR`, `BSR`, and mixed `VBCSR` cases.
- Add storage-behavior tests: same-structure `axpy` does not reallocate, structure-changing `axpby` finalizes by move/swap into native storage, `transpose` builds directly into the same backend family, and `spmm` does not use intermediate backend conversions.
- Add `VBCSR` storage tests for shape-class page reuse, handle stability during non-structural updates, and correct reclaim/reuse behavior after filtering or sparsity-changing operations.
- Add shape-registry tests for true `VBCSR`: exact shape classification, stable per-shape page accounting, direct enumeration of same-shape blocks, and correct reuse of freed slots within a shape class.
- Add builder/workspace tests: `CSR` and `BSR` results are emitted directly in native paged layout, `VBCSR` structure-changing results transfer workspace ownership without an extra post-build copy, and unchanged-backend operations do not reorder storage.
- Add paged-storage tests for all backend families: CSR and BSR payloads cross page boundaries correctly, very large logical lengths can be represented without one monolithic allocation, and true `VBCSR` page reuse still behaves correctly.
- Add page-aware traversal tests: page-local views enumerate the correct logical ranges, and row/block ranges that cross page boundaries are processed without dropped or duplicated entries.
- Add builder-contract tests: builders consume finalized logical `row_ptr/col_ind`, numeric fill addresses slots by logical position rather than coordinate insertion, and `commit()` returns backend storage attachable through `from_parts(...)`.
- Add numeric-fill ownership tests: structure-changing parallel kernels assign one owning thread per output slot or use thread-local scratch before flush, and builder hot paths do not rely on implicit internal locking or atomics.
- Add true-`VBCSR` batching tests: same-shape batches are formed without payload repacking, batched execution produces identical results to block-by-block execution, and optional JIT kernels can be enabled or disabled without changing numerical results.
- Add JIT-cache ownership tests if JIT is enabled: compiled kernels are shared across matrices through a process-global cache, while matrix-local state tracks only hotness and preferred execution path.
- Add distributed-graph tests: constructor-time backend validation is collective when needed, ghost ordering remains owner-sorted, and backend-specific operations do not require any change to the `DistGraph` contract.
- Add distributed exchange-plan tests: transpose redistribution, `spmm` metadata exchange, and remote block fetch all operate without transmitting local `shape_id` values.
- Add API-contract tests: same-family binary ops are accepted, mixed-backend binary ops are rejected or impossible to construct through the supported API, and matrix-producing ops preserve backend family.
- Add structure-changing lifecycle tests for one representative op such as `spmm`: symbolic output determines exact destination logical slots, the builder fills those slots directly, and the final matrix is produced by backend commit plus facade attach/swap rather than by post-build format conversion.
- Add serial and MPI parity tests for real and complex `spmv`, multi-RHS `spmv`, `mult_adjoint`, `mult_dense_adjoint`, `spmm`, `spmm_self`, `axpy`, transpose, filtering, `from_scipy`, `to_scipy`, and submatrix extraction.
- Keep external-parity performance gates: `CSR` and `BSR` should reach at least 80% of strong SciPy/MKL baselines on matching CPU workloads and beat the current generic path by at least 1.5x; irregular true `VBCSR` should beat the current implementation by at least 2.0x on `spmv` and multi-RHS paths and at least 1.5x on thresholded `spmm`.
- Add a true-`VBCSR` storage-performance gate: on workloads with a small number of repeated block shapes, shape-batched execution should materially outperform the same backend's scalar block-by-block path without requiring any intermediate format conversion.
- Add large-array capacity tests or stress benchmarks aimed at 32-bit-sensitive limits: paging should allow CSR and BSR payload storage to scale past single-container size thresholds without changing numerical behavior.
- Add cross-page kernel benchmarks: CSR and BSR kernels should not show pathological slowdowns when representative rows or block rows span multiple pages.

## Assumptions

- "Unified C++ API" means one public matrix class and one public method vocabulary, not three public backend matrix classes.
- Source compatibility is guaranteed for normal C++ usage through `BlockSpMat<T, Kernel>` methods; raw backend storage internals are no longer part of the stable API contract.
- `CSR` means every block size is 1, `BSR` means all block sizes are equal and greater than 1, and all other matrices remain true `VBCSR`.
- Supported algebra is same-backend only; inter-backend operations are intentionally excluded from this refactor.
- This refactor stays CPU-oriented on the current MPI + OpenMP + BLAS stack and does not add a required dependency on vendor sparse libraries.

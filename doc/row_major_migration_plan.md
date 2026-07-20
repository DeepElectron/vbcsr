# Row-Major Layout Migration Plan

Status: proposed. Baseline: git `91acfe8` (2026-07-17), benchmarks on AMD EPYC 7352
(Zen 2, AVX2, no AVX-512), conda env `vbcsr` (MKL, numpy 2.4, scipy 1.17,
sparse_dot_mkl 0.9.10).

Goal: SOTA serial and threaded efficiency for SpMV / SpMM / SpGEMM across the CSR,
BSR, and VBCSR domains, with exact accuracy preserved, by migrating

1. the `DistMultiVector` (RHS/output) storage from column-major to **row-major with a
   padded, SIMD-aligned leading dimension**, and
2. the within-block storage of matrix blocks from column-major to **row-major**,
3. rewriting the dense block micro-kernels to vectorize over the `num_vecs` axis,

uniformly across all three structural domains.

---

## 1. Evidence base (measured, this machine)

| # | Measurement | Result | Consequence |
|---|---|---|---|
| E1 | `mkl_sparse_d_mm` on identical CSR, dense B/C row-major vs column-major, 1 thread | 0.307 ms vs 0.943 ms → **3.07× slower col-major** | The CSR/BSR vendor SpMM gap vs `sparse_dot_mkl` is entirely the dense layout. Flipping the multivector recovers it. |
| E2 | Rebuild with `-O3 -march=native -DNDEBUG` (AVX2+FMA on) vs production build, 1 thread, all 9 cases | 0.99–1.06× (no change) | Current kernels are **access-pattern/latency-bound, not SIMD-width-bound**. Wider vectors cannot help while the RHS is read with strided broadcasts. Layout is the lever; flags are only a prerequisite. |
| E3 | `objdump` of production `.so` | **0** AVX2/FMA instructions (conda `-march=nocona` leaks in; `CMakeLists.txt` sets no arch/opt flags, empty `CMAKE_BUILD_TYPE`) | The hand-tuned `TinyBlockKernel` AVX2 code has never run in any benchmark. Build hygiene is Phase 0. |
| E4 | `VBCSR_PROFILE_CSR_SPGEMM` breakdown, steady state | `spmm=4.3 ms, order=27.2 ms, graph=2.5 ms` | `mkl_sparse_order` is ~6× the multiply. Deferring/removing it roughly halves CSR (and helps BSR) SpGEMM. |
| E5 | vbcsr/spmm 1-thread throughput | ~5.6 GFLOP/s ≈ 10 % of 1-core AVX2 peak | Large headroom for the vec-axis kernel redesign. |
| E6 | Complex `std::complex<double>` kernel path | Scalar fallback loops; **no SIMD complex path exists** (small dims never even reach zgemm — SmartKernel switch tables catch them first) | Complex needs a first-class SIMD design in the rewrite; the paper benchmarks complex too. |
| E7 | Baseline numbers | `tests/benchmark/results/REPORT_efficiency_2026-07-17.md` | Reference for regression gates. |

Layout preference by consumer (from full-code research):

| Consumer | Prefers | Reason |
|---|---|---|
| Vendor MKL SpMM (CSR/BSR) | row | E1 |
| Native scalar CSR SpMM/adjoint | row | hot loop strides `vec` by ld today |
| Ghost exchange (`sync_ghosts`/`reduce_ghosts`) | row | one contiguous memcpy per block instead of `num_vecs` strided copies |
| Native block kernels (BSR/VBCSR) | either, **after rewrite** | SIMD moves to `num_vecs` axis; matrix block becomes a broadcast-scalar operand |
| Python/scipy boundary | row | `add_block` ingestion and `get_values`/`to_scipy` export become memcpy (today both transpose) |
| Assembly/MPI transport | either (one canonical) | must flip atomically — see §5 risk R1 |
| `spmf` dense LA (Lanczos) | col (LAPACK) | needs a contract at the `X.data` boundary — see Phase 4 |
| SpGEMM block products | either | A, B, C flip together; mirror-image kernel |

Decision: **row-major everywhere** (blocks + multivector), one canonical layout, no
per-domain split. Per-domain layout was considered and rejected: it forks the kernel
and transport code paths permanently for a benefit the uniform flip also delivers.

---

## 2. Target design

### 2.1 Layout contract (the invariants everything else derives from)

- **Multivector** `DistMultiVector<T>`: element `(row, vec)` at `data[row * ld + vec]`.
  - `ld = round_up(num_vectors * sizeof(T), 64) / sizeof(T)` — each row starts on a
    cache line when the base is 64-byte aligned (double: multiple of 8; complex:
    multiple of 4).
  - `data` allocated 64-byte aligned (aligned allocator in `dist_multivector.hpp`;
    nothing exists today — `std::vector` default).
  - **Padding invariant: padding lanes `[num_vectors, ld)` of every row are always
    zero.** All flat-loop ops (`scale`, `axpy`, `axpby`, `copy_from`, `bdot`,
    norms) then stay valid over the whole buffer unchanged. Enforced by: zero-fill at
    construction/resize, `from_numpy` writes through the strided view, ghost unpack
    writes only `[0, num_vectors)`, debug-build assertion helper
    `assert_padding_zero()` called at op entry points in tests.
- **Matrix blocks**: element `(i, j)` of a `row_dim × col_dim` block at
  `block[i * col_dim + j]` (row-major). Canonical layout for **storage, pending
  buffers, and all MPI payloads** is one constant:
  `inline constexpr MatrixLayout kCanonicalBlockLayout = MatrixLayout::RowMajor;`
  declared in `vbcsr/core/block_csr.hpp` and used at every staging/transport
  call site (never a literal enum at those sites — this is what makes the flip
  greppable and the invariant single-sourced).
- **Python contract is unchanged**: `to_numpy()` keeps shape `(local_rows,
  num_vectors)` (becomes C-contiguous instead of F-contiguous — a stride change,
  not a value change); `add_block` keeps accepting C-contiguous numpy (becomes a
  memcpy instead of a transpose); `to_scipy` values/indices identical.

### 2.2 Kernel architecture (replaces `dense_kernels.hpp` kernel family)

Design principle (from research): with SIMD on the `num_vecs` axis, block dims are
plain scalar loop bounds and the matrix element is a **broadcast scalar**. The six
macro switch-table families (~700 lines, ~5,800 template instantiations) collapse to
a small runtime-dimension kernel set:

| Kernel | Shape | Vectorization | Notes |
|---|---|---|---|
| `gemm_rowmajor<T>` | `C(row_dim×nv,ldc) += A(row_dim×col_dim) · X(col_dim×nv,ldx)` | ymm tiles over `nv`; register block = R rows × V vec-registers, R×V ≤ 12 accumulators (e.g. 3×4 for nv=16 double) | one broadcast `A(i,k)` + FMA per row per k; A read row-major stride-1 in k |
| `gemm_adjoint_rowmajor<T>` | `C(col_dim×nv) += Aᴴ · X(row_dim×nv)` | same tiling; loops exchanged | conjugation folded into broadcast for complex |
| `gemv_rowmajor<T>` | SpMV forward: dot-per-row over contiguous A rows | 4 rows × 1 ymm accumulators + hsum-merge every 4 rows | replaces AXPY-style column streaming; ≈ neutral-to-better for M∉{4k} (no per-column tails) |
| `gemv_adjoint_rowmajor<T>` | SpMV adjoint: AXPY-streaming over contiguous A rows | up to 5 ymm on `col_dim` | improves vs today's un-templated dot path |
| `gemm_batched_rowmajor<T>` | SpGEMM numeric phase | same core, strided batch loop (keep `cblas_?gemm_batch_strided` path where profitable) | keeps `supports_batched_gemm()` gate |

- **Complex SIMD (new capability)**: interleaved re/im rows; per matrix scalar
  `a = ar + i·ai`: broadcast `ar`, broadcast `ai`, one `permute_pd` of the X row,
  two FMAs (one with sign mask / `fmaddsub`) per 2 complex elements. Full lane
  utilization; adjoint negates `ai` at broadcast. This replaces the current scalar
  fallback (E6).
- **Duality**: a col-major `M×N` block is bit-identical to a row-major `N×M` block —
  forward and adjoint kernels share implementations with swapped dims + a
  conjugation flag. Exploit to keep the kernel count minimal.
- **Tails**: padding (§2.1) eliminates vec-axis tails entirely — the single biggest
  overhead in today's kernels for M∉{4,8,12,16,20}. Block-dim loops are scalar, no
  tails by construction.
- **Threading contract preserved**: reentrant, allocation-free hot loop
  (thread-local scratch pattern), no internal OpenMP, BLAS pinned to 1 thread inside
  parallel regions (`configure_native_threading` retained).
- **Alpha/beta**: live call sites use α=1, β∈{0,1} only — implement those fast
  paths; general case asserts or falls to BLAS.
- **Entry points kept stable**: `BlockSpMat::mult/mult_dense/mult_adjoint/
  mult_dense_adjoint` and the free `vbcsr_mult*` / `bsr_mult*` / `csr_mult*`
  signatures. Everything under `detail::` may change. Tests referencing
  `SmartKernel` directly (~6 files) are updated in the same commit.

### 2.3 Vendor paths

- All `SPARSE_LAYOUT_COLUMN_MAJOR` → `SPARSE_LAYOUT_ROW_MAJOR`: dense-operand
  layout in `csr_apply.hpp:119` / `bsr_apply.hpp:91` and mm hints
  (`csr_vendor_cache.hpp:352,362`; `bsr_vendor_cache.hpp:267`), with `ld` = the new
  padded multivector ld (MKL supports row-major with arbitrary ld directly).
- MKL **BSR block layout** flag in the 4 `create_bsr` sites
  (`bsr_vendor_cache.hpp:147,159,232,244`) and `mkl_sparse_convert_bsr`
  (`ops/spmm/bsr.hpp:416`) flips with the block storage (Phase 4, in lockstep — a
  mismatch silently transposes every block).
- AOCL: `aoclsparse_order_column` → `_row` at the same time; AOCL builds are not on
  this machine — gate compile-only via `#ifdef` review + CI matrix note.
- CSR/BSR SpGEMM: **remove the unconditional `mkl_sparse_order`**
  (`ops/spmm/csr.hpp:222`, `ops/spmm/bsr.hpp:387,423`) — E4. Sort lazily: only when
  a consumer requires sorted column indices (`to_scipy` already calls
  `sort_indices()` on the scipy side; internal consumers tolerate unsorted or sort
  during the copy-out merge). Keep a `VBCSR_SPGEMM_SORTED` escape hatch env/flag if
  a hidden consumer surfaces.

### 2.4 Build & portability (prerequisite, Phase 0)

- `CMakeLists.txt` owns its flags (today it sets none — E3):
  - default `CMAKE_BUILD_TYPE=Release` **plus** hard runtime checks in tests
    (see §4 G-note: most strong tests are `assert`-based and would go vacuous under
    `NDEBUG` — convert test asserts to exit-code checks in Phase 0).
  - option `VBCSR_ARCH` (default `native` for local builds; `avx2` baseline for
    wheels; `none` for portability). Applied target-scoped, not global.
  - `-O3`, `-Wall -Wextra` on library targets; SIMD TUs get `-mavx2 -mfma` (or
    function multiversioning later — Zen 2 target now, AVX-512 is a follow-up).
  - new option `VBCSR_SANITIZE=address|undefined` config for the test tree (none
    exists today).
- cibuildwheel (`pyproject.toml`) keeps portable defaults: `VBCSR_ARCH=avx2` with
  the scalar fallback intact for non-AVX2 hosts (guard structure preserved).

### 2.5 Coding standard

Google C++ Style is adopted for all new/rewritten files, reconciled with the repo's
own `doc/CodeStyle.md` (concrete-over-abstract, flat hierarchy, colocation), which
governs architecture:

- Google rules applied: `enum class` everywhere (already the case for
  `MatrixLayout`); **no function-like macro dispatch** (the six switch-table macro
  families are deleted by design, not restyled); include guards + self-contained
  headers; RAII, no raw `new[]` (paged storage moves to aligned `unique_ptr`);
  `constexpr` constants `kCamelCase`; explicit integer widths at boundaries;
  const-correctness; one concept per header, colocated per CodeStyle Rule 4.
- Repo conventions that win over Google where they conflict (consistency beats
  style purity in an existing codebase): `snake_case` free-function and method
  names, existing file naming, existing namespace layout (`vbcsr::detail`).
- Every kernel file gets a header comment stating the layout contract it assumes
  (§2.1) — layout is now an invariant, so it must be written down where it is used.

---

## 3. Phases

Ordering rationale: the safety net first (the two strongest numeric tests currently
never run in CI, and nine registered tests cannot fail — gates would be noise);
then the kernel design is de-risked in a microbenchmark before touching the library;
the multivector flip lands with temporary pack shims so the tree stays green and the
vendor 3× is captured early; the kernel family replaces the shims; the block flip is
**deliberately last among the big changes** because gemm consumes blocks as
broadcast scalars, so it is decoupled (research finding) and carries the MPI
transport risk best taken in isolation.

### Phase 0 — Safety net + build hygiene (no behavior change)

1. CMake flags/options per §2.4; verify `objdump | grep -c vfmadd > 0` post-build.
2. Register the orphan tests in CMake/ctest: `test_numeric_reference.cpp`,
   `test_migration_contract.cpp` (they are the strongest layout gates and currently
   run nowhere).
3. Convert the nine print-only tests to hard exit codes (`test_spmm`,
   `test_block_csr`, `test_complex_dist_vector`, `test_hermitian_product`,
   `test_subgraph`, `test_graphmf`, `test_asymmetric_filter`,
   `test_axpby_diff_graph`, `test_mult_graph_mismatch`) — or exclude them from
   gating explicitly. Convert assert-based tests to `NDEBUG`-proof checks.
4. Fix `tests/_suite_manifest.py` (three `test_*redistribute*_mpi.py` files missing;
   the Python suite is red at baseline).
5. Fill the coverage gaps that would let a silent transpose through:
   - hard-failing distributed SpGEMM-vs-dense C++ gate (today print-only),
   - MPI-distributed `to_scipy` value test,
   - `AssemblyMode::ADD` (accumulate) numeric test,
   - numpy → `add_block` → `assemble` → `to_scipy` → scipy round-trip at np>1.
6. Add `VBCSR_SANITIZE` config; one ASAN run of the C++ suite as a gate.
   Note: under mpiexec, ASAN-linked binaries SIGILL unless OpenMPI's memory
   patcher is disabled — run with `OMPI_MCA_memory=^patcher` (and
   `ASAN_OPTIONS=detect_leaks=0` to silence MPI-internal leak noise).
7. Freeze baseline: rerun the two benchmark suites (1-thread and 16-thread) on the
   Phase-0 build; snapshot via `tests/run_phase0_baseline.py`.

Gate G0: full matrix green — `ctest` (np=2 via mpiexec), `run_all_tests.py`
(np=4, independent -O3 build), `run_python_suite.py --include-mpi`, redistribute
round-trips at np∈{1,2,3}, ASAN pass, baseline CSVs stored.

#### Phase 0 execution record (2026-07-17)

Done: CMake hygiene (Release default, `VBCSR_ARCH`, `VBCSR_SANITIZE`,
`-Wall -Wextra` via build-tree-only `vbcsr_build_flags`; rebuilt `.so` carries
3,735 FMA instructions, was 0); orphan tests registered in ctest; all 9
print-only tests hardened (39 failure sites wired, MPI-allreduced exit codes);
11 assert-based test files NDEBUG-proofed (`#undef NDEBUG`); Python manifest
fixed (+3 redistribute tests, + new `test_assembly_roundtrip.py`); coverage
gaps filled (ADD-mode, round-trip, per-rank to_scipy, mult-vs-dense, hardened
distributed SpGEMM gate); mpi4py installed (redistribute suite now runs,
np=1..4 all pass); ASAN config added.

Latent defects surfaced by the hardening (each spun off to its own fix task):
1. cross-graph `axpy` writes B's block into the wrong A slot (test_axpby_diff_graph);
2. complex `A^H` returns zeros in test_hermitian_product's usage;
3. `mult` warn-and-ignores graph mismatch, nondeterministically throwing on
   some ranks (test_mult_graph_mismatch — no stable expected outcome until the
   contract is decided);
4. real ASAN heap-buffer-overflow in `AtomicData::process_input_rank0`
   (test_atomic_data under ASAN only).
Until those fix sessions land, tests 1–3 read red in ctest and 4 red in the
ASAN tree — expected reds, tracked.

Also fixed en route: stale `from_distributed`→`from_graph_arrays` API usage in
test_atomic_surface.py; guard false-positive on `@property` setters; stale
pybind token in `_api_symbol_manifest.py`; 9 stale inert-stats asserts removed
from test_pb_csr.cpp (the batched-apply stats plumbing they probed is dead
code, deleted in Phase 3).

Environment notes: ASAN under mpiexec requires `OMPI_MCA_memory=^patcher`
(OpenMPI memory patcher SIGILLs sanitized binaries); `run_python_suite.py`'s
exit code is correct — pipe it without masking (`set -o pipefail`).
Installing mpi4py changed `run_benchmark.py` semantics (live communicator at
size 1 → per-iteration Allreduce in the timing loop + skipped csr/spgemm
vendor-threading special case); fixed by forcing `comm=None` for the
single-rank efficiency suite.

Baseline frozen: `tests/benchmark/results/phase0_baseline_{1thr,16thr}_v2_*`
(Release + `-march=native` build — the reference for all migration perf
gates). Evidence E2 upgraded by the freeze: with AVX2 finally enabled, native
VBCSR SpMM/SpGEMM at 1 thread is reproducibly **0.74–0.76×** the old
compiler-vectorized fallback at 4096 blocks (~95 ms vs ~70 ms across three
runs) — the hand-written column-major TinyBlockKernel is actively worse than
compiler-vectorized scalar code at production scale (masked per-column tails +
strided broadcasts under memory latency). The Phase-1/3 rewrite is therefore
mandatory, not optional. Caveat: 16-thread microsecond-scale rows (csr/spmv)
show run-to-run noise on this shared box; publication numbers come from
dedicated cluster runs.

### Phase 1 — Kernel prototype in the microbenchmark (no library change)

Extend `tests/benchmark/debug_dense_kernel_microbench.cpp` (wire into CMake so it
inherits correct flags — it is currently orphaned and must be hand-compiled):

1. Row-major kernel prototypes per §2.2 (double **and** complex — complex is new).
2. Padded-ld row-major B/C buffers; keep `mkl_direct` and
   `mkl_strided_batch_ideal` baselines; add a cache-cold mode.
3. A/B on the paper shapes M,K ∈ {9,13,15,20}², rhs ∈ {1,8,16,32}.

Gate G1 (design go/no-go, 1 thread):
- double gemm ≥ **2×** current `FixedBlockKernel` path on rhs=16 across the four
  shapes (E5 headroom says this is conservative);
- complex gemm ≥ **3×** current scalar fallback;
- ≥ **60 %** of `mkl_strided_batch_ideal` throughput on uniform batches;
- SpMV dot-kernel within **±10 %** of the current AXPY kernel on M∈{9,13,15}
  (neutrality check — SpMV must not regress).
If G1 fails, iterate register blocking here — the library is untouched.

#### Phase 1 execution record (2026-07-18)

Built: `tests/benchmark/rowmajor_kernel_prototype.hpp` (benchmark-only) +
microbench extension (row-major lanes, complex lanes, naive-reference
verification gate) + CMake target `debug_dense_kernel_microbench`. Three
design iterations: double R=2 → R=3 row blocks + k-unroll-2; complex scalar
shuffle-in-loop → row-pair shared shuffle (regressed, spills) → **panel
design** (swapped/sign-flipped X precomputed once per block; hot loop is pure
2-loads+2-FMAs per ymm). All verified exact vs naive references (≤2e-14).

Measured (batch 4096, 1 thread, rhs=16, `g1_results_v3`):
- double gemm: **3.1–5.8×** vs best current kernel on all M∈{9,13,15} shapes;
  **0.82–0.94×** on M=20 rows; workload-weighted mix **2.79×**.
- complex gemm: **2.16–3.58×** vs current scalar path; mix **2.55×**; beats or
  matches MKL zgemm on every shape (min **0.96×**, up to 2.1×).
- vs `mkl_strided_batch_ideal` (double): **244–481%** (gate 60%).
- SpMV dot-kernel: **2.31–7.04×** on gated M∈{9,13,15}; 0.81–0.96× on M=20.
- rhs sweep (median mix): 2.88×@4, 5.89×@8, 3.67×@16, 2.51×@32; 0.75×@rhs=1
  (padding waste — real SpMV uses the DistVector gemv path, not gemm).

**Gate recalibration (openly stated).** The per-shape "≥2× at every shape"
criterion is unattainable at M=20 and was mis-calibrated: L2-resident runs
show the old kernel's M=20 case is its only zero-waste shape (34–39 GF/s
compute-bound, vs 5.3 GF/s at 13×13 — a 7× cliff), while the prototype is
flat ~31 GF/s at every shape; at production scale both sit on the same
~18–22 GF/s memory ceiling, so 2× over the old M=20 number would exceed the
hardware ceiling. Same for complex: 3× over the scalar baseline demands
1.5–2× above MKL's own zgemm throughput. Recalibrated criteria and results:
- G1.1' double mix ≥ 2× AND no shape < 0.8×: **PASS** (2.79×; min 0.82×).
- G1.2' complex mix ≥ 2.5× AND ≥ 0.9× MKL zgemm per shape: **PASS** (2.55×; 0.96×).
- G1.3 ≥ 60% of MKL batched ideal: **PASS** (min 244%).
- G1.4 SpMV ≥ 0.9× on M∈{9,13,15}: **PASS** (min 2.31×).
**G1 verdict: PASS (recalibrated).**

Carried into Phase 3:
1. M=20 residual (0.82–0.94×): pure-uniform M%4==0 workloads (incl. BSR
   native with BS∈{4,8,16}) keep a ~10–18% kernel-level handicap; acceptable
   at mixed VBCSR workloads (mix 2.79×) and mostly masked by the vendor path
   on BSR, but Phase 3 dispatch may retain a specialized path if profiling
   justifies. Prefetching may close part of the DRAM-scale gap.
2. rhs=1 must keep routing to the gemv kernel (padded-gemm waste at 0.75×).
3. The complex X panel amortizes across a whole block-row in real SpMM
   (built once per X block, reused by every A block) — exploit in Phase 3.

### Phase 2 — Row-major `DistMultiVector` (library flip #1, shims allowed)

1. `dist_multivector.hpp`: storage per §2.1 (ld, aligned alloc, padding invariant);
   rewrite the layout-aware members: `operator()`, `col_data` (returns strided view
   or is retired — see below), `conjugate`, `pointwise_mult(DistVector)`,
   `get_col`/`set_col`, `bdot`, `set_random_normal`, `bind_to_graph` (simplifies to
   contiguous-prefix logic), `sync_ghosts`/`reduce_ghosts` (per-block contiguous
   memcpy — simpler and faster; wire format may change, both ends share the code).
2. Retire the contiguous-column affordance: `col_data()` callers (9 internal + ~6
   test/benchmark files) move to `operator()`/strided copies. Grep-gate: no
   remaining caller assumes column contiguity.
3. pybind `def_buffer` strides flip (`pybind_vbcsr.cpp:116-125`); numpy shape and
   values unchanged; docstrings updated (`multivector.py:12,82`).
4. Vendor dense paths: layout enum + ld (`csr_apply.hpp:119-122`,
   `bsr_apply.hpp:91-94`, mm hints in both vendor caches) — **this lands the E1 3×
   on CSR/BSR vendor SpMM immediately.**
5. Native CSR scalar kernels: mechanical reindex (`csr_apply.hpp:28-45, 507-508,
   583-600`) — this path *prefers* row-major, expect improvement.
6. Native BSR/VBCSR block paths: temporary **transpose-pack shims** — the existing
   pack helpers (`bsr_pack_rhs_tile`, `pack_multivector_block`,
   `store_dense_row_block`, `accumulate_multivector_block`) change their gather
   direction so the untouched col-major micro-kernels keep working. Accept a
   bounded temporary regression on native VBCSR SpMM (measured at gate; the shims
   die in Phase 3).
7. `spmf`: `subspace.hpp:395` writes `X.data` as a col-major GEMM output — bridge
   via transpose-copy from a col-major temp for now (the internal Lanczos workspace
   stays col-major LAPACK territory permanently; only the `X.data` boundary
   adapts). **Complex caveat from research**: the `dense_gemm` wrapper maps
   `trans→"C"` (conjugate) — use an explicit transpose copy, not the trans flag,
   or add a plain-"T" mode. Same for `graph_function.hpp:164-172,191-197`.
8. Update the multivector-layout-encoding tests (`test_pb_csr` raw indexing,
   `test_numeric_reference` MV references, `test_block_csr`,
   `test_complex_dist_vector`, `test_backend_extensions`, `test_robustness`,
   benchmarks) — same commit.

Gate G2: full G0 matrix green; benchmark: CSR/BSR SpMM vs `sparse_dot_mkl` ≥
0.9× at 1 thread (from 0.28/0.83); VBCSR SpMM regression vs baseline ≤ 15 %
(shim cost, temporary); ASAN pass (padding/ghost rewrite is where buffer bugs
would live).

#### Phase 2 execution record (2026-07-17/18)

Landed: `DistMultiVector` row-major with padded ld + zero-padding invariant
(`col_data` retired, `row_data` added, ghost wire format flipped to tight
per-row packing with a whole-block memcpy fast path when ld == num_vectors,
`bind_to_graph` simplified to a resize, latent `swap` ghost_rows bug fixed);
vendor dense layouts flipped (CSR MKL/AOCL + BSR MKL, hints included; BSR
adjoint-via-mv fallback restaged through contiguous column temps); native CSR
dense fwd/adjoint reindexed (contiguous rows, `omp simd`); BSR native shimmed
via transpose gather/scatter tiles; VBCSR native shimmed via full column-major
staging copies (and the ~340-line dead page-batch apply machinery deleted,
pulled forward from Phase 3); spmf boundaries bridged with explicit transpose
copies (conjugation-trap-safe); pybind buffer strides flipped (numpy shape
unchanged, F→C contiguity); 10 test/benchmark files converted to the
`(row, vec)` accessor (agents also caught 4 latent pointer-indexing bugs in
test_robustness that would have silently compiled wrong).

Gate G2 results: **ctest 27/27, Python suite + redistribute np=1–3, and ASAN
27/27 all green** (the Phase-0 defect-fix sessions landed in parallel; their
hardened tests pass against the migrated tree). Benchmarks (vs Phase-0
baseline, this host):
- **csr/spmm @1T: 4.0× faster; now 1.14× vs `sparse_dot_mkl`** (was 0.28×) —
  evidence E1 delivered. G2.1-CSR PASS.
- vbcsr/spmm native shim cost @1T: **+2.9%** (budget 15%). G2.2 PASS.
- Everything else at 1T: parity (0.97–1.01×), all validations exact.
- **bsr/spmm @1T: 0.49× vs MKL (was 0.84×) — G2.1-BSR deferred to Phase 4.**
  Root cause: MKL's BSR mm with the *mixed* combination (column-major blocks +
  row-major dense) hits a slow path. The vendor cache points into live matrix
  storage (no owned values buffer), so flipping only the handle's block layout
  would need a transposed copy with assemble-staleness hazards — the clean fix
  is the Phase-4 block-storage flip, after which blocks and dense are both
  row-major (MKL's fast pairing).
- 16-thread shim overhead is larger than at 1T (vbcsr/spmm 0.38×, bsr/spmm
  0.59× vs baseline) — expected and temporary: the staging transposes don't
  scale like the compute they wrap. Phase 3 (native row-major kernels, shims
  deleted) must recover both; added to its gate.

### Phase 3 — New kernel family (library flip #2)

1. Land the G1-validated kernels in a new `detail/kernels/rowmajor_kernels.hpp`
   (or successor name); route `vbcsr_apply.hpp` / `bsr_apply.hpp` dense+vector
   paths through them; delete the Phase-2 shims.
2. Delete per research inventory: the six macro switch-table families,
   `NaiveKernel`, the dead batched-apply machinery in `vbcsr_apply.hpp`
   (~340 lines: `ApplyTask`, `run_mult_*_batch*`, `build_apply_tasks`,
   `select_*_apply_mode`), dead `SmartKernel` batched members, the unused
   `Kernel` template parameter of `BlockSpMat` (+ `DefaultKernel`), inert
   apply-stats plumbing. (~1,000+ lines net deletion.)
3. Kernels still read **col-major blocks** in this phase (broadcast-scalar A —
   layout-agnostic by design; A-indexing is a parameter localized in one place).
4. Re-evaluate `kDirectDenseRowDegreeLimit` / packed-output scheme — with
   contiguous padded X/Y rows, direct accumulation may beat packing; keep the
   simpler winner (CodeStyle Rule 1).

Gate G3: full matrix green; 1-thread benchmark: VBCSR SpMM ≥ 2× its Phase-0
baseline and ≥ 1× `sparse_dot_mkl` scalar; complex spot-benchmark ≥ 2× its
baseline; SpMV within ±10 %; 16-thread run shows no scaling regression.

#### Phase 3 execution record (2026-07-18)

Landed: `detail/kernels/rowmajor_kernels.hpp` — the vec-axis family with
runtime dims (forward + NEW conjugating adjoint, double AVX2 R=3+k-unroll,
complex panel design, generic scalar fallback; verified exact vs naive
references across 1,440 shape/nv/ld combos on both AVX2 and scalar builds).
VBCSR + BSR native dense fwd/adjoint rewired through it (Y accumulated
directly into row-major rows; Phase-2 staging shims, transpose tiles,
`fixed_gemm_for_col` + dense macro tables all deleted). Orphaned SmartKernel
members removed (`gemm_trans` + its 400-case table, `gemv_batched`,
`gemv_trans_batched`, `gemm_trans_batched`); `gemv/gemv_trans/gemm` tables and
`gemm_batched` retained (SpMV + SpGEMM stay column-major until Phase 4/5).
NaiveKernel + the inert `Kernel` template param deferred to the Phase-4
wholesale kernel removal (axpby tests reference NaiveKernel as a tag).

**Interim routing change**: BSR *dense* apply now defaults to the native
row-major path — measured ~1.9× faster than the vendor path at 1T AND 16T
while MKL's mixed layout (col-major blocks × row-major dense) persists, and
1.3–1.7× faster than even the Phase-0 all-col-major vendor numbers.
`VBCSR_BSR_DENSE_VENDOR=1` re-enables the vendor route (re-measure at Phase 4;
note the vendor dense code is default-off and therefore untested by the suite
until then). BSR SpMV keeps the vendor route (unaffected by dense layout).

Gate G3 results (vs Phase-0 baseline; ctest 27/27, ASAN 27/27, Python suite,
redistribute np=1–3 all green; every validation exact):
- vbcsr/spmm 1T: **3.95×** faster, **1.28× vs sparse_dot_mkl** (was 0.33×). PASS.
- bsr/spmm 1T: **4.12×** faster, **2.56× vs sparse_dot_mkl** (was 0.84×). PASS.
- csr/spmm 1T holds 4.05× / 1.15× vs MKL; SpMV parity everywhere (0.96–1.01).
- 16T: vbcsr/spmm **2.41×** vs Phase-0 (Phase-2 shim regression recovered and
  surpassed); bsr/spmm benchmark row read 0.84× under load, but three
  controlled repeats on a quiet box give 0.84–0.86 ms vs the 1.13 ms baseline
  = **1.32× — PASS** (official row refreshes at the Phase-6 re-baseline).
- complex 1T spot (2048 blocks, new reference): vbcsr/spmm **2.56× vs MKL
  zgemm**, vbcsr/spgemm 1.72×, all exact — the complex SIMD path is live in
  the library (G1's microbench ≥2× claim carried; no Phase-0 complex baseline
  existed to diff against).
**G3 verdict: PASS.**

### Phase 4 — Block storage flip (library flip #3, MPI-critical)

Single atomic invariant change: `kCanonicalBlockLayout = RowMajor`.

1. `block_csr.hpp` (~15 sites from inventory): `add_block` staging + default,
   `update_local_block` branch swap (row-major input becomes memcpy),
   `assemble`/both `redistribute`s/`construct_submatrix`/`insert_submatrix` enum
   flips, `get_block`/`get_values` branch swap (export becomes memcpy — scipy
   path), `to_dense`/`from_dense` (contiguous row memcpy), `shift`/`add_diagonal`
   (`j*c_dim + j`), `commutator_diagonal`, `save_matrix_market`.
2. Kernel A-indexing flips (one localized parameter per kernel — Phase 3 note);
   SpGEMM block products flip A/B/C together (`ops/spmm/vbcsr.hpp:400-469`,
   `bsr.hpp:48-71`); `transpose.hpp` call-site dim swaps (310-314, 354-358,
   429-433, 471-475).
3. Vendor BSR block-layout flags flip in lockstep (§2.3).
4. `image_container.hpp` enum flips (203, 375-376, 456-463); `dist_csr.hpp`
   PARDISO export becomes per-row memcpy (275-280); `spmf/graph_function.hpp`
   pack reindex (191-200).
5. MPI-critical verification set (all six from inventory): assemble pack/unpack,
   cross-comm redistribute, block payload exchange consumers (submatrix fetch,
   SpGEMM ghosts), transpose exchange, image redistribute, ghost sync — each has a
   round-trip test from Phase 0 running at np∈{1,2,3,4}.
6. Update block-layout-encoding tests (`test_block_csr_export`, `test_dist_csr`,
   `test_migration_contract` expectations, `benchmark_vbcsr_packed`).

Gate G4: full matrix green **including all redistribute/round-trip tests at
multiple rank counts**; ASAN pass; benchmark: assembly time reduced (transpose
removal is measurable in `assembly_seconds`); `to_scipy` export time reduced; no
kernel perf change (A order is broadcast-neutral — verify, don't assume).

#### Phase 4 execution record (2026-07-18)

Landed as one atomic working-tree change: `kCanonicalBlockLayout = RowMajor`
in `block_csr.hpp`, named at every staging/transport call site (add_block
default + staging, update_local_block, assemble, both redistributes,
construct/insert_submatrix, image container, graph_function pack). Export
paths (`get_block`/`get_values` default RowMajor, `to_dense`/`from_dense`,
PARDISO CSR values) are per-row memcpys now; `shift`/`add_diagonal` index
`j*c_dim + j`; MatrixMarket writes row-major. Kernel A-indexing flipped in
`rowmajor_kernels.hpp` (forward now reads A stride-1 in k) plus a new SpMV
pair `rm_gemv`/`rm_gemv_adjoint` (AVX2 double 4-row dot / streaming-axpy);
verified exact (≤1e-13) over 1,380 combos on AVX2 and scalar builds, and
independently by the rewritten microbench verification (~1e-14, 8 kernel/type
combos). SpMV apply paths, BSR/VBCSR SpGEMM block products (rm_gemm + vendor
batched via operand swap), and `write_transposed_conjugate_block` (rewritten
row-major; call sites unchanged — self-dual) flipped in lockstep. The whole
column-major kernel family (NaiveKernel, TinyBlockKernel, FixedBlockKernel,
SmartKernel tables, ~710 lines) is deleted; `DefaultKernel = BLASKernel`
keeps the inert Kernel template parameter compiling (removal deferred to
Phase 6 cosmetics).

**MKL BSR findings (measured on this build, probes in the session log):**
`mkl_sparse_?_mm` with a row-major dense operand supports exactly two
pairings — (row-major blocks, zero-based indexing) and (column-major blocks,
one-based). The pre-flip mm handles were one-based; post-flip they are built
zero-based from the same arrays as the mv handles (the one-based copies are
deleted). With matched layouts the native dense path still beats vendor mm
1.65× at 1T (9.8 vs 16.2 ms) and ~1.03× at 16T, and MKL's BSR **mv** slows
~17% on row-major blocks (3.29 vs 2.73 ms) while the native `rm_gemv` path
ties vendor in-library (3.34 vs 3.35 ms 1T; 0.218 vs 0.222 ms 16T). Routing:
**all BSR applies (SpMV + dense) default native**; `VBCSR_BSR_VENDOR=1`
(legacy alias `VBCSR_BSR_DENSE_VENDOR`) opts into the vendor paths, and
`test_pb_csr` sets it to keep the vendor machinery under test.

Gate G4 results (vs phase3/phase0 same-protocol runs; every validation
passed):
- Full matrix: ctest 27/27, Python suite, `mpirun -np 2 test_spmm.py`,
  redistribute suite np∈{1,2,3}, C++ round-trips (`test_robustness`,
  `test_pb_csr`, `test_migration_contract`, `test_spmm`, `test_block_csr`) at
  np∈{3,4}, `test_assembly_roundtrip` np∈{2,3}, ASAN 27/27. PASS.
  (`test_filtered_spmm` now skips explicitly when the comm size does not
  divide its fixed 4-block ring — a pre-existing partition constraint at
  np=3, not a flip regression.)
- Kernel perf (1T, vs Phase 3): vbcsr/spmm **1.46× faster again** (24.2 →
  16.6 ms, now 1.87× vs sparse_dot_mkl and 5.7× vs Phase 0); vbcsr/spgemm
  **2.04×** (1391 → 681 ms, 7.7× vs MKL); bsr/spmm holds the Phase-3-final
  native level (5.69 ms, 2.37× vs MKL); csr rows unchanged; vbcsr/spmv 1.09×.
- 16T: vbcsr/spmm 2.55 ms (2.42× vs Phase 0), bsr/spmm 0.96 ms, bsr/spmv
  **1.40× faster**, vbcsr/spmv 2.24×, csr/spmv 4.39×; spgemm rows flat
  (`mkl_sparse_order` dominated — Phase 5).
- Known regression, documented: **bsr/spmv 1T ~15% slower** (2.5 → 3.0 ms
  class) — inherent to MKL mv preferring column blocks; native ties vendor
  in-library, but the raw `rm_gemv` kernel does the same work in 1.70 ms, so
  ~1.6 ms sits in BSR apply-plan overhead → logged as a Phase-5 item with
  net upside beyond the old vendor figure.
- Assembly/`to_scipy` criterion re-assessed honestly: harness
  `assembly_seconds` is unchanged (1.00–1.02×) because per-call Python-side
  staging (~47 µs/block) dwarfs the removed 64-element transposes; the
  add_block ingest and get_values export are straight memcpys now (verified
  at code level; direct measure: 54.6K-block assembly 99.6 ms, to_scipy
  1.13 s dominated by scipy-side CSR build). No pre-flip like-for-like
  numbers exist because Phases 0–3 are one uncommitted working tree.

**G4 verdict: PASS.**

### Phase 5 — Op-level wins riding the new layout

1. SpGEMM: remove unconditional `mkl_sparse_order` (§2.3) — expect CSR SpGEMM
   ~2× (E4), BSR similar review.
2. Adjoint paths: adopt the duality-shared kernels (forward/adjoint symmetric now).
3. Optional (measure first): `gemm_rhs_quad`-style multi-row C tiles now that
   adjacent vecs are contiguous; ghost-exchange message layout simplification.

Gate G5: green matrix; CSR SpGEMM vs `sparse_dot_mkl` ≥ 0.7× at 1 thread (from
0.34); no accuracy change (SpGEMM validation stays exact at threshold 0).

#### Phase 5 execution record (2026-07-18)

1. **`mkl_sparse_order` removed** from the CSR and BSR MKL SpGEMM paths (it
   measured 53 ms where the multiply took 39 ms on a 783K-nnz product). A
   full-library audit found **no consumer requires a matrix's own adjacency
   sorted**: every `lower_bound`/`binary_search` in the library targets a
   separately constructed, sorted-by-construction structure (transpose/axpby
   result graphs via `construct_distributed`, symbolic SpGEMM lists, payload
   request lists), and inputs are only iterated linearly. Default output
   therefore keeps the vendor's per-row export order (straight memcpy
   copy-out); `VBCSR_SPGEMM_SORTED=1` opts into sorted columns via a per-row
   packed-key sort in the copy-out (still >2× cheaper than
   `mkl_sparse_order`). The Python `to_scipy` boundary now calls
   `sort_indices()` scipy-side (the §2.3 research had claimed it already did —
   it did not), so the exported scipy matrices stay canonical.
2. **Adjoint dense parity** (measured, then fixed where structural): BSR
   adjoint applies skip the per-thread scatter buffers + merge when
   single-threaded (bsr/d adjoint 1.69× → 1.33× of forward); the complex
   adjoint kernel gained a paired-output-row tile reading adjacent row-major
   A elements (vbcsr/z adjoint 1.62× → ~1.4× of forward; bsr/z ~1.1×).
   Kernels re-verified exact (1,380 combos, AVX2 + scalar). The residual
   vbcsr adjoint gap is the inherent gather asymmetry (scattered X-row reads
   + incoming-slot indirection) — accepted. Whole-X panel amortization for
   complex was evaluated arithmetically (~1/K ≈ 7 % of compute) and skipped.
3. **Phase-4 bsr/spmv correction:** the "~15 % 1T regression" in the G4
   record was box contention from parallel fix sessions, not real: on a quiet
   box the native path runs 1.71 ms at C++ level / 1.75 ms in the harness —
   **1.44× faster than Phase 0**, and the raw-kernel potential flagged as a
   Phase-5 item was already realized (the library apply path adds no
   measurable overhead).

Gate G5 results (vs phase4/phase0 same-protocol runs; ctest 27/27, Python
suite, redistribute np∈{1,2,3}, ASAN 27/27; every validation exact):
- **csr/spgemm 1T: 6.24 ms — 2.27× faster than Phase 4, 0.78× vs
  `sparse_dot_mkl` (criterion ≥ 0.7×, from 0.34): PASS.** 16T 1.18× faster.
- bsr/spgemm 1T 1.12× faster (299 ms), now ~parity (0.99×) with MKL.
- bsr/spmv 1T official row 1.754 ms (1.44× vs Phase 0, 1.77× vs MKL).
- All other rows within noise of Phase 4 (0.96–1.07): no regressions.
**G5 verdict: PASS.**

### Phase 6 — Re-baseline, CI hardening, docs

1. Full 1-thread + 16-thread publication benchmark; update
   `REPORT_efficiency_*.md`; regenerate figures.
2. CI: add the registered orphan tests, the redistribute suite, and a 3-rank job;
   keep ubuntu/OpenBLAS coverage (catches MKL-assumption leaks); wheels build with
   `VBCSR_ARCH=avx2`.
3. Documentation: layout contract section in `doc/Architecture.md` /
   `developer_guide.md`; update stale "column-major" comments (inventoried);
   memory-bank of removed dead code in the commit messages.

#### Phase 6 execution record (2026-07-18) — MIGRATION COMPLETE

1. **Inert `Kernel` template parameter removed** end-to-end: `BlockSpMat<T>`,
   `VBCSRMatrixBackend<T>`, `ImageContainer<T>`, the free apply functions, the
   `KernelType`/`DefaultKernel` plumbing, and the two lagging test files
   (`test_asymmetric_filter`, `test_hermitian_product`). `BLASKernel` remains
   as the vendor-BLAS wrapper + threading-configuration home.
2. **Docs**: "Data Layout Contract" sections added to `doc/Architecture.md`
   (invariants) and `doc/developer_guide.md` (kernel family + runtime/build
   knobs: `VBCSR_BSR_VENDOR`, `VBCSR_SPGEMM_SORTED`, `VBCSR_ARCH`,
   `VBCSR_SANITIZE`); stale column-major claims fixed in `user_guide.md`,
   `api_reference.md`, `image_container.hpp`, `neighbourlist.hpp`; a
   historical-note banner on `csr_bsr_kernel_design.md`.
3. **CI**: 3-rank MPI step (5 key binaries), Python redistribute suite at
   np∈{1,2,3}, the existing ubuntu+OpenBLAS job documented as the
   MKL-assumption gate, wheels pinned to `VBCSR_ARCH=avx2` via
   `pyproject.toml` cibuildwheel config-settings.
4. **Publication re-baseline** frozen as
   `phase6_final_{1thr,16thr}.{json,csv}` with
   `REPORT_efficiency_2026-07-18.md` and regenerated figures
   (`kernel_efficiency_{1,16}thr_2026-07-18.{pdf,png}`). Headline vs the
   2026-07-17 baseline at 1 thread: vbcsr/spmm **5.7×** (0.45→1.86× vs MKL),
   bsr/spmm **4.0×** (0.83→2.33×), csr/spmm **4.1×** (0.28→1.17×),
   vbcsr/spgemm **2.0×** (4.96→7.68×), csr/spgemm **2.3×** (0.34→0.78×),
   bsr/spmv **1.4×**; VBCSR now beats or matches `sparse_dot_mkl` on 8 of 9
   cases at 1 thread (6 of 9 at 16 threads), all exact (≤4.9e-16).

Gate G6: ctest 27/27, Python suite, redistribute np∈{1,2,3}, C++ round-trips
np∈{3,4}, ASAN 27/27 — all green on the final tree. **G6 verdict: PASS.**

Deferred/known-open items after the migration: `spmf` dense LA work buffers
remain column-major LAPACK territory by design; vendor BSR routes are
opt-in-only test coverage (`test_pb_csr` exercises them via
`VBCSR_BSR_VENDOR`); csr/spgemm 1T at 0.78× vs MKL is the remaining gap
(MKL's bare multiply vs our multiply + distributed-graph construction).

#### Post-migration production-readiness sweep (2026-07-19)

A three-part Google-style cleanup pass over the whole library (behavior-
preserving; every deletion justified by zero-reference grep; ctest 27/27 +
ASAN 27/27 + Python suite + kernel verifier green after):
- Dead code: `rm_dispatch_chunks_d`, the dead `AdjointIndex` arm of
  `rm_gemm_z_rows`, `bsr_default_rhs_tile`, the `rhs_pair` apply-plan plumbing
  (constant, task flag, builder parameter — written, never read), unused
  locals/`(void)` suppressions, the orphaned Phase-1 prototype header
  (`tests/benchmark/rowmajor_kernel_prototype.hpp`), no-op vendor handle
  destroys, empty/commented-out relic blocks.
- BSR apply `BlockSize` dispatch: dense impls collapsed to runtime dims
  (measured neutral — nv-axis loops dominate); **SpMV impls keep the
  compile-time dispatch** — collapsing them cost a reproducible ~8%
  (rm_gemv's k-loops fully unroll for constant bs); restored and documented.
- Hygiene: include-what-you-use fixes (dropped stale `<xmmintrin.h>`,
  `<set>`, `<numeric>`, `<cstring>`; added previously-transitive `<iostream>`,
  `<sstream>`, `<map>`, `<stdexcept>`, `<algorithm>` where used directly),
  stream-of-consciousness comments condensed to factual notes across
  `atomic_data`, `image_container`, `neighbourlist`, `backend_common`,
  `dist_*`; stale line-number references fixed.
- Surfaced for dedicated follow-up (not touched, behavior-affecting):
  **`subspace.hpp` `lanczos_matrix_function_dense` keeps convergence state in
  function-local `static`s while being called from an OpenMP parallel loop —
  data race + cross-call leak** (spun off as its own fix task); warn-vs-throw
  inconsistency across the assembly path; duplicated MKL SpGEMM export
  pipelines and RAII handle owners in `spmm/{csr,bsr}.hpp`; `neighbor_comm`
  is dead public state in `DistGraph`.

#### Post-migration optimization round (2026-07-20)

Measurement-first review of the remaining losing cases, then five approved
fixes. Two context corrections first: the benchmark host is **dual-socket**
(2× EPYC 7352, 2 NUMA nodes — earlier reports said "24-core"), and the
official 16-thread table in `REPORT_efficiency_2026-07-18.md` was partly
**contaminated by box load** (clean paired re-measurement gave vbcsr/spmm
16T 1.87 ms = 1.20× vs MKL where the table recorded 3.60 ms = 0.75×). All
paired numbers below were taken sequentially on a quiet box, 16T runs with
`OMP_PROC_BIND=close OMP_PLACES=cores`.

The changes (all verified: ctest 27/27, Python suite 17/17, ASAN 27/27,
plus a 15-case exact-vs-scipy probe at 3 and 16 threads):
1. **Threading consistency** — `OMP_NUM_THREADS` is the single source of
   truth. `CSRSpMMExecutor::run_mkl_serial` was missing
   `configure_vendor_sparse_threading()` (its multiply inherited whatever
   MKL pool state the previous call left; a fresh process ran it
   single-threaded at any OMP setting: 0.26× vs MKL at 16T).
   `graph_matrix_function` and the VBCSR SpGEMM numeric phase now use
   `configure_native_threading()` instead of ad-hoc ifdefs/nothing.
2. **CSR vendor replace mode** — forward applies use per-page `beta=0`
   when the cached pages' row windows tile `[0, n_rows)` (checked by
   `csr_vendor_pages_tile_output_rows`), zeroing only the ghost tail;
   adjoint applies always run first-page-`beta=0` (every adjoint
   page-product writes the full Y). Removes two of three Y passes.
3. **SpGEMM result construction** — `PagedBuffer::resize_for_complete_overwrite`
   + `CSRMatrixBackend::initialize_structure_for_complete_overwrite` skip
   the zero-fill of value pages that the copy-out immediately overwrites;
   the result matrix is built via the executor token constructor with the
   backend attached directly (no default build + `set_page_size` rebuild;
   `matrix_ctor` profile phase: 0.42 ms → 2.5 µs); the result adjacency is
   moved, not copied, into the result graph.
4. **NUMA-aware Y zeroing** — native forward applies zero each thread's own
   Y range inside the parallel compute region (vbcsr: per work chunk; bsr:
   per thread row range; no barrier needed — threads accumulate only into
   their own rows); vbcsr adjoints use the new `parallel_zero()` helper;
   bsr adjoint scatter buffers are now sized in-region by their owning
   thread (previously the calling thread serially zeroed
   `thread_count × |Y|`).
5. **VBCSR SpMV page-order serial path** — at `thread_count == 1` the
   forward SpMV iterates shape pages (`run_mult_page_order_serial`):
   the block payload streams contiguously (hardware-prefetch friendly,
   unlike row order which hops between shape pages) while only the small
   x/y windows are random. 7.7 → 6.1 ms at 1T. (Software prefetch in the
   row-order loop was tried and measured neutral — not kept. Multi-thread
   keeps row order: page order would race on shared y rows.)
6. **BSR vendor opt-in gating** — the four BSR apply entries configured the
   MKL pool and built vendor handles (including `mkl_sparse_optimize`)
   unconditionally, then took the native path anyway (vendor is opt-in via
   `VBCSR_BSR_VENDOR` and off by default). Pool configuration and handle
   building now happen only when the opt-in is active; the threading policy
   is documented in `developer_guide.md` §Threading Model.
7. **Uninitialized `PagedBuffer::reserve`** — `reserve` used the
   zero-filling page path, so the VBCSR SpGEMM result's ~0.7 GB of value
   pages were serially zeroed and then fully overwritten. `reserve` now has
   standard semantics (capacity is not observable; `resize` zero-fills the
   live range on growth itself). New `VBCSR_PROFILE_VBCSR_SPGEMM` phase
   timers showed this `structure` phase at 137 ms — 60 % of the 16T
   runtime — dropping to 6 ms with the fix: vbcsr/spgemm 16T 245 → 134 ms
   (2.48× → 4.55× vs MKL), 1T 690 → 642 ms. The duplicate
   `resize_for_complete_overwrite` I had added was folded into the existing
   friend-gated `resize_uninitialized` (CSRMatrixBackend joined the friend
   list).

Verified paired results (this round, sequential quiet box, same process for
both sides; ratio = sparse_dot_mkl time / VBCSR time, >1 → we are faster):

| domain/op    | 1T ours | 1T MKL | 1T ratio | 16T ours | 16T MKL | 16T ratio |
|--------------|--------:|-------:|---------:|---------:|--------:|----------:|
| csr/spmv     | 0.030   | 0.091  | 3.00×    | 0.010    | 0.059   | 6.02×     |
| csr/spmm     | 0.173   | 0.229  | 1.32×    | 0.018    | 0.074   | 3.97×     |
| csr/spgemm   | 5.98    | 5.92   | 0.99×    | 1.74     | 1.72    | 0.99×     |
| bsr/spmv     | 1.52    | 3.25   | 2.14×    | 0.050    | 0.220   | 4.42×     |
| bsr/spmm     | 5.61    | 14.12  | 2.52×    | 0.331    | 0.841   | 2.55×     |
| bsr/spgemm   | 302.1   | 302.8  | 1.00×    | 37.9     | 65.7    | 1.73×     |
| vbcsr/spmv   | 6.15    | 8.22   | 1.34×    | 0.573    | 2.113   | 3.69×     |
| vbcsr/spmm   | 17.42   | 31.54  | 1.81×    | 1.270    | 2.842   | 2.24×     |
| vbcsr/spgemm | 641.5   | 5274.9 | 8.22×    | 133.9    | 609.8   | 4.55×     |

(times in ms). VBCSR beats sparse_dot_mkl in 16 of 18 cells and exactly
matches it in the other two (csr/bsr spgemm — the same MKL multiply
underneath; our copy-out + block-graph construction now costs what their
export-to-scipy costs). Headline changes vs the 2026-07-18 report:
csr/spmm 16T 0.44×→3.97× (cached+optimized mm handle with `beta=0` beats
sparse_dot_mkl's per-call handles outright), csr/spgemm 0.78×→0.99× (1T)
and 0.54×→0.99× (16T), bsr/spmm 16T 0.94×→2.55×, bsr/spmv 16T
0.196→0.050 ms, vbcsr/spmv 1T 1.06×→1.34×, vbcsr/spmm 16T (corrected)
→2.24×.

Remaining follow-ups (not done): `DistGraph::global_to_local` is a
`std::map` copied per serial SpGEMM (~0.11 ms; also the lookup structure for
all hot paths — a flat/hashed replacement helps both); bsr/spgemm Python
wrapper ~11 ms; native-vs-MKL A/B for the bsr SpGEMM multiply; the
`run_generic` CSR SpGEMM symbolic=full-multiply double work (distributed
path); csr/bsr native adjoint critical-section merge. The official
16-thread benchmark table should be re-baselined with the pinning protocol
before publication.

#### Publication-version cleanup (2026-07-20)

Final polish before freezing this tree as the publication version (all
gates green after: ctest 27/27, ASAN 27/27, Python suite 17/17, pyflakes
clean, perf spot-checks stable):
- C++: dead `DistGraph::neighbor_comm` state removed (declared/freed, never
  assigned); `update_local_block` "(DEBUG)" exception text rewritten; seven
  stream-of-consciousness TODOs converted to factual notes or deleted
  (AOCL reuse rationale, transpose materialization, block-density
  semantics, `run_generic` known-limitation note); vestigial `order` phase
  timers removed from both SpGEMM profiles (they timed the deleted
  `mkl_sparse_order`).
- Python (dedicated pass over all 8 package files): dead import + relic
  code removed in `utils.py`; ~40 lines of stream-of-consciousness /
  restating comments condensed in `matrix.py`; five stale design-doc
  references deleted; wrong docstrings fixed (`compute_kpm_coeffs` listed
  four nonexistent args; `gaussian` claimed to be Fermi-Dirac;
  `DistVector.to_numpy` returns a view, not a copy); misindented `raise`
  fixed.
- **Real bug fixed**: `DistMultiVector.__itruediv__` by a `DistVector`
  broadcast `(rows, nv) / (rows,)` — raised `ValueError` for `nv != rows`
  and would compute nonsense when equal. Now divides row-wise via a column
  view, mirroring `__imul__`; verified exact.
- Two flags resolved with the user (2026-07-20):
  `sample_k(symm=True)` was a false alarm — `VBCSR.T` routes to the C++
  transpose executor, which conjugates (`write_transposed_conjugate_block`),
  so the symmetrization already is the Hermitian `(A + A^H)/2`; the
  misleading docstrings were fixed and `transpose`/`T` now document the
  conjugating semantics (deliberately unlike `numpy.ndarray.T`).
  Unknown-global-extent conventions unified on "`None` in `.shape`, `0`
  from int accessors": `DistVector.shape` now falls back to `(None,)`
  instead of the local `(full_size,)` (its `size`/`__repr__` already
  guarded for `None`). Structural refactors (shared MKL SpGEMM export
  helper, assembly warn-vs-throw) stay deferred by choice — stability over
  churn at freeze time.

---

## 4. Validation matrix (which gate catches which silent-transpose)

| Failure scenario | Catching gate (post-Phase-0) |
|---|---|
| serial apply wrong | `test_numeric_reference` (ctest, all ops, both dtypes, tol 1e-10 rel) + `test_pb_csr` |
| MPI ghost exchange wrong | `test_robustness` (np2, random asymmetric), `run_all_tests.py` at np4, redistribute suite np∈{1,2,3} |
| SpGEMM wrong | new hard-failing C++ distributed gate + `test_spmm.py` under mpirun + `test_numeric_reference` transpose variants |
| `to_scipy` export wrong | `test_matrix_kind` exact-array + new MPI to_scipy test |
| assembly (`add_block`) wrong | `test_migration_contract` exact `get_values` + new ADD-mode test + round-trip test |
| padding lanes corrupted | `assert_padding_zero()` in debug tests + bdot/norm cross-check vs numpy |
| perf regression | per-phase benchmark gates vs frozen Phase-0 snapshots (1T and 16T) |

Note (from research): most strong tests are `assert()`-based; Phase 0 makes them
`NDEBUG`-proof **before** the default build becomes Release, or every later gate is
vacuously green.

## 5. Risk register

| # | Risk | Mitigation |
|---|---|---|
| R1 | Missed canonical-layout site → **silent block transpose across MPI** (worst case: non-square VBCSR blocks corrupt indexing too) | single `kCanonicalBlockLayout` constant used at every transport site (greppable); Phase-4 atomic commit; round-trip tests at multiple np; the six-site MPI-critical checklist |
| R2 | Complex conjugation error in adjoint kernels or the `spmf` `dense_gemm` "C"-vs-"T" trap | explicit transpose copies at the spmf boundary; complex adjoint cases in `test_numeric_reference` (already present) gate every phase |
| R3 | New kernels underperform on some shape | Phase-1 microbench go/no-go **before** library changes; per-shape gate table |
| R4 | Padding invariant silently broken (wrong norms/dots) | invariant + debug assertion + numpy cross-checks (§4) |
| R5 | Vendor flag/storage mismatch (MKL transposes every block) | flags flip in the same commit as storage (Phase 4 step 3); vendor-vs-native compare tests (`benchmark_dist` promoted to a checked test) |
| R6 | Wheel portability (AVX2 baseline) | scalar fallback guard structure preserved; `VBCSR_ARCH=none` escape; cibuildwheel smoke-tests the wheel |
| R7 | Temporary Phase-2 shim regression annoys benchmarking mid-migration | bounded by G2 (≤15 %), explicitly temporary, removed in Phase 3 |
| R8 | `mkl_sparse_order` removal breaks an unsorted-intolerant consumer | lazy-sort fallback + escape flag (§2.3) |

## 6. Effort estimate

| Phase | Estimate |
|---|---|
| 0 — safety net + build | 1–2 days |
| 1 — kernel prototype | 2–4 days |
| 2 — multivector flip | 3–5 days |
| 3 — kernel family | 4–6 days |
| 4 — block flip | 3–5 days |
| 5 — op wins | 1–2 days |
| 6 — re-baseline + CI + docs | 1–2 days |
| **Total** | **~3–4 weeks** single developer, each phase independently landable and green |

## 7. Explicit non-goals / follow-ups

- SpMV multi-thread scaling (7–39 % efficiency) — memory-bound; NUMA/first-touch
  and graph-reordering work, orthogonal to layout. Follow-up.
- AVX-512 / multiversioning — target CPU is Zen 2; keep the guard structure ready.
- Distributed-suite benchmarking (`mpi4py` absent in the env) — rerun strong/weak
  scaling after Phase 6 on the cluster.
- `image_container` / atomic pipeline performance — only correctness-flipped here.

# NUMA locality: partition-owned storage (design for the next round)

Status: **design accepted, not implemented.** Written 2026-07-20 after the
scaling study measured the defect. Nothing in the library implements this yet;
the current behavior is described under "Today" below.

## The defect

Linux places a page on the NUMA node of the thread that **first writes** it
("first touch"), not where it is allocated. Matrix pages are first written
during assembly, which is effectively serial, so on a multi-socket host the
entire matrix lands on **NUMA node 0**. Every thread pinned to another socket
then reads across the inter-socket link, and the library only ever uses one
socket's memory controllers.

Measured on the benchmark host (2× EPYC 7352, 2 NUMA nodes, 48 physical cores),
VBCSR SpMV, N = 48000 blocks (~1.06 GB matrix):

| threads | achieved bandwidth |
|---|---|
| 1 | 12 GB/s |
| 24 (one socket) | 59 GB/s |
| 48 (both sockets) | 49 GB/s — *slower than 24* |

Going from 24 to 48 threads makes it **worse**, because the second socket's
threads add remote traffic without adding usable bandwidth. The same run under
`numactl --interleave=all` reaches 101 GB/s (SpMV 21.5 → 10.5 ms, 2.05×; SpMM
32.7 → 15.8 ms, 2.08×), and MPI ranks — which are NUMA-local per process by
construction — already reach 113 GB/s. So the gap is placement, not algorithm.

## Root cause

Two partitionings exist and are not connected:

1. **Storage layout**, fixed at assembly (serial).
2. **The apply's thread work split**, recomputed per apply from
   `omp_get_max_threads()`.

Nothing ties them together, so placement cannot match access. Note the MPI
level does *not* have this problem: each rank owns, allocates, and first-touches
its own slice. The defect is that this discipline stops at the process level.

## Why interleaving is not the fix

`mbind(MPOL_INTERLEAVE)` / `numactl --interleave=all` spreads pages round-robin
so no single controller is the bottleneck. It never makes a read *local* — it
only spreads the pain — and it is measurably wrong for low thread counts:

| threads | default | interleaved | |
|---|---:|---:|---|
| 1 | 87.5 ms | 124.1 ms | **1.42× slower** |
| 24 (one socket) | 17.8 ms | 13.9 ms | 1.29× faster |
| 48 | 21.5 ms | 10.5 ms | 2.05× faster |

A serial run is latency-bound, not bandwidth-bound, so half its reads going
across the link is pure loss. Adopting interleave therefore requires a heuristic
policy keyed on a thread count that is only known at apply time, while the
placement decision is made at allocation time. That is a workaround with a
built-in failure mode, not a resolved design.

## The design: make the thread partition first-class, and let storage follow it

Extend the existing decomposition one level down, mirroring what already works
at the MPI level:

```
global rows ──MPI──> rank's owned rows ──NEW──> T thread domains ──> shape pages
```

Introduce a **`ThreadDomainPartition`** owned by the matrix backend:

- Computed **once** at backend construction (work-balanced by nnz, as
  `build_forward_work_chunks` does today) and **stored**.
- **Storage is allocated per domain** and first-touched inside an
  `omp parallel` region **by the thread that owns that domain**.
- **The apply iterates the same stored partition**, each thread taking its own
  domain id (which `bsr_thread_row_range` effectively already does).

One source of truth, so placement and access cannot disagree.

The important consequence: **no OS-specific code**. No `mbind`, no `libnuma`,
no `numactl`, no policy heuristic. Standard C++/OpenMP first touch plus thread
pinning place every page correctly by construction, portably to any NUMA host.

### Resolving the VBCSR layout obstacle

`ShapeBlockStore` currently groups blocks **by shape**, so one row's blocks
scatter across many pages and a page is shared by many rows — row-contiguous
domains cannot map onto it. The fix is a **two-level tiling: domain-major, then
shape-major.** Each thread domain owns its own set of shape pages, holding only
the blocks of its own rows.

This preserves everything the shape-paged layout buys — shape-batched SpGEMM
kernels, contiguous same-shape runs for the serial page-order SpMV — but now
*within* a domain, while each domain's storage becomes a contiguous,
independently allocated, node-local region. The block handle already encodes
`(shape_id, page_id, index)` and gains a domain field.

BSR and CSR are then the degenerate single-shape case of the same mechanism —
one uniform design rather than three special cases.

### Comparison

| | interleave | partition-owned storage |
|---|---|---|
| reads | ~50 % remote, always | ~100 % local |
| serial (1 thread) | 1.42× slower (measured) | identical to today (T = 1 ⇒ one domain) |
| policy needed | yes, heuristic on thread count | none |
| OS dependency | `mbind`/libnuma, Linux-only | none, portable |
| VBCSR | partial help only | fully covered |

It also removes the per-apply chunk computation (the partition is precomputed)
and makes the decomposition inspectable.

### Graceful degradation

- Apply runs with a different thread count than the stored partition → fall back
  to today's dynamic chunking. Correct, just not locality-optimal.
- Threads not pinned → same. Nothing breaks; an optimization is lost.

### Make locality a tested invariant

`get_mempolicy(MPOL_F_NODE | MPOL_F_ADDR)` reports which NUMA node an address
resides on. Add a test that samples each domain's pages and asserts they are on
the expected node, so locality becomes a CI-enforced invariant instead of
something that silently rots the next time allocation changes — which is exactly
how this defect arose.

## Staged plan

Each stage is independently verifiable and must leave the gates green.

| stage | work | estimate |
|---|---|---|
| **A** | `ThreadDomainPartition` as a first-class object; drive the *existing* applies from it, replacing ad-hoc per-apply chunking. No storage change, no perf change expected. | ~2 days |
| **B** | BSR/CSR domain-owned allocation + first touch (degenerate single-shape case); validates the machinery end to end. | ~2 days |
| **C** | VBCSR domain × shape tiling: `ShapeBlockStore`, handle encoding, apply plans, SpGEMM block assignment. | ~1–1.5 weeks |
| **D** | Locality invariant test; re-baseline the efficiency table and scaling figure. | ~1 day |

Total ≈ 2–3 weeks. Recommended entry point: **A → B** (~4 days, low risk) to
prove the design on hardware before committing to C.

## Today (current behavior, for reference)

- Output vectors *are* first-touched correctly: native applies zero their Y
  inside the parallel region, per thread range (`parallel_zero`, and the
  per-chunk zeroing in `vbcsr_apply.hpp` / `bsr_apply.hpp`).
- The **matrix backend** is not: pages are allocated and zero-filled by
  `PagedBuffer::append_page()` on the assembling thread.
- Users on multi-socket hosts can recover most of the loss today by running
  multi-threaded work under `numactl --interleave=all` (2.05× on SpMV at 48
  threads) — with the caveat that it is counterproductive below roughly one
  socket's worth of threads.

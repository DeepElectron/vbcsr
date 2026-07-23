# VBCSR multi-node campaign — protocol

Goal: establish (or refute) weak scalability at the production regime —
~15x15 blocks, ~500 neighbors per block row — on a real interconnect.
Everything measured so far is single-node; this campaign is the missing
evidence. Run the stages in order and do not skip stage 0.

## Prerequisites (once per cluster)

1. Build the library on the cluster with the cluster's MPI:
   `cmake -B build && cmake --build build -j` (needs MKL + pybind11 + the
   site MPI; check `ctest --test-dir build` passes on a login/compute node).
2. Python env with numpy, scipy, and **mpi4py compiled against the same
   MPI** the library uses.
3. A shared-filesystem scratch directory for the geometry cache.

## Stage 0 — 2-node smoke (30 min, decisive)

    sbatch tests/benchmark/cluster/run_smoke_2node.sbatch

Edit the CHANGE-ME lines first (`VBCSR_ROOT`, `PYTHON`, partition/account).
This runs the MPI correctness tests and one small validated weak point
across a real network. Most multi-node failures (transport configuration,
MPI env mismatches, ghost-pattern issues at real latency) show up here at
near-zero cost. **Do not proceed until it passes.**

## Stage 1 — warm the geometry caches (login/fat node, no allocation)

Geometry generation runs on rank 0 only and is expensive at million-atom
sizes; warm every weak point's cache up front so jobs only mmap:

    for NP in 64 128 256 512; do
        python tests/benchmark/cluster/warm_geometry.py \
            --blocks $((8000 * NP)) --target-degree 500 \
            --cache-dir /shared/scratch/vbcsr_geom
    done

Memory note: rank-0 generation at N blocks needs roughly the global
adjacency in RAM (~N * 500 * 12 B, i.e. ~25 GB at N = 4M) — use a fat
node for the largest sizes.

## Stage 2 — weak-scaling curve

    sbatch --nodes=1 tests/benchmark/cluster/run_weak_scaling.sbatch
    sbatch --nodes=2 tests/benchmark/cluster/run_weak_scaling.sbatch
    sbatch --nodes=4 tests/benchmark/cluster/run_weak_scaling.sbatch
    sbatch --nodes=8 tests/benchmark/cluster/run_weak_scaling.sbatch
    ...

Fixed per-rank problem, growing node count: flat per-op times = perfect
weak scaling. Knobs (env vars, see the sbatch header):

- `RANKS_PER_NODE` / `THREADS_PER_RANK`: run BOTH recommended layouts —
  rank-per-core (SpGEMM-optimal; single-node evidence: 48 ranks beat 48
  threads 0.37 s vs 1.06 s on bsr SpGEMM) and rank-per-NUMA-domain with
  threads (halo volume per rank shrinks with fewer, larger subdomains).
- `WEAK_BLOCKS_PER_RANK`: memory/rank ~ blocks * 0.9 MB at degree 500 /
  15x15. Budget ~2.2x that (matrix + thresholded C + workspace) against
  RAM per rank. Default 8000 => ~7.2 GB matrix per rank.
- `SPGEMM_THRESHOLD` (default 1e-3) with `VALUE_FILL=physical`: the
  production-representative configuration. Do NOT run threshold=0 SpGEMM
  at scale — unfiltered C = A*A is ~8x the matrix (multi-TB globally); the
  norm-filtered symbolic path is the designed route and bounds memory
  before allocation.

Scale up only after each point's numbers are understood (see below); a
1M-atom run is `WEAK_BLOCKS_PER_RANK * NP = 1e6` at whatever layout fits.

## Reading the results

    python tests/benchmark/cluster/plot_weak_scaling.py \
        --results-dir tests/benchmark/results/cluster_weak \
        --prefix weak_bsr_d500_b8000 --output-prefix weak_d500

- **Weak efficiency** (top row; ideal 1.0): SpMV/SpMM should hold >= ~0.9 —
  their per-rank work is bandwidth-bound and the halo is a few percent of
  it. Thresholded SpGEMM is the one to watch; its ghost metadata/payload
  volumes are the largest movers.
- **Ghost comm fraction** (bottom row; from the new `ghost_comm_*` fields
  in each case's `vbcsr_internal` JSON): this is the diagnosis channel. A
  drooping efficiency with FLAT comm fraction means load imbalance or
  per-rank slowdown, not the network; a RISING comm fraction localizes the
  problem to halo exchange (then: fewer/larger ranks per node, or ping the
  interconnect config).
- SpGEMM phase splits: rerun one point with
  `VBCSR_PROFILE_BSR_SPGEMM=1` (or `_VBCSR_`/`_CSR_`) in the environment —
  per-rank stderr lines give the metadata/symbolic/numeric/fill split.

## Known limits to respect at scale (audited 2026-07-23, commit 2f96ecc)

- Construct through `VBCSR.create_distributed_flat` (rank-local arrays).
  Never `construct_serial`/global-scipy paths — they replicate the global
  adjacency on every rank.
- Per-rank result block count must stay < 2^31; the library now throws
  "distribute over more ranks" instead of wrapping.
- `--value-fill physical` computes decay statistics in Python per row;
  it is fine at cluster per-rank sizes (rows/rank are modest), but if a
  point's build phase dominates, switch to `--value-fill random` for
  apply-only points (SpGEMM threshold behaviour then changes: random
  values do not decay, so the filter keeps nearly everything — keep
  physical fill for any thresholded-SpGEMM measurement).

## What "confident" looks like

1. Stage 0 passes (correctness across the wire).
2. SpMV/SpMM weak efficiency >= 0.9 with comm fraction < 10% out to the
   largest node count.
3. Thresholded SpGEMM weak efficiency understood (>= ~0.7 with a comm
   fraction that explains the rest), no memory or overflow failures.
4. One 1M-atom-scale point (blocks_per_rank x np = 1e6) completes with
   sane per-op times.

Anything that fails these produces exactly the diagnostics needed to fix
it: comm fractions per op, per-rank SpGEMM phase splits, and validated
correctness from stage 0.

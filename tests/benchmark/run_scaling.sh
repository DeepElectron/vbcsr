#!/usr/bin/env bash
# VBCSR strong scaling study (VBCSR-only, --no-baselines). Two sweeps on one node:
#   1. OpenMP thread strong scaling  (1 process, threads pinned to cores)
#   2. MPI process strong scaling    (fixed global size, 1 thread/rank, bound to cores)
# Results land in tests/benchmark/results/scaling/ as scaling_<mode>_<workers>.{json,csv}.
#
# Weak scaling is deliberately NOT run here. On a single node every rank shares
# the same memory controllers, so a bandwidth-bound kernel cannot hold constant
# time as ranks grow: the efficiency ceiling is (node bandwidth) / (P x
# single-core bandwidth), about 0.12 at 48 ranks on this host. That measures the
# machine, not the library. Run `--suite distributed-weak` on a multi-node
# machine, where each added rank brings its own memory subsystem.
#
# NOTE: this script only writes its own result files; it does not clear the
# output directory. Callers that wipe results/scaling before a rerun will also
# delete the written report living there -- keep a copy outside that directory.
#
# Usage: PYTHONPATH=<build> PYTHON=$(which python) bash tests/benchmark/run_scaling.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/results/scaling"
mkdir -p "${OUT_DIR}"

# Interpreter: pass PYTHON=<abs path to the env python> so the sweep does not
# depend on the caller's PATH (mpiexec children inherit it explicitly).
PYTHON="${PYTHON:-python}"

# Worker counts. 24 = one socket, 48 = both sockets (all physical cores) on the
# dual-socket EPYC 7352 host.
WORKERS=(1 2 4 8 16 24 48)

# Strong scaling: fixed global problem, sized by *stored bytes* rather than block
# count. A fixed block count is not a fixed problem: with 8x8 BSR blocks and
# ~14x14 VBCSR blocks it spans a ~130x range in footprint, which left scalar CSR
# entirely inside the node's ~192 MB aggregate L3 (16 MB/CCX x 6 CCX/socket x 2
# sockets) while the other domains were far into DRAM.
#
# The target must clear aggregate L3 by a wide margin, not merely exceed it.
# At 256 MB (1.3x L3) the repeated-apply loop still ran largely out of cache
# once all 48 threads contributed their L3 slices: SpMV reported 129-155 GB/s
# against a measured DRAM roofline of 125.7 GB/s, i.e. above what the memory
# system can deliver, with a spurious 2.8x jump between 24 and 48 threads.
# 1 GiB is ~5.6x aggregate L3 and reproduces the clean DRAM-bound behaviour the
# earlier ~1 GB VBCSR case showed.
#
# Sanity check after any resize: achieved bandwidth at the highest worker count
# must stay BELOW the roofline. Above it means you are timing cache, not memory.
STRONG_BYTES=$((1024 * 1024 * 1024))

# Timing controls (VBCSR-only, so we can afford steady medians without baselines).
# min-seconds drives many iterations for the sub-millisecond apply ops; the low
# min-iterations floor keeps the heavy SpGEMM points affordable.
TIMING=(--warmups 2 --repeats 3 --min-seconds 0.3 --min-iterations 2)
COMMON=(--target-degree 12 --rhs 16 --no-baselines --output-dir "${OUT_DIR}")
SIZING=(--target-storage-bytes "${STRONG_BYTES}")

export MKL_NUM_THREADS=1

echo "=== [1/2] OpenMP thread strong scaling (${STRONG_BYTES} B/domain) ==="
for t in "${WORKERS[@]}"; do
    echo "--- threads=${t} ---"
    OMP_NUM_THREADS="${t}" OMP_PROC_BIND=close OMP_PLACES=cores \
        "${PYTHON}" "${SCRIPT_DIR}/run_benchmark.py" --suite efficiency \
        "${SIZING[@]}" "${COMMON[@]}" "${TIMING[@]}" \
        --label "scaling_thread_strong_w${t}"
done

echo "=== [2/2] MPI process strong scaling (${STRONG_BYTES} B/domain) ==="
for p in "${WORKERS[@]}"; do
    echo "--- ranks=${p} ---"
    OMP_NUM_THREADS=1 \
        mpiexec --bind-to core --map-by core -n "${p}" \
        "${PYTHON}" "${SCRIPT_DIR}/run_benchmark.py" --suite distributed-strong \
        "${SIZING[@]}" "${COMMON[@]}" "${TIMING[@]}" \
        --label "scaling_mpi_strong_w${p}"
done

echo "=== scaling sweep complete ==="

#!/bin/bash
#SBATCH --job-name=vbcsr-weak
#SBATCH --time=02:00:00
### CHANGE-ME: set your partition/account and node count per submission:
###   sbatch --nodes=N run_weak_scaling.sbatch
#SBATCH --exclusive
##SBATCH --partition=CHANGE-ME
##SBATCH --account=CHANGE-ME

# Weak-scaling point at the million-atom target regime (15x15 blocks,
# ~500 neighbors). Submit once per node count (1, 2, 4, 8, ...); every
# submission runs the same per-rank problem, so the resulting per-op times
# ARE the weak-scaling curve. See README_CLUSTER.md for the protocol.

set -euo pipefail

# ---------------------------------------------------------------- knobs ---
# CHANGE-ME: absolute repo path on the cluster (with build/ compiled there).
VBCSR_ROOT=${VBCSR_ROOT:-/path/to/vbcsr}
# CHANGE-ME: python with numpy+mpi4py matching the cluster MPI.
PYTHON=${PYTHON:-python}
# Shared-filesystem geometry cache (pre-warm with warm_geometry.py!).
GEOM_CACHE=${GEOM_CACHE:-$VBCSR_ROOT/tests/benchmark/results/geom_cache}
OUT_DIR=${OUT_DIR:-$VBCSR_ROOT/tests/benchmark/results/cluster_weak}

# Layout: ranks per node and threads per rank. Two recommended configs:
#   SpGEMM-heavy:  RANKS_PER_NODE = cores, THREADS_PER_RANK = 1
#   halo-lean:     RANKS_PER_NODE = NUMA domains, THREADS_PER_RANK = cores/NUMA
RANKS_PER_NODE=${RANKS_PER_NODE:-16}
THREADS_PER_RANK=${THREADS_PER_RANK:-1}

# Per-rank problem: memory/rank ~= WEAK_BLOCKS_PER_RANK * degree * 15*15 * 8 B
# (8000 blocks/rank * 500 * 1.8 kB ~= 7.2 GB/rank). Size so that
# 2 x that (matrix + thresholded SpGEMM result + workspace) fits RAM/rank.
WEAK_BLOCKS_PER_RANK=${WEAK_BLOCKS_PER_RANK:-8000}
TARGET_DEGREE=${TARGET_DEGREE:-500}
BSR_BLOCK_SIZE=${BSR_BLOCK_SIZE:-15}
RHS=${RHS:-16}
# Production-representative SpGEMM: thresholded (exact threshold=0 C=A*A is
# multi-TB at scale — do not run it in the campaign).
SPGEMM_THRESHOLD=${SPGEMM_THRESHOLD:-1e-3}
# physical = decay-model values (realistic norms for the threshold filter).
VALUE_FILL=${VALUE_FILL:-physical}
DOMAINS=${DOMAINS:-bsr}
# --------------------------------------------------------------------------

NODES=${SLURM_JOB_NUM_NODES:?run under sbatch --nodes=N}
NP=$((NODES * RANKS_PER_NODE))
LABEL="weak_${DOMAINS}_d${TARGET_DEGREE}_b${WEAK_BLOCKS_PER_RANK}_r${RANKS_PER_NODE}x${THREADS_PER_RANK}_np${NP}"

cd "$VBCSR_ROOT"
export VBCSR_BUILD_DIR=build PYTHONPATH=build
export OMP_NUM_THREADS=$THREADS_PER_RANK
export OMP_PROC_BIND=close OMP_PLACES=cores
export MKL_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
mkdir -p "$OUT_DIR"

echo "=== $LABEL: $NODES nodes x $RANKS_PER_NODE ranks x $THREADS_PER_RANK threads ==="
date

# srun is the SLURM-native launcher; swap for mpirun/mpiexec if your site
# requires it (keep the binding flags equivalent).
srun --ntasks="$NP" --ntasks-per-node="$RANKS_PER_NODE" \
     --cpus-per-task="$THREADS_PER_RANK" --cpu-bind=cores \
    "$PYTHON" tests/benchmark/run_benchmark.py \
        --suite distributed-weak \
        --weak-blocks-per-rank "$WEAK_BLOCKS_PER_RANK" \
        --domains "$DOMAINS" \
        --target-degree "$TARGET_DEGREE" \
        --bsr-block-size "$BSR_BLOCK_SIZE" \
        --rhs "$RHS" \
        --spgemm-threshold "$SPGEMM_THRESHOLD" \
        --value-fill "$VALUE_FILL" \
        --no-baselines \
        --warmups 2 --repeats 3 --min-seconds 0.3 --min-iterations 2 \
        --geometry-cache-dir "$GEOM_CACHE" \
        --output-dir "$OUT_DIR" \
        --label "$LABEL"

date
echo "=== done: $OUT_DIR/$LABEL.{json,csv} ==="

#!/bin/bash
#SBATCH --job-name=vbcsr-smoke
#SBATCH --nodes=2
#SBATCH --time=00:30:00
#SBATCH --exclusive
##SBATCH --partition=CHANGE-ME
##SBATCH --account=CHANGE-ME

# Stage-0 smoke test: SMALL problem, 2 nodes. Purpose is not performance —
# it is to prove the stack end-to-end on a real interconnect before burning
# allocation: MPI transport config, correctness across a network, and a
# first look at comm fractions. Runs the python MPI test suite pieces plus
# a small weak point, and validates SpMV/SpGEMM against serial references
# (baselines stay ON at this size).

set -euo pipefail

VBCSR_ROOT=${VBCSR_ROOT:-/path/to/vbcsr}     # CHANGE-ME
PYTHON=${PYTHON:-python}                      # CHANGE-ME
RANKS_PER_NODE=${RANKS_PER_NODE:-8}
OUT_DIR=${OUT_DIR:-$VBCSR_ROOT/tests/benchmark/results/cluster_smoke}

cd "$VBCSR_ROOT"
export VBCSR_BUILD_DIR=build PYTHONPATH=build
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT_DIR"
NP=$((2 * RANKS_PER_NODE))

echo "=== correctness across the interconnect (np=$NP over 2 nodes) ==="
srun --ntasks="$NP" --ntasks-per-node="$RANKS_PER_NODE" --cpu-bind=cores \
    "$PYTHON" tests/test_spmm.py
srun --ntasks="$NP" --ntasks-per-node="$RANKS_PER_NODE" --cpu-bind=cores \
    "$PYTHON" tests/test_api_mpi.py

echo "=== small weak point with validation (baselines ON) ==="
srun --ntasks="$NP" --ntasks-per-node="$RANKS_PER_NODE" --cpu-bind=cores \
    "$PYTHON" tests/benchmark/run_benchmark.py \
        --suite distributed-weak \
        --weak-blocks-per-rank 1000 \
        --domains bsr --target-degree 100 --bsr-block-size 15 --rhs 16 \
        --spgemm-threshold 1e-3 \
        --warmups 1 --repeats 2 --min-seconds 0.1 --min-iterations 1 \
        --output-dir "$OUT_DIR" --label "smoke_2node_np${NP}"

echo "=== SMOKE PASSED — inspect ghost_comm_* in $OUT_DIR/smoke_2node_np${NP}.json ==="

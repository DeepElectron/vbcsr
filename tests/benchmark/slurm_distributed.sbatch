#!/usr/bin/env bash
#SBATCH --job-name=vbcsr-scaling
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=tests/benchmark/results/slurm-scaling-%j.out
#SBATCH --error=tests/benchmark/results/slurm-scaling-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
CONDA_ENV="${CONDA_ENV:-vbcsr}"
VBCSR_BUILD_DIR="${VBCSR_BUILD_DIR:-build}"
SUITE="${SUITE:-distributed-strong}"
SRUN_MPI_TYPE="${SRUN_MPI_TYPE:-pmix}"
LABEL="${LABEL:-paper_${SUITE}_np${SLURM_NTASKS}_${SLURM_JOB_ID}}"

cd "${REPO_ROOT}"
mkdir -p tests/benchmark/results

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export VBCSR_BUILD_DIR
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK}}"

COMMON_ARGS=(
  --suite "${SUITE}"
  --target-degree "${TARGET_DEGREE:-12}"
  --rhs "${RHS:-16}"
  --repeats "${REPEATS:-7}"
  --min-seconds "${MIN_SECONDS:-1.0}"
  --min-iterations "${MIN_ITERATIONS:-5}"
  --output-dir tests/benchmark/results
  --label "${LABEL}"
)

if [[ "${SUITE}" == "distributed-weak" ]]; then
  COMMON_ARGS+=(--weak-blocks-per-rank "${WEAK_BLOCKS_PER_RANK:-2048}")
else
  COMMON_ARGS+=(--blocks "${BLOCKS:-8192}")
fi

srun --mpi="${SRUN_MPI_TYPE}" -n "${SLURM_NTASKS}" \
  python tests/benchmark/run_benchmark.py "${COMMON_ARGS[@]}"

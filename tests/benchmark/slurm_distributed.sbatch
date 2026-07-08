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
PYTHON_VENV="${PYTHON_VENV:-${REPO_ROOT}/.venv-vbcsr}"
VBCSR_BUILD_DIR="${VBCSR_BUILD_DIR:-build}"
SUITE="${SUITE:-distributed-strong}"
SRUN_MPI_TYPE="${SRUN_MPI_TYPE:-pmix}"
LABEL="${LABEL:-paper_${SUITE}_np${SLURM_NTASKS}_${SLURM_JOB_ID}}"
ALLIANCE_MODULES="${ALLIANCE_MODULES:-StdEnv/2023 python scipy-stack mpi4py flexiblas}"
FLEXIBLAS_BACKEND="${FLEXIBLAS_BACKEND:-${FLEXIBLAS:-}}"
SPGEMM_THRESHOLD="${SPGEMM_THRESHOLD:-0.0}"
SPGEMM_THRESHOLDS="${SPGEMM_THRESHOLDS:-${SPGEMM_THRESHOLD}}"

cd "${REPO_ROOT}"
mkdir -p tests/benchmark/results

if command -v module >/dev/null 2>&1; then
  module --force purge
  # shellcheck disable=SC2086
  module load ${ALLIANCE_MODULES}
  module list
fi

if [[ ! -d "${PYTHON_VENV}" ]]; then
  echo "Python virtualenv not found: ${PYTHON_VENV}" >&2
  echo "Create it on the cluster with the same modules, e.g.:" >&2
  echo "  module load ${ALLIANCE_MODULES}" >&2
  echo "  virtualenv --no-download ${PYTHON_VENV}" >&2
  echo "  source ${PYTHON_VENV}/bin/activate" >&2
  echo "  pip install --no-index --upgrade pip" >&2
  echo "  pip install --no-index mpi4py sparse-dot-mkl" >&2
  exit 2
fi
source "${PYTHON_VENV}/bin/activate"

export PYTHON_VENV
export VBCSR_BUILD_DIR
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK}}"
if [[ -n "${FLEXIBLAS_BACKEND}" ]]; then
  export FLEXIBLAS="${FLEXIBLAS_BACKEND}"
fi
if command -v flexiblas >/dev/null 2>&1; then
  flexiblas list || true
fi

python - <<'PY'
import importlib.util
missing = [name for name in ("numpy", "scipy", "mpi4py") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing required Python modules: {missing}")
PY

COMMON_ARGS=(
  --suite "${SUITE}"
  --target-degree "${TARGET_DEGREE:-12}"
  --rhs "${RHS:-16}"
  --dtype "${DTYPE:-real}"
  --seed "${SEED:-1729}"
  --bsr-block-size "${BSR_BLOCK_SIZE:-8}"
  --spgemm-threshold "${SPGEMM_THRESHOLD}"
  --spgemm-thresholds "${SPGEMM_THRESHOLDS}"
  --spgemm-audit-limit "${SPGEMM_AUDIT_LIMIT:-200000}"
  --geometry-dim "${GEOMETRY_DIM:-3}"
  --geometry-spacing "${GEOMETRY_SPACING:-1.0}"
  --geometry-jitter "${GEOMETRY_JITTER:-0.12}"
  --geometry-cutoff-quantile "${GEOMETRY_CUTOFF_QUANTILE:-0.90}"
  --repeats "${REPEATS:-7}"
  --min-seconds "${MIN_SECONDS:-1.0}"
  --min-iterations "${MIN_ITERATIONS:-5}"
  --warmups "${WARMUPS:-3}"
  --output-dir tests/benchmark/results
  --label "${LABEL}"
)
if [[ -n "${GEOMETRY_CUTOFF:-}" ]]; then
  COMMON_ARGS+=(--geometry-cutoff "${GEOMETRY_CUTOFF}")
fi

if [[ "${SUITE}" == "distributed-weak" ]]; then
  COMMON_ARGS+=(--weak-blocks-per-rank "${WEAK_BLOCKS_PER_RANK:-4096}")
else
  COMMON_ARGS+=(--blocks "${BLOCKS:-32768}")
fi

srun --mpi="${SRUN_MPI_TYPE}" -n "${SLURM_NTASKS}" \
  python tests/benchmark/run_benchmark.py "${COMMON_ARGS[@]}"

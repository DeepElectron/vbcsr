#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VIRTUAL_ENV:-/home/zhanghao/envs/vbcsr}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Could not find Python in ${VENV_DIR}."
    echo "Activate the environment first, or set VIRTUAL_ENV to the vbcsr venv."
    exit 1
fi

source "${VENV_DIR}/bin/activate"

if ! command -v module >/dev/null 2>&1; then
    # Compute Canada exposes the module command through the Lmod init script.
    # Non-interactive shells often need this explicit initialization.
    # shellcheck disable=SC1091
    source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
fi

echo "Loading Compute Canada modules..."
module load StdEnv/2023
module load gcc/12.3 openmpi/4.1.5
module load imkl/2025.2.0
module load metis/5.1.0 parmetis/4.0.3

# Loading modules can prepend their own Python. Re-activate the venv so the
# build always uses the requested environment.
source "${VENV_DIR}/bin/activate"

PYTHON="${VENV_DIR}/bin/python"
PIP=("${PYTHON}" -m pip)

echo "Using Python: $(${PYTHON} -c 'import sys; print(sys.executable)')"
echo "Python version: $(${PYTHON} -c 'import sys; print(sys.version.split()[0])')"
echo "Using mpicc: $(command -v mpicc)"
echo "Using mpicxx: $(command -v mpicxx)"

find_one() {
    local root="$1"
    local name="$2"
    local path
    path="$(find "${root}" -name "${name}" | sort | head -n 1 || true)"
    if [[ -z "${path}" ]]; then
        echo "Could not find ${name} under ${root}" >&2
        exit 1
    fi
    printf '%s\n' "${path}"
}

require_env() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "${name} is not set. Did the required module fail to load?" >&2
        exit 1
    fi
}

require_env MKLROOT
require_env EBROOTMETIS
require_env EBROOTPARMETIS

MKL_RT="$(find_one "${MKLROOT}" "libmkl_rt.so")"
METIS_LIB="$(find_one "${EBROOTMETIS}" "libmetis.so")"
PARMETIS_LIB="$(find_one "${EBROOTPARMETIS}" "libparmetis.so")"
METIS_INCLUDE="${EBROOTMETIS}/include"
PARMETIS_INCLUDE="${EBROOTPARMETIS}/include"

if [[ ! -f "${METIS_INCLUDE}/metis.h" ]]; then
    echo "Could not find metis.h in ${METIS_INCLUDE}" >&2
    exit 1
fi
if [[ ! -f "${PARMETIS_INCLUDE}/parmetis.h" ]]; then
    echo "Could not find parmetis.h in ${PARMETIS_INCLUDE}" >&2
    exit 1
fi

export CC="${CC:-$(command -v mpicc)}"
export CXX="${CXX:-$(command -v mpicxx)}"
export MPICC="${MPICC:-$(command -v mpicc)}"
export MPICXX="${MPICXX:-$(command -v mpicxx)}"
export CMAKE_PREFIX_PATH="${MKLROOT}:${EBROOTMETIS}:${EBROOTPARMETIS}:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$(dirname "${MKL_RT}"):$(dirname "${METIS_LIB}"):$(dirname "${PARMETIS_LIB}"):${LD_LIBRARY_PATH:-}"

echo "Installing Python build/runtime dependencies..."
"${PIP[@]}" install --no-index scikit-build-core pybind11 cmake ninja fypp numpy scipy matplotlib ase

if ! "${PYTHON}" -c 'import mpi4py' >/dev/null 2>&1; then
    echo "Installing mpi4py from source for the active venv Python..."
    echo "Compute Canada's mpi4py pip package is a dummy; building avoids mixing Python ABIs."
    "${PIP[@]}" install --no-binary=mpi4py mpi4py
fi

echo "Installing vbcsr..."
cd "${ROOT_DIR}"
"${PIP[@]}" install . --no-build-isolation -v \
    -C cmake.define.CMAKE_C_COMPILER="${CC}" \
    -C cmake.define.CMAKE_CXX_COMPILER="${CXX}" \
    -C cmake.define.BLA_VENDOR=Intel10_64lp \
    -C cmake.define.BLAS_FOUND=ON \
    -C cmake.define.LAPACK_FOUND=ON \
    -C cmake.define.BLAS_LIBRARIES="${MKL_RT}" \
    -C cmake.define.LAPACK_LIBRARIES="${MKL_RT}" \
    -C cmake.define.PARMETIS_LIB="${PARMETIS_LIB}" \
    -C cmake.define.METIS_LIB="${METIS_LIB}" \
    -C cmake.define.PARMETIS_INCLUDE_DIR="${PARMETIS_INCLUDE}" \
    -C cmake.define.METIS_INCLUDE_DIR="${METIS_INCLUDE}" \
    -C cmake.define.VBCSR_USE_ILP64=OFF

echo "Checking import..."
"${PYTHON}" - <<'PY'
import sys
import mpi4py
import vbcsr

print("Python:", sys.executable)
print("mpi4py:", mpi4py.__version__, mpi4py.__file__)
print("vbcsr:", vbcsr.__version__, vbcsr.__file__)
PY

echo "vbcsr installation completed."

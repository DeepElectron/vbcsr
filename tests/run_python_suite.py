import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = Path(os.environ.get("VBCSR_BUILD_DIR", REPO_ROOT / "build"))
SERIAL_TESTS = [
    "test_matrix_kind.py",
    "test_scipy_adapter.py",
    "test_wrapper_contracts.py",
    "test_api_serial.py",
    "test_api_compliance.py",
    "test_spmf.py",
    "test_kpm.py",
    "test_vbcsr.py",
]
MPI_TESTS = [
    "test_api_mpi.py",
    "test_from_scipy_collective.py",
]


def have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def make_test_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [str(REPO_ROOT)]
    if BUILD_DIR.is_dir():
        pythonpath_entries.append(str(BUILD_DIR))

    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)

    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env.setdefault("VBCSR_BUILD_DIR", str(BUILD_DIR))
    return env


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Python VBCSR regression suite.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    parser.add_argument("--include-mpi", action="store_true", help="Also run the MPI Python tests")
    parser.add_argument("--np", type=int, default=2, help="MPI rank count for --include-mpi")
    parser.add_argument("--mpirun", default=shutil.which("mpirun") or "mpirun", help="MPI launcher")
    args = parser.parse_args()

    missing = [name for name in ("numpy", "scipy") if not have_module(name)]
    if missing:
        print(f"Missing Python dependencies: {', '.join(missing)}")
        return 2

    test_env = make_test_env()
    failed: list[str] = []

    for test_name in SERIAL_TESTS:
        code = run_command([args.python, str(REPO_ROOT / "tests" / test_name)], REPO_ROOT, test_env)
        if code != 0:
            failed.append(test_name)

    if args.include_mpi:
        if not have_module("mpi4py"):
            print("Skipping MPI Python tests because mpi4py is not installed.")
        elif shutil.which(args.mpirun) is None and args.mpirun == "mpirun":
            print("Skipping MPI Python tests because `mpirun` is not available.")
        else:
            for test_name in MPI_TESTS:
                code = run_command(
                    [args.mpirun, "-np", str(args.np), args.python, str(REPO_ROOT / "tests" / test_name)],
                    REPO_ROOT,
                    test_env,
                )
                if code != 0:
                    failed.append(test_name)

    if failed:
        print("Python test suite failed:")
        for name in failed:
            print(f"  - {name}")
        return 1

    print("Python test suite passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

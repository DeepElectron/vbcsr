import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

from _workspace_bootstrap import REPO_ROOT, build_subprocess_env
from _suite_manifest import MPI_TESTS, SERIAL_TESTS


def have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


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

    test_env = build_subprocess_env()
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

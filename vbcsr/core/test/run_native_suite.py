import argparse
import os
import subprocess
import sys
from pathlib import Path


def prepend_env_path(env: dict[str, str], key: str, path: Path) -> None:
    if not path.is_dir():
        return
    existing = env.get(key)
    env[key] = str(path) if not existing else f"{path}{os.pathsep}{existing}"


def build_runner_env(python_executable: str) -> dict[str, str]:
    env = os.environ.copy()
    python_bin = Path(python_executable).resolve().parent
    prepend_env_path(env, "PATH", python_bin)

    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lib = Path(conda_prefix) / "lib"
        prepend_env_path(env, "LD_LIBRARY_PATH", conda_lib)
        prepend_env_path(env, "LIBRARY_PATH", conda_lib)

    return env


def resolve_tool(default_value: str, python_executable: str) -> str:
    if default_value != Path(default_value).name:
        return default_value
    sibling = Path(python_executable).resolve().parent / default_value
    if sibling.exists():
        return str(sibling)
    return default_value


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    print(f"$ {' '.join(str(part) for part in cmd)}")
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the maintained native VBCSR regression workflow.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for the runner scripts")
    parser.add_argument("--mpicxx", default="mpicxx", help="MPI C++ compiler for the direct core runner")
    parser.add_argument("--mpirun", default="mpirun", help="MPI launcher for the direct core runner")
    parser.add_argument("--np", type=int, default=2, help="MPI rank count for the direct core runner")
    parser.add_argument("--openblas", default="openblas", help="BLAS library name without the -l prefix")
    parser.add_argument("--std", default="c++17", help="C++ language standard for the direct core runner")
    parser.add_argument("--cmake", default="cmake", help="CMake executable for registered tests")
    parser.add_argument("--ctest", default="ctest", help="CTest executable for registered tests")
    parser.add_argument("--cmake-build-dir", default=None, help="Build directory for the CMake-registered tests")
    parser.add_argument("--skip-core", action="store_true", help="Skip the direct core C++ runner")
    parser.add_argument("--skip-cmake", action="store_true", help="Skip the CMake-registered test runner")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    runner_env = build_runner_env(args.python)
    args.mpicxx = resolve_tool(args.mpicxx, args.python)
    args.mpirun = resolve_tool(args.mpirun, args.python)
    failed: list[str] = []

    if not args.skip_core:
        core_cmd = [
            args.python,
            str(script_dir / "run_all_tests.py"),
            "--mpicxx",
            args.mpicxx,
            "--mpirun",
            args.mpirun,
            "--np",
            str(args.np),
            "--openblas",
            args.openblas,
            "--std",
            args.std,
        ]
        if run_command(core_cmd, script_dir, runner_env) != 0:
            failed.append("run_all_tests.py")

    if not args.skip_cmake:
        cmake_cmd = [
            args.python,
            str(script_dir / "run_cmake_registered_tests.py"),
            "--cmake",
            args.cmake,
            "--ctest",
            args.ctest,
        ]
        if args.cmake_build_dir:
            cmake_cmd.extend(["--build-dir", args.cmake_build_dir])
        if run_command(cmake_cmd, script_dir, runner_env) != 0:
            failed.append("run_cmake_registered_tests.py")

    if failed:
        print("Native regression workflow failed:")
        for name in failed:
            print(f"  - {name}")
        return 1

    print("Native regression workflow passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

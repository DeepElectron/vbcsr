import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_TESTS = [
    "test_block_csr_export",
    "test_atomic_data",
    "test_image_container",
    "test_neighbourlist",
    "test_neighbourlist_repro",
]


def run_command(cmd: list[str], cwd: Path) -> int:
    print(f"$ {' '.join(str(part) for part in cmd)}")
    return subprocess.run(cmd, cwd=cwd).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Configure, build, and run the maintained CMake-registered VBCSR tests."
    )
    parser.add_argument("--cmake", default=shutil.which("cmake") or "cmake", help="CMake executable")
    parser.add_argument("--ctest", default=shutil.which("ctest") or "ctest", help="CTest executable")
    parser.add_argument(
        "--build-dir",
        default=None,
        help="Out-of-source build directory for the maintained CMake tests",
    )
    parser.add_argument("--tests", nargs="*", default=DEFAULT_TESTS, help="CTest targets to build and run")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    build_dir = Path(args.build_dir) if args.build_dir else repo_root / "build" / "cmake-tests"

    if shutil.which(args.cmake) is None and args.cmake == "cmake":
        print("`cmake` is not available.")
        return 2
    if shutil.which(args.ctest) is None and args.ctest == "ctest":
        print("`ctest` is not available.")
        return 2

    configure_cmd = [args.cmake, "-S", str(repo_root), "-B", str(build_dir), "-DVBCSR_ENABLE_TESTS=ON"]
    if run_command(configure_cmd, repo_root) != 0:
        return 1

    build_cmd = [args.cmake, "--build", str(build_dir), "--target", *args.tests]
    if run_command(build_cmd, repo_root) != 0:
        return 1

    failed: list[str] = []
    for test_name in args.tests:
        ctest_cmd = [args.ctest, "--test-dir", str(build_dir), "--output-on-failure", "-R", f"^{test_name}$"]
        if run_command(ctest_cmd, repo_root) != 0:
            failed.append(test_name)

    if failed:
        print("CMake-registered test suite failed:")
        for test_name in failed:
            print(f"  - {test_name}")
        return 1

    print("CMake-registered test suite passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

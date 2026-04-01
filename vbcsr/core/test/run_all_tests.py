import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


TESTS = [
    "test_asymmetric_filter.cpp",
    "test_axpby.cpp",
    "test_axpby_diff_graph.cpp",
    "test_backend_extensions.cpp",
    "test_block_arena.cpp",
    "test_block_csr.cpp",
    "test_block_csr_export.cpp",
    "test_complex_dist_vector.cpp",
    "test_density.cpp",
    "test_dist_csr.cpp",
    "test_dist_graph.cpp",
    "test_extract_batched.cpp",
    "test_extract_batched_extended.cpp",
    "test_extract_batched_robust.cpp",
    "test_graphmf.cpp",
    "test_hermitian_product.cpp",
    "test_migration_contract.cpp",
    "test_mult_graph_mismatch.cpp",
    "test_multi_sparsity.cpp",
    "test_pb_csr.cpp",
    "test_robustness.cpp",
    "test_spmm.cpp",
    "test_subgraph.cpp",
]


def make_tool_env(mpicxx: str, mpirun: str) -> dict[str, str]:
    env = os.environ.copy()
    path_entries: list[str] = []
    for tool in (mpicxx, mpirun):
        resolved = shutil.which(tool) if Path(tool).name == tool else tool
        if resolved:
            tool_dir = str(Path(resolved).resolve().parent)
            if tool_dir not in path_entries:
                path_entries.append(tool_dir)

    existing = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(path_entries + ([existing] if existing else []))
    return env


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile and run the VBCSR C++ regression suite.")
    parser.add_argument("--mpicxx", default=shutil.which("mpicxx") or "mpicxx", help="MPI C++ compiler")
    parser.add_argument("--mpirun", default=shutil.which("mpirun") or "mpirun", help="MPI launcher")
    parser.add_argument("--np", type=int, default=4, help="MPI rank count for test execution")
    parser.add_argument("--openblas", default="openblas", help="BLAS library name without the -l prefix")
    parser.add_argument("--std", default="c++17", help="C++ language standard")
    parser.add_argument("--tests", nargs="*", default=TESTS, help="Specific test files to compile and run")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    include_dir = script_dir.parent

    if shutil.which(args.mpicxx) is None and args.mpicxx == "mpicxx":
        print("`mpicxx` is not available.")
        return 2
    if shutil.which(args.mpirun) is None and args.mpirun == "mpirun":
        print("`mpirun` is not available.")
        return 2

    failed: list[str] = []
    passed: list[str] = []
    tool_env = make_tool_env(args.mpicxx, args.mpirun)

    print(f"Running {len(args.tests)} C++ tests...")
    for test_file in args.tests:
        exe_name = f"exec_{Path(test_file).stem}"
        compile_cmd = [
            args.mpicxx,
            f"-std={args.std}",
            "-fopenmp",
            "-O3",
            f"-I{include_dir}",
            test_file,
            "-o",
            exe_name,
            f"-l{args.openblas}",
        ]

        print("--------------------------------------------------")
        print(f"Compiling {test_file}...")
        compile_result = run_command(compile_cmd, script_dir, tool_env)
        if compile_result.returncode != 0:
            print(compile_result.stdout)
            print(compile_result.stderr)
            failed.append(test_file)
            continue

        run_cmd = [args.mpirun, "-np", str(args.np), f"./{exe_name}"]
        print(f"Running {exe_name}...")
        run_result = run_command(run_cmd, script_dir, tool_env)
        if run_result.returncode != 0:
            print(run_result.stdout)
            print(run_result.stderr)
            failed.append(test_file)
            continue

        print(run_result.stdout)
        passed.append(test_file)

    print("==================================================")
    print(f"Summary: {len(passed)}/{len(args.tests)} passed.")
    if failed:
        print("Failed tests:")
        for test_name in failed:
            print(f"  - {test_name}")
        return 1

    print("All C++ tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

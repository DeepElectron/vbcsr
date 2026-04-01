import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK = REPO_ROOT / "tests" / "benchmark_large_scale.py"


def run_command(cmd, cwd):
    print("Running:", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def build_cases(families, profiles, rank_counts, modes):
    for family in families:
        for profile in profiles:
            for rank_count in rank_counts:
                for mode in modes:
                    yield family, profile, rank_count, mode


def main():
    parser = argparse.ArgumentParser(description="Freeze phase 0 baseline snapshots")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "phase0_snapshots", help="Directory for snapshot JSON files")
    parser.add_argument("--families", nargs="+", default=["csr", "bsr", "vbcsr"], help="Matrix families to benchmark")
    parser.add_argument("--profiles", nargs="+", default=["small", "medium"], help="Preset profiles to run")
    parser.add_argument("--rank-counts", nargs="+", type=int, default=[1, 2], help="MPI rank counts to run")
    parser.add_argument("--modes", nargs="+", default=["mult", "mult_dense", "spmm"], help="Benchmark modes to run")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the Python smoke tests")
    parser.add_argument("--min-seconds", type=float, default=None, help="Forward minimum benchmark duration to the benchmark runner")
    parser.add_argument("--min-iterations", type=int, default=None, help="Forward minimum benchmark iterations to the benchmark runner")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_smoke:
        smoke_scripts = [
            REPO_ROOT / "tests" / "test_scipy_adapter.py",
            REPO_ROOT / "tests" / "test_api_serial.py",
            REPO_ROOT / "tests" / "test_api_mpi.py",
            REPO_ROOT / "tests" / "test_api_compliance.py",
            REPO_ROOT / "tests" / "test_matrix_kind.py",
        ]
        for script in smoke_scripts:
            run_command([sys.executable, str(script)], cwd=REPO_ROOT)

    manifest = []
    for family, profile, rank_count, mode in build_cases(args.families, args.profiles, args.rank_counts, args.modes):
        label = f"{family}-{profile}-np{rank_count}-{mode}"
        snapshot_path = args.output_dir / f"{label}.json"
        cmd = [
            sys.executable,
            str(BENCHMARK),
            "--family",
            family,
            "--profile",
            profile,
            "--mode",
            mode,
            "--label",
            label,
            "--snapshot-out",
            str(snapshot_path),
        ]
        if args.min_seconds is not None:
            cmd.extend(["--min-seconds", str(args.min_seconds)])
        if args.min_iterations is not None:
            cmd.extend(["--min-iterations", str(args.min_iterations)])
        if profile == "small" and rank_count == 1:
            cmd.append("--scipy")
        if rank_count > 1:
            cmd = ["mpirun", "-np", str(rank_count)] + cmd
        run_command(cmd, cwd=REPO_ROOT)
        manifest.append({"label": label, "snapshot": str(snapshot_path)})

    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump({"cases": manifest}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - exercised by direct script use.
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parent
BENCHMARK = REPO_ROOT / "tests" / "benchmark_large_scale.py"

PLOT_STYLE = {
    "font.size": 14,
    "font.family": "sans-serif",
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "lines.linewidth": 2.5,
    "lines.markersize": 10,
    "figure.figsize": (10, 6),
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "savefig.bbox": "tight",
}


@dataclass
class BenchmarkResult:
    label: str
    x_value: int
    mode: str
    matrix_kind: str
    baseline: dict[str, object]
    timings: dict[str, float]
    comparisons: dict[str, object]
    command: str


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one integer")
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def format_seconds(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.6g}"


def format_speedup(baseline: float | None, vbcsr: float | None) -> str:
    if baseline is None or vbcsr is None:
        return "-"
    if not math.isfinite(baseline) or not math.isfinite(vbcsr) or vbcsr <= 0.0:
        return "-"
    return f"{baseline / vbcsr:.2f}x"


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def density_for_blocks(blocks: int, target_neighbors: int) -> float:
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    return min(float(target_neighbors) / float(blocks), 1.0)


def run_benchmark(
    *,
    label: str,
    x_value: int,
    blocks: int,
    mode: str,
    args: argparse.Namespace,
    snapshot_dir: Path,
    num_vecs: int | None = None,
) -> BenchmarkResult | None:
    density = density_for_blocks(blocks, args.target_neighbors)
    snapshot_path = snapshot_dir / f"{label}.json"

    cmd = [
        sys.executable,
        str(BENCHMARK),
        "--family",
        args.family,
        "--blocks",
        str(blocks),
        "--min-block",
        str(args.min_block),
        "--max-block",
        str(args.max_block),
        "--density",
        str(density),
        "--mode",
        mode,
        "--min-seconds",
        str(args.min_seconds),
        "--min-iterations",
        str(args.min_iterations),
        "--seed",
        str(args.seed),
        "--label",
        label,
        "--snapshot-out",
        str(snapshot_path),
    ]
    if args.scipy:
        cmd.append("--scipy")
    if args.use_mkl:
        cmd.append("--mkl")
    if num_vecs is not None:
        cmd.extend(["--num-vecs", str(num_vecs)])
    if args.complex:
        cmd.append("--complex")

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(f"Benchmark failed for {label}:")
        print(result.stdout)
        print(result.stderr)
        return None
    if not snapshot_path.exists():
        print(f"Benchmark did not write a snapshot for {label}. Output was:")
        print(result.stdout)
        print(result.stderr)
        return None

    with snapshot_path.open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)

    timings = {
        key: float(value)
        for key, value in snapshot.get("timings", {}).items()
        if isinstance(value, (int, float))
    }
    comparisons = snapshot.get("comparisons", {})
    return BenchmarkResult(
        label=label,
        x_value=x_value,
        mode=str(snapshot.get("mode", mode)),
        matrix_kind=str(snapshot.get("matrix_kind", "unknown")),
        baseline=snapshot.get("baseline", {}),
        timings=timings,
        comparisons=comparisons if isinstance(comparisons, dict) else {},
        command=str(snapshot.get("command", " ".join(cmd))),
    )


def finite_series(results: Iterable[BenchmarkResult], key: str) -> bool:
    return any(math.isfinite(result.timings.get(key, math.nan)) for result in results)


def plot_performance(
    results: list[BenchmarkResult],
    *,
    title: str,
    filename: Path,
    xlabel: str,
) -> bool:
    if not results:
        print(f"Skipping {filename.name}: no successful benchmark data")
        return False

    series = [
        ("scipy", "SciPy CSR", "#E63946"),
        ("mkl", "sparse_dot_mkl", "#457B9D"),
        ("vbcsr", "VBCSR", "#2A9D8F"),
    ]
    active_series = [
        item
        for item in series
        if item[0] == "vbcsr" or finite_series(results, item[0])
    ]
    if not active_series:
        print(f"Skipping {filename.name}: no plottable timing series")
        return False

    x = [float(i) for i in range(len(results))]
    width = min(0.8 / len(active_series), 0.25)
    fig, ax = plt.subplots(figsize=(10, 6))

    for series_idx, (key, label, color) in enumerate(active_series):
        offset = (series_idx - (len(active_series) - 1) / 2.0) * width
        values = [result.timings.get(key, math.nan) for result in results]
        ax.bar(
            [position + offset for position in x],
            values,
            width,
            label=label,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel("Execution Time (s)", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([str(result.x_value) for result in results])
    ax.legend(frameon=True, fancybox=True, shadow=True)

    for i, result in enumerate(results):
        vbcsr_time = result.timings.get("vbcsr")
        annotations = []
        for key, label, _ in active_series:
            if key == "vbcsr":
                continue
            text = format_speedup(result.timings.get(key), vbcsr_time)
            if text != "-":
                annotations.append(f"{text} vs {label.split()[0]}")
        if not annotations:
            continue

        heights = [
            result.timings.get(key, math.nan)
            for key, _, _ in active_series
            if math.isfinite(result.timings.get(key, math.nan))
        ]
        if not heights:
            continue
        ax.annotate(
            "\n".join(annotations),
            xy=(x[i], max(heights)),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        )

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")
    return True


def write_results_table(handle, results: list[BenchmarkResult]) -> None:
    handle.write(
        "| Case | Kind | MKL format | VBCSR (s) | SciPy (s) | MKL (s) | "
        "SciPy/VBCSR | MKL/VBCSR | SciPy/MKL |\n"
    )
    handle.write("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for result in results:
        vbcsr_time = result.timings.get("vbcsr")
        scipy_time = result.timings.get("scipy")
        mkl_time = result.timings.get("mkl")
        mkl_format = result.baseline.get("mkl_sparse_format", "-")
        mkl_blocksize = result.baseline.get("mkl_blocksize")
        if mkl_blocksize is not None:
            mkl_format = f"{mkl_format} {mkl_blocksize}"
        handle.write(
            f"| {result.x_value} | {result.matrix_kind} | "
            f"{mkl_format} | "
            f"{format_seconds(vbcsr_time)} | "
            f"{format_seconds(scipy_time)} | "
            f"{format_seconds(mkl_time)} | "
            f"{format_speedup(scipy_time, vbcsr_time)} | "
            f"{format_speedup(mkl_time, vbcsr_time)} | "
            f"{format_speedup(scipy_time, mkl_time)} |\n"
        )
    handle.write("\n")


def write_report(
    *,
    output_dir: Path,
    spmv_results: list[BenchmarkResult],
    dense_results: list[BenchmarkResult],
    sparse_results: list[BenchmarkResult],
    include_mkl: bool,
) -> None:
    report_path = output_dir / "benchmark_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("# VBCSR Performance Benchmarks\n\n")
        handle.write(
            "Generated from `tests/benchmark_large_scale.py` using its JSON snapshots. "
            "Modes are mapped to the current benchmark driver as follows: "
            "`mult` for SpMV, `mult_dense` for sparse-dense matrix multiplication, "
            "and `spmm` for sparse matrix-matrix multiplication.\n\n"
        )
        handle.write(
            "The MKL baseline is measured through the Python `sparse_dot_mkl` package. "
            "For small SpMV cases this includes per-call Python wrapper and MKL sparse "
            "handle setup costs, so it is not the same measurement as a persistent "
            "C++ MKL handle cache.\n\n"
        )
        if not include_mkl:
            handle.write("MKL comparison was not enabled or `sparse_dot_mkl` was unavailable.\n\n")

        handle.write("## 1. SpMV Performance\n\n")
        handle.write("Matrix-vector multiplication (`A * x`).\n\n")
        if spmv_results:
            handle.write("![SpMV Benchmark](benchmark_spmv.png)\n\n")
            write_results_table(handle, spmv_results)
        else:
            handle.write("No successful SpMV benchmark runs.\n\n")

        handle.write("## 2. Sparse-Dense Performance\n\n")
        handle.write("Sparse matrix-dense matrix multiplication (`A * X`), scaling RHS count.\n\n")
        if dense_results:
            handle.write("![Sparse-Dense Benchmark](benchmark_mult_dense_k.png)\n\n")
            write_results_table(handle, dense_results)
        else:
            handle.write("No successful sparse-dense benchmark runs.\n\n")

        handle.write("## 3. Sparse Matrix-Matrix Performance\n\n")
        handle.write("Sparse matrix-matrix multiplication (`A * A`).\n\n")
        if sparse_results:
            handle.write("![Sparse Matrix-Matrix Benchmark](benchmark_spmm.png)\n\n")
            write_results_table(handle, sparse_results)
        else:
            handle.write("No successful sparse matrix-matrix benchmark runs.\n\n")

    print(f"Saved {report_path}")


def write_json_summary(output_dir: Path, sections: dict[str, list[BenchmarkResult]]) -> None:
    summary = {
        section: [
            {
                "label": result.label,
                "x_value": result.x_value,
                "mode": result.mode,
                "matrix_kind": result.matrix_kind,
                "baseline": result.baseline,
                "timings": result.timings,
                "comparisons": result.comparisons,
                "command": result.command,
            }
            for result in results
        ]
        for section, results in sections.items()
    }
    path = output_dir / "benchmark_report_data.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Saved {path}")


def collect_series(
    *,
    section: str,
    values: list[int],
    mode: str,
    args: argparse.Namespace,
    snapshot_dir: Path,
    blocks_for_value,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    print(f"\n--- Running {section} ---")
    for value in values:
        blocks, num_vecs = blocks_for_value(value)
        label = f"{section.lower().replace(' ', '_')}_{value}"
        print(f"  {label}: blocks={blocks}", end="")
        if num_vecs is not None:
            print(f", num_vecs={num_vecs}", end="")
        print("...", flush=True)
        result = run_benchmark(
            label=label,
            x_value=value,
            blocks=blocks,
            mode=mode,
            args=args,
            snapshot_dir=snapshot_dir,
            num_vecs=num_vecs,
        )
        if result is not None:
            results.append(result)
            print("  Done.")
        else:
            print("  Failed.")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark plots and a Markdown report from tests/benchmark_large_scale.py",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for report images and data")
    parser.add_argument("--family", default="random", choices=["random", "csr", "bsr", "vbcsr"], help="Matrix family passed to benchmark_large_scale.py")
    parser.add_argument("--min-block", type=int, default=16, help="Minimum block size for random family")
    parser.add_argument("--max-block", type=int, default=20, help="Maximum block size for random family")
    parser.add_argument("--spmv-blocks", type=parse_int_list, default=parse_int_list("500,1000,2000"), help="Comma-separated block counts for SpMV")
    parser.add_argument("--dense-blocks", type=int, default=1000, help="Fixed block count for sparse-dense runs")
    parser.add_argument("--k-values", type=parse_int_list, default=parse_int_list("16,32,64"), help="Comma-separated RHS counts for sparse-dense runs")
    parser.add_argument("--sparse-blocks", type=parse_int_list, default=parse_int_list("100,300,500"), help="Comma-separated block counts for sparse matrix-matrix runs")
    parser.add_argument("--target-neighbors", type=int, default=200, help="Approximate row neighbor target used to derive density")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-seconds", type=float, default=1.0)
    parser.add_argument("--min-iterations", type=int, default=5)
    parser.add_argument("--complex", action="store_true", help="Use complex-valued benchmark matrices")
    parser.add_argument("--no-scipy", dest="scipy", action="store_false", help="Do not run SciPy comparison")
    parser.set_defaults(scipy=True)
    parser.add_argument(
        "--mkl",
        choices=["auto", "on", "off"],
        default="auto",
        help="Run sparse_dot_mkl comparison: auto uses it only when importable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if plt is None:
        print(
            "matplotlib is required to generate benchmark plots. "
            "Activate the vbcsr conda environment or install matplotlib.",
            file=sys.stderr,
        )
        print(f"Import error: {MATPLOTLIB_IMPORT_ERROR}", file=sys.stderr)
        return 1

    if args.min_block > args.max_block:
        print("--min-block must be <= --max-block", file=sys.stderr)
        return 1

    mkl_available = module_available("sparse_dot_mkl")
    args.use_mkl = args.mkl == "on" or (args.mkl == "auto" and mkl_available)
    if args.mkl == "on" and not mkl_available:
        print("sparse_dot_mkl is not importable, but --mkl=on was requested.", file=sys.stderr)
        return 1
    if args.mkl == "auto" and not mkl_available:
        print("sparse_dot_mkl is not importable; generating report without MKL comparison.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLOT_STYLE)

    with tempfile.TemporaryDirectory(prefix="vbcsr_benchmark_snapshots_") as tmp:
        snapshot_dir = Path(tmp)
        spmv_results = collect_series(
            section="SpMV",
            values=args.spmv_blocks,
            mode="mult",
            args=args,
            snapshot_dir=snapshot_dir,
            blocks_for_value=lambda blocks: (blocks, None),
        )
        dense_results = collect_series(
            section="Sparse Dense",
            values=args.k_values,
            mode="mult_dense",
            args=args,
            snapshot_dir=snapshot_dir,
            blocks_for_value=lambda k: (args.dense_blocks, k),
        )
        sparse_results = collect_series(
            section="Sparse Matrix Matrix",
            values=args.sparse_blocks,
            mode="spmm",
            args=args,
            snapshot_dir=snapshot_dir,
            blocks_for_value=lambda blocks: (blocks, None),
        )

    plot_performance(
        spmv_results,
        title="SpMV Performance: VBCSR vs Baselines",
        filename=args.output_dir / "benchmark_spmv.png",
        xlabel="Number of Blocks",
    )
    plot_performance(
        dense_results,
        title=f"Sparse-Dense Performance (Blocks={args.dense_blocks})",
        filename=args.output_dir / "benchmark_mult_dense_k.png",
        xlabel="Number of RHS Vectors (K)",
    )
    plot_performance(
        sparse_results,
        title="Sparse Matrix-Matrix Performance",
        filename=args.output_dir / "benchmark_spmm.png",
        xlabel="Number of Blocks",
    )
    sections = {
        "spmv": spmv_results,
        "mult_dense": dense_results,
        "spmm": sparse_results,
    }
    write_json_summary(args.output_dir, sections)
    write_report(
        output_dir=args.output_dir,
        spmv_results=spmv_results,
        dense_results=dense_results,
        sparse_results=sparse_results,
        include_mkl=args.use_mkl,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

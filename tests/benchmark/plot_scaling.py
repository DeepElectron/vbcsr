#!/usr/bin/env python3
"""Publication scaling figures from the run_scaling.sh sweep.

Reads tests/benchmark/results/scaling/scaling_<mode>_w<workers>.csv and writes:
  - scaling_strong_<stamp>.{pdf,png}: 2 rows (OpenMP threads, MPI ranks) x 3 op
    panels; speedup S(p)=T(1)/T(p) vs worker count, log-log, with the ideal
    linear reference.
Lines are the three storage domains (CSR / BSR / VBCSR); the library under test
is VBCSR in every panel.

Weak scaling is not plotted: it is not measurable on a single node, where all
ranks share one set of memory controllers. See run_scaling.sh.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SCALING_DIR = SCRIPT_DIR / "results" / "scaling"

OPERATION_ORDER = ("spmv", "spmm", "spgemm")
OPERATION_LABELS = {"spmv": "SpMV", "spmm": "SpMM", "spgemm": "SpGEMM"}
DOMAIN_ORDER = ("csr", "bsr", "vbcsr")
DOMAIN_LABELS = {"csr": "CSR", "bsr": "BSR", "vbcsr": "VBCSR"}
DOMAIN_STYLE = {
    "csr": ("#4c78a8", "o"),
    "bsr": ("#dd8452", "s"),
    "vbcsr": ("#0e7c86", "^"),
}
IDEAL_COLOR = "#9aa5b1"
TEXT_MUTED = "#4b5563"
GRID_MAJOR = "#dde2e8"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#3b4552",
            "lines.linewidth": 1.5,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def worker_from_name(path: Path) -> int:
    match = re.search(r"_w(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"cannot parse worker count from {path.name}")
    return int(match.group(1))


def load_mode(prefix: str) -> dict[tuple[str, str], dict[int, float]]:
    """Return {(domain, op): {workers: median_seconds}} for one sweep mode."""
    series: dict[tuple[str, str], dict[int, float]] = {}
    for path in sorted(SCALING_DIR.glob(f"{prefix}_w*.csv")):
        workers = worker_from_name(path)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("suite") is None:
                    continue
                domain = row.get("domain")
                operation = row.get("operation")
                value = row.get("vbcsr_median_seconds")
                if domain not in DOMAIN_ORDER or operation not in OPERATION_ORDER:
                    continue
                if value in (None, ""):
                    continue
                series.setdefault((domain, operation), {})[workers] = float(value)
    if not series:
        raise SystemExit(f"No data found for prefix {prefix!r} in {SCALING_DIR}")
    return series


def check_cache_residency(prefix: str, roofline_gbs: float) -> list[str]:
    """Flag sweep points whose apparent bandwidth exceeds what DRAM can deliver.

    A strong-scaling problem that fits in the machine's aggregate last-level
    cache is served from cache, not memory, and reports speedups that are an
    artifact of the timing loop rather than parallel efficiency. Exceeding the
    measured streaming roofline is proof of it -- no arrangement of DRAM
    accesses can beat the memory system's peak.

    Merely clearing aggregate L3 is not enough: a 256 MB problem against
    ~192 MB of L3 still tripped this. Size several times above it.
    """
    warnings: list[str] = []
    for path in sorted(SCALING_DIR.glob(f"{prefix}_w*.csv")):
        workers = worker_from_name(path)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                seconds = row.get("vbcsr_median_seconds")
                storage = row.get("estimated_vbcsr_storage_bytes_sum")
                if not seconds or not storage or row.get("operation") != "spmv":
                    continue
                achieved = float(storage) / float(seconds) / 1e9
                if achieved > roofline_gbs:
                    warnings.append(
                        f"{prefix} w={workers} {row['domain']}/{row['operation']}: "
                        f"{achieved:.1f} GB/s exceeds the {roofline_gbs:.1f} GB/s roofline "
                        f"-- working set is cache-resident, speedups are not meaningful"
                    )
    return warnings


def xaxis(axis, workers: list[int]) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xticks(workers)
    axis.set_xticklabels([str(w) for w in workers])
    axis.set_xlim(workers[0] * 0.85, workers[-1] * 1.18)
    axis.minorticks_off()
    axis.grid(True, which="major", color=GRID_MAJOR, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(width=0.6, length=2.6)


def domain_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=DOMAIN_STYLE[d][0], marker=DOMAIN_STYLE[d][1],
               linewidth=1.5, markersize=4.5, label=DOMAIN_LABELS[d])
        for d in DOMAIN_ORDER
    ]


def plot_strong(thread_series, mpi_series, workers, output_prefix, formats, dpi, footer):
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.75), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.115, top=0.885, wspace=0.24, hspace=0.34)

    rows = [("Speedup\n(shared mem., OpenMP threads)", thread_series),
            ("Speedup\n(distributed mem., MPI ranks)", mpi_series)]

    for row_index, (row_label, series) in enumerate(rows):
        for col_index, operation in enumerate(OPERATION_ORDER):
            axis = axes[row_index][col_index]
            present = sorted({w for d in DOMAIN_ORDER for w in series.get((d, operation), {})})
            lo, hi = present[0], present[-1]
            axis.plot([lo, hi], [lo, hi], color=IDEAL_COLOR, linestyle=(0, (4, 3)),
                      linewidth=1.0, zorder=1, label="ideal")
            panel_speedups = []
            for domain in DOMAIN_ORDER:
                data = series.get((domain, operation), {})
                if data.get(1) is None:
                    continue
                base = data[1]
                xs = sorted(data)
                ys = [base / data[w] for w in xs]
                panel_speedups.extend(ys)
                color, marker = DOMAIN_STYLE[domain]
                axis.plot(xs, ys, color=color, marker=marker, zorder=3)
            axis.set_yscale("log", base=2)
            # y range spans the ideal (up to max workers) and any sub-linear
            # dips (e.g. distributed CSR SpGEMM's serial->generic transition),
            # so no curve is silently clipped off-axis.
            y_hi = workers[-1] * 1.25
            y_lo = min(0.85, (min(panel_speedups) * 0.8) if panel_speedups else 0.85)
            y_lo = max(y_lo, 1.0 / 32)  # floor: keep the axis readable
            axis.set_ylim(y_lo, y_hi)
            ticks = [w for w in workers if y_lo <= w <= y_hi]
            axis.set_yticks(ticks)
            axis.set_yticklabels([str(w) for w in ticks])
            xaxis(axis, workers)
            if row_index == 0:
                axis.set_title(OPERATION_LABELS[operation], pad=5)
            if col_index == 0:
                axis.set_ylabel(row_label, fontsize=8.6, linespacing=1.3)

    for axis in axes[1]:
        axis.set_xlabel("Workers")

    handles = domain_legend_handles()
    handles.append(Line2D([0], [0], color=IDEAL_COLOR, linestyle=(0, (4, 3)),
                          linewidth=1.0, label="ideal linear"))
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
               handlelength=1.8, columnspacing=1.7)
    fig.text(0.5, 0.008, footer, ha="center", va="bottom", fontsize=7.2, color=TEXT_MUTED)
    _save(fig, output_prefix, formats, dpi)


def _save(fig, output_prefix: Path, formats, dpi: int) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_prefix.with_suffix(f".{fmt}")
        save_kwargs = {"dpi": dpi} if fmt in {"png", "jpg", "jpeg"} else {}
        fig.savefig(path, **save_kwargs)
        print(f"Wrote {path}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default="2026-07-20")
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--strong-mib", type=int, default=1024)
    parser.add_argument(
        "--roofline-gbs",
        type=float,
        default=125.7,
        help="measured streaming-read bandwidth of the host; sweep points above it are cache-resident",
    )
    args = parser.parse_args()
    formats = [f.strip().lstrip(".").lower() for f in args.formats.split(",") if f.strip()]

    residency = (check_cache_residency("scaling_thread_strong", args.roofline_gbs)
                 + check_cache_residency("scaling_mpi_strong", args.roofline_gbs))
    for warning in residency:
        print(f"WARNING: {warning}", file=sys.stderr)
    if residency:
        print(
            "WARNING: re-run with a larger --target-storage-bytes before using these figures.",
            file=sys.stderr,
        )

    thread_series = load_mode("scaling_thread_strong")
    mpi_series = load_mode("scaling_mpi_strong")

    workers = sorted({w for s in (thread_series, mpi_series) for m in s.values() for w in m})

    strong_footer = (
        f"Strong scaling: fixed global problem, {args.strong_mib} MiB of stored blocks per domain, "
        f"RHS=16. Speedup = T(1 worker) / T(p workers); dashed = ideal linear. Dual-socket EPYC 7352."
    )

    plot_strong(thread_series, mpi_series, workers,
                SCALING_DIR / f"scaling_strong_{args.stamp}", formats, args.dpi, strong_footer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

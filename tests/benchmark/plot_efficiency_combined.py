#!/usr/bin/env python3
"""Combined 1-thread + 16-thread kernel-efficiency figure (publication).

Two rows (thread counts) x three panels (SpMV / SpMM / SpGEMM), grouped bars
of median times (log scale) for SciPy, sparse-dot-mkl, and VBCSR, with the
VBCSR speedup over the fastest reference annotated per group. Data loading
and case selection are shared with plot_efficiency.py.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

from plot_efficiency import (  # noqa: E402
    DOMAIN_ORDER,
    OPERATION_ORDER,
    OPERATION_LABELS,
    RESULTS_DIR,
    block_size_labels,
    efficiency_parameters_by_domain,
    format_speedup,
    metadata_line,
    method_times,
    fastest_reference,
    paired_json_path,
    parse_float,
    read_payload,
    read_rows,
    select_rows,
)

# Method key -> (display label, fill, edge). A calm neutral for SciPy, a warm
# amber for the vendor baseline, and the VBCSR teal carrying the accent.
METHOD_STYLE = (
    ("scipy_median_seconds", "SciPy", "#aeb8c2", "#7d8894"),
    ("mkl_median_seconds", "sparse-dot-mkl", "#e3a13d", "#a86f14"),
    ("vbcsr_median_seconds", "VBCSR (this work)", "#0e7c86", "#07474d"),
)
SPEEDUP_POSITIVE = "#0f6f37"
SPEEDUP_NEGATIVE = "#b23a48"
SPEEDUP_NEUTRAL = "#52606d"  # parity band 0.95x-1.05x


def speedup_color(speedup: float) -> str:
    if speedup >= 1.05:
        return SPEEDUP_POSITIVE
    if speedup <= 0.95:
        return SPEEDUP_NEGATIVE
    return SPEEDUP_NEUTRAL


DOMAIN_FORMAT_NAMES = {"csr": "CSR", "bsr": "BSR", "vbcsr": "VBCSR"}


def two_line_domain_label(domain: str, bsz_label: str) -> str:
    # "bsz=9/13/15/20" -> "9-20" keeps the variable-block label compact.
    value = bsz_label.removeprefix("bsz=")
    if "/" in value:
        parts = value.split("/")
        value = f"{parts[0]}–{parts[-1]}"
    return f"{DOMAIN_FORMAT_NAMES[domain]}\nbsz {value}"
GRID_MAJOR = "#dde2e8"
GRID_MINOR = "#f0f3f6"
TEXT_MUTED = "#4b5563"


def format_seconds_compact(value: float, _pos=None) -> str:
    if value >= 1.0:
        return f"{value:g}s"
    if value >= 1e-3:
        return f"{value * 1e3:g}ms"
    if value >= 1e-6:
        return f"{value * 1e6:g}µs"
    return f"{value * 1e9:g}ns"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#3b4552",
            "text.color": "#1f2733",
            "axes.labelcolor": "#1f2733",
            "xtick.color": "#1f2733",
            "ytick.color": "#3b4552",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def load_selected(csv_path: Path, spgemm_threshold: float):
    rows = read_rows(csv_path)
    payload = read_payload(paired_json_path(csv_path))
    selected = select_rows(rows, spgemm_threshold)
    labels = block_size_labels(selected, efficiency_parameters_by_domain(payload))
    return selected, labels


def panel_payload(selected, operation):
    items = []
    for domain in DOMAIN_ORDER:
        row = selected[(operation, domain)]
        times = method_times(row, require_all=True)
        vbcsr = parse_float(row.get("vbcsr_median_seconds"))
        items.append(
            {
                "domain": domain,
                "times": {label: value for label, value, _c, _e in times},
                "speedup": fastest_reference(row) / vbcsr,
            }
        )
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-1t", type=Path, required=True)
    parser.add_argument("--csv-16t", type=Path, required=True)
    parser.add_argument("--threads-16t", type=int, default=16)
    parser.add_argument("--spgemm-threshold", type=float, default=0.0)
    parser.add_argument("--output-prefix", type=Path, default=RESULTS_DIR / "kernel_efficiency_combined")
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    set_style()

    row_specs = [
        ("1 thread", *load_selected(args.csv_1t, args.spgemm_threshold)),
        (f"{args.threads_16t} threads", *load_selected(args.csv_16t, args.spgemm_threshold)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.75))
    fig.subplots_adjust(
        left=0.088, right=0.995, bottom=0.125, top=0.855, wspace=0.10, hspace=0.52
    )

    bar_width = 0.24
    offsets = [-bar_width, 0.0, bar_width]
    legend_handles: dict[str, object] = {}

    for row_index, (row_label, selected, domain_labels) in enumerate(row_specs):
        # One log range per thread-count row: the two rows differ by ~10x and
        # forcing a shared range would flatten the 16T bars.
        row_times: list[float] = []
        panels = {}
        for operation in OPERATION_ORDER:
            panels[operation] = panel_payload(selected, operation)
            for item in panels[operation]:
                row_times.extend(item["times"].values())
        ymin = 10 ** math.floor(math.log10(min(row_times))) / 1.5
        ymax = 10 ** math.ceil(math.log10(max(row_times))) * 2.2

        for col_index, operation in enumerate(OPERATION_ORDER):
            axis = axes[row_index][col_index]
            x_positions = list(range(len(DOMAIN_ORDER)))

            for method_index, (key, label, fill, edge) in enumerate(METHOD_STYLE):
                display = label.split(" (")[0]
                y_values = [item["times"][display if display in item["times"] else label] for item in panels[operation]]
                bars = axis.bar(
                    [x + offsets[method_index] for x in x_positions],
                    y_values,
                    width=bar_width * 0.9,
                    color=fill,
                    edgecolor=edge,
                    linewidth=0.4,
                    zorder=3,
                )
                legend_handles.setdefault(label, bars[0])

            for x, item in zip(x_positions, panels[operation]):
                speedup = float(item["speedup"])
                axis.text(
                    x,
                    max(item["times"].values()) * 1.25,
                    format_speedup(speedup),
                    ha="center",
                    va="bottom",
                    color=speedup_color(speedup),
                    fontsize=7.6,
                    fontweight="bold",
                    zorder=4,
                )

            if row_index == 0:
                axis.set_title(OPERATION_LABELS[operation], pad=6)
            axis.set_xticks(x_positions)
            axis.set_xticklabels(
                [two_line_domain_label(domain, domain_labels[domain]) for domain in DOMAIN_ORDER],
                linespacing=1.25,
            )
            axis.set_yscale("log")
            axis.set_ylim(ymin, ymax)
            axis.set_xlim(-0.55, len(DOMAIN_ORDER) - 0.45)
            axis.grid(True, axis="y", which="major", color=GRID_MAJOR, linewidth=0.6, zorder=0)
            axis.grid(True, axis="y", which="minor", color=GRID_MINOR, linewidth=0.4, zorder=0)
            axis.set_axisbelow(True)
            axis.tick_params(axis="x", length=0)
            axis.tick_params(axis="y", width=0.6, length=2.6)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
            axis.yaxis.set_major_locator(LogLocator(base=10))
            axis.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
            if col_index == 0:
                axis.yaxis.set_major_formatter(FuncFormatter(format_seconds_compact))
            else:
                axis.set_yticklabels([])
                axis.tick_params(axis="y", which="both", length=0)

        # Row tag: thread count, anchored to the row's left panel.
        axes[row_index][0].annotate(
            row_label,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(1, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            color="#1f2733",
        )
        axes[row_index][0].set_ylabel("Median time")

    fig.legend(
        [legend_handles[label] for _k, label, _f, _e in METHOD_STYLE],
        [label for _k, label, _f, _e in METHOD_STYLE],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        handlelength=1.35,
        handleheight=1.0,
        columnspacing=1.6,
    )

    _first_row_selected = row_specs[0][1]
    footer = metadata_line(_first_row_selected.values(), args.spgemm_threshold)
    fig.text(
        0.5,
        0.012,
        f"{footer}. bsz: DOFs per graph node. Labels: VBCSR speedup vs. the fastest reference; >1× is faster.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color=TEXT_MUTED,
    )

    written = []
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in (item.strip().lstrip(".").lower() for item in args.formats.split(",") if item.strip()):
        path = args.output_prefix.with_suffix(f".{fmt}")
        save_kwargs = {"dpi": args.dpi} if fmt in {"png", "jpg", "jpeg", "tif", "tiff"} else {}
        fig.savefig(path, **save_kwargs)
        written.append(path)
    plt.close(fig)
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

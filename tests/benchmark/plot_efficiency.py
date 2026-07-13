#!/usr/bin/env python3
"""Create a publication-ready kernel-efficiency figure from benchmark CSV data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
DOMAIN_ORDER = ("csr", "bsr", "vbcsr")
OPERATION_ORDER = ("spmv", "spmm", "spgemm")
OPERATION_LABELS = {"spmv": "SpMV", "spmm": "SpMM", "spgemm": "SpGEMM"}
METHODS = (
    ("scipy_median_seconds", "SciPy", "#7f8b96", "#4f5963"),
    ("mkl_median_seconds", "sparse-dot-mkl", "#d98c00", "#915f00"),
    ("vbcsr_median_seconds", "VBCSR", "#00777f", "#03464b"),
)


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def format_seconds(value: float, _position: int | None = None) -> str:
    if value >= 1.0:
        return f"{value:g} s"
    if value >= 1e-3:
        return f"{value * 1e3:g} ms"
    if value >= 1e-6:
        return f"{value * 1e6:g} us"
    return f"{value * 1e9:g} ns"


def format_speedup(value: float) -> str:
    if value >= 10.0:
        return f"{value:.0f}x"
    if value >= 1.0:
        return f"{value:.2g}x"
    if value >= 0.1:
        return f"{value:.2f}x"
    return f"{value:.3f}x"


def latest_csv(results_dir: Path) -> Path:
    candidates = [
        path
        for path in results_dir.iterdir()
        if path.is_file()
        and (path.suffix == ".csv" or path.name == ".csv")
        and not path.name.startswith("efficiency_summary")
        and not path.name.startswith("kernel_efficiency")
    ]
    if not candidates:
        raise SystemExit(f"No benchmark CSV files found in {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    return rows


def paired_json_path(csv_path: Path) -> Path | None:
    if csv_path.name == ".csv":
        candidate = csv_path.with_name(".json")
    else:
        candidate = csv_path.with_suffix(".json")
    return candidate if candidate.exists() else None


def read_payload(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def efficiency_parameters_by_domain(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for case in payload.get("cases", []):
        if not isinstance(case, dict):
            continue
        if case.get("suite") != "efficiency" or case.get("operation") != "spmv":
            continue
        domain = case.get("domain")
        params = case.get("parameters")
        if isinstance(domain, str) and isinstance(params, dict):
            result[domain] = params
    return result


def threshold_value(row: dict[str, str]) -> float | None:
    value = row.get("spgemm_threshold")
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def choose_spgemm_row(rows: list[dict[str, str]], threshold: float) -> dict[str, str]:
    with_threshold = [(row, threshold_value(row)) for row in rows if threshold_value(row) is not None]
    if not with_threshold:
        raise SystemExit("SpGEMM rows do not contain spgemm_threshold values")
    exact = [
        row
        for row, value in with_threshold
        if value is not None and math.isclose(value, threshold, rel_tol=1e-9, abs_tol=1e-300)
    ]
    if exact:
        return exact[0]
    row, value = min(with_threshold, key=lambda item: abs(float(item[1]) - threshold))
    print(
        f"Requested SpGEMM threshold {threshold:g} is unavailable; using {float(value):g}.",
        file=sys.stderr,
    )
    return row


def select_rows(rows: list[dict[str, str]], spgemm_threshold: float) -> dict[tuple[str, str], dict[str, str]]:
    efficiency_rows = [row for row in rows if row.get("suite") == "efficiency"]
    if not efficiency_rows:
        raise SystemExit("No efficiency rows found in benchmark CSV")

    selected: dict[tuple[str, str], dict[str, str]] = {}
    for operation in OPERATION_ORDER:
        for domain in DOMAIN_ORDER:
            candidates = [
                row
                for row in efficiency_rows
                if row.get("operation") == operation and row.get("domain") == domain
            ]
            if not candidates:
                raise SystemExit(f"Missing efficiency row for {operation}/{domain}")
            if operation == "spgemm":
                selected[(operation, domain)] = choose_spgemm_row(candidates, spgemm_threshold)
            else:
                selected[(operation, domain)] = candidates[0]
    return selected


def method_times(row: dict[str, str], *, require_all: bool) -> list[tuple[str, float, str, str]]:
    values = []
    missing = []
    for key, label, color, edgecolor in METHODS:
        value = parse_float(row.get(key))
        if value is None:
            missing.append(label)
            continue
        values.append((label, value, color, edgecolor))
    if require_all and missing:
        raise SystemExit(f"Missing method time(s) for {row.get('label', '<unknown>')}: {', '.join(missing)}")
    if not values:
        raise SystemExit(f"No plottable method times found for {row.get('label', '<unknown>')}")
    return values


def fastest_reference(row: dict[str, str]) -> float:
    values = []
    for key in ("scipy_median_seconds", "mkl_median_seconds"):
        value = parse_float(row.get(key))
        if value is not None:
            values.append(value)
    if not values:
        raise SystemExit(f"No reference time found for {row.get('label', '<unknown>')}")
    return min(values)


def infer_uniform_block_size(row: dict[str, str]) -> int | None:
    blocks = parse_float(row.get("blocks"))
    scalar_rows = parse_float(row.get("scalar_rows"))
    if blocks is None or scalar_rows is None:
        return None
    ratio = scalar_rows / blocks
    rounded = int(round(ratio))
    return rounded if abs(ratio - rounded) < 1e-6 else None


def block_size_labels(
    selected: dict[tuple[str, str], dict[str, str]],
    params_by_domain: dict[str, dict[str, object]],
) -> dict[str, str]:
    labels = {"csr": "bsz=1", "bsr": "bsz=?", "vbcsr": "bsz=var"}

    bsr_params = params_by_domain.get("bsr", {})
    bsr_size = bsr_params.get("bsr_block_size")
    if isinstance(bsr_size, int):
        labels["bsr"] = f"bsz={bsr_size}"
    else:
        inferred = infer_uniform_block_size(selected[("spmv", "bsr")])
        if inferred is not None:
            labels["bsr"] = f"bsz={inferred}"

    vbcsr_params = params_by_domain.get("vbcsr", {})
    vbcsr_sizes = vbcsr_params.get("vbcsr_block_sizes")
    if isinstance(vbcsr_sizes, list) and vbcsr_sizes:
        sizes = [str(int(size)) for size in vbcsr_sizes]
        labels["vbcsr"] = "bsz=" + "/".join(sizes)
    return labels


def metadata_line(rows: Iterable[dict[str, str]], spgemm_threshold: float) -> str:
    first = next(iter(rows))
    blocks = first.get("blocks", "?")
    rhs = first.get("rhs", "?")
    degree = first.get("degree_mean", "")
    degree_text = ""
    if degree:
        try:
            degree_text = f", mean degree {float(degree):.1f}"
        except ValueError:
            degree_text = f", mean degree {degree}"
    decay = first.get("magnitude_decay_length", "")
    decay_text = f", decay length {decay}" if decay else ""
    return f"N={blocks} blocks{degree_text}, RHS={rhs}, SpGEMM threshold {spgemm_threshold:g}{decay_text}"


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 10.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def plot_efficiency(
    selected: dict[tuple[str, str], dict[str, str]],
    *,
    domain_labels: dict[str, str],
    output_prefix: Path,
    spgemm_threshold: float,
    dpi: int,
    title: str | None,
    formats: list[str],
    require_all_methods: bool,
) -> list[Path]:
    set_plot_style()

    positive_color = "#1b7f3a"
    negative_color = "#b23a48"
    bar_width = 0.23
    offsets = [-bar_width, 0.0, bar_width]

    all_times: list[float] = []
    panel_data: dict[str, list[dict[str, object]]] = {}
    for operation in OPERATION_ORDER:
        panel_rows = []
        for domain in DOMAIN_ORDER:
            row = selected[(operation, domain)]
            times = method_times(row, require_all=require_all_methods)
            vbcsr = parse_float(row.get("vbcsr_median_seconds"))
            if vbcsr is None:
                raise SystemExit(f"Missing VBCSR time for {operation}/{domain}")
            speedup = fastest_reference(row) / vbcsr
            panel_rows.append({"domain": domain, "times": times, "speedup": speedup, "vbcsr": vbcsr})
            all_times.extend(time for _, time, _, _ in times)
        panel_data[operation] = panel_rows

    ymin = 10 ** math.floor(math.log10(min(all_times))) / 1.6
    ymax = 10 ** math.ceil(math.log10(max(all_times))) * 1.8

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.70), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.26, top=0.78, wspace=0.08)
    if title:
        fig.suptitle(title, y=0.98, fontsize=10.5)

    legend_by_label = {}
    for axis, operation in zip(axes, OPERATION_ORDER):
        x_positions = list(range(len(DOMAIN_ORDER)))
        for method_index, (method_key, method_label, color, edgecolor) in enumerate(METHODS):
            x_values = []
            y_values = []
            for x, item in zip(x_positions, panel_data[operation]):
                times = {
                    label: (time, item_color, item_edgecolor)
                    for label, time, item_color, item_edgecolor in item["times"]  # type: ignore[index]
                }
                if method_label not in times:
                    continue
                time, _, _ = times[method_label]
                x_values.append(x + offsets[method_index])
                y_values.append(time)
            if not y_values:
                continue
            bars = axis.bar(
                x_values,
                y_values,
                width=bar_width * 0.92,
                color=color,
                edgecolor=edgecolor,
                linewidth=0.45,
                label=method_label,
            )
            legend_by_label.setdefault(method_label, bars[0])

        for x, item in zip(x_positions, panel_data[operation]):
            times = [time for _, time, _, _ in item["times"]]  # type: ignore[index]
            speedup = float(item["speedup"])
            label_y = max(times) * 1.22
            axis.text(
                x,
                label_y,
                format_speedup(speedup),
                ha="center",
                va="bottom",
                color=positive_color if speedup >= 1.0 else negative_color,
                fontsize=7.5,
                fontweight="bold",
            )

        axis.set_title(OPERATION_LABELS[operation])
        axis.set_xticks(x_positions)
        axis.set_xticklabels([domain_labels[domain] for domain in DOMAIN_ORDER])
        axis.set_yscale("log")
        axis.set_ylim(ymin, ymax)
        axis.grid(True, axis="y", which="major", color="#d6dbe0", linewidth=0.6)
        axis.grid(True, axis="y", which="minor", color="#edf0f2", linewidth=0.4)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", length=0)
        axis.tick_params(axis="y", width=0.6)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)

    axes[0].set_ylabel("Median time per operation")
    axes[0].yaxis.set_major_locator(LogLocator(base=10))
    axes[0].yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    axes[0].yaxis.set_major_formatter(FuncFormatter(format_seconds))

    if legend_by_label:
        legend_labels = [label for _, label, _, _ in METHODS if label in legend_by_label]
        legend_handles = [legend_by_label[label] for label in legend_labels]
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99 if title else 0.965),
            ncol=3,
            frameon=False,
            handlelength=1.4,
            columnspacing=1.5,
        )

    footer = metadata_line(selected.values(), spgemm_threshold)
    fig.text(
        0.5,
        0.035,
        f"{footer}. bsz: DOFs per graph node. Labels: VBCSR speedup vs. fastest reference; >1x is faster.",
        ha="center",
        va="bottom",
        fontsize=7.3,
        color="#4b5563",
    )

    written = []
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        path = output_prefix.with_suffix(f".{fmt}")
        save_kwargs = {"dpi": dpi} if fmt in {"png", "jpg", "jpeg", "tif", "tiff"} else {}
        fig.savefig(path, **save_kwargs)
        written.append(path)
    plt.close(fig)
    return written


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot the VBCSR kernel-efficiency benchmark as a publication figure.")
    parser.add_argument("--csv", type=Path, default=None, help="Benchmark CSV file. Defaults to newest CSV in tests/benchmark/results.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output-prefix", type=Path, default=RESULTS_DIR / "kernel_efficiency")
    parser.add_argument("--spgemm-threshold", type=float, default=0.0, help="SpGEMM threshold row to use in the main figure.")
    parser.add_argument("--formats", default="pdf,png", help="Comma-separated output formats.")
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--title", default=None)
    parser.add_argument("--allow-missing-methods", action="store_true", help="Plot available methods instead of requiring SciPy, sparse-dot-mkl, and VBCSR.")
    return parser


def main() -> int:
    args = make_parser().parse_args()
    csv_path = args.csv if args.csv is not None else latest_csv(args.results_dir)
    rows = read_rows(csv_path)
    payload = read_payload(paired_json_path(csv_path))
    selected = select_rows(rows, args.spgemm_threshold)
    domain_labels = block_size_labels(selected, efficiency_parameters_by_domain(payload))
    formats = [item.strip() for item in args.formats.split(",") if item.strip()]
    written = plot_efficiency(
        selected,
        domain_labels=domain_labels,
        output_prefix=args.output_prefix,
        spgemm_threshold=args.spgemm_threshold,
        dpi=args.dpi,
        title=args.title,
        formats=formats,
        require_all_methods=not args.allow_missing_methods,
    )
    print(f"Read {csv_path}")
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

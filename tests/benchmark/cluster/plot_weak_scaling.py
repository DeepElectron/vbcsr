#!/usr/bin/env python
"""Weak-scaling figure from cluster campaign CSV/JSON outputs.

Reads every `<prefix>*_np<N>.csv` (plus its .json for comm fractions) in a
results directory and renders, per operation:
  - weak efficiency  E(np) = T(np_ref) / T(np)   (ideal = 1.0)
  - ghost-communication fraction of the apply median (spmv/spmm)

Usage:
    python plot_weak_scaling.py --results-dir .../cluster_weak \
        --prefix weak_bsr_d500_b8000_r16x1 --output-prefix weak_d500
"""
import argparse
import collections
import csv
import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OPS = ("spmv", "spmm", "spgemm")


def load_points(results_dir: Path, prefix: str):
    medians = collections.defaultdict(dict)   # (domain, op) -> {np: seconds}
    comm_frac = collections.defaultdict(dict)  # (domain, op) -> {np: fraction}
    pattern = re.compile(re.escape(prefix) + r".*_np(\d+)\.csv$")
    for csv_path in sorted(results_dir.glob(prefix + "*_np*.csv")):
        match = pattern.search(csv_path.name)
        if not match:
            continue
        np_count = int(match.group(1))
        with io.open(csv_path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row.get("domain"), row.get("operation"))
                value = row.get("vbcsr_median_seconds") or row.get("median_seconds")
                if value:
                    medians[key][np_count] = float(value)
        json_path = csv_path.with_suffix(".json")
        if json_path.exists():
            payload = json.load(io.open(json_path, encoding="utf-8"))
            for case in payload.get("cases", []):
                key = (case.get("domain"), case.get("operation"))
                fraction = case.get("vbcsr_internal", {}).get("ghost_comm_fraction_of_median")
                if fraction is not None:
                    comm_frac[key][np_count] = float(fraction)
    return medians, comm_frac


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="weak_")
    parser.add_argument("--output-prefix", default="weak_scaling")
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    medians, comm_frac = load_points(args.results_dir, args.prefix)
    if not medians:
        raise SystemExit(f"no '{args.prefix}*_np<N>.csv' files under {args.results_dir}")

    fig, axes = plt.subplots(2, len(OPS), figsize=(4.2 * len(OPS), 7.2), sharex=True)
    domains = sorted({domain for (domain, _op) in medians})
    for col, op in enumerate(OPS):
        ax_eff = axes[0][col]
        ax_comm = axes[1][col]
        plotted_comm = False
        for domain in domains:
            series = medians.get((domain, op), {})
            if len(series) < 2:
                continue
            counts = sorted(series)
            ref = series[counts[0]]
            ax_eff.plot(counts, [ref / series[n] for n in counts], marker="o", label=domain)
            frac = comm_frac.get((domain, op), {})
            if frac:
                fcounts = sorted(frac)
                ax_comm.plot(fcounts, [frac[n] for n in fcounts], marker="s", label=domain)
                plotted_comm = True
        ax_eff.axhline(1.0, color="grey", linestyle="--", linewidth=1)
        ax_eff.set_title(op.upper())
        ax_eff.set_ylabel("weak efficiency T(ref)/T(np)" if col == 0 else "")
        ax_eff.set_ylim(bottom=0.0)
        ax_eff.set_xscale("log", base=2)
        ax_eff.grid(alpha=0.3)
        ax_comm.set_ylabel("ghost comm fraction" if col == 0 else "")
        ax_comm.set_xlabel("MPI ranks")
        ax_comm.set_ylim(0.0, 1.0)
        ax_comm.set_xscale("log", base=2)
        ax_comm.grid(alpha=0.3)
        if not plotted_comm:
            ax_comm.text(
                0.5,
                0.5,
                "not instrumented",
                ha="center",
                va="center",
                transform=ax_comm.transAxes,
                color="0.35",
            )
    axes[0][0].legend()
    fig.suptitle("VBCSR weak scaling (fixed per-rank problem)")
    fig.tight_layout()

    for fmt in args.formats.split(","):
        out = args.results_dir / f"{args.output_prefix}.{fmt.strip()}"
        fig.savefig(out, dpi=args.dpi)
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

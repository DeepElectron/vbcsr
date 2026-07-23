#!/usr/bin/env python
"""Pre-generate the on-disk geometry cache for a cluster campaign size.

Geometry generation (KD-tree neighbor search + recursive-bisection ordering)
runs on rank 0 only and can take minutes and tens of GB at million-atom
sizes — run this once per (blocks, degree, ...) on a login/fat node so the
compute allocation never pays for it. All job ranks then mmap the cached
arrays from the shared filesystem.

The cache key is the GLOBAL problem, so warm one entry per weak-scaling
point: blocks = weak_blocks_per_rank * total_ranks.

Example (weak points for 8000 blocks/rank at 1, 2, 4, 8 nodes x 64 ranks):
    for np in 64 128 256 512; do
        python warm_geometry.py --blocks $((8000 * np)) --target-degree 500 \
            --cache-dir /shared/scratch/vbcsr_geom
    done
"""
import argparse
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

import run_benchmark as rb  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, required=True, help="GLOBAL block count for this point")
    parser.add_argument("--target-degree", type=int, default=500)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--geometry-dim", type=int, default=3)
    parser.add_argument("--geometry-spacing", type=float, default=1.0)
    parser.add_argument("--geometry-jitter", type=float, default=0.12)
    parser.add_argument("--geometry-ordering", choices=("bisection", "lexicographic"), default="bisection")
    parser.add_argument("--geometry-cutoff", type=float, default=None)
    parser.add_argument("--geometry-cutoff-quantile", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    spec = rb.BenchmarkSpec(
        suite="efficiency",
        domain="bsr",
        operation="spmv",
        blocks=int(args.blocks),
        target_degree=int(args.target_degree),
        rhs=1,
        dtype=rb.np.float64,
        seed=int(args.seed),
        bsr_block_size=15,
        spgemm_threshold=0.0,
        spgemm_audit_limit=0,
        geometry_dim=int(args.geometry_dim),
        geometry_spacing=float(args.geometry_spacing),
        geometry_jitter=float(args.geometry_jitter),
        geometry_ordering=str(args.geometry_ordering),
        geometry_cutoff=args.geometry_cutoff,
        geometry_cutoff_quantile=float(args.geometry_cutoff_quantile),
        magnitude_decay_length=0.5,
        offdiagonal_scale=1.0,
        diagonal_shift=2.0,
        value_fill="random",
    )

    t0 = time.perf_counter()
    rb.cached_geometric_adjacency(spec, (0, spec.blocks), None, 0, args.cache_dir)
    print(
        "warmed cache for blocks=%d degree=%d in %.1f s -> %s"
        % (spec.blocks, spec.target_degree, time.perf_counter() - t0, args.cache_dir)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

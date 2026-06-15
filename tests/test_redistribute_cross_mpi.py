"""Cross-comm BlockSpMat.redistribute (doc/design/35 incr2) — broadcast + reduce gate.

Emulates the LCAO L1<->L3 topology: ``world`` is split into ``npool`` pools of
``domain`` ranks. L1 is a contiguous world partition (unique owners); L3 is a
contiguous per-pool partition replicated across pools (the same position owns the
same global blocks in every pool). Transport runs on ``world``.

  * Copy (send-down):  L1 (unique) --Copy--> L3 (replicated). Each block fans out to
    every pool, so each pool's L3 ends up holding the full matrix.
  * Sum  (reduce-up):  L3 (replicated partials) --Sum--> L1 (unique). Each of the
    npool source ranks holding a block contributes; the unique L1 owner accumulates,
    so the result is npool * partial.

Block sizes are non-uniform -> VBCSR backend (the real LCAO case).

    OMP_NUM_THREADS=1 mpirun --bind-to none -n {1,2,4} python tests/test_redistribute_cross_mpi.py
"""
import numpy as np
from mpi4py import MPI
from vbcsr import VBCSR
from vbcsr_core import AssemblyMode, RedistOp

world = MPI.COMM_WORLD
rank, size = world.rank, world.size

# Pool split: as many pools as cleanly divide world (1 -> degenerate single pool).
npool = 2 if size % 2 == 0 else 1
domain = size // npool
pool_id = rank // domain
position = rank % domain
pool = world.Split(pool_id, position)          # rank within pool == position

n = 6
gsizes = [2, 3, 2, 3, 4, 2]                     # non-uniform -> VBCSR
def neigh(i): return sorted({(i - 1) % n, i, (i + 1) % n})
gadj = [neigh(i) for i in range(n)]
def val(gi, gj): return float(gi * 100 + gj)

def contig(idx, parts):
    b = [round(i * n / parts) for i in range(parts + 1)]
    return list(range(b[idx], b[idx + 1]))

l1_owned = contig(rank, size)                   # L1: unique, world-contiguous
l3_owned = contig(position, domain)             # L3: per-pool, replicated across pools

def build(owned, comm, fill):
    m = VBCSR.create_distributed(owned, [gsizes[g] for g in owned], [gadj[g] for g in owned],
                                 dtype=np.float64, comm=comm)
    for gi in owned:
        for gj in gadj[gi]:
            v = val(gi, gj) if fill else 0.0
            m.add_block(gi, gj, np.full((gsizes[gi], gsizes[gj]), v, np.float64), AssemblyMode.INSERT)
    m.assemble()                                # backfills ghost block sizes
    return m

def check(mat, owned, expect, tag):
    err, nb = 0.0, 0
    for gi in owned:
        for gj in gadj[gi]:
            b = mat.get_block(gi, gj)
            if b is None:
                err = 1e30; continue
            err = max(err, float(np.max(np.abs(np.asarray(b) - expect(gi, gj))))); nb += 1
    gerr = world.allreduce(err, MPI.MAX); gnb = world.allreduce(nb, MPI.SUM)
    if rank == 0:
        print(f"[n={size} npool={npool}] {tag}: blocks={gnb} err={gerr:.3e} "
              f"-> {'PASS' if gerr < 1e-12 else 'FAIL'}", flush=True)
    return gerr < 1e-12

# --- Copy: L1 (world, unique) -> L3 (pool, replicated) = send-down/broadcast ---
src_l1 = build(l1_owned, world, fill=True)
tgt_l3 = build(l3_owned, pool, fill=False)      # held so its graph survives
res_l3 = src_l1.redistribute_cross(tgt_l3.graph, RedistOp.Copy, world, target_comm=pool)
ok = check(res_l3, l3_owned, val, "Copy L1->L3 (send-down)")

# --- Sum: L3 (pool, replicated partials) -> L1 (world, unique) = reduce-up ---
src_l3 = build(l3_owned, pool, fill=True)        # every pool holds the same partial
tgt_l1 = build(l1_owned, world, fill=False)
res_l1 = src_l3.redistribute_cross(tgt_l1.graph, RedistOp.Sum, world, target_comm=world)
ok &= check(res_l1, l1_owned, lambda gi, gj: npool * val(gi, gj), "Sum L3->L1 (reduce-up)")

raise SystemExit(0 if ok else 1)

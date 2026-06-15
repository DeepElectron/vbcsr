"""ImageContainer.redistribute_into (doc/design/35 incr3) — batched send-down + reduce-up.

A periodic chain (pbc along x) gives real multi-R images (R=0, +x, -x). The SAME
geometry is built on two partitions: L1 on ``world`` (unique atom owners) and L3 on
``pool_comm`` (per-pool, replicated across pools). All images ride one Alltoallv.

  * Copy: L1 -> L3 = send-down/broadcast (every pool's L3 gets the full image set).
  * Sum:  L3 -> L1 = reduce-up (npool pools each hold the same partial, so the L1
    result is npool * partial).

Blocks are filled by walking each container's own edges + the R=0 diagonal, so the
test assumes no partition layout. Correctness is checked by sampling each container at
several k against an independently-filled reference on the same comm.

    OMP_NUM_THREADS=1 mpirun --bind-to none -n {1,2,4} python tests/test_image_redistribute_mpi.py
"""
import numpy as np
from mpi4py import MPI
import vbcsr
from vbcsr_core import RedistOp

world = MPI.COMM_WORLD
rank, size = world.rank, world.size

npool = 2 if size % 2 == 0 else 1
domain = size // npool
pool = world.Split(rank // domain, rank % domain)

na = 4
period = float(na)                          # spacing 1, cutoff 1.5 -> +-1 neighbour + image
pos = np.array([[float(i), 0.0, 0.0] for i in range(na)], dtype=np.float64)
z = np.array([1 if i % 2 == 0 else 8 for i in range(na)], dtype=np.int32)
cell = np.diag([period, 20.0, 20.0]).astype(np.float64)
pbc = [True, False, False]
cutoff = {1: 1.5, 8: 1.5}
norb = {1: 2, 8: 3}
nb = {i: norb[int(z[i])] for i in range(na)}

def val(R, gi, gj):
    return float(1000 * (R[0] + 5) + 100 * gi + gj)

def make(comm):
    ad = vbcsr.AtomicData.from_points(pos, z, cell, pbc, cutoff, norb, comm=comm)
    return ad, vbcsr.ImageContainer(ad, dtype=np.float64)

def fill(ad, ic, value):
    """Fill every owned block (edges + R0 diagonal) with value(R, gi, gj)."""
    g = ad.graph
    ei = np.asarray(ad.edge_index)
    es = np.asarray(ad.edge_shift)
    seen = set()
    for e in range(ei.shape[0]):
        gi = int(g.get_global_index(int(ei[e, 0])))
        gj = int(g.get_global_index(int(ei[e, 1])))
        R = [int(es[e, 0]), int(es[e, 1]), int(es[e, 2])]
        ic.add_block(gi, gj, np.full((nb[gi], nb[gj]), value(R, gi, gj), np.float64),
                     R=R, mode="insert")
        seen.add((tuple(R), gi, gj))
    for gi in (int(x) for x in np.asarray(ad.atom_indices)):
        if ((0, 0, 0), gi, gi) not in seen:
            ic.add_block(gi, gi, np.full((nb[gi], nb[gi]), value([0, 0, 0], gi, gi), np.float64),
                         R=[0, 0, 0], mode="insert")
    ic.assemble()

def check(comm, ad, ic, expect, tag):
    """Sample ic and an independently-filled reference at several k; compare blocks."""
    ref_ad, ref = make(comm)
    fill(ref_ad, ref, expect)
    err = 0.0
    for kf in ([0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.2, 0.1]):
        a = ic.sample_k(kf)
        b = ref.sample_k(kf)
        for gi in (int(x) for x in np.asarray(ad.atom_indices)):
            for gj in range(na):
                ba, bb = a.get_block(gi, gj), b.get_block(gi, gj)
                if (ba is None) != (bb is None):
                    err = 1e30; continue
                if ba is None:
                    continue
                err = max(err, float(np.max(np.abs(np.asarray(ba) - np.asarray(bb)))))
    gerr = world.allreduce(err, MPI.MAX)
    if rank == 0:
        print(f"[n={size} npool={npool}] {tag}: err={gerr:.3e} -> "
              f"{'PASS' if gerr < 1e-10 else 'FAIL'}", flush=True)
    return gerr < 1e-10

# --- Copy: L1 (world, unique) -> L3 (pool, replicated) = send-down ---
src_ad, src = make(world)
fill(src_ad, src, val)
tgt_ad, tgt = make(pool)
src.redistribute_into(tgt, RedistOp.Copy, world)
ok = check(pool, tgt_ad, tgt, val, "Copy L1->L3 (send-down)")

# --- Sum: L3 (pool, replicated partials) -> L1 (world, unique) = reduce-up ---
src2_ad, src2 = make(pool)
fill(src2_ad, src2, val)
tgt2_ad, tgt2 = make(world)
src2.redistribute_into(tgt2, RedistOp.Sum, world)
ok &= check(world, tgt2_ad, tgt2, lambda R, gi, gj: npool * val(R, gi, gj),
            "Sum L3->L1 (reduce-up)")

raise SystemExit(0 if ok else 1)

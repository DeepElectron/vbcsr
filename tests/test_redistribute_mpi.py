"""BlockSpMat.redistribute (doc/design/35) — same-comm repartition gate.

Repartition a distributed BlockSpMat from one contiguous partition (P1) to another
(P2) and back, verifying every block lands at its new owner with the right value.

NOTE VBCSR requires **contiguous, rank-ordered** partitions (owner lookup is via
cumulative block counts; a non-contiguous partition breaks the ghost-size fetch).
Target graphs must be assembled (ghost block sizes backfilled), as every real
operator graph is. Block sizes here are non-uniform to exercise the VBCSR backend
(the LCAO case); CSR (uniform 1) and BSR (uniform >1) redistribute equally well —
all three backends are validated distributed.

    OMP_NUM_THREADS=1 mpirun --bind-to none -n {1,2,3} python tests/test_redistribute_mpi.py
"""
import numpy as np
from mpi4py import MPI
from vbcsr import VBCSR
from vbcsr_core import AssemblyMode

comm = MPI.COMM_WORLD
rank, size = comm.rank, comm.size
n = 6
gsizes = [2, 3, 2, 3, 4, 2]               # non-uniform -> VBCSR backend
def neigh(i): return sorted({(i - 1) % n, i, (i + 1) % n})
gadj = [neigh(i) for i in range(n)]

def bounds(variant):
    b = [round(i * n / size) for i in range(size + 1)]
    if variant == 1:                       # a different contiguous partition
        for i in range(1, size):
            b[i] = min(b[i] + 1, n)
    return b

def owned(kind):
    b = bounds(0 if kind == "P1" else 1)
    return list(range(b[rank], b[rank + 1]))

def make(kind, fill):
    own = owned(kind)
    m = VBCSR.create_distributed(own, [gsizes[g] for g in own], [gadj[g] for g in own],
                                 dtype=np.float64, comm=comm)
    if fill:
        for gi in own:
            for gj in gadj[gi]:
                m.add_block(gi, gj, np.full((gsizes[gi], gsizes[gj]), gi * 100 + gj, np.float64),
                            AssemblyMode.INSERT)
        m.assemble()
    return m

def check(mat, kind, tag):
    err, nb = 0.0, 0
    for gi in owned(kind):
        for gj in gadj[gi]:
            b = mat.get_block(gi, gj)
            if b is None:
                err = 1e30; continue
            err = max(err, float(np.max(np.abs(np.asarray(b) - (gi * 100 + gj))))); nb += 1
    gerr = comm.allreduce(err, MPI.MAX); gnb = comm.allreduce(nb, MPI.SUM)
    if rank == 0:
        print(f"[n={size}] {tag}: blocks={gnb} err={gerr:.3e} -> {'PASS' if gerr < 1e-12 else 'FAIL'}",
              flush=True)
    return gerr < 1e-12

src = make("P1", True)
tgt_p2 = make("P2", True)   # held so the graph (with ghost sizes) survives
tgt_p1 = make("P1", True)
ok = check(src.redistribute(tgt_p2.graph), "P2", "P1->P2")
ok &= check(src.redistribute(tgt_p2.graph).redistribute(tgt_p1.graph), "P1", "P1->P2->P1 roundtrip")
raise SystemExit(0 if ok else 1)

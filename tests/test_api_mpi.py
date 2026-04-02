import _workspace_bootstrap
from _api_smoke import run_api_smoke

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from vbcsr import VBCSR, DistVector, DistGraph, DistMultiVector


def test_api_mpi():
    if MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        import vbcsr_core

        graph = vbcsr_core.DistGraph(None)
        comm = None
        rank = graph.rank
        size = graph.size

    run_api_smoke(VBCSR, DistVector, DistMultiVector, DistGraph, comm=comm, rank=rank, size=size, label="MPI")


if __name__ == "__main__":
    test_api_mpi()

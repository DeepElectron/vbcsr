from vbcsr_core import AssemblyMode
from .vector import DistVector
from .multivector import DistMultiVector
from .matrix import VBCSR

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    # Define a dummy MPI module or communicator if needed
    class DummyMPI:
        COMM_WORLD = None
    MPI = DummyMPI()

__all__ = ["VBCSR", "DistVector", "DistMultiVector", "AssemblyMode", "HAS_MPI", "MPI"]

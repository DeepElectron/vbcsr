try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    class DummyMPI:
        COMM_WORLD = None
        SUM = None
    MPI = DummyMPI()

import vbcsr_core
from vbcsr_core import AssemblyMode
from .vector import DistVector
from .multivector import DistMultiVector
from .matrix import VBCSR
from .atomic_data import AtomicData
from .image_container import ImageContainer

# If mpi4py is not present, we might still have initialized MPI in C++ (via mpirun)
# We need to ensure MPI_Finalize is called.
if not HAS_MPI:
    import atexit
    atexit.register(vbcsr_core.finalize_mpi)

__version__ = "0.1.1"

__all__ = ["VBCSR", "DistVector", "DistMultiVector", "AssemblyMode", "HAS_MPI", "MPI", 
           "AtomicData", "ImageContainer"]

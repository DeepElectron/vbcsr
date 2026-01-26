__version__ = "0.1.0"
import vbcsr_core
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
    
    # If mpi4py is not present, we might still have initialized MPI in C++ (via mpirun)
    # We need to ensure MPI_Finalize is called.
    import atexit
    atexit.register(vbcsr_core.finalize_mpi)

__all__ = ["VBCSR", "DistVector", "DistMultiVector", "AssemblyMode", "HAS_MPI", "MPI"]

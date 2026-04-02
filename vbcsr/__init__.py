import os
import sys
from pathlib import Path

try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    class DummyMPI:
        COMM_WORLD = None
        SUM = None
    MPI = DummyMPI()

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BUILD_DIR = _REPO_ROOT / "build"
_BUILD_DIR = Path(os.environ.get("VBCSR_BUILD_DIR", _DEFAULT_BUILD_DIR))
if os.environ.get("VBCSR_PREFER_BUILD", "1") != "0" and _BUILD_DIR.is_dir():
    if any(_BUILD_DIR.glob("vbcsr_core*.so")) and str(_BUILD_DIR) not in sys.path:
        sys.path.insert(0, str(_BUILD_DIR))

import vbcsr_core
from vbcsr_core import AssemblyMode, DistGraph
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

__version__ = "0.2.2"

__all__ = [
    "VBCSR",
    "DistVector",
    "DistMultiVector",
    "DistGraph",
    "AssemblyMode",
    "HAS_MPI",
    "MPI",
    "AtomicData",
    "ImageContainer",
]

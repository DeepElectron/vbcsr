import sys
from unittest.mock import patch

import _workspace_bootstrap
from _api_smoke import run_api_smoke


with patch.dict(sys.modules, {"mpi4py": None}):
    import vbcsr
    from vbcsr import VBCSR, DistVector, DistGraph, DistMultiVector, HAS_MPI


def test_api_serial():
    print(f"HAS_MPI: {HAS_MPI}")
    assert not HAS_MPI
    run_api_smoke(VBCSR, DistVector, DistMultiVector, DistGraph, comm=None, rank=0, size=1, label="Serial")


if __name__ == "__main__":
    test_api_serial()

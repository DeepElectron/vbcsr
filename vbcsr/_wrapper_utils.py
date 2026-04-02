from __future__ import annotations

from typing import Any, Optional

import numpy as np


def infer_dtype_from_core(core_obj: Any) -> np.dtype:
    if "Complex" in core_obj.__class__.__name__:
        return np.dtype(np.complex128)
    return np.dtype(np.float64)


def reduced_extent(comm: Any, local_extent: int) -> Optional[int]:
    if comm is None:
        return int(local_extent)
    if hasattr(comm, "allreduce"):
        try:
            return int(comm.allreduce(int(local_extent)))
        except Exception:
            return None
    return None


def core_buffer(core_obj: Any) -> np.ndarray:
    return np.array(core_obj, copy=False)


def duplicate_wrapper(wrapper: Any) -> Any:
    duplicate = wrapper.duplicate()
    duplicate.comm = getattr(wrapper, "comm", None)
    return duplicate

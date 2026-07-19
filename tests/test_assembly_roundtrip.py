"""Assembly / export round-trip gates (doc/row_major_migration_plan.md Phase 0).

Covers the silent-transpose scenarios the migration plan flagged as gaps:
- AssemblyMode.ADD accumulation (each block assembled as two halves),
- numpy -> add_block -> assemble -> get_block exact round-trip,
- to_scipy value correctness on every rank (local columns remapped to global),
- mult against a dense reference (exercises ghost exchange at np > 1).

The structure is deliberately asymmetric and the blocks non-square with
non-symmetric values: a silently transposed block or swapped dimension cannot
cancel out. Runs serially and under mpirun (no mpi4py required).
"""

import _workspace_bootstrap  # noqa: F401
import unittest

import numpy as np
import scipy.sparse as sp

import vbcsr
from vbcsr_core import AssemblyMode

N_BLOCKS = 8
BLOCK_SIZES = [1, 3, 2, 4, 1, 2, 3, 2]


def adjacency_row(gi: int) -> list:
    # Ring neighbours plus an asymmetric skip link (i -> i+3 only).
    cols = {(gi - 1) % N_BLOCKS, gi, (gi + 1) % N_BLOCKS, (gi + 3) % N_BLOCKS}
    return sorted(cols)


def block_value(gi: int, gj: int, dtype) -> np.ndarray:
    r, c = BLOCK_SIZES[gi], BLOCK_SIZES[gj]
    base = np.arange(r * c, dtype=np.float64).reshape(r, c)
    data = base * 0.01 + gi + 0.1 * gj + 0.5
    if np.dtype(dtype) == np.dtype(np.complex128):
        data = data + 1j * (base * 0.02 - gj + 0.25)
    return np.ascontiguousarray(data, dtype=dtype)


def scalar_offsets() -> np.ndarray:
    offsets = np.zeros(N_BLOCKS + 1, dtype=np.int64)
    np.cumsum(BLOCK_SIZES, out=offsets[1:])
    return offsets


def reference_dense(dtype) -> np.ndarray:
    offsets = scalar_offsets()
    dense = np.zeros((offsets[-1], offsets[-1]), dtype=dtype)
    for gi in range(N_BLOCKS):
        for gj in adjacency_row(gi):
            dense[offsets[gi]:offsets[gi + 1], offsets[gj]:offsets[gj + 1]] = block_value(gi, gj, dtype)
    return dense


def partition_range(n_items: int, size: int, rank: int) -> range:
    base, rem = divmod(n_items, size)
    start = rank * base + min(rank, rem)
    return range(start, start + base + (1 if rank < rem else 0))


class TestAssemblyRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from mpi4py import MPI
            cls.rank = MPI.COMM_WORLD.Get_rank()
            cls.size = MPI.COMM_WORLD.Get_size()
        except ImportError:
            import vbcsr_core
            probe = vbcsr_core.DistGraph(None)
            cls.rank = int(probe.rank)
            cls.size = int(probe.size)

    def build(self, dtype):
        owned = list(partition_range(N_BLOCKS, self.size, self.rank))
        matrix = vbcsr.VBCSR.create_distributed(
            owned_indices=owned,
            block_sizes=[BLOCK_SIZES[g] for g in owned],
            adjacency=[adjacency_row(g) for g in owned],
            dtype=dtype,
        )
        # ADD-mode gate: each block is assembled as two halves that must sum.
        for gi in owned:
            for gj in adjacency_row(gi):
                full = block_value(gi, gj, dtype)
                matrix.add_block(gi, gj, np.ascontiguousarray(0.25 * full), AssemblyMode.ADD)
                matrix.add_block(gi, gj, np.ascontiguousarray(0.75 * full), AssemblyMode.ADD)
        matrix.assemble()
        return matrix, owned

    def check_dtype(self, dtype):
        matrix, owned = self.build(dtype)
        offsets = scalar_offsets()
        reference = reference_dense(dtype)

        # 1. get_block returns exactly what was assembled (ADD halves summed).
        for gi in owned:
            for gj in adjacency_row(gi):
                got = matrix.get_block(gi, gj)
                self.assertIsNotNone(got, f"missing block ({gi},{gj})")
                np.testing.assert_allclose(
                    got, block_value(gi, gj, dtype), rtol=0, atol=1e-14,
                    err_msg=f"block ({gi},{gj}) mismatch after ADD assembly")

        # 2. to_scipy local values match the reference rows exactly once local
        #    columns are remapped to global scalar columns.
        local = matrix.to_scipy(format="csr").tocsr()
        graph = matrix.graph
        local_blocks = list(graph.owned_global_indices) + list(graph.ghost_global_indices)
        local_block_sizes = list(graph.block_sizes)
        self.assertEqual(len(local_block_sizes), len(local_blocks))
        local_col_offsets = np.zeros(len(local_blocks) + 1, dtype=np.int64)
        np.cumsum(local_block_sizes, out=local_col_offsets[1:])
        col_map = np.zeros(local_col_offsets[-1], dtype=np.int64)
        for j, g in enumerate(local_blocks):
            self.assertEqual(local_block_sizes[j], BLOCK_SIZES[g])
            span = local_col_offsets[j + 1] - local_col_offsets[j]
            col_map[local_col_offsets[j]:local_col_offsets[j + 1]] = offsets[g] + np.arange(span)

        row_start = offsets[owned[0]] if owned else 0
        dense_local = np.zeros((local.shape[0], offsets[-1]), dtype=dtype)
        arr = local.toarray()
        dense_local[:, col_map[:arr.shape[1]]] = arr
        np.testing.assert_allclose(
            dense_local, reference[row_start:row_start + local.shape[0], :],
            rtol=0, atol=1e-14, err_msg="to_scipy local values mismatch reference")

        # 3. mult against dense reference (ghost exchange at np > 1).
        rng = np.random.default_rng(7)
        x_global = rng.standard_normal(offsets[-1])
        if np.dtype(dtype) == np.dtype(np.complex128):
            x_global = x_global + 1j * rng.standard_normal(offsets[-1])
        x = matrix.create_vector()
        y = matrix.create_vector()
        x.from_numpy(np.ascontiguousarray(
            x_global[row_start:row_start + local.shape[0]], dtype=dtype))
        matrix.mult(x, y)
        np.testing.assert_allclose(
            y.to_numpy(), (reference @ x_global)[row_start:row_start + local.shape[0]],
            rtol=1e-12, atol=1e-12, err_msg="mult mismatch vs dense reference")

    def test_real(self):
        self.check_dtype(np.float64)

    def test_complex(self):
        self.check_dtype(np.complex128)


if __name__ == "__main__":
    unittest.main()

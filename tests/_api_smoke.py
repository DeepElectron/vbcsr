from __future__ import annotations

import numpy as np


def run_api_smoke(VBCSR, DistVector, DistMultiVector, DistGraph, comm, rank: int, size: int, label: str) -> None:
    graph = DistGraph(comm)
    if size == 1:
        graph.construct_serial(2, [2, 3], [[0, 1], [1]])
        assert graph.owned_global_indices == [0, 1]
        assert graph.ghost_global_indices == []
        assert graph.block_sizes == [2, 3]
        assert graph.owned_scalar_rows == 5
        assert graph.local_scalar_cols == 5
        assert graph.global_scalar_rows == 5
        assert graph.get_local_index(1) == 1
    else:
        owned = [rank]
        graph.construct_distributed(owned, [2], [[rank, (rank + 1) % size]])
        assert graph.rank == rank
        assert graph.size == size
        assert graph.owned_global_indices == owned
        assert graph.owned_scalar_rows == 2
        assert graph.global_scalar_rows == 2 * size
        assert graph.get_local_index(rank) == 0

    if rank == 0:
        print("DistGraph assertions passed")

    n_blocks = 4
    block_size = 2
    global_blocks = n_blocks * size

    mat = VBCSR.create_random(global_blocks, block_size, block_size, density=0.1, seed=42, comm=comm)

    if rank == 0:
        print(f"Matrix created: {mat}")

    assert mat.ndim == 2
    assert mat.shape == (global_blocks * block_size, global_blocks * block_size)
    assert mat.nnz >= 0
    assert mat.matrix_kind == "bsr"
    assert len(mat) == mat.shape[0]

    if rank == 0:
        print("Matrix assertions passed")

    mat_T = mat.T
    assert mat_T.shape == (mat.shape[1], mat.shape[0])

    if rank == 0:
        print("Transpose passed")

    mat_conj = mat.conj()
    mat_conjugate = mat.conjugate()
    assert mat_conj.shape == mat.shape
    assert mat_conjugate.shape == mat.shape
    assert mat_conj.dtype == mat_conjugate.dtype

    if rank == 0:
        print("Conj passed")

    mat_copy = mat.copy()
    assert mat_copy.shape == mat.shape

    if rank == 0:
        print("Copy passed")

    mat_T_inplace = mat.copy()
    mat_T_inplace.transpose_()
    assert mat_T_inplace.shape == (mat.shape[1], mat.shape[0])

    if rank == 0:
        print("In-place Transpose passed")

    mat_conj_inplace = mat.copy()
    mat_conj_inplace.conj_()
    assert mat_conj_inplace.shape == mat.shape

    if rank == 0:
        print("In-place Conj passed")

    mat_neg = -mat
    assert mat_neg.shape == mat.shape

    mat_sub = mat - mat
    assert mat_sub.shape == mat.shape

    mat_real = mat.real
    assert mat_real.shape == mat.shape
    assert mat_real.dtype == np.dtype(np.float64)

    mat_imag = mat.imag
    assert mat_imag.shape == mat.shape
    assert mat_imag.dtype == np.dtype(np.float64)

    if rank == 0:
        print("Numerical Ops passed")

    vec = mat.create_vector()
    vec.set_constant(1.0)

    if rank == 0:
        print(f"Vector created: {vec}")

    assert vec.ndim == 1
    assert vec.shape == (mat.shape[1],)
    assert vec.size == mat.shape[1]
    assert len(vec) == vec.size
    assert vec.T is vec

    vec_copy = vec.copy()
    assert vec_copy.shape == vec.shape

    res = mat.dot(vec)
    assert isinstance(res, DistVector)
    assert res.shape == (mat.shape[0],)

    res2 = mat @ vec
    assert isinstance(res2, DistVector)

    k = 3
    mv = mat.create_multivector(k)
    mv.set_constant(1.0)

    if rank == 0:
        print(f"MultiVector created: {mv}")

    assert mv.ndim == 2
    assert mv.shape == (mat.shape[1], k)
    assert mv.size == mat.shape[1] * k
    assert len(mv) == mat.shape[1]

    mv_copy = mv.copy()
    assert mv_copy.shape == mv.shape

    res_mv = mat @ mv
    assert isinstance(res_mv, DistMultiVector)
    assert res_mv.shape == (mat.shape[0], k)

    if rank == 0:
        print(f"API Compliance Test Passed ({label})!")

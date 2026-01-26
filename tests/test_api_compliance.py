import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from vbcsr import VBCSR, DistVector, DistMultiVector, HAS_MPI, MPI

def test_api_compliance():
    if HAS_MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1
    
    # Create matrix
    n_blocks = 4
    block_size = 2
    global_blocks = n_blocks * size
    
    # Use create_random for simplicity
    mat = VBCSR.create_random(comm, global_blocks, block_size, block_size, density=0.1, seed=42)
    
    if rank == 0:
        print(f"Matrix created: {mat}")
    
    # Check Matrix API
    assert mat.ndim == 2
    assert mat.shape == (global_blocks * block_size, global_blocks * block_size)
    assert mat.nnz >= 0
    assert len(mat) == mat.shape[0]
    
    if rank == 0:
        print("Matrix assertions passed")

    # Transpose
    mat_T = mat.T
    assert mat_T.shape == (mat.shape[1], mat.shape[0])
    
    if rank == 0:
        print("Transpose passed")

    # Conj
    mat_conj = mat.conj()
    assert mat_conj.shape == mat.shape
    
    if rank == 0:
        print("Conj passed")

    # Copy
    mat_copy = mat.copy()
    assert mat_copy.shape == mat.shape
    
    if rank == 0:
        print("Copy passed")

    # In-place Transpose
    mat_T_inplace = mat.copy()
    mat_T_inplace.transpose_()
    assert mat_T_inplace.shape == (mat.shape[1], mat.shape[0])
    
    if rank == 0:
        print("In-place Transpose passed")

    # In-place Conj
    mat_conj_inplace = mat.copy()
    mat_conj_inplace.conj_()
    assert mat_conj_inplace.shape == mat.shape
    
    if rank == 0:
        print("In-place Conj passed")

    # Numerical Ops
    mat_neg = -mat
    assert mat_neg.shape == mat.shape
    
    mat_sub = mat - mat
    assert mat_sub.shape == mat.shape
    
    mat_real = mat.real
    assert mat_real.shape == mat.shape
    assert mat_real.dtype == np.float64
    
    mat_imag = mat.imag
    assert mat_imag.shape == mat.shape
    assert mat_imag.dtype == np.float64
    
    if rank == 0:
        print("Numerical Ops passed")

    # Vector API
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
    
    # Operations
    # dot
    res = mat.dot(vec)
    assert isinstance(res, DistVector)
    assert res.shape == (mat.shape[0],)
    
    # @ operator
    res2 = mat @ vec
    assert isinstance(res2, DistVector)
    
    # MultiVector API
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
    
    # Operations
    res_mv = mat @ mv
    assert isinstance(res_mv, DistMultiVector)
    assert res_mv.shape == (mat.shape[0], k)
    
    if rank == 0:
        print("API Compliance Test Passed!")

if __name__ == "__main__":
    test_api_compliance()

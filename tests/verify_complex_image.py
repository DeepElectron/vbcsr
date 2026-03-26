import numpy as np
from vbcsr.atomic_data import AtomicData
from vbcsr.image_container import ImageContainer
import os

def test_complex_image():
    print("Starting verification of Complex ImageContainer...")
    # Create simple AtomicData
    pos = np.array([[0.0, 0.0, 0.0]])
    cell = np.eye(3)
    pbc = [True, True, True]
    # small cutoff
    ad = AtomicData.from_points(pos, np.array([1]), cell, pbc, 2.0, type_norb=np.array([1]))
    
    # Create Complex ImageContainer
    print("Creating ImageContainer(dtype=complex128)...")
    ic = ImageContainer(ad, dtype=np.complex128)
    assert ic.dtype == np.complex128, f"Expected complex128, got {ic.dtype}"
    
    # Add block
    # Local R=0 block
    print("Adding complex block...")
    data = np.array([[1.0 + 2.0j]])
    ic.add_block(0, 0, data, R=[0,0,0])
    
    # Sample k
    print("Sampling k=[0,0,0]...")
    k = [0.0, 0.0, 0.0]
    res = ic.sample_k(k)
    
    # Verify result
    print("Verifying result...")
    assert res.dtype == np.complex128, f"Result dtype mismatch: {res.dtype}"
    val = res.to_dense()
    print("Result dense matrix:\n", val)
    
    expected = np.array([[1.0 + 2.0j]])
    assert np.allclose(val, expected), f"Expected {expected}, got {val}"
    
    print("Complex ImageContainer verified successfully!")

if __name__ == "__main__":
    test_complex_image()

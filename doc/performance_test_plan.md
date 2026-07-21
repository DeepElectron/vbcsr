# First state
Strict demand oriented
We need CSR and BSR to work well in the moire delocalization calculation, as well as the benchmark paper. So CSR and BSR is the first data target to work with.

Benchmark operations (ordered by priority): SPMV, adjoint SPMV, SPMM; both serial and distributed version.
Then, benchmark B-SPMV, adjoint B-SPMV.
Later AXPBY.

We benchmark serical case with MKL, and then benchmark the distributed case on ourselve for scalability test.

# Second State
VBCSR case.
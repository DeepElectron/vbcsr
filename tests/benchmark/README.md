# VBCSR Publication Benchmarks

`run_benchmark.py` is the data-generation script for the benchmark section of
`doc/main.tex`. It uses an atom-like periodic geometric finite-cutoff graph
with exponentially decaying off-diagonal block magnitudes, and runs the CSR,
BSR, and VBCSR domains for SpMV, SpMM, and SpGEMM.

The JSON output records the full reproducibility state: git revision, CMake
cache entries, Python packages, NumPy BLAS configuration, FlexiBLAS state,
loaded modules, Slurm allocation, CPU model, MPI library, graph parameters,
matrix storage, communication volume, assembly time, atomistic conversion time,
validation error, value-decay parameters, and SpGEMM threshold/fill statistics.
The CSV output contains one plotting row per benchmark case.

## Alliance/Compute Canada Environment

Create the Python virtual environment with the same module stack that will be
used by the jobs:

```bash
module load StdEnv/2023 python scipy-stack mpi4py flexiblas
virtualenv --no-download $PWD/.venv-vbcsr
source $PWD/.venv-vbcsr/bin/activate
pip install --no-index --upgrade pip
pip install --no-index mpi4py sparse-dot-mkl
```

Build VBCSR with the same compiler, MPI, and BLAS/LAPACK modules. The Slurm
scripts use `VBCSR_BUILD_DIR=build` by default and add that directory to
`PYTHONPATH` at runtime.

FlexiBLAS backend switching is controlled through `FLEXIBLAS_BACKEND`. Use the
backend name printed by `flexiblas list`; the scripts export it as `FLEXIBLAS`
before Python starts.

## Kernel Efficiency

Run on one rank. The production default is `4096` graph blocks, which is a
`16 x 16 x 16` periodic lattice before jitter.

```bash
sbatch \
  --export=ALL,REPO_ROOT=$PWD,PYTHON_VENV=$PWD/.venv-vbcsr,VBCSR_BUILD_DIR=build,BLOCKS=4096,TARGET_DEGREE=12,RHS=16,REPEATS=7,MAGNITUDE_DECAY_LENGTH=0.5,SPGEMM_THRESHOLDS=0.0 \
  tests/benchmark/slurm_efficiency.sbatch
```

For the SpGEMM threshold/fill/error tradeoff rows, submit an additional
efficiency job with a threshold sweep. Nonzero thresholds are approximate
SpGEMM measurements: the script records the relative error against the exact
SciPy product and does not treat that expected approximation error as a failed
correctness check.

```bash
sbatch \
  --export=ALL,REPO_ROOT=$PWD,PYTHON_VENV=$PWD/.venv-vbcsr,VBCSR_BUILD_DIR=build,BLOCKS=4096,TARGET_DEGREE=12,RHS=16,REPEATS=7,MAGNITUDE_DECAY_LENGTH=0.5,SPGEMM_THRESHOLDS='0.0 1e-6 1e-4 1e-2 1e-1' \
  tests/benchmark/slurm_efficiency.sbatch
```

The efficiency job requires `sparse_dot_mkl` because the paper compares VBCSR
against SciPy CSR and an MKL sparse baseline.

## Strong Scaling

Use the same global `BLOCKS` for every rank count. The production default is
`32768` graph blocks, a `32 x 32 x 32` periodic lattice.

```bash
TASKS_PER_NODE=32
for np in 1 2 4 8 16 32 64; do
  nodes=$(( (np + TASKS_PER_NODE - 1) / TASKS_PER_NODE ))
  sbatch --nodes=${nodes} --ntasks=${np} --ntasks-per-node=${TASKS_PER_NODE} \
    --export=ALL,REPO_ROOT=$PWD,PYTHON_VENV=$PWD/.venv-vbcsr,VBCSR_BUILD_DIR=build,SUITE=distributed-strong,BLOCKS=32768,TARGET_DEGREE=12,RHS=16,REPEATS=7,MAGNITUDE_DECAY_LENGTH=0.5,LABEL=paper_strong_np${np} \
    tests/benchmark/slurm_distributed.sbatch
done
```

Distributed paper data requires `mpi4py`; the script aborts without it because
rank-level timing and metadata must be reduced correctly. Set
`SRUN_MPI_TYPE=pmi2` or another site-specific Slurm MPI type if PMIx is not the
launcher interface on the target cluster.

## Weak Scaling

Use the same `WEAK_BLOCKS_PER_RANK` for every rank count. The production default
is `4096` graph blocks per rank.

```bash
TASKS_PER_NODE=32
for np in 1 2 4 8 16 32 64; do
  nodes=$(( (np + TASKS_PER_NODE - 1) / TASKS_PER_NODE ))
  sbatch --nodes=${nodes} --ntasks=${np} --ntasks-per-node=${TASKS_PER_NODE} \
    --export=ALL,REPO_ROOT=$PWD,PYTHON_VENV=$PWD/.venv-vbcsr,VBCSR_BUILD_DIR=build,SUITE=distributed-weak,WEAK_BLOCKS_PER_RANK=4096,TARGET_DEGREE=12,RHS=16,REPEATS=7,MAGNITUDE_DECAY_LENGTH=0.5,LABEL=paper_weak_np${np} \
    tests/benchmark/slurm_distributed.sbatch
done
```

The strong and weak runs should include at least one multi-node allocation for
the publication scaling figure.

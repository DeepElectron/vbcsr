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

Threading policy for the publication jobs: `OMP_NUM_THREADS` controls VBCSR's
outer OpenMP parallelism. BLAS runtime thread pools used under VBCSR are pinned
to one thread by default through `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `BLIS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, and
`VECLIB_MAXIMUM_THREADS=1`, avoiding nested BLAS threading inside VBCSR apply
kernels. `OMP_DYNAMIC=FALSE` and `MKL_DYNAMIC=FALSE` are set in the Slurm
wrappers.

The sparse-dot-mkl reference is a top-level MKL sparse baseline, so it has its
own explicit control: `SPARSE_DOT_MKL_NUM_THREADS`. For the one-rank efficiency
job the Slurm default is `SPARSE_DOT_MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}`,
matching the VBCSR CPU budget. The Python driver applies this with
`sparse_dot_mkl.mkl_set_num_threads(...)`, records the reported MKL thread
state for every MKL baseline row, and restores `MKL_NUM_THREADS` afterward so
the next VBCSR case is not contaminated by the reference baseline.

## Kernel Efficiency

Run on one rank. The production default is `4096` graph blocks, which is a
`16 x 16 x 16` periodic lattice before jitter.

```bash
sbatch --cpus-per-task=32 \
  --export=ALL,REPO_ROOT=$PWD,PYTHON_VENV=$PWD/.venv-vbcsr,VBCSR_BUILD_DIR=build,BLOCKS=4096,TARGET_DEGREE=12,RHS=16,REPEATS=7,MAGNITUDE_DECAY_LENGTH=0.5,SPGEMM_THRESHOLDS=0.0 \
  tests/benchmark/slurm_efficiency.sbatch
```

For the SpGEMM threshold/fill/error tradeoff rows, submit an additional
efficiency job with a threshold sweep. Nonzero thresholds are approximate
SpGEMM measurements: the script records the relative error against the exact
SciPy product and does not treat that expected approximation error as a failed
correctness check.

```bash
sbatch --cpus-per-task=32 \
  --export=ALL,REPO_ROOT=$PWD,PYTHON_VENV=$PWD/.venv-vbcsr,VBCSR_BUILD_DIR=build,BLOCKS=4096,TARGET_DEGREE=12,RHS=16,REPEATS=7,MAGNITUDE_DECAY_LENGTH=0.5,SPGEMM_THRESHOLDS='0.0 1e-6 1e-4 1e-2 1e-1' \
  tests/benchmark/slurm_efficiency.sbatch
```

The efficiency job requires `sparse_dot_mkl` because the paper compares VBCSR
against SciPy CSR and an MKL sparse baseline.

For exact SpGEMM, the candidate block-product count scales approximately as
`blocks * target_degree^2`. A run with `BLOCKS=4096` and `TARGET_DEGREE=100`
therefore has about 41 million candidate block products per exact SpGEMM call,
before repeats and reference implementations are counted. Use that setting only
for a deliberately high-connectivity stress case.

Create the paper figure from the newest efficiency CSV:

```bash
python tests/benchmark/plot_efficiency.py \
  --output-prefix tests/benchmark/results/kernel_efficiency
```

The script writes both PDF and PNG, with SciPy, sparse-dot-mkl, and VBCSR shown
as separate bars. The x-axis labels identify the tested block-size model. The
PDF is the preferred format for the manuscript because it preserves vector text
and bars.

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

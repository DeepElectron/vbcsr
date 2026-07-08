# VBCSR Publication Benchmarks

`run_benchmark.py` is the data-generation script for the benchmark section of
`doc/main.tex`. It only uses an atom-like geometric finite-cutoff graph and
always runs the full CSR/BSR/VBCSR x SpMV/SpMM/SpGEMM benchmark matrix.

## Kernel Efficiency

Run on one rank. For local or interactive jobs:

```bash
VBCSR_BUILD_DIR=build_dbg conda run -n vbcsr \
  python tests/benchmark/run_benchmark.py \
  --suite efficiency \
  --blocks 2048 \
  --target-degree 12 \
  --rhs 16 \
  --repeats 7 \
  --min-seconds 1.0 \
  --min-iterations 5 \
  --require-mkl \
  --output-dir tests/benchmark/results \
  --label paper_efficiency
```

This records VBCSR, SciPy CSR, and MKL sparse timings when MKL is available.
Use `--require-mkl` for final paper data so missing MKL fails loudly.

For Slurm:

```bash
sbatch \
  --export=ALL,REPO_ROOT=$PWD,CONDA_ENV=vbcsr,VBCSR_BUILD_DIR=build,BLOCKS=2048,TARGET_DEGREE=12,RHS=16,REPEATS=7 \
  tests/benchmark/slurm_efficiency.sbatch
```

## Strong Scaling

Use the same global `--blocks` for every rank count. On Slurm, submit one job
per rank count so `SLURM_NTASKS` is the MPI size:

```bash
for np in 1 2 4 8; do
  sbatch -n ${np} \
    --export=ALL,REPO_ROOT=$PWD,CONDA_ENV=vbcsr,VBCSR_BUILD_DIR=build,SUITE=distributed-strong,BLOCKS=8192,TARGET_DEGREE=12,RHS=16,REPEATS=7,LABEL=paper_strong_np${np} \
    tests/benchmark/slurm_distributed.sbatch
done
```

Distributed paper data requires `mpi4py`; the script aborts without it because
rank-level metadata cannot otherwise be reduced correctly. Set
`SRUN_MPI_TYPE=pmi2` or another site-specific value in `--export` if your
cluster does not use Slurm PMIx.

## Weak Scaling

Use the same `--weak-blocks-per-rank` for every rank count:

```bash
for np in 1 2 4 8; do
  sbatch -n ${np} \
    --export=ALL,REPO_ROOT=$PWD,CONDA_ENV=vbcsr,VBCSR_BUILD_DIR=build,SUITE=distributed-weak,WEAK_BLOCKS_PER_RANK=2048,TARGET_DEGREE=12,RHS=16,REPEATS=7,LABEL=paper_weak_np${np} \
    tests/benchmark/slurm_distributed.sbatch
done
```

Each run writes a JSON file with full reproducibility metadata and a CSV table
with one row per domain/operation case.

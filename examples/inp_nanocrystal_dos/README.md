# InP nanocrystal DOS with VBCSR and KPM

This Python example constructs the finite zinc-blende InP nanocrystals of
Sapra, Viswanatha, and Sarma, *J. Phys. D* **36**, 1595 (2003), assembles the
paper's `In(sp3)-P(sp3d5)` tight-binding Hamiltonian directly into VBCSR, and
computes the density of states with an MPI-parallel kernel polynomial method.
The parameter source is [arXiv:cond-mat/0308038](https://arxiv.org/abs/cond-mat/0308038).

The default is the largest system explicitly studied in the paper: 22
coordination shells around a central In atom. It contains 5,083 In and 4,444 P
atoms, has a 77.1 Angstrom effective diameter, and gives exactly 60,328 core
orbitals. Surface passivation adds 2,556 one-orbital pseudo-H atoms.

## Files

- `structure.py` generates exact alternating zinc-blende coordination shells.
- `parameters.py` records the basis, onsite energies, and hopping table in eV.
- `slater_koster.py` rotates sigma/pi/delta channels into real-harmonic blocks.
- `hamiltonian.py` builds the distributed neighbour graph and uses batched
  `ImageContainer.add_blocks` assembly. Rank-local edges are classified by
  vectorized NumPy bond codes, so there is no Python loop over atoms or edges.
- `kpm.py` implements the RSATB-style Hermitian two-moments-per-matvec DOS
  recurrence and Jackson reconstruction.
- `run.py` is the MPI command-line driver.
- `cluster/` provides a restart-safe Alliance/Compute Canada Slurm campaign,
  including the paper, 100k, and million-core-atom DOS runs and an optional
  million-atom strong-scaling sweep.

## Running

From the VBCSR source root, a small validation run is:

```bash
mpirun -np 2 python -m examples.inp_nanocrystal_dos.run \
  --shells 4 --moments 256 --random-vectors 8 --batch-size 4 \
  --output-dir inp_dos_small
```

The paper-scale default is:

```bash
mpirun -np 8 python -m examples.inp_nanocrystal_dos.run \
  --moments 1024 --random-vectors 16 --batch-size 8 \
  --output-dir inp_dos_shell22
```

The output directory contains `dos.dat`, raw `moments.npz`, and a complete
`metadata.json` parameter/provenance record. `dos.dat` includes both the
single-spin DOS produced by the spinless Hamiltonian and the factor-of-two
spin-degenerate DOS. Its energy points are an ascending Gauss-Chebyshev grid;
the last column is the integration weight, so summing
`DOS_single_spin * weight` returns the single-spin state count.

An exact small-system size-series validation complements the stochastic DOS
run and checks that the confinement gap decreases with diameter:

```bash
python -m examples.inp_nanocrystal_dos.validate --shells 2 4 6 8
```

It reports HOMO/LUMO H weights and compares the exciton-corrected confinement
shift with the paper's printed size-fit. Each result is referenced to its own
bulk gap: the rounded Table-I parameters give 1.58903 eV, whereas the paper's
experimental reference is 1.4 eV. The comparison remains diagnostic rather
than exact because the finite-dot values are only plotted and the In-H table is
under-specified.

For visual or VASP-side inspection, export centered Cartesian POSCAR files with
10 Angstrom vacuum to every cell face:

```bash
python -m examples.inp_nanocrystal_dos.export_vasp \
  --shells 2 4 6 8 22 --output-dir inp_vasp_structures
```

Each structure is written as `shell_NN/POSCAR`. The exporter reads every file
back and checks species counts, vacuum clearance, and nearest In-P, In-In, P-P,
and In-H distances; the results are recorded in `manifest.json`.

## Million-atom extensions

Size selection is analytic and does not require coordinate allocation:

```bash
python -m examples.inp_nanocrystal_dos.run --list-sizes
python -m examples.inp_nanocrystal_dos.run --target-core-atoms 3000000 --dry-run
```

The standard targets are:

| requested core atoms | shells | actual In+P atoms | atoms with H | orbitals with H |
|---:|---:|---:|---:|---:|
| paper system | 22 | 9,527 | 12,083 | 62,884 |
| 100,000 | 50 | 107,401 | 120,205 | 702,908 |
| 1,000,000 | 106 | 1,006,789 | 1,063,609 | 6,565,436 |
| 3,000,000 | 154 | 3,073,533 | 3,193,041 | 20,022,780 |

An actual large run uses the identical physical construction:

```bash
mpirun -np 256 python -m examples.inp_nanocrystal_dos.run \
  --target-core-atoms 1000000 --moments 2048 \
  --random-vectors 16 --batch-size 8 --output-dir inp_dos_1m
```

The million-atom cases require an HPC allocation sized for the VBCSR matrix,
graph/ghost metadata, and three complex KPM multivectors. Use `--dry-run` for
planning, and do not add `--write-xyz` at those sizes unless the large text
output is genuinely needed.

For a production Alliance campaign, including submission previews, immutable
result publication, checksums, Slurm accounting collection, and recommended
calibration order, see [`cluster/README.md`](cluster/README.md).

## Model scope and passivation caveat

The implemented bulk terms are nearest-neighbour In-P and next-nearest-neighbour
In-In/P-P hoppings, with the orbital order written into `metadata.json`. Table I
declares one H `s` orbital but lists three In-H numbers under `ss`, `sp`, and
`sd`, even though the surface In basis has no `d`. The table therefore does not
uniquely define the rectangular In-H block. This implementation reads the first
two entries as host `s`/`p` to H `s`, using `ss=-2.944` eV and
`p(In)-s(H)=2.76` eV, and preserves the unused `-1.36` eV source value and this
caveat in `metadata.json`. The nominal 1.80 Angstrom In-H distance only places H
and builds connectivity; the paper's tabulated hoppings are not distance
dependent.

## Scaling behavior

No Numba, Cython, or compiled example-specific extension is required. VBCSR
distributes the graph and matrix by owned atom rows, builds neighbour halos on
each rank, and performs KPM sparse multivector products and ghost exchange in
native MPI/OpenMP kernels. The Python model operates only on each rank's local
edge arrays. It classifies them in bounded chunks with NumPy, reduces the zinc-
blende geometry to a small species/direction block codebook, and evaluates each
distinct SK block once.

As a validation on four MPI ranks with two OpenMP threads per rank, vectorizing
the example reduced shell-22 Hamiltonian assembly from 0.547 s to 0.079 s. The
120,205-atom shell-50 case assembled 1,685,308 directed hopping blocks in 0.728
s; its 702,908-state DOS integrated to exactly 702,908 using the output
quadrature weights. These timings are machine-specific, but the approximately
linear change in assembly time with local edge count is the intended scaling
property. `metadata.json` records maximum-rank timings and distributed counts
for every run so larger allocations can be evaluated in the same way.

Coordinate enumeration is still a one-time rank-0 Python operation before
`AtomicData.from_points` scatters the arrays. It is O(N), preallocated, and took
0.75 s for shell 50 in the validation above; unlike SK assembly or KPM, it is
not repeated with the number of moments.

# Hg-DNA transport

This code loads the 7-base-pair Hg-DNA Hamiltonian, checks that its orbitals
match the atoms and partitions in the PDB file, and calculates coherent or
DOS-weighted transmission one energy point at a time.

The model has 5,083 orbitals split across 15 partitions. Partition 1 is the
left contact and partition 7 is the right contact. Both contact strengths are
1 eV. The remaining 13 partitions receive decoherence probes during a
decoherent calculation.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

`data/xx7tg6.mat` is about 191 MB and is not stored in Git. Place it in the
`data` folder before running the model. It must contain a MATLAB variable named
`xx7tg6` with shape `5083 x 5083`. The Hamiltonian and energy inputs are in eV.

## Check the model

Run the model inspection before calculating transport:

```bash
python3 inspect_model.py \
  --pdb data/xx7tg6.pdb \
  --hamiltonian data/xx7tg6.mat \
  --variable xx7tg6
```

This checks the atom order, partition order, orbital ranges, matrix dimensions,
finite values, and Hermitian condition. It writes the partition table to
`outputs/model_inspection/partition_table.csv`.

## Coherent transport

The confirmed energy grid is -10 to 0 eV in steps of 0.01 eV:

```bash
python3 run_transport.py \
  --pdb data/xx7tg6.pdb \
  --hamiltonian data/xx7tg6.mat \
  --variable xx7tg6 \
  --mode coherent \
  --left-partition 1 \
  --right-partition 7 \
  --gamma-left 1 \
  --gamma-right 1 \
  --energy-min -10 \
  --energy-max 0 \
  --energy-step 0.01 \
  --output-dir outputs/real_coherent
```

The grid contains 1,001 points. A dense Green-function calculation takes
several seconds per point, so benchmark a short range before starting the full
sweep. This command calculates only index 500, which is -5 eV:

```bash
python3 run_transport.py \
  --pdb data/xx7tg6.pdb \
  --hamiltonian data/xx7tg6.mat \
  --variable xx7tg6 \
  --mode coherent \
  --start-index 500 \
  --stop-index 501 \
  --save-history-index 500 \
  --output-dir outputs/coherent_benchmark
```

`--start-index` is included and `--stop-index` is excluded. These options can
split a long calculation into smaller ranges.

## DOS-weighted transport

Decoherent mode also requires `D0`, `alpha`, and a maximum iteration count.
They do not have defaults. The values below are an example run configuration:

```bash
python3 run_transport.py \
  --pdb data/xx7tg6.pdb \
  --hamiltonian data/xx7tg6.mat \
  --variable xx7tg6 \
  --mode decoherent \
  --left-partition 1 \
  --right-partition 7 \
  --gamma-left 1 \
  --gamma-right 1 \
  --energy-min -10 \
  --energy-max 0 \
  --energy-step 0.01 \
  --d0 0.1 \
  --alpha 0.3 \
  --max-iterations 500 \
  --start-index 500 \
  --stop-index 501 \
  --save-history-index 500 \
  --output-dir outputs/real_decoherent
```

`D0` is in eV squared. `alpha` is the unitless linear-mixing value. The solver
stops only when both the DOS and effective-transmission changes are below
0.1 percent. It stops with an error if the maximum iteration count is reached.

## Saved results

Each completed energy point is written immediately. Running the same command
with the same output directory skips completed points. A run with different
physical or solver settings must use a different output directory.

- `run_config.json` contains the settings used for the run.
- `checkpoints/` contains one JSON file per completed energy point.
- `transport.csv` contains DOS, direct transmission, effective transmission,
  convergence changes, and iteration counts.
- `partition_results.csv` contains the 15 partition LDOS, self-energy, and
  probe-broadening values at every completed energy.
- `history/` contains iteration history for indices passed with
  `--save-history-index`.
- `failures/` records a point that reached the iteration limit without
  converging.

## Tests

Run the normal test suite with:

```bash
python3 -m pytest
```

The normal suite checks the equations and workflow with small matrices. It
loads the real data for structural checks but does not invert the 5,083 by
5,083 Hamiltonian.

Run the real single-energy benchmark separately with:

```bash
RUN_REAL_BENCHMARK=1 python3 -m pytest tests/test_real_model_benchmark.py -s
```

The earlier 20-orbital GGGGG calculation is kept in
[`reference/ggggg`](reference/ggggg/README.md).

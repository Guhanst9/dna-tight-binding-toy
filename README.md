# Hg-DNA transport

This repo now uses the 7-base-pair Hg-DNA Hamiltonian in `data/`. The first
step is checking that the PDB atoms line up with all 5,083 rows and columns of
the Hamiltonian.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

The Hamiltonian is too large for GitHub, so `xx7tg6.mat` has to be placed in
the `data` folder before running anything.

## Check the model

```bash
python3 inspect_model.py \
  --pdb data/xx7tg6.pdb \
  --hamiltonian data/xx7tg6.mat \
  --variable xx7tg6
```

This checks the atom order, partitions, orbital ranges, matrix size, and
Hermitian condition. The partition table is saved to
`outputs/model_inspection/partition_table.csv`.

The real-system transport calculation has not been added yet. The contact
partitions and run values still need to be chosen first.

## Tests

```bash
python3 -m pytest
```

The older 20-orbital GGGGG model is kept in
[`reference/ggggg`](reference/ggggg/README.md).

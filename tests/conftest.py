from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat


@pytest.fixture
def repo_root():
    return Path(__file__).resolve().parents[1]


def write_pdb(path, atom_specs):
    lines = []

    for serial, atom_name, residue_name, partition_id in atom_specs:
        line = (
            f"ATOM {serial} {atom_name} {residue_name} {partition_id} "
            "0.0 0.0 0.0 1.00 0.00\n"
        )
        lines.append(line)

    path.write_text("".join(lines), encoding="ascii")


def write_mat(path, variable_name, matrix, extra_variables=None):
    values = {variable_name: matrix}

    if extra_variables is not None:
        for name, value in extra_variables.items():
            values[name] = value

    savemat(path, values)


@pytest.fixture
def small_model_files(tmp_path):
    pdb_path = tmp_path / "small.pdb"
    mat_path = tmp_path / "small.mat"
    atom_specs = [
        (1, "H1", "DA", 1),
        (2, "C1", "DT", 2),
        (3, "HG", "HG", 3),
    ]
    write_pdb(pdb_path, atom_specs)
    write_mat(mat_path, "small", np.eye(40))
    return pdb_path, mat_path

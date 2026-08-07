from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from src.model_data import LoadedModel, ModelPartition


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


@pytest.fixture
def interleaved_model():
    hamiltonian = np.array(
        [
            [0.0, 0.2, 0.0, 0.1],
            [0.2, 0.1, 0.3, 0.0],
            [0.0, 0.3, -0.2, 0.4],
            [0.1, 0.0, 0.4, 0.3],
        ]
    )
    left = ModelPartition(
        partition_id=1,
        residue_names=("L1", "L2"),
        atom_start=1,
        atom_stop=4,
        atom_count=2,
        element_counts=(("H", 2),),
        orbital_start=0,
        orbital_stop=4,
        orbital_ranges=((0, 1), (3, 4)),
        source_partition_ids=(1, 4),
    )
    probe = ModelPartition(
        partition_id=2,
        residue_names=("P",),
        atom_start=2,
        atom_stop=2,
        atom_count=1,
        element_counts=(("H", 1),),
        orbital_start=1,
        orbital_stop=2,
        orbital_ranges=((1, 2),),
        source_partition_ids=(2,),
    )
    right = ModelPartition(
        partition_id=3,
        residue_names=("R",),
        atom_start=3,
        atom_stop=3,
        atom_count=1,
        element_counts=(("H", 1),),
        orbital_start=2,
        orbital_stop=3,
        orbital_ranges=((2, 3),),
        source_partition_ids=(3,),
    )

    return LoadedModel(
        hamiltonian=hamiltonian,
        atoms=tuple(),
        atom_blocks=tuple(),
        partitions=(left, probe, right),
        variable_name="interleaved",
        max_hermitian_error=0.0,
    )

import subprocess
import sys

import numpy as np

from conftest import write_mat, write_pdb


def test_inspection_command_prints_and_exports_partition_table(
    tmp_path,
    repo_root,
):
    pdb_path = tmp_path / "inspect.pdb"
    mat_path = tmp_path / "inspect.mat"
    output_path = tmp_path / "partition_table.csv"
    atom_specs = []

    for partition_id in range(1, 15):
        atom_specs.append((partition_id, "H1", "DA", partition_id))

    atom_specs.append((15, "HG", "HG", 15))
    write_pdb(pdb_path, atom_specs)
    write_mat(mat_path, "inspect", np.eye(90))
    command = [
        sys.executable,
        "inspect_model.py",
        "--pdb",
        str(pdb_path),
        "--hamiltonian",
        str(mat_path),
        "--variable",
        "inspect",
        "--output",
        str(output_path),
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "model checkpoints" in result.stdout
    assert "15 passed" in result.stdout
    assert "90 passed" in result.stdout
    assert "one-based range" in result.stdout
    assert output_path.exists()

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 16
    assert "source_partition_ids" in lines[0]
    assert "python_orbital_start" in lines[0]
    assert "one_based_orbital_end" in lines[0]

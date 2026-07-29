from pathlib import Path

import numpy as np
import pytest

from conftest import write_mat, write_pdb
from src.model_data import (
    ModelValidationError,
    ORBITALS_BY_ELEMENT,
    load_model,
    parse_pdb,
)


def test_orbital_counts_match_supplied_basis():
    assert ORBITALS_BY_ELEMENT["H"] == 5
    assert ORBITALS_BY_ELEMENT["C"] == 15
    assert ORBITALS_BY_ELEMENT["O"] == 15
    assert ORBITALS_BY_ELEMENT["N"] == 15
    assert ORBITALS_BY_ELEMENT["P"] == 19
    assert ORBITALS_BY_ELEMENT["Hg"] == 20


def test_small_model_maps_atoms_and_partitions(small_model_files):
    pdb_path, mat_path = small_model_files
    model = load_model(
        pdb_path,
        mat_path,
        "small",
        expected_partition_count=3,
    )

    assert len(model.atoms) == 3
    assert len(model.partitions) == 3
    assert model.hamiltonian.shape == (40, 40)

    expected_atom_ranges = [(0, 5), (5, 20), (20, 40)]

    for index in range(len(expected_atom_ranges)):
        expected_start, expected_stop = expected_atom_ranges[index]
        block = model.atom_blocks[index]
        assert block.orbital_start == expected_start
        assert block.orbital_stop == expected_stop

    assert model.partitions[0].orbital_start == 0
    assert model.partitions[0].orbital_stop == 5
    assert model.partitions[1].orbital_start == 5
    assert model.partitions[1].orbital_stop == 20
    assert model.partitions[2].orbital_start == 20
    assert model.partitions[2].orbital_stop == 40
    assert model.partitions[2].element_counts == (("Hg", 1),)


def test_parser_ignores_non_atom_records(tmp_path):
    pdb_path = tmp_path / "records.pdb"
    pdb_path.write_text(
        "ATOM 1 H1 DA 1 0 0 0 1 0\n"
        "TER\n"
        "ATOM 2 HG HG 2 0 0 0 1 0\n"
        "END\n",
        encoding="ascii",
    )
    atoms = parse_pdb(pdb_path)
    assert len(atoms) == 2
    assert atoms[1].element == "Hg"


def test_parser_rejects_unknown_element(tmp_path):
    pdb_path = tmp_path / "unknown.pdb"
    write_pdb(pdb_path, [(1, "S1", "DA", 1)])

    with pytest.raises(ModelValidationError, match="unsupported element"):
        parse_pdb(pdb_path)


def test_parser_rejects_serial_gap(tmp_path):
    pdb_path = tmp_path / "serial_gap.pdb"
    atom_specs = [
        (1, "H1", "DA", 1),
        (3, "HG", "HG", 2),
    ]
    write_pdb(pdb_path, atom_specs)

    with pytest.raises(ModelValidationError, match="atom serials"):
        parse_pdb(pdb_path)


def test_loader_rejects_partition_gap(tmp_path):
    pdb_path = tmp_path / "partition_gap.pdb"
    mat_path = tmp_path / "partition_gap.mat"
    atom_specs = [
        (1, "H1", "DA", 1),
        (2, "HG", "HG", 3),
    ]
    write_pdb(pdb_path, atom_specs)
    write_mat(mat_path, "model", np.eye(25))

    with pytest.raises(ModelValidationError, match="partition IDs"):
        load_model(
            pdb_path,
            mat_path,
            "model",
            expected_partition_count=3,
        )


def test_loader_rejects_partition_split_into_two_blocks(tmp_path):
    pdb_path = tmp_path / "split_partition.pdb"
    mat_path = tmp_path / "split_partition.mat"
    atom_specs = [
        (1, "H1", "DA", 1),
        (2, "C1", "DT", 2),
        (3, "H2", "DA", 1),
        (4, "HG", "HG", 3),
    ]
    write_pdb(pdb_path, atom_specs)
    write_mat(mat_path, "model", np.eye(45))

    with pytest.raises(ModelValidationError, match="not one atom block"):
        load_model(
            pdb_path,
            mat_path,
            "model",
            expected_partition_count=3,
        )


def test_loader_rejects_hg_that_is_not_its_own_final_partition(tmp_path):
    pdb_path = tmp_path / "bad_hg.pdb"
    mat_path = tmp_path / "bad_hg.mat"
    atom_specs = [
        (1, "H1", "DA", 1),
        (2, "HG", "HG", 2),
        (3, "C1", "DT", 2),
    ]
    write_pdb(pdb_path, atom_specs)
    write_mat(mat_path, "model", np.eye(40))

    with pytest.raises(ModelValidationError, match="Hg must be the final atom"):
        load_model(
            pdb_path,
            mat_path,
            "model",
            expected_partition_count=2,
        )


def test_loader_rejects_missing_matlab_variable(small_model_files):
    pdb_path, mat_path = small_model_files

    with pytest.raises(ModelValidationError, match="was not found"):
        load_model(
            pdb_path,
            mat_path,
            "missing",
            expected_partition_count=3,
        )


def test_loader_rejects_extra_matlab_variable(tmp_path):
    pdb_path = tmp_path / "extra.pdb"
    mat_path = tmp_path / "extra.mat"
    atom_specs = [
        (1, "H1", "DA", 1),
        (2, "C1", "DT", 2),
        (3, "HG", "HG", 3),
    ]
    write_pdb(pdb_path, atom_specs)
    write_mat(
        mat_path,
        "model",
        np.eye(40),
        extra_variables={"other": np.eye(2)},
    )

    with pytest.raises(ModelValidationError, match="only the requested"):
        load_model(
            pdb_path,
            mat_path,
            "model",
            expected_partition_count=3,
        )


def test_loader_rejects_nonsquare_hamiltonian(tmp_path):
    pdb_path = tmp_path / "nonsquare.pdb"
    mat_path = tmp_path / "nonsquare.mat"
    atom_specs = [
        (1, "H1", "DA", 1),
        (2, "C1", "DT", 2),
        (3, "HG", "HG", 3),
    ]
    write_pdb(pdb_path, atom_specs)
    write_mat(mat_path, "model", np.zeros((40, 39)))

    with pytest.raises(ModelValidationError, match="must be square"):
        load_model(
            pdb_path,
            mat_path,
            "model",
            expected_partition_count=3,
        )


def test_loader_rejects_nonfinite_hamiltonian(tmp_path):
    pdb_path = tmp_path / "nonfinite.pdb"
    mat_path = tmp_path / "nonfinite.mat"
    atom_specs = [
        (1, "H1", "DA", 1),
        (2, "C1", "DT", 2),
        (3, "HG", "HG", 3),
    ]
    matrix = np.eye(40)
    matrix[0, 0] = np.nan
    write_pdb(pdb_path, atom_specs)
    write_mat(mat_path, "model", matrix)

    with pytest.raises(ModelValidationError, match="nonfinite"):
        load_model(
            pdb_path,
            mat_path,
            "model",
            expected_partition_count=3,
        )


def test_loader_rejects_orbital_count_mismatch(tmp_path):
    pdb_path = tmp_path / "mismatch.pdb"
    mat_path = tmp_path / "mismatch.mat"
    atom_specs = [
        (1, "H1", "DA", 1),
        (2, "C1", "DT", 2),
        (3, "HG", "HG", 3),
    ]
    write_pdb(pdb_path, atom_specs)
    write_mat(mat_path, "model", np.eye(39))

    with pytest.raises(ModelValidationError, match="does not match"):
        load_model(
            pdb_path,
            mat_path,
            "model",
            expected_partition_count=3,
        )


def test_loader_rejects_nonhermitian_hamiltonian(tmp_path):
    pdb_path = tmp_path / "nonhermitian.pdb"
    mat_path = tmp_path / "nonhermitian.mat"
    atom_specs = [
        (1, "H1", "DA", 1),
        (2, "C1", "DT", 2),
        (3, "HG", "HG", 3),
    ]
    matrix = np.eye(40)
    matrix[0, 1] = 1.0
    write_pdb(pdb_path, atom_specs)
    write_mat(mat_path, "model", matrix)

    with pytest.raises(ModelValidationError, match="not Hermitian"):
        load_model(
            pdb_path,
            mat_path,
            "model",
            expected_partition_count=3,
        )


def test_real_model_matches_audited_dimensions(repo_root):
    pdb_path = repo_root / "data" / "xx7tg6.pdb"
    mat_path = repo_root / "data" / "xx7tg6.mat"

    if not mat_path.exists():
        pytest.skip("local real Hamiltonian is not available")

    model = load_model(pdb_path, mat_path, "xx7tg6")
    assert len(model.atoms) == 444
    assert len(model.partitions) == 15
    assert model.hamiltonian.shape == (5083, 5083)
    assert model.partitions[-1].orbital_count == 20
    assert model.partitions[-1].orbital_stop == 5083
    assert model.max_hermitian_error <= 1e-7


def test_real_pdb_partition_orbital_counts(repo_root):
    pdb_path = Path(repo_root) / "data" / "xx7tg6.pdb"
    atoms = parse_pdb(pdb_path)
    counts = {}

    for atom in atoms:
        if atom.partition_id not in counts:
            counts[atom.partition_id] = 0

        counts[atom.partition_id] = (
            counts[atom.partition_id] + ORBITALS_BY_ELEMENT[atom.element]
        )

    expected = [
        320,
        389,
        364,
        364,
        364,
        389,
        394,
        300,
        344,
        374,
        364,
        374,
        344,
        379,
        20,
    ]

    for partition_index in range(len(expected)):
        partition_id = partition_index + 1
        assert counts[partition_id] == expected[partition_index]

    assert sum(expected) == 5083

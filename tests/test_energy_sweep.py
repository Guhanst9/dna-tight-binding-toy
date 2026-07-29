import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from conftest import write_mat, write_pdb
from src.energy_solver import ConvergenceError
from src.energy_sweep import (
    SweepConfig,
    build_energy_grid,
    run_energy_sweep,
    validate_sweep_config,
)
from src.model_data import LoadedModel, ModelPartition
from src.transport_setup import build_transport_setup


def build_three_site_model():
    hamiltonian = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    partitions = []

    for partition_index in range(3):
        partition_id = partition_index + 1
        partition = ModelPartition(
            partition_id=partition_id,
            residue_names=(f"P{partition_id}",),
            atom_start=partition_id,
            atom_stop=partition_id,
            atom_count=1,
            element_counts=(("H", 1),),
            orbital_start=partition_index,
            orbital_stop=partition_index + 1,
        )
        partitions.append(partition)

    return LoadedModel(
        hamiltonian=hamiltonian,
        atoms=tuple(),
        atom_blocks=tuple(),
        partitions=tuple(partitions),
        variable_name="three_site",
        max_hermitian_error=0.0,
    )


def build_sweep_values(mode="coherent", d0=None, alpha=None, max_iterations=None):
    model = build_three_site_model()
    setup = build_transport_setup(
        model,
        left_partition_id=1,
        right_partition_id=3,
        gamma_left=1.0,
        gamma_right=1.0,
        energy_minimum=-1.0,
        energy_maximum=1.0,
        energy_step=0.5,
    )
    config = SweepConfig(
        mode=mode,
        energy_minimum=-1.0,
        energy_maximum=1.0,
        energy_step=0.5,
        d0=d0,
        alpha=alpha,
        max_iterations=max_iterations,
    )
    return model, setup, config


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def test_energy_grid_includes_both_endpoints():
    energies = build_energy_grid(-1.0, 1.0, 0.5)
    assert np.allclose(energies, [-1.0, -0.5, 0.0, 0.5, 1.0])


def test_energy_step_must_divide_range():
    with pytest.raises(ValueError, match="divide"):
        build_energy_grid(-1.0, 1.0, 0.3)


def test_coherent_sweep_writes_checkpoints_and_csv_files(tmp_path):
    model, setup, config = build_sweep_values()
    summary = run_energy_sweep(
        model,
        setup,
        config,
        tmp_path,
        history_indices=(2,),
    )

    assert summary.energy_count == 5
    assert summary.processed_count == 5
    assert summary.skipped_count == 0
    assert len(list((tmp_path / "checkpoints").glob("*.json"))) == 5

    transport_rows = read_csv(tmp_path / "transport.csv")
    partition_rows = read_csv(tmp_path / "partition_results.csv")
    history_rows = read_csv(tmp_path / "history" / "energy_000002.csv")

    assert len(transport_rows) == 5
    assert len(partition_rows) == 15
    assert len(history_rows) == 3
    assert transport_rows[0]["energy_index"] == "0"
    assert transport_rows[-1]["energy_index"] == "4"
    assert partition_rows[0]["partition_id"] == "1"
    assert partition_rows[1]["is_probe"] == "True"


def test_resume_skips_completed_energy_points(tmp_path, monkeypatch):
    model, setup, config = build_sweep_values()
    first_summary = run_energy_sweep(model, setup, config, tmp_path)
    assert first_summary.processed_count == 5

    def fail_if_called(model_value, setup_value, energy_value):
        raise AssertionError("completed energy was recalculated")

    monkeypatch.setattr(
        "src.energy_sweep.run_coherent_energy",
        fail_if_called,
    )
    second_summary = run_energy_sweep(model, setup, config, tmp_path)

    assert second_summary.processed_count == 0
    assert second_summary.skipped_count == 5
    assert len(read_csv(tmp_path / "transport.csv")) == 5
    assert len(read_csv(tmp_path / "partition_results.csv")) == 15


def test_bounded_ranges_can_complete_one_output_directory(tmp_path):
    model, setup, config = build_sweep_values()
    first_summary = run_energy_sweep(
        model,
        setup,
        config,
        tmp_path,
        start_index=0,
        stop_index=2,
    )
    second_summary = run_energy_sweep(
        model,
        setup,
        config,
        tmp_path,
        start_index=2,
        stop_index=5,
    )

    assert first_summary.processed_count == 2
    assert second_summary.processed_count == 3
    assert len(read_csv(tmp_path / "transport.csv")) == 5


def test_summary_rows_are_sorted_after_reverse_bounded_ranges(tmp_path):
    model, setup, config = build_sweep_values()
    run_energy_sweep(
        model,
        setup,
        config,
        tmp_path,
        start_index=2,
        stop_index=5,
    )
    run_energy_sweep(
        model,
        setup,
        config,
        tmp_path,
        start_index=0,
        stop_index=2,
    )

    rows = read_csv(tmp_path / "transport.csv")
    energy_indices = []

    for row in rows:
        energy_indices.append(int(row["energy_index"]))

    assert energy_indices == [0, 1, 2, 3, 4]


def test_successful_retry_removes_old_failure_file(tmp_path):
    model, setup, config = build_sweep_values()
    failure_directory = tmp_path / "failures"
    failure_directory.mkdir()
    old_failure_path = failure_directory / "energy_000000.json"
    old_failure_path.write_text("{}\n", encoding="utf-8")

    run_energy_sweep(
        model,
        setup,
        config,
        tmp_path,
        start_index=0,
        stop_index=1,
    )

    assert not old_failure_path.exists()


def test_different_settings_cannot_resume_same_output(tmp_path):
    model, setup, config = build_sweep_values()
    run_energy_sweep(model, setup, config, tmp_path, stop_index=1)
    different_config = SweepConfig(
        mode="coherent",
        energy_minimum=-1.0,
        energy_maximum=1.0,
        energy_step=0.25,
        d0=None,
        alpha=None,
        max_iterations=None,
    )
    different_setup = build_transport_setup(
        model,
        left_partition_id=1,
        right_partition_id=3,
        energy_minimum=-1.0,
        energy_maximum=1.0,
        energy_step=0.25,
    )

    with pytest.raises(ValueError, match="different run settings"):
        run_energy_sweep(
            model,
            different_setup,
            different_config,
            tmp_path,
            stop_index=1,
        )


def test_decoherent_sweep_requires_solver_settings():
    config = SweepConfig(
        mode="decoherent",
        energy_minimum=-1.0,
        energy_maximum=1.0,
        energy_step=0.5,
        d0=None,
        alpha=None,
        max_iterations=None,
    )

    with pytest.raises(ValueError, match="requires d0"):
        validate_sweep_config(config)


def test_zero_d0_decoherent_sweep_converges(tmp_path):
    model, setup, config = build_sweep_values(
        mode="decoherent",
        d0=0.0,
        alpha=0.3,
        max_iterations=10,
    )
    summary = run_energy_sweep(
        model,
        setup,
        config,
        tmp_path,
        start_index=2,
        stop_index=3,
        history_indices=(2,),
    )
    checkpoint_path = tmp_path / "checkpoints" / "energy_000002.json"

    with checkpoint_path.open("r", encoding="utf-8") as checkpoint_file:
        checkpoint = json.load(checkpoint_file)

    assert summary.processed_count == 1
    assert checkpoint["converged"]
    assert checkpoint["iterations"] == 2
    assert len(checkpoint["partitions"]) == 3
    assert len(checkpoint["history"]) == 6


def test_nonconvergence_writes_failure_and_stops(tmp_path):
    model, setup, config = build_sweep_values(
        mode="decoherent",
        d0=1.0,
        alpha=1.0,
        max_iterations=2,
    )

    with pytest.raises(ConvergenceError):
        run_energy_sweep(
            model,
            setup,
            config,
            tmp_path,
            start_index=2,
            stop_index=3,
            history_indices=(2,),
        )

    failure_path = tmp_path / "failures" / "energy_000002.json"
    assert failure_path.exists()
    assert not (tmp_path / "checkpoints" / "energy_000002.json").exists()


@pytest.mark.parametrize(
    "start_index,stop_index",
    [
        (-1, 2),
        (2, 2),
        (3, 2),
        (0, 6),
    ],
)
def test_invalid_index_range_is_rejected(tmp_path, start_index, stop_index):
    model, setup, config = build_sweep_values()

    with pytest.raises(ValueError):
        run_energy_sweep(
            model,
            setup,
            config,
            tmp_path,
            start_index=start_index,
            stop_index=stop_index,
        )


def test_cli_runs_one_coherent_energy(tmp_path, repo_root):
    pdb_path = tmp_path / "cli.pdb"
    mat_path = tmp_path / "cli.mat"
    output_path = tmp_path / "cli_output"
    atom_specs = []

    for partition_id in range(1, 15):
        atom_specs.append((partition_id, "H1", "DA", partition_id))

    atom_specs.append((15, "HG", "HG", 15))
    write_pdb(pdb_path, atom_specs)
    write_mat(mat_path, "cli", np.eye(90))
    command = [
        sys.executable,
        "run_transport.py",
        "--pdb",
        str(pdb_path),
        "--hamiltonian",
        str(mat_path),
        "--variable",
        "cli",
        "--mode",
        "coherent",
        "--energy-min",
        "-1",
        "--energy-max",
        "1",
        "--energy-step",
        "1",
        "--start-index",
        "0",
        "--stop-index",
        "1",
        "--output-dir",
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
    assert "completed 0: -1.00 eV" in result.stdout
    assert "processed: 1" in result.stdout
    assert (output_path / "checkpoints" / "energy_000000.json").exists()

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import time

import numpy as np

from .energy_solver import (
    ConvergenceError,
    run_coherent_energy,
    run_dos_weighted_energy,
    validate_solver_settings,
)


@dataclass(frozen=True)
class SweepConfig:
    mode: str
    energy_minimum: float
    energy_maximum: float
    energy_step: float
    d0: float | None
    alpha: float | None
    max_iterations: int | None


@dataclass(frozen=True)
class SweepSummary:
    energy_count: int
    processed_count: int
    skipped_count: int
    elapsed_seconds: float


def build_energy_grid(minimum, maximum, step):
    if not np.isfinite(minimum):
        raise ValueError("energy minimum must be finite")

    if not np.isfinite(maximum):
        raise ValueError("energy maximum must be finite")

    if not np.isfinite(step):
        raise ValueError("energy step must be finite")

    if minimum >= maximum:
        raise ValueError("energy minimum must be less than energy maximum")

    if step <= 0:
        raise ValueError("energy step must be positive")

    interval_count = int(round((maximum - minimum) / step))
    calculated_maximum = minimum + interval_count * step

    if not np.isclose(calculated_maximum, maximum, atol=1e-12, rtol=0):
        raise ValueError("energy step must divide the requested range")

    energies = np.zeros(interval_count + 1)

    for energy_index in range(interval_count + 1):
        energies[energy_index] = minimum + energy_index * step

    energies[-1] = maximum
    return energies


def validate_sweep_config(config):
    if config.mode != "coherent" and config.mode != "decoherent":
        raise ValueError("mode must be coherent or decoherent")

    build_energy_grid(
        config.energy_minimum,
        config.energy_maximum,
        config.energy_step,
    )

    if config.mode == "decoherent":
        if config.d0 is None:
            raise ValueError("decoherent mode requires d0")

        if config.alpha is None:
            raise ValueError("decoherent mode requires alpha")

        if config.max_iterations is None:
            raise ValueError("decoherent mode requires max iterations")

        validate_solver_settings(
            config.d0,
            config.alpha,
            config.max_iterations,
        )


def validate_setup_grid(setup, config):
    if setup.energy_grid.minimum != config.energy_minimum:
        raise ValueError("setup energy minimum does not match the sweep")

    if setup.energy_grid.maximum != config.energy_maximum:
        raise ValueError("setup energy maximum does not match the sweep")

    if setup.energy_grid.step != config.energy_step:
        raise ValueError("setup energy step does not match the sweep")


def validate_index_range(start_index, stop_index, energy_count):
    if not isinstance(start_index, int):
        raise ValueError("start index must be an integer")

    if not isinstance(stop_index, int):
        raise ValueError("stop index must be an integer")

    if start_index < 0:
        raise ValueError("start index must be nonnegative")

    if stop_index > energy_count:
        raise ValueError("stop index exceeds the energy grid")

    if start_index >= stop_index:
        raise ValueError("start index must be less than stop index")


def validate_history_indices(history_indices, energy_count):
    values = []

    for energy_index in history_indices:
        if not isinstance(energy_index, int):
            raise ValueError("history energy indices must be integers")

        if energy_index < 0 or energy_index >= energy_count:
            raise ValueError("history energy index is outside the energy grid")

        if energy_index not in values:
            values.append(energy_index)

    return tuple(values)


def build_run_settings(model, setup, config):
    partition_ranges = []

    for partition in setup.partitions:
        values = {
            "partition_id": partition.partition_id,
            "orbital_start": partition.orbital_start,
            "orbital_stop": partition.orbital_stop,
        }

        if model.partition_scheme != "pdb":
            values["source_partition_ids"] = list(
                partition.source_partition_ids
            )
            values["orbital_ranges"] = []

            for orbital_start, orbital_stop in partition.orbital_ranges:
                values["orbital_ranges"].append(
                    [orbital_start, orbital_stop]
                )

        partition_ranges.append(values)

    settings = {
        "mode": config.mode,
        "variable_name": model.variable_name,
        "hamiltonian_size": model.hamiltonian.shape[0],
        "partition_ranges": partition_ranges,
        "left_partition_id": setup.left_partition.partition_id,
        "right_partition_id": setup.right_partition.partition_id,
        "gamma_left_ev": setup.gamma_left,
        "gamma_right_ev": setup.gamma_right,
        "energy_minimum_ev": config.energy_minimum,
        "energy_maximum_ev": config.energy_maximum,
        "energy_step_ev": config.energy_step,
        "d0_ev2": config.d0,
        "alpha": config.alpha,
        "max_iterations": config.max_iterations,
    }

    if model.partition_scheme != "pdb":
        settings["partition_scheme"] = model.partition_scheme

    return settings


def write_json(path, values):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(values, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")

    temporary_path.replace(output_path)


def load_or_write_run_settings(output_directory, settings):
    settings_path = output_directory / "run_config.json"

    if settings_path.exists():
        with settings_path.open("r", encoding="utf-8") as settings_file:
            existing_settings = json.load(settings_file)

        if existing_settings != settings:
            raise ValueError("output directory contains different run settings")

        return

    write_json(settings_path, settings)


def optional_number(value):
    if value is None:
        return None

    if not np.isfinite(value):
        return None

    return float(value)


def build_partition_rows(result, setup, energy_index):
    rows = []

    for partition_index in range(len(setup.partitions)):
        partition = setup.partitions[partition_index]
        sigma_value = result.sigma_decoherence_by_partition[partition_index]
        row = {
            "energy_index": energy_index,
            "energy_ev": float(result.energy),
            "partition_id": partition.partition_id,
            "is_probe": bool(setup.probe_mask[partition_index]),
            "ldos_per_ev": float(result.partition_ldos[partition_index]),
            "sigma_real_ev": float(np.real(sigma_value)),
            "sigma_imaginary_ev": float(np.imag(sigma_value)),
            "gamma_b_ev": float(
                result.gamma_decoherence_by_partition[partition_index]
            ),
        }
        rows.append(row)

    return rows


def build_checkpoint(result, setup, energy_index, elapsed_seconds, save_history):
    checkpoint = {
        "energy_index": energy_index,
        "energy_ev": float(result.energy),
        "converged": bool(result.converged),
        "iterations": result.iterations,
        "dos_per_ev": float(result.dos),
        "t_lr": float(result.t_lr),
        "t_eff": float(result.t_eff),
        "dos_change_percent": optional_number(result.dos_change_percent),
        "transmission_change_percent": optional_number(
            result.transmission_change_percent
        ),
        "elapsed_seconds": float(elapsed_seconds),
        "partitions": build_partition_rows(result, setup, energy_index),
    }

    if save_history:
        history_rows = build_history_rows(result, setup, energy_index)
        checkpoint["history"] = history_rows

    return checkpoint


def build_history_rows(result, setup, energy_index):
    rows = []

    for record in result.history:
        for partition_index in range(len(setup.partitions)):
            partition = setup.partitions[partition_index]
            sigma_value = record.sigma_decoherence_by_partition[
                partition_index
            ]
            green_sum = record.partition_green_sums[partition_index]
            row = {
                "energy_index": energy_index,
                "energy_ev": float(result.energy),
                "iteration": record.iteration,
                "partition_id": partition.partition_id,
                "is_probe": bool(setup.probe_mask[partition_index]),
                "dos_per_ev": float(record.dos),
                "t_lr": float(record.t_lr),
                "t_eff": float(record.t_eff),
                "dos_change_percent": optional_number(
                    record.dos_change_percent
                ),
                "transmission_change_percent": optional_number(
                    record.transmission_change_percent
                ),
                "partition_ldos_per_ev": float(
                    record.partition_ldos[partition_index]
                ),
                "green_sum_real_per_ev": float(np.real(green_sum)),
                "green_sum_imaginary_per_ev": float(np.imag(green_sum)),
                "sigma_real_ev": float(np.real(sigma_value)),
                "sigma_imaginary_ev": float(np.imag(sigma_value)),
                "gamma_b_ev": float(
                    record.gamma_decoherence_by_partition[partition_index]
                ),
            }
            rows.append(row)

    return rows


def checkpoint_path(output_directory, energy_index):
    filename = f"energy_{energy_index:06d}.json"
    return output_directory / "checkpoints" / filename


def failure_path(output_directory, energy_index):
    filename = f"energy_{energy_index:06d}.json"
    return output_directory / "failures" / filename


def history_path(output_directory, energy_index):
    filename = f"energy_{energy_index:06d}.csv"
    return output_directory / "history" / filename


def validate_checkpoint(checkpoint, energies, partition_count):
    energy_index = checkpoint.get("energy_index")

    if not isinstance(energy_index, int):
        raise ValueError("checkpoint energy index is invalid")

    if energy_index < 0 or energy_index >= len(energies):
        raise ValueError("checkpoint energy index is outside the grid")

    checkpoint_energy = checkpoint.get("energy_ev")

    if not isinstance(checkpoint_energy, (int, float)):
        raise ValueError("checkpoint energy is invalid")

    if not np.isfinite(checkpoint_energy):
        raise ValueError("checkpoint energy is invalid")

    expected_energy = energies[energy_index]

    if not np.isclose(checkpoint_energy, expected_energy, atol=1e-12):
        raise ValueError("checkpoint energy does not match its index")

    partitions = checkpoint.get("partitions")

    if not isinstance(partitions, list):
        raise ValueError("checkpoint partition data is missing")

    if len(partitions) != partition_count:
        raise ValueError("checkpoint partition count is incorrect")


def load_checkpoints(output_directory, energies, partition_count):
    checkpoints = {}
    checkpoint_directory = output_directory / "checkpoints"

    if not checkpoint_directory.exists():
        return checkpoints

    paths = sorted(checkpoint_directory.glob("energy_*.json"))

    for path in paths:
        with path.open("r", encoding="utf-8") as checkpoint_file:
            checkpoint = json.load(checkpoint_file)

        validate_checkpoint(checkpoint, energies, partition_count)
        energy_index = checkpoint["energy_index"]

        if energy_index in checkpoints:
            raise ValueError(f"duplicate checkpoint for energy {energy_index}")

        checkpoints[energy_index] = checkpoint

    return checkpoints


def write_csv(path, fieldnames, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path, fieldnames, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerows(rows)


def transport_fieldnames():
    return [
        "energy_index",
        "energy_ev",
        "converged",
        "iterations",
        "dos_per_ev",
        "t_lr",
        "t_eff",
        "dos_change_percent",
        "transmission_change_percent",
        "elapsed_seconds",
    ]


def partition_fieldnames():
    return [
        "energy_index",
        "energy_ev",
        "partition_id",
        "is_probe",
        "ldos_per_ev",
        "sigma_real_ev",
        "sigma_imaginary_ev",
        "gamma_b_ev",
    ]


def history_fieldnames():
    return [
        "energy_index",
        "energy_ev",
        "iteration",
        "partition_id",
        "is_probe",
        "dos_per_ev",
        "t_lr",
        "t_eff",
        "dos_change_percent",
        "transmission_change_percent",
        "partition_ldos_per_ev",
        "green_sum_real_per_ev",
        "green_sum_imaginary_per_ev",
        "sigma_real_ev",
        "sigma_imaginary_ev",
        "gamma_b_ev",
    ]


def checkpoint_transport_row(checkpoint):
    return {
        "energy_index": checkpoint["energy_index"],
        "energy_ev": checkpoint["energy_ev"],
        "converged": checkpoint["converged"],
        "iterations": checkpoint["iterations"],
        "dos_per_ev": checkpoint["dos_per_ev"],
        "t_lr": checkpoint["t_lr"],
        "t_eff": checkpoint["t_eff"],
        "dos_change_percent": checkpoint["dos_change_percent"],
        "transmission_change_percent": checkpoint[
            "transmission_change_percent"
        ],
        "elapsed_seconds": checkpoint["elapsed_seconds"],
    }


def rebuild_summary_files(output_directory, checkpoints):
    transport_rows = []
    partition_rows = []
    energy_indices = sorted(checkpoints)

    for energy_index in energy_indices:
        checkpoint = checkpoints[energy_index]
        transport_rows.append(checkpoint_transport_row(checkpoint))

        for row in checkpoint["partitions"]:
            partition_rows.append(row)

    write_csv(
        output_directory / "transport.csv",
        transport_fieldnames(),
        transport_rows,
    )
    write_csv(
        output_directory / "partition_results.csv",
        partition_fieldnames(),
        partition_rows,
    )


def write_history_file(output_directory, energy_index, history_rows):
    write_csv(
        history_path(output_directory, energy_index),
        history_fieldnames(),
        history_rows,
    )


def write_existing_history_files(
    output_directory,
    checkpoints,
    history_indices,
):
    for energy_index in history_indices:
        if energy_index not in checkpoints:
            continue

        checkpoint = checkpoints[energy_index]

        if "history" not in checkpoint:
            raise ValueError(
                f"checkpoint {energy_index} does not contain iteration history"
            )

        write_history_file(
            output_directory,
            energy_index,
            checkpoint["history"],
        )


def solve_energy(model, setup, config, energy):
    if config.mode == "coherent":
        return run_coherent_energy(model, setup, energy)

    return run_dos_weighted_energy(
        model,
        setup,
        energy,
        config.d0,
        config.alpha,
        config.max_iterations,
    )


def run_energy_sweep(
    model,
    setup,
    config,
    output_directory,
    start_index=0,
    stop_index=None,
    history_indices=tuple(),
    progress_callback=None,
):
    validate_sweep_config(config)
    validate_setup_grid(setup, config)
    energies = build_energy_grid(
        config.energy_minimum,
        config.energy_maximum,
        config.energy_step,
    )

    if stop_index is None:
        stop_index = len(energies)

    validate_index_range(start_index, stop_index, len(energies))
    history_indices = validate_history_indices(history_indices, len(energies))
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    settings = build_run_settings(model, setup, config)
    load_or_write_run_settings(output_directory, settings)
    checkpoints = load_checkpoints(
        output_directory,
        energies,
        len(setup.partitions),
    )
    rebuild_summary_files(output_directory, checkpoints)
    write_existing_history_files(
        output_directory,
        checkpoints,
        history_indices,
    )
    processed_count = 0
    skipped_count = 0
    sweep_start = time.perf_counter()

    for energy_index in range(start_index, stop_index):
        energy = float(energies[energy_index])

        if energy_index in checkpoints:
            skipped_count = skipped_count + 1

            if progress_callback is not None:
                progress_callback("skipped", energy_index, energy, None)

            continue

        point_start = time.perf_counter()

        try:
            result = solve_energy(model, setup, config, energy)
        except ConvergenceError as error:
            elapsed_seconds = time.perf_counter() - point_start
            save_history = energy_index in history_indices
            failure = build_checkpoint(
                error.result,
                setup,
                energy_index,
                elapsed_seconds,
                save_history,
            )
            write_json(
                failure_path(output_directory, energy_index),
                failure,
            )

            if save_history:
                write_history_file(
                    output_directory,
                    energy_index,
                    failure["history"],
                )

            raise

        elapsed_seconds = time.perf_counter() - point_start
        save_history = energy_index in history_indices
        checkpoint = build_checkpoint(
            result,
            setup,
            energy_index,
            elapsed_seconds,
            save_history,
        )
        write_json(
            checkpoint_path(output_directory, energy_index),
            checkpoint,
        )
        old_failure_path = failure_path(output_directory, energy_index)

        if old_failure_path.exists():
            old_failure_path.unlink()

        checkpoints[energy_index] = checkpoint
        transport_rows = [checkpoint_transport_row(checkpoint)]
        append_csv(
            output_directory / "transport.csv",
            transport_fieldnames(),
            transport_rows,
        )
        append_csv(
            output_directory / "partition_results.csv",
            partition_fieldnames(),
            checkpoint["partitions"],
        )

        if save_history:
            write_history_file(
                output_directory,
                energy_index,
                checkpoint["history"],
            )

        processed_count = processed_count + 1

        if progress_callback is not None:
            progress_callback(
                "completed",
                energy_index,
                energy,
                elapsed_seconds,
            )

    rebuild_summary_files(output_directory, checkpoints)
    elapsed_seconds = time.perf_counter() - sweep_start
    return SweepSummary(
        energy_count=len(energies),
        processed_count=processed_count,
        skipped_count=skipped_count,
        elapsed_seconds=elapsed_seconds,
    )

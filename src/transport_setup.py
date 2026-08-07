from dataclasses import dataclass

import numpy as np

from .model_data import ModelPartition


@dataclass(frozen=True)
class EnergyGridSettings:
    minimum: float
    maximum: float
    step: float


@dataclass(frozen=True)
class TransportSetup:
    left_partition: ModelPartition
    right_partition: ModelPartition
    probe_partitions: tuple
    partitions: tuple
    gamma_left: float
    gamma_right: float
    sigma_left_diagonal: np.ndarray
    sigma_right_diagonal: np.ndarray
    probe_mask: np.ndarray
    sigma_decoherence_by_partition: np.ndarray
    gamma_decoherence_by_partition: np.ndarray
    energy_grid: EnergyGridSettings


def validate_contact_strength(value, name):
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")

    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def validate_energy_grid(minimum, maximum, step):
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


def validate_partition_coverage(model):
    if len(model.partitions) < 2:
        raise ValueError("at least two partitions are required")

    partition_ids = set()
    orbital_ranges = []
    hamiltonian_size = model.hamiltonian.shape[0]

    for partition in model.partitions:
        if partition.partition_id in partition_ids:
            raise ValueError("partition IDs must be unique")

        partition_ids.add(partition.partition_id)

        if len(partition.orbital_ranges) == 0:
            raise ValueError("partition orbital ranges must not be empty")

        for orbital_start, orbital_stop in partition.orbital_ranges:
            if orbital_start < 0 or orbital_stop > hamiltonian_size:
                raise ValueError("partition orbital range is out of bounds")

            if orbital_stop <= orbital_start:
                raise ValueError("partition orbital ranges must not be empty")

            orbital_ranges.append((orbital_start, orbital_stop))

    orbital_ranges.sort()
    expected_start = 0

    for orbital_start, orbital_stop in orbital_ranges:
        if orbital_start != expected_start:
            raise ValueError("partition orbital ranges have a gap or overlap")

        expected_start = orbital_stop

    if expected_start != hamiltonian_size:
        raise ValueError(
            "partition orbital ranges do not cover the Hamiltonian"
        )


def find_partition(partitions, partition_id):
    for partition in partitions:
        if partition.partition_id == partition_id:
            return partition

    raise ValueError(f"partition {partition_id} does not exist")


def build_contact_diagonal(size, partition, gamma):
    self_energy = np.zeros(size, dtype=complex)
    value = -1j * gamma / 2

    for orbital_start, orbital_stop in partition.orbital_ranges:
        for orbital_index in range(orbital_start, orbital_stop):
            self_energy[orbital_index] = value

    return self_energy


def build_transport_setup(
    model,
    left_partition_id=1,
    right_partition_id=7,
    gamma_left=1.0,
    gamma_right=1.0,
    energy_minimum=-10.0,
    energy_maximum=0.0,
    energy_step=0.01,
):
    validate_partition_coverage(model)
    validate_contact_strength(gamma_left, "gamma left")
    validate_contact_strength(gamma_right, "gamma right")
    validate_energy_grid(energy_minimum, energy_maximum, energy_step)

    if left_partition_id == right_partition_id:
        raise ValueError("left and right contacts must use different partitions")

    left_partition = find_partition(model.partitions, left_partition_id)
    right_partition = find_partition(model.partitions, right_partition_id)
    probe_partitions = []
    probe_mask = np.zeros(len(model.partitions), dtype=bool)

    for partition_index in range(len(model.partitions)):
        partition = model.partitions[partition_index]

        if partition.partition_id == left_partition_id:
            continue

        if partition.partition_id == right_partition_id:
            continue

        probe_partitions.append(partition)
        probe_mask[partition_index] = True

    expected_probe_count = len(model.partitions) - 2

    if len(probe_partitions) != expected_probe_count:
        raise ValueError("contact and probe partitions overlap")

    size = model.hamiltonian.shape[0]
    sigma_left_diagonal = build_contact_diagonal(
        size,
        left_partition,
        gamma_left,
    )
    sigma_right_diagonal = build_contact_diagonal(
        size,
        right_partition,
        gamma_right,
    )

    overlap = np.logical_and(
        sigma_left_diagonal != 0,
        sigma_right_diagonal != 0,
    )

    if np.any(overlap):
        raise ValueError("left and right contact orbitals overlap")

    partition_count = len(model.partitions)
    sigma_decoherence_by_partition = np.zeros(
        partition_count,
        dtype=complex,
    )
    gamma_decoherence_by_partition = np.zeros(partition_count)
    energy_grid = EnergyGridSettings(
        minimum=energy_minimum,
        maximum=energy_maximum,
        step=energy_step,
    )

    return TransportSetup(
        left_partition=left_partition,
        right_partition=right_partition,
        probe_partitions=tuple(probe_partitions),
        partitions=model.partitions,
        gamma_left=gamma_left,
        gamma_right=gamma_right,
        sigma_left_diagonal=sigma_left_diagonal,
        sigma_right_diagonal=sigma_right_diagonal,
        probe_mask=probe_mask,
        sigma_decoherence_by_partition=sigma_decoherence_by_partition,
        gamma_decoherence_by_partition=gamma_decoherence_by_partition,
        energy_grid=energy_grid,
    )

from dataclasses import dataclass

import numpy as np


CONVERGENCE_PERCENT = 0.1
NEGATIVE_TOLERANCE = 1e-10
MINIMUM_PROBE_GAMMA = 1e-14


class ConvergenceError(RuntimeError):
    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    dos: float
    t_lr: float
    t_eff: float
    partition_green_sums: np.ndarray
    partition_ldos: np.ndarray
    sigma_decoherence_by_partition: np.ndarray
    gamma_decoherence_by_partition: np.ndarray
    active_probe_partition_ids: tuple
    dos_change_percent: float | None
    transmission_change_percent: float | None


@dataclass(frozen=True)
class EnergyResult:
    energy: float
    converged: bool
    iterations: int
    dos: float
    t_lr: float
    t_eff: float
    partition_ldos: np.ndarray
    sigma_decoherence_by_partition: np.ndarray
    gamma_decoherence_by_partition: np.ndarray
    dos_change_percent: float | None
    transmission_change_percent: float | None
    history: tuple


def validate_energy(energy):
    if not np.isfinite(energy):
        raise ValueError("energy must be finite")


def validate_solver_settings(d0, alpha, max_iterations):
    if not np.isfinite(d0):
        raise ValueError("d0 must be finite")

    if d0 < 0:
        raise ValueError("d0 must be nonnegative")

    if not np.isfinite(alpha):
        raise ValueError("alpha must be finite")

    if alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be greater than 0 and no more than 1")

    if not isinstance(max_iterations, int):
        raise ValueError("max iterations must be an integer")

    if max_iterations < 2:
        raise ValueError("max iterations must be at least 2")


def get_partition_index(setup, partition_id):
    for partition_index in range(len(setup.partitions)):
        partition = setup.partitions[partition_index]

        if partition.partition_id == partition_id:
            return partition_index

    raise ValueError(f"partition {partition_id} does not exist")


def validate_partition_values(values, setup, name):
    expected_shape = (len(setup.partitions),)

    if values.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains nonfinite values")


def build_decoherence_diagonal(setup, sigma_by_partition):
    validate_partition_values(
        sigma_by_partition,
        setup,
        "decoherence self-energy",
    )
    size = len(setup.sigma_left_diagonal)
    diagonal = np.zeros(size, dtype=complex)

    for partition_index in range(len(setup.partitions)):
        partition = setup.partitions[partition_index]
        value = sigma_by_partition[partition_index]

        if not setup.probe_mask[partition_index]:
            if value != 0:
                raise ValueError(
                    "contact partitions cannot have decoherence self-energy"
                )

            continue

        for orbital_start, orbital_stop in partition.orbital_ranges:
            for orbital_index in range(orbital_start, orbital_stop):
                diagonal[orbital_index] = value

    return diagonal


def solve_green_function(model, setup, energy, sigma_by_partition):
    validate_energy(energy)
    sigma_decoherence_diagonal = build_decoherence_diagonal(
        setup,
        sigma_by_partition,
    )
    matrix = -model.hamiltonian.astype(complex)
    diagonal_indices = np.diag_indices_from(matrix)
    matrix[diagonal_indices] = (
        matrix[diagonal_indices]
        + energy
        - setup.sigma_left_diagonal
        - setup.sigma_right_diagonal
        - sigma_decoherence_diagonal
    )

    try:
        green_function = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("green function matrix could not be inverted") from error

    if not np.all(np.isfinite(green_function)):
        raise ValueError("green function contains nonfinite values")

    return green_function


def calculate_partition_values(green_function, setup):
    expected_size = len(setup.sigma_left_diagonal)

    if green_function.shape != (expected_size, expected_size):
        raise ValueError("green function shape does not match the setup")

    diagonal = np.diag(green_function)
    partition_count = len(setup.partitions)
    green_sums = np.zeros(partition_count, dtype=complex)
    partition_ldos = np.zeros(partition_count)

    for partition_index in range(partition_count):
        partition = setup.partitions[partition_index]
        value = 0.0j

        for orbital_start, orbital_stop in partition.orbital_ranges:
            value = value + np.sum(diagonal[orbital_start:orbital_stop])

        ldos = -np.imag(value) / np.pi

        if ldos < -NEGATIVE_TOLERANCE:
            raise ValueError(
                f"partition {partition.partition_id} has negative LDOS"
            )

        if ldos < 0:
            ldos = 0.0

        green_sums[partition_index] = value
        partition_ldos[partition_index] = ldos

    dos = float(np.sum(partition_ldos))

    if not np.isfinite(dos):
        raise ValueError("DOS is not finite")

    return green_sums, partition_ldos, dos


def calculate_probe_gammas(sigma_by_partition, setup):
    validate_partition_values(
        sigma_by_partition,
        setup,
        "decoherence self-energy",
    )
    gamma_values = np.zeros(len(setup.partitions))

    for partition_index in range(len(setup.partitions)):
        value = -2 * np.imag(sigma_by_partition[partition_index])

        if not setup.probe_mask[partition_index]:
            if abs(value) > NEGATIVE_TOLERANCE:
                raise ValueError(
                    "contact partitions cannot have probe broadening"
                )

            continue

        if value < -NEGATIVE_TOLERANCE:
            partition = setup.partitions[partition_index]
            raise ValueError(
                f"partition {partition.partition_id} has negative probe broadening"
            )

        if value < 0:
            value = 0.0

        gamma_values[partition_index] = float(value)

    return gamma_values


def calculate_region_transmission(
    green_function,
    first_partition,
    first_gamma,
    second_partition,
    second_gamma,
):
    if first_gamma < 0 or second_gamma < 0:
        raise ValueError("region broadening must be nonnegative")

    block_sum = 0.0

    for first_start, first_stop in first_partition.orbital_ranges:
        for second_start, second_stop in second_partition.orbital_ranges:
            green_block = green_function[
                first_start:first_stop,
                second_start:second_stop,
            ]
            block_sum = block_sum + np.sum(np.abs(green_block) ** 2)

    transmission = first_gamma * second_gamma * block_sum
    transmission = float(np.real(transmission))

    if not np.isfinite(transmission):
        raise ValueError("transmission is not finite")

    if transmission < -NEGATIVE_TOLERANCE:
        raise ValueError("transmission is negative")

    if transmission < 0:
        transmission = 0.0

    return transmission


def get_active_probes(setup, gamma_by_partition):
    active_partitions = []
    active_gammas = []
    active_partition_ids = []

    for partition in setup.probe_partitions:
        partition_index = get_partition_index(setup, partition.partition_id)
        gamma_value = gamma_by_partition[partition_index]

        if gamma_value > MINIMUM_PROBE_GAMMA:
            active_partitions.append(partition)
            active_gammas.append(gamma_value)
            active_partition_ids.append(partition.partition_id)

    return active_partitions, active_gammas, tuple(active_partition_ids)


def build_w_matrix(probe_to_left, probe_to_right, probe_to_probe):
    probe_count = len(probe_to_left)

    if probe_to_right.shape != (probe_count,):
        raise ValueError("probe-to-right transmission shape is incorrect")

    if probe_to_probe.shape != (probe_count, probe_count):
        raise ValueError("probe transmission matrix shape is incorrect")

    w_matrix = np.zeros((probe_count, probe_count))

    for row in range(probe_count):
        diagonal_value = probe_to_left[row] + probe_to_right[row]

        for column in range(probe_count):
            if row == column:
                continue

            diagonal_value = diagonal_value + probe_to_probe[row, column]
            w_matrix[row, column] = -probe_to_probe[row, column]

        w_matrix[row, row] = diagonal_value

    if not np.all(np.isfinite(w_matrix)):
        raise ValueError("W matrix contains nonfinite values")

    return w_matrix


def calculate_effective_transmission(
    t_lr,
    left_to_probe,
    w_matrix,
    probe_to_right,
):
    if len(left_to_probe) == 0:
        return float(t_lr)

    try:
        probe_potentials = np.linalg.solve(w_matrix, probe_to_right)
    except np.linalg.LinAlgError as error:
        raise ValueError("W matrix is singular") from error

    t_eff = t_lr + np.dot(left_to_probe, probe_potentials)
    t_eff = float(np.real(t_eff))

    if not np.isfinite(t_eff):
        raise ValueError("effective transmission is not finite")

    if t_eff < -NEGATIVE_TOLERANCE:
        raise ValueError("effective transmission is negative")

    if t_eff < 0:
        t_eff = 0.0

    return t_eff


def calculate_transport(green_function, setup, gamma_by_partition):
    validate_partition_values(
        gamma_by_partition,
        setup,
        "probe broadening",
    )

    for partition_index in range(len(setup.partitions)):
        gamma_value = gamma_by_partition[partition_index]

        if gamma_value < -NEGATIVE_TOLERANCE:
            raise ValueError("probe broadening must be nonnegative")

        if not setup.probe_mask[partition_index]:
            if abs(gamma_value) > NEGATIVE_TOLERANCE:
                raise ValueError(
                    "contact partitions cannot have probe broadening"
                )

    t_lr = calculate_region_transmission(
        green_function,
        setup.left_partition,
        setup.gamma_left,
        setup.right_partition,
        setup.gamma_right,
    )
    active_partitions, active_gammas, active_partition_ids = get_active_probes(
        setup,
        gamma_by_partition,
    )
    probe_count = len(active_partitions)
    left_to_probe = np.zeros(probe_count)
    probe_to_left = np.zeros(probe_count)
    probe_to_right = np.zeros(probe_count)
    probe_to_probe = np.zeros((probe_count, probe_count))

    for probe_index in range(probe_count):
        probe_partition = active_partitions[probe_index]
        probe_gamma = active_gammas[probe_index]
        left_to_probe[probe_index] = calculate_region_transmission(
            green_function,
            setup.left_partition,
            setup.gamma_left,
            probe_partition,
            probe_gamma,
        )
        probe_to_left[probe_index] = calculate_region_transmission(
            green_function,
            probe_partition,
            probe_gamma,
            setup.left_partition,
            setup.gamma_left,
        )
        probe_to_right[probe_index] = calculate_region_transmission(
            green_function,
            probe_partition,
            probe_gamma,
            setup.right_partition,
            setup.gamma_right,
        )

    for row in range(probe_count):
        for column in range(probe_count):
            if row == column:
                continue

            probe_to_probe[row, column] = calculate_region_transmission(
                green_function,
                active_partitions[row],
                active_gammas[row],
                active_partitions[column],
                active_gammas[column],
            )

    w_matrix = build_w_matrix(
        probe_to_left,
        probe_to_right,
        probe_to_probe,
    )
    t_eff = calculate_effective_transmission(
        t_lr,
        left_to_probe,
        w_matrix,
        probe_to_right,
    )

    return t_lr, t_eff, active_partition_ids


def calculate_next_self_energy(
    current_green_sums,
    previous_green_sums,
    setup,
    d0,
    alpha,
):
    validate_partition_values(current_green_sums, setup, "current Green sums")
    validate_partition_values(previous_green_sums, setup, "previous Green sums")
    next_self_energy = np.zeros(len(setup.partitions), dtype=complex)
    factor = d0 / (2 * np.pi)

    for partition_index in range(len(setup.partitions)):
        if not setup.probe_mask[partition_index]:
            continue

        mixed_green_sum = (
            alpha * current_green_sums[partition_index]
            + (1 - alpha) * previous_green_sums[partition_index]
        )
        value = factor * mixed_green_sum
        gamma_value = -2 * np.imag(value)

        if gamma_value < -NEGATIVE_TOLERANCE:
            partition = setup.partitions[partition_index]
            raise ValueError(
                f"partition {partition.partition_id} has negative probe broadening"
            )

        if gamma_value < 0:
            value = complex(np.real(value), 0.0)

        next_self_energy[partition_index] = value

    if not np.all(np.isfinite(next_self_energy)):
        raise ValueError("decoherence self-energy contains nonfinite values")

    return next_self_energy


def calculate_percent_change(current_value, previous_value):
    if previous_value == 0:
        if current_value == 0:
            return 0.0

        return np.inf

    return abs(current_value - previous_value) / abs(previous_value) * 100


def build_iteration_record(
    iteration,
    dos,
    t_lr,
    t_eff,
    green_sums,
    partition_ldos,
    sigma_by_partition,
    gamma_by_partition,
    active_partition_ids,
    dos_change,
    transmission_change,
):
    return IterationRecord(
        iteration=iteration,
        dos=dos,
        t_lr=t_lr,
        t_eff=t_eff,
        partition_green_sums=green_sums.copy(),
        partition_ldos=partition_ldos.copy(),
        sigma_decoherence_by_partition=sigma_by_partition.copy(),
        gamma_decoherence_by_partition=gamma_by_partition.copy(),
        active_probe_partition_ids=active_partition_ids,
        dos_change_percent=dos_change,
        transmission_change_percent=transmission_change,
    )


def build_energy_result(energy, converged, history):
    final_record = history[-1]

    return EnergyResult(
        energy=energy,
        converged=converged,
        iterations=final_record.iteration,
        dos=final_record.dos,
        t_lr=final_record.t_lr,
        t_eff=final_record.t_eff,
        partition_ldos=final_record.partition_ldos.copy(),
        sigma_decoherence_by_partition=(
            final_record.sigma_decoherence_by_partition.copy()
        ),
        gamma_decoherence_by_partition=(
            final_record.gamma_decoherence_by_partition.copy()
        ),
        dos_change_percent=final_record.dos_change_percent,
        transmission_change_percent=(
            final_record.transmission_change_percent
        ),
        history=tuple(history),
    )


def run_coherent_energy(model, setup, energy):
    validate_energy(energy)
    sigma_by_partition = np.zeros(len(setup.partitions), dtype=complex)
    green_function = solve_green_function(
        model,
        setup,
        energy,
        sigma_by_partition,
    )
    green_sums, partition_ldos, dos = calculate_partition_values(
        green_function,
        setup,
    )
    gamma_by_partition = calculate_probe_gammas(
        sigma_by_partition,
        setup,
    )
    t_lr, t_eff, active_partition_ids = calculate_transport(
        green_function,
        setup,
        gamma_by_partition,
    )
    history = []
    history.append(
        build_iteration_record(
            1,
            dos,
            t_lr,
            t_eff,
            green_sums,
            partition_ldos,
            sigma_by_partition,
            gamma_by_partition,
            active_partition_ids,
            None,
            None,
        )
    )
    del green_function

    return build_energy_result(energy, True, history)


def run_dos_weighted_energy(
    model,
    setup,
    energy,
    d0,
    alpha,
    max_iterations,
):
    validate_energy(energy)
    validate_solver_settings(d0, alpha, max_iterations)
    partition_count = len(setup.partitions)
    sigma_by_partition = np.zeros(partition_count, dtype=complex)
    previous_green_sums = np.zeros(partition_count, dtype=complex)
    previous_dos = None
    previous_t_eff = None
    history = []

    for iteration in range(1, max_iterations + 1):
        green_function = solve_green_function(
            model,
            setup,
            energy,
            sigma_by_partition,
        )
        green_sums, partition_ldos, dos = calculate_partition_values(
            green_function,
            setup,
        )
        gamma_by_partition = calculate_probe_gammas(
            sigma_by_partition,
            setup,
        )
        t_lr, t_eff, active_partition_ids = calculate_transport(
            green_function,
            setup,
            gamma_by_partition,
        )
        dos_change = None
        transmission_change = None

        if previous_dos is not None:
            dos_change = calculate_percent_change(dos, previous_dos)
            transmission_change = calculate_percent_change(
                t_eff,
                previous_t_eff,
            )

        record = build_iteration_record(
            iteration,
            dos,
            t_lr,
            t_eff,
            green_sums,
            partition_ldos,
            sigma_by_partition,
            gamma_by_partition,
            active_partition_ids,
            dos_change,
            transmission_change,
        )
        history.append(record)

        if dos_change is not None and transmission_change is not None:
            if dos_change < CONVERGENCE_PERCENT:
                if transmission_change < CONVERGENCE_PERCENT:
                    del green_function
                    return build_energy_result(energy, True, history)

        if iteration == max_iterations:
            del green_function
            result = build_energy_result(energy, False, history)
            raise ConvergenceError(
                f"solver did not converge within {max_iterations} iterations",
                result,
            )

        next_self_energy = calculate_next_self_energy(
            green_sums,
            previous_green_sums,
            setup,
            d0,
            alpha,
        )
        previous_green_sums = green_sums.copy()
        previous_dos = dos
        previous_t_eff = t_eff
        sigma_by_partition = next_self_energy
        del green_function

    result = build_energy_result(energy, False, history)
    raise ConvergenceError("solver did not converge", result)

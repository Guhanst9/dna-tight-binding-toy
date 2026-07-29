import numpy as np

from src.greens import solve_green_function
from .partition_transport import calculate_partition_transport


def validate_solver_settings(d0, tolerance, max_iterations, alpha):
    if d0 < 0:
        raise ValueError("d0 must be nonnegative")

    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")

    if max_iterations < 1:
        raise ValueError("max iterations must be at least 1")

    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1")


def validate_energy_grid(energies):
    if len(energies) < 3:
        raise ValueError("energy grid must contain at least three points")

    spacings = np.diff(energies)

    for spacing in spacings:
        if spacing <= 0:
            raise ValueError("energy grid must be strictly increasing")

    if not np.allclose(spacings, spacings[0]):
        raise ValueError("energy grid must have a constant step")


def solve_green_grid(hamiltonian, contact_setup, energies, sigma_decoherence):
    energy_count = len(energies)
    size = hamiltonian.shape[0]
    green_functions = np.zeros((energy_count, size, size), dtype=complex)

    for energy_index in range(energy_count):
        green_functions[energy_index] = solve_green_function(
            hamiltonian=hamiltonian,
            energy=energies[energy_index],
            sigma_left=contact_setup["sigma_left"],
            sigma_right=contact_setup["sigma_right"],
            sigma_decoherence=sigma_decoherence[energy_index],
        )

    return green_functions


def calculate_partition_ldos(green_functions, probe_partitions):
    energy_count = green_functions.shape[0]
    partition_count = len(probe_partitions)
    ldos = np.zeros((energy_count, partition_count))

    for energy_index in range(energy_count):
        for partition_index in range(partition_count):
            imaginary_sum = 0.0
            partition = probe_partitions[partition_index]

            for orbital_index in partition.orbital_indices:
                imaginary_sum = imaginary_sum + np.imag(
                    green_functions[energy_index, orbital_index, orbital_index]
                )

            ldos[energy_index, partition_index] = -imaginary_sum / np.pi

    return ldos


def calculate_total_dos(green_functions):
    energy_count = green_functions.shape[0]
    dos = np.zeros(energy_count)

    for energy_index in range(energy_count):
        diagonal = np.diag(green_functions[energy_index])
        dos[energy_index] = float(np.sum(-np.imag(diagonal) / np.pi))

    return dos


def calculate_partition_self_energy_values(sigma_decoherence, probe_partitions):
    energy_count = sigma_decoherence.shape[0]
    partition_count = len(probe_partitions)
    real_values = np.zeros((energy_count, partition_count))
    imaginary_values = np.zeros((energy_count, partition_count))

    for energy_index in range(energy_count):
        for partition_index in range(partition_count):
            partition = probe_partitions[partition_index]
            first_index = partition.orbital_indices[0]
            value = sigma_decoherence[energy_index, first_index, first_index]
            real_values[energy_index, partition_index] = np.real(value)
            imaginary_values[energy_index, partition_index] = np.imag(value)

    return real_values, imaginary_values


def calculate_imaginary_candidates(partition_ldos, d0):
    return -d0 * partition_ldos / 2


def mix_imaginary_self_energy(candidate, previous_candidate, alpha):
    return alpha * candidate + (1 - alpha) * previous_candidate


def build_energy_cell_edges(energies):
    energy_count = len(energies)
    edges = np.zeros(energy_count + 1)
    first_spacing = energies[1] - energies[0]
    last_spacing = energies[-1] - energies[-2]
    edges[0] = energies[0] - first_spacing / 2
    edges[-1] = energies[-1] + last_spacing / 2

    for index in range(1, energy_count):
        edges[index] = (energies[index - 1] + energies[index]) / 2

    return edges


def build_kramers_kronig_operator(energies):
    # principal-value quadrature over uniform energy cells.
    energy_count = len(energies)
    cell_edges = build_energy_cell_edges(energies)
    operator = np.zeros((energy_count, energy_count))

    for target_index in range(energy_count):
        target_energy = energies[target_index]
        upper_differences = cell_edges[1:] - target_energy
        lower_differences = cell_edges[:-1] - target_energy
        weights = np.log(abs(upper_differences / lower_differences))
        weights[target_index] = 0.0
        operator[target_index] = weights / np.pi

    return operator


def calculate_real_self_energy(kramers_kronig_operator, imaginary_self_energy):
    real_self_energy = np.matmul(kramers_kronig_operator, imaginary_self_energy)

    return real_self_energy


def build_decoherence_self_energy_grid(
    size,
    probe_partitions,
    real_self_energy,
    imaginary_self_energy,
):
    energy_count = real_self_energy.shape[0]
    sigma_decoherence = np.zeros((energy_count, size, size), dtype=complex)

    for energy_index in range(energy_count):
        for partition_index in range(len(probe_partitions)):
            value = (
                real_self_energy[energy_index, partition_index]
                + 1j * imaginary_self_energy[energy_index, partition_index]
            )
            partition = probe_partitions[partition_index]

            for orbital_index in partition.orbital_indices:
                sigma_decoherence[energy_index, orbital_index, orbital_index] = value

    return sigma_decoherence


def calculate_transport_grid(
    green_functions,
    contact_setup,
    sigma_decoherence,
):
    energy_count = green_functions.shape[0]
    direct_transmission = np.zeros(energy_count)
    effective_transmission = np.zeros(energy_count)

    for energy_index in range(energy_count):
        transport = calculate_partition_transport(
            green_function=green_functions[energy_index],
            sigma_left=contact_setup["sigma_left"],
            sigma_right=contact_setup["sigma_right"],
            sigma_decoherence=sigma_decoherence[energy_index],
            probe_partitions=contact_setup["probe_partitions"],
        )
        direct_transmission[energy_index] = transport["t_lr"]
        effective_transmission[energy_index] = transport["t_eff"]

    return direct_transmission, effective_transmission


def calculate_percent_changes(current_values, previous_values):
    changes = np.zeros(len(current_values))

    for index in range(len(current_values)):
        previous_value = previous_values[index]
        current_value = current_values[index]

        if previous_value == 0:
            if current_value == 0:
                changes[index] = 0.0
            else:
                changes[index] = np.inf
        else:
            changes[index] = (
                abs(current_value - previous_value)
                / abs(previous_value)
                * 100
            )

    return changes


def has_converged(dos_changes, transmission_changes, tolerance):
    if not np.all(np.isfinite(dos_changes)):
        return False

    if not np.all(np.isfinite(transmission_changes)):
        return False

    if np.any(dos_changes >= tolerance):
        return False

    if np.any(transmission_changes >= tolerance):
        return False

    return True


def build_history_row(
    iteration,
    dos,
    direct_transmission,
    effective_transmission,
    gamma_values,
    probe_self_energy_real,
    probe_self_energy_imaginary,
    dos_changes,
    transmission_changes,
):
    return {
        "iteration": iteration,
        "dos": dos.copy(),
        "t_lr": direct_transmission.copy(),
        "t_eff": effective_transmission.copy(),
        "probe_gammas": gamma_values.copy(),
        "probe_self_energy_real": probe_self_energy_real.copy(),
        "probe_self_energy_imaginary": probe_self_energy_imaginary.copy(),
        "dos_changes": None if dos_changes is None else dos_changes.copy(),
        "transmission_changes": (
            None if transmission_changes is None else transmission_changes.copy()
        ),
    }


def package_results(
    converged,
    iteration,
    energies,
    green_functions,
    partition_ldos,
    dos,
    direct_transmission,
    effective_transmission,
    sigma_decoherence,
    gamma_values,
    dos_changes,
    transmission_changes,
    history,
):
    return {
        "converged": converged,
        "iterations": iteration,
        "energies": energies,
        "green_functions": green_functions,
        "partition_ldos": partition_ldos,
        "dos": dos,
        "t_lr": direct_transmission,
        "t_eff": effective_transmission,
        "sigma_decoherence": sigma_decoherence,
        "probe_gammas": gamma_values,
        "dos_changes": dos_changes,
        "transmission_changes": transmission_changes,
        "history": history,
    }


def run_dos_weighted_solver(
    hamiltonian,
    contact_setup,
    energies,
    d0,
    tolerance,
    max_iterations,
    alpha,
):
    validate_solver_settings(d0, tolerance, max_iterations, alpha)
    validate_energy_grid(energies)

    size = hamiltonian.shape[0]
    energy_count = len(energies)
    partition_count = len(contact_setup["probe_partitions"])
    sigma_current = np.zeros((energy_count, size, size), dtype=complex)
    kramers_kronig_operator = build_kramers_kronig_operator(energies)
    previous_candidate_imaginary = np.zeros((energy_count, partition_count))
    previous_dos = None
    previous_t_eff = None
    history = []
    final_results = None

    for iteration in range(1, max_iterations + 1):
        green_functions = solve_green_grid(
            hamiltonian,
            contact_setup,
            energies,
            sigma_current,
        )
        partition_ldos = calculate_partition_ldos(
            green_functions,
            contact_setup["probe_partitions"],
        )
        dos = calculate_total_dos(green_functions)
        direct_transmission, effective_transmission = calculate_transport_grid(
            green_functions,
            contact_setup,
            sigma_current,
        )
        gamma_values = -2 * np.imag(
            np.diagonal(sigma_current, axis1=1, axis2=2)
        )
        partition_self_energy_real, partition_self_energy_imaginary = (
            calculate_partition_self_energy_values(
                sigma_current,
                contact_setup["probe_partitions"],
            )
        )
        partition_gammas = np.zeros((energy_count, partition_count))

        for energy_index in range(energy_count):
            for partition_index in range(partition_count):
                partition = contact_setup["probe_partitions"][partition_index]
                first_index = partition.orbital_indices[0]
                partition_gammas[energy_index, partition_index] = gamma_values[
                    energy_index,
                    first_index
                ]

        dos_changes = None
        transmission_changes = None

        if previous_dos is not None:
            dos_changes = calculate_percent_changes(dos, previous_dos)
            transmission_changes = calculate_percent_changes(
                effective_transmission,
                previous_t_eff,
            )

        history.append(
            build_history_row(
                iteration,
                dos,
                direct_transmission,
            effective_transmission,
            partition_gammas,
            partition_self_energy_real,
            partition_self_energy_imaginary,
            dos_changes,
                transmission_changes,
            )
        )

        if previous_dos is not None:
            if has_converged(dos_changes, transmission_changes, tolerance):
                return package_results(
                    True,
                    iteration,
                    energies,
                    green_functions,
                    partition_ldos,
                    dos,
                    direct_transmission,
                    effective_transmission,
                    sigma_current,
                    partition_gammas,
                    dos_changes,
                    transmission_changes,
                    history,
                )

        final_results = package_results(
            False,
            iteration,
            energies,
            green_functions,
            partition_ldos,
            dos,
            direct_transmission,
            effective_transmission,
            sigma_current,
            partition_gammas,
            dos_changes,
            transmission_changes,
            history,
        )
        candidate_imaginary = calculate_imaginary_candidates(partition_ldos, d0)
        mixed_imaginary = mix_imaginary_self_energy(
            candidate_imaginary,
            previous_candidate_imaginary,
            alpha,
        )
        real_self_energy = calculate_real_self_energy(
            kramers_kronig_operator,
            mixed_imaginary,
        )
        sigma_current = build_decoherence_self_energy_grid(
            size,
            contact_setup["probe_partitions"],
            real_self_energy,
            mixed_imaginary,
        )
        previous_candidate_imaginary = candidate_imaginary
        previous_dos = dos
        previous_t_eff = effective_transmission

    return final_results

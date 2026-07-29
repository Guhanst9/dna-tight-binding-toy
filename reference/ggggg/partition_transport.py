import numpy as np

minimum_probe_gamma = 1e-14


def calculate_partition_gamma(sigma_decoherence, partition):
    gamma_values = []

    for index in partition.orbital_indices:
        gamma_value = -2 * np.imag(sigma_decoherence[index, index])
        gamma_values.append(gamma_value)

    first_gamma = gamma_values[0]

    for gamma_value in gamma_values:
        if not np.isclose(gamma_value, first_gamma, atol=1e-12):
            raise ValueError("orbitals in one partition must have the same probe gamma")

    if first_gamma < 0 and abs(first_gamma) < 1e-12:
        first_gamma = 0.0

    if first_gamma < 0:
        raise ValueError("probe gamma must be nonnegative")

    return float(first_gamma)


def calculate_region_gamma(self_energy, indices):
    gamma_values = []

    for index in indices:
        gamma_values.append(-2 * np.imag(self_energy[index, index]))

    first_gamma = gamma_values[0]

    for gamma_value in gamma_values:
        if not np.isclose(gamma_value, first_gamma, atol=1e-12):
            raise ValueError("all orbitals in one region must have the same gamma")

    return float(first_gamma)


def calculate_region_transmission(
    green_function,
    left_indices,
    left_gamma,
    right_indices,
    right_gamma,
):
    green_block = green_function[np.ix_(left_indices, right_indices)]
    transmission = left_gamma * right_gamma * np.sum(abs(green_block) ** 2)
    return float(np.real(transmission))


def get_active_partition_data(sigma_decoherence, probe_partitions):
    active_partitions = []
    active_gammas = []

    for partition in probe_partitions:
        gamma_value = calculate_partition_gamma(sigma_decoherence, partition)

        if gamma_value > minimum_probe_gamma:
            active_partitions.append(partition)
            active_gammas.append(gamma_value)

    return active_partitions, active_gammas


def calculate_probe_transmissions(
    green_function,
    left_indices,
    gamma_left,
    right_indices,
    gamma_right,
    active_partitions,
    active_gammas,
):
    probe_count = len(active_partitions)
    left_to_probe = np.zeros(probe_count)
    probe_to_right = np.zeros(probe_count)
    probe_to_probe = np.zeros((probe_count, probe_count))

    for position in range(probe_count):
        partition = active_partitions[position]
        gamma_probe = active_gammas[position]
        left_to_probe[position] = calculate_region_transmission(
            green_function,
            left_indices,
            gamma_left,
            partition.orbital_indices,
            gamma_probe,
        )
        probe_to_right[position] = calculate_region_transmission(
            green_function,
            partition.orbital_indices,
            gamma_probe,
            right_indices,
            gamma_right,
        )

    for row in range(probe_count):
        for column in range(probe_count):
            if row != column:
                left_partition = active_partitions[row]
                right_partition = active_partitions[column]
                probe_to_probe[row, column] = calculate_region_transmission(
                    green_function,
                    left_partition.orbital_indices,
                    active_gammas[row],
                    right_partition.orbital_indices,
                    active_gammas[column],
                )

    return {
        "left_to_probe": left_to_probe,
        "probe_to_right": probe_to_right,
        "probe_to_probe": probe_to_probe,
    }


def build_w_matrix(left_to_probe, probe_to_right, probe_to_probe):
    probe_count = len(left_to_probe)
    w_matrix = np.zeros((probe_count, probe_count))

    for row in range(probe_count):
        diagonal_value = left_to_probe[row] + probe_to_right[row]

        for column in range(probe_count):
            if row != column:
                diagonal_value = diagonal_value + probe_to_probe[row, column]
                w_matrix[row, column] = -probe_to_probe[row, column]

        w_matrix[row, row] = diagonal_value

    return w_matrix


def calculate_effective_transmission(t_lr, left_to_probe, w_matrix, probe_to_right):
    if len(left_to_probe) == 0:
        return float(t_lr)

    try:
        probe_potentials = np.linalg.solve(w_matrix, probe_to_right)
    except np.linalg.LinAlgError:
        raise ValueError("probe transmission matrix could not be inverted")

    probe_term = np.dot(left_to_probe, probe_potentials)
    return float(np.real(t_lr + probe_term))


def calculate_partition_transport(
    green_function,
    sigma_left,
    sigma_right,
    sigma_decoherence,
    probe_partitions,
):
    left_indices = []
    right_indices = []

    for index in range(len(sigma_left)):
        if abs(sigma_left[index, index]) > 0:
            left_indices.append(index)

        if abs(sigma_right[index, index]) > 0:
            right_indices.append(index)

    gamma_left = calculate_region_gamma(sigma_left, left_indices)
    gamma_right = calculate_region_gamma(sigma_right, right_indices)
    t_lr = calculate_region_transmission(
        green_function,
        left_indices,
        gamma_left,
        right_indices,
        gamma_right,
    )
    active_partitions, active_gammas = get_active_partition_data(
        sigma_decoherence,
        probe_partitions,
    )
    probe_transmissions = calculate_probe_transmissions(
        green_function,
        left_indices,
        gamma_left,
        right_indices,
        gamma_right,
        active_partitions,
        active_gammas,
    )
    w_matrix = build_w_matrix(
        probe_transmissions["left_to_probe"],
        probe_transmissions["probe_to_right"],
        probe_transmissions["probe_to_probe"],
    )
    t_eff = calculate_effective_transmission(
        t_lr,
        probe_transmissions["left_to_probe"],
        w_matrix,
        probe_transmissions["probe_to_right"],
    )

    return {
        "t_lr": t_lr,
        "t_eff": t_eff,
        "active_partitions": active_partitions,
        "active_gammas": active_gammas,
        "left_to_probe": probe_transmissions["left_to_probe"],
        "probe_to_right": probe_transmissions["probe_to_right"],
        "probe_to_probe": probe_transmissions["probe_to_probe"],
        "w_matrix": w_matrix,
    }

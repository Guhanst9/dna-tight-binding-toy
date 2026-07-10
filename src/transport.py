import numpy as np

from .greens import calculate_broadening, calculate_direct_transmission

minimum_probe_gamma = 1e-14

def build_empty_decoherence_self_energy(size):
    return np.zeros((size, size), dtype=complex)

def calculate_probe_gamma_values(sigma_decoherence, probe_indices):
    gamma_values = []

    for index in probe_indices:
        gamma_value = -2 * np.imag(sigma_decoherence[index, index])

        if gamma_value < 0 and abs(gamma_value) < 1e-12:
            gamma_value = 0.0

        if gamma_value < 0:
            raise ValueError("probe gamma must be nonnegative")

        gamma_values.append(float(gamma_value))

    return gamma_values

def get_active_probes(probe_indices, probe_gamma_values):
    active_probe_indices = []
    active_probe_gammas = []

    for position in range(len(probe_indices)):
        gamma_value = probe_gamma_values[position]

        if gamma_value > minimum_probe_gamma:
            active_probe_indices.append(probe_indices[position])
            active_probe_gammas.append(gamma_value)

    return active_probe_indices, active_probe_gammas

def build_probe_gamma_matrix(size, probe_index, gamma_value):
    gamma_matrix = np.zeros((size, size), dtype=complex)
    gamma_matrix[probe_index, probe_index] = gamma_value
    return gamma_matrix

def build_probe_gamma_matrices(size, active_probe_indices, active_probe_gammas):
    probe_gamma_matrices = []

    for position in range(len(active_probe_indices)):
        probe_index = active_probe_indices[position]
        gamma_value = active_probe_gammas[position]
        gamma_matrix = build_probe_gamma_matrix(size, probe_index, gamma_value)
        probe_gamma_matrices.append(gamma_matrix)

    return probe_gamma_matrices

def calculate_probe_transmissions(
    green_function,
    gamma_left,
    gamma_right,
    probe_gamma_matrices,
):
    probe_count = len(probe_gamma_matrices)
    left_to_probe = np.zeros(probe_count)
    probe_to_left = np.zeros(probe_count)
    probe_to_right = np.zeros(probe_count)
    probe_to_probe = np.zeros((probe_count, probe_count))

    for position in range(probe_count):
        gamma_probe = probe_gamma_matrices[position]

        left_to_probe[position] = calculate_direct_transmission(
            green_function,
            gamma_left,
            gamma_probe,
        )
        probe_to_left[position] = calculate_direct_transmission(
            green_function,
            gamma_probe,
            gamma_left,
        )
        probe_to_right[position] = calculate_direct_transmission(
            green_function,
            gamma_probe,
            gamma_right,
        )

    for row in range(probe_count):
        for column in range(probe_count):
            if row != column:
                probe_to_probe[row, column] = calculate_direct_transmission(
                    green_function,
                    probe_gamma_matrices[row],
                    probe_gamma_matrices[column],
                )

    return {
        "left_to_probe": left_to_probe,
        "probe_to_left": probe_to_left,
        "probe_to_right": probe_to_right,
        "probe_to_probe": probe_to_probe,
    }

def build_w_matrix(probe_to_left, probe_to_right, probe_to_probe):
    probe_count = len(probe_to_left)
    w_matrix = np.zeros((probe_count, probe_count))

    for row in range(probe_count):
        diagonal_value = probe_to_left[row] + probe_to_right[row]

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
        w_inverse = np.linalg.inv(w_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("probe transmission matrix could not be inverted")

    product = np.matmul(left_to_probe, w_inverse)
    probe_term = np.matmul(product, probe_to_right)
    t_eff = t_lr + probe_term

    return float(np.real(t_eff))

def run_transport_calculation(green_function, contact_setup, sigma_decoherence):
    size = green_function.shape[0]
    gamma_left = calculate_broadening(contact_setup["sigma_left"])
    gamma_right = calculate_broadening(contact_setup["sigma_right"])
    t_lr = calculate_direct_transmission(green_function, gamma_left, gamma_right)

    probe_gamma_values = calculate_probe_gamma_values(
        sigma_decoherence,
        contact_setup["probe_indices"],
    )
    active_probe_indices, active_probe_gammas = get_active_probes(
        contact_setup["probe_indices"],
        probe_gamma_values,
    )
    probe_gamma_matrices = build_probe_gamma_matrices(
        size,
        active_probe_indices,
        active_probe_gammas,
    )
    probe_transmissions = calculate_probe_transmissions(
        green_function,
        gamma_left,
        gamma_right,
        probe_gamma_matrices,
    )
    w_matrix = build_w_matrix(
        probe_transmissions["probe_to_left"],
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
        "probe_gammas": probe_gamma_values,
        "active_probe_indices": active_probe_indices,
        "active_probe_gammas": active_probe_gammas,
        "left_to_probe": probe_transmissions["left_to_probe"],
        "probe_to_right": probe_transmissions["probe_to_right"],
        "probe_to_probe": probe_transmissions["probe_to_probe"],
        "w_matrix": w_matrix,
    }

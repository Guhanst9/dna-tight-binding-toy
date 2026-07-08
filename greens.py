import numpy as np

def solve_green_function(hamiltonian, energy, sigma_left, sigma_right):
    size = hamiltonian.shape[0]
    identity = np.eye(size, dtype=complex)

    matrix_to_invert = (
        energy * identity
        - hamiltonian
        - sigma_left
        - sigma_right
    )

    try:
        green_function = np.linalg.inv(matrix_to_invert)
    except np.linalg.LinAlgError:
        raise ValueError("green's function matrix could not be inverted")

    return green_function

def calculate_ldos(green_function):
    diagonal = np.diag(green_function)
    ldos = -np.imag(diagonal) / np.pi
    return ldos

def calculate_dos(ldos):
    return float(np.sum(ldos))

def calculate_broadening(self_energy):
    return 1j * (self_energy - self_energy.conj().T)

def calculate_direct_transmission(green_function, gamma_left, gamma_right):
    advanced_green_function = green_function.conj().T

    product = np.matmul(gamma_left, green_function)
    product = np.matmul(product, gamma_right)
    product = np.matmul(product, advanced_green_function)

    return float(np.real(np.trace(product)))

def run_coherent_calculation(hamiltonian, contact_setup, energy):
    green_function = solve_green_function(
        hamiltonian=hamiltonian,
        energy=energy,
        sigma_left=contact_setup["sigma_left"],
        sigma_right=contact_setup["sigma_right"],
    )

    ldos = calculate_ldos(green_function)
    dos = calculate_dos(ldos)
    gamma_left = calculate_broadening(contact_setup["sigma_left"])
    gamma_right = calculate_broadening(contact_setup["sigma_right"])
    transmission = calculate_direct_transmission(
        green_function=green_function,
        gamma_left=gamma_left,
        gamma_right=gamma_right,
    )

    return {
        "green_function": green_function,
        "ldos": ldos,
        "dos": dos,
        "t_lr": transmission,
    }

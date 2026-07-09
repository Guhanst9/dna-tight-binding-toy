import numpy as np

from greens import calculate_dos, calculate_ldos, solve_green_function
from transport import (
    build_empty_decoherence_self_energy,
    run_transport_calculation,
)

def validate_solver_settings(d0, tolerance, max_iterations, alpha):
    if d0 < 0:
        raise ValueError("d0 must be nonnegative")

    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")

    if max_iterations < 1:
        raise ValueError("max iterations must be at least 1")

    if alpha < 0:
        raise ValueError("alpha must be between 0 and 1")

    if alpha > 1:
        raise ValueError("alpha must be between 0 and 1")

def build_decoherence_self_energy(green_function, probe_indices, d0):
    size = green_function.shape[0]
    sigma_decoherence = build_empty_decoherence_self_energy(size)

    for index in probe_indices:
        sigma_decoherence[index, index] = d0 / (2 * np.pi) * green_function[index, index]

    return sigma_decoherence

def mix_self_energy(candidate, previous_candidate, alpha):
    return alpha * candidate + (1 - alpha) * previous_candidate

def calculate_percent_change(current, previous):
    scale = abs(previous)

    if abs(current) > scale:
        scale = abs(current)

    if scale < 1e-30:
        return 0.0

    return abs(current - previous) / scale * 100

def is_converged(dos_change, transmission_change, tolerance):
    if dos_change > tolerance:
        return False

    if transmission_change > tolerance:
        return False

    return True

def package_solver_results(
    converged,
    iteration,
    green_function,
    ldos,
    dos,
    sigma_decoherence,
    transport_results,
    dos_change,
    transmission_change,
):
    return {
        "converged": converged,
        "iterations": iteration,
        "green_function": green_function,
        "ldos": ldos,
        "dos": dos,
        "sigma_decoherence": sigma_decoherence,
        "transport_results": transport_results,
        "t_lr": transport_results["t_lr"],
        "t_eff": transport_results["t_eff"],
        "dos_change_percent": dos_change,
        "transmission_change_percent": transmission_change,
    }

def run_self_consistent_solver(
    hamiltonian,
    contact_setup,
    energy,
    d0,
    tolerance,
    max_iterations,
    alpha,
):
    validate_solver_settings(d0, tolerance, max_iterations, alpha)

    size = hamiltonian.shape[0]
    sigma_current = build_empty_decoherence_self_energy(size)
    previous_candidate_sigma = build_empty_decoherence_self_energy(size)
    previous_dos = None
    previous_t_eff = None
    final_results = None

    for iteration in range(1, max_iterations + 1):
        green_function = solve_green_function(
            hamiltonian=hamiltonian,
            energy=energy,
            sigma_left=contact_setup["sigma_left"],
            sigma_right=contact_setup["sigma_right"],
            sigma_decoherence=sigma_current,
        )
        ldos = calculate_ldos(green_function)
        dos = calculate_dos(ldos)
        transport_results = run_transport_calculation(
            green_function=green_function,
            contact_setup=contact_setup,
            sigma_decoherence=sigma_current,
        )

        dos_change = None
        transmission_change = None

        if previous_dos is not None:
            dos_change = calculate_percent_change(dos, previous_dos)
            transmission_change = calculate_percent_change(
                transport_results["t_eff"],
                previous_t_eff,
            )

            if is_converged(dos_change, transmission_change, tolerance):
                return package_solver_results(
                    True,
                    iteration,
                    green_function,
                    ldos,
                    dos,
                    sigma_current,
                    transport_results,
                    dos_change,
                    transmission_change,
                )

        final_results = package_solver_results(
            False,
            iteration,
            green_function,
            ldos,
            dos,
            sigma_current,
            transport_results,
            dos_change,
            transmission_change,
        )
        candidate_sigma = build_decoherence_self_energy(
            green_function,
            contact_setup["probe_indices"],
            d0,
        )
        sigma_next = mix_self_energy(candidate_sigma, previous_candidate_sigma, alpha)

        previous_dos = dos
        previous_t_eff = transport_results["t_eff"]
        previous_candidate_sigma = candidate_sigma
        sigma_current = sigma_next

    return final_results

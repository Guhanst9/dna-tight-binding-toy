import numpy as np

from src.greens import run_coherent_calculation
from src.solver import run_self_consistent_solver

def calculate_eigenvalues(hamiltonian):
    eigenvalues = np.linalg.eigvalsh(hamiltonian)
    eigenvalues = np.sort(eigenvalues)
    return eigenvalues

def validate_energy_grid(energy_min, energy_max, energy_points):
    if not np.isfinite(energy_min):
        raise ValueError("energy min must be finite")

    if not np.isfinite(energy_max):
        raise ValueError("energy max must be finite")

    if energy_min >= energy_max:
        raise ValueError("energy min must be less than energy max")

    if energy_points < 2:
        raise ValueError("energy points must be at least 2")

def validate_trace_energy(trace_energy):
    if not np.isfinite(trace_energy):
        raise ValueError("trace energy must be finite")

def build_energy_grid(energy_min, energy_max, energy_points):
    validate_energy_grid(energy_min, energy_max, energy_points)
    return np.linspace(energy_min, energy_max, energy_points)

def run_coherent_sweep(hamiltonian, contact_setup, energy_min, energy_max, energy_points):
    energies = build_energy_grid(energy_min, energy_max, energy_points)
    rows = []

    for energy in energies:
        coherent_results = run_coherent_calculation(
            hamiltonian=hamiltonian,
            contact_setup=contact_setup,
            energy=energy,
        )
        row = {
            "energy": float(energy),
            "coherent_t_lr": coherent_results["t_lr"],
        }
        rows.append(row)

    return rows

def run_transmission_sweep(
    hamiltonian,
    contact_setup,
    energy_min,
    energy_max,
    energy_points,
    d0,
    tolerance,
    max_iterations,
    alpha,
):
    energies = build_energy_grid(energy_min, energy_max, energy_points)
    rows = []

    for energy in energies:
        coherent_results = run_coherent_calculation(
            hamiltonian=hamiltonian,
            contact_setup=contact_setup,
            energy=energy,
        )
        solver_results = run_self_consistent_solver(
            hamiltonian=hamiltonian,
            contact_setup=contact_setup,
            energy=energy,
            d0=d0,
            tolerance=tolerance,
            max_iterations=max_iterations,
            alpha=alpha,
        )

        row = {
            "energy": float(energy),
            "coherent_t_lr": coherent_results["t_lr"],
            "decoherent_t_eff": solver_results["t_eff"],
            "converged": solver_results["converged"],
            "iterations": solver_results["iterations"],
        }
        rows.append(row)

    return rows

def get_orbital_by_index(orbitals, index):
    for orbital in orbitals:
        if orbital.index == index:
            return orbital

    raise ValueError("orbital index was not found")

def build_probe_gamma_rows(orbitals, probe_indices, probe_gammas):
    rows = []

    for position in range(len(probe_indices)):
        probe_index = probe_indices[position]
        gamma = probe_gammas[position]
        orbital = get_orbital_by_index(orbitals, probe_index)
        row = {
            "probe_index": probe_index,
            "base_pair": orbital.base_pair,
            "strand": orbital.strand,
            "base": orbital.base,
            "gamma_b": gamma,
        }
        rows.append(row)

    return rows

def get_converged_probe_gammas(orbitals, contact_setup, solver_results):
    probe_indices = contact_setup["probe_indices"]
    probe_gammas = solver_results["transport_results"]["probe_gammas"]
    return build_probe_gamma_rows(orbitals, probe_indices, probe_gammas)

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.contacts import build_contact_self_energy
from src.greens import calculate_broadening, solve_green_function
from src.parameters import (
    get_nearest_cross_hopping,
    get_onsite,
    get_strand_hopping,
)
from src.transport import run_transport_calculation

complements = {
    "G": "C",
    "C": "G",
    "A": "T",
    "T": "A",
}

@dataclass(frozen=True)
class FullOrbital:
    index: int
    base_pair: int
    strand: int
    base: str
    band: str

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="GGGGG")
    parser.add_argument("--gamma-left", type=float, default=1.0)
    parser.add_argument("--gamma-right", type=float, default=1.0)
    parser.add_argument("--probe-gamma", type=float, default=0.01)
    parser.add_argument("--energy-min", type=float, default=-10.0)
    parser.add_argument("--energy-max", type=float, default=10.0)
    parser.add_argument("--energy-step", type=float, default=0.01)
    parser.add_argument("--output-dir", default="outputs/sequence_ggggg_20_orbital_fixed_probe")
    return parser.parse_args()

def validate_args(args):
    sequence = args.sequence.upper()

    if len(sequence) < 2:
        raise ValueError("sequence must have at least two base pairs")

    for base in sequence:
        if base not in complements:
            raise ValueError("sequence can only contain A, T, G, and C")

    if sequence != "GGGGG":
        raise ValueError("only GGGGG is supported for this test")

    if args.gamma_left < 0:
        raise ValueError("gamma left must be nonnegative")

    if args.gamma_right < 0:
        raise ValueError("gamma right must be nonnegative")

    if args.probe_gamma < 0:
        raise ValueError("probe gamma must be nonnegative")

    if args.energy_min >= args.energy_max:
        raise ValueError("energy min must be less than energy max")

    if args.energy_step <= 0:
        raise ValueError("energy step must be positive")

    if args.energy_step > 0.01:
        raise ValueError("energy step must be 0.01 eV or smaller")

def build_full_basis(sequence):
    orbitals = []
    bands = ["homo", "lumo"]

    for band in bands:
        for base_pair_index in range(len(sequence)):
            first_base = sequence[base_pair_index]
            second_base = complements[first_base]
            bases = [
                (1, first_base),
                (2, second_base),
            ]

            for strand, base in bases:
                orbital = FullOrbital(
                    index=len(orbitals),
                    base_pair=base_pair_index + 1,
                    strand=strand,
                    base=base,
                    band=band,
                )
                orbitals.append(orbital)

    return orbitals

def build_full_hamiltonian(sequence):
    orbitals = build_full_basis(sequence)
    size = len(orbitals)
    hamiltonian = np.zeros((size, size))

    for orbital in orbitals:
        hamiltonian[orbital.index, orbital.index] = get_onsite(
            orbital.base,
            orbital.band,
        )

    add_same_strand_hopping(hamiltonian, orbitals)
    add_same_base_pair_hopping(hamiltonian, orbitals)
    check_hermitian(hamiltonian)

    return hamiltonian, orbitals

def add_same_strand_hopping(hamiltonian, orbitals):
    for left in orbitals:
        for right in orbitals:
            if right.base_pair != left.base_pair + 1:
                continue

            if right.strand != left.strand:
                continue

            if right.band != left.band:
                continue

            value = get_strand_hopping(left.base, left.band)
            hamiltonian[left.index, right.index] = value
            hamiltonian[right.index, left.index] = value

def add_same_base_pair_hopping(hamiltonian, orbitals):
    for left in orbitals:
        for right in orbitals:
            if right.base_pair != left.base_pair:
                continue

            if left.strand != 1 or right.strand != 2:
                continue

            if right.band != left.band:
                continue

            pair = (left.base + right.base).lower()
            value = get_nearest_cross_hopping(pair, left.band)
            hamiltonian[left.index, right.index] = value
            hamiltonian[right.index, left.index] = value

def check_hermitian(matrix):
    if not np.allclose(matrix, matrix.conj().T):
        raise ValueError("hamiltonian is not hermitian")

def build_terminal_contact_setup(orbitals, gamma_left, gamma_right):
    size = len(orbitals)
    last_base_pair = orbitals[-1].base_pair
    left_contact = []
    right_contact = []
    probe_indices = []

    for orbital in orbitals:
        if orbital.base_pair == 1:
            left_contact.append(orbital.index)

        if orbital.base_pair == last_base_pair:
            right_contact.append(orbital.index)

        if orbital.base_pair != 1:
            if orbital.base_pair != last_base_pair:
                probe_indices.append(orbital.index)

    return {
        "left_contact": left_contact,
        "right_contact": right_contact,
        "probe_indices": probe_indices,
        "sigma_left": build_contact_self_energy(size, left_contact, gamma_left),
        "sigma_right": build_contact_self_energy(size, right_contact, gamma_right),
    }

def build_fixed_probe_self_energy(size, probe_indices, probe_gamma):
    sigma_probe = np.zeros((size, size), dtype=complex)

    for index in probe_indices:
        sigma_probe[index, index] = -1j * probe_gamma / 2

    return sigma_probe

def build_energy_grid(energy_min, energy_max, energy_step):
    point_count = int(round((energy_max - energy_min) / energy_step)) + 1
    energies = []

    for point_index in range(point_count):
        energy = energy_min + point_index * energy_step
        energies.append(float(energy))

    if energies[-1] < energy_max:
        energies.append(float(energy_max))

    return energies

def calculate_direct_transmission_complex(green_function, gamma_left, gamma_right):
    advanced_green_function = green_function.conj().T
    product = np.matmul(gamma_left, green_function)
    product = np.matmul(product, gamma_right)
    product = np.matmul(product, advanced_green_function)
    return np.trace(product)

def check_transmission_is_real(value):
    if abs(np.imag(value)) > 1e-10:
        raise ValueError("transmission has a nonzero imaginary part")

def run_coherent_sweep(hamiltonian, contact_setup, energies):
    gamma_left = calculate_broadening(contact_setup["sigma_left"])
    gamma_right = calculate_broadening(contact_setup["sigma_right"])
    rows = []

    for energy in energies:
        green_function = solve_green_function(
            hamiltonian=hamiltonian,
            energy=energy,
            sigma_left=contact_setup["sigma_left"],
            sigma_right=contact_setup["sigma_right"],
        )
        transmission = calculate_direct_transmission_complex(
            green_function,
            gamma_left,
            gamma_right,
        )
        check_transmission_is_real(transmission)
        row = {
            "energy_ev": energy,
            "coherent_transmission": float(np.real(transmission)),
            "coherent_transmission_imag": float(np.imag(transmission)),
        }
        rows.append(row)

    return rows

def run_fixed_probe_sweep(hamiltonian, contact_setup, energies, probe_gamma):
    gamma_left = calculate_broadening(contact_setup["sigma_left"])
    gamma_right = calculate_broadening(contact_setup["sigma_right"])
    sigma_probe = build_fixed_probe_self_energy(
        len(hamiltonian),
        contact_setup["probe_indices"],
        probe_gamma,
    )
    rows = []

    for energy in energies:
        coherent_green_function = solve_green_function(
            hamiltonian=hamiltonian,
            energy=energy,
            sigma_left=contact_setup["sigma_left"],
            sigma_right=contact_setup["sigma_right"],
        )
        coherent_transmission = calculate_direct_transmission_complex(
            coherent_green_function,
            gamma_left,
            gamma_right,
        )
        check_transmission_is_real(coherent_transmission)

        probe_green_function = solve_green_function(
            hamiltonian=hamiltonian,
            energy=energy,
            sigma_left=contact_setup["sigma_left"],
            sigma_right=contact_setup["sigma_right"],
            sigma_decoherence=sigma_probe,
        )
        transport_results = run_transport_calculation(
            probe_green_function,
            contact_setup,
            sigma_probe,
        )
        fixed_probe_transmission = transport_results["t_eff"]

        row = {
            "energy_ev": energy,
            "coherent_transmission": float(np.real(coherent_transmission)),
            "coherent_transmission_imag": float(np.imag(coherent_transmission)),
            "fixed_probe_t_eff": fixed_probe_transmission,
            "probe_gamma_ev": probe_gamma,
            "active_probe_count": len(transport_results["active_probe_indices"]),
        }
        rows.append(row)

    return rows

def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

def write_basis(path, orbitals):
    rows = []

    for orbital in orbitals:
        row = {
            "index": orbital.index,
            "base_pair": orbital.base_pair,
            "strand": orbital.strand,
            "base": orbital.base,
            "band": orbital.band,
        }
        rows.append(row)

    write_csv(path, ["index", "base_pair", "strand", "base", "band"], rows)

def write_contact_indices(path, contact_setup):
    rows = []

    for index in contact_setup["left_contact"]:
        rows.append({"contact": "left", "orbital_index": index})

    for index in contact_setup["right_contact"]:
        rows.append({"contact": "right", "orbital_index": index})

    for index in contact_setup["probe_indices"]:
        rows.append({"contact": "fixed_probe", "orbital_index": index})

    write_csv(path, ["contact", "orbital_index"], rows)

def write_hamiltonian(path, hamiltonian):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, hamiltonian, delimiter=",")

def write_eigenvalues(path, hamiltonian, orbitals):
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    rows = []

    for index in range(len(eigenvalues)):
        vector = eigenvectors[:, index]
        homo_weight = 0.0
        lumo_weight = 0.0

        for orbital in orbitals:
            weight = abs(vector[orbital.index]) ** 2

            if orbital.band == "homo":
                homo_weight = homo_weight + weight
            else:
                lumo_weight = lumo_weight + weight

        dominant_band = "homo"

        if lumo_weight > homo_weight:
            dominant_band = "lumo"

        row = {
            "index": index,
            "eigenvalue_ev": eigenvalues[index],
            "homo_weight": homo_weight,
            "lumo_weight": lumo_weight,
            "dominant_band": dominant_band,
        }
        rows.append(row)

    write_csv(
        path,
        ["index", "eigenvalue_ev", "homo_weight", "lumo_weight", "dominant_band"],
        rows,
    )

    return rows

def write_coherent_transmission(path, rows):
    coherent_rows = []

    for row in rows:
        coherent_row = {
            "energy_ev": row["energy_ev"],
            "coherent_transmission": row["coherent_transmission"],
            "coherent_transmission_imag": row["coherent_transmission_imag"],
        }
        coherent_rows.append(coherent_row)

    write_csv(
        path,
        [
            "energy_ev",
            "coherent_transmission",
            "coherent_transmission_imag",
        ],
        coherent_rows,
    )

def write_fixed_probe_transmission(path, rows):
    write_csv(
        path,
        [
            "energy_ev",
            "coherent_transmission",
            "coherent_transmission_imag",
            "fixed_probe_t_eff",
            "probe_gamma_ev",
            "active_probe_count",
        ],
        rows,
    )

def get_frontier_values(eigenvalue_rows):
    homo_values = []
    lumo_values = []

    for row in eigenvalue_rows:
        value = float(row["eigenvalue_ev"])

        if row["dominant_band"] == "homo":
            homo_values.append(value)
        else:
            lumo_values.append(value)

    return {
        "highest_homo": max(homo_values),
        "lowest_lumo": min(lumo_values),
    }

def plot_transmission(path, rows, frontier_values, y_scale):
    energies = []
    transmissions = []

    for row in rows:
        if row["coherent_transmission"] > 0:
            energies.append(row["energy_ev"])
            transmissions.append(row["coherent_transmission"])

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    plt.plot(energies, transmissions)

    if y_scale == "log":
        plt.yscale("log")
        y_label = "coherent transmission (unitless, log y-axis)"
        title = "GGGGG 20-orbital coherent transmission (log scale)"
    else:
        y_label = "coherent transmission (unitless, linear y-axis)"
        title = "GGGGG 20-orbital coherent transmission (linear scale)"

    add_frontier_lines(frontier_values)
    plt.xlim(-10, 2.5)
    plt.xlabel("energy (eV)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", linewidth=0.4)
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_transmission_comparison(path, rows, frontier_values, y_scale):
    coherent_energies = []
    coherent_transmissions = []
    probe_energies = []
    probe_transmissions = []

    for row in rows:
        if row["coherent_transmission"] > 0:
            coherent_energies.append(row["energy_ev"])
            coherent_transmissions.append(row["coherent_transmission"])

        if row["fixed_probe_t_eff"] > 0:
            probe_energies.append(row["energy_ev"])
            probe_transmissions.append(row["fixed_probe_t_eff"])

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    plt.plot(coherent_energies, coherent_transmissions, label="coherent")
    plt.plot(probe_energies, probe_transmissions, label="fixed Gamma_B = 0.01 eV")

    if y_scale == "log":
        plt.yscale("log")
        y_label = "transmission (unitless, log y-axis)"
        title = "GGGGG 20-orbital transmission with fixed probes (log scale)"
    else:
        y_label = "transmission (unitless, linear y-axis)"
        title = "GGGGG 20-orbital transmission with fixed probes (linear scale)"

    add_frontier_lines(frontier_values)
    plt.xlim(-10, 2.5)
    plt.xlabel("energy (eV)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", linewidth=0.4)
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def add_frontier_lines(frontier_values):
    highest_homo = frontier_values["highest_homo"]
    lowest_lumo = frontier_values["lowest_lumo"]

    plt.axvline(
        highest_homo,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.2,
        label=f"highest HOMO = {highest_homo:.3f} eV",
    )
    plt.axvline(
        lowest_lumo,
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.2,
        label=f"lowest LUMO = {lowest_lumo:.3f} eV",
    )

def plot_eigenvalues(path, eigenvalue_rows):
    homo_values = []
    lumo_values = []

    for row in eigenvalue_rows:
        value = float(row["eigenvalue_ev"])

        if row["dominant_band"] == "homo":
            homo_values.append(value)
        else:
            lumo_values.append(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
    plot_eigenvalue_axis(axes[0], homo_values, "mostly HOMO eigenvalues", "#1f77b4")
    plot_eigenvalue_axis(axes[1], lumo_values, "mostly LUMO eigenvalues", "#ff7f0e")
    figure.suptitle("GGGGG 20-orbital Hamiltonian eigenvalues", fontsize=16)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_eigenvalue_axis(axis, values, title, color):
    indices = []

    for index in range(len(values)):
        indices.append(index)

    axis.scatter(values, indices, color=color, s=42)
    margin = max((max(values) - min(values)) * 0.08, 0.03)
    axis.set_xlim(min(values) - margin, max(values) + margin)
    axis.set_ylim(-0.6, len(values) - 0.1)

    for index in range(len(values)):
        value = values[index]
        axis.text(
            value,
            index + 0.22,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )

    axis.set_title(title)
    axis.set_xlabel("eigenvalue (eV)")
    axis.set_ylabel("level index")
    axis.set_yticks(indices)
    axis.grid(True, axis="x", linewidth=0.4)
    axis.grid(True, axis="y", linewidth=0.2)

def plot_eigenvalue_zoom_panels(path, eigenvalue_rows):
    homo_values = []
    lumo_values = []

    for row in eigenvalue_rows:
        value = float(row["eigenvalue_ev"])

        if row["dominant_band"] == "homo":
            homo_values.append(value)
        else:
            lumo_values.append(value)

    panels = [
        ("HOMO lower cluster", homo_values[:5], "#1f77b4"),
        ("HOMO upper cluster", homo_values[5:], "#1f77b4"),
        ("LUMO lower cluster", lumo_values[:5], "#ff7f0e"),
        ("LUMO upper cluster", lumo_values[5:], "#ff7f0e"),
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(9, 5.5))
    flat_axes = axes.flatten()

    for panel_index in range(len(panels)):
        title, values, color = panels[panel_index]
        axis = flat_axes[panel_index]
        y_values = []

        for index in range(len(values)):
            y_values.append(0)

        axis.scatter(values, y_values, color=color, s=48)

        for index in range(len(values)):
            value = values[index]
            y_offset = 0.10

            if index % 2 == 1:
                y_offset = -0.16

            axis.text(
                value,
                y_offset,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )

        margin = max((max(values) - min(values)) * 0.25, 0.01)
        axis.set_xlim(min(values) - margin, max(values) + margin)
        axis.set_ylim(-0.35, 0.35)
        axis.set_yticks([])
        axis.set_title(title)
        axis.set_xlabel("eigenvalue (eV)")
        axis.grid(True, axis="x", linewidth=0.4)

    figure.suptitle("GGGGG 20-orbital eigenvalue clusters", fontsize=16)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def print_written(path):
    print(f"wrote {path}")

def main():
    args = parse_args()
    args.sequence = args.sequence.upper()

    try:
        validate_args(args)
        output_dir = Path(args.output_dir)
        hamiltonian, orbitals = build_full_hamiltonian(args.sequence)
        contact_setup = build_terminal_contact_setup(
            orbitals,
            args.gamma_left,
            args.gamma_right,
        )
        energies = build_energy_grid(
            args.energy_min,
            args.energy_max,
            args.energy_step,
        )
        transmission_rows = run_fixed_probe_sweep(
            hamiltonian,
            contact_setup,
            energies,
            args.probe_gamma,
        )
    except ValueError as error:
        raise SystemExit(f"error: {error}")

    basis_path = output_dir / "basis.csv"
    contact_path = output_dir / "contacts.csv"
    hamiltonian_path = output_dir / "hamiltonian.csv"
    eigenvalues_path = output_dir / "eigenvalues.csv"
    transmission_path = output_dir / "transmission_coherent.csv"
    transmission_fixed_probe_path = output_dir / "transmission_fixed_probe.csv"
    transmission_plot_path = output_dir / "transmission_coherent.png"
    transmission_linear_plot_path = output_dir / "transmission_coherent_linear.png"
    fixed_probe_plot_path = output_dir / "transmission_fixed_probe.png"
    fixed_probe_linear_plot_path = output_dir / "transmission_fixed_probe_linear.png"
    eigenvalue_plot_path = output_dir / "eigenvalues.png"
    eigenvalue_zoom_path = output_dir / "eigenvalues_zoom.png"

    write_basis(basis_path, orbitals)
    write_contact_indices(contact_path, contact_setup)
    write_hamiltonian(hamiltonian_path, hamiltonian)
    eigenvalue_rows = write_eigenvalues(eigenvalues_path, hamiltonian, orbitals)
    frontier_values = get_frontier_values(eigenvalue_rows)
    write_coherent_transmission(transmission_path, transmission_rows)
    write_fixed_probe_transmission(transmission_fixed_probe_path, transmission_rows)
    plot_transmission(transmission_plot_path, transmission_rows, frontier_values, "log")
    plot_transmission(
        transmission_linear_plot_path,
        transmission_rows,
        frontier_values,
        "linear",
    )
    plot_transmission_comparison(
        fixed_probe_plot_path,
        transmission_rows,
        frontier_values,
        "log",
    )
    plot_transmission_comparison(
        fixed_probe_linear_plot_path,
        transmission_rows,
        frontier_values,
        "linear",
    )
    plot_eigenvalues(eigenvalue_plot_path, eigenvalue_rows)
    plot_eigenvalue_zoom_panels(eigenvalue_zoom_path, eigenvalue_rows)

    print_written(basis_path)
    print_written(contact_path)
    print_written(hamiltonian_path)
    print_written(eigenvalues_path)
    print_written(transmission_path)
    print_written(transmission_fixed_probe_path)
    print_written(transmission_plot_path)
    print_written(transmission_linear_plot_path)
    print_written(fixed_probe_plot_path)
    print_written(fixed_probe_linear_plot_path)
    print_written(eigenvalue_plot_path)
    print_written(eigenvalue_zoom_path)

if __name__ == "__main__":
    main()

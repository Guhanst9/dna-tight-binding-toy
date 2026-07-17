import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.basis import Orbital
from src.builder import build_hamiltonian
from src.contacts import build_contact_self_energy
from src.greens import (
    calculate_broadening,
    solve_green_function,
)
from src.transport import run_transport_calculation

complements = {
    "G": "C",
    "C": "G",
    "A": "T",
    "T": "A",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="GGGGG")
    parser.add_argument("--gamma-left", type=float, default=1.0)
    parser.add_argument("--gamma-right", type=float, default=1.0)
    parser.add_argument("--probe-gamma", type=float, default=0.01)
    parser.add_argument("--energy-min", type=float, default=-8.0)
    parser.add_argument("--energy-max", type=float, default=3.0)
    parser.add_argument("--energy-step", type=float, default=0.01)
    parser.add_argument("--output-dir", default="outputs/sequence_ggggg_fixed_probe")
    return parser.parse_args()

def validate_args(args):
    sequence = args.sequence.upper()

    if len(sequence) < 2:
        raise ValueError("sequence must have at least two base pairs")

    for base in sequence:
        if base not in complements:
            raise ValueError("sequence can only contain A, T, G, and C")

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

def build_sequence_basis(sequence):
    orbitals = []

    for base_pair_index in range(len(sequence)):
        first_base = sequence[base_pair_index]
        second_base = complements[first_base]

        orbitals.append(
            Orbital(
                index=len(orbitals),
                base_pair=base_pair_index + 1,
                strand=1,
                base=first_base,
            )
        )
        orbitals.append(
            Orbital(
                index=len(orbitals),
                base_pair=base_pair_index + 1,
                strand=2,
                base=second_base,
            )
        )

    return orbitals

def build_sequence_hamiltonian(sequence, band):
    if sequence.upper() == "GGGGG":
        hamiltonian, unused_orbitals = build_hamiltonian(
            pair="gc",
            band=band,
            base_pair_count=len(sequence),
        )
        orbitals = build_sequence_basis(sequence)
        return hamiltonian, orbitals

    raise ValueError("only GGGGG is supported for this fixed-probe test")

def build_fixed_probe_contact_setup(orbitals, gamma_left, gamma_right):
    size = len(orbitals)
    left_contact = []
    right_contact = []
    probe_indices = []

    for orbital in orbitals:
        if orbital.base_pair == 1:
            left_contact.append(orbital.index)

        if orbital.base_pair == 2:
            right_contact.append(orbital.index)

        probe_indices.append(orbital.index)

    return {
        "left_contact": left_contact,
        "right_contact": right_contact,
        "probe_indices": probe_indices,
        "sigma_left": build_contact_self_energy(size, left_contact, gamma_left),
        "sigma_right": build_contact_self_energy(size, right_contact, gamma_right),
    }

def build_fixed_probe_self_energy(size, probe_indices, probe_gamma):
    self_energy = np.zeros((size, size), dtype=complex)

    for index in probe_indices:
        self_energy[index, index] = -1j * probe_gamma / 2

    return self_energy

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

def calculate_coherent_transmission(hamiltonian, contact_setup, energy):
    green_function = solve_green_function(
        hamiltonian=hamiltonian,
        energy=energy,
        sigma_left=contact_setup["sigma_left"],
        sigma_right=contact_setup["sigma_right"],
    )
    gamma_left = calculate_broadening(contact_setup["sigma_left"])
    gamma_right = calculate_broadening(contact_setup["sigma_right"])
    transmission = calculate_direct_transmission_complex(
        green_function,
        gamma_left,
        gamma_right,
    )
    check_transmission_is_real(transmission)
    return transmission

def calculate_fixed_probe_transmission(
    hamiltonian,
    contact_setup,
    sigma_probe,
    energy,
):
    green_function = solve_green_function(
        hamiltonian=hamiltonian,
        energy=energy,
        sigma_left=contact_setup["sigma_left"],
        sigma_right=contact_setup["sigma_right"],
        sigma_decoherence=sigma_probe,
    )
    gamma_left = calculate_broadening(contact_setup["sigma_left"])
    gamma_right = calculate_broadening(contact_setup["sigma_right"])
    direct_transmission = calculate_direct_transmission_complex(
        green_function,
        gamma_left,
        gamma_right,
    )
    check_transmission_is_real(direct_transmission)
    transport_results = run_transport_calculation(
        green_function,
        contact_setup,
        sigma_probe,
    )
    return {
        "t_eff": transport_results["t_eff"],
        "direct_t_lr": direct_transmission,
    }

def run_band_sweep(sequence, band, args):
    hamiltonian, orbitals = build_sequence_hamiltonian(sequence, band)
    contact_setup = build_fixed_probe_contact_setup(
        orbitals,
        args.gamma_left,
        args.gamma_right,
    )
    sigma_probe = build_fixed_probe_self_energy(
        len(orbitals),
        contact_setup["probe_indices"],
        args.probe_gamma,
    )
    energies = build_energy_grid(
        args.energy_min,
        args.energy_max,
        args.energy_step,
    )
    rows = []

    for energy in energies:
        coherent_t_lr = calculate_coherent_transmission(
            hamiltonian,
            contact_setup,
            energy,
        )
        decoherent_t_eff = calculate_fixed_probe_transmission(
            hamiltonian,
            contact_setup,
            sigma_probe,
            energy,
        )
        row = {
            "energy_ev": energy,
            "coherent_t_lr": float(np.real(coherent_t_lr)),
            "coherent_t_lr_imag": float(np.imag(coherent_t_lr)),
            "decoherent_t_eff": float(decoherent_t_eff["t_eff"]),
            "decoherent_direct_t_lr_imag": float(np.imag(decoherent_t_eff["direct_t_lr"])),
        }
        rows.append(row)

    return {
        "hamiltonian": hamiltonian,
        "orbitals": orbitals,
        "contact_setup": contact_setup,
        "rows": rows,
    }

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
        }
        rows.append(row)

    write_csv(path, ["index", "base_pair", "strand", "base"], rows)

def write_hamiltonian(path, hamiltonian):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, hamiltonian, delimiter=",")

def write_eigenvalues(path, hamiltonian):
    eigenvalues = np.linalg.eigvalsh(hamiltonian)
    rows = []

    for index in range(len(eigenvalues)):
        row = {
            "index": index,
            "eigenvalue_ev": eigenvalues[index],
        }
        rows.append(row)

    write_csv(path, ["index", "eigenvalue_ev"], rows)

def write_band_rows(path, rows):
    write_csv(
        path,
        [
            "energy_ev",
            "coherent_t_lr",
            "coherent_t_lr_imag",
            "decoherent_t_eff",
            "decoherent_direct_t_lr_imag",
        ],
        rows,
    )

def write_combined_rows(path, homo_rows, lumo_rows):
    rows = []

    for index in range(len(homo_rows)):
        row = {
            "energy_ev": homo_rows[index]["energy_ev"],
            "homo_coherent_t_lr": homo_rows[index]["coherent_t_lr"],
            "homo_coherent_t_lr_imag": homo_rows[index]["coherent_t_lr_imag"],
            "homo_decoherent_t_eff": homo_rows[index]["decoherent_t_eff"],
            "homo_decoherent_direct_t_lr_imag": homo_rows[index]["decoherent_direct_t_lr_imag"],
            "lumo_coherent_t_lr": lumo_rows[index]["coherent_t_lr"],
            "lumo_coherent_t_lr_imag": lumo_rows[index]["coherent_t_lr_imag"],
            "lumo_decoherent_t_eff": lumo_rows[index]["decoherent_t_eff"],
            "lumo_decoherent_direct_t_lr_imag": lumo_rows[index]["decoherent_direct_t_lr_imag"],
        }
        rows.append(row)

    write_csv(
        path,
        [
            "energy_ev",
            "homo_coherent_t_lr",
            "homo_coherent_t_lr_imag",
            "homo_decoherent_t_eff",
            "homo_decoherent_direct_t_lr_imag",
            "lumo_coherent_t_lr",
            "lumo_coherent_t_lr_imag",
            "lumo_decoherent_t_eff",
            "lumo_decoherent_direct_t_lr_imag",
        ],
        rows,
    )

def plot_band(path, rows, band):
    energies = []
    coherent = []
    decoherent = []

    for row in rows:
        if row["coherent_t_lr"] > 0:
            energies.append(row["energy_ev"])
            coherent.append(row["coherent_t_lr"])

    decoherent_energies = []
    for row in rows:
        if row["decoherent_t_eff"] > 0:
            decoherent_energies.append(row["energy_ev"])
            decoherent.append(row["decoherent_t_eff"])

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(energies, coherent, label=f"{band.upper()} Hamiltonian, coherent")
    plt.plot(decoherent_energies, decoherent, label=f"{band.upper()} Hamiltonian, fixed Gamma_B")
    plt.yscale("log")
    plt.xlabel("energy (eV)")
    plt.ylabel("transmission (unitless, log y-axis)")
    plt.title(f"GGGGG {band.upper()} transmission (log y-axis)")
    plt.grid(True, which="both", linewidth=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_combined(path, homo_rows, lumo_rows):
    series = [
        ("HOMO Hamiltonian, coherent", homo_rows, "coherent_t_lr"),
        ("HOMO Hamiltonian, fixed Gamma_B", homo_rows, "decoherent_t_eff"),
        ("LUMO Hamiltonian, coherent", lumo_rows, "coherent_t_lr"),
        ("LUMO Hamiltonian, fixed Gamma_B", lumo_rows, "decoherent_t_eff"),
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))

    for label, rows, key in series:
        energies = []
        transmissions = []

        for row in rows:
            if row[key] > 0:
                energies.append(row["energy_ev"])
                transmissions.append(row[key])

        plt.plot(energies, transmissions, label=label)

    plt.yscale("log")
    plt.xlabel("energy (eV)")
    plt.ylabel("transmission (unitless, log y-axis)")
    plt.title("GGGGG coherent and fixed-probe transmission (log y-axis)")
    plt.grid(True, which="both", linewidth=0.4)
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_eigenvalues(path, homo_hamiltonian, lumo_hamiltonian):
    homo_eigenvalues = np.linalg.eigvalsh(homo_hamiltonian)
    lumo_eigenvalues = np.linalg.eigvalsh(lumo_hamiltonian)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=False)

    plot_eigenvalue_axis(
        axes[0],
        homo_eigenvalues,
        "HOMO Hamiltonian eigenvalues",
        "#1f77b4",
    )
    plot_eigenvalue_axis(
        axes[1],
        lumo_eigenvalues,
        "LUMO Hamiltonian eigenvalues",
        "#ff7f0e",
    )

    figure.suptitle("GGGGG Hamiltonian eigenvalues", fontsize=16)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_eigenvalue_axis(axis, eigenvalues, title, color):
    indices = []

    for index in range(len(eigenvalues)):
        indices.append(index)

    axis.scatter(eigenvalues, indices, color=color, s=42)
    axis_margin = max((max(eigenvalues) - min(eigenvalues)) * 0.08, 0.03)
    axis.set_xlim(min(eigenvalues) - axis_margin, max(eigenvalues) + axis_margin)
    axis.set_ylim(-0.6, len(eigenvalues) - 0.1)

    for index in range(len(eigenvalues)):
        value = eigenvalues[index]
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

def plot_eigenvalue_zoom_panels(path, homo_hamiltonian, lumo_hamiltonian):
    homo_eigenvalues = np.linalg.eigvalsh(homo_hamiltonian)
    lumo_eigenvalues = np.linalg.eigvalsh(lumo_hamiltonian)
    panels = [
        ("HOMO lower cluster", homo_eigenvalues[:5], "#1f77b4"),
        ("HOMO upper cluster", homo_eigenvalues[5:], "#1f77b4"),
        ("LUMO lower cluster", lumo_eigenvalues[:5], "#ff7f0e"),
        ("LUMO upper cluster", lumo_eigenvalues[5:], "#ff7f0e"),
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

    figure.suptitle("GGGGG eigenvalue clusters", fontsize=16)
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
        homo_result = run_band_sweep(args.sequence, "homo", args)
        lumo_result = run_band_sweep(args.sequence, "lumo", args)
    except ValueError as error:
        raise SystemExit(f"error: {error}")

    write_basis(output_dir / "basis.csv", homo_result["orbitals"])
    write_hamiltonian(output_dir / "hamiltonian_homo.csv", homo_result["hamiltonian"])
    write_hamiltonian(output_dir / "hamiltonian_lumo.csv", lumo_result["hamiltonian"])
    write_eigenvalues(output_dir / "eigenvalues_homo.csv", homo_result["hamiltonian"])
    write_eigenvalues(output_dir / "eigenvalues_lumo.csv", lumo_result["hamiltonian"])
    write_band_rows(output_dir / "transmission_homo.csv", homo_result["rows"])
    write_band_rows(output_dir / "transmission_lumo.csv", lumo_result["rows"])
    write_combined_rows(
        output_dir / "transmission_combined.csv",
        homo_result["rows"],
        lumo_result["rows"],
    )
    plot_band(output_dir / "transmission_homo.png", homo_result["rows"], "homo")
    plot_band(output_dir / "transmission_lumo.png", lumo_result["rows"], "lumo")
    plot_combined(
        output_dir / "transmission_combined.png",
        homo_result["rows"],
        lumo_result["rows"],
    )
    plot_eigenvalues(
        output_dir / "eigenvalues.png",
        homo_result["hamiltonian"],
        lumo_result["hamiltonian"],
    )
    plot_eigenvalue_zoom_panels(
        output_dir / "eigenvalues_zoom.png",
        homo_result["hamiltonian"],
        lumo_result["hamiltonian"],
    )

    print_written(output_dir / "basis.csv")
    print_written(output_dir / "hamiltonian_homo.csv")
    print_written(output_dir / "hamiltonian_lumo.csv")
    print_written(output_dir / "eigenvalues_homo.csv")
    print_written(output_dir / "eigenvalues_lumo.csv")
    print_written(output_dir / "transmission_homo.csv")
    print_written(output_dir / "transmission_lumo.csv")
    print_written(output_dir / "transmission_combined.csv")
    print_written(output_dir / "transmission_homo.png")
    print_written(output_dir / "transmission_lumo.png")
    print_written(output_dir / "transmission_combined.png")
    print_written(output_dir / "eigenvalues.png")
    print_written(output_dir / "eigenvalues_zoom.png")

if __name__ == "__main__":
    main()

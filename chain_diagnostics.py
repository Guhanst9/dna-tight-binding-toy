import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.greens import (
    calculate_broadening,
    calculate_direct_transmission,
    solve_green_function,
)
from src.transport import run_transport_calculation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=["all", "coherent-10", "coherent-100", "decoherent-10"],
        default="all",
    )
    parser.add_argument("--onsite", type=float, default=0.0)
    parser.add_argument("--hopping", type=float, default=1.0)
    parser.add_argument("--gamma-left", type=float, default=1.0)
    parser.add_argument("--gamma-right", type=float, default=1.0)
    parser.add_argument("--probe-gamma", type=float, default=0.01)
    parser.add_argument("--energy-min", type=float, default=-3.0)
    parser.add_argument("--energy-max", type=float, default=3.0)
    parser.add_argument("--energy-points", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/chain_diagnostics")
    return parser.parse_args()

def validate_args(args):
    if args.gamma_left < 0:
        raise ValueError("gamma left must be nonnegative")

    if args.gamma_right < 0:
        raise ValueError("gamma right must be nonnegative")

    if args.probe_gamma < 0:
        raise ValueError("probe gamma must be nonnegative")

    if args.energy_min >= args.energy_max:
        raise ValueError("energy min must be less than energy max")

    if args.energy_points < 2:
        raise ValueError("energy points must be at least 2")

def build_chain_hamiltonian(size, onsite, hopping):
    hamiltonian = np.zeros((size, size))

    for index in range(size):
        hamiltonian[index, index] = onsite

    for index in range(size - 1):
        hamiltonian[index, index + 1] = hopping
        hamiltonian[index + 1, index] = hopping

    return hamiltonian

def build_contact_self_energy(size, index, gamma):
    self_energy = np.zeros((size, size), dtype=complex)
    self_energy[index, index] = -1j * gamma / 2
    return self_energy

def build_chain_contact_setup(size, gamma_left, gamma_right, include_probes):
    left_contact = [0]
    right_contact = [size - 1]
    probe_indices = []

    if include_probes:
        for index in range(1, size - 1):
            probe_indices.append(index)

    return {
        "left_contact": left_contact,
        "right_contact": right_contact,
        "probe_indices": probe_indices,
        "sigma_left": build_contact_self_energy(size, 0, gamma_left),
        "sigma_right": build_contact_self_energy(size, size - 1, gamma_right),
    }

def build_probe_self_energy(size, probe_indices, probe_gamma):
    self_energy = np.zeros((size, size), dtype=complex)

    for index in probe_indices:
        self_energy[index, index] = -1j * probe_gamma / 2

    return self_energy

def calculate_coherent_transmission(hamiltonian, contact_setup, energy):
    green_function = solve_green_function(
        hamiltonian=hamiltonian,
        energy=energy,
        sigma_left=contact_setup["sigma_left"],
        sigma_right=contact_setup["sigma_right"],
    )
    gamma_left = calculate_broadening(contact_setup["sigma_left"])
    gamma_right = calculate_broadening(contact_setup["sigma_right"])
    transmission = calculate_direct_transmission(
        green_function,
        gamma_left,
        gamma_right,
    )
    return transmission

def run_coherent_sweep(size, args):
    hamiltonian = build_chain_hamiltonian(size, args.onsite, args.hopping)
    contact_setup = build_chain_contact_setup(
        size,
        args.gamma_left,
        args.gamma_right,
        False,
    )
    energies = np.linspace(args.energy_min, args.energy_max, args.energy_points)
    rows = []

    for energy in energies:
        transmission = calculate_coherent_transmission(
            hamiltonian,
            contact_setup,
            energy,
        )
        row = {
            "energy": float(energy),
            "transmission": transmission,
        }
        rows.append(row)

    return hamiltonian, rows

def run_decoherent_sweep(size, args):
    hamiltonian = build_chain_hamiltonian(size, args.onsite, args.hopping)
    contact_setup = build_chain_contact_setup(
        size,
        args.gamma_left,
        args.gamma_right,
        True,
    )
    sigma_decoherence = build_probe_self_energy(
        size,
        contact_setup["probe_indices"],
        args.probe_gamma,
    )
    energies = np.linspace(args.energy_min, args.energy_max, args.energy_points)
    rows = []

    for energy in energies:
        coherent_transmission = calculate_coherent_transmission(
            hamiltonian,
            contact_setup,
            energy,
        )
        green_function = solve_green_function(
            hamiltonian=hamiltonian,
            energy=energy,
            sigma_left=contact_setup["sigma_left"],
            sigma_right=contact_setup["sigma_right"],
            sigma_decoherence=sigma_decoherence,
        )
        transport_results = run_transport_calculation(
            green_function,
            contact_setup,
            sigma_decoherence,
        )
        row = {
            "energy": float(energy),
            "coherent_t_lr": coherent_transmission,
            "decoherent_t_eff": transport_results["t_eff"],
            "gamma_b": args.probe_gamma,
        }
        rows.append(row)

    return hamiltonian, rows

def write_hamiltonian(path, hamiltonian):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, hamiltonian, delimiter=",")

def write_coherent_transmission(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["energy", "transmission"])
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

def write_decoherent_transmission(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "energy",
        "coherent_t_lr",
        "decoherent_t_eff",
        "gamma_b",
    ]

    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

def plot_coherent_transmission(path, rows, title):
    energies = []
    transmissions = []

    for row in rows:
        if row["transmission"] > 0:
            energies.append(row["energy"])
            transmissions.append(row["transmission"])

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(energies, transmissions)
    plt.xlabel("energy")
    plt.ylabel("transmission (unitless)")
    plt.title(title)
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_decoherent_transmission(path, rows, title):
    energies = []
    coherent = []
    decoherent = []

    for row in rows:
        energies.append(row["energy"])
        coherent.append(row["coherent_t_lr"])
        decoherent.append(row["decoherent_t_eff"])

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    plt.plot(energies, coherent, label="coherent")
    plt.plot(energies, decoherent, label="with decoherence")
    plt.xlabel("energy")
    plt.ylabel("transmission (unitless)")
    plt.title(title)
    plt.grid(True, linewidth=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def write_coherent_case(output_dir, hamiltonian, rows, title):
    hamiltonian_path = output_dir / "hamiltonian.csv"
    transmission_path = output_dir / "transmission_vs_energy.csv"
    plot_path = output_dir / "transmission_vs_energy.png"

    write_hamiltonian(hamiltonian_path, hamiltonian)
    write_coherent_transmission(transmission_path, rows)
    plot_coherent_transmission(plot_path, rows, title)

    print(f"wrote {hamiltonian_path}")
    print(f"wrote {transmission_path}")
    print(f"wrote {plot_path}")

def write_decoherent_case(output_dir, hamiltonian, rows, title):
    hamiltonian_path = output_dir / "hamiltonian.csv"
    transmission_path = output_dir / "transmission_vs_energy.csv"
    plot_path = output_dir / "transmission_vs_energy.png"

    write_hamiltonian(hamiltonian_path, hamiltonian)
    write_decoherent_transmission(transmission_path, rows)
    plot_decoherent_transmission(plot_path, rows, title)

    print(f"wrote {hamiltonian_path}")
    print(f"wrote {transmission_path}")
    print(f"wrote {plot_path}")

def run_case(case_name, args):
    output_root = Path(args.output_dir)

    if case_name == "coherent-10":
        hamiltonian, rows = run_coherent_sweep(10, args)
        write_coherent_case(
            output_root / "coherent_10",
            hamiltonian,
            rows,
            "10-site chain coherent transmission",
        )

    if case_name == "coherent-100":
        hamiltonian, rows = run_coherent_sweep(100, args)
        write_coherent_case(
            output_root / "coherent_100",
            hamiltonian,
            rows,
            "100-site chain coherent transmission",
        )

    if case_name == "decoherent-10":
        hamiltonian, rows = run_decoherent_sweep(10, args)
        write_decoherent_case(
            output_root / "decoherent_10",
            hamiltonian,
            rows,
            "10-site chain transmission with fixed probe decoherence",
        )

def main():
    args = parse_args()

    try:
        validate_args(args)

        if args.case == "all":
            run_case("coherent-10", args)
            run_case("coherent-100", args)
            run_case("decoherent-10", args)
        else:
            run_case(args.case, args)
    except ValueError as error:
        raise SystemExit(f"error: {error}")

if __name__ == "__main__":
    main()

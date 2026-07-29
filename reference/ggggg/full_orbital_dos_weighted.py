import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from reference.ggggg.dos_weighted_solver import run_dos_weighted_solver
from reference.ggggg.full_orbital_model import (
    build_full_hamiltonian,
    build_partitioned_contact_setup,
    validate_ggggg_sequence,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="GGGGG")
    parser.add_argument("--gamma-left", type=float, default=1.0)
    parser.add_argument("--gamma-right", type=float, default=1.0)
    parser.add_argument("--d0", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--energy-min", type=float, default=-10.0)
    parser.add_argument("--energy-max", type=float, default=10.0)
    parser.add_argument("--energy-step", type=float, default=0.01)
    parser.add_argument("--trace-energy", type=float, default=-4.0)
    parser.add_argument(
        "--output-dir",
        default="outputs/sequence_ggggg_20_orbital_dos_weighted",
    )
    return parser.parse_args()


def validate_args(args):
    validate_ggggg_sequence(args.sequence)

    if args.gamma_left < 0:
        raise ValueError("gamma left must be nonnegative")

    if args.gamma_right < 0:
        raise ValueError("gamma right must be nonnegative")

    if args.d0 < 0:
        raise ValueError("d0 must be nonnegative")

    if args.alpha < 0 or args.alpha > 1:
        raise ValueError("alpha must be between 0 and 1")

    if args.tolerance < 0:
        raise ValueError("tolerance must be nonnegative")

    if args.max_iterations < 1:
        raise ValueError("max iterations must be at least 1")

    if args.energy_min >= args.energy_max:
        raise ValueError("energy min must be less than energy max")

    if args.energy_step <= 0:
        raise ValueError("energy step must be positive")

    if args.energy_step > 0.01:
        raise ValueError("energy step must be 0.01 eV or smaller")

    if args.trace_energy < args.energy_min:
        raise ValueError("trace energy must be inside the energy range")

    if args.trace_energy > args.energy_max:
        raise ValueError("trace energy must be inside the energy range")


def build_energy_grid(energy_min, energy_max, energy_step):
    point_count = int(round((energy_max - energy_min) / energy_step)) + 1
    energies = np.zeros(point_count)

    for index in range(point_count):
        energies[index] = energy_min + index * energy_step

    return energies


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
        rows.append(
            {
                "index": orbital.index,
                "base_pair": orbital.base_pair,
                "strand": orbital.strand,
                "base": orbital.base,
                "band": orbital.band,
            }
        )

    write_csv(path, ["index", "base_pair", "strand", "base", "band"], rows)


def write_partitions(path, contact_setup):
    rows = []

    for partition_index in range(len(contact_setup["probe_partitions"])):
        partition = contact_setup["probe_partitions"][partition_index]
        rows.append(
            {
                "probe_index": partition_index,
                "base_pair": partition.base_pair,
                "strand": partition.strand,
                "base": partition.base,
                "orbital_indices": " ".join(
                    str(index) for index in partition.orbital_indices
                ),
            }
        )

    write_csv(
        path,
        ["probe_index", "base_pair", "strand", "base", "orbital_indices"],
        rows,
    )


def write_contacts(path, contact_setup):
    rows = []

    for index in contact_setup["left_indices"]:
        rows.append({"contact": "left", "orbital_index": index})

    for index in contact_setup["right_indices"]:
        rows.append({"contact": "right", "orbital_index": index})

    write_csv(path, ["contact", "orbital_index"], rows)


def write_hamiltonian(path, hamiltonian):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, hamiltonian, delimiter=",")


def write_eigenvalues(path, hamiltonian):
    eigenvalues = np.linalg.eigvalsh(hamiltonian)
    rows = []

    for index in range(len(eigenvalues)):
        rows.append({"index": index, "eigenvalue_ev": eigenvalues[index]})

    write_csv(path, ["index", "eigenvalue_ev"], rows)


def get_frontier_values(hamiltonian):
    size = hamiltonian.shape[0]
    half_size = size // 2
    homo_eigenvalues = np.linalg.eigvalsh(hamiltonian[:half_size, :half_size])
    lumo_eigenvalues = np.linalg.eigvalsh(hamiltonian[half_size:, half_size:])

    return {
        "highest_homo": float(np.max(homo_eigenvalues)),
        "lowest_lumo": float(np.min(lumo_eigenvalues)),
    }


def format_change(value):
    if value is None:
        return ""

    return float(value)


def write_transmission(path, results):
    coherent_transmission = results["history"][0]["t_lr"]
    rows = []

    for energy_index in range(len(results["energies"])):
        rows.append(
            {
                "energy_ev": results["energies"][energy_index],
                "coherent_t_lr": coherent_transmission[energy_index],
                "final_t_lr": results["t_lr"][energy_index],
                "final_t_eff": results["t_eff"][energy_index],
                "final_dos": results["dos"][energy_index],
                "dos_change_percent": format_change(
                    None
                    if results["dos_changes"] is None
                    else results["dos_changes"][energy_index]
                ),
                "transmission_change_percent": format_change(
                    None
                    if results["transmission_changes"] is None
                    else results["transmission_changes"][energy_index]
                ),
            }
        )

    write_csv(
        path,
        [
            "energy_ev",
            "coherent_t_lr",
            "final_t_lr",
            "final_t_eff",
            "final_dos",
            "dos_change_percent",
            "transmission_change_percent",
        ],
        rows,
    )


def write_iteration_history(path, results):
    rows = []

    for history_row in results["history"]:
        for energy_index in range(len(results["energies"])):
            rows.append(
                {
                    "iteration": history_row["iteration"],
                    "energy_ev": results["energies"][energy_index],
                    "dos": history_row["dos"][energy_index],
                    "t_lr": history_row["t_lr"][energy_index],
                    "t_eff": history_row["t_eff"][energy_index],
                    "dos_change_percent": format_change(
                        None
                        if history_row["dos_changes"] is None
                        else history_row["dos_changes"][energy_index]
                    ),
                    "transmission_change_percent": format_change(
                        None
                        if history_row["transmission_changes"] is None
                        else history_row["transmission_changes"][energy_index]
                    ),
                }
            )

    write_csv(
        path,
        [
            "iteration",
            "energy_ev",
            "dos",
            "t_lr",
            "t_eff",
            "dos_change_percent",
            "transmission_change_percent",
        ],
        rows,
    )


def integrate_dos(energies, dos):
    total = 0.0

    for energy_index in range(len(energies) - 1):
        energy_step = energies[energy_index + 1] - energies[energy_index]
        average_dos = (dos[energy_index] + dos[energy_index + 1]) / 2
        total = total + average_dos * energy_step

    return total


def write_total_dos_history(path, results):
    rows = []

    for history_row in results["history"]:
        total_dos = integrate_dos(results["energies"], history_row["dos"])
        rows.append(
            {
                "iteration": history_row["iteration"],
                "integrated_total_dos_states": total_dos,
            }
        )

    write_csv(
        path,
        ["iteration", "integrated_total_dos_states"],
        rows,
    )


def write_probe_gamma_history(path, results, probe_partitions):
    rows = []

    for history_row in results["history"]:
        for energy_index in range(len(results["energies"])):
            for partition_index in range(len(probe_partitions)):
                partition = probe_partitions[partition_index]
                rows.append(
                    {
                        "iteration": history_row["iteration"],
                        "energy_ev": results["energies"][energy_index],
                        "probe_index": partition_index,
                        "base_pair": partition.base_pair,
                        "strand": partition.strand,
                        "base": partition.base,
                        "gamma_d_ev": history_row["probe_gammas"][
                            energy_index,
                            partition_index
                        ],
                    }
                )

    write_csv(
        path,
        [
            "iteration",
            "energy_ev",
            "probe_index",
            "base_pair",
            "strand",
            "base",
            "gamma_d_ev",
        ],
        rows,
    )


def get_trace_index(energies, trace_energy):
    differences = abs(energies - trace_energy)
    return int(np.argmin(differences))


def write_trace_probe_gammas(path, results, probe_partitions, trace_index):
    rows = []

    for history_row in results["history"]:
        for partition_index in range(len(probe_partitions)):
            partition = probe_partitions[partition_index]
            rows.append(
                {
                    "iteration": history_row["iteration"],
                    "energy_ev": results["energies"][trace_index],
                    "probe_index": partition_index,
                    "base_pair": partition.base_pair,
                    "strand": partition.strand,
                    "base": partition.base,
                    "gamma_d_ev": history_row["probe_gammas"][
                        trace_index,
                        partition_index
                    ],
                }
            )

    write_csv(
        path,
        [
            "iteration",
            "energy_ev",
            "probe_index",
            "base_pair",
            "strand",
            "base",
            "gamma_d_ev",
        ],
        rows,
    )


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


def plot_transmission(path, results, frontier_values, d0, alpha, y_scale):
    coherent_transmission = results["history"][0]["t_lr"]
    coherent_energies = []
    coherent_values = []
    effective_energies = []
    effective_values = []

    for energy_index in range(len(results["energies"])):
        energy = results["energies"][energy_index]
        coherent_value = coherent_transmission[energy_index]
        effective_value = results["t_eff"][energy_index]

        if y_scale == "linear" or coherent_value > 0:
            coherent_energies.append(energy)
            coherent_values.append(coherent_value)

        if y_scale == "linear" or effective_value > 0:
            effective_energies.append(energy)
            effective_values.append(effective_value)

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 5))
    plt.plot(coherent_energies, coherent_values, label="coherent")
    plt.plot(effective_energies, effective_values, label="DOS-weighted decoherence")

    if y_scale == "log":
        plt.yscale("log")
        y_label = "transmission (unitless, log y-axis)"
        title = (
            "GGGGG 20-orbital DOS-weighted transmission "
            f"(D0 = {d0:g} eV^2, alpha = {alpha:g}, log scale)"
        )
    else:
        y_label = "transmission (unitless, linear y-axis)"
        title = (
            "GGGGG 20-orbital DOS-weighted transmission "
            f"(D0 = {d0:g} eV^2, alpha = {alpha:g}, linear scale)"
        )

    add_frontier_lines(frontier_values)
    plt.xlabel("energy (eV)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", linewidth=0.4)
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_residue_loss(path, results, probe_partitions, trace_index, d0, alpha):
    iterations = []

    for history_row in results["history"]:
        iterations.append(history_row["iteration"])

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 5))

    for partition_index in range(len(probe_partitions)):
        partition = probe_partitions[partition_index]
        gamma_values = []

        for history_row in results["history"]:
            gamma_values.append(
                history_row["probe_gammas"][trace_index, partition_index]
            )

        label = (
            f"bp{partition.base_pair} "
            f"strand{partition.strand} {partition.base}"
        )
        plt.plot(iterations, gamma_values, marker="o", label=label)

    trace_energy = results["energies"][trace_index]
    plt.xlabel("iteration")
    plt.ylabel("loss per residue, Gamma_D (eV)")
    plt.title(
        "DOS-weighted loss per residue "
        f"(D0 = {d0:g} eV^2, alpha = {alpha:g}, E = {trace_energy:.3f} eV)"
    )
    plt.grid(True, linewidth=0.4)
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_total_dos_history(path, results, d0, alpha):
    iterations = []
    total_dos_values = []

    for history_row in results["history"]:
        iterations.append(history_row["iteration"])
        total_dos_values.append(
            integrate_dos(results["energies"], history_row["dos"])
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 5))
    plt.plot(iterations, total_dos_values, marker="o")
    plt.xlabel("iteration")
    plt.ylabel("integrated total DOS (states)")
    plt.title(
        "Integrated total DOS vs iteration "
        f"(D0 = {d0:g} eV^2, alpha = {alpha:g})"
    )
    plt.grid(True, linewidth=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def print_summary(args, results, trace_index):
    print("inputs:")
    print(f"sequence: {args.sequence.upper()}")
    print(f"Gamma_L: {args.gamma_left:g} eV")
    print(f"Gamma_R: {args.gamma_right:g} eV")
    print(f"D0: {args.d0:g} eV^2")
    print(f"alpha: {args.alpha:g}")
    print(f"energy range: {args.energy_min:g} to {args.energy_max:g} eV")
    print(f"energy step: {args.energy_step:g} eV")
    print(f"trace energy: {results['energies'][trace_index]:.3f} eV")
    print()
    print("self-consistent solver:")
    print(f"converged: {results['converged']}")
    print(f"iterations: {results['iterations']}")
    print(f"probe partitions: {results['probe_gammas'].shape[1]}")


def main():
    args = parse_args()
    validate_args(args)
    sequence = validate_ggggg_sequence(args.sequence)
    energies = build_energy_grid(args.energy_min, args.energy_max, args.energy_step)
    hamiltonian, orbitals = build_full_hamiltonian(sequence)
    contact_setup = build_partitioned_contact_setup(
        orbitals,
        args.gamma_left,
        args.gamma_right,
    )
    results = run_dos_weighted_solver(
        hamiltonian=hamiltonian,
        contact_setup=contact_setup,
        energies=energies,
        d0=args.d0,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        alpha=args.alpha,
    )
    output_dir = Path(args.output_dir)
    trace_index = get_trace_index(energies, args.trace_energy)
    frontier_values = get_frontier_values(hamiltonian)

    write_basis(output_dir / "basis.csv", orbitals)
    write_partitions(output_dir / "probe_partitions.csv", contact_setup)
    write_contacts(output_dir / "contacts.csv", contact_setup)
    write_hamiltonian(output_dir / "hamiltonian.csv", hamiltonian)
    write_eigenvalues(output_dir / "eigenvalues.csv", hamiltonian)
    write_transmission(output_dir / "transmission.csv", results)
    write_iteration_history(output_dir / "iteration_history.csv", results)
    write_total_dos_history(output_dir / "total_dos_vs_iteration.csv", results)
    write_probe_gamma_history(
        output_dir / "probe_gamma_history.csv",
        results,
        contact_setup["probe_partitions"],
    )
    write_trace_probe_gammas(
        output_dir / "probe_gamma_trace.csv",
        results,
        contact_setup["probe_partitions"],
        trace_index,
    )
    plot_transmission(
        output_dir / "transmission_log.png",
        results,
        frontier_values,
        args.d0,
        args.alpha,
        "log",
    )
    plot_transmission(
        output_dir / "transmission_linear.png",
        results,
        frontier_values,
        args.d0,
        args.alpha,
        "linear",
    )
    plot_residue_loss(
        output_dir / "loss_per_residue_vs_iteration.png",
        results,
        contact_setup["probe_partitions"],
        trace_index,
        args.d0,
        args.alpha,
    )
    plot_total_dos_history(
        output_dir / "total_dos_vs_iteration.png",
        results,
        args.d0,
        args.alpha,
    )
    print_summary(args, results, trace_index)


if __name__ == "__main__":
    main()

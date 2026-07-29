import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from reference.ggggg.dos_weighted_solver import run_dos_weighted_solver
from reference.ggggg.full_orbital_model import (
    apply_linear_bias,
    build_full_hamiltonian,
    build_partitioned_contact_setup,
    validate_ggggg_sequence,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default="GGGGG")
    parser.add_argument(
        "--bias-voltages",
        type=float,
        nargs=3,
        default=[-0.5, 0.0, 0.5],
    )
    parser.add_argument("--gamma-left", type=float, default=1.0)
    parser.add_argument("--gamma-right", type=float, default=1.0)
    parser.add_argument("--d0", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=250)
    parser.add_argument("--energy-min", type=float, default=-10.0)
    parser.add_argument("--energy-max", type=float, default=10.0)
    parser.add_argument("--energy-step", type=float, default=0.01)
    parser.add_argument("--trace-energy", type=float, default=-4.0)
    parser.add_argument(
        "--output-dir",
        default="outputs/ggggg_bias_voltage_comparison",
    )
    return parser.parse_args()


def validate_args(args):
    validate_ggggg_sequence(args.sequence)

    for bias_voltage in args.bias_voltages:
        if not np.isfinite(bias_voltage):
            raise ValueError("bias voltages must be finite")

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

    for point_index in range(point_count):
        energies[point_index] = energy_min + point_index * energy_step

    return energies


def calculate_maximum_change(values):
    if values is None:
        return None

    maximum = 0.0

    for value in values:
        if value > maximum:
            maximum = value

    return float(maximum)


def integrate_dos(energies, dos):
    total = 0.0

    for energy_index in range(len(energies) - 1):
        energy_step = energies[energy_index + 1] - energies[energy_index]
        average_dos = (dos[energy_index] + dos[energy_index + 1]) / 2
        total = total + average_dos * energy_step

    return total


def get_trace_index(energies, trace_energy):
    differences = abs(energies - trace_energy)
    return int(np.argmin(differences))


def build_probe_labels(probe_partitions):
    labels = []

    for partition in probe_partitions:
        label = f"bp{partition.base_pair} s{partition.strand} {partition.base}"
        labels.append(label)

    return labels


def build_voltage_file_label(bias_voltage):
    label = f"{bias_voltage:g}"
    label = label.replace("-", "minus_")
    label = label.replace(".", "p")
    return label


def run_bias_case(
    hamiltonian,
    orbitals,
    contact_setup,
    energies,
    bias_voltage,
    d0,
    tolerance,
    max_iterations,
    alpha,
):
    biased_hamiltonian, shifts = apply_linear_bias(
        hamiltonian,
        orbitals,
        bias_voltage,
    )
    results = run_dos_weighted_solver(
        hamiltonian=biased_hamiltonian,
        contact_setup=contact_setup,
        energies=energies,
        d0=d0,
        tolerance=tolerance,
        max_iterations=max_iterations,
        alpha=alpha,
    )

    if not results["converged"]:
        raise ValueError(
            f"bias voltage {bias_voltage:g} V did not converge within "
            f"{max_iterations} iterations"
        )

    return {
        "bias_voltage": bias_voltage,
        "shifts": shifts,
        "results": results,
    }


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def write_bias_shifts(path, cases):
    rows = []

    for case in cases:
        for base_pair in sorted(case["shifts"]):
            rows.append(
                {
                    "bias_voltage_v": case["bias_voltage"],
                    "base_pair": base_pair,
                    "onsite_shift_ev": case["shifts"][base_pair],
                }
            )

    write_csv(path, ["bias_voltage_v", "base_pair", "onsite_shift_ev"], rows)


def write_transmission(path, cases, energies):
    rows = []

    for energy_index in range(len(energies)):
        row = {"energy_ev": energies[energy_index]}

        for case in cases:
            voltage = case["bias_voltage"]
            results = case["results"]
            label = f"{voltage:g}_v"
            row[f"coherent_{label}"] = results["history"][0]["t_lr"][energy_index]
            row[f"decoherent_{label}"] = results["t_eff"][energy_index]

        rows.append(row)

    fieldnames = ["energy_ev"]

    for case in cases:
        voltage = case["bias_voltage"]
        label = f"{voltage:g}_v"
        fieldnames.append(f"coherent_{label}")
        fieldnames.append(f"decoherent_{label}")

    write_csv(path, fieldnames, rows)


def write_summary(path, cases, energies):
    rows = []

    for case in cases:
        results = case["results"]
        right_base_pair = len(case["shifts"])
        rows.append(
            {
                "bias_voltage_v": case["bias_voltage"],
                "right_onsite_shift_ev": case["shifts"][right_base_pair],
                "converged": results["converged"],
                "iterations": results["iterations"],
                "integrated_converged_dos_states": integrate_dos(
                    energies,
                    results["dos"],
                ),
                "max_dos_change_percent": calculate_maximum_change(
                    results["dos_changes"]
                ),
                "max_transmission_change_percent": calculate_maximum_change(
                    results["transmission_changes"]
                ),
            }
        )

    write_csv(
        path,
        [
            "bias_voltage_v",
            "right_onsite_shift_ev",
            "converged",
            "iterations",
            "integrated_converged_dos_states",
            "max_dos_change_percent",
            "max_transmission_change_percent",
        ],
        rows,
    )


def write_trace_self_energy(path, cases, probe_partitions, trace_index, energies):
    rows = []

    for case in cases:
        results = case["results"]

        for history_row in results["history"]:
            for partition_index in range(len(probe_partitions)):
                partition = probe_partitions[partition_index]
                rows.append(
                    {
                        "bias_voltage_v": case["bias_voltage"],
                        "iteration": history_row["iteration"],
                        "energy_ev": energies[trace_index],
                        "base_pair": partition.base_pair,
                        "strand": partition.strand,
                        "base": partition.base,
                        "gamma_b_ev": history_row["probe_gammas"][
                            trace_index,
                            partition_index
                        ],
                        "real_sigma_d_ev": history_row["probe_self_energy_real"][
                            trace_index,
                            partition_index
                        ],
                        "imaginary_sigma_d_ev": history_row[
                            "probe_self_energy_imaginary"
                        ][trace_index, partition_index],
                    }
                )

    write_csv(
        path,
        [
            "bias_voltage_v",
            "iteration",
            "energy_ev",
            "base_pair",
            "strand",
            "base",
            "gamma_b_ev",
            "real_sigma_d_ev",
            "imaginary_sigma_d_ev",
        ],
        rows,
    )


def write_converged_probe_gamma_spectra(path, cases, probe_partitions, energies):
    rows = []

    for case in cases:
        gamma_values = case["results"]["probe_gammas"]

        for energy_index in range(len(energies)):
            for partition_index in range(len(probe_partitions)):
                partition = probe_partitions[partition_index]
                rows.append(
                    {
                        "bias_voltage_v": case["bias_voltage"],
                        "energy_ev": energies[energy_index],
                        "base_pair": partition.base_pair,
                        "strand": partition.strand,
                        "base": partition.base,
                        "gamma_b_ev": gamma_values[
                            energy_index,
                            partition_index
                        ],
                    }
                )

    write_csv(
        path,
        [
            "bias_voltage_v",
            "energy_ev",
            "base_pair",
            "strand",
            "base",
            "gamma_b_ev",
        ],
        rows,
    )


def plot_transmission(path, cases, energies, d0, mode, y_scale):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 5))

    for case in cases:
        voltage = case["bias_voltage"]
        results = case["results"]

        if mode == "coherent":
            values = results["history"][0]["t_lr"]
        else:
            values = results["t_eff"]

        plot_energies = []
        plot_values = []

        for energy_index in range(len(energies)):
            value = values[energy_index]

            if y_scale == "log" and value <= 0:
                continue

            plot_energies.append(energies[energy_index])
            plot_values.append(value)

        plt.plot(plot_energies, plot_values, label=f"Vbias = {voltage:g} V")

    if y_scale == "log":
        plt.yscale("log")
        y_label = "transmission (unitless, log y-axis)"
        scale_label = "log scale"
    else:
        y_label = "transmission (unitless, linear y-axis)"
        scale_label = "linear scale"

    if mode == "coherent":
        title = f"GGGGG 20-orbital coherent transmission ({scale_label})"
    else:
        title = (
            "GGGGG 20-orbital DOS-weighted transmission "
            f"(D0 = {d0:g} eV^2, {scale_label})"
        )

    plt.xlabel("energy (eV)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", linewidth=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_probe_gamma_spectrum(path, cases, energies, partition, partition_index):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.5))

    for case in cases:
        values = case["results"]["probe_gammas"][:, partition_index]
        voltage = case["bias_voltage"]
        plt.plot(energies, values, label=f"Vbias = {voltage:g} V")

    label = f"bp{partition.base_pair} strand{partition.strand} {partition.base}"
    plt.xlabel("energy (eV)")
    plt.ylabel("converged Gamma_B (eV)")
    plt.title(f"Converged Gamma_B versus energy: {label}")
    plt.ylim(bottom=0)
    plt.grid(True, linewidth=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_all_probe_gamma_spectra(path, cases, energies, probe_partitions):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    flat_axes = axes.flatten()

    for partition_index in range(len(probe_partitions)):
        partition = probe_partitions[partition_index]
        axis = flat_axes[partition_index]

        for case in cases:
            values = case["results"]["probe_gammas"][:, partition_index]
            voltage = case["bias_voltage"]
            axis.plot(energies, values, label=f"Vbias = {voltage:g} V")

        label = f"bp{partition.base_pair} strand{partition.strand} {partition.base}"
        axis.set_title(label)
        axis.set_ylabel("Gamma_B (eV)")
        axis.set_ylim(bottom=0)
        axis.grid(True, linewidth=0.4)
        axis.legend(fontsize="small")

    for axis in flat_axes[-2:]:
        axis.set_xlabel("energy (eV)")

    figure.suptitle("Converged Gamma_B versus energy for internal bases")
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_converged_probe_gammas(
    path,
    cases,
    probe_partitions,
    trace_index,
    trace_energy,
):
    labels = build_probe_labels(probe_partitions)
    positions = np.arange(len(probe_partitions))
    width = 0.23
    offsets = [-width, 0.0, width]

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9.5, 5))

    for case_index in range(len(cases)):
        case = cases[case_index]
        gamma_values = case["results"]["probe_gammas"][trace_index]
        bar_positions = positions + offsets[case_index]
        plt.bar(
            bar_positions,
            gamma_values,
            width=width,
            label=f"Vbias = {case['bias_voltage']:g} V",
        )

    plt.xlabel("probe site")
    plt.ylabel("converged Gamma_B (eV, log y-axis)")
    plt.title(
        "Converged Gamma_B by site "
        f"at E = {trace_energy:.3f} eV (log y-axis)"
    )
    plt.xticks(positions, labels, rotation=35, ha="right")
    plt.yscale("log")
    plt.grid(True, axis="y", which="both", linewidth=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_probe_gamma_history(
    path,
    case,
    probe_partitions,
    trace_index,
    trace_energy,
):
    results = case["results"]
    labels = build_probe_labels(probe_partitions)
    iterations = []

    for history_row in results["history"]:
        iterations.append(history_row["iteration"])

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 5))

    for partition_index in range(len(probe_partitions)):
        values = []

        for history_row in results["history"]:
            values.append(
                history_row["probe_gammas"][trace_index, partition_index]
            )

        plt.plot(iterations, values, marker="o", label=labels[partition_index])

    plt.xlabel("iteration")
    plt.ylabel("Gamma_B (eV)")
    plt.title(
        "Gamma_B by iteration "
        f"(Vbias = {case['bias_voltage']:g} V, E = {trace_energy:.3f} eV)"
    )
    plt.grid(True, linewidth=0.4)
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_probe_self_energy_history(
    path,
    case,
    probe_partitions,
    trace_index,
    trace_energy,
):
    results = case["results"]
    labels = build_probe_labels(probe_partitions)
    iterations = []

    for history_row in results["history"]:
        iterations.append(history_row["iteration"])

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 2, figsize=(12, 10))
    flat_axes = axes.flatten()

    for partition_index in range(len(probe_partitions)):
        real_values = []
        imaginary_values = []
        axis = flat_axes[partition_index]

        for history_row in results["history"]:
            real_values.append(
                history_row["probe_self_energy_real"][
                    trace_index,
                    partition_index
                ]
            )
            imaginary_values.append(
                history_row["probe_self_energy_imaginary"][
                    trace_index,
                    partition_index
                ]
            )

        axis.plot(
            iterations,
            real_values,
            marker="o",
            label="Re[Sigma_D]",
        )
        axis.plot(
            iterations,
            imaginary_values,
            marker="o",
            label="Im[Sigma_D]",
        )
        axis.set_title(labels[partition_index])
        axis.set_xlabel("iteration")
        axis.set_ylabel("self-energy (eV)")
        axis.grid(True, linewidth=0.4)
        axis.legend(fontsize="small")

    figure.suptitle(
        "Real and imaginary probe self-energy by iteration "
        f"(Vbias = {case['bias_voltage']:g} V, E = {trace_energy:.3f} eV)"
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main():
    args = parse_args()
    validate_args(args)
    sequence = validate_ggggg_sequence(args.sequence)
    hamiltonian, orbitals = build_full_hamiltonian(sequence)
    contact_setup = build_partitioned_contact_setup(
        orbitals,
        args.gamma_left,
        args.gamma_right,
    )
    energies = build_energy_grid(
        args.energy_min,
        args.energy_max,
        args.energy_step,
    )
    trace_index = get_trace_index(energies, args.trace_energy)
    trace_energy = energies[trace_index]
    cases = []

    for bias_voltage in args.bias_voltages:
        case = run_bias_case(
            hamiltonian,
            orbitals,
            contact_setup,
            energies,
            bias_voltage,
            args.d0,
            args.tolerance,
            args.max_iterations,
            args.alpha,
        )
        cases.append(case)

    output_dir = Path(args.output_dir)
    write_bias_shifts(output_dir / "onsite_bias_shifts.csv", cases)
    write_transmission(output_dir / "transmission_vs_energy.csv", cases, energies)
    write_summary(output_dir / "solver_summary.csv", cases, energies)
    write_trace_self_energy(
        output_dir / "probe_self_energy_trace.csv",
        cases,
        contact_setup["probe_partitions"],
        trace_index,
        energies,
    )
    write_converged_probe_gamma_spectra(
        output_dir / "converged_gamma_b_vs_energy.csv",
        cases,
        contact_setup["probe_partitions"],
        energies,
    )

    for mode in ["coherent", "decoherent"]:
        plot_transmission(
            output_dir / f"{mode}_transmission_log.png",
            cases,
            energies,
            args.d0,
            mode,
            "log",
        )
        plot_transmission(
            output_dir / f"{mode}_transmission_linear.png",
            cases,
            energies,
            args.d0,
            mode,
            "linear",
        )

    plot_converged_probe_gammas(
        output_dir / "converged_gamma_b_by_site.png",
        cases,
        contact_setup["probe_partitions"],
        trace_index,
        trace_energy,
    )
    plot_all_probe_gamma_spectra(
        output_dir / "converged_gamma_b_vs_energy_all_sites.png",
        cases,
        energies,
        contact_setup["probe_partitions"],
    )

    for partition_index in range(len(contact_setup["probe_partitions"])):
        partition = contact_setup["probe_partitions"][partition_index]
        file_name = (
            f"converged_gamma_b_vs_energy_bp{partition.base_pair}_"
            f"s{partition.strand}_{partition.base.lower()}.png"
        )
        plot_probe_gamma_spectrum(
            output_dir / file_name,
            cases,
            energies,
            partition,
            partition_index,
        )

    for case in cases:
        voltage_label = build_voltage_file_label(case["bias_voltage"])
        plot_probe_gamma_history(
            output_dir / f"gamma_b_vs_iteration_{voltage_label}_v.png",
            case,
            contact_setup["probe_partitions"],
            trace_index,
            trace_energy,
        )
        plot_probe_self_energy_history(
            output_dir / f"probe_self_energy_vs_iteration_{voltage_label}_v.png",
            case,
            contact_setup["probe_partitions"],
            trace_index,
            trace_energy,
        )

    print(f"wrote {output_dir}")

    for case in cases:
        results = case["results"]
        print(
            f"Vbias = {case['bias_voltage']:g} V: "
            f"converged in {results['iterations']} iterations"
        )


if __name__ == "__main__":
    main()

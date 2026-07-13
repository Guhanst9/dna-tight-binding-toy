from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

def finish_plot(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_transmission(path, rows, transmission_key, title):
    energies = []
    transmissions = []

    for row in rows:
        transmission = row[transmission_key]

        if transmission > 0:
            energies.append(row["energy"])
            transmissions.append(transmission)

    plt.figure(figsize=(7, 4.5))
    plt.plot(energies, transmissions)
    plt.yscale("log")
    plt.xlabel("energy (eV)")
    plt.ylabel("transmission (unitless, log y-axis)")
    plt.title(f"{title} (log y-axis)")
    plt.grid(True, which="both", linewidth=0.4)
    finish_plot(path)

def plot_coherent_transmission(path, rows):
    plot_transmission(
        path,
        rows,
        "coherent_t_lr",
        "coherent transmission vs energy",
    )

def plot_decoherent_transmission(path, rows):
    plot_transmission(
        path,
        rows,
        "decoherent_t_eff",
        "decoherent transmission vs energy",
    )

def plot_gamma_by_iteration(path, orbitals, probe_indices, history):
    plt.figure(figsize=(8, 5))

    for position in range(len(probe_indices)):
        probe_index = probe_indices[position]
        orbital = orbitals[probe_index]
        iterations = []
        gamma_values = []

        for history_row in history:
            iterations.append(history_row["iteration"])
            gamma_values.append(history_row["probe_gammas"][position])

        label = (
            f"bp{orbital.base_pair} "
            f"s{orbital.strand} "
            f"{orbital.base}"
        )
        plt.plot(iterations, gamma_values, label=label)

    plt.xlabel("iteration")
    plt.yscale("symlog", linthresh=1e-12)
    plt.ylabel("Gamma_B (eV, symlog y-axis)")
    plt.title("probe Gamma_B vs iteration (symlog y-axis)")
    plt.grid(True, which="both", linewidth=0.4)
    plt.legend(fontsize="x-small", ncol=2)
    finish_plot(path)

def plot_gamma_by_residue(path, gamma_rows):
    labels = []
    gamma_values = []

    for row in gamma_rows:
        label = (
            f"bp{row['base_pair']} "
            f"s{row['strand']} "
            f"{row['base']}"
        )
        labels.append(label)
        gamma_values.append(row["gamma_b"])

    positions = list(range(len(labels)))

    plt.figure(figsize=(8, 4.5))
    plt.bar(positions, gamma_values)
    plt.xticks(positions, labels, rotation=45, ha="right")
    plt.xlabel("residue")
    plt.ylabel("Gamma_B (eV, log y-axis)")
    plt.title("converged Gamma_B by residue (log y-axis)")
    plt.yscale("log")
    plt.grid(True, axis="y", which="both", linewidth=0.4)
    finish_plot(path)

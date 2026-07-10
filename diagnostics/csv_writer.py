import csv
from pathlib import Path

def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

def write_eigenvalues(path, eigenvalues):
    rows = []

    for index in range(len(eigenvalues)):
        row = {
            "index": index,
            "energy_ev": eigenvalues[index],
        }
        rows.append(row)

    write_csv(path, ["index", "energy_ev"], rows)

def write_transmission_sweep(path, rows):
    fieldnames = [
        "energy_ev",
        "coherent_t_lr",
        "decoherent_t_eff",
        "converged",
        "iterations",
    ]
    csv_rows = []

    for row in rows:
        csv_row = {
            "energy_ev": row["energy"],
            "coherent_t_lr": row["coherent_t_lr"],
            "decoherent_t_eff": row["decoherent_t_eff"],
            "converged": row["converged"],
            "iterations": row["iterations"],
        }
        csv_rows.append(csv_row)

    write_csv(path, fieldnames, csv_rows)

def write_gamma_by_iteration(path, orbitals, probe_indices, history):
    fieldnames = [
        "iteration",
        "probe_index",
        "base_pair",
        "strand",
        "base",
        "gamma_b_ev",
    ]
    rows = []

    for history_row in history:
        for position in range(len(probe_indices)):
            probe_index = probe_indices[position]
            orbital = orbitals[probe_index]
            gamma = history_row["probe_gammas"][position]
            row = {
                "iteration": history_row["iteration"],
                "probe_index": probe_index,
                "base_pair": orbital.base_pair,
                "strand": orbital.strand,
                "base": orbital.base,
                "gamma_b_ev": gamma,
            }
            rows.append(row)

    write_csv(path, fieldnames, rows)

def write_gamma_by_residue(path, rows):
    fieldnames = [
        "probe_index",
        "base_pair",
        "strand",
        "base",
        "gamma_b_ev",
    ]
    csv_rows = []

    for row in rows:
        csv_row = {
            "probe_index": row["probe_index"],
            "base_pair": row["base_pair"],
            "strand": row["strand"],
            "base": row["base"],
            "gamma_b_ev": row["gamma_b"],
        }
        csv_rows.append(csv_row)

    write_csv(path, fieldnames, csv_rows)

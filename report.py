import argparse
from pathlib import Path

from diagnostics.analysis import (
    calculate_eigenvalues,
    get_converged_probe_gammas,
    run_transmission_sweep,
    validate_trace_energy,
)

from diagnostics.csv_writer import (
    write_eigenvalues,
    write_gamma_by_iteration,
    write_gamma_by_residue,
    write_transmission_sweep,
)

from src.builder import build_hamiltonian
from src.contacts import build_contact_setup
from src.solver import run_self_consistent_solver

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=["gc", "at"], required=True)
    parser.add_argument("--band", choices=["homo", "lumo"], required=True)
    parser.add_argument("--base-pairs", type=int, default=10)
    parser.add_argument("--gamma-left", type=float, default=0.5)
    parser.add_argument("--gamma-right", type=float, default=0.5)
    parser.add_argument("--d0", type=float, default=0.01)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--energy-min", type=float, required=True)
    parser.add_argument("--energy-max", type=float, required=True)
    parser.add_argument("--energy-points", type=int, required=True)
    parser.add_argument("--trace-energy", type=float, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()

def print_written_file(path):
    print(f"wrote {path}")

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    try:
        validate_trace_energy(args.trace_energy)
        hamiltonian, orbitals = build_hamiltonian(
            pair=args.pair,
            band=args.band,
            base_pair_count=args.base_pairs,
        )
        contact_setup = build_contact_setup(
            orbitals=orbitals,
            base_pair_count=args.base_pairs,
            gamma_left=args.gamma_left,
            gamma_right=args.gamma_right,
        )
        eigenvalues = calculate_eigenvalues(hamiltonian)
        transmission_rows = run_transmission_sweep(
            hamiltonian=hamiltonian,
            contact_setup=contact_setup,
            energy_min=args.energy_min,
            energy_max=args.energy_max,
            energy_points=args.energy_points,
            d0=args.d0,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
            alpha=args.alpha,
        )
        trace_results = run_self_consistent_solver(
            hamiltonian=hamiltonian,
            contact_setup=contact_setup,
            energy=args.trace_energy,
            d0=args.d0,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
            alpha=args.alpha,
        )
        gamma_rows = get_converged_probe_gammas(
            orbitals,
            contact_setup,
            trace_results,
        )
        
    except ValueError as error:
        raise SystemExit(f"error: {error}")

    eigenvalues_path = output_dir / "eigenvalues.csv"
    transmission_path = output_dir / "transmission_vs_energy.csv"
    gamma_iteration_path = output_dir / "gamma_by_iteration.csv"
    gamma_residue_path = output_dir / "gamma_by_residue.csv"

    write_eigenvalues(eigenvalues_path, eigenvalues)
    write_transmission_sweep(transmission_path, transmission_rows)
    write_gamma_by_iteration(
        gamma_iteration_path,
        orbitals,
        contact_setup["probe_indices"],
        trace_results["history"],
    )
    write_gamma_by_residue(gamma_residue_path, gamma_rows)

    print_written_file(eigenvalues_path)
    print_written_file(transmission_path)
    print_written_file(gamma_iteration_path)
    print_written_file(gamma_residue_path)

if __name__ == "__main__":
    main()

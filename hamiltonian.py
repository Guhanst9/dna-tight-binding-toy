import argparse

from builder import build_hamiltonian
from contacts import build_contact_setup
from greens import run_coherent_calculation
from printing import (
    print_basis,
    print_coherent_results,
    print_contact_setup,
    print_inputs,
    print_ldos,
    print_matrix,
    print_solver_results,
    print_transport_results,
)
from solver import run_self_consistent_solver

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=["gc", "at"], required=True)
    parser.add_argument("--band", choices=["homo", "lumo"], required=True)
    parser.add_argument("--base-pairs", type=int, default=10)
    parser.add_argument("--gamma-left", type=float, default=0.5)
    parser.add_argument("--gamma-right", type=float, default=0.5)
    parser.add_argument("--energy", type=float, required=True)
    parser.add_argument("--d0", type=float, default=0.01)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--show-basis", action="store_true")
    parser.add_argument("--show-contacts", action="store_true")
    parser.add_argument("--show-ldos", action="store_true")
    parser.add_argument("--show-transport", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()

    try:
        matrix, orbitals = build_hamiltonian(
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
        coherent_results = run_coherent_calculation(
            hamiltonian=matrix,
            contact_setup=contact_setup,
            energy=args.energy,
        )
        solver_results = run_self_consistent_solver(
            hamiltonian=matrix,
            contact_setup=contact_setup,
            energy=args.energy,
            d0=args.d0,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
            alpha=args.alpha,
        )
    except ValueError as error:
        raise SystemExit(f"error: {error}")

    if args.show_basis:
        print_basis(orbitals)

    print_matrix(matrix)
    print()
    print_inputs(args)
    print()

    if args.show_contacts:
        print_contact_setup(contact_setup)
        print()

    print_coherent_results(coherent_results)
    print()
    print_solver_results(solver_results)

    if args.show_ldos:
        print()
        print_ldos(solver_results)

    if args.show_transport:
        print()
        print_transport_results(solver_results["transport_results"])

if __name__ == "__main__":
    main()

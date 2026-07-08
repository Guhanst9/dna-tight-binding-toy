import argparse

from builder import build_hamiltonian
from printing import print_basis, print_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", choices=["gc", "at"], required=True)
    parser.add_argument("--band", choices=["homo", "lumo"], required=True)
    parser.add_argument("--base-pairs", type=int, default=10)
    parser.add_argument("--show-basis", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()

    try:
        matrix, orbitals = build_hamiltonian(
            pair=args.pair,
            band=args.band,
            base_pair_count=args.base_pairs,
        )
    except ValueError as error:
        raise SystemExit(f"error: {error}")

    if args.show_basis:
        print_basis(orbitals)

    print_matrix(matrix)

if __name__ == "__main__":
    main()

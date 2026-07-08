import argparse

import numpy as np

mev_to_ev = 1e-3
table = {
    "G": {"e_h": -4278.0, "e_l": 1137.0, "t_l": 19.0, "t_h": -115.0},
    "C": {"e_h": -6519.0, "e_l": -1065.0, "t_l": -61.0, "t_h": -24.0},
    "A": {"e_h": -5245.0, "e_l": 259.0, "t_l": 24.0, "t_h": 21.0},
    "T": {"e_h": -6298.0, "e_l": -931.0, "t_l": -23.0, "t_h": -98.0},
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--band", choices=["homo", "lumo"], required=True)
    parser.add_argument("--sequence", required=True)
    return parser.parse_args()

def onsite(base, band):
    if band == "homo":
        key = "e_h"
    else:
        key = "e_l"
    return table[base][key] * mev_to_ev

def hopping(base, band):
    if band == "homo":
        key = "t_h"
    else:
        key = "t_l"    
    return table[base][key] * mev_to_ev

def build_hamiltonian(sequence, band):
    sequence = sequence.upper()
    if not sequence:
        raise ValueError("sequence must contain at least one base")

    unknown_bases = []
    for base in sequence:
        if base not in table and base not in unknown_bases:
            unknown_bases.append(base)

    if unknown_bases:
        raise ValueError(f"unknown base(s): {', '.join(unknown_bases)}")

    size = len(sequence)
    hamiltonian = np.zeros((size, size))

    for index, base in enumerate(sequence):
        hamiltonian[index, index] = onsite(base, band)

    for index in range(size - 1):
        left = sequence[index]
        right = sequence[index + 1]
        if left != right:
            raise ValueError(
                "table 1 only gives same-base hopping: "
                f"missing {left}-{right} hopping"
            )
        value = hopping(left, band)
        hamiltonian[index, index + 1] = value
        hamiltonian[index + 1, index] = value

    return hamiltonian

def print_matrix(matrix):
    print(np.array2string(matrix, precision=3, suppress_small=True))

def main():
    args = parse_args()
    try:
        matrix = build_hamiltonian(args.sequence, args.band)
    except ValueError as error:
        raise SystemExit(f"error: {error}")
    print_matrix(matrix)

if __name__ == "__main__":
    main()

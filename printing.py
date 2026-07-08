import numpy as np

def print_basis(orbitals):
    print("basis:")
    for orbital in orbitals:
        print(
            f"{orbital.index}: bp{orbital.base_pair} "
            f"strand{orbital.strand} {orbital.base}"
        )

def print_matrix(matrix):
    print("hamiltonian:")
    print(np.array2string(matrix, precision=3, suppress_small=True))

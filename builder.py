import numpy as np

from basis import build_basis
from parameters import (
    get_nearest_cross_hopping,
    get_onsite,
    get_strand_hopping,
    validate_band,
)

def build_hamiltonian(pair, band, base_pair_count):
    pair = pair.lower()
    band = band.lower()
    validate_band(band)

    orbitals = build_basis(pair, base_pair_count)
    size = len(orbitals)
    matrix = np.zeros((size, size))

    for orbital in orbitals:
        matrix[orbital.index, orbital.index] = get_onsite(orbital.base, band)

    add_strand_hopping(matrix, orbitals, band)
    add_nearest_cross_hopping(matrix, orbitals, pair, band)
    check_hermitian(matrix)

    return matrix, orbitals

def check_hermitian(matrix):
    if not np.allclose(matrix, matrix.conj().T):
        raise ValueError("hamiltonian is not hermitian")

def add_strand_hopping(matrix, orbitals, band):
    for index in range(len(orbitals) - 2):
        left = orbitals[index]
        right = orbitals[index + 2]
        if left.strand != right.strand:
            continue
        value = get_strand_hopping(left.base, band)
        matrix[left.index, right.index] = value
        matrix[right.index, left.index] = value

def add_nearest_cross_hopping(matrix, orbitals, pair, band):
    value = get_nearest_cross_hopping(pair, band)

    for index in range(0, len(orbitals), 2):
        first = orbitals[index]
        second = orbitals[index + 1]
        matrix[first.index, second.index] = value
        matrix[second.index, first.index] = value

from dataclasses import dataclass

import numpy as np

from src.contacts import build_contact_self_energy
from .parameters import (
    get_nearest_cross_hopping,
    get_onsite,
    get_strand_hopping,
)


complements = {
    "G": "C",
    "C": "G",
    "A": "T",
    "T": "A",
}


@dataclass(frozen=True)
class FullOrbital:
    index: int
    base_pair: int
    strand: int
    base: str
    band: str


@dataclass(frozen=True)
class BasePartition:
    base_pair: int
    strand: int
    base: str
    orbital_indices: tuple


def validate_ggggg_sequence(sequence):
    normalized_sequence = sequence.upper()

    if normalized_sequence != "GGGGG":
        raise ValueError("only GGGGG is supported for this calculation")

    return normalized_sequence


def build_full_basis(sequence):
    orbitals = []
    bands = ["homo", "lumo"]

    for band in bands:
        for base_pair_index in range(len(sequence)):
            first_base = sequence[base_pair_index]
            second_base = complements[first_base]
            bases = [(1, first_base), (2, second_base)]

            for strand, base in bases:
                orbitals.append(
                    FullOrbital(
                        index=len(orbitals),
                        base_pair=base_pair_index + 1,
                        strand=strand,
                        base=base,
                        band=band,
                    )
                )

    return orbitals


def build_full_hamiltonian(sequence):
    orbitals = build_full_basis(sequence)
    size = len(orbitals)
    hamiltonian = np.zeros((size, size))

    for orbital in orbitals:
        hamiltonian[orbital.index, orbital.index] = get_onsite(
            orbital.base,
            orbital.band,
        )

    add_same_strand_hopping(hamiltonian, orbitals)
    add_same_base_pair_hopping(hamiltonian, orbitals)
    check_hermitian(hamiltonian)

    return hamiltonian, orbitals


def add_same_strand_hopping(hamiltonian, orbitals):
    for left in orbitals:
        for right in orbitals:
            if right.base_pair != left.base_pair + 1:
                continue

            if right.strand != left.strand:
                continue

            if right.band != left.band:
                continue

            value = get_strand_hopping(left.base, left.band)
            hamiltonian[left.index, right.index] = value
            hamiltonian[right.index, left.index] = value


def add_same_base_pair_hopping(hamiltonian, orbitals):
    for left in orbitals:
        for right in orbitals:
            if right.base_pair != left.base_pair:
                continue

            if left.strand != 1 or right.strand != 2:
                continue

            if right.band != left.band:
                continue

            pair = (left.base + right.base).lower()
            value = get_nearest_cross_hopping(pair, left.band)
            hamiltonian[left.index, right.index] = value
            hamiltonian[right.index, left.index] = value


def check_hermitian(matrix):
    if not np.allclose(matrix, matrix.conj().T):
        raise ValueError("hamiltonian is not hermitian")


def calculate_linear_bias_shifts(orbitals, bias_voltage):
    if not np.isfinite(bias_voltage):
        raise ValueError("bias voltage must be finite")

    base_pair_count = 0

    for orbital in orbitals:
        if orbital.base_pair > base_pair_count:
            base_pair_count = orbital.base_pair

    if base_pair_count < 2:
        raise ValueError("at least two base pairs are required for a bias")

    shifts = {}

    for base_pair in range(1, base_pair_count + 1):
        fraction = (base_pair - 1) / (base_pair_count - 1)
        shifts[base_pair] = -bias_voltage * fraction

    return shifts


def apply_linear_bias(hamiltonian, orbitals, bias_voltage):
    if hamiltonian.shape != (len(orbitals), len(orbitals)):
        raise ValueError("hamiltonian shape does not match the orbital basis")

    shifts = calculate_linear_bias_shifts(orbitals, bias_voltage)
    biased_hamiltonian = hamiltonian.astype(complex).copy()

    for orbital in orbitals:
        energy_shift = shifts[orbital.base_pair]
        biased_hamiltonian[orbital.index, orbital.index] = (
            biased_hamiltonian[orbital.index, orbital.index] + energy_shift
        )

    check_hermitian(biased_hamiltonian)

    return biased_hamiltonian, shifts


def build_base_partitions(orbitals):
    partitions_by_base = {}

    for orbital in orbitals:
        key = (orbital.base_pair, orbital.strand)

        if key not in partitions_by_base:
            partitions_by_base[key] = []

        partitions_by_base[key].append(orbital)

    partitions = []

    for key in sorted(partitions_by_base):
        base_orbitals = partitions_by_base[key]
        base_orbitals.sort(key=lambda orbital: orbital.band)
        first_orbital = base_orbitals[0]
        indices = []

        for orbital in base_orbitals:
            indices.append(orbital.index)

        partitions.append(
            BasePartition(
                base_pair=first_orbital.base_pair,
                strand=first_orbital.strand,
                base=first_orbital.base,
                orbital_indices=tuple(indices),
            )
        )

    return partitions


def build_partitioned_contact_setup(orbitals, gamma_left, gamma_right):
    if gamma_left < 0:
        raise ValueError("gamma left must be nonnegative")

    if gamma_right < 0:
        raise ValueError("gamma right must be nonnegative")

    partitions = build_base_partitions(orbitals)
    last_base_pair = 0

    for orbital in orbitals:
        if orbital.base_pair > last_base_pair:
            last_base_pair = orbital.base_pair

    left_indices = []
    right_indices = []
    probe_partitions = []

    for partition in partitions:
        if partition.base_pair == 1:
            for index in partition.orbital_indices:
                left_indices.append(index)
        elif partition.base_pair == last_base_pair:
            for index in partition.orbital_indices:
                right_indices.append(index)
        else:
            probe_partitions.append(partition)

    left_indices.sort()
    right_indices.sort()
    size = len(orbitals)

    return {
        "left_indices": left_indices,
        "right_indices": right_indices,
        "probe_partitions": probe_partitions,
        "sigma_left": build_contact_self_energy(size, left_indices, gamma_left),
        "sigma_right": build_contact_self_energy(size, right_indices, gamma_right),
    }

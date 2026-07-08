import numpy as np

def get_contact_indices(orbitals, base_pair_count):
    left_contact = []
    right_contact = []

    for orbital in orbitals:
        if orbital.base_pair == 1:
            left_contact.append(orbital.index)
        if orbital.base_pair == base_pair_count:
            right_contact.append(orbital.index)

    return left_contact, right_contact

def get_probe_indices(orbitals, base_pair_count):
    probe_indices = []

    for orbital in orbitals:
        if orbital.base_pair != 1 and orbital.base_pair != base_pair_count:
            probe_indices.append(orbital.index)

    return probe_indices

def build_contact_self_energy(size, contact_indices, gamma):
    if gamma < 0:
        raise ValueError("gamma must be nonnegative")

    self_energy = np.zeros((size, size), dtype=complex)

    for index in contact_indices:
        self_energy[index, index] = -1j * gamma / 2

    return self_energy

def build_contact_setup(orbitals, base_pair_count, gamma_left, gamma_right):
    if base_pair_count < 2:
        raise ValueError("base pair count must be at least 2 for contacts")

    left_contact, right_contact = get_contact_indices(orbitals, base_pair_count)
    probe_indices = get_probe_indices(orbitals, base_pair_count)
    size = len(orbitals)

    sigma_left = build_contact_self_energy(size, left_contact, gamma_left)
    sigma_right = build_contact_self_energy(size, right_contact, gamma_right)

    return {
        "left_contact": left_contact,
        "right_contact": right_contact,
        "probe_indices": probe_indices,
        "sigma_left": sigma_left,
        "sigma_right": sigma_right,
    }

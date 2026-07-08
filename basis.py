from dataclasses import dataclass
from parameters import get_base_pair

@dataclass(frozen=True)
class Orbital:
    index: int
    base_pair: int
    strand: int
    base: str

def build_basis(pair, base_pair_count):
    if base_pair_count < 1:
        raise ValueError("base pair count must be at least 1")

    first_base, second_base = get_base_pair(pair)
    orbitals = []

    for base_pair in range(1, base_pair_count + 1):
        orbitals.append(
            Orbital(
                index=len(orbitals),
                base_pair=base_pair,
                strand=1,
                base=first_base,
            )
        )
        orbitals.append(
            Orbital(
                index=len(orbitals),
                base_pair=base_pair,
                strand=2,
                base=second_base,
            )
        )

    return orbitals

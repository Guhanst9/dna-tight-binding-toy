from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat, whosmat


ORBITALS_BY_ELEMENT = {
    "H": 5,
    "C": 15,
    "O": 15,
    "N": 15,
    "P": 19,
    "Hg": 20,
}


class ModelValidationError(ValueError):
    pass


@dataclass(frozen=True)
class AtomRecord:
    serial: int
    atom_name: str
    residue_name: str
    partition_id: int
    element: str


@dataclass(frozen=True)
class AtomOrbitalBlock:
    atom: AtomRecord
    orbital_start: int
    orbital_stop: int

    @property
    def orbital_count(self):
        return self.orbital_stop - self.orbital_start


@dataclass(frozen=True)
class ModelPartition:
    partition_id: int
    residue_names: tuple
    atom_start: int
    atom_stop: int
    atom_count: int
    element_counts: tuple
    orbital_start: int
    orbital_stop: int
    atom_ranges: tuple = ()
    orbital_ranges: tuple = ()
    source_partition_ids: tuple = ()

    def __post_init__(self):
        if len(self.atom_ranges) == 0:
            object.__setattr__(
                self,
                "atom_ranges",
                ((self.atom_start, self.atom_stop),),
            )

        if len(self.orbital_ranges) == 0:
            object.__setattr__(
                self,
                "orbital_ranges",
                ((self.orbital_start, self.orbital_stop),),
            )

        if len(self.source_partition_ids) == 0:
            object.__setattr__(
                self,
                "source_partition_ids",
                (self.partition_id,),
            )

    @property
    def orbital_count(self):
        total = 0

        for orbital_start, orbital_stop in self.orbital_ranges:
            total = total + orbital_stop - orbital_start

        return total


@dataclass(frozen=True)
class LoadedModel:
    hamiltonian: np.ndarray
    atoms: tuple
    atom_blocks: tuple
    partitions: tuple
    variable_name: str
    max_hermitian_error: float
    partition_scheme: str = "pdb"


def infer_element(atom_name, residue_name):
    normalized_atom = atom_name.upper()
    normalized_residue = residue_name.upper()

    if normalized_atom == "HG" and normalized_residue == "HG":
        return "Hg"

    match = re.search(r"[A-Z]", normalized_atom)

    if match is None:
        raise ModelValidationError(
            f"cannot infer an element from atom name {atom_name}"
        )

    element = match.group(0)

    if element not in ORBITALS_BY_ELEMENT:
        raise ModelValidationError(
            f"unsupported element in atom name {atom_name}"
        )

    return element


def parse_pdb(path):
    pdb_path = Path(path)
    atoms = []

    with pdb_path.open("r", encoding="ascii") as pdb_file:
        for line_number, line in enumerate(pdb_file, start=1):
            fields = line.split()

            if len(fields) == 0:
                continue

            if fields[0] != "ATOM" and fields[0] != "HETATM":
                continue

            if len(fields) < 5:
                raise ModelValidationError(
                    f"invalid ATOM record on line {line_number}"
                )

            try:
                serial = int(fields[1])
                partition_id = int(fields[4])
            except ValueError as error:
                raise ModelValidationError(
                    f"invalid atom serial or partition on line {line_number}"
                ) from error

            atom_name = fields[2]
            residue_name = fields[3]
            element = infer_element(atom_name, residue_name)
            atom = AtomRecord(
                serial=serial,
                atom_name=atom_name,
                residue_name=residue_name,
                partition_id=partition_id,
                element=element,
            )
            atoms.append(atom)

    if len(atoms) == 0:
        raise ModelValidationError("the PDB file contains no ATOM records")

    validate_atom_order(atoms)
    return tuple(atoms)


def validate_atom_order(atoms):
    for atom_index in range(len(atoms)):
        expected_serial = atom_index + 1

        if atoms[atom_index].serial != expected_serial:
            raise ModelValidationError(
                "atom serials must be contiguous and remain in file order"
            )


def validate_partition_order(atoms, expected_partition_count):
    partition_ids = []
    closed_partitions = set()
    previous_partition = None

    for atom in atoms:
        current_partition = atom.partition_id

        if current_partition != previous_partition:
            if current_partition in closed_partitions:
                raise ModelValidationError(
                    f"partition {current_partition} is not one atom block"
                )

            if previous_partition is not None:
                closed_partitions.add(previous_partition)

            partition_ids.append(current_partition)
            previous_partition = current_partition

    expected_ids = []

    for partition_id in range(1, expected_partition_count + 1):
        expected_ids.append(partition_id)

    if partition_ids != expected_ids:
        raise ModelValidationError(
            "partition IDs must be contiguous from 1 through "
            f"{expected_partition_count}"
        )


def validate_final_hg(atoms, expected_partition_count):
    hg_atoms = []

    for atom in atoms:
        if atom.element == "Hg":
            hg_atoms.append(atom)

    if len(hg_atoms) != 1:
        raise ModelValidationError("the model must contain exactly one Hg atom")

    final_atom = atoms[-1]

    if final_atom.element != "Hg":
        raise ModelValidationError("Hg must be the final atom")

    if final_atom.partition_id != expected_partition_count:
        raise ModelValidationError("Hg must use the final partition")

    final_partition_atoms = []

    for atom in atoms:
        if atom.partition_id == expected_partition_count:
            final_partition_atoms.append(atom)

    if len(final_partition_atoms) != 1:
        raise ModelValidationError("Hg must be its own partition")


def find_hg_partition_ids(atoms):
    partition_ids = []

    for atom in atoms:
        if atom.element != "Hg":
            continue

        if atom.partition_id not in partition_ids:
            partition_ids.append(atom.partition_id)

    if len(partition_ids) == 0:
        raise ModelValidationError("the model must contain at least one Hg atom")

    return tuple(partition_ids)


def validate_hg_partitions_are_separate(atoms):
    hg_partition_ids = find_hg_partition_ids(atoms)

    for partition_id in hg_partition_ids:
        atoms_in_partition = []

        for atom in atoms:
            if atom.partition_id == partition_id:
                atoms_in_partition.append(atom)

        if len(atoms_in_partition) != 1:
            raise ModelValidationError(
                f"Hg partition {partition_id} is not a single atom"
            )

        if atoms_in_partition[0].element != "Hg":
            raise ModelValidationError(
                f"partition {partition_id} is not an Hg partition"
            )


def build_partition_groups(atoms, expected_partition_count, partition_scheme):
    pdb_groups = []

    for partition_id in range(1, expected_partition_count + 1):
        pdb_groups.append((partition_id,))

    if partition_scheme == "pdb":
        return tuple(pdb_groups)

    hg_partition_ids = find_hg_partition_ids(atoms)
    validate_hg_partitions_are_separate(atoms)

    if partition_scheme == "metals-separate":
        return tuple(pdb_groups)

    dna_partition_ids = []

    for partition_id in range(1, expected_partition_count + 1):
        if partition_id not in hg_partition_ids:
            dna_partition_ids.append(partition_id)

    if partition_scheme == "metals-single":
        groups = []

        for partition_id in dna_partition_ids:
            groups.append((partition_id,))

        groups.append(hg_partition_ids)
        return tuple(groups)

    if partition_scheme == "base-pair":
        if len(dna_partition_ids) != 14:
            raise ModelValidationError(
                "base-pair partitioning expects 14 DNA base partitions"
            )

        groups = []

        for base_pair_index in range(7):
            left_partition_id = base_pair_index + 1
            right_partition_id = 14 - base_pair_index
            group = [left_partition_id, right_partition_id]

            if base_pair_index == 3:
                for hg_partition_id in hg_partition_ids:
                    group.append(hg_partition_id)

            groups.append(tuple(group))

        return tuple(groups)

    raise ModelValidationError(f"unknown partition scheme {partition_scheme}")


def build_atom_blocks(atoms):
    atom_blocks = []
    orbital_start = 0

    for atom in atoms:
        orbital_count = ORBITALS_BY_ELEMENT[atom.element]
        orbital_stop = orbital_start + orbital_count
        block = AtomOrbitalBlock(
            atom=atom,
            orbital_start=orbital_start,
            orbital_stop=orbital_stop,
        )
        atom_blocks.append(block)
        orbital_start = orbital_stop

    return tuple(atom_blocks)


def build_ranges_from_blocks(selected_blocks):
    atom_ranges = []
    orbital_ranges = []

    atom_start = selected_blocks[0].atom.serial
    atom_stop = selected_blocks[0].atom.serial
    orbital_start = selected_blocks[0].orbital_start
    orbital_stop = selected_blocks[0].orbital_stop

    for block in selected_blocks[1:]:
        next_atom = block.atom.serial

        if next_atom == atom_stop + 1:
            atom_stop = next_atom
            orbital_stop = block.orbital_stop
            continue

        atom_ranges.append((atom_start, atom_stop))
        orbital_ranges.append((orbital_start, orbital_stop))
        atom_start = next_atom
        atom_stop = next_atom
        orbital_start = block.orbital_start
        orbital_stop = block.orbital_stop

    atom_ranges.append((atom_start, atom_stop))
    orbital_ranges.append((orbital_start, orbital_stop))
    return tuple(atom_ranges), tuple(orbital_ranges)


def build_partitions_from_groups(atom_blocks, partition_groups):
    partitions = []

    for group_index in range(len(partition_groups)):
        partition_id = group_index + 1
        source_partition_ids = partition_groups[group_index]
        selected_blocks = []

        if len(source_partition_ids) == 0:
            raise ModelValidationError(
                f"partition group {partition_id} has no source partitions"
            )

        if len(set(source_partition_ids)) != len(source_partition_ids):
            raise ModelValidationError(
                f"partition group {partition_id} repeats a source partition"
            )

        for block in atom_blocks:
            if block.atom.partition_id in source_partition_ids:
                selected_blocks.append(block)

        if len(selected_blocks) == 0:
            raise ModelValidationError(
                f"partition group {partition_id} contains no atoms"
            )

        residue_names = []
        element_counts = {}

        for block in selected_blocks:
            residue_name = block.atom.residue_name

            if residue_name not in residue_names:
                residue_names.append(residue_name)

            element = block.atom.element

            if element not in element_counts:
                element_counts[element] = 0

            element_counts[element] = element_counts[element] + 1

        ordered_counts = []

        for element in ORBITALS_BY_ELEMENT:
            if element in element_counts:
                ordered_counts.append((element, element_counts[element]))

        atom_ranges, orbital_ranges = build_ranges_from_blocks(selected_blocks)
        partition = ModelPartition(
            partition_id=partition_id,
            residue_names=tuple(residue_names),
            atom_start=selected_blocks[0].atom.serial,
            atom_stop=selected_blocks[-1].atom.serial,
            atom_count=len(selected_blocks),
            element_counts=tuple(ordered_counts),
            orbital_start=selected_blocks[0].orbital_start,
            orbital_stop=selected_blocks[-1].orbital_stop,
            atom_ranges=atom_ranges,
            orbital_ranges=orbital_ranges,
            source_partition_ids=source_partition_ids,
        )
        partitions.append(partition)

    validate_orbital_ranges(atom_blocks, partitions)
    return tuple(partitions)


def build_partitions(atom_blocks, expected_partition_count):
    partition_groups = []

    for partition_id in range(1, expected_partition_count + 1):
        partition_groups.append((partition_id,))

    return build_partitions_from_groups(atom_blocks, tuple(partition_groups))


def validate_orbital_ranges(atom_blocks, partitions):
    if len(atom_blocks) == 0:
        raise ModelValidationError("the model must contain at least one atom")

    if len(partitions) == 0:
        raise ModelValidationError("the model must contain at least one partition")

    expected_start = 0

    for block in atom_blocks:
        if block.orbital_start != expected_start:
            raise ModelValidationError("atom orbital ranges have a gap or overlap")

        expected_start = block.orbital_stop

    ranges = []
    final_orbital = atom_blocks[-1].orbital_stop

    for partition in partitions:
        if len(partition.orbital_ranges) == 0:
            raise ModelValidationError(
                f"partition {partition.partition_id} has no orbital ranges"
            )

        for orbital_range in partition.orbital_ranges:
            if len(orbital_range) != 2:
                raise ModelValidationError("orbital ranges must have two values")

            orbital_start, orbital_stop = orbital_range

            if orbital_start < 0 or orbital_stop > final_orbital:
                raise ModelValidationError("partition orbital range is out of bounds")

            if orbital_stop <= orbital_start:
                raise ModelValidationError("partition orbital ranges must not be empty")

            ranges.append((orbital_start, orbital_stop))

    ranges.sort()
    expected_start = 0

    for orbital_start, orbital_stop in ranges:
        if orbital_start != expected_start:
            raise ModelValidationError(
                "partition orbital ranges have a gap or overlap"
            )

        expected_start = orbital_stop

    if expected_start != final_orbital:
        raise ModelValidationError(
            "the final partition does not end at the final atom orbital"
        )


def inspect_mat_variables(path, variable_name):
    variables = whosmat(path)
    variable_names = []

    for variable in variables:
        variable_names.append(variable[0])

    if variable_name not in variable_names:
        raise ModelValidationError(
            f"MATLAB variable {variable_name} was not found"
        )

    if len(variable_names) != 1:
        raise ModelValidationError(
            "the MATLAB file must contain only the requested Hamiltonian variable"
        )


def load_hamiltonian(path, variable_name):
    inspect_mat_variables(path, variable_name)
    loaded_data = loadmat(path, variable_names=[variable_name])
    hamiltonian = loaded_data[variable_name]

    if not isinstance(hamiltonian, np.ndarray):
        raise ModelValidationError("the Hamiltonian must be a dense array")

    if hamiltonian.ndim != 2:
        raise ModelValidationError("the Hamiltonian must be two-dimensional")

    row_count = hamiltonian.shape[0]
    column_count = hamiltonian.shape[1]

    if row_count != column_count:
        raise ModelValidationError("the Hamiltonian must be square")

    if not np.issubdtype(hamiltonian.dtype, np.number):
        raise ModelValidationError("the Hamiltonian must contain numeric values")

    if not np.all(np.isfinite(hamiltonian)):
        raise ModelValidationError("the Hamiltonian contains nonfinite values")

    return hamiltonian


def calculate_hermitian_error(matrix, block_size=256):
    size = matrix.shape[0]
    max_error = 0.0

    for row_start in range(0, size, block_size):
        row_stop = min(row_start + block_size, size)

        for column_start in range(0, size, block_size):
            column_stop = min(column_start + block_size, size)
            left_block = matrix[
                row_start:row_stop,
                column_start:column_stop,
            ]
            right_block = matrix[
                column_start:column_stop,
                row_start:row_stop,
            ].conj().T
            block_error = float(np.max(np.abs(left_block - right_block)))

            if block_error > max_error:
                max_error = block_error

    return max_error


def load_model(
    pdb_path,
    hamiltonian_path,
    variable_name,
    expected_partition_count=None,
    hermitian_tolerance=1e-7,
    require_final_hg=True,
    partition_scheme="pdb",
):
    atoms = parse_pdb(pdb_path)

    if expected_partition_count is None:
        expected_partition_count = 0

        for atom in atoms:
            if atom.partition_id > expected_partition_count:
                expected_partition_count = atom.partition_id

    if expected_partition_count < 1:
        raise ModelValidationError("expected partition count must be positive")

    validate_partition_order(atoms, expected_partition_count)

    if require_final_hg:
        validate_final_hg(atoms, expected_partition_count)

    atom_blocks = build_atom_blocks(atoms)
    partition_groups = build_partition_groups(
        atoms,
        expected_partition_count,
        partition_scheme,
    )
    partitions = build_partitions_from_groups(atom_blocks, partition_groups)
    hamiltonian = load_hamiltonian(hamiltonian_path, variable_name)
    orbital_count = atom_blocks[-1].orbital_stop

    if hamiltonian.shape != (orbital_count, orbital_count):
        raise ModelValidationError(
            "Hamiltonian size does not match the PDB orbital count: "
            f"{hamiltonian.shape[0]} != {orbital_count}"
        )

    max_hermitian_error = calculate_hermitian_error(hamiltonian)

    if max_hermitian_error > hermitian_tolerance:
        raise ModelValidationError(
            "Hamiltonian is not Hermitian within tolerance: "
            f"{max_hermitian_error:.6e} > {hermitian_tolerance:.6e}"
        )

    covered_orbitals = 0

    for partition in partitions:
        covered_orbitals = covered_orbitals + partition.orbital_count

    if covered_orbitals != hamiltonian.shape[0]:
        raise ModelValidationError(
            "partition orbitals do not cover the full Hamiltonian"
        )

    return LoadedModel(
        hamiltonian=hamiltonian,
        atoms=atoms,
        atom_blocks=atom_blocks,
        partitions=partitions,
        variable_name=variable_name,
        max_hermitian_error=max_hermitian_error,
        partition_scheme=partition_scheme,
    )


def format_element_counts(element_counts):
    values = []

    for element, count in element_counts:
        values.append(f"{element}:{count}")

    return " ".join(values)

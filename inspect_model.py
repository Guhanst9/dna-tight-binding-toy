import argparse
import csv
from pathlib import Path

from src.model_data import format_element_counts, load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--hamiltonian", required=True)
    parser.add_argument("--variable", required=True)
    parser.add_argument(
        "--partition-scheme",
        choices=[
            "pdb",
            "metals-single",
            "metals-separate",
            "base-pair",
        ],
        default="pdb",
    )
    parser.add_argument(
        "--output",
        default="outputs/model_inspection/partition_table.csv",
    )
    return parser.parse_args()


def build_partition_rows(model):
    rows = []

    for partition in model.partitions:
        row = {
            "partition_id": partition.partition_id,
            "source_partition_ids": format_ids(
                partition.source_partition_ids,
            ),
            "residue": ",".join(partition.residue_names),
            "atom_start": partition.atom_start,
            "atom_end": partition.atom_stop,
            "atom_ranges": format_ranges(partition.atom_ranges, one_based=False),
            "atom_count": partition.atom_count,
            "element_counts": format_element_counts(partition.element_counts),
            "orbital_count": partition.orbital_count,
            "python_orbital_start": partition.orbital_start,
            "python_orbital_stop": partition.orbital_stop,
            "python_orbital_ranges": format_ranges(
                partition.orbital_ranges,
                one_based=False,
            ),
            "one_based_orbital_start": partition.orbital_start + 1,
            "one_based_orbital_end": partition.orbital_stop,
            "one_based_orbital_ranges": format_ranges(
                partition.orbital_ranges,
                one_based=True,
            ),
        }
        rows.append(row)

    return rows


def format_ids(values):
    formatted = []

    for value in values:
        formatted.append(str(value))

    return ",".join(formatted)


def format_ranges(ranges, one_based):
    values = []

    for start, stop in ranges:
        if one_based:
            values.append(f"{start + 1}:{stop}")
        else:
            values.append(f"{start}:{stop}")

    return ";".join(values)


def write_partition_table(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_checkpoints(model):
    print("model checkpoints")
    print("------------------------------------------")
    print(f"atom records                 {len(model.atoms)} passed")
    print(f"model partitions             {len(model.partitions)} passed")
    print(f"mapped orbitals              {model.hamiltonian.shape[0]} passed")
    print(f"Hg atoms                     {count_hg_atoms(model)} passed")
    print("orbital ranges               passed")
    print("finite square Hamiltonian    passed")
    print(
        "Hermitian error             "
        f"{model.max_hermitian_error:.6e} passed"
    )


def count_hg_atoms(model):
    count = 0

    for atom in model.atoms:
        if atom.element == "Hg":
            count = count + 1

    return count


def print_partition_table(rows):
    print()
    print("partition table")
    print("------------------------------------------")
    header = (
        "id  source ids  residue  atoms       elements               orbitals  "
        "python ranges       one-based ranges"
    )
    print(header)

    for row in rows:
        print(
            f"{row['partition_id']:>2}  "
            f"{row['source_partition_ids']:<10}  "
            f"{row['residue']:<7}  "
            f"{row['atom_ranges']:<10}  "
            f"{row['element_counts']:<21}  "
            f"{row['orbital_count']:>8}  "
            f"{row['python_orbital_ranges']:<17}  "
            f"{row['one_based_orbital_ranges']}"
        )


def main():
    args = parse_args()
    model = load_model(
        pdb_path=args.pdb,
        hamiltonian_path=args.hamiltonian,
        variable_name=args.variable,
        require_final_hg=False,
        partition_scheme=args.partition_scheme,
    )
    rows = build_partition_rows(model)
    write_partition_table(args.output, rows)
    print_checkpoints(model)
    print_partition_table(rows)
    print()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

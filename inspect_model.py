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
        "--output",
        default="outputs/model_inspection/partition_table.csv",
    )
    return parser.parse_args()


def build_partition_rows(model):
    rows = []

    for partition in model.partitions:
        row = {
            "partition_id": partition.partition_id,
            "residue": ",".join(partition.residue_names),
            "atom_start": partition.atom_start,
            "atom_end": partition.atom_stop,
            "atom_count": partition.atom_count,
            "element_counts": format_element_counts(partition.element_counts),
            "orbital_count": partition.orbital_count,
            "python_orbital_start": partition.orbital_start,
            "python_orbital_stop": partition.orbital_stop,
            "one_based_orbital_start": partition.orbital_start + 1,
            "one_based_orbital_end": partition.orbital_stop,
        }
        rows.append(row)

    return rows


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
    print(f"contiguous partitions        {len(model.partitions)} passed")
    print(f"mapped orbitals              {model.hamiltonian.shape[0]} passed")
    print("final Hg partition           passed")
    print("orbital ranges               passed")
    print("finite square Hamiltonian    passed")
    print(
        "Hermitian error             "
        f"{model.max_hermitian_error:.6e} passed"
    )


def print_partition_table(rows):
    print()
    print("partition table")
    print("------------------------------------------")
    header = (
        "id  residue  atoms       elements               orbitals  "
        "python range  one-based range"
    )
    print(header)

    for row in rows:
        atom_range = f"{row['atom_start']}:{row['atom_end']}"
        python_range = (
            f"{row['python_orbital_start']}:{row['python_orbital_stop']}"
        )
        one_based_range = (
            f"{row['one_based_orbital_start']}:{row['one_based_orbital_end']}"
        )
        print(
            f"{row['partition_id']:>2}  "
            f"{row['residue']:<7}  "
            f"{atom_range:<10}  "
            f"{row['element_counts']:<21}  "
            f"{row['orbital_count']:>8}  "
            f"{python_range:<12}  "
            f"{one_based_range}"
        )


def main():
    args = parse_args()
    model = load_model(
        pdb_path=args.pdb,
        hamiltonian_path=args.hamiltonian,
        variable_name=args.variable,
    )
    rows = build_partition_rows(model)
    write_partition_table(args.output, rows)
    print_checkpoints(model)
    print_partition_table(rows)
    print()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

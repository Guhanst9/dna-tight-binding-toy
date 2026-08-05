import argparse

from src.energy_solver import ConvergenceError
from src.energy_sweep import (
    SweepConfig,
    build_energy_grid,
    run_energy_sweep,
    validate_index_range,
    validate_sweep_config,
)
from src.model_data import load_model
from src.transport_setup import build_transport_setup
from tqdm import tqdm


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--hamiltonian", required=True)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--mode", choices=["coherent", "decoherent"], required=True)
    parser.add_argument("--left-partition", type=int, default=1)
    parser.add_argument("--right-partition", type=int, default=7)
    parser.add_argument("--gamma-left", type=float, default=1.0)
    parser.add_argument("--gamma-right", type=float, default=1.0)
    parser.add_argument("--energy-min", type=float, default=-10.0)
    parser.add_argument("--energy-max", type=float, default=0.0)
    parser.add_argument("--energy-step", type=float, default=0.01)
    parser.add_argument("--d0", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--stop-index", type=int)
    parser.add_argument(
        "--save-history-index",
        type=int,
        action="append",
        default=[],
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = SweepConfig(
        mode=args.mode,
        energy_minimum=args.energy_min,
        energy_maximum=args.energy_max,
        energy_step=args.energy_step,
        d0=args.d0,
        alpha=args.alpha,
        max_iterations=args.max_iterations,
    )

    try:
        validate_sweep_config(config)
        model = load_model(
            args.pdb,
            args.hamiltonian,
            args.variable,
        )
        setup = build_transport_setup(
            model,
            left_partition_id=args.left_partition,
            right_partition_id=args.right_partition,
            gamma_left=args.gamma_left,
            gamma_right=args.gamma_right,
            energy_minimum=args.energy_min,
            energy_maximum=args.energy_max,
            energy_step=args.energy_step,
        )
        energies = build_energy_grid(
            args.energy_min,
            args.energy_max,
            args.energy_step,
        )
        stop_index = args.stop_index

        if stop_index is None:
            stop_index = len(energies)

        validate_index_range(
            args.start_index,
            stop_index,
            len(energies),
        )
        progress_total = stop_index - args.start_index

        with tqdm(
            total=progress_total,
            desc="Transport",
            unit="point",
            dynamic_ncols=True,
        ) as progress_bar:
            def update_progress(status, energy_index, energy, elapsed_seconds):
                progress_bar.set_postfix_str(f"E={energy:.2f} eV")
                progress_bar.update(1)

            summary = run_energy_sweep(
                model,
                setup,
                config,
                args.output_dir,
                start_index=args.start_index,
                stop_index=stop_index,
                history_indices=tuple(args.save_history_index),
                progress_callback=update_progress,
            )
    except (ValueError, ConvergenceError) as error:
        parser.error(str(error))

    print(f"processed: {summary.processed_count}")
    print(f"skipped: {summary.skipped_count}")
    print(f"energy points: {summary.energy_count}")
    print(f"elapsed seconds: {summary.elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()

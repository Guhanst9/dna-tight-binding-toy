import numpy as np

def print_table(title, rows):
    print(title)
    print("-" * 42)

    for row in rows:
        print(f"{row[0]:<28} {row[1]}")


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


def print_inputs(args):
    rows = [
        ("pair", args.pair.upper()),
        ("band", args.band.upper()),
        ("base pairs", args.base_pairs),
        ("energy (eV)", args.energy),
        ("gamma left (eV)", args.gamma_left),
        ("gamma right (eV)", args.gamma_right),
        ("d0 (eV^2)", args.d0),
        ("tolerance (%)", args.tolerance),
        ("max iterations", args.max_iterations),
        ("alpha (unitless)", args.alpha),
    ]
    print_table("inputs", rows)


def print_contact_setup(contact_setup):
    print("left contact indices:")
    print(contact_setup["left_contact"])
    print("right contact indices:")
    print(contact_setup["right_contact"])
    print("buttiker probe indices:")
    print(contact_setup["probe_indices"])
    print("sigma_l:")
    print(np.array2string(contact_setup["sigma_left"], precision=3, suppress_small=True))
    print("sigma_r:")
    print(np.array2string(contact_setup["sigma_right"], precision=3, suppress_small=True))

def print_coherent_results(results):
    rows = [
        ("coherent dos (1/eV)", f"{results['dos']:.8f}"),
        ("coherent t_lr (unitless)", f"{results['t_lr']:.8e}"),
    ]
    print_table("coherent reference", rows)

def print_ldos(results):
    print("ldos:")
    print(np.array2string(results["ldos"], precision=8, suppress_small=True))

def print_transport_results(results):
    rows = []

    if len(results["active_probe_indices"]) > 0:
        for position in range(len(results["active_probe_indices"])):
            index = results["active_probe_indices"][position]
            gamma = results["active_probe_gammas"][position]
            rows.append((f"({index}, {index})", f"{gamma:.8e} eV"))
    else:
        rows.append(("none active", ""))

    print_table("probe matrix entries", rows)

def print_solver_results(results):
    if results["converged"]:
        converged = "yes"
    else:
        converged = "no"

    active_gammas = results["transport_results"]["active_probe_gammas"]
    rows = [
        ("converged", converged),
        ("iterations", results["iterations"]),
        ("final dos (1/eV)", f"{results['dos']:.8f}"),
        ("final t_lr (unitless)", f"{results['t_lr']:.8e}"),
        ("final t_eff (unitless)", f"{results['t_eff']:.8e}"),
        ("active probes", len(active_gammas)),
    ]

    if results["dos_change_percent"] is not None:
        rows.append(("dos change (%)", f"{results['dos_change_percent']:.8f}"))

    if results["transmission_change_percent"] is not None:
        rows.append(
            (
                "transmission change (%)",
                f"{results['transmission_change_percent']:.8f}",
            )
        )

    print_table("self-consistent result", rows)

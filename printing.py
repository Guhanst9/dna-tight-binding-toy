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
    print("coherent calculation:")
    print(f"dos: {results['dos']:.8f}")
    print(f"t_lr: {results['t_lr']:.8e}")

def print_ldos(results):
    print("ldos:")
    print(np.array2string(results["ldos"], precision=8, suppress_small=True))

def print_transport_results(results):
    print("transport calculation:")
    print(f"active probe count: {len(results['active_probe_indices'])}")
    print(f"t_eff: {results['t_eff']:.8e}")

    if len(results["active_probe_indices"]) > 0:
        print("active probe indices:")
        print(results["active_probe_indices"])
        print("active probe gammas:")
        print(np.array2string(np.array(results["active_probe_gammas"]), precision=8))

def print_solver_results(results):
    print("self-consistent solver:")
    print(f"converged: {results['converged']}")
    print(f"iterations: {results['iterations']}")
    print(f"dos: {results['dos']:.8f}")
    print(f"t_lr: {results['t_lr']:.8e}")
    print(f"t_eff: {results['t_eff']:.8e}")

    if results["dos_change_percent"] is not None:
        print(f"dos change percent: {results['dos_change_percent']:.8f}")

    if results["transmission_change_percent"] is not None:
        print(
            "transmission change percent: "
            f"{results['transmission_change_percent']:.8f}"
        )

    active_gammas = results["transport_results"]["active_probe_gammas"]
    print(f"active probe count: {len(active_gammas)}")

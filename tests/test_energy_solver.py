from dataclasses import replace

import numpy as np
import pytest

from src.energy_solver import (
    CONVERGENCE_PERCENT,
    ConvergenceError,
    build_decoherence_diagonal,
    build_w_matrix,
    calculate_effective_transmission,
    calculate_next_self_energy,
    calculate_partition_values,
    calculate_percent_change,
    calculate_probe_gammas,
    calculate_region_transmission,
    calculate_transport,
    run_coherent_energy,
    run_dos_weighted_energy,
    validate_solver_settings,
)
from src.model_data import LoadedModel, ModelPartition
from src.transport_setup import build_transport_setup


def build_three_site_model():
    hamiltonian = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    partitions = []

    for partition_index in range(3):
        partition_id = partition_index + 1
        partition = ModelPartition(
            partition_id=partition_id,
            residue_names=(f"P{partition_id}",),
            atom_start=partition_id,
            atom_stop=partition_id,
            atom_count=1,
            element_counts=(("H", 1),),
            orbital_start=partition_index,
            orbital_stop=partition_index + 1,
        )
        partitions.append(partition)

    return LoadedModel(
        hamiltonian=hamiltonian,
        atoms=tuple(),
        atom_blocks=tuple(),
        partitions=tuple(partitions),
        variable_name="three_site",
        max_hermitian_error=0.0,
    )


def build_three_site_setup():
    model = build_three_site_model()
    setup = build_transport_setup(
        model,
        left_partition_id=1,
        right_partition_id=3,
        gamma_left=1.0,
        gamma_right=1.0,
    )
    return model, setup


def test_coherent_three_site_transmission_is_one():
    model, setup = build_three_site_setup()
    result = run_coherent_energy(model, setup, 0.0)

    assert result.converged
    assert result.iterations == 1
    assert result.t_lr == pytest.approx(1.0)
    assert result.t_eff == pytest.approx(1.0)
    assert result.dos > 0
    assert result.partition_ldos.shape == (3,)
    assert np.all(result.gamma_decoherence_by_partition == 0)
    assert result.history[0].active_probe_partition_ids == tuple()
    assert result.history[0].dos_change_percent is None
    assert result.history[0].transmission_change_percent is None


def test_partition_ldos_uses_diagonal_green_entries():
    model, setup = build_three_site_setup()
    green_function = np.diag([1.0 - 2.0j, 2.0 - 4.0j, 3.0 - 6.0j])
    green_sums, partition_ldos, dos = calculate_partition_values(
        green_function,
        setup,
    )

    assert np.allclose(green_sums, [1.0 - 2.0j, 2.0 - 4.0j, 3.0 - 6.0j])
    assert np.allclose(partition_ldos, [2.0 / np.pi, 4.0 / np.pi, 6.0 / np.pi])
    assert dos == pytest.approx(12.0 / np.pi)


def test_split_partition_ldos_uses_only_owned_orbitals(interleaved_model):
    setup = build_transport_setup(
        interleaved_model,
        left_partition_id=1,
        right_partition_id=3,
    )
    green_function = np.diag([-1.0j, -2.0j, -3.0j, -4.0j])
    green_sums, partition_ldos, dos = calculate_partition_values(
        green_function,
        setup,
    )

    assert green_sums[0] == pytest.approx(-5.0j)
    assert np.allclose(partition_ldos, [5.0 / np.pi, 2.0 / np.pi, 3.0 / np.pi])
    assert dos == pytest.approx(10.0 / np.pi)


def test_split_partition_decoherence_uses_only_probe_orbitals(
    interleaved_model,
):
    setup = build_transport_setup(
        interleaved_model,
        left_partition_id=1,
        right_partition_id=3,
    )
    sigma_by_partition = np.array([0.0j, -0.05j, 0.0j])
    diagonal = build_decoherence_diagonal(setup, sigma_by_partition)

    assert np.array_equal(diagonal, [0.0j, -0.05j, 0.0j, 0.0j])


def test_split_partition_transmission_uses_only_owned_blocks(
    interleaved_model,
):
    green_function = np.zeros((4, 4), dtype=complex)
    green_function[0, 2] = 1.0
    green_function[3, 2] = 2.0j
    green_function[1, 2] = 100.0
    transmission = calculate_region_transmission(
        green_function,
        interleaved_model.partitions[0],
        2.0,
        interleaved_model.partitions[2],
        3.0,
    )

    assert transmission == pytest.approx(30.0)


def test_first_eq_37_update_uses_zero_for_g0():
    model, setup = build_three_site_setup()
    current = np.array([1.0 - 2.0j, 2.0 - 4.0j, 3.0 - 6.0j])
    previous = np.zeros(3, dtype=complex)
    result = calculate_next_self_energy(
        current,
        previous,
        setup,
        d0=0.1,
        alpha=0.3,
    )
    expected_probe_value = 0.1 / (2 * np.pi) * 0.3 * (2.0 - 4.0j)

    assert result[0] == 0
    assert result[1] == pytest.approx(expected_probe_value)
    assert result[2] == 0


def test_later_eq_37_update_uses_current_and_previous_green_sums():
    model, setup = build_three_site_setup()
    current = np.array([0.0, 4.0 - 8.0j, 0.0])
    previous = np.array([0.0, 2.0 - 4.0j, 0.0])
    result = calculate_next_self_energy(
        current,
        previous,
        setup,
        d0=0.2,
        alpha=0.25,
    )
    mixed = 0.25 * (4.0 - 8.0j) + 0.75 * (2.0 - 4.0j)
    expected_probe_value = 0.2 / (2 * np.pi) * mixed

    assert result[1] == pytest.approx(expected_probe_value)


def test_contact_partitions_keep_zero_decoherence():
    model, setup = build_three_site_setup()
    current = np.array([5.0 - 5.0j, 2.0 - 4.0j, 7.0 - 7.0j])
    previous = current.copy()
    result = calculate_next_self_energy(
        current,
        previous,
        setup,
        d0=1.0,
        alpha=0.5,
    )

    assert result[0] == 0
    assert result[2] == 0
    assert result[1] != 0


def test_probe_gamma_comes_from_imaginary_self_energy():
    model, setup = build_three_site_setup()
    sigma = np.array([0.0, 0.25 - 0.05j, 0.0], dtype=complex)
    gamma = calculate_probe_gammas(sigma, setup)

    assert np.allclose(gamma, [0.0, 0.1, 0.0])


def test_region_transmission_uses_green_block_norm():
    model, setup = build_three_site_setup()
    green_function = np.array(
        [
            [0.0, 2.0 + 1.0j, 3.0 - 2.0j],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=complex,
    )
    transmission = calculate_region_transmission(
        green_function,
        setup.left_partition,
        0.5,
        setup.right_partition,
        0.25,
    )

    assert transmission == pytest.approx(0.5 * 0.25 * 13.0)


def test_w_matrix_uses_all_outgoing_probe_transmissions():
    probe_to_left = np.array([0.2, 0.3])
    probe_to_right = np.array([0.4, 0.5])
    probe_to_probe = np.array(
        [
            [0.0, 0.1],
            [0.15, 0.0],
        ]
    )
    w_matrix = build_w_matrix(
        probe_to_left,
        probe_to_right,
        probe_to_probe,
    )
    expected = np.array(
        [
            [0.7, -0.1],
            [-0.15, 0.95],
        ]
    )

    assert np.allclose(w_matrix, expected)


def test_effective_transmission_solves_probe_network():
    left_to_probe = np.array([0.2])
    probe_to_right = np.array([0.4])
    w_matrix = np.array([[0.6]])
    result = calculate_effective_transmission(
        0.1,
        left_to_probe,
        w_matrix,
        probe_to_right,
    )

    assert result == pytest.approx(0.1 + 0.2 * 0.4 / 0.6)


def test_singular_w_matrix_is_rejected():
    with pytest.raises(ValueError, match="W matrix is singular"):
        calculate_effective_transmission(
            0.1,
            np.array([0.2]),
            np.zeros((1, 1)),
            np.array([0.4]),
        )


def test_percent_change_keeps_paper_definition():
    assert calculate_percent_change(11.0, 10.0) == pytest.approx(10.0)
    assert calculate_percent_change(0.0, 0.0) == 0.0
    assert np.isinf(calculate_percent_change(1.0, 0.0))
    assert CONVERGENCE_PERCENT == 0.1


def test_dos_weighted_solver_converges_and_keeps_history():
    model, setup = build_three_site_setup()
    result = run_dos_weighted_energy(
        model,
        setup,
        energy=0.0,
        d0=0.01,
        alpha=0.3,
        max_iterations=100,
    )

    assert result.converged
    assert result.iterations >= 2
    assert result.dos_change_percent < CONVERGENCE_PERCENT
    assert result.transmission_change_percent < CONVERGENCE_PERCENT
    assert len(result.history) == result.iterations
    assert result.history[0].active_probe_partition_ids == tuple()
    assert result.gamma_decoherence_by_partition[0] == 0
    assert result.gamma_decoherence_by_partition[2] == 0
    assert result.gamma_decoherence_by_partition[1] > 0


def test_zero_d0_converges_to_coherent_result():
    model, setup = build_three_site_setup()
    coherent = run_coherent_energy(model, setup, 0.0)
    result = run_dos_weighted_energy(
        model,
        setup,
        energy=0.0,
        d0=0.0,
        alpha=0.3,
        max_iterations=10,
    )

    assert result.converged
    assert result.iterations == 2
    assert result.dos == pytest.approx(coherent.dos)
    assert result.t_lr == pytest.approx(coherent.t_lr)
    assert result.t_eff == pytest.approx(coherent.t_eff)
    assert np.all(result.gamma_decoherence_by_partition == 0)


def test_nonconvergence_raises_with_final_result():
    model, setup = build_three_site_setup()

    with pytest.raises(ConvergenceError) as error:
        run_dos_weighted_energy(
            model,
            setup,
            energy=0.0,
            d0=1.0,
            alpha=1.0,
            max_iterations=2,
        )

    assert not error.value.result.converged
    assert error.value.result.iterations == 2
    assert len(error.value.result.history) == 2


@pytest.mark.parametrize(
    "d0,alpha,max_iterations",
    [
        (-0.1, 0.5, 10),
        (np.nan, 0.5, 10),
        (0.1, -0.1, 10),
        (0.1, 0.0, 10),
        (0.1, 1.1, 10),
        (0.1, np.inf, 10),
        (0.1, 0.5, 1),
        (0.1, 0.5, 2.5),
    ],
)
def test_invalid_solver_settings_are_rejected(d0, alpha, max_iterations):
    with pytest.raises(ValueError):
        validate_solver_settings(d0, alpha, max_iterations)


def test_negative_partition_ldos_is_rejected():
    model, setup = build_three_site_setup()
    green_function = np.diag([1.0j, -1.0j, -1.0j])

    with pytest.raises(ValueError, match="negative LDOS"):
        calculate_partition_values(green_function, setup)


def test_contact_probe_broadening_is_rejected():
    model, setup = build_three_site_setup()
    bad_sigma = np.array([-0.1j, -0.1j, 0.0], dtype=complex)

    with pytest.raises(ValueError, match="contact partitions"):
        calculate_probe_gammas(bad_sigma, setup)


def test_transport_rejects_negative_probe_broadening():
    model, setup = build_three_site_setup()
    green_function = np.eye(3, dtype=complex)
    gamma = np.array([0.0, -0.1, 0.0])

    with pytest.raises(ValueError, match="must be nonnegative"):
        calculate_transport(green_function, setup, gamma)


def test_result_does_not_store_green_function():
    model, setup = build_three_site_setup()
    result = run_coherent_energy(model, setup, 0.0)

    assert not hasattr(result, "green_function")
    assert not hasattr(result.history[0], "green_function")


def test_decoherence_values_must_match_partition_count():
    model, setup = build_three_site_setup()
    bad_setup = replace(
        setup,
        sigma_decoherence_by_partition=np.zeros(2, dtype=complex),
    )

    with pytest.raises(ValueError, match="shape"):
        calculate_probe_gammas(
            bad_setup.sigma_decoherence_by_partition,
            setup,
        )

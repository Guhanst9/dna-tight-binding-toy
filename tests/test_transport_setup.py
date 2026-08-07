from dataclasses import replace

import numpy as np
import pytest

from src.model_data import load_model
from src.transport_setup import build_transport_setup


def test_small_model_contact_and_probe_setup(small_model_files):
    pdb_path, mat_path = small_model_files
    model = load_model(
        pdb_path,
        mat_path,
        "small",
        expected_partition_count=3,
    )
    setup = build_transport_setup(
        model,
        left_partition_id=1,
        right_partition_id=3,
        gamma_left=0.4,
        gamma_right=0.8,
    )

    assert setup.left_partition.partition_id == 1
    assert setup.right_partition.partition_id == 3
    assert len(setup.probe_partitions) == 1
    assert setup.probe_partitions[0].partition_id == 2
    assert np.all(setup.sigma_left_diagonal[0:5] == -0.2j)
    assert np.all(setup.sigma_left_diagonal[5:40] == 0)
    assert np.all(setup.sigma_right_diagonal[0:20] == 0)
    assert np.all(setup.sigma_right_diagonal[20:40] == -0.4j)
    assert np.array_equal(setup.probe_mask, [False, True, False])
    assert np.all(setup.sigma_decoherence_by_partition == 0)
    assert np.all(setup.gamma_decoherence_by_partition == 0)


def test_default_energy_grid(small_model_files):
    pdb_path, mat_path = small_model_files
    model = load_model(
        pdb_path,
        mat_path,
        "small",
        expected_partition_count=3,
    )
    setup = build_transport_setup(
        model,
        left_partition_id=1,
        right_partition_id=3,
    )

    assert setup.energy_grid.minimum == -10.0
    assert setup.energy_grid.maximum == 0.0
    assert setup.energy_grid.step == 0.01


def test_contacts_use_only_their_split_orbital_ranges(interleaved_model):
    setup = build_transport_setup(
        interleaved_model,
        left_partition_id=1,
        right_partition_id=3,
        gamma_left=0.4,
        gamma_right=0.8,
    )

    assert np.array_equal(
        setup.sigma_left_diagonal,
        [-0.2j, 0.0j, 0.0j, -0.2j],
    )
    assert np.array_equal(
        setup.sigma_right_diagonal,
        [0.0j, 0.0j, -0.4j, 0.0j],
    )
    assert np.array_equal(setup.probe_mask, [False, True, False])


def test_real_model_uses_confirmed_contacts_and_thirteen_probes(repo_root):
    pdb_path = repo_root / "data" / "xx7tg6.pdb"
    mat_path = repo_root / "data" / "xx7tg6.mat"

    if not mat_path.exists():
        pytest.skip("local real Hamiltonian is not available")

    model = load_model(pdb_path, mat_path, "xx7tg6")
    setup = build_transport_setup(model)

    assert setup.left_partition.partition_id == 1
    assert setup.left_partition.orbital_start == 0
    assert setup.left_partition.orbital_stop == 320
    assert setup.right_partition.partition_id == 7
    assert setup.right_partition.orbital_start == 2190
    assert setup.right_partition.orbital_stop == 2584
    assert setup.gamma_left == 1.0
    assert setup.gamma_right == 1.0
    assert len(setup.probe_partitions) == 13

    expected_probe_ids = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
    actual_probe_ids = []

    for partition in setup.probe_partitions:
        actual_probe_ids.append(partition.partition_id)

    assert actual_probe_ids == expected_probe_ids
    assert setup.probe_mask.shape == (15,)
    assert not setup.probe_mask[0]
    assert not setup.probe_mask[6]
    assert int(np.count_nonzero(setup.probe_mask)) == 13
    assert setup.sigma_left_diagonal.shape == (5083,)
    assert setup.sigma_right_diagonal.shape == (5083,)
    assert np.all(setup.sigma_left_diagonal[0:320] == -0.5j)
    assert np.all(setup.sigma_left_diagonal[320:5083] == 0)
    assert np.all(setup.sigma_right_diagonal[0:2190] == 0)
    assert np.all(setup.sigma_right_diagonal[2190:2584] == -0.5j)
    assert np.all(setup.sigma_right_diagonal[2584:5083] == 0)
    assert setup.sigma_decoherence_by_partition.shape == (15,)
    assert setup.gamma_decoherence_by_partition.shape == (15,)


def test_contacts_must_use_different_partitions(small_model_files):
    pdb_path, mat_path = small_model_files
    model = load_model(
        pdb_path,
        mat_path,
        "small",
        expected_partition_count=3,
    )

    with pytest.raises(ValueError, match="different partitions"):
        build_transport_setup(
            model,
            left_partition_id=1,
            right_partition_id=1,
        )


def test_contact_partition_must_exist(small_model_files):
    pdb_path, mat_path = small_model_files
    model = load_model(
        pdb_path,
        mat_path,
        "small",
        expected_partition_count=3,
    )

    with pytest.raises(ValueError, match="partition 4 does not exist"):
        build_transport_setup(
            model,
            left_partition_id=1,
            right_partition_id=4,
        )


@pytest.mark.parametrize("gamma", [-0.1, np.nan, np.inf])
def test_invalid_contact_strength_is_rejected(small_model_files, gamma):
    pdb_path, mat_path = small_model_files
    model = load_model(
        pdb_path,
        mat_path,
        "small",
        expected_partition_count=3,
    )

    with pytest.raises(ValueError, match="gamma left"):
        build_transport_setup(
            model,
            left_partition_id=1,
            right_partition_id=3,
            gamma_left=gamma,
        )


@pytest.mark.parametrize(
    "minimum,maximum,step",
    [
        (0.0, 0.0, 0.01),
        (1.0, 0.0, 0.01),
        (-10.0, 0.0, 0.0),
        (-10.0, 0.0, -0.01),
        (np.nan, 0.0, 0.01),
        (-10.0, np.inf, 0.01),
        (-10.0, 0.0, np.nan),
    ],
)
def test_invalid_energy_grid_is_rejected(
    small_model_files,
    minimum,
    maximum,
    step,
):
    pdb_path, mat_path = small_model_files
    model = load_model(
        pdb_path,
        mat_path,
        "small",
        expected_partition_count=3,
    )

    with pytest.raises(ValueError):
        build_transport_setup(
            model,
            left_partition_id=1,
            right_partition_id=3,
            energy_minimum=minimum,
            energy_maximum=maximum,
            energy_step=step,
        )


def test_partition_ranges_must_cover_the_hamiltonian(small_model_files):
    pdb_path, mat_path = small_model_files
    model = load_model(
        pdb_path,
        mat_path,
        "small",
        expected_partition_count=3,
    )
    bad_partition = replace(
        model.partitions[1],
        orbital_ranges=((6, 20),),
    )
    bad_partitions = list(model.partitions)
    bad_partitions[1] = bad_partition
    bad_model = replace(model, partitions=tuple(bad_partitions))

    with pytest.raises(ValueError, match="gap or overlap"):
        build_transport_setup(
            bad_model,
            left_partition_id=1,
            right_partition_id=3,
        )

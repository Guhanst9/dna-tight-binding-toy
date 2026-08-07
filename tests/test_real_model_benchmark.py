import os
import time

import numpy as np
import pytest

from src.energy_solver import run_coherent_energy
from src.model_data import load_model
from src.transport_setup import build_transport_setup


def test_real_model_coherent_energy_benchmark(repo_root):
    if os.environ.get("RUN_REAL_BENCHMARK") != "1":
        pytest.skip("set RUN_REAL_BENCHMARK=1 to run the dense benchmark")

    pdb_path = repo_root / "data" / "xx7tg6" / "xx7tg6.pdb"
    mat_path = repo_root / "data" / "xx7tg6" / "xx7tg6.mat"

    if not mat_path.exists():
        pytest.skip("local real Hamiltonian is not available")

    model = load_model(pdb_path, mat_path, "xx7tg6")
    setup = build_transport_setup(model)
    start = time.perf_counter()
    result = run_coherent_energy(model, setup, -5.0)
    elapsed_seconds = time.perf_counter() - start

    assert result.converged
    assert result.iterations == 1
    assert result.partition_ldos.shape == (15,)
    assert np.all(np.isfinite(result.partition_ldos))
    assert result.dos == pytest.approx(2.415034636217, rel=1e-8)
    assert result.t_lr == pytest.approx(6.445297588003e-11, rel=1e-8)
    assert result.t_eff == pytest.approx(result.t_lr)
    print(f"real coherent energy completed in {elapsed_seconds:.2f} seconds")


@pytest.mark.parametrize(
    "strand,hamiltonian_size",
    [
        ("xx7tg6", 5083),
        ("xx7sm0", 5103),
    ],
)
def test_real_base_pair_coherent_energy_smoke(
    repo_root,
    strand,
    hamiltonian_size,
):
    if os.environ.get("RUN_REAL_BENCHMARK") != "1":
        pytest.skip("set RUN_REAL_BENCHMARK=1 to run the dense benchmark")

    pdb_path = repo_root / "data" / strand / f"{strand}.pdb"
    mat_path = repo_root / "data" / strand / f"{strand}.mat"

    if not mat_path.exists():
        pytest.skip(f"local {strand} Hamiltonian is not available")

    model = load_model(
        pdb_path,
        mat_path,
        strand,
        require_final_hg=False,
        partition_scheme="base-pair",
    )
    setup = build_transport_setup(model)
    result = run_coherent_energy(model, setup, -5.0)

    assert model.hamiltonian.shape == (hamiltonian_size, hamiltonian_size)
    assert len(model.partitions) == 7
    assert result.converged
    assert result.iterations == 1
    assert result.partition_ldos.shape == (7,)
    assert np.all(np.isfinite(result.partition_ldos))
    assert np.isfinite(result.dos)
    assert np.isfinite(result.t_lr)
    assert result.dos >= 0
    assert result.t_lr >= 0
    assert result.t_eff == pytest.approx(result.t_lr)

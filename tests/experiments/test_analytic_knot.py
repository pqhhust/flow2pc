"""Analytic evaluation-law regressions; execute only with test authorization."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flow2pc.experiments import adapters
from flow2pc.experiments.analytic_knot import contract, parameters, reference, source_banks
from flow2pc.experiments.e3_1.config import resolve
from flow2pc.utils.config import load_config


@pytest.fixture
def dataset():
    times = np.linspace(0, 1, 120).tolist()
    return SimpleNamespace(
        dim=2, params={"size": 120, "noise_std": .01},
        metadata=SimpleNamespace(name="knot", model_times=times),
        preprocessing={"kind": "affine", "shift": [0., 0.], "scale": [1., 1.]},
        contract=lambda: {"initial_law": {"kind": "empirical_earliest_marginal"}},
    )


@pytest.fixture
def settings():
    return resolve({"source_split": "analytic_knot", "reference_split": "analytic_knot",
                    "seeds": [0, 1], "bank_sizes": [10, 20], "tune_samples": 10,
                    "energy_samples": 10, "max_test_samples": 10000,
                    "bandwidth_scale": "model_std"})


def test_fresh_reference_uses_saved_mean_shift_and_noise(dataset, settings):
    means, noise = parameters(dataset)
    values, split, count = reference(dataset, 60, settings)
    assert means[60, 1] > 1.9  # The lifted middle branch reaches y ~= 2.
    assert split == "analytic_knot" and count == 10000
    np.testing.assert_allclose(values.mean(0), means[60], atol=.0006)
    np.testing.assert_allclose(values.std(0), noise, rtol=.05)
    np.testing.assert_array_equal(values, reference(dataset, 60, settings)[0])
    assert not np.array_equal(values - means[60], reference(dataset, 61, settings)[0] - means[61])
    changed = {**settings, "analytic_seed": settings["analytic_seed"] + 1}
    assert not np.array_equal(values, reference(dataset, 60, changed)[0])


def test_source_banks_are_disjoint_fresh_draws_and_keep_model_coordinates(dataset, settings):
    dataset.preprocessing = {"kind": "affine", "shift": [-3., .5], "scale": [2., .25]}
    source, plans, sampling = source_banks(dataset, settings)
    indices = np.concatenate([block for plan in plans.values() for block in plan.values()])
    assert len(np.unique(indices)) == len(source) == len(indices)
    assert sampling["source_resampling_with_replacement"] is False
    first_child = np.random.SeedSequence([settings["analytic_seed"], 1, 0]).spawn(5)[0]
    means, noise = parameters(dataset)
    raw = means[0] + np.random.default_rng(first_child).standard_normal((20, 2)) * noise
    expected = (torch.tensor(raw, dtype=torch.float32) - torch.tensor([-3., .5])) / torch.tensor([2., .25])
    torch.testing.assert_close(source[plans[0]["fit"]], expected)
    assert contract(dataset, settings)["evaluation_initial_law"]["mean_raw"] == means[0].tolist()


def test_analytic_adapter_never_draws_saved_pools_and_shares_particles(dataset, settings, monkeypatch):
    seen = []

    def forbidden(*args, **kwargs):
        raise AssertionError("Analytic evaluation must not resample the saved ten-point source")

    def fake_rollout(model, source, grid, options, *, student=False):
        seen.append(source.clone())
        return torch.stack([source.clone() for _ in grid]), 0

    monkeypatch.setattr(adapters, "population", forbidden)
    monkeypatch.setattr(adapters, "rollout", fake_rollout)
    paths, plans, sampling = adapters.paired_banks(object(), object(), dataset, [40, 80, 119], settings)
    torch.testing.assert_close(seen[0], seen[1])
    assert paths["teacher"].shape[0] == 3
    assert len(plans) == 2 and sampling["source_split"] == "analytic_knot"
    assert adapters.real_population(dataset, 40, settings)[1] == "analytic_knot"


def test_analytic_mode_rejects_wrong_dataset_or_grid(dataset, settings):
    dataset.metadata.name = "gom"
    with pytest.raises(ValueError, match="saved 2D knot"):
        parameters(dataset)
    dataset.metadata.name = "knot"
    dataset.metadata.model_times = [0., 1.]
    with pytest.raises(ValueError, match="grid/noise"):
        parameters(dataset)
    with pytest.raises(ValueError, match="both source_split"):
        resolve({"source_split": "analytic_knot"})


def test_knot_presets_enable_both_analytic_laws_and_fresh_references():
    from flow2pc.utils.paths import REPO_ROOT

    for method in ("alicfm", "otpfm"):
        path = REPO_ROOT / f"configs/experiments/e3_1/{method}/{method}_knot_hclt.yaml"
        config = load_config(path)
        assert config["evaluation"]["source_split"] == "analytic_knot"
        assert config["evaluation"]["reference_split"] == "analytic_knot"
        assert config["evaluation"]["max_test_samples"] == 10000
        assert "knot-analytic-v1" in config["output"]["dir"]
        with pytest.raises(ValueError, match="only for dataset: knot"):
            load_config(path, ["dataset=gom"])

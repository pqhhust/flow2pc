"""Conditional inference/scoring regressions. Execution requires authorization."""

from pathlib import Path

import numpy as np
import pytest

from flow2pc.experiments.e3_2.conditional import ConditionalKDE, masks_for_dimension
from flow2pc.experiments.e3_2.config import resolve
from flow2pc.experiments.e3_2.metrics import aggregate, conditional_scores
from flow2pc.utils.config import load_config, resolve_config

ROOT = Path(__file__).resolve().parents[2]


def test_one_dimensional_beijing_cannot_be_a_missing_coordinate_task():
    with pytest.raises(ValueError, match=">=2"):
        masks_for_dimension([[0]], 1)
    assert masks_for_dimension([[0], [1]], 2) == [([0], [1]), ([1], [0])]
    for invalid in ([[-1]], [[0, 0]], [[2]], [[0, 1]], [[]]):
        with pytest.raises(ValueError, match="mask"):
            masks_for_dimension(invalid, 2)


def test_conditional_posterior_matches_two_component_bayes_rule():
    estimator = ConditionalKDE(np.array([[-1., -2.], [1., 2.]]), [0], 1.)
    conditioned = estimator.condition(np.array([1.]))
    np.testing.assert_allclose(conditioned.weights, [1 / (1 + np.exp(2)), 1 / (1 + np.exp(-2))])
    np.testing.assert_array_equal(conditioned.centers, [[-2.], [2.]])
    assert conditioned.effective_components < 2
    original = estimator.centers.copy()
    estimator.condition(np.array([-100.]))
    np.testing.assert_array_equal(estimator.centers, original)


def test_duplicate_particles_keep_mass_but_do_not_inflate_effective_support():
    estimator = ConditionalKDE(np.array([[0., 0.], [0., 0.], [2., 2.]]), [0], 1.)
    mixture = estimator.condition(np.array([0.]))
    assert len(mixture.centers) == 2
    expected = np.array([2., np.exp(-2)])
    expected /= expected.sum()
    np.testing.assert_allclose(mixture.weights, expected)
    assert mixture.effective_components == pytest.approx(1 / np.square(expected).sum())


def test_density_and_sampling_describe_same_normalized_smoothed_law():
    mixture = ConditionalKDE(np.zeros((3, 2)), [0], .2).condition(np.array([0.]))
    assert mixture.log_prob(np.array([[0.]]))[0] == pytest.approx(-np.log(.2 * np.sqrt(2 * np.pi)))
    draws = mixture.sample(30000, np.random.default_rng(7))
    assert draws.mean() == pytest.approx(0., abs=.004)
    assert draws.var() == pytest.approx(.04, abs=.0015)
    assert mixture.effective_components == 1.


def test_conditional_energy_and_crps_known_point_mass():
    draws = np.full((4, 1), 2.)
    rows = conditional_scores(np.array([0.]), draws, draws, np.array([-2., -1.]), -3., [.5, .9],
                              validation_log_density=np.array([-1., -1.]))
    assert rows[0]["energy_score"] == rows[0]["conditional_crps"] == 2.
    assert rows[0]["hdr_covered"] == 0.
    assert rows[0]["independent_model_mass"] == 1.
    assert rows[1]["mean_interval_score"] == pytest.approx(40.)
    assert "sd_ece" not in rows[0]


def test_multivariate_energy_uses_joint_hidden_vector():
    draws = np.tile([3., 4.], (5, 1))
    row = conditional_scores(np.zeros(2), draws, draws, np.array([-2., -1.]), -3., [.9])[0]
    assert row["energy_score"] == 5.
    assert row["conditional_crps"] == 3.5


def test_hdr_ece_aggregates_observations_before_absolute_error():
    rows = []
    for index in range(10):
        values = conditional_scores(np.zeros(1), np.zeros((2, 1)), np.zeros((2, 1)),
                                    np.zeros(2), 0., [.5, .9])
        for row in values:
            row["hdr_covered"] = float(index < (5 if row["nominal_alpha"] == .5 else 9))
        rows.extend(values)
    result = aggregate(rows)
    assert result[0]["coverage"] == .5
    assert result[1]["coverage"] == .9
    assert all(row["hdr_ece"] == 0 for row in result)


def test_fixed_bandwidth_does_not_change_when_fit_bank_repeats():
    values = np.array([[0., 0.], [1., 1.]])
    first = ConditionalKDE(values, [0], .1).condition(np.array([.2]))
    repeated = ConditionalKDE(np.repeat(values, 3, axis=0), [0], .1).condition(np.array([.2]))
    np.testing.assert_allclose(first.log_prob(np.array([[0.], [.5]])), repeated.log_prob(np.array([[0.], [.5]])))


def test_st_presets_use_reserved_cells_shared_masks_mse_and_flow2pc():
    for method in ("alicfm", "otpfm"):
        path = ROOT / f"configs/experiments/e3_2/{method}/{method}_st_hclt.yaml"
        config = load_config(path)
        assert config["selection"]["metric"] == "validation_selection_score"
        assert config["evaluation"]["observed_masks"] == [[0], [1]]
        assert config["evaluation"]["time_indices"] == [1]
        assert config["evaluation"]["reference_split"] == "test"
        assert config["logging"]["project"] == "flow2pc"
        assert config["experiment"] == "e3_2" and config["stage"] == "evaluate"
        assert "training" not in config and "distillation" not in config
        assert resolve_config(config, base=path.parent) == config


@pytest.mark.parametrize("settings", [
    {"levels": [.5, .7]}, {"condition_bandwidths": [.2, .1]},
    {"primary_bandwidth": .123}, {"observed_masks": [[0], [0]]},
    {"conditional_draw_sizes": [1]}, {"reference_split": "all"},
    {"hdr_representation": "pca2"},
])
def test_invalid_scientific_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        resolve(settings)


def test_invalid_score_dimensions_fail_clearly():
    with pytest.raises(ValueError):
        conditional_scores(np.zeros(2), np.zeros((3, 1)), np.zeros((3, 1)), np.zeros(3), 0., [.9])


@pytest.mark.parametrize("method", ["alicfm", "otpfm"])
@pytest.mark.parametrize("dataset", ["cite5", "multi5"])
def test_single_cell_pilots_preserve_temporal_test_and_score_full_hidden_state(method, dataset):
    path = ROOT / f"configs/experiments/e3_2/{method}/{method}_{dataset}_hclt.yaml"
    config = load_config(path)
    settings = config["evaluation"]
    assert config["stage"] == "evaluate" and config["experiment"] == "e3_2"
    assert config["selection"]["metric"] == "validation_selection_score"
    assert config["logging"]["project"] == "flow2pc"
    assert settings["reference_split"] == "test" and settings["time_indices"] == [1]
    assert settings["max_test_samples"] == 512
    assert settings["condition_bandwidths"] == [.05, .1, .2]
    assert settings["primary_bandwidth"] == .1
    assert masks_for_dimension(settings["observed_masks"], 5) == [([0, 1], [2, 3, 4]), ([2, 3, 4], [0, 1])]
    assert "hdr_representation" not in settings
    assert "training" not in config and "distillation" not in config
    assert resolve_config(config, base=path.parent) == config

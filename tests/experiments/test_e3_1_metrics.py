"""Numerical/calibration regressions; run only with explicit test authorization."""

import numpy as np
import pytest

from flow2pc.experiments.e3_1.metrics import EstimatedHDR, Representation, energy_score


def test_energy_score_has_known_two_point_value():
    # P(-1)=P(+1)=1/2, x=0: E|Y-x|=1, E|Y-Y'|=1.
    draws = np.array([[-1.0], [1.0]])
    assert energy_score(np.array([[0.0]]), draws, draws) == pytest.approx(.5)


def test_gaussian_coverage_uses_independent_banks_and_does_not_refit_on_real(monkeypatch):
    from sklearn.neighbors import KernelDensity

    streams = [np.random.default_rng(seed) for seed in range(4)]
    fit, tune, threshold, real = [rng.normal(size=(3000, 1)) for rng in streams]
    estimator = EstimatedHDR(fit, tune, [.5, 1, 2])
    search = list(estimator.search)

    def forbidden(*args, **kwargs):
        raise AssertionError("Real observations must never refit/tune the model density")

    monkeypatch.setattr(KernelDensity, "fit", forbidden)
    scores, ece = estimator.coverage(threshold, real, [.5, .7, .9, .95])
    assert all(abs(row["coverage"] - row["nominal_alpha"]) < .06 for row in scores)
    assert ece < .05
    assert estimator.search == search
    # A far-away population must under-cover, not get recalibrated into coverage.
    bad, _ = estimator.coverage(threshold, np.full((100, 1), 100.0), [.9])
    assert bad[0]["coverage"] == 0


def test_projected_hdr_does_not_project_the_energy_score_state():
    training = np.random.default_rng(5).normal(size=(300, 50))
    representation = Representation(training, "pca2")
    assert representation.hdr(training).shape == (300, 2)
    assert representation.energy(training).shape == (300, 50)
    center = representation.center.copy()
    representation.hdr(np.full((10, 50), 1000.0))
    np.testing.assert_array_equal(representation.center, center)
    with pytest.raises(ValueError, match="restricted"):
        Representation(training, "full")


def test_discrete_density_score_ties_are_recorded():
    values = np.zeros((30, 1))
    estimator = EstimatedHDR(values, values, [1])
    scores, _ = estimator.coverage(values, values, [.9])
    assert scores[0]["threshold_tie_fraction"] == 1
    assert scores[0]["coverage"] == 1


def test_narrow_model_bank_can_scale_the_bandwidth_search():
    rng = np.random.default_rng(2026)
    fit, tune = rng.normal(0, .001, (300, 2)), rng.normal(0, .001, (200, 2))
    estimator = EstimatedHDR(fit, tune, [1], bandwidth_scale="model_std")
    expected_scale = np.exp(np.mean(np.log(fit.std(0))))
    assert estimator.scale == pytest.approx(expected_scale)
    assert estimator.bandwidth == pytest.approx(expected_scale * len(fit) ** (-1 / 6))
    assert EstimatedHDR(fit, tune, [1]).scale == 1.0

"""Numerical/calibration regressions; run only with explicit test authorization."""

import numpy as np
import pytest

from flow2pc.experiments.e3_1.metrics import (
    EstimatedHDR, Representation, distribution_fidelity, energy_score, region_fidelity,
)


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


def test_explicit_five_dimensional_hdr_retains_every_coordinate():
    training = np.random.default_rng(7).normal(size=(300, 5))
    representation = Representation(training, "full")
    assert not representation.projected
    assert representation.hdr(training).shape == (300, 5)
    np.testing.assert_array_equal(representation.axes, np.eye(5))


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


def test_fidelity_separates_location_error_from_interval_width_error():
    teacher = np.array([[-1., -2.], [0., 0.], [1., 2.]])
    shifted = teacher + np.array([3., 4.])
    row = distribution_fidelity(teacher, shifted, [.9])[0]
    assert row["mean_shift_l2"] == pytest.approx(5.)
    assert row["paired_ode_rmse"] == pytest.approx(np.sqrt(12.5))
    assert row["relative_covariance_error"] == pytest.approx(0.)
    assert row["mean_interval_width_ratio"] == pytest.approx(1.)
    assert row["relative_interval_width_error"] == pytest.approx(0.)
    scaled = distribution_fidelity(teacher, 2 * teacher, [.9])[0]
    assert scaled["mean_shift_l2"] == pytest.approx(0.)
    assert scaled["relative_covariance_error"] == pytest.approx(3.)
    assert scaled["mean_interval_width_ratio"] == pytest.approx(2.)


def test_identical_model_draws_have_zero_fidelity_errors():
    teacher = np.array([[-1., -2.], [0., 0.], [1., 2.]])
    row = distribution_fidelity(teacher, teacher.copy(), [.5, .9])[1]
    assert row["paired_ode_rmse"] == row["mean_shift_l2"] == row["relative_covariance_error"] == 0.
    assert row["mean_interval_width_ratio"] == pytest.approx(1.)


def test_degenerate_spread_is_reported_undefined_instead_of_clamped():
    row = distribution_fidelity(np.zeros((4, 2)), np.ones((4, 2)), [.9])[0]
    assert row["relative_covariance_error"] is None
    assert row["mean_interval_width_ratio"] is None
    assert row["relative_interval_width_error"] is None


def test_region_fidelity_detects_disjoint_models_and_identity():
    class Ranking:
        def __init__(self, center):
            self.center = center

        def score(self, values):
            return -((values - self.center) ** 2).sum(1)

    teacher, student = np.array([[-1.], [0.], [1.]]), np.array([[9.], [10.], [11.]])
    regions = [{"nominal_alpha": .9, "log_density_threshold": -1.}]
    row = region_fidelity(Ranking(0.), Ranking(10.), teacher, student, regions, regions)[0]
    assert row["teacher_region_teacher_mass"] == row["student_region_student_mass"] == 1.
    assert row["teacher_region_student_mass"] == row["student_region_teacher_mass"] == 0.
    assert row["teacher_region_mass_gap"] == row["student_region_mass_gap"] == -1.
    identical = region_fidelity(Ranking(0.), Ranking(0.), teacher, teacher, regions, regions)[0]
    assert identical["teacher_region_mass_gap"] == identical["student_region_mass_gap"] == 0.

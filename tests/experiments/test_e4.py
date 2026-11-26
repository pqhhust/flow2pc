"""E4 regression cases; execution requires explicit user authorization."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from flow2pc.experiments.e4.config import resolve
from flow2pc.experiments.e4.metrics import mmd2, response, uncertainty
from flow2pc.experiments.e4.perturbations import bank_plans, conditions, perturb
from flow2pc.experiments.e4.summary import E41, summarize
from flow2pc.utils.config import load_config


def test_identity_and_spread_keep_the_declared_initial_law():
    x = np.array([[1., 2.], [3., 5.], [6., 8.]])
    for condition in ({"shift_type": "clean", "severity": 0.},
                      {"shift_type": "mean", "severity": 0.},
                      {"shift_type": "spread", "severity": 1.}):
        np.testing.assert_allclose(perturb(x, condition, [1., 0.]), x)
    widened = perturb(x, {"shift_type": "spread", "severity": 1.5}, [1., 0.])
    np.testing.assert_allclose(widened.mean(0), x.mean(0))
    np.testing.assert_allclose(widened.var(0), x.var(0)*2.25)


def test_bank_streams_are_reproducible_without_claiming_disjoint_cell_ids():
    settings = resolve({"direction": [1.]})
    first, audit = bank_plans(10, settings)
    second, other = bank_plans(10, settings)
    assert audit == other
    for seed, banks in first.items():
        for key, ids in banks.items():
            np.testing.assert_array_equal(ids, second[seed][key])
        assert not np.array_equal(banks["region"], banks["reference"])
        assert len(np.intersect1d(banks["region"], banks["reference"])) > 0
    assert conditions(settings)[0]["condition"] == "clean"


def test_duplicate_compression_is_the_biased_mmd_estimator():
    a = np.array([[0.], [0.], [1.], [2.]])
    b = np.array([[0.], [3.], [3.]])
    def kernel(x, y):
        return np.exp(-cdist(x, y, "sqeuclidean")/2)
    expected = kernel(a, a).mean()+kernel(b, b).mean()-2*kernel(a, b).mean()
    assert mmd2(a, b, [1.], chunk=2) == pytest.approx(expected)
    assert mmd2(a, a, [1.], chunk=2) == pytest.approx(0.)
    assert mmd2(a, b+20, [1.]) > mmd2(a, b, [1.])


def test_population_agreement_does_not_imply_paired_agreement():
    teacher = np.array([[-1.], [1.]])
    student = teacher[::-1].copy()
    assert mmd2(teacher, student, [1.]) == pytest.approx(0.)
    assert response(teacher, student, teacher, student)["trajectory_rmse"] == pytest.approx(2.)
    shifted = teacher+3
    record = response(shifted, shifted, teacher, teacher)
    assert record == {"mean_shift_error": 0., "variance_shift_error": 0., "trajectory_rmse": 0.}


def test_interval_coverage_errors_are_computed_before_feature_averaging():
    region = np.array([[-1., -1.], [1., 1.]])
    target = np.array([[0., 9.], [0., 9.]])
    rows, scores = uncertainty(region, target, region, region, [.5, .9])
    assert rows[0]["coverage_sd"] == .5
    assert rows[0]["interval_ece_at_level"] == .5
    assert scores["hdr_ece"] is None
    assert scores["sd_ece"] == .5
    assert "no validated" in scores["hdr_reason"]


def test_proper_score_has_known_point_mass_value():
    point = np.zeros((4, 2))
    target = np.array([[3., 4.]])
    _, scores = uncertainty(point, target, point, point, [.9])
    assert scores["energy_score"] == pytest.approx(5.)
    assert scores["crps"] == pytest.approx(3.5)


def test_refuses_unobserved_ground_truth_and_unvalidated_hdr():
    with pytest.raises(ValueError, match="ground truth"):
        resolve({"reference_mode": "ground-truth-shifted", "direction": [1.]})
    with pytest.raises(ValueError, match="validated density"):
        resolve({"hdr_method": "pc_density", "direction": [1.]})
    with pytest.raises(ValueError, match="unit shift direction"):
        resolve({"direction": [2.]})


def test_time_zero_does_not_improve_headline_and_rmse_uses_squared_reduction():
    base = {"method": "fixture", "dataset": "fixture", "reference_mode": "teacher-relative",
            "eval_target": "teacher", "condition": "clean", "shift_type": "clean", "severity": 0.,
            "role": "student", "score_space": "fixed", "seed": 0}
    rows = [{**base, "time_index": i, **{k: float(v) for k in E41}} for i, v in enumerate((0, 3, 4))]
    result = summarize(rows, E41, "e4.1")[0]
    assert result["mean_shift_error"] == 3.5
    assert result["trajectory_rmse"] == pytest.approx(np.sqrt(12.5))
    assert result["trajectory_rmse_mc_se"] is None


def test_pilot_configs_pin_frozen_selected_checkpoints_and_explicit_directions():
    root = Path(__file__).resolve().parents[2]
    files = list((root/"configs/experiments/e4").rglob("*.yaml"))
    assert len(files) == 4
    for path in files:
        config = load_config(path)
        assert config["stage"] == "evaluate" and config["experiment"] == "e4"
        assert config["selection"]["metric"] == "validation_selection_score"
        assert config["selection"]["pinned_checkpoint"].endswith("checkpoints/best.pt")
        assert config["evaluation"]["reference_mode"] == "teacher-relative"
        assert config["logging"]["project"] == "flow2pc"
        assert config["evaluation"]["direction"] == ([1.] if config["dataset"] == "beijing" else [1., 0.])

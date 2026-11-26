"""Source-density objective regressions; execute only with test authorization."""

from pathlib import Path

import numpy as np
import pytest
import torch

from flow2pc.distill.objectives import (
    anchored_velocity_loss, source_nll_per_dimension, validate_source_density,
)
from flow2pc.distill.sampling import DistillationBatch
from flow2pc.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class GaussianStudent(torch.nn.Module):
    dim = 2

    def __init__(self):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.tensor(.3))

    def velocity(self, values, time):
        return self.mean.expand_as(values)

    def log_prob(self, values, time):
        assert time.ndim == 0
        return -.5 * ((values - self.mean).square() + np.log(2 * np.pi)).sum(-1)


class FrozenTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.eval()

    def velocity(self, values, time, *, context=None):
        return self.weight.expand_as(values)


def query():
    return DistillationBatch(torch.zeros(3, 2), torch.full((3,), .5))


def test_weighted_objective_and_gradient_match_known_gaussian():
    student, teacher = GaussianStudent(), FrozenTeacher()
    source = torch.full((5, 2), 2.)
    loss, mse, nll = anchored_velocity_loss(student, teacher, query(), weight=.2, source=source)
    assert float(mse) == pytest.approx(.3**2)
    assert float(nll) == pytest.approx(.5 * ((2 - .3)**2 + np.log(2 * np.pi)))
    assert float(loss) == pytest.approx(float(mse + .2 * nll))
    loss.backward()
    assert float(student.mean.grad) == pytest.approx(.6 + .2 * (.3 - 2))
    assert teacher.weight.grad is None and not teacher.training


def test_zero_weight_never_queries_density_or_consumes_rng(monkeypatch):
    student = GaussianStudent()
    def forbidden(*args, **kwargs):
        raise AssertionError("Velocity-only baseline must not evaluate source density")
    monkeypatch.setattr(student, "log_prob", forbidden)
    before = torch.get_rng_state().clone()
    loss, mse, nll = anchored_velocity_loss(student, FrozenTeacher(), query())
    assert loss is mse and nll is None
    assert torch.equal(before, torch.get_rng_state())


def test_validation_weights_partial_batches_and_does_not_change_parameters():
    student = GaussianStudent()
    source = torch.arange(14).reshape(7, 2).float()
    expected = float(source_nll_per_dimension(student, source))
    before = student.mean.detach().clone()
    assert validate_source_density(student, source, 0., 3) == pytest.approx(expected)
    torch.testing.assert_close(student.mean, before)
    assert student.mean.grad is None
    with pytest.raises(ValueError, match="reserved"):
        validate_source_density(student, source[:0], 0., 3)


@pytest.mark.parametrize("weight", [-1., float("nan"), float("inf"), True])
def test_invalid_weights_fail_in_runtime_and_config(weight):
    with pytest.raises(ValueError, match="source_nll_weight"):
        anchored_velocity_loss(GaussianStudent(), FrozenTeacher(), query(), weight=weight)
    path = ROOT / "configs/distill/alicfm/alicfm_st_hclt.yaml"
    encoded = {float("inf"): ".inf"}.get(weight, str(weight)) if not np.isnan(weight) else ".nan"
    if weight is True:
        encoded = "true"
    with pytest.raises(ValueError, match="source_nll_weight"):
        load_config(path, [f"distillation.source_nll_weight={encoded}"])


def test_positive_weight_requires_training_source_and_checks_finite_density(monkeypatch):
    student = GaussianStudent()
    with pytest.raises(ValueError, match="requires source"):
        anchored_velocity_loss(student, FrozenTeacher(), query(), weight=1.)
    monkeypatch.setattr(student, "log_prob", lambda values, time: torch.full((len(values),), float("nan")))
    with pytest.raises(FloatingPointError, match="log probabilities"):
        source_nll_per_dimension(student, torch.ones(4, 2))


def test_pilot_has_eight_fresh_runs_and_preserves_each_selected_recipe(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import run_source_nll_pilot as pilot

    args = pilot.parser().parse_args(["--output-root", str(tmp_path / "pilot")])
    runs = pilot.build_runs(args)
    assert len(runs) == len({run.output for run in runs}) == 8
    for method in ("alicfm", "otpfm"):
        selected = load_config(ROOT / f"configs/distill/{method}/{method}_st_hclt.yaml")
        group = [run for run in runs if run.config["teacher"]["name"] == method]
        assert {run.config["distillation"]["source_nll_weight"] for run in group} == {0., .01, .1, 1.}
        for run in group:
            assert run.config["teacher"] == selected["teacher"]
            assert run.config["student"] == selected["student"]
            assert run.config["distillation"]["sampler"] == selected["distillation"]["sampler"]
            assert run.config["logging"]["project"] == "flow2pc"
            assert run.output != Path(selected["output"]["dir"])
            assert not run.output.exists()


def test_refinement_grid_changes_only_nll_and_learning_rate(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import run_source_nll_pilot as pilot

    args = pilot.parser().parse_args([
        "--weights", "0", "0.003", "0.01", "0.03", "0.1", "0.3", "1",
        "--lr-factors", "1", "0.3333333333333333", "0.1",
        "--output-root", str(tmp_path / "refinement"),
    ])
    runs = pilot.build_runs(args)
    assert len(runs) == len({r.output for r in runs}) == 42
    for method in ("alicfm", "otpfm"):
        baseline = load_config(ROOT / f"configs/distill/{method}/source_anchor/{method}_st_hclt.yaml")
        group = [r for r in runs if r.config["teacher"]["name"] == method]
        assert len(group) == 21
        assert {r.config["distillation"]["lr"] for r in group} == {
            baseline["distillation"]["lr"] * f for f in args.lr_factors}
        for run in group:
            assert run.config["teacher"] == baseline["teacher"]
            assert run.config["student"] == baseline["student"]
            assert run.config["evaluation"] == baseline["evaluation"]
            expected = dict(baseline["distillation"])
            actual = dict(run.config["distillation"])
            for key in ("source_nll_weight", "lr"):
                expected.pop(key)
                actual.pop(key)
            assert actual == expected
            assert run.config["logging"]["project"] == "flow2pc"
            assert "pc_velocity_weight" not in run.config["distillation"]
            assert f"lr{run.config['distillation']['lr']:g}" in run.name
            assert not run.output.exists()


@pytest.mark.parametrize("factors", [[0.], [-1.], [float("nan")], [float("inf")], [1., 1.]])
def test_refinement_rejects_invalid_lr_factors(monkeypatch, tmp_path, factors):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import run_source_nll_pilot as pilot

    args = pilot.parser().parse_args(["--output-root", str(tmp_path)])
    args.lr_factors = factors
    with pytest.raises(ValueError, match="LR factors"):
        pilot.build_runs(args)

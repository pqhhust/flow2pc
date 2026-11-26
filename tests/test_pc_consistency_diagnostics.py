"""Diagnostic regressions. Do not execute without test authorization."""

import json
from pathlib import Path

import pytest
import torch
import yaml

from flow2pc.evaluation.pc_consistency import velocity_support_metrics
from flow2pc.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class Field(torch.nn.Module):
    dim = 2

    def __init__(self, multiplier=0., offset=0.):
        super().__init__()
        self.multiplier = torch.nn.Parameter(torch.tensor(float(multiplier)))
        self.offset = offset
        self.eval()

    def velocity(self, values, time):
        return values * self.multiplier + self.offset


class DensityField(Field):
    def log_prob(self, values, time):
        return -.5 * values.square().sum(-1)


def test_optional_tail_records_identify_exact_rows_without_changing_metrics():
    values = torch.tensor([[1., 0.], [0., 0.], [30., 40.], [2., 0.]])
    student, teacher = DensityField(1.), Field()
    baseline = velocity_support_metrics(student, teacher, values, .5, 3)
    recorded = velocity_support_metrics(student, teacher, values, .5, 3, tail_count=2)
    tails = recorded.pop("tail_samples")
    assert recorded == baseline
    assert [r["population_index"] for r in tails] == [2, 3]
    assert tails[0]["state_model"] == [30., 40.]
    assert tails[0]["student_velocity_model"] == [30., 40.]
    assert tails[0]["teacher_velocity_model"] == [0., 0.]
    assert tails[0]["population_mse_fraction"] == pytest.approx(2500 / 2505)
    assert tails[0]["pc_log_probability_model"] == -1250.
    assert tails[0]["pc_nll_per_dimension_model"] == 625.
    assert student.multiplier.grad is None and teacher.multiplier.grad is None


def test_tail_count_rejects_negative_or_noninteger_values():
    for count in (-1, .5, True):
        with pytest.raises(ValueError, match="tail_count"):
            velocity_support_metrics(Field(), Field(), torch.zeros(2, 2), .5, 2,
                                     tail_count=count)


def test_velocity_support_matches_known_error_and_weights_partial_batches():
    values = torch.tensor([[0., 2.], [1., 3.], [2., 4.]])
    student, teacher = Field(1.), Field(0., 2.)
    original = student.multiplier.detach().clone()
    result = velocity_support_metrics(student, teacher, values, .4, 2)
    mse = float((values - 2).square().mean())
    assert result["velocity_mse_model"] == pytest.approx(mse)
    assert result["teacher_velocity_energy_model"] == 4.
    assert result["relative_velocity_rmse"] == pytest.approx((mse / 4)**.5)
    assert result["state_count"] == 3
    assert student.multiplier.grad is None and teacher.multiplier.grad is None
    torch.testing.assert_close(student.multiplier, original)


def test_tail_errors_are_never_trimmed_and_zero_teacher_speed_is_na():
    values = torch.zeros(100, 2)
    values[-1] = 1000.
    result = velocity_support_metrics(Field(1.), Field(), values, .5, 16)
    assert result["velocity_mse_model"] == 10000.
    assert result["relative_velocity_rmse"] is None
    assert result["point_velocity_rmse_max"] == 1000.
    assert result["point_velocity_rmse_median"] == 0.


def test_nonfinite_states_are_not_silently_discarded():
    values = torch.zeros(5, 2)
    values[-1, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        velocity_support_metrics(Field(), Field(), values, .5, 2)


def test_preview_never_launches_workers_queries_gpus_or_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import run_pc_consistency_pilot as pilot

    root = tmp_path / "pilot"
    entries = []
    for weight in (0., .1):
        run = root / str(weight)
        (run / "checkpoints").mkdir(parents=True)
        config = load_config(ROOT / "configs/distill/alicfm/source_anchor/alicfm_st_hclt.yaml")
        config["distillation"]["source_nll_weight"] = weight
        config["output"]["dir"] = str(run)
        (run / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        (run / "status.json").write_text(json.dumps({"status": "complete"}))
        (run / "checkpoints/best.pt").write_bytes(b"metadata-only checkpoint fixture")
        entries.append({"config": config})
    (root / "distillation_plan.json").write_text(json.dumps({"runs": entries}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Metadata preview must not run GPUs or workers")
    monkeypatch.setattr(pilot.batch, "gpu_inventory", forbidden)
    monkeypatch.setattr(pilot.Queue, "run", forbidden)
    output = tmp_path / "diagnostics"
    assert pilot.main(["--dry-run", "--methods", "alicfm", "--pilot-root", str(root),
                       "--output-root", str(output)]) == 0
    assert not output.exists()


def test_numeric_failure_report_is_not_a_success(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import run_pc_consistency_pilot as pilot

    task = pilot.Task("fixture", "alicfm", .1, tmp_path / "config.yaml", tmp_path / "best.pt",
                      {"checkpoint_sha256": "fixture"}, tmp_path / "output")
    task.output.mkdir()
    (task.output / "results.json").write_text(json.dumps({
        "status": "completed_with_errors", "protocol": task.expected,
        "numerical_errors": [{"phase": "student_ODE_from_PC_initial", "error": "budget exceeded"}],
    }))
    pilot.inspect(task)
    assert task.status == "failed" and "budget exceeded" in task.error


def test_direct_diagnostic_scores_multiple_times_without_rollouts(tmp_path, monkeypatch):
    """A population label must not overwrite the observation loader after t=0."""
    from types import SimpleNamespace

    np = pytest.importorskip("numpy")
    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import check_pc_density_consistency as diagnostic
    from flow2pc.data import restore
    from flow2pc.evaluation import entrypoint, ode, runner
    from flow2pc.pretrain import runner as pretrain
    from flow2pc.utils import tracking

    class SampleField(DensityField):
        def sample(self, count, time, generator):
            return torch.randn(count, self.dim, generator=generator) + time

    class Logger:
        run = None

        def __init__(self, *args):
            pass

        def log(self, *args):
            pass

        def close(self):
            pass

    def forbidden(*args, **kwargs):
        raise AssertionError("Direct-only diagnostics must not integrate an ODE")

    preprocessing = {"shift": [0., 0.], "scale": [1., 1.]}
    cells = torch.tensor([[0., 1.], [1., 0.], [2., 1.]])
    dataset = SimpleNamespace(
        metadata=SimpleNamespace(name="fixture", model_times=[0., 1.], physical_times=[1., 3.]),
        preprocessing=preprocessing, dim=2, observed=[0],
        get_marginal=lambda index, split: SimpleNamespace(materialize=lambda: cells + index))
    teacher = Field(0., 1.)
    teacher.preprocessing = preprocessing
    teacher.manifest = {"dataset": {}, "checkpoint_sha256": "teacher-fixture"}
    student = SampleField(1.)
    original = student.multiplier.detach().clone()
    visited = []
    original_population = runner.population

    def observations(data, index, split):
        visited.append(index)
        return original_population(data, index, split)

    monkeypatch.setattr(pretrain, "load_teacher", lambda *a, **k: teacher)
    monkeypatch.setattr(restore, "restore_dataset", lambda *a, **k: dataset)
    monkeypatch.setattr(entrypoint, "_load_student", lambda *a, **k: (student, 20))
    monkeypatch.setattr(runner, "population", observations)
    monkeypatch.setattr(ode, "rollout", forbidden)
    # This regression covers orchestration and retained samples, not the OT solver.
    monkeypatch.setattr(ode, "emd", lambda a, b: float((a.mean(0) - b.mean(0)).norm()))
    monkeypatch.setattr(tracking, "LocalLogger", Logger)
    output = tmp_path / "diagnostic"
    args = diagnostic.parser().parse_args([
        "--device", "cpu", "--samples", "4", "--metric-samples", "2",
        "--batch-size", "3", "--tail-count", "1", "--direct-only",
        "--time-indices", "0", "1", "--mode", "disabled", "--output-dir", str(output)])
    config = load_config(ROOT / "configs/distill/alicfm/source_anchor/alicfm_st_hclt.yaml")
    result = diagnostic.run(args, config, tmp_path / "best.pt",
                            {"checkpoint_sha256": "student-fixture"})
    assert visited == [0, 1]
    assert result["status"] == "complete"
    assert [r["time_index"] for r in result["rows"]] == [0, 1]
    assert all(r["direct_vs_observed_w1_raw"] is not None for r in result["rows"])
    assert len(result["velocity_support"]) == 4
    assert all(r["status"] == "complete" for r in result["velocity_support"])
    assert result["own_initial_ode_nfe"] is None
    with np.load(result["direct_sample_bank"]) as bank:
        assert bank["direct_raw_0"].shape == bank["direct_raw_1"].shape == (4, 2)
        assert bank["metric_reference_raw_1"].shape == (2, 2)
    assert Path(result["media"]).is_file()
    torch.testing.assert_close(student.multiplier, original)
    assert student.multiplier.grad is None

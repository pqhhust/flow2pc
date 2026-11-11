"""CITE student-sweep contracts; execute only with explicit test authorization."""

import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sweep(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "_otp_cite_student_sweep", ROOT / "scripts/run_otpfm_distill_sweep.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_two_hundred_unique_runs_preserve_teachers_and_training_contract(sweep, tmp_path):
    args = sweep.parser().parse_args(["--output-root", str(tmp_path)])
    runs = sweep.build_runs(args)
    assert Counter(run.dataset for run in runs) == {"cite50": 100, "cite100": 100}
    assert len({run.output for run in runs}) == len({run.name for run in runs}) == 200
    assert [run.dataset for run in runs[:4]] == ["cite50", "cite100"] * 2
    assert [run.name for run in sweep.build_runs(args)] == [run.name for run in runs]
    for run in runs:
        base = sweep.load_config(ROOT / f"configs/distill/otpfm/otpfm_{run.dataset}_hclt.yaml")
        config = copy.deepcopy(run.config)
        assert config["teacher"] == base["teacher"]
        assert config["evaluation"] == base["evaluation"]
        assert config["logging"]["project"] == "flow2pc"
        assert config["logging"]["name"].startswith(run.dataset + "d-")
        assert run.output.is_relative_to(tmp_path)
        assert config["output"]["dir"] != base["output"]["dir"]
        for key in ("num_latents", "time_embedding_dim"):
            config["student"]["params"][key] = base["student"]["params"][key]
        config["distillation"]["lr"] = base["distillation"]["lr"]
        assert sweep.batch.scientific_config(config) == sweep.batch.scientific_config(base)
    for run in runs[:2]:
        assert run.config["student"]["params"]["num_latents"] == 128
        assert run.config["student"]["params"]["time_embedding_dim"] == 32
        assert run.config["distillation"]["lr"] == 0.001
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("options", [
    ["--datasets", "knot"], ["--datasets", "cite50", "cite50"],
    ["--latents", "128", "128"], ["--time-features", "3"],
    ["--learning-rates", "nan"], ["--learning-rates", "0"], ["--steps", "0"],
])
def test_invalid_sweep_axes_are_rejected(sweep, options):
    with pytest.raises(ValueError):
        sweep.build_runs(sweep.parser().parse_args(options))


def test_changed_training_budget_uses_new_paths(sweep, tmp_path):
    options = ["--output-root", str(tmp_path), "--latents", "128",
               "--time-features", "32", "--learning-rates", "0.001"]
    original = sweep.build_runs(sweep.parser().parse_args(options))
    changed = sweep.build_runs(sweep.parser().parse_args(options + ["--steps", "50000"]))
    assert all(a.output != b.output for a, b in zip(original, changed, strict=True))
    assert all(run.config["distillation"]["steps"] == 50000 for run in changed)


def test_ranking_uses_velocity_mse_and_retains_late_emd_diagnostics(sweep, tmp_path):
    args = sweep.parser().parse_args([
        "--datasets", "cite50", "--output-root", str(tmp_path),
        "--latents", "64", "128", "--time-features", "32", "--learning-rates", "0.001",
    ])
    runs = sweep.build_runs(args)
    for run, mse, emd in zip(runs, (0.01, 0.02), (100.0, 1.0), strict=True):
        run.status = "complete"
        run.state = {"best_selection_score": mse, "best_step": 1000}
        run.metrics = {"student_ode_emd_mean": emd, "student_ode_emd_heldout": emd,
                       "by_time": {"0": {}, "3": {"teacher_ode_emd": 41.8,
                                                   "student_ode_emd": 50.8,
                                                   "paired_ode_mse": 4.0}, "1": {}}}
    sweep.summary(args, runs, [], "complete")
    best = json.loads((tmp_path / "cite50/best_models.json").read_text())
    assert best["population_emd_used_for_selection"] is False
    assert best["best"][0]["run"] == runs[0].name
    assert best["best"][0]["student_ode_emd_last"] == 50.8
    assert best["best"][0]["teacher_ode_emd_last"] == 41.8


def test_dry_run_never_queries_gpus_writes_outputs_or_starts_workers(sweep, tmp_path, monkeypatch):
    monkeypatch.setattr(sweep.batch, "inspect_run", lambda *args: None)
    monkeypatch.setattr(sweep.batch, "gpu_inventory", lambda: pytest.fail("GPU query"))
    monkeypatch.setattr(sweep.batch, "DistillationQueue", lambda *args, **kwargs: pytest.fail("worker launch"))
    output = tmp_path / "not-created"
    assert sweep.main(["--dry-run", "--output-root", str(output)]) == 0
    assert not output.exists()

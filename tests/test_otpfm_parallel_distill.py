"""Parallel-controller contracts; execute only with explicit test authorization."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("_otp_distill_parallel", ROOT / "scripts/run_otpfm_distill_parallel.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_batch_uses_only_distill_presets_and_preserves_training_settings(launcher, tmp_path):
    from flow2pc.utils.config import load_config

    args = launcher.parser().parse_args(["--dry-run", "--output-root", str(tmp_path)])
    runs = launcher.build_runs(args)
    assert len(runs) == 10
    assert len({run.output for run in runs}) == 10
    for run in runs:
        original = load_config(ROOT / f"configs/distill/otpfm/otpfm_{run.dataset}_hclt.yaml")
        assert launcher.scientific_config(run.config) == launcher.scientific_config(original)
        assert run.config["stage"] == "distill"
        assert run.config["teacher"] == original["teacher"]
        assert run.config["distillation"]["device"] == "cuda:0"
        assert run.config["logging"]["project"] == "flow2pc"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("action", ("distill", "evaluate", "restart"))
def test_saved_checkpoint_retry_never_dispatches_teacher_evaluation(launcher, monkeypatch, action):
    calls = []
    monkeypatch.setattr(launcher.ParallelSweep, "launch", lambda self, task, *args: calls.append(task.action))
    queue = object.__new__(launcher.DistillationQueue)
    archived = []
    queue.archive = lambda task: archived.append(task)
    task = SimpleNamespace(action=action)
    queue.launch(task, "0", {}, None)
    assert calls == ["distill"]
    assert bool(archived) == (action == "restart")


def test_missing_synthetic_data_is_blocked_before_launch(launcher, tmp_path):
    checkpoint = tmp_path / "teacher/checkpoints/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (checkpoint.parents[1] / "manifest.json").write_text(json.dumps({
        "method": "otpfm", "model_config": {"native_config": {
            "potential": "w2inf", "solver_type": "direct",
        }},
        "dataset": {"name": "arch", "root": str(tmp_path / "missing-data"),
                    "source": {"prepared_identity": "teacher-bound-identity"}},
    }))
    run = launcher.Run("arch", "fixture", {"teacher": {
        "name": "otpfm", "checkpoint": str(checkpoint),
    }}, tmp_path / "student")
    launcher.inspect_run(run)
    assert run.status == "blocked"
    assert "manifest.json" in run.error and "missing-data" in run.error
    assert not run.output.exists()


def test_offline_fallback_sdk_collision_does_not_abort_local_logging(tmp_path, monkeypatch):
    from flow2pc.utils.tracking import LocalLogger

    class CommunicationError(Exception):
        pass

    def initialize(**kwargs):
        if kwargs["mode"] == "online":
            raise CommunicationError("online startup timeout")
        raise RuntimeError("run ID is in use")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=initialize, errors=SimpleNamespace(CommError=CommunicationError),
    ))
    logger = LocalLogger(tmp_path, {"stage": "distill", "logging": {
        "mode": "online", "project": "flow2pc", "name": "local-fallback-fixture",
    }})
    logger.log({"train/velocity_mse": 0.2})
    logger.close()
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["train/velocity_mse"] == 0.2
    state = json.loads((tmp_path / "tracking_status.json").read_text())
    assert state["status"] == "local_only"
    assert len(state["errors"]) == 2

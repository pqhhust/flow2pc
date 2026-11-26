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


def test_selected_ali_configs_keep_winning_settings_and_declared_teachers(launcher, tmp_path):
    args = launcher.parser().parse_args([
        "--method", "alicfm", "--output-root", str(tmp_path), "--dry-run",
    ])
    runs = launcher.build_runs(args)
    features = {"cite5": 64, "cite50": 64, "cite100": 32,
                "multi5": 64, "multi50": 32, "multi100": 32, "cell_tracking": 32}
    assert {r.dataset for r in runs} == set(features)
    assert len({r.output for r in runs}) == 7
    for run in runs:
        config = run.config
        assert config["student"]["params"] == {
            "num_latents": 256, "time_embedding_dim": features[run.dataset], "time_feature_seed": 0,
        }
        assert config["seed"] == 0
        assert config["distillation"]["steps"] == 50000
        assert config["distillation"]["batch_size"] == 64
        assert config["distillation"]["lr"] == .001
        assert config["distillation"]["sampler"]["name"] == "teacher_interpolant"
        assert config["logging"]["project"] == "flow2pc"
        assert config["evaluation"]["enabled"]
        if run.dataset == "cell_tracking":
            assert config["teacher"]["format"] == "legacy"
            assert config["teacher"]["checkpoint"].endswith("ali-cfm/changed_files/ali_cfm_model.pth")
            assert Path(config["teacher"]["legacy_metadata"]).is_file()
            assert config["evaluation"]["time_indices"] == [0, 24, 49, 74, 99, 114]
        else:
            assert config["teacher"]["checkpoint"] == (
                f"/data/lthung/flow2pc/pretrained-teachers/alicfm/{run.dataset}/checkpoints/last.pt"
            )
    assert not list(tmp_path.iterdir())


def test_ali_dry_run_never_starts_workers_or_queries_gpus(launcher, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preview must not launch workers or query GPUs")

    monkeypatch.setattr(launcher, "teacher_identity", lambda c: ("alicfm", "hash", None, None))
    monkeypatch.setattr(launcher, "gpu_inventory", forbidden)
    monkeypatch.setattr(launcher.DistillationQueue, "run", forbidden)
    output = tmp_path / "preview"
    assert launcher.main([
        "--method", "alicfm", "--output-root", str(output), "--dry-run",
    ]) == 0
    assert not output.exists()


def test_combined_batch_interleaves_30_students_and_keeps_method_settings(launcher, tmp_path):
    from flow2pc.utils.config import load_config

    args = launcher.parser().parse_args([
        "--method", "both", "--output-root", str(tmp_path / "batch"),
        "--alicfm-output-root", str(tmp_path / "ali"),
        "--otpfm-output-root", str(tmp_path / "otp"), "--dry-run",
    ])
    runs = launcher.build_runs(args)
    assert len(runs) == 30
    assert len({run.output for run in runs}) == 30
    for index, run in enumerate(runs):
        method = "alicfm" if index % 2 == 0 else "otpfm"
        assert run.config["teacher"].get("name", "alicfm") == method
        original = load_config(ROOT / f"configs/distill/{method}/{method}_{run.dataset}_hclt.yaml")
        assert launcher.scientific_config(run.config) == launcher.scientific_config(original)
        assert run.output.is_relative_to(tmp_path / ("ali" if method == "alicfm" else "otp"))
    assert {run.dataset for run in runs if run.config["teacher"].get("name") == "otpfm"} == (
        set(launcher.ALL_DATASETS) - {"gom", "beijing"} | {"gom_w2", "beijing_w2"}
    )
    assert not list(tmp_path.iterdir())


def test_combined_batch_selects_teacher_manifests_without_replacing_legacy(launcher, tmp_path):
    teacher_root = tmp_path / "teachers"
    for method, dataset, checkpoint in (("alicfm", "gom", "last.pt"),
                                        ("otpfm", "gom_w2", "best.pt"),
                                        ("otpfm", "cell_tracking", "last.pt")):
        folder = teacher_root / method / dataset
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(json.dumps({
            "method": method, "checkpoint_relpath": f"checkpoints/{checkpoint}",
        }))
    args = launcher.parser().parse_args([
        "--method", "both", "--datasets", "gom", "cell_tracking",
        "--teacher-root", str(teacher_root), "--output-root", str(tmp_path / "batch"), "--dry-run",
    ])
    runs = launcher.build_runs(args)
    assert [r.config["teacher"]["checkpoint"] for r in runs[:2]] == [
        str(teacher_root / "alicfm/gom/checkpoints/last.pt"),
        str(teacher_root / "otpfm/gom_w2/checkpoints/best.pt"),
    ]
    assert runs[2].config["teacher"]["format"] == "legacy"
    assert runs[2].config["teacher"]["checkpoint"].endswith("ali-cfm/changed_files/ali_cfm_model.pth")


def test_legacy_teacher_preview_checks_pinned_hash_without_loading_weights(launcher, tmp_path):
    import hashlib

    path = tmp_path / "teacher.pth"
    path.write_bytes(b"metadata-only-checkpoint-fixture")
    metadata = tmp_path / "metadata.json"
    content = {"data": {"root": str(tmp_path)}, "time_input": "global",
               "velocity_time_units": "model", "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    metadata.write_text(json.dumps(content))
    config = {"teacher": {"name": "alicfm", "checkpoint": str(path), "legacy_metadata": str(metadata)}}
    assert launcher.teacher_identity(config) == (
        "alicfm", content["checkpoint_sha256"], hashlib.sha256(metadata.read_bytes()).hexdigest(), None,
    )
    path.write_bytes(b"different-teacher")
    with pytest.raises(ValueError, match="hash differs"):
        launcher.teacher_identity(config)


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

"""Config/queue guards; no model work is allowed during previews."""

from pathlib import Path

import pytest

from flow2pc.experiments import parallel
from flow2pc.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def test_all_e3_1_configs_are_evaluation_only_and_have_one_teacher():
    presets = list((ROOT / "configs/experiments/e3_1").rglob("*.yaml"))
    assert len(presets) == 28
    for path in presets:
        config = load_config(path)
        assert config["stage"] == "evaluate"
        assert config["experiment"] == "e3_1"
        assert config["logging"]["project"] == "flow2pc"
        assert config["evaluation"]["levels"] == [.5, .7, .9, .95]
        assert "distillation" not in config and "training" not in config
    with pytest.raises(ValueError, match="stage: evaluate"):
        load_config(presets[0], ["stage=pretrain"])


def test_dry_run_does_not_query_gpu_start_workers_or_write_files(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Metadata preview must not perform runtime work")

    monkeypatch.setattr(parallel, "gpu_inventory", forbidden)
    monkeypatch.setattr(parallel.ExperimentQueue, "run", forbidden)
    monkeypatch.setattr(parallel, "select_student", lambda config: {
        "checkpoint": str(tmp_path / "best.pt"), "run_dir": str(tmp_path / "student"),
        "teacher_sha256": "fixture", "score": 1.,
    })
    output = tmp_path / "preview"
    assert parallel.main(["--dry-run", "--methods", "alicfm", "--datasets", "beijing",
                          "--output-root", str(output)]) == 0
    assert not output.exists()

"""Static config naming/provenance contracts; run only at the user's request."""

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_runnable_configs_use_method_folders_and_explicit_student_names():
    for path in (ROOT / "configs").rglob("*.yaml"):
        if "upstream" in path.parts:
            continue
        config = yaml.safe_load(path.read_text())
        if "experiment" in config:
            relative = path.relative_to(ROOT / "configs/experiments" / config["experiment"])
            method = relative.parts[0]
            assert config["stage"] == "evaluate"
            assert method == config["teacher"]["name"]
            assert path.stem == f"{method}_{config['dataset']}_hclt"
            continue
        stage = config["stage"]
        relative = path.relative_to(ROOT / "configs" / stage)
        method = relative.parts[0]
        assert method in ("alicfm", "otpfm", "mfm", "mmfm")
        assert path.stem.startswith(method + "_")
        if stage == "pretrain":
            assert config["model"]["name"] == method
        elif stage == "distill":
            assert config.get("teacher", {}).get("name", "alicfm") == method
            assert path.stem.endswith("_" + config["student"]["name"])


def test_selected_presets_match_recorded_scientific_settings():
    index = json.loads((ROOT / "docs/selected_teachers.json").read_text())
    assert len(index["teachers"]) == 18
    for selected in index["teachers"]:
        config = yaml.safe_load((ROOT / selected["pretrain_config"]).read_text())
        assert config["model"]["name"] == selected["method"]
        assert config["logging"]["project"] == "flow2pc"
        assert config["evaluation"]["enabled"]
        assert 0 < selected["completed_runs"] <= selected["planned_runs"]
        assert config["output"]["dir"] != str(Path(selected["checkpoint"]).parents[1])
        for key in ("output", "logging"):
            config.pop(key)
        config["training"].pop("device", None)
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        assert digest == selected["scientific_config_sha256"]


def test_transport_students_load_selected_teachers_and_use_temporal_validation():
    index = json.loads((ROOT / "docs/selected_teachers.json").read_text())
    for selected in index["teachers"]:
        if selected["method"] != "alicfm" or selected["dataset"] not in ("gom", "beijing"):
            continue
        config = yaml.safe_load((ROOT / selected["distill_config"]).read_text())
        assert config["teacher"]["checkpoint"] == selected["checkpoint"]
        assert config["student"]["name"] == "hclt"
        assert config["distillation"]["validation_scope"] == "heldout_times"
        assert config["distillation"]["sampler"]["name"] == "teacher_interpolant"


def test_synthetic_selection_uses_endpoint_validation_and_preserves_test_times():
    index = json.loads((ROOT / "docs/selected_teachers.json").read_text())
    synthetic = [r for r in index["teachers"] if r["dataset"] in ("arch", "sphere", "lidar")]
    assert len(synthetic) == 6
    for selected in synthetic:
        assert selected["metric"] == "validation_teacher_mean_emd"
        expected = [0, 1] if selected["dataset"] == "lidar" else [0, 2]
        assert selected["validation_time_indices"] == expected
        assert selected["excluded_test_times"] == ([] if selected["dataset"] == "lidar" else [1])
        config = yaml.safe_load((ROOT / selected["distill_config"]).read_text())
        assert config["teacher"]["checkpoint"] == selected["checkpoint"]
        assert config["distillation"]["validation_scope"] == "observed_rows"

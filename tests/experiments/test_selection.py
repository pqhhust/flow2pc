"""One EMD-selected saved student, bound to one exact frozen teacher."""

import json
from pathlib import Path

import pytest

from flow2pc.experiments.selection import digest, select_student


@pytest.fixture
def experiment(tmp_path):
    teacher = tmp_path / "teacher/checkpoints/last.pt"
    teacher.parent.mkdir(parents=True)
    teacher.write_bytes(b"teacher-fixture")
    teacher.parent.parent.joinpath("manifest.json").write_text(json.dumps({
        "method": "alicfm", "checkpoint_sha256": digest(teacher),
        "checkpoint_relpath": "checkpoints/last.pt", "dataset": {"name": "beijing"},
    }))
    config = {"teacher": {"name": "alicfm", "checkpoint": str(teacher)}, "dataset": "beijing",
              "selection": {"student_roots": [str(tmp_path / "students")], "metric": "auto"}}

    def candidate(name, score, *, teacher_sha=None, report_step=10, space="raw"):
        folder = tmp_path / "students" / name
        (folder / "checkpoints").mkdir(parents=True)
        (folder / "eval/population").mkdir(parents=True)
        (folder / "checkpoints/best.pt").write_bytes(b"selected-weights")
        (folder / "resolved_config.yaml").write_text("stage: distill\n")
        (folder / "manifest.json").write_text(json.dumps({
            "artifact_type": "student", "teacher_sha256": teacher_sha or digest(teacher),
            "legacy_metadata_sha256": None, "generator_sha256": None,
        }))
        (folder / "status.json").write_text(json.dumps({"status": "complete", "best_step": 10}))
        (folder / "eval/population/evaluation.json").write_text(json.dumps({
            "checkpoint_step": report_step, "student_ode_emd_heldout": score,
            "teacher_ode_emd_heldout": 2., "direct_pc_emd_heldout": 100 - score,
            "heldout_indices": [1], "space": space, "protocol": {"origin": "earliest"},
        }))
        return folder

    return config, candidate


def test_selection_uses_ode_emd_and_rejects_wrong_teacher_or_report(experiment):
    config, candidate = experiment
    candidate("worse", 4)
    chosen = candidate("best", 2)
    candidate("foreign", .01, teacher_sha="other-teacher")
    candidate("wrong-step", .001, report_step=5)
    selected = select_student(config)
    assert selected["checkpoint"] == str(chosen / "checkpoints/best.pt")
    assert selected["candidate_count"] == 2
    assert len(selected["rejected"]) == 2
    assert selected["metric"] == "student_ode_emd_heldout"


def test_incomparable_emd_units_cannot_be_ranked(experiment):
    config, candidate = experiment
    candidate("raw", 3)
    candidate("standardized", .1, space="standardized")
    with pytest.raises(ValueError, match="protocols/units differ"):
        select_student(config)


def test_launch_pin_keeps_selection_fixed_when_another_candidate_completes(experiment):
    config, candidate = experiment
    chosen = candidate("first", 3)
    config["selection"]["pinned_checkpoint"] = str(chosen / "checkpoints/best.pt")
    candidate("finished-later", 2)
    selected = select_student(config)
    assert Path(selected["run_dir"]) == chosen
    assert selected["candidate_count"] == 2


def test_single_cell_profiles_use_saved_family_and_dimension(experiment):
    config, candidate = experiment
    candidate("single-cell", 3)
    manifest_path = Path(config["teacher"]["checkpoint"]).parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset"] = {"name": "cite", "dim": 50}
    manifest_path.write_text(json.dumps(manifest))
    config["dataset"] = "cite50"
    assert select_student(config)["score"] == 3
    config["dataset"] = "cite100"
    with pytest.raises(ValueError, match="dataset differs"):
        select_student(config)

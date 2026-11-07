"""Reviewable configs for the local MFM catalog (no model/data imports)."""

from pathlib import Path

import yaml

from flow2pc.data.mfm_catalog import PREPARED_ROOT, prepared_path, profile
from flow2pc.utils.paths import MFM_PRETRAIN_ROOT, REPO_ROOT


def native_recipe(name):
    profile(name)
    template = REPO_ROOT / "configs/pretrain/mfm" / f"mfm_{name}.yaml"
    preset = yaml.safe_load(template.read_text())
    if (preset["model"]["params"].get("backend") != "snapshots"
            or preset["data"]["params"].get("profile") != name):
        raise ValueError(f"Invalid MFM snapshot preset: {template}")
    return dict(preset["model"]["params"]["native"]), str(template.relative_to(REPO_ROOT))


def configs(name, *, seed=42, data_root=PREPARED_ROOT, output_root=None, run_name="default", source_root=MFM_PRETRAIN_ROOT):
    profile(name)
    native, template = native_recipe(name)
    base = Path(output_root or REPO_ROOT / "outputs").resolve()
    teacher_output = base / "pretrain/mfm" / name / run_name
    student_output = base / "distill/mfm" / name / run_name
    prepared = prepared_path(data_root, name, seed)
    pretrain = {
        "stage": "pretrain", "seed": seed,
        "model": {"name": "mfm", "params": {"backend": "snapshots", "source_root": str(source_root), "native": native}},
        "data": {"name": "mfm_snapshots", "root": str(prepared), "params": {"profile": name}},
        "training": {"epochs": 1000, "batch_size": 128, "device": "cuda:0",
                     "patience_geopath": 5, "patience_flow": 5, "validate_every": 1},
        "output": {"dir": str(teacher_output)}, "wandb": {"mode": "disabled"},
    }
    # Preserve dataset-specific stopping patience from the checked-in preset.
    raw = yaml.safe_load((REPO_ROOT / template).read_text())
    pretrain["training"]["patience_geopath"] = raw["training"].get("patience_geopath", 5)
    pretrain["training"]["patience_flow"] = raw["training"].get("patience_flow", 5)
    students = {}
    for student in ("hclt", "gaussian"):
        students[student] = {
            "stage": "distill", "seed": seed,
            "teacher": {"name": "mfm", "checkpoint": str(teacher_output / "checkpoints/flow.pt"),
                        "params": {"repo_path": str(source_root), "geopath_checkpoint": str(teacher_output / "checkpoints/geopath.pt")}},
            "data": {"name": "mfm_snapshots", "root": str(prepared)},
            "student": {"name": student, "params": (
                {"num_latents": 64, "time_embedding_dim": 64, "topology": "learned", "topology_samples": 2048, "time_order": 4}
                if student == "hclt" else {"num_components": 64, "time_order": 4})},
            "distillation": {"steps": 20000, "batch_size": 128, "lr": 0.001, "device": "auto",
                             "gradient_clip": 1.0, "log_every": 100, "validate_every": 1000,
                             "checkpoint_every": 5000, "resume_checkpoint": None},
            "output": {"dir": str(student_output / student)},
            "evaluation": {"enabled": False}, "wandb": {"mode": "disabled"},
        }
    return pretrain, students, template

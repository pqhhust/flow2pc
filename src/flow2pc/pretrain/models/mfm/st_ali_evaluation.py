"""Evaluate an ST MFM teacher with the shared ALI-CFM ODE protocol."""

import json
from pathlib import Path

import torch
import yaml

from flow2pc.data.temporal import SpatialTranscriptomics
from flow2pc.evaluation.runner import evaluate, publish_evaluation
from flow2pc.pretrain.models.mfm.snapshots import load_teacher
from flow2pc.utils.checkpoint import atomic_json


def evaluate_run(run_dir, device):
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    pair = json.loads((run_dir / "teacher_pair.json").read_text())
    binding = pair["binding"]
    from flow2pc.data.mfm_snapshots import read_manifest
    prepared = read_manifest(binding["prepared_path"])
    if prepared["identity"] != binding["snapshot_identity"] or prepared["split_protocol"] != "st_native_independent_validation":
        raise ValueError("ST ALI evaluation requires the bound native-protocol data")
    source = prepared["provenance"]
    contract = source["ali_dataset_contract"]
    dataset = SpatialTranscriptomics(name="st", root=contract["root"], seed=config["seed"],
                                    params=contract["params"], preprocessing=source["ali_preprocessing"])
    if dataset.contract() != contract:
        raise ValueError("ST source or dataset contract changed after training")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    teacher = load_teacher({"checkpoint_path": str(run_dir / "checkpoints/flow.pt"),
                            "geopath_checkpoint": str(run_dir / "checkpoints/geopath.pt"),
                            "repo_path": config["model"]["params"]["source_root"]}, device)

    class SharedTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = teacher.model
            self.manifest = teacher.metadata

        def ode_velocity(self, t, x):
            return self.model(t, x)

    return evaluate(SharedTeacher(), dataset, config["evaluation"], run_dir / "eval")


def publish(run_dir, report):
    run_dir = Path(run_dir)
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    tracking = config.get("wandb", {})
    if tracking.get("mode", "disabled") == "disabled":
        return
    sessions = [p for p in (run_dir / "wandb").glob("run-*") if p.is_dir()]
    if len(sessions) != 1:
        raise ValueError("Expected exactly one W&B session for this MFM teacher")
    import wandb
    run_id = sessions[0].name.rsplit("-", 1)[-1]
    run = wandb.init(project=tracking.get("project", "flow2pc"), group=tracking.get("group"),
                     id=run_id, resume="allow", mode=tracking["mode"], dir=str(run_dir))
    try:
        figures = json.loads((run_dir / "eval/media.json").read_text())
        scalars = {key: value for key, value in report.items()
                   if key.startswith("teacher_") and value is not None}
        publish_evaluation(run, report["scores"], scalars, figures, run_dir / "eval")
    finally:
        run.finish()
    atomic_json(run_dir / "eval/wandb_status.json", {"status": "complete", "run_id": run_id})

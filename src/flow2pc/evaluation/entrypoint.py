"""CLI checkpoint evaluation. Model construction is separate from shared metrics."""

import json
from pathlib import Path

from flow2pc.data.restore import restore_dataset
from flow2pc.evaluation.runner import evaluate
from flow2pc.pretrain.runner import load_teacher
from flow2pc.utils.checkpoint import atomic_json
from flow2pc.utils.tracking import LocalLogger


def _load_student(path, teacher, dataset, device):
    """Restore the shared HCLT checkpoint; never initialize or train its source."""
    import torch

    from flow2pc.distill.student import build_student

    saved = torch.load(path, map_location="cpu", weights_only=True)
    binding, model = saved["binding"], saved["model_config"]
    if binding["teacher_sha256"] != teacher.manifest["checkpoint_sha256"]:
        raise ValueError("Student checkpoint belongs to another teacher")
    if binding["dataset"] != dataset.contract():
        raise ValueError("Student checkpoint data/preprocessing differs from this teacher")
    if saved.get("best_step") != saved["step"]:
        raise ValueError("Student evaluation requires the selected best checkpoint")
    if model["name"] != "hclt" or model.get("fourier_zero_center", False):
        raise ValueError("Expected the shared free-source HCLT checkpoint format")
    student = build_student(
        model["parents"],
        {key: model[key] for key in ("num_latents", "time_embedding_dim", "time_feature_seed") if key in model},
        clamp_derivative=model.get("clamp_scale_derivative", False),
        time_order=model.get("time_order", 4),
    ).to(device)
    student.load_state_dict(saved["state_dict"], strict=True)
    student.eval()
    return student, saved["step"]


def run(config, role):
    if config.get("evaluation", {}).get("protocol") == "native_otpfm":
        if role != "teacher" or config["stage"] != "pretrain" or config["model"]["name"] != "otpfm":
            raise ValueError("Native OTP-FM evaluation requires an OTP-FM pretrain config and teacher role")
        from flow2pc.pretrain.models.otpfm.evaluation import evaluate_checkpoint

        return evaluate_checkpoint(config)
    if config.get("model", {}).get("name") == "otpfm":
        from flow2pc.pretrain.models.otpfm.profiles import MarginalProfile, get_profile

        if config["evaluation"].get("protocol") != "shared_ode" or not isinstance(get_profile(config["data"]["name"]), MarginalProfile):
            raise ValueError("Native transport OTP-FM uses native_otpfm; shared-data profiles use explicit shared_ode evaluation")
    if config["stage"] == "pretrain":
        if role != "teacher":
            raise ValueError("Student evaluation requires its distill config")
        run_dir = Path(config["output"]["dir"])
        teacher = load_teacher(
            run_dir / "checkpoints/last.pt", config["training"]["device"]
        )
        student, step = None, None
        dataset = restore_dataset(teacher.manifest["dataset"], teacher.preprocessing)
        reference = run_dir / "eval"
    else:
        # Imports for the student entry point never occur during teacher training.
        from flow2pc.distill.runner import prepare

        teacher, dataset, _ = prepare(config)
        reference = (
            Path(config["teacher"]["checkpoint"]).resolve().parent.parent / "eval"
        )
        if role == "student":
            run_dir = Path(config["output"]["dir"])
            student, step = _load_student(
                run_dir / "checkpoints/best.pt",
                teacher,
                dataset,
                config["distillation"]["device"],
            )
        else:
            # A separate teacher-only output/run is useful for legacy checkpoints.
            student_dir = Path(config["output"]["dir"])
            run_dir = student_dir.parent / (student_dir.name + "-teacher-reference")
            student, step = None, None
    settings = {**config["evaluation"], "enabled": True}
    run_dir.mkdir(parents=True, exist_ok=True)
    tracking_config = config
    if role == "teacher" and config["stage"] == "distill":
        tracking_config = {
            **config,
            "output": {"dir": str(run_dir)},
            "logging": {
                **config["logging"],
                "name": config["logging"].get("name", dataset.metadata.name)
                + "-teacher-reference",
            },
        }
    logger = LocalLogger(run_dir, tracking_config, append=True)
    try:
        evaluator = evaluate
        if config["stage"] == "pretrain" and dataset.metadata.name in ("arch", "sphere", "lidar"):
            from flow2pc.evaluation.prepared_mfm import evaluate as evaluator
        result = evaluator(
            teacher,
            dataset,
            settings,
            run_dir / "eval",
            student=student,
            logger=logger,
            checkpoint_step=step,
            reference_dir=reference,
        )
    except BaseException:
        logger.close(failed=True)
        raise
    logger.close()
    if config["stage"] == "pretrain":
        status_path = run_dir / "status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        status.pop("error", None)
        atomic_json(
            status_path,
            {**status, "status": "complete", "evaluation_status": "complete"},
        )
    return {
        "evaluation_dir": str((run_dir / "eval").resolve()),
        "media_manifest": str((run_dir / "eval/media.json").resolve()),
        "model": result["model"],
    }

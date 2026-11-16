"""Post-training teacher/student ODE evaluation; independent of PC implementations."""

import csv
import hashlib
import json
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from flow2pc.data.base import heldout_indices
from flow2pc.evaluation.config import TRANSPORT_W2_POPULATIONS, TRANSPORT_W2_PROTOCOL
from flow2pc.evaluation.figures import TRACKING_FRAMES, save_figures, validate_resources
from flow2pc.evaluation.ode import (
    emd,
    fixed_subset,
    metric_coordinates,
    raw_coordinates,
    rollout,
)
from flow2pc.utils.checkpoint import atomic_json, atomic_path, identity
from flow2pc.utils.experiment_protocols import dataset_tag


def population(dataset, index, split):
    if split == "all":
        if hasattr(dataset, "clouds"):
            return dataset.clouds[index]
        if dataset.preprocessing.get("kind") == "all_marginals_sequential_float32":
            return dataset.get_marginal(index, "all").materialize()
        return torch.cat(
            [
                dataset.get_marginal(index, name).materialize()
                for name in ("train", "val", "test")
            ]
        )
    return dataset.get_marginal(index, split).materialize()


def save_tensor(path, payload):
    with atomic_path(path) as temporary:
        torch.save(payload, temporary)


def population_scores(real, predicted, *, transport):
    scores = {"emd": emd(real, predicted)}
    if transport:
        count = min(len(real), len(predicted))
        scores["w2"] = emd(real[:count], predicted[:count], order=2)
    return scores


def evaluate(
    teacher,
    dataset,
    settings,
    output,
    *,
    student=None,
    logger=None,
    checkpoint_step=None,
    reference_dir=None,
):
    """Compare ODE flows from identical empirical source particles and model times.

    Settings explicitly record all-population versus held-out-row evaluation.
    Teacher/student observation EMD and paired flow error are separate metrics.
    """
    if settings.get("protocol") == "native_otpfm":
        raise ValueError("Use the OTP-FM native finite-interval evaluator for native_otpfm; shared ODE metrics are a separate protocol")
    transport = dataset_tag(dataset) in ("gom", "beijing")
    if transport and settings["metric_space"] != "raw":
        raise ValueError("Transport W2 pretraining evaluation requires evaluation.metric_space: raw")
    validate_resources(dataset, settings)
    source = fixed_subset(
        population(dataset, 0, settings["source_split"]),
        settings["num_samples"],
        settings["seed"],
    )
    if not len(source):
        raise ValueError("Evaluation source split is empty")
    times = torch.tensor(dataset.metadata.model_times, dtype=torch.float32)
    protocol = {k: v for k, v in settings.items() if k not in ("enabled", "image_dir")}
    protocol.update(
        source_index=0,
        initial_law="same_observed_source_cells_for_teacher_and_student",
        source_count=len(source),
        model_times=list(dataset.metadata.model_times),
        source_sha256=hashlib.sha256(source.contiguous().numpy().tobytes()).hexdigest(),
        emd_formula="uniform-mass W1 with Euclidean costs; full feature dimension",
        units=(
            "standardized_feature_coordinates"
            if settings["metric_space"] == "historical"
            and dataset.preprocessing.get("whiten")
            else "raw_feature_coordinates"
        ),
    )
    legacy_binding = identity(
        {
            "teacher": teacher.manifest,
            "dataset": dataset.contract(),
            "protocol": protocol,
        }
    )
    if transport:
        protocol.update(
            metric_protocol_version=TRANSPORT_W2_PROTOCOL,
            primary_distance="w2", primary_metric="teacher_heldout_w2",
            w2_formula="sqrt(uniform-mass OT with squared Euclidean costs); full feature dimension",
            w2_population_policy=TRANSPORT_W2_POPULATIONS,
            time_rule="exact observation times", sampler="instantaneous velocity ODE",
        )
    binding = identity({"teacher": teacher.manifest, "dataset": dataset.contract(), "protocol": protocol})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if transport and (output / "metrics.json").is_file():
        old_metrics = json.loads((output / "metrics.json").read_text())
        backup = output.with_name(output.name + "-w1-before-w2")
        if (old_metrics.get("protocol", {}).get("metric_protocol_version") != TRANSPORT_W2_PROTOCOL
                and not backup.exists()):
            shutil.copytree(output, backup)
    atomic_json(
        output / "status.json", {"status": "running", "binding_sha256": binding}
    )
    try:
        teacher_path, teacher_nfe, cache_reused = None, None, None
        cache_directories = [Path(reference_dir)] if reference_dir is not None else []
        cache_directories.append(output)
        for directory in cache_directories:
            cache = directory / "teacher_predictions.pt"
            if cache.is_file():
                cached = torch.load(cache, map_location="cpu", weights_only=True)
                # Adding a distance does not change a trajectory. The exact old
                # binding still checks teacher, data, preprocessing, particles,
                # solver and EVERY old evaluation setting before reuse.
                accepted = {binding, legacy_binding} if transport else {binding}
                if cached.get("binding_sha256") in accepted:
                    teacher_path, teacher_nfe = cached["teacher_ode"], cached["nfe"]
                    if (
                        teacher_path.shape != (len(times), len(source), dataset.dim)
                        or not torch.equal(cached["times"], times)
                        or not torch.isfinite(teacher_path).all()
                    ):
                        raise ValueError("Invalid cached teacher rollout")
                    cache_reused = "current_binding" if cached["binding_sha256"] == binding else "legacy_W1_binding"
                    break
        if teacher_path is None:
            teacher_path, teacher_nfe = rollout(teacher, source, times, settings)
        save_tensor(
            output / "teacher_predictions.pt",
            {
                "binding_sha256": binding,
                "teacher_ode": teacher_path,
                "times": times,
                "nfe": teacher_nfe,
            },
        )
        student_path, student_nfe = None, None
        if student is not None:
            student_path, student_nfe = rollout(
                student, source, times, settings, student=True
            )
            save_tensor(
                output / "predictions.pt",
                {
                    "binding_sha256": binding,
                    "times": times,
                    "student_ode": student_path,
                    "teacher_ode": teacher_path,
                },
            )
        indices = settings.get("time_indices") or (
            list(TRACKING_FRAMES)
            if dataset.metadata.name == "cell_tracking"
            else list(range(len(times)))
        )
        rows = []
        for index in indices:
            real = fixed_subset(
                population(dataset, index, settings["reference_split"]),
                settings["metric_samples"],
                settings["seed"] + index + 1,
            )
            if not len(real):
                continue
            predicted_teacher = fixed_subset(
                teacher_path[index], settings["metric_samples"], settings["seed"] + 1000
            )
            real = metric_coordinates(
                real, dataset.preprocessing, settings["metric_space"]
            )
            predicted_teacher = metric_coordinates(
                predicted_teacher, dataset.preprocessing, settings["metric_space"]
            )
            row = {
                "index": index,
                "physical_time": dataset.metadata.physical_times[index],
                "model_time": float(times[index]),
                "held_out": index in heldout_indices(dataset),
                "num_reference": len(real),
                "num_predicted": len(predicted_teacher),
            }
            row.update({"teacher_" + key: value for key, value in
                        population_scores(real, predicted_teacher, transport=transport).items()})
            if transport:
                row["w2_num_samples"] = min(len(real), len(predicted_teacher))
            if student_path is not None:
                predicted_student = fixed_subset(
                    student_path[index],
                    settings["metric_samples"],
                    settings["seed"] + 1000,
                )
                predicted_student = metric_coordinates(
                    predicted_student, dataset.preprocessing, settings["metric_space"]
                )
                for prefix, reference in (("student", real), ("student_teacher", predicted_teacher)):
                    row.update({prefix + "_" + key: value for key, value in
                                population_scores(reference, predicted_student, transport=transport).items()})
            rows.append(row)
        noninitial = [row for row in rows if row["index"] != 0]
        if not noninitial:
            raise ValueError(
                "No noninitial evaluation populations in the selected reference split"
            )
        scalars = {
            "teacher_mean_emd": float(np.mean([r["teacher_emd"] for r in noninitial])),
            "teacher_final_emd": next(
                (r["teacher_emd"] for r in rows if r["index"] == len(times) - 1), None
            ),
            "teacher_ode_nfe": teacher_nfe,
        }
        heldout_rows = [row for row in rows if row["held_out"]]
        heldout = next(iter(heldout_rows), None)
        if heldout is not None:
            scalars["teacher_heldout_emd"] = float(np.mean([r["teacher_emd"] for r in heldout_rows]))
        if student_path is not None:
            difference = raw_coordinates(
                student_path, dataset.preprocessing
            ) - raw_coordinates(teacher_path, dataset.preprocessing)
            paired_rmse = difference.square().sum(-1).mean(-1).sqrt()
            late = times >= 0.8
            scalars.update(
                student_mean_emd=float(np.mean([r["student_emd"] for r in noninitial])),
                student_final_emd=next(
                    (r["student_emd"] for r in rows if r["index"] == len(times) - 1),
                    None,
                ),
                student_teacher_mean_emd=float(
                    np.mean([r["student_teacher_emd"] for r in noninitial])
                ),
                teacher_student_final_paired_rmse_raw=float(paired_rmse[-1]),
                teacher_student_late_paired_rmse_raw=float(paired_rmse[late].mean()),
                teacher_student_max_paired_rmse_raw=float(paired_rmse.max()),
                student_ode_nfe=student_nfe,
            )
            if heldout is not None:
                scalars["student_heldout_emd"] = float(np.mean([r["student_emd"] for r in heldout_rows]))
        if checkpoint_step is not None:
            scalars["checkpoint_step"] = checkpoint_step
        if transport:
            for role in (("teacher", "student", "student_teacher") if student is not None else ("teacher",)):
                key = role + "_w2"
                scalars[role + "_mean_w2"] = float(np.mean([r[key] for r in noninitial]))
                scalars[role + "_final_w2"] = next((r[key] for r in rows if r["index"] == len(times) - 1), None)
                if heldout_rows:
                    scalars[role + "_heldout_w2"] = float(np.mean([r[key] for r in heldout_rows]))
        result = {
            "schema_version": 1,
            "model": "student" if student is not None else "teacher",
            "binding_sha256": binding,
            "protocol": protocol,
            "scores": rows,
            **scalars,
        }
        if transport:
            result["execution"] = {"teacher_rollout_reused": cache_reused is not None,
                                   "prediction_cache_binding": cache_reused}
        atomic_json(output / "metrics.json", result)
        with atomic_path(output / "marginal_metrics.csv") as temporary:
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        figures = save_figures(dataset, teacher_path, student_path, output, settings)
        atomic_json(
            output / "media.json",
            {key: str(path.resolve()) for key, path in figures.items()},
        )
        # Local computation is complete before optional remote publication.
        atomic_json(
            output / "status.json", {"status": "complete", "binding_sha256": binding}
        )
        if logger is not None:
            logger.log(
                {
                    f"eval/{key}": value
                    for key, value in scalars.items()
                    if value is not None
                }
            )
            if logger.run is not None:
                # Already-running teachers may hold the earlier logger class.
                guard = logger.remote("evaluation publication") if hasattr(logger, "remote") else nullcontext()
                with guard:
                    publish_evaluation(logger.run, rows, scalars, figures, output)
        return result
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise


def publish_evaluation(run, rows, scalars, figures, output):
    """Publish already-saved evaluation outputs; no model/data execution."""
    output = Path(output)
    if run is None:
        return
    import wandb

    run.log({
        "eval/marginal_metrics": wandb.Table(columns=list(rows[0]), data=[list(row.values()) for row in rows]),
        **{f"eval/{key}": wandb.Image(str(path)) for key, path in figures.items()},
        **{f"eval/marginal_{row['index']}/{key}": value for row in rows
           for key, value in row.items() if key.endswith(("emd", "w2"))},
    })
    run.summary.update({f"eval/{key}": value for key, value in scalars.items() if value is not None})
    artifact = wandb.Artifact(f"{run.id}-evaluation", type="evaluation")
    artifacts = [output / "metrics.json", output / "marginal_metrics.csv",
                 output / "media.json", output / "teacher_predictions.pt", *figures.values()]
    if (output / "predictions.pt").is_file():
        artifacts.append(output / "predictions.pt")
    for path in artifacts:
        artifact.add_file(str(path))
    run.log_artifact(artifact)

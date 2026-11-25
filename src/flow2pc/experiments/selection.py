"""Select one evaluated student per frozen teacher using a declared saved score."""

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from flow2pc.utils.experiment_protocols import dataset_tag


def read_json(path):
    path = Path(path)
    value = json.loads(path.read_text()) if path.is_file() else {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def implementation_identity():
    package = Path(__file__).parents[1]
    files = [*Path(__file__).parent.rglob("*.py")]
    files.extend(package / name for name in (
        "evaluation/ode.py", "evaluation/entrypoint.py", "distill/student.py",
        "pc/base.py", "pc/gaussian_pc.py", "pc/hclt_pc.py",
    ))
    return {str(path.relative_to(package)): digest(path) for path in sorted(files)}


def teacher_contract(source):
    path = Path(source["checkpoint"])
    if not path.is_file():
        raise ValueError(f"Missing frozen teacher: {path}")
    if source.get("legacy_metadata"):
        metadata = read_json(source["legacy_metadata"])
        expected = metadata.get("checkpoint_sha256")
        if not expected:
            raise ValueError("Historical teacher metadata needs its pinned checkpoint hash")
        return {"method": source["name"], "checkpoint_sha256": expected,
                "dataset": {"name": metadata["data"]["name"]}}, digest(source["legacy_metadata"])
    manifest = read_json(path.parent.parent / "manifest.json")
    if manifest.get("method") != source["name"] or not manifest.get("checkpoint_sha256"):
        raise ValueError("Missing or incompatible teacher manifest")
    if (path.parent.parent / manifest["checkpoint_relpath"]).resolve() != path.resolve():
        raise ValueError("Teacher path is not the manifest-selected checkpoint")
    return manifest, None


def validation_contract(binding):
    """Compare validation queries, never capacity/LR or population EMD settings."""
    settings = binding.get("settings")
    if not isinstance(settings, dict) or not binding.get("dataset") or binding.get("seed") is None:
        raise ValueError("Missing saved dataset/seed/validation settings for MSE selection")
    return {
        "dataset": identity(binding["dataset"]), "seed": binding["seed"],
        "sampler": binding.get("sampler"),
        "temporal_validation": binding.get("validation_protocol"),
        "validation_scope": settings.get("validation_scope", "dataset"),
        "validation_batches": settings.get("validation_batches", 4),
        "validation_batch_size": settings.get("validation_batch_size", 128),
        "teacher_path_particles": settings.get("teacher_path_particles", 128),
    }


def select_student(config):
    """Metadata only. Weight hashes/bindings are checked again in the worker."""
    source = config["teacher"]
    teacher, metadata_sha = teacher_contract(source)
    expected_dataset = config["dataset"].removesuffix("_w2")
    actual_dataset = dataset_tag(SimpleNamespace(metadata=SimpleNamespace(
        name=teacher["dataset"]["name"], dim=teacher["dataset"].get("dim"))))
    if actual_dataset != expected_dataset:
        raise ValueError("Teacher dataset differs from experiment profile")
    generator_sha = digest(source["generator_checkpoint"]) if source.get("generator_checkpoint") else None
    roots = [Path(p) for p in config["selection"]["student_roots"]]
    directories = sorted({p.parent for root in roots if root.is_dir()
                          for p in root.rglob("manifest.json")})
    candidates, rejected = [], []
    for directory in directories:
        try:
            binding = read_json(directory / "manifest.json")
            if binding.get("artifact_type") != "student":
                continue
            if (binding.get("teacher_sha256") != teacher["checkpoint_sha256"]
                    or binding.get("legacy_metadata_sha256") != metadata_sha
                    or binding.get("generator_sha256") != generator_sha):
                raise ValueError("Different frozen teacher/metadata/generator")
            state = read_json(directory / "status.json")
            if state.get("status") != "complete":
                raise ValueError("Student run is not complete")
            report = read_json(directory / "eval/population/evaluation.json")
            for item in [report, *report.get("by_time", {}).values()]:
                if any(key.startswith(("teacher_ode", "student_ode")) and key.endswith("error") and value
                       for key, value in item.items()):
                    raise ValueError("Saved teacher/student ODE evaluation has numerical errors")
            if report.get("checkpoint_step") != state.get("best_step") or state.get("best_step") is None:
                raise ValueError("Population report does not describe the saved selected checkpoint")
            checkpoint = directory / "checkpoints/best.pt"
            if not checkpoint.is_file() or not (directory / "resolved_config.yaml").is_file():
                raise ValueError("Missing selected weights/resolved configuration")
            metric = config["selection"]["metric"]
            if metric == "auto":
                metric = "student_ode_emd_heldout" if report.get("heldout_indices") else "student_ode_emd_mean"
            mse_selection = metric == "validation_selection_score"
            score = state.get("best_selection_score") if mse_selection else report.get(metric)
            if type(score) not in (int, float) or not math.isfinite(score) or score < 0:
                raise ValueError(f"Missing finite saved {metric}")
            protocol = ({"metric": metric, "validation": validation_contract(binding)} if mse_selection else
                        {"space": report.get("space"), "protocol": report.get("protocol"),
                         "heldout_indices": report.get("heldout_indices"), "metric": metric})
            reference_metric = "teacher_ode_emd_heldout" if report.get("heldout_indices") else "teacher_ode_emd_mean"
            candidates.append({"checkpoint": str(checkpoint), "run_dir": str(directory),
                               "score": score, "metric": metric, "checkpoint_step": state["best_step"],
                               "teacher_emd": report.get(reference_metric if mse_selection else metric.replace("student_", "teacher_", 1)),
                               "protocol": protocol, "binding_sha256": binding.get("binding_sha256")})
        except (ValueError, KeyError, TypeError, OSError) as error:
            rejected.append({"run_dir": str(directory), "reason": str(error)})
    if not candidates:
        raise ValueError(f"No completed, teacher-matched candidate for {source['name']}/{config['dataset']}; "
                         f"rejections: {rejected[:3]}")
    if len({identity(c["protocol"]) for c in candidates}) != 1:
        raise ValueError("Candidate validation or EMD protocols/units differ; narrow selection.student_roots to a comparable cohort")
    pinned = config["selection"].get("pinned_checkpoint")
    if pinned is not None:
        matches = [c for c in candidates if Path(c["checkpoint"]).resolve() == Path(pinned).resolve()]
        if not matches:
            raise ValueError("The launch-pinned student is no longer an eligible completed candidate")
        chosen = matches[0]
    else:
        chosen = min(candidates, key=lambda c: (c["score"], c["checkpoint"]))
    mse_selection = chosen["metric"] == "validation_selection_score"
    progress = []
    for root in roots:
        saved = read_json(root / "selected_student.json")
        if saved.get("teacher_sha256") == teacher["checkpoint_sha256"] and type(saved.get("planned")) is int:
            progress.append({"root": str(root), "planned": saved["planned"],
                             "eligible_completed": sum(Path(c["run_dir"]).is_relative_to(root) for c in candidates)})
    final = (all(p["eligible_completed"] == p["planned"] for p in progress) if progress else None)
    return {**chosen, "teacher_sha256": teacher["checkpoint_sha256"],
            "candidate_count": len(candidates), "candidates": candidates, "rejected": rejected,
            "selection_cohorts": progress, "selection_final": final,
            "across_runs_selection": ("lowest saved validation velocity-MSE selection score, for this exact teacher"
                                      if mse_selection else "lowest saved student ODE W1 EMD, for this exact teacher"),
            "within_run_selection": "existing best.pt chosen by validation velocity MSE; no weights changed",
            "data_reuse": "retrospective evaluation; selection populations are not claimed to be untouched test data"}

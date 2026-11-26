"""Strict stage schema; algorithm parameters are validated by their method."""

import copy
import os
import re
from pathlib import Path

import yaml

from flow2pc.data.registry import datasets
from flow2pc.pretrain.registry import teachers


def mapping(value, label, allowed, required=()):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")  # noqa: TRY004
    unknown, missing = set(value) - set(allowed), set(required) - set(value)
    if unknown or missing:
        raise ValueError(
            f"{label}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}"
        )
    return value


def expand(value):
    if isinstance(value, str):

        def replace(match):
            key = match.group(1)
            if not os.environ.get(key):
                raise ValueError(f"Environment variable {key} is unset or empty")
            return os.environ[key]

        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, value)
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value


def resolve_path(value, base, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty path")
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def load_config(path, overrides=()):
    path = Path(path).resolve()
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    for override in overrides:
        if "=" not in override:
            raise ValueError("Overrides must use dotted.key=value")
        name, value = override.split("=", 1)
        keys = name.split(".")
        target = raw
        if not all(keys):
            raise ValueError("Override keys cannot be empty")
        for key in keys[:-1]:
            target = target.setdefault(key, {})
            if not isinstance(target, dict):
                raise ValueError(f"Override parent is not a mapping: {name}")
        target[keys[-1]] = yaml.safe_load(value)
    return resolve_config(raw, base=path.parent)


def resolve_config(raw, *, base):
    raw = expand(copy.deepcopy(raw))
    if isinstance(raw, dict) and "experiment" in raw:
        from flow2pc.experiments.config import resolve

        return resolve(raw, base)
    if isinstance(raw, dict) and (raw.get("stage") == "evaluate" or (
        raw.get("stage") == "pretrain"
        and isinstance(raw.get("model"), dict)
        and raw["model"].get("name") in ("mfm", "mmfm")
    ) or (
        raw.get("stage") == "distill"
        and isinstance(raw.get("teacher"), dict)
        and raw["teacher"].get("name") in ("mfm", "mmfm")
    )):
        from flow2pc.utils.metric_config import prepare

        return prepare(raw)
    config = expand(copy.deepcopy(raw))
    mapping(
        config,
        "config",
        {
            "stage",
            "seed",
            "model",
            "data",
            "training",
            "teacher",
            "student",
            "distillation",
            "output",
            "logging",
            "evaluation",
        },
        {"stage", "output"},
    )
    if config["stage"] not in ("pretrain", "distill"):
        raise ValueError("stage must be pretrain or distill")
    config["seed"] = config.get("seed", 42)
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    output = mapping(config["output"], "output", {"dir"}, {"dir"})
    output["dir"] = resolve_path(output["dir"], base, "output.dir")
    logging = mapping(
        config.setdefault("logging", {}),
        "logging",
        {"mode", "project", "entity", "name"},
    )
    if logging.setdefault("mode", "disabled") not in ("disabled", "offline", "online"):
        raise ValueError("logging.mode must be disabled, offline or online")
    logging.setdefault("project", "flow2pc")
    from flow2pc.evaluation.config import resolve_evaluation

    config["evaluation"] = resolve_evaluation(config.get("evaluation", {}))
    if "image_dir" in config["evaluation"]:
        config["evaluation"]["image_dir"] = resolve_path(
            config["evaluation"]["image_dir"], base, "evaluation.image_dir"
        )
    if config["stage"] == "pretrain":
        if {"teacher", "student", "distillation"} & config.keys():
            raise ValueError("Pretrain cannot contain distillation-stage keys")
        for key in ("model", "data", "training"):
            if key not in config:
                raise ValueError(f"Missing pretrain config key: {key}")
        model = mapping(config["model"], "model", {"name", "params"}, {"name"})
        teachers.require(model["name"])
        if not isinstance(model.setdefault("params", {}), dict):
            raise ValueError("model.params must be a mapping")
        data = mapping(config["data"], "data", {"name", "root", "params"}, {"name"})
        datasets.require(data["name"])
        if not isinstance(data.setdefault("params", {}), dict):
            raise ValueError("data.params must be a mapping")
        fallback = os.environ.get("FLOW2PC_DATA_ROOT", str(Path.cwd() / "data"))
        data["root"] = resolve_path(data.get("root", fallback), base, "data.root")
        if not isinstance(config["training"], dict):
            raise ValueError("training must be a mapping")
    else:
        if {"model", "data", "training"} & config.keys():
            raise ValueError(
                "Distill data/preprocessing must come from the teacher manifest"
            )
        for key in ("teacher", "student", "distillation"):
            if key not in config:
                raise ValueError(f"Missing distill config key: {key}")
        teacher = mapping(
            config["teacher"],
            "teacher",
            {"name", "checkpoint", "format", "legacy_metadata", "generator_checkpoint", "data_root"},
            {"checkpoint"},
        )
        teacher["checkpoint"] = resolve_path(
            teacher["checkpoint"], base, "teacher.checkpoint"
        )
        if "data_root" in teacher:
            teacher["data_root"] = resolve_path(teacher["data_root"], base, "teacher.data_root")
        if "name" in teacher and teacher["name"] not in ("alicfm", "otpfm"):
            raise ValueError("Use the method-specific schema for this teacher")
        teacher_format = teacher.setdefault(
            "format", "legacy" if "legacy_metadata" in teacher else "framework"
        )
        if teacher_format not in ("framework", "legacy"):
            raise ValueError("teacher.format must be framework or legacy")
        if teacher_format == "legacy" and "legacy_metadata" not in teacher:
            raise ValueError(
                "Existing checkpoints require their explicit legacy_metadata contract"
            )
        if teacher.get("name") == "otpfm" and teacher_format != "framework":
            raise ValueError("OTP distillation requires a migrated framework checkpoint with saved native configuration")
        if teacher_format == "framework" and (
            "legacy_metadata" in teacher or "generator_checkpoint" in teacher
        ):
            raise ValueError(
                "Legacy metadata/generator options require teacher.format: legacy"
            )
        for key in ("legacy_metadata", "generator_checkpoint"):
            if key in teacher:
                teacher[key] = resolve_path(teacher[key], base, f"teacher.{key}")
        student = mapping(config["student"], "student", {"name", "params"}, {"name"})
        if student["name"] != "hclt":
            raise ValueError("The retained distillation student is hclt")
        if not isinstance(student.setdefault("params", {}), dict):
            raise ValueError("student.params must be a mapping")
        d = mapping(
            config["distillation"],
            "distillation",
            {
                "sampler",
                "steps",
                "batch_size",
                "lr",
                "device",
                "resume",
                "validate_every",
                "checkpoint_every",
                "log_every",
                "validation_batches",
                "validation_batch_size",
                "teacher_path_particles",
                "validation_scope",
                "validation_population_metrics",
                "source_nll_weight",
                "student_data",
            },
            {"sampler", "steps", "batch_size", "lr"},
        )
        mapping(d["sampler"], "distillation.sampler", {"name", "params"}, {"name"})
        if not isinstance(d["sampler"]["name"], str) or not d["sampler"]["name"]:
            raise ValueError("sampler.name must be a nonempty string")
        if not isinstance(d["sampler"].setdefault("params", {}), dict):
            raise ValueError("sampler.params must be a mapping")
        if teacher.get("name") == "otpfm" and d["sampler"]["name"] != "otpfm_conditional":
            raise ValueError("OTP-FM requires sampler.name: otpfm_conditional")
        from math import isfinite

        from flow2pc.utils.experiment_protocols import validation_scope
        validation_scope(None, {"validation_scope": d.get("validation_scope", "observed_rows")})
        if type(d.get("validation_population_metrics", True)) is not bool:
            raise ValueError("validation_population_metrics must be boolean")
        source_weight = d.get("source_nll_weight", 0.0)
        if (type(source_weight) not in (int, float)
                or not isfinite(source_weight) or source_weight < 0):
            raise ValueError("distillation.source_nll_weight must be finite and nonnegative")
        if "student_data" in d:
            student_data = mapping(d["student_data"], "distillation.student_data",
                                   {"name", "samples_per_time", "seed"},
                                   {"name", "samples_per_time", "seed"})
            if (student_data["name"] != "independent_knot"
                    or type(student_data["samples_per_time"]) is not int
                    or student_data["samples_per_time"] < 11
                    or type(student_data["seed"]) is not int
                    or not 0 <= student_data["seed"] < 2**32):
                raise ValueError("student_data requires independent_knot, >=11 samples per time and an unsigned seed")

        for key in ("steps", "batch_size"):
            if type(d[key]) is not int or d[key] < 1:
                raise ValueError(f"distillation.{key} must be a positive integer")
        if type(d["lr"]) not in (int, float) or not isfinite(d["lr"]) or d["lr"] <= 0:
            raise ValueError("distillation.lr must be finite and positive")
        for key in (
            "validate_every",
            "checkpoint_every",
            "log_every",
            "validation_batches",
            "validation_batch_size",
            "teacher_path_particles",
        ):
            if key in d and (type(d[key]) is not int or d[key] < 1):
                raise ValueError(f"distillation.{key} must be a positive integer")
        if type(d.setdefault("resume", False)) is not bool:
            raise ValueError("distillation.resume must be boolean")
        if not isinstance(d.setdefault("device", "cpu"), str):
            raise ValueError("distillation.device must be a device string")
    return config

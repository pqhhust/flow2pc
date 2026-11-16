"""Shared experiment schema, resolved without Torch, data loads or output writes."""

from pathlib import Path
from importlib import import_module

from flow2pc.experiments import EXPERIMENTS


def resolve(raw, base):
    from flow2pc.utils.config import mapping, resolve_path

    mapping(raw, "experiment config", {
        "stage", "experiment", "teacher", "dataset", "selection", "device",
        "evaluation", "output", "logging",
    }, {"stage", "experiment", "teacher", "dataset", "selection", "output"})
    if raw["stage"] != "evaluate" or raw["experiment"] not in EXPERIMENTS:
        raise ValueError("Expected stage: evaluate and a registered experiment")
    if not isinstance(raw["dataset"], str) or not raw["dataset"]:
        raise ValueError("dataset must be an explicit profile name")
    teacher = mapping(raw["teacher"], "teacher", {
        "name", "checkpoint", "legacy_metadata", "generator_checkpoint", "data_root",
    }, {"name", "checkpoint"})
    if teacher["name"] not in ("alicfm", "otpfm"):
        raise ValueError("E3.1 currently supports ALI-CFM and OTP-FM")
    for key in ("checkpoint", "legacy_metadata", "generator_checkpoint", "data_root"):
        if key in teacher:
            teacher[key] = resolve_path(teacher[key], base, f"teacher.{key}")
    if teacher["name"] != "alicfm" and "legacy_metadata" in teacher:
        raise ValueError("Only ALI-CFM supports explicit historical metadata")
    selection = mapping(raw["selection"], "selection", {"student_roots", "metric", "pinned_checkpoint"}, {"student_roots"})
    roots = selection["student_roots"]
    if not isinstance(roots, list) or not roots:
        raise ValueError("selection.student_roots must be a nonempty list")
    selection["student_roots"] = [resolve_path(p, base, "student_roots") for p in roots]
    if "pinned_checkpoint" in selection:
        selection["pinned_checkpoint"] = resolve_path(selection["pinned_checkpoint"], base, "selection.pinned_checkpoint")
    if selection.setdefault("metric", "auto") not in (
        "auto", "student_ode_emd_heldout", "student_ode_emd_mean",
    ):
        raise ValueError("Select by saved student ODE W1 EMD, never direct-PC EMD")
    raw.setdefault("device", "cpu")
    if not isinstance(raw["device"], str):
        raise ValueError("device must be a device string")
    output = mapping(raw["output"], "output", {"dir"}, {"dir"})
    output["dir"] = resolve_path(output["dir"], base, "output.dir")
    logging = mapping(raw.setdefault("logging", {}), "logging", {"mode", "project", "entity", "name"})
    if logging.setdefault("mode", "online") not in ("online", "offline", "disabled"):
        raise ValueError("Invalid logging.mode")
    if logging.setdefault("project", "flow2pc") != "flow2pc":
        raise ValueError("These experiments log to project flow2pc")
    logging.setdefault("name", f"{raw['dataset']}-{teacher['name']}-{raw['experiment']}")
    module = import_module(EXPERIMENTS[raw["experiment"]] + ".config")
    raw["evaluation"] = module.resolve(raw.get("evaluation", {}))
    if raw["evaluation"].get("source_split") == "analytic_knot" and raw["dataset"] != "knot":
        raise ValueError("Analytic knot evaluation is available only for dataset: knot")
    # Never let evaluation overwrite its input student or teacher directory.
    out = Path(output["dir"])
    if any(out == Path(root) or out.is_relative_to(Path(root)) for root in selection["student_roots"]):
        raise ValueError("Experiment output must be outside the student search roots")
    path = Path(teacher["checkpoint"])
    teacher_run = path.parent if "legacy_metadata" in teacher else path.parent.parent
    if out == teacher_run or out.is_relative_to(teacher_run):
        raise ValueError("Experiment output must be outside the frozen teacher directory")
    return raw

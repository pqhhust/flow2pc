"""Frozen checkpoint adapters with explicitly declared ODE initial laws."""

from pathlib import Path

import numpy as np
import torch

from flow2pc.data.base import heldout_indices
from flow2pc.data.restore import restore_dataset
from flow2pc.evaluation.entrypoint import _load_student
from flow2pc.evaluation.ode import raw_coordinates, rollout
from flow2pc.evaluation.runner import population
from flow2pc.experiments.selection import digest
from flow2pc.pretrain.runner import load_teacher
from flow2pc.utils.config import load_config


def load_pair(config, selection):
    source = config["teacher"]
    if digest(source["checkpoint"]) != selection["teacher_sha256"]:
        raise ValueError("Teacher weight hash differs from selected student binding")
    teacher = load_teacher(source["checkpoint"], config["device"],
                           legacy_metadata=source.get("legacy_metadata"),
                           generator_checkpoint=source.get("generator_checkpoint"))
    dataset = restore_dataset(teacher.manifest["dataset"], teacher.preprocessing,
                              data_root=source.get("data_root"))
    saved_config = load_config(Path(selection["run_dir"]) / "resolved_config.yaml")
    for key in ("legacy_metadata", "generator_checkpoint"):
        old, new = saved_config["teacher"].get(key), source.get(key)
        if (old is None) != (new is None) or (old and digest(old) != digest(new)):
            raise ValueError(f"Selected student has different {key}")
    student, step = _load_student(Path(selection["checkpoint"]), teacher, dataset, config["device"])
    if step != selection["checkpoint_step"]:
        raise ValueError("Student weights and EMD report have different steps")
    teacher.eval().requires_grad_(False)
    # PC velocity internally differentiates its density. Do not put inference
    # under inference_mode or change its mathematical implementation.
    student.eval()
    return teacher, student, dataset


def evaluation_times(dataset, settings):
    heldout = heldout_indices(dataset)
    chosen = settings["time_indices"]
    if chosen is None:
        chosen = heldout or list(range(1, len(dataset.metadata.model_times)))
    if any(index >= len(dataset.metadata.model_times) for index in chosen):
        raise ValueError("Evaluation time index outside the saved dataset grid")
    # Earliest observations define the empirical initial law, not a test of a
    # transported prediction. Frame 1 remains in the audit as an excluded source.
    chosen = [index for index in chosen if index != 0]
    if not chosen:
        raise ValueError("No noninitial calibration times")
    return chosen


def real_population(dataset, index, settings):
    if settings["reference_split"] == "analytic_knot":
        from flow2pc.experiments.analytic_knot import reference

        return reference(dataset, index, settings)
    split = settings["reference_split"]
    if split == "protocol":
        counts = dataset.contract()["split_counts"][index]
        split = "test" if counts.get("test", 0) else "val"
    values = population(dataset, index, split)
    # No fallback to training/all observations when a reserved split is empty.
    count = len(values)
    if count > settings["max_test_samples"]:
        rng = torch.Generator().manual_seed(2026 + index)
        values = values[torch.randperm(count, generator=rng)[:settings["max_test_samples"]]]
    return raw_coordinates(values, dataset.preprocessing).numpy(), split, count


def training_coordinates(dataset, limit):
    """Fit scoring coordinates only on observed training rows, never test rows."""
    observed = list(dataset.contract()["observed_indices"])
    per_time = max(2, (limit + len(observed) - 1) // len(observed))
    rng = torch.Generator().manual_seed(8137)
    pieces = []
    for index in observed:
        values = population(dataset, index, "train")
        if len(values):
            subset = values[torch.randperm(len(values), generator=rng)[:per_time]]
            pieces.append(raw_coordinates(subset, dataset.preprocessing).numpy())
    if not pieces:
        raise ValueError("No observed training data for the fixed scoring representation")
    return np.concatenate(pieces)[:limit]


def bank_plan(source_count, settings):
    if source_count < 2:
        raise ValueError("The declared empirical source needs at least two cells")
    maximum = max(settings["bank_sizes"])
    sizes = {"fit": maximum, "tune": settings["tune_samples"], "threshold": maximum,
             "energy_a": settings["energy_samples"], "energy_b": settings["energy_samples"]}
    plans = {}
    for seed in settings["seeds"]:
        children = np.random.SeedSequence(seed).spawn(len(sizes))
        plans[seed] = {name: np.random.default_rng(child).integers(source_count, size=count)
                       for (name, count), child in zip(sizes.items(), children)}
    return plans


def paired_banks(teacher, student, dataset, times, settings, *, include_student=True):
    if settings["source_split"] == "analytic_knot":
        from flow2pc.experiments.analytic_knot import source_banks

        source, plans, sampling = source_banks(dataset, settings)
        unique = np.arange(len(source))
    else:
        source = population(dataset, 0, settings["source_split"])
        plans = bank_plan(len(source), settings)
        requested = np.concatenate([indices for plan in plans.values() for indices in plan.values()])
        unique = np.unique(requested)
        sampling = {"initial_law": "empirical earliest marginal, resampled with replacement",
                    "source_split": settings["source_split"], "source_count": len(source),
                    "unique_source_rows_integrated": len(unique),
                    "source_resampling_with_replacement": True,
                    "bank_independence": "independent RNG streams; shared source indices across roles"}
    grid = torch.tensor([dataset.metadata.model_times[i] for i in [0, *times]], dtype=torch.float32)
    result, nfe = {}, {}
    # The empirical mode integrates distinct requested source rows once. The
    # analytic mode integrates independent fresh initial particles for each bank.
    # Both use identical particles across teacher/student, with no real pairings.
    models = (("teacher", teacher), ("student", student)) if include_student else (("teacher", teacher),)
    for role, model in models:
        path, nfe[role] = rollout(model, source[unique], grid, settings, student=role == "student")
        result[role] = raw_coordinates(path[1:], dataset.preprocessing).numpy()
    mapped = {seed: {name: np.searchsorted(unique, indices) for name, indices in plan.items()}
              for seed, plan in plans.items()}
    return result, mapped, {**sampling, "nfe": nfe, "rollout_origin": "earliest"}


def query_banks(teacher, student, dataset, times, settings):
    """Numerical teacher banks and independent direct-PC student banks.

    The common array offsets are storage addresses, never particle pairings.
    Student draws do not inherit the teacher's finite empirical source IDs.
    """
    from time import perf_counter

    start = perf_counter()
    teacher_paths, old_plans, sampling = paired_banks(
        teacher, student, dataset, times, settings, include_student=False)
    teacher_seconds = perf_counter() - start
    plans, requested, cursor = {}, [], 0
    for seed, plan in old_plans.items():
        plans[seed] = {}
        for name, indices in plan.items():
            plans[seed][name] = np.arange(cursor, cursor + len(indices))
            requested.append(indices)
            cursor += len(indices)
    paths = {"teacher": teacher_paths["teacher"][:, np.concatenate(requested)]}
    parameter = next(student.parameters())
    start = perf_counter()
    draws = []
    with torch.no_grad():
        for index in times:
            pieces = []
            time = parameter.new_tensor(dataset.metadata.model_times[index])
            for seed, plan in plans.items():
                for bank_number, indices in enumerate(plan.values()):
                    rng = np.random.default_rng(np.random.SeedSequence([seed, index, bank_number, 31032]))
                    generator = torch.Generator(device=parameter.device).manual_seed(int(rng.integers(0, 2**63 - 1)))
                    for offset in range(0, len(indices), settings["batch_size"]):
                        values = student.sample(min(settings["batch_size"], len(indices) - offset), time, generator)
                        pieces.append(raw_coordinates(values.cpu(), dataset.preprocessing).numpy())
            draws.append(np.concatenate(pieces))
    paths["student"] = np.stack(draws)
    return paths, plans, {**sampling, "nfe": {**sampling["nfe"], "student": 0},
        "teacher_inference": "ODE pushforward from declared teacher initial law",
        "student_inference": "direct_pc", "direct_pc_sampling": True,
        "student_initial_law": "the saved PC's own normalized law; no empirical-source substitution",
        "bank_independence": "separate RNG streams per role, time, seed and bank; teacher source-ID overlap expected",
        "cross_model_particle_pairing": False,
        "teacher_bank_preparation_seconds": teacher_seconds,
        "student_direct_sampling_seconds": perf_counter() - start,
        "timing_note": "end-to-end bank preparation, not a controlled per-query speed benchmark"}

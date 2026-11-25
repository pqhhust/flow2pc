"""E3.2: frozen teacher/student ODE banks, same-time missing-coordinate tasks."""

import fcntl
from pathlib import Path

import numpy as np
import yaml

from flow2pc.experiments.adapters import (
    evaluation_times, load_pair, paired_banks, real_population, training_coordinates,
)
from flow2pc.experiments.analytic_knot import array_sha256
from flow2pc.experiments.e3_2.conditional import ConditionalKDE, masks_for_dimension
from flow2pc.experiments.e3_2.figures import save as save_figures
from flow2pc.experiments.e3_2.metrics import aggregate, conditional_scores
from flow2pc.experiments.io import write_csv
from flow2pc.experiments.selection import digest, identity, implementation_identity, read_json, select_student
from flow2pc.utils.checkpoint import atomic_json, atomic_path
from flow2pc.utils.tracking import LocalLogger

METRICS = ("coverage90", "hdr_ece", "energy_score", "conditional_crps",
           "estimated_conditional_nll_per_dim", "interval_coverage_ece", "mean_interval_score",
           "effective_components_mean", "low_ess_fraction", "independent_model_mass90")


def summarize(rows, settings):
    primary = [r for r in rows if r["bank_size"] == max(settings["bank_sizes"])
               and r["conditional_draws"] == max(settings["conditional_draw_sizes"])
               and r["bandwidth"] == settings["primary_bandwidth"]]
    summary = []
    for role in ("teacher", "student"):
        result = {"role": role, "bank_size": max(settings["bank_sizes"]),
                  "conditional_draws": max(settings["conditional_draw_sizes"]),
                  "bandwidth": settings["primary_bandwidth"]}
        for metric in METRICS:
            means = []
            for seed in settings["seeds"]:
                chosen = [r for r in primary if r["role"] == role and r["seed"] == seed
                          and r["nominal_alpha"] == (.9 if metric in ("coverage90", "independent_model_mass90")
                                                       else settings["levels"][0])]
                key = {"coverage90": "coverage", "independent_model_mass90": "independent_model_mass"}.get(metric, metric)
                # Mean interval score averages levels; other scores repeat by level.
                if metric == "mean_interval_score":
                    chosen = [r for r in primary if r["role"] == role and r["seed"] == seed]
                if chosen:
                    means.append(float(np.mean([r[key] for r in chosen])))
            result[metric] = float(np.mean(means)) if means else None
            result[metric + "_mc_std"] = float(np.std(means)) if means else None
        result["uncertainty_note"] = "MC standard deviation only; repeated masked views are not independent test observations"
        summary.append(result)
    return summary


def run(config):
    selection = select_student(config)
    selection["student_sha256"] = digest(selection["checkpoint"])
    output = Path(config["output"]["dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another experiment worker owns {output}") from error
        return _run(config, selection, output)


def _run(config, selection, output):
    settings = config["evaluation"]
    binding = {"config": {k: v for k, v in config.items() if k not in ("device", "logging", "output")},
               "teacher_sha256": selection["teacher_sha256"], "student_sha256": selection["student_sha256"],
               "implementation": implementation_identity()}
    fingerprint = identity(binding)
    previous, old_binding = read_json(output / "status.json"), read_json(output / "binding.json")
    if old_binding and old_binding != binding:
        raise ValueError("Experiment inputs/code changed; use a new output.dir")
    if previous.get("status") == "complete":
        result, media = read_json(output / "results.json"), read_json(output / "media.json")
        if (previous.get("binding_sha256") != fingerprint or not (output / "summary.csv").is_file()
                or result.get("selection", {}).get("student_sha256") != selection["student_sha256"]
                or not media or any(not Path(p).is_file() for p in media.values())):
            raise ValueError("Completed E3.2 artifacts are incomplete")
        return result
    atomic_json(output / "binding.json", binding)
    atomic_json(output / "selection.json", selection)
    with atomic_path(output / "resolved_config.yaml") as temporary:
        temporary.write_text(yaml.safe_dump(config, sort_keys=False))
    atomic_json(output / "status.json", {"status": "running", "binding_sha256": fingerprint})
    logger = None
    try:
        teacher, student, dataset = load_pair(config, selection)
        masks = masks_for_dimension(settings["observed_masks"], dataset.dim)
        times = evaluation_times(dataset, settings)
        training = training_coordinates(dataset, settings["representation_samples"])
        center, scale = training.mean(0), training.std(0)
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("E3.2 needs nonconstant observed-training coordinates for standardization")
        representation = {"center": center.tolist(), "scale": scale.tolist(), "state_dim": dataset.dim,
                          "fit_scope": "observed training rows only", "projection": None,
                          "score_space": "hidden coordinates, training-standardized",
                          "conditional_law": "Gaussian joint KDE conditioned analytically; evaluation-only smoothing"}
        representation_sha = identity(representation)
        atomic_json(output / "representation.json", {**representation, "sha256": representation_sha})
        atomic_json(output / "dataset_contract.json", dataset.contract())
        references, audit = {}, []
        for index in times:
            values, split, count = real_population(dataset, index, settings)
            audit.append({"experiment": "e3_2", "method": config["teacher"]["name"], "dataset": config["dataset"],
                          "time_index": index, "physical_time": dataset.metadata.physical_times[index],
                          "model_time": dataset.metadata.model_times[index], "state_dim": dataset.dim,
                          "reference_split": split, "n_available": count, "n_evaluated": len(values),
                          "reference_sha256": array_sha256(values), "eligible": len(values) >= settings["min_test_samples"],
                          "split_definition": dataset.contract()["split_scheme"],
                          "data_reuse": "retrospective; teacher selection/preprocessing may have used temporal holdouts",
                          "group_ids_available": dataset.metadata.has_trajectory_ids,
                          "observational_ci": "not estimated; independent groups not established"})
            if len(values) >= settings["min_test_samples"]:
                references[index] = (values - center) / scale
        write_csv(output / "dataset_audit.csv", audit)
        times = [i for i in times if i in references]
        if not times:
            raise ValueError("No eligible reserved observations for E3.2; see dataset_audit.csv")
        logger = LocalLogger(output, config, append=(output / "metrics.jsonl").is_file())
        logger.log({"e3_2/phase": "sampling_frozen_ODEs"})
        print(f"E3.2 {config['teacher']['name']}/{config['dataset']}: {len(times)} times, {len(masks)} masks; frozen ODE banks", flush=True)
        paths, plans, sampling = paired_banks(teacher, student, dataset, times, settings)
        paths = {role: (values - center) / scale for role, values in paths.items()}
        atomic_json(output / "sampling.json", {**sampling, "direct_pc_sampling": False,
                    "conditional_law": representation["conditional_law"],
                    "bandwidth_selection": "predeclared primary and sensitivity values; never chosen using test coverage",
                    "smoothing_scope": "evaluation estimator only; teacher/student ODEs and initial law unchanged",
                    "random_streams": "independent threshold / score-a / score-b; common random numbers across roles",
                    "masks": [{"observed": o, "hidden": h} for o, h in masks]})
        rows, observations = [], []
        largest = max(settings["conditional_draw_sizes"])
        for position, index in enumerate(times):
            real = references[index]
            for seed, plan in plans.items():
                for mask_number, (observed, hidden) in enumerate(masks):
                    mask_id = ",".join(map(str, observed))
                    # Training-only quartiles for calibration stratification.
                    boundaries = np.quantile((training[:, observed[0]] - center[observed[0]]) / scale[observed[0]], [.25, .5, .75])
                    for budget in settings["bank_sizes"]:
                        for bandwidth in settings["condition_bandwidths"]:
                            for role in ("teacher", "student"):
                                estimator = ConditionalKDE(paths[role][position][plan["fit"][:budget]], observed, bandwidth)
                                groups = {n: [] for n in settings["conditional_draw_sizes"]}
                                ess = []
                                for row_index, target in enumerate(real):
                                    mixture = estimator.condition(target[observed])
                                    ess.append(mixture.effective_components)
                                    streams = np.random.SeedSequence([seed, index, mask_number, row_index, 3202]).spawn(3)
                                    threshold, a, b = [mixture.sample(largest, np.random.default_rng(s)) for s in streams]
                                    threshold_scores = mixture.log_prob(threshold, settings["kde_chunk_size"])
                                    validation_scores = mixture.log_prob(a, settings["kde_chunk_size"])
                                    real_score = mixture.log_prob(target[hidden][None])[0]
                                    for draws in settings["conditional_draw_sizes"]:
                                        values = conditional_scores(target[hidden], a[:draws], b[:draws], threshold_scores[:draws],
                                                                    real_score, settings["levels"], validation_log_density=validation_scores[:draws])
                                        groups[draws].extend(values)
                                        if (budget == max(settings["bank_sizes"]) and bandwidth == settings["primary_bandwidth"]
                                                and draws == largest):
                                            low, high = np.quantile(a[:, 0], [.05, .95])
                                            for value in values:
                                                observations.append({"role": role, "seed": seed, "time_index": index,
                                                    "row_index": row_index, "mask_id": mask_id, "observed_dim": len(observed),
                                                    "hidden_dim": len(hidden), "effective_components": mixture.effective_components,
                                                    "low_ess": mixture.effective_components < settings["min_effective_components"],
                                                    "observed_value": float(target[observed[0]]),
                                                    "observed_quartile": int(np.searchsorted(boundaries, target[observed[0]], side="right")),
                                                    "hidden_value": float(target[hidden[0]]), "conditional_mean": float(a[:, 0].mean()),
                                                    "lower90": float(low), "upper90": float(high), **value})
                                for draws, scores in groups.items():
                                    for value in aggregate(scores):
                                        row = {"role": role, "seed": seed, "time_index": index,
                                               "physical_time": dataset.metadata.physical_times[index],
                                               "model_time": dataset.metadata.model_times[index], "mask_id": mask_id,
                                               "observed_dim": len(observed), "hidden_dim": len(hidden), "bank_size": budget,
                                               "unique_fit_components": len(estimator.centers), "conditional_draws": draws,
                                               "bandwidth": bandwidth, "effective_components_mean": float(np.mean(ess)),
                                               "effective_components_min": float(np.min(ess)),
                                               "low_ess_fraction": float(np.mean(np.array(ess) < settings["min_effective_components"])),
                                               **value}
                                        rows.append(row)
                                        logger.log({"e3_2/" + k: v for k, v in row.items()})
                                write_csv(output / "by_time_mask.csv", rows)
                                print(f"E3.2 {role}: time={index} mask={mask_id} bank={budget} h={bandwidth:g} finished", flush=True)
        headline = summarize(rows, settings)
        write_csv(output / "by_condition.csv", observations)
        write_csv(output / "summary.csv", headline)
        stratified = []
        for role in ("teacher", "student"):
            for index in times:
                for observed, _ in masks:
                    mask_id = ",".join(map(str, observed))
                    for seed in settings["seeds"]:
                        for quartile in range(4):
                            group = [r for r in observations if r["role"] == role and r["time_index"] == index
                                     and r["mask_id"] == mask_id and r["seed"] == seed and r["observed_quartile"] == quartile]
                            if group:
                                stratified.extend({"role": role, "time_index": index, "mask_id": mask_id, "seed": seed,
                                                   "observed_training_quartile": quartile, **r} for r in aggregate(group))
        write_csv(output / "by_observed_quartile.csv", stratified)
        comparison = {"method": config["teacher"]["name"], "dataset": config["dataset"], "representation_sha256": representation_sha}
        for key in METRICS:
            before, after = headline[0][key], headline[1][key]
            comparison.update({"teacher_" + key: before, "student_" + key: after,
                               "student_minus_teacher_" + key: after - before if before is not None and after is not None else None})
        write_csv(output / "comparison.csv", [comparison])
        figures = save_figures(rows, observations, settings, output / "figures")
        atomic_json(output / "media.json", figures)
        diagnostics = {"source_support": sampling["unique_source_rows_integrated"],
                       "source_law": sampling["initial_law"], "effective_component_threshold": settings["min_effective_components"],
                       "reliability_note": "small ESS flags sparse conditional support, not an automatic calibration verdict; no observations dropped",
                       "density_note": "all probabilities/scores concern conditional KDE, not exact unsmoothed ODE density",
                       "primary_bandwidth": settings["primary_bandwidth"],
                       "bandwidths_reported_without_test_selection": settings["condition_bandwidths"],
                       "high_hidden_dimension_warning": any(len(hidden) > 3 for _, hidden in masks)}
        atomic_json(output / "estimator_diagnostics.json", diagnostics)
        result = {"schema_version": 1, "experiment": "e3_2", "method": config["teacher"]["name"], "dataset": config["dataset"],
                  "selection": selection, "summary": headline, "comparison": comparison,
                  "representation_sha256": representation_sha, "media": figures, "estimator_diagnostics": diagnostics,
                  "interpretation": "retrospective same-time conditional KDE calibration; aggregate over conditions, not pointwise calibration or forecasting"}
        atomic_json(output / "results.json", result)
        for row in headline:
            logger.log({f"e3_2/summary/{row['role']}/{k}": v for k, v in row.items() if type(v) in (int, float)})
        if logger.run is not None:
            import wandb

            with logger.remote("E3.2 comparison images"):
                logger.run.log({"e3_2/" + key: wandb.Image(path) for key, path in figures.items()})
        logger.close()
        logger = None
        atomic_json(output / "status.json", {"status": "complete", "binding_sha256": fingerprint})
        return result
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "failed", "binding_sha256": fingerprint, "error": str(error)})
        if logger is not None:
            logger.close(failed=True)
        raise

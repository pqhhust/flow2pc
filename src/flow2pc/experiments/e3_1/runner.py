"""E3.1: numerical teacher reference, direct frozen-PC student queries."""

import fcntl
from pathlib import Path

import numpy as np
import yaml

from flow2pc.experiments.adapters import (
    evaluation_times,
    load_pair,
    query_banks,
    real_population,
    training_coordinates,
)
from flow2pc.experiments.analytic_knot import array_sha256, contract as analytic_contract, enabled as analytic_enabled
from flow2pc.experiments.e3_1.figures import save_fidelity, save_regions, save_summary
from flow2pc.experiments.e3_1.metrics import (
    EstimatedHDR, Representation, distribution_fidelity, energy_score, region_fidelity,
)
from flow2pc.experiments.io import write_csv
from flow2pc.experiments.pc_queries import DirectPC, PCDensityHDR
from flow2pc.experiments.selection import digest, identity, implementation_identity, read_json, select_student
from flow2pc.utils.checkpoint import atomic_json, atomic_path
from flow2pc.utils.tracking import LocalLogger


def summarize(rows, settings):
    maximum, headline = max(settings["bank_sizes"]), []
    for role in ("teacher", "student"):
        chosen = [r for r in rows if r["role"] == role and r["bank_size"] == maximum]
        result = {"role": role, "bank_size": maximum}
        result["inference_method"] = "direct_pc" if role == "student" else "ode_kde"
        for metric, alpha in (("coverage", .9), ("hdr_ece", settings["levels"][0]),
                              ("energy_score", settings["levels"][0])):
            per_seed = []
            for seed in settings["seeds"]:
                values = [r[metric] for r in chosen if r["seed"] == seed
                          and r["nominal_alpha"] == alpha and r[metric] is not None]
                if values:
                    per_seed.append(float(np.mean(values)))
            key = "coverage90" if metric == "coverage" else metric
            result[key] = float(np.mean(per_seed)) if per_seed else None
            result[key + "_mc_std"] = float(np.std(per_seed)) if per_seed else None
        result["evaluated_times"] = len({r["time_index"] for r in chosen})
        result["coverage_eligible_times"] = len({r["time_index"] for r in chosen if r["coverage"] is not None})
        result["uncertainty_note"] = "MC standard deviation across resampling seeds; not independent training runs or a test-data confidence interval"
        headline.append(result)
    return headline


def summarize_fidelity(rows, settings):
    fields = ("paired_ode_rmse", "mean_shift_l2", "relative_covariance_error",
              "mean_interval_width_ratio", "relative_interval_width_error",
              "teacher_region_teacher_mass", "teacher_region_student_mass",
              "student_region_student_mass", "student_region_teacher_mass",
              "teacher_region_mass_gap", "student_region_mass_gap")
    summary = []
    for level in settings["levels"]:
        selected = [r for r in rows if r["bank_size"] == max(settings["bank_sizes"])
                    and r["nominal_alpha"] == level]
        if not selected:
            continue
        row = {"nominal_alpha": level, "bank_size": max(settings["bank_sizes"])}
        for key in fields:
            means = []
            for seed in settings["seeds"]:
                values = [r[key] for r in selected if r["seed"] == seed and r[key] is not None]
                if values:
                    means.append(float(np.mean(values)))
            row[key] = float(np.mean(means)) if means else None
            row[key + "_mc_std"] = float(np.std(means)) if means else None
        summary.append(row)
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
    analytic = analytic_enabled(settings)
    reference_note = (
        f"fresh independently seeded analytic observations, unused in training/model selection; selected by saved {selection['metric']}"
        if analytic else
        "retrospective; EMD/velocity model selection and possibly preprocessing already used held-out data"
    )
    selection["evaluation_reference_protocol"] = "analytic_knot" if analytic else "saved_dataset_splits"
    selection["data_reuse"] = reference_note
    scientific = {key: value for key, value in config.items() if key not in ("device", "logging", "output")}
    implementation = implementation_identity()
    binding = {"config": scientific, "teacher_sha256": selection["teacher_sha256"],
               "student_sha256": selection["student_sha256"], "implementation": implementation}
    fingerprint = identity(binding)
    previous = read_json(output / "status.json")
    saved_binding = read_json(output / "binding.json")
    if saved_binding and saved_binding != binding:
        raise ValueError("Experiment inputs/code differ from saved outputs; select a new output.dir")
    if previous.get("status") == "complete":
        media = read_json(output / "media.json")
        result = read_json(output / "results.json")
        if (previous.get("binding_sha256") != fingerprint or not (output / "summary.csv").is_file()
                or result.get("selection", {}).get("student_sha256") != selection["student_sha256"]
                or not media or any(not Path(p).is_file() for p in media.values())):
            raise ValueError("Completed experiment artifacts are inconsistent")
        return result
    atomic_json(output / "binding.json", binding)
    atomic_json(output / "selection.json", selection)
    with atomic_path(output / "resolved_config.yaml") as temporary:
        temporary.write_text(yaml.safe_dump(config, sort_keys=False))
    atomic_json(output / "status.json", {"status": "running", "binding_sha256": fingerprint})
    logger = None
    try:
        teacher, student, dataset = load_pair(config, selection)
        contract = dataset.contract()
        if analytic:
            atomic_json(output / "analytic_protocol.json", analytic_contract(dataset, settings))
        times = evaluation_times(dataset, settings)
        references, audit = {}, []
        for index in times:
            real, split, total = real_population(dataset, index, settings)
            eligible = len(real) >= settings["min_test_samples"]
            references[index] = real
            audit.append({
                "experiment": "e3_1", "method": config["teacher"]["name"], "dataset": config["dataset"],
                "time_index": index, "physical_time": dataset.metadata.physical_times[index],
                "model_time": dataset.metadata.model_times[index], "state_dim": dataset.dim,
                "reference_split": split, "n_available": total, "n_evaluated": len(real),
                "n_train": contract["split_counts"][index].get("train", 0),
                "split_definition": "fresh independent analytic marginals; checkpoint splits unchanged" if analytic else contract["split_scheme"],
                "saved_split_definition": contract["split_scheme"],
                "reference_sha256": array_sha256(real),
                "reference_rng_seed": settings["analytic_seed"] if analytic else "fixed subset seed 2026 + time index",
                "source_split": settings["source_split"], "hdr_eligible": eligible,
                "energy_eligible": bool(len(real)),
                "exclusion_reason": "" if eligible else f"Fewer than {settings['min_test_samples']} evaluation observations",
                "group_ids_available": dataset.metadata.has_trajectory_ids,
                "observational_ci": "not estimated; independent sampling/group units have not been established",
                "data_reuse": reference_note,
                "teacher_checkpoint": config["teacher"]["checkpoint"], "student_checkpoint": selection["checkpoint"],
            })
        write_csv(output / "dataset_audit.csv", audit)
        atomic_json(output / "dataset_contract.json", contract)
        nonempty = [index for index in times if len(references[index])]
        if not nonempty:
            raise ValueError("No reference observations at the requested times; see dataset_audit.csv")
        times = nonempty
        representation = Representation(training_coordinates(dataset, settings["representation_samples"]),
                                          settings["hdr_representation"])
        if representation.projected:
            raise ValueError("Direct PC HDR cannot use a rotated PCA density. Select a low-dimensional full-state E3.1 profile; a coordinate-marginal protocol is needed for projected high-dimensional reports. No student KDE/ODE fallback is used.")
        direct_pc = DirectPC(student, dataset.preprocessing, representation.center,
                             representation.scale, settings["kde_chunk_size"])
        coordinate_contract = representation.contract()
        coordinate_contract["density"] = "teacher: Gaussian KDE of ODE draws; student: direct PC density, no KDE"
        coordinate_sha = identity(coordinate_contract)
        atomic_json(output / "representation.json", {**coordinate_contract, "sha256": coordinate_sha})
        logger = LocalLogger(output, config, append=(output / "metrics.jsonl").is_file())
        logger.log({"e3_1/phase": "teacher_ODE_and_direct_PC", "e3_1/analytic_knot": analytic})
        print(f"E3.1 {config['teacher']['name']}/{config['dataset']}: teacher ODE and direct student PC "
              f"at {len(times)} times; initial law={settings['source_split']}", flush=True)
        paths, plans, sampling = query_banks(teacher, student, dataset, times, settings)
        atomic_json(output / "sampling.json", {**sampling, "evaluation_settings": settings,
                    "physical_times": [dataset.metadata.physical_times[i] for i in times],
                    "initial_time_excluded": True})
        logger.log({"e3_1/phase": "scoring", "e3_1/source_count": sampling["source_count"],
                    "e3_1/teacher_nfe": sampling["nfe"]["teacher"], "e3_1/student_nfe": sampling["nfe"]["student"]})
        rows, fidelity_rows, regions = [], [], {}
        region_times = {times[0], times[-1]}
        for position, index in enumerate(times):
            pc_density = PCDensityHDR(direct_pc.at(dataset.metadata.model_times[index]), representation.project_scale)
            real = references[index]
            energy_real, hdr_real = representation.energy(real), representation.hdr(real)
            sufficient = len(real) >= settings["min_test_samples"]
            for seed, plan in plans.items():
                model_regions = {}
                for role in ("teacher", "student"):
                    predicted = paths[role][position]
                    energy = energy_score(energy_real, representation.energy(predicted[plan["energy_a"]]),
                                          representation.energy(predicted[plan["energy_b"]]), settings["kde_chunk_size"])
                    for budget in settings["bank_sizes"]:
                        estimator, ece = None, None
                        coverage = [{"nominal_alpha": a, "coverage": None,
                                     "log_density_threshold": None, "threshold_bank_mass": None,
                                     "threshold_tie_fraction": None} for a in settings["levels"]]
                        if sufficient:
                            estimator = pc_density if role == "student" else EstimatedHDR(
                                representation.hdr(predicted[plan["fit"][:budget]]),
                                representation.hdr(predicted[plan["tune"]]),
                                settings["bandwidth_multipliers"], settings["kde_chunk_size"],
                                bandwidth_scale=settings["bandwidth_scale"])
                            coverage, ece = estimator.coverage(
                                representation.hdr(predicted[plan["threshold"][:budget]]), hdr_real, settings["levels"])
                            bandwidth_path = output / "density_diagnostics" / f"time{index}-{role}-s{seed}-n{budget}.json"
                            diagnostic = {"density_method": "direct_pc", "smoothing": False, "threshold_draws": budget}
                            if role == "teacher":
                                diagnostic = {"search": estimator.search, "selected_bandwidth": estimator.bandwidth,
                                                        "bandwidth_scale": settings["bandwidth_scale"], "model_fit_spread": estimator.scale,
                                                        "selected_index": estimator.selected_index,
                                                        "at_search_boundary": estimator.at_search_boundary,
                                                        "unique_fit_rows": len(np.unique(predicted[plan["fit"][:budget]], axis=0)),
                                                        "tuning_data": "independent model draws only", "density_method": "ode_kde"}
                            atomic_json(bandwidth_path, diagnostic)
                            model_regions.setdefault(budget, {})[role] = (estimator, coverage)
                        for score in coverage:
                            row = {"method": config["teacher"]["name"], "dataset": config["dataset"],
                                   "role": role, "seed": seed,
                                   "inference_method": "direct_pc" if role == "student" else "ode_kde",
                                   "density_exact_under_model": role == "student",
                                   "hdr_threshold_method": "independent-model-draw density quantile",
                                   "seed_kind": "Monte Carlo analytic initial draws" if analytic else "Monte Carlo bank resampling",
                                   "time_index": index, "physical_time": dataset.metadata.physical_times[index],
                                   "model_time": dataset.metadata.model_times[index], "bank_size": budget,
                                   "n_test": len(real), "state_dim": dataset.dim,
                                   "hdr_dim": coordinate_contract["hdr_dim"], "projected_hdr": representation.projected,
                                   "representation_sha256": coordinate_sha, **score,
                                   "hdr_ece": ece, "energy_score": energy,
                                   "bandwidth": estimator.bandwidth if estimator is not None else None,
                                   "bandwidth_at_search_boundary": estimator.at_search_boundary if estimator is not None else None}
                            rows.append(row)
                            logger.log({"e3_1/" + key: value for key, value in row.items()})
                        if (estimator is not None and index in region_times and seed == settings["seeds"][0]
                                and budget == max(settings["bank_sizes"])):
                            threshold = next((r["log_density_threshold"] for r in coverage if r["nominal_alpha"] == .9), None)
                            if threshold is not None:
                                regions.setdefault(index, []).append({"role": role, "time": dataset.metadata.physical_times[index],
                                    "real": hdr_real, "model": representation.hdr(predicted[plan["fit"][:budget]]),
                                    "estimator": estimator, "threshold": threshold})
                if model_regions:
                    teacher_draws = paths["teacher"][position][plan["energy_a"]]
                    student_draws = paths["student"][position][plan["energy_a"]]
                    statistics = distribution_fidelity(representation.energy(teacher_draws),
                                                       representation.energy(student_draws), settings["levels"], paired=False)
                    for budget, estimators in model_regions.items():
                        teacher_estimator, teacher_regions = estimators["teacher"]
                        student_estimator, student_regions = estimators["student"]
                        masses = region_fidelity(teacher_estimator, student_estimator,
                                                  representation.hdr(teacher_draws), representation.hdr(student_draws),
                                                  teacher_regions, student_regions)
                        for stats, mass in zip(statistics, masses, strict=True):
                            row = {"method": config["teacher"]["name"], "dataset": config["dataset"],
                                   "seed": seed, "time_index": index,
                                   "physical_time": dataset.metadata.physical_times[index],
                                   "model_time": dataset.metadata.model_times[index], "bank_size": budget,
                                   "representation_sha256": coordinate_sha, **stats, **mass}
                            fidelity_rows.append(row)
                            logger.log({"e3_1/fidelity/" + key: value for key, value in row.items()})
            write_csv(output / "by_time.csv", rows)
            write_csv(output / "fidelity_by_time.csv", fidelity_rows)
            print(f"E3.1 {config['teacher']['name']}/{config['dataset']}: time {index} finished; {len(real)} real observations", flush=True)
        headline = summarize(rows, settings)
        write_csv(output / "summary.csv", headline)
        comparison = {"method": config["teacher"]["name"], "dataset": config["dataset"],
                      "representation_sha256": coordinate_sha}
        for metric in ("coverage90", "hdr_ece", "energy_score"):
            before, after = headline[0][metric], headline[1][metric]
            comparison.update({f"teacher_{metric}": before, f"student_{metric}": after,
                               f"student_minus_teacher_{metric}": after - before if before is not None and after is not None else None})
        write_csv(output / "comparison.csv", [comparison])
        fidelity_summary = summarize_fidelity(fidelity_rows, settings)
        write_csv(output / "fidelity_summary.csv", fidelity_summary)
        figures = save_summary(rows, output / "figures")
        figures.update(save_fidelity(fidelity_summary, output / "figures"))
        figures.update(save_regions(regions, output / "figures"))
        if analytic:
            from flow2pc.experiments.e3_1.figures import save_knot_geometry

            figures.update(save_knot_geometry(paths, references, times, dataset,
                                              plans[settings["seeds"][0]]["fit"], output / "figures"))
        atomic_json(output / "media.json", figures)
        estimator_diagnostics = {
            "bandwidth_boundary_fraction": float(np.mean([r["bandwidth_at_search_boundary"] for r in rows
                if r["nominal_alpha"] == settings["levels"][0] and r["bandwidth_at_search_boundary"] is not None]))
                if any(r["bandwidth_at_search_boundary"] is not None for r in rows) else None,
            "source_support": sampling["unique_source_rows_integrated"],
            "source_law": sampling["initial_law"],
            "qualification": ("fresh analytic draws" if analytic else
                              "banks resample a finite empirical source law; larger banks do not create new source support"),
            "bandwidth_boundary_note": "a boundary choice requires sensitivity review; it is not proof of KDE convergence",
        }
        atomic_json(output / "estimator_diagnostics.json", estimator_diagnostics)
        result = {"schema_version": 2, "experiment": "e3_1", "method": config["teacher"]["name"], "dataset": config["dataset"],
                  "output_dir": str(output), "summary": headline, "selection": selection,
                  "comparison": comparison,
                  "fidelity_summary": fidelity_summary,
                  "fidelity_protocol": {"statistics_space": "full-state training-standardized coordinates",
                      "region_space": coordinate_contract["hdr_space"],
                      "bank": "energy_a: independent of density fit, tune and threshold banks",
                      "pairing": "independent direct-PC and teacher-ODE draws; paired_ode_rmse is N/A",
                      "interpretation": "cross-region mass gaps, location/covariance/width errors are diagnostics, not equivalence proofs",
                      "uncertainty": "MC standard deviations only; no independent-observation confidence interval"},
                  "estimator_diagnostics": estimator_diagnostics,
                  "representation_sha256": coordinate_sha, "media": figures,
                  "student_inference": "direct_pc", "teacher_inference": "ode_kde",
                  "interpretation": ("direct PC calibration against fresh analytic knot references; analytic source applies only to teacher; fixed checkpoints, MC variability only"
                                     if analytic else "direct PC retrospective calibration; teacher uses numerical ODE/KDE; no untouched-test claim")}
        atomic_json(output / "results.json", result)
        if logger.run is not None:
            import wandb

            with logger.remote("E3.1 comparison images"):
                logger.run.log({f"e3_1/{key}": wandb.Image(path) for key, path in figures.items()})
        for row in headline:
            logger.log({f"e3_1/summary/{row['role']}/{k}": v for k, v in row.items() if type(v) in (int, float)})
        logger.close()
        logger = None
        atomic_json(output / "status.json", {"status": "complete", "binding_sha256": fingerprint})
        return result
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "failed", "binding_sha256": fingerprint, "error": str(error)})
        if logger is not None:
            logger.close(failed=True)
        raise

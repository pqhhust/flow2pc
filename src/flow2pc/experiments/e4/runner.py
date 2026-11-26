"""E4 evaluation only: fixed checkpoints, empirical shifted laws, ODE outputs."""

import fcntl
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from flow2pc.data.preprocessing import transform
from flow2pc.evaluation.ode import raw_coordinates, rollout
from flow2pc.evaluation.runner import population
from flow2pc.experiments.adapters import evaluation_times, load_pair, training_coordinates
from flow2pc.experiments.analytic_knot import array_sha256
from flow2pc.experiments.e3_1.metrics import Representation
from flow2pc.experiments.e4.figures import save as save_figures, time_curves
from flow2pc.experiments.e4.metrics import HDR_REASON, kernel_widths, mmd2, response, uncertainty
from flow2pc.experiments.e4.perturbations import bank_plans, conditions, perturb
from flow2pc.experiments.e4.summary import E41, E42, rectangular, summarize
from flow2pc.experiments.io import write_csv
from flow2pc.experiments.selection import digest, identity, implementation_identity, read_json, select_student
from flow2pc.utils.checkpoint import atomic_json, atomic_path
from flow2pc.utils.tracking import LocalLogger
from flow2pc.utils.paths import REPO_ROOT


def condition_paths(teacher, student, source, dataset, grid, settings, representation,
                    condition, output, fingerprint):
    """Integrate each empirical source row once; cache exact pushforwards."""
    raw = raw_coordinates(source, dataset.preprocessing).numpy()
    standardized = representation.energy(raw)
    shifted = perturb(standardized, condition, settings["direction"])
    # Preserve original float32 source states exactly for the clean condition.
    initial = (source if condition["shift_type"] == "clean" else
               transform(shifted * representation.scale + representation.center, dataset.preprocessing))
    cache_key = identity({"experiment": fingerprint, "condition": condition,
                          "source_model_sha256": array_sha256(initial.numpy())})
    cache = output / "cache" / (condition["condition"]+".npz")
    reused = False
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["binding"]) != cache_key:
                raise ValueError("E4 cache inputs differ; use a new output directory")
            paths = {role: saved[role].copy() for role in ("teacher", "student")}
            nfe = {role: int(saved[role+"_nfe"]) for role in paths}
        reused = True
    else:
        paths, nfe = {}, {}
        for role, model in (("teacher", teacher), ("student", student)):
            path, nfe[role] = rollout(model, initial, grid, settings, student=role == "student")
            paths[role] = raw_coordinates(path, dataset.preprocessing).numpy()
        cache.parent.mkdir(parents=True, exist_ok=True)
        with atomic_path(cache) as temporary:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, binding=cache_key, **paths,
                                    **{role+"_nfe": value for role, value in nfe.items()})
    shape = (len(grid), len(source), dataset.dim)
    if any(p.shape != shape or not np.isfinite(p).all() for p in paths.values()):
        raise ValueError("Invalid E4 trajectory cache")
    raw_initial = raw_coordinates(initial, dataset.preprocessing).numpy()
    return paths, {**condition, "cache": str(cache), "cache_reused": reused,
                   "cache_binding_sha256": cache_key, "nfe": nfe,
                   "initial_raw_sha256": array_sha256(raw_initial),
                   "initial_model_sha256": array_sha256(initial.numpy()),
                   "negative_initial_coordinate_fraction": float(np.mean(raw_initial < 0)),
                   "outside_clean_coordinate_range_fraction": float(np.mean(
                       (raw_initial < raw.min(0)) | (raw_initial > raw.max(0)))),
                   "initial_raw_min": raw_initial.min(0).tolist(),
                   "initial_raw_max": raw_initial.max(0).tolist(),
                   "clipping": False, "numerical_failures": 0}


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
    previous, saved = read_json(output / "status.json"), read_json(output / "binding.json")
    if saved and saved != binding:
        raise ValueError("E4 inputs/code differ from saved results; use a new output.dir")
    if previous.get("status") == "complete":
        result, media = read_json(output / "results.json"), read_json(output / "media.json")
        if (previous.get("binding_sha256") != fingerprint or not result or not media
                or not (output / "summary.csv").is_file()
                or result.get("selection", {}).get("student_sha256") != selection["student_sha256"]
                or any(not Path(p).is_file() for p in media.values())):
            raise ValueError("Completed E4 artifacts are inconsistent")
        return result
    atomic_json(output / "binding.json", binding)
    atomic_json(output / "selection.json", selection)
    with atomic_path(output / "resolved_config.yaml") as temporary:
        temporary.write_text(yaml.safe_dump(config, sort_keys=False))
    atomic_json(output / "status.json", {"status": "running", "binding_sha256": fingerprint})
    logger = None
    try:
        teacher, student, dataset = load_pair(config, selection)
        if len(settings["direction"]) != dataset.dim:
            raise ValueError("E4 direction and saved state dimension differ")
        times = [0, *evaluation_times(dataset, settings)]
        if times != sorted(set(times)):
            raise ValueError("E4 requires an increasing nonextrapolated time grid")
        grid = torch.tensor([dataset.metadata.model_times[i] for i in times], dtype=torch.float32)
        source = population(dataset, 0, settings["source_split"]).cpu()
        training = training_coordinates(dataset, settings["representation_samples"])
        representation = Representation(training, "full")
        widths = kernel_widths(representation.energy(training), settings)
        representation_contract = {**representation.contract(), "mmd_bandwidths": widths,
                                   "kernel_fit": "fixed observed-training-row subset, seed 8137; no shifted outcomes"}
        representation_sha = identity(representation_contract)
        atomic_json(output / "representation.json", {**representation_contract, "sha256": representation_sha})
        plans, plan_audit = bank_plans(len(source), settings)
        atomic_json(output / "samples_manifest.json", {
            "initial_law": "uniform empirical earliest-time population",
            "source_count": len(source), "source_model_sha256": array_sha256(source.numpy()),
            "bank_law": "independent IID draws with replacement conditional on finite source law",
            "shared_support": "source IDs may overlap across banks; draw streams, not cell identities, are independent",
            "paired_models": "identical initial source IDs for teacher and student",
            "paired_conditions": "same source IDs for clean and shifted; fixed full-source center for spread",
            "plans": plan_audit, "reference_mode": "teacher-relative",
            "ground_truth_shifted": "not_evaluable: no independent outcomes under these interventions"})
        atomic_json(output / "dataset_contract.json", dataset.contract())
        audit = [{"experiment": "e4", "method": config["teacher"]["name"], "dataset": config["dataset"],
                  "time_index": index, "physical_time": dataset.metadata.physical_times[index],
                  "model_time": dataset.metadata.model_times[index], "state_dim": dataset.dim,
                  "reference_split": "independent_teacher", "n_available": len(source),
                  "n_evaluated": settings["max_test_samples"], "included_in_summary": index != 0,
                  "data_reuse": "retrospective checkpoint selection; synthetic teacher-reference outcomes only"}
                 for index in times]
        write_csv(output / "dataset_audit.csv", audit)
        logger = LocalLogger(output, config, append=(output / "metrics.jsonl").exists())
        all_e41, all_e42, all_coverage, sampling = [], [], [], []
        clean_paths = clean_mmd = clean_uncertainty = None
        schedule = conditions(settings)
        for condition in schedule:
            print(f"E4 {config['teacher']['name']}/{config['dataset']}: {condition['condition']}", flush=True)
            raw_paths, sample_info = condition_paths(teacher, student, source, dataset, grid, settings,
                                                      representation, condition, output, fingerprint)
            sampling.append(sample_info)
            paths = {role: representation.energy(v.reshape(-1, dataset.dim)).reshape(v.shape)
                     for role, v in raw_paths.items()}
            if condition["shift_type"] == "clean":
                clean_paths, clean_mmd, clean_uncertainty = paths, {}, {}
            for seed, plan in plans.items():
                for position, index in enumerate(times):
                    shared = {"method": config["teacher"]["name"], "dataset": config["dataset"],
                              "reference_mode": "teacher-relative", "eval_target": "shifted_teacher_ODE",
                              **condition, "seed": seed, "time_index": index,
                              "physical_time": dataset.metadata.physical_times[index],
                              "model_time": dataset.metadata.model_times[index],
                              "score_space": "observed-training-standardized full state"}
                    t, s = (paths[role][position][plan["initial"]] for role in ("teacher", "student"))
                    value = mmd2(t, s, widths, settings["chunk_size"])
                    if condition["shift_type"] == "clean":
                        clean_mmd[seed, index] = value
                    e41 = {**shared, "role": "student", "n_initial": settings["n_initial"],
                           "shifted_mmd2": value, "clean_mmd2": clean_mmd[seed, index],
                           "delta_mmd2": value-clean_mmd[seed, index],
                           "teacher_teacher_mmd2_floor": mmd2(t, paths["teacher"][position][plan["floor"]], widths,
                                                              settings["chunk_size"]),
                           **response(t, s, clean_paths["teacher"][position][plan["initial"]],
                                      clean_paths["student"][position][plan["initial"]])}
                    all_e41.append(e41)
                    logger.log({"e4/e41/"+k: v for k, v in e41.items()})
                    reference = paths["teacher"][position][plan["reference"]]
                    for role in ("teacher", "student"):
                        prediction = paths[role][position]
                        coverage, scores = uncertainty(prediction[plan["region"]], reference,
                                                       prediction[plan["energy_a"]], prediction[plan["energy_b"]],
                                                       settings["levels"], settings["chunk_size"])
                        if condition["shift_type"] == "clean":
                            clean_uncertainty[seed, index, role] = scores
                        baseline = clean_uncertainty[seed, index, role]
                        record = {**shared, "role": role, "n_reference": len(reference),
                                  "n_region": settings["n_region"], "n_es": settings["n_es"],
                                  "density_method": "unavailable: empirical ODE law", **scores,
                                  "coverage90": next(r["coverage_sd"] for r in coverage if r["nominal_alpha"] == .9),
                                  **{"delta_"+k: scores[k]-baseline[k] for k in ("sd_ece", "energy_score", "crps")}}
                        all_e42.append(record)
                        logger.log({"e4/e42/"+k: v for k, v in record.items()})
                        all_coverage.extend({**shared, "role": role, "n_reference": len(reference),
                                             "n_region": settings["n_region"], "density_method": record["density_method"],
                                             **r} for r in coverage)
            write_csv(output / "e41_per_time_severity.csv", all_e41)
            write_csv(output / "e42_per_time_severity.csv", all_e42)
            write_csv(output / "e42_per_time_alpha.csv", all_coverage)
            atomic_json(output / "sampling.json", {"conditions": sampling, "rollout_origin": "earliest",
                                                   "direct_pc_sampling": False, "initial_law": "empirical earliest marginal"})
        e41_summary, e42_summary = summarize(all_e41, E41, "e4.1"), summarize(all_e42, E42, "e4.2")
        write_csv(output / "e41_summary.csv", e41_summary)
        write_csv(output / "e42_summary.csv", e42_summary)
        summary = rectangular(e41_summary+e42_summary)
        write_csv(output / "summary.csv", summary)
        media = save_figures(e41_summary, e42_summary, all_coverage, output / "figures")
        media.update(time_curves(all_e41, output / "figures"))
        atomic_json(output / "media.json", media)
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        atomic_json(output / "run_metadata.json", {"git_commit": revision,
                    "checkpoint_code_revision": teacher.manifest.get("code_revision"),
                    "teacher_sha256": selection["teacher_sha256"], "student_sha256": selection["student_sha256"],
                    "reference_mode": "teacher-relative", "hdr_gate": {"status": "not_evaluable", "reason": HDR_REASON},
                    "mmd_estimator": "biased multiscale Gaussian; same-initial-particle pairing; independent teacher floor uses equal-sized banks",
                    "variance_estimator": "population variance ddof=0", "time_aggregation": "equal selected noninitial physical times"})
        (output / "notes_and_limitations.md").write_text(
            "# E4 teacher-relative shift fidelity\n\n"
            "This evaluates frozen ODE pushforwards of perturbations of a finite empirical initial law. "
            "References are independent teacher draws, not observed shifted real-world outcomes. "
            "Ground-truth-shifted calibration is not evaluable. Shared source IDs across IID banks are expected; "
            "larger banks do not create new source support. No initial-law smoothing or clipping is applied.\n\n"
            "SD-ECE is operational central-interval ECE, averaged after per-feature coverage errors. "
            "Discrete atoms/ties can make attainable coverage differ from nominal. Teacher self-checks "
            "quantify sampling/discreteness effects, not real-data calibration. HDR-ECE is N/A: "+HDR_REASON+"\n\n"
            "MC dispersion/Student-t intervals are conditional on fixed checkpoints and empirical source. "
            "Timesteps are not independent replicates. No multiple-training-seed claim is made. "
            "The paired MMD comparison and independent teacher floor have different couplings; "
            "the floor is diagnostic, not a pass threshold. Extrapolation beyond observed source ranges "
            "and negative initial values are flagged in sampling.json. Training-standardized metric units "
            "are not pooled across datasets.\n")
        result = {"schema_version": 1, "experiment": "e4", "method": config["teacher"]["name"],
                  "dataset": config["dataset"], "selection": selection, "summary": summary,
                  "representation_sha256": representation_sha, "media": media,
                  "reference_mode": "teacher-relative", "hdr_reason": HDR_REASON,
                  "interpretation": "initial-law shift preservation relative to frozen teacher; no real-world shifted calibration claim"}
        atomic_json(output / "results.json", result)
        if logger.run is not None:
            import wandb

            with logger.remote("E4 figures"):
                logger.run.log({"e4/"+key: wandb.Image(path) for key, path in media.items()})
        logger.close()
        logger = None
        atomic_json(output / "status.json", {"status": "complete", "binding_sha256": fingerprint})
        return result
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "failed", "binding_sha256": fingerprint, "error": str(error)})
        if logger is not None:
            logger.close(failed=True)
        raise

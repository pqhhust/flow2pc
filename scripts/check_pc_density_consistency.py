#!/usr/bin/env python3
"""Explicit diagnostic: direct q_t versus PC-velocity transport of its own q_0.

Only --dry-run is metadata-only. Running without it evaluates the selected
checkpoint and writes a report; it never optimizes or changes model weights.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from flow2pc.utils.config import load_config  # noqa: E402


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, default=ROOT / "configs/distill/alicfm/alicfm_st_hclt.yaml")
    cli.add_argument("--checkpoint", type=Path)
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--device", default="cuda:0")
    cli.add_argument("--samples", type=int, default=512)
    cli.add_argument("--metric-samples", type=int,
                     help="Optional W1 subset cap; velocity/tail diagnostics still use every direct sample")
    cli.add_argument("--direct-only", action="store_true",
                     help="Evaluate direct PC states and reserved cells, without any ODE rollout")
    cli.add_argument("--query-checks", action="store_true",
                     help="Audit full-dimensional E3 density and conditional query calculations")
    cli.add_argument("--batch-size", type=int, default=64)
    cli.add_argument("--tail-count", type=int, default=0,
                     help="Save this many worst velocity-error states per population/time; zero disables density queries")
    cli.add_argument("--time-indices", nargs="+", type=int)
    cli.add_argument("--seed", type=int, default=20261125)
    cli.add_argument("--solver", choices=("dopri5", "rk4"), default="dopri5")
    cli.add_argument("--tolerance", type=float, default=1e-5)
    cli.add_argument("--ode-steps", type=int, default=400)
    cli.add_argument("--max-nfe", type=int, default=50000)
    cli.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    cli.add_argument("--dry-run", action="store_true")
    return cli


def protocol_spec(args, config, checkpoint):
    """Torch-free identity, shared by the single worker and parallel controller."""
    spec = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint) if checkpoint.is_file() else None,
            "teacher_checkpoint": config["teacher"]["checkpoint"],
            "student_training_seed": config["seed"],
            "student_learning_rate": config["distillation"]["lr"],
            "samples": args.samples, "batch_size": args.batch_size, "seed": args.seed,
            "tail_count": args.tail_count,
            "metric_samples": getattr(args, "metric_samples", None),
            "direct_only": getattr(args, "direct_only", False),
            "solver": args.solver, "tolerance": args.tolerance, "ode_steps": args.ode_steps, "max_nfe": args.max_nfe,
            "requested_time_indices": args.time_indices,
            "protocol": ("direct PC and reserved-cell diagnostics; no ODE"
                         if getattr(args, "direct_only", False) else
                         "direct PC versus own-initial-law ODE; separate empirical-source diagnostic")}
    spec["implementation_sha256"] = {name: digest(ROOT / name) for name in (
        "scripts/check_pc_density_consistency.py", "src/flow2pc/distill/student.py",
        "src/flow2pc/distill/objectives.py", "src/flow2pc/pc/hclt_pc.py",
        "src/flow2pc/pc/gaussian_pc.py", "src/flow2pc/evaluation/ode.py",
        "src/flow2pc/evaluation/pc_consistency.py", "src/flow2pc/data/preprocessing.py",
        "src/flow2pc/pretrain/models/alicfm/adapter.py", "src/flow2pc/pretrain/models/otpfm/adapter.py",
    )}
    if config["distillation"].get("student_data") is not None:
        spec["student_data"] = config["distillation"]["student_data"]
        spec["implementation_sha256"]["src/flow2pc/data/student_pools.py"] = digest(ROOT / "src/flow2pc/data/student_pools.py")
    if getattr(args, "query_checks", False):
        spec["query_checks"] = True
        for name in ("src/flow2pc/evaluation/pc_query_consistency.py", "src/flow2pc/experiments/pc_queries.py"):
            spec["implementation_sha256"][name] = digest(ROOT / name)
    return spec


def run(args, config, checkpoint, spec):
    import math
    from time import perf_counter

    import torch

    from flow2pc.distill.objectives import validate_source_density
    from flow2pc.data.restore import restore_dataset
    from flow2pc.pretrain.runner import load_teacher
    from flow2pc.evaluation.entrypoint import _load_student
    from flow2pc.evaluation.ode import emd, fixed_subset, raw_coordinates, rollout
    from flow2pc.evaluation.runner import population as observed_population
    from flow2pc.evaluation.pc_consistency import diagnostic_failure, velocity_support_metrics
    from flow2pc.utils.checkpoint import atomic_json, atomic_path
    from flow2pc.utils.tracking import LocalLogger
    from flow2pc.utils.experiment_protocols import dataset_tag

    source_config = config["teacher"]
    teacher = load_teacher(source_config["checkpoint"], args.device,
                           legacy_metadata=source_config.get("legacy_metadata"),
                           generator_checkpoint=source_config.get("generator_checkpoint"))
    dataset = restore_dataset(teacher.manifest["dataset"], teacher.preprocessing,
                              data_root=source_config.get("data_root"))
    student, step = _load_student(checkpoint, teacher, dataset, args.device)
    parameter = next(student.parameters())
    available = dataset.metadata.model_times
    indices = args.time_indices or sorted({0, len(available) // 3, 2 * len(available) // 3, len(available) - 1})
    indices = sorted({0, *indices})
    if any(index < 0 or index >= len(available) for index in indices) or len(indices) < 2:
        raise ValueError("Choose valid observation indices including at least one noninitial time")
    grid = parameter.new_tensor([available[index] for index in indices])
    spec.update(teacher_sha256=teacher.manifest["checkpoint_sha256"], checkpoint_step=step,
                time_indices=indices, physical_times=[dataset.metadata.physical_times[i] for i in indices])
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    output = args.output_dir or Path("/data/lthung/flow2pc/diagnostics/pc-density-consistency-v1") / config["teacher"].get("name", "alicfm") / spec["checkpoint_sha256"][:12] / fingerprint[:12]
    if (output / "results.json").exists():
        old = json.loads((output / "results.json").read_text())
        if old.get("fingerprint") == fingerprint:
            return old
        raise ValueError("Diagnostic output contains a different protocol; use another --output-dir")
    if output.exists():
        raise ValueError("Partial diagnostic directory exists; preserve it and choose another --output-dir")
    from flow2pc.data.student_pools import training_dataset

    student_data, student_data_contract = training_dataset(dataset, config["distillation"])
    source = student_data.get_marginal(0, "train").materialize()
    reserved = student_data.get_marginal(0, "val").materialize()
    if not len(source) or not len(reserved):
        raise ValueError("Diagnostic needs initial train and val cells; no test fallback")
    empirical = fixed_subset(source, args.samples, args.seed)
    settings = {"batch_size": args.batch_size, "solver": args.solver,
                "tolerance": args.tolerance, "ode_steps": args.ode_steps, "max_nfe": args.max_nfe}

    @torch.no_grad()
    def draw(time, seed):
        generator = torch.Generator(device=parameter.device).manual_seed(seed)
        result = torch.cat([student.sample(min(args.batch_size, args.samples - start), time, generator).cpu()
                            for start in range(0, args.samples, args.batch_size)])
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite direct PC samples")
        return result

    output.mkdir(parents=True)
    config["output"]["dir"] = str(output)
    tag = dataset_tag(dataset)
    name = f"{tag}-{config['teacher'].get('name', 'alicfm')}-pc-density-consistency-{spec['checkpoint_sha256'][:8]}"
    if args.direct_only:
        name = (f"{tag}-{config['teacher'].get('name', 'alicfm')}-pc-direct"
                f"-nll{config['distillation'].get('source_nll_weight', 0):g}-s{config['seed']}-mc{args.seed}")
    config["logging"].update(mode=args.mode, project="flow2pc", name=name)
    logger = LocalLogger(output, config)
    errors, phases, rows, support, tails = [], {}, [], [], []
    sample_arrays, state_summaries = {}, []
    query_checks = []
    metric_count = args.metric_samples or args.samples

    def metric_subset(values, seed):
        return fixed_subset(values, metric_count, seed)

    def record_error(phase, error):
        if not diagnostic_failure(error):
            raise error
        errors.append({"phase": phase, "error": str(error)})
        print(f"Diagnostic numerical failure: {phase}: {error}", flush=True)
        atomic_json(output / "numerical_errors.json", errors)

    def transport(name, model, initial, is_student):
        if initial is None:
            return None, None
        print(f"Starting {name}: {len(initial)} particles", flush=True)
        atomic_json(output / "status.json", {"status": "running", "phase": name, "fingerprint": fingerprint})
        started = perf_counter()
        try:
            return rollout(model, initial, grid, settings, student=is_student)
        except (FloatingPointError, RuntimeError, AssertionError) as error:
            record_error(name, error)
            return None, None
        finally:
            phases[name] = perf_counter() - started

    def write_rows(name, records):
        if records:
            with atomic_path(output / name) as temporary:
                with temporary.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                    writer.writeheader()
                    writer.writerows(records)

    try:
        atomic_json(output / "protocol.json", {**spec, "fingerprint": fingerprint})
        own_path = empirical_path = teacher_path = None
        own_nfe = empirical_nfe = teacher_nfe = None
        if not args.direct_only:
            initial = None
            try:
                initial = draw(grid[0], args.seed + 1)
            except (FloatingPointError, RuntimeError, AssertionError) as error:
                record_error("initial_PC_sampling", error)
            own_path, own_nfe = transport("student_ODE_from_PC_initial", student, initial, True)
            empirical_path, empirical_nfe = transport("student_ODE_from_empirical_initial", student, empirical, True)
            teacher_path, teacher_nfe = transport("teacher_ODE_from_empirical_initial", teacher, empirical, False)
        for position, index in enumerate(indices):
            row = {"time_index": index, "physical_time": dataset.metadata.physical_times[index],
                   "model_time": available[index], "samples": args.samples,
                   "heldout": index not in dataset.observed,
                   "metric_samples": metric_count, "reference_count": None,
                   "direct_vs_observed_w1_raw": None,
                   "empirical_source_count": len(empirical),
                   "direct_vs_own_initial_ode_w1_raw": None,
                   "independent_direct_vs_direct_w1_raw": None,
                   "direct_vs_empirical_initial_ode_w1_raw": None,
                   "direct_vs_own_initial_ode_mean_rmse_raw": None}
            direct = None
            try:
                direct = draw(grid[position], args.seed + 1000 + index)
                independent = draw(grid[position], args.seed + 2000 + index)
                if getattr(args, "query_checks", False):
                    from flow2pc.evaluation.pc_query_consistency import query_consistency

                    audit = query_consistency(student, direct, available[index], dataset.preprocessing,
                                              seed=args.seed + index, batch_size=args.batch_size)
                    audit["time_index"] = index
                    query_checks.append(audit)
                    atomic_json(output / "query_consistency.json", query_checks)
                    if not audit["passed"]:
                        raise ValueError(f"Direct-PC query consistency failed at time {index}; see query_consistency.json")
                direct_raw = raw_coordinates(direct, dataset.preprocessing)
                metric_direct = metric_subset(direct_raw, args.seed + 6000 + index)
                if args.direct_only:
                    reference = fixed_subset(observed_population(dataset, index, "all"), metric_count, args.seed + 7000 + index)
                    reference_raw = raw_coordinates(reference, dataset.preprocessing)
                    row["reference_count"] = len(reference)
                    sample_arrays[f"direct_model_{index}"] = direct.numpy()
                    sample_arrays[f"direct_raw_{index}"] = direct_raw.numpy()
                    sample_arrays[f"metric_reference_raw_{index}"] = reference_raw.numpy()
                    state_summaries.append({"time_index": index, "state_count": len(direct),
                        "min_raw": direct_raw.min(0).values.tolist(), "max_raw": direct_raw.max(0).values.tolist(),
                        "p01_raw": direct_raw.quantile(.01, dim=0).tolist(),
                        "median_raw": direct_raw.median(0).values.tolist(),
                        "p99_raw": direct_raw.quantile(.99, dim=0).tolist()})
                    row["direct_vs_observed_w1_raw"] = emd(reference_raw, metric_direct)
                row["independent_direct_vs_direct_w1_raw"] = emd(metric_direct, metric_subset(
                    raw_coordinates(independent, dataset.preprocessing), args.seed + 8000 + index))
                if own_path is not None:
                    own_raw = raw_coordinates(own_path[position], dataset.preprocessing)
                    row["direct_vs_own_initial_ode_w1_raw"] = emd(metric_direct, metric_subset(own_raw, args.seed + 9000 + index))
                    row["direct_vs_own_initial_ode_mean_rmse_raw"] = float((direct_raw.mean(0) - own_raw.mean(0)).square().mean().sqrt())
                if empirical_path is not None:
                    row["direct_vs_empirical_initial_ode_w1_raw"] = emd(metric_direct, metric_subset(
                        raw_coordinates(empirical_path[position], dataset.preprocessing), args.seed + 10000 + index))
            except (FloatingPointError, RuntimeError, AssertionError) as error:
                record_error(f"consistency_time_{index}", error)
            if not all(math.isfinite(v) for v in row.values() if v is not None):
                raise FloatingPointError("Nonfinite consistency diagnostic")
            rows.append(row)
            populations = {
                "direct_PC": direct,
                "student_ODE_from_PC_initial": own_path[position] if own_path is not None else None,
                "student_ODE_from_empirical_initial": empirical_path[position] if empirical_path is not None else None,
                "teacher_ODE_from_empirical_initial": teacher_path[position] if teacher_path is not None else None,
                "reserved_observed_val_cells": fixed_subset(student_data.get_marginal(index, "val").materialize(), args.samples, args.seed + index),
            }
            if args.direct_only:
                populations = {key: value for key, value in populations.items()
                               if key in ("direct_PC", "reserved_observed_val_cells")}
            for population, values in populations.items():
                record = {"time_index": index, "physical_time": row["physical_time"],
                          "model_time": available[index], "population": population,
                          "status": "complete", "error": "", "state_count": None,
                          "velocity_mse_model": None, "velocity_rmse_model": None,
                          "teacher_velocity_energy_model": None, "relative_velocity_rmse": None,
                          "point_velocity_rmse_median": None, "point_velocity_rmse_p90": None,
                          "point_velocity_rmse_p99": None, "point_velocity_rmse_max": None}
                if values is None or not len(values):
                    record.update(status="not_available", error=("upstream sampling/rollout failed" if values is None
                                  else "no reserved val cells at this time; no test fallback"))
                else:
                    try:
                        scores = velocity_support_metrics(
                            student, teacher, values, available[index], args.batch_size,
                            tail_count=args.tail_count)
                        selected_tails = scores.pop("tail_samples", [])
                        if selected_tails:
                            raw = raw_coordinates(torch.tensor(
                                [item["state_model"] for item in selected_tails], dtype=torch.float64),
                                dataset.preprocessing)
                            for item, raw_state in zip(selected_tails, raw.tolist()):
                                item["state_raw"] = raw_state
                            tails.append({"time_index": index, "physical_time": row["physical_time"],
                                          "model_time": available[index], "population": population,
                                          "state_count": scores["state_count"], "samples": selected_tails})
                        record.update(scores)
                    except (FloatingPointError, RuntimeError, AssertionError) as error:
                        record_error(f"velocity_{population}_time_{index}", error)
                        record.update(status="failed", error=str(error))
                support.append(record)
            write_rows("by_time.csv", rows)
            write_rows("velocity_support.csv", support)
            if args.tail_count:
                atomic_json(output / "tail_samples.json", {
                    "schema_version": 1, "tail_count": args.tail_count,
                    "selection": "largest per-state coordinate-mean squared velocity errors; no trimming",
                    "velocity_units": "saved model coordinates per model-time unit",
                    "density_units": "log density in saved model coordinates",
                    "coverage_limit": "States and velocities are retained; distance to the distillation query distribution is not measured",
                    "populations": tails})
            print(f"Completed diagnostic scoring at model time {available[index]:g}", flush=True)
        source_nll = None
        try:
            source_nll = validate_source_density(student, reserved, available[0], args.batch_size)
        except (FloatingPointError, RuntimeError, AssertionError) as error:
            record_error("source_validation_density", error)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sample_bank = direct_figure = None
        if sample_arrays:
            import numpy as np

            sample_bank = output / "direct_samples.npz"
            with atomic_path(sample_bank) as temporary:
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **sample_arrays)
            if dataset.dim == 2:
                fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 4), squeeze=False)
                for ax, index in zip(axes[0], indices):
                    real = sample_arrays.get(f"metric_reference_raw_{index}")
                    predicted = sample_arrays.get(f"direct_raw_{index}")
                    if real is not None and predicted is not None:
                        ax.scatter(real[:, 0], real[:, 1], s=5, alpha=.45, label="Observed")
                        ax.scatter(predicted[:, 0], predicted[:, 1], s=4, alpha=.25, label="Direct PC")
                        ax.set_title(f"Physical time {dataset.metadata.physical_times[index]:g}; all PC samples")
                        ax.legend(fontsize=7)
                direct_figure = output / "direct_samples.png"
                fig.savefig(direct_figure, dpi=170, bbox_inches="tight")
                plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        ax = axes[0]
        for key, label in (("direct_vs_own_initial_ode_w1_raw", "Direct PC vs ODE from PC initial law"),
                           ("independent_direct_vs_direct_w1_raw", "Independent direct-PC sample discrepancy"),
                           ("direct_vs_empirical_initial_ode_w1_raw", "Direct PC vs ODE from source cells"),
                           ("direct_vs_observed_w1_raw", "Direct PC vs observed population")):
            if not any(r[key] is not None for r in rows):
                continue
            ax.plot([r["physical_time"] for r in rows], [r[key] if r[key] is not None else float("nan") for r in rows], "o-", label=label)
        ax.set(xlabel="Physical observation time", ylabel="W1 in raw coordinates",
               title="Direct PC diagnostics; no ODE" if args.direct_only else "Density–velocity consistency; fixed checkpoint")
        ax.legend(fontsize=8)
        for population in sorted({r["population"] for r in support}):
            selected = [r for r in support if r["population"] == population and r["relative_velocity_rmse"] is not None]
            if selected:
                axes[1].plot([r["physical_time"] for r in selected], [r["relative_velocity_rmse"] for r in selected], "o-", label=population)
        axes[1].set(xlabel="Physical observation time", ylabel="Velocity RMSE / teacher RMS speed",
                    title="Velocity fidelity on different state populations")
        axes[1].legend(fontsize=6)
        figure = output / "consistency.png"
        fig.savefig(figure, dpi=170, bbox_inches="tight")
        plt.close(fig)
        result = {"status": "complete" if not errors else "completed_with_errors",
                  "fingerprint": fingerprint, "protocol": spec, "rows": rows,
                  "velocity_support": support, "numerical_errors": errors, "phase_seconds": phases,
                  "tail_audit": str(output / "tail_samples.json") if args.tail_count else None,
                  "direct_sample_bank": str(sample_bank) if sample_bank else None,
                  "direct_sample_figure": str(direct_figure) if direct_figure else None,
                  "direct_state_summaries": state_summaries,
                  "query_consistency": query_checks,
                  "student_training_data": student_data_contract,
                  "source_validation_nll_per_dimension": source_nll,
                  "source_validation_cell_count": len(reserved), "source_nll_space": "teacher-input coordinates",
                  "own_initial_ode_nfe": own_nfe, "empirical_initial_ode_nfe": empirical_nfe,
                  "teacher_empirical_initial_ode_nfe": teacher_nfe,
                  "velocity_score_space": "saved model coordinates per model-time unit; zero teacher energy gives N/A relative RMSE",
                  "observed_state_split": "reserved val cells only; heldout test populations are not substituted",
                  "teacher_rollout_origin": ("not evaluated; direct-only mode" if args.direct_only else
                                             "earliest; no preceding-observed resets in this diagnostic"),
                  "media": str(figure), "output_dir": str(output),
                  "interpretation": ("Independent direct/direct W1 estimates finite-sample discrepancy, not a statistical pass threshold. "
                    "Direct-only mode does not test ODE consistency. W1 may use a declared subset; every direct sample is retained and used for velocity/tail scoring. "
                    "Held-out observations are reporting references only. No empirical-source/PC-source equivalence is assumed.")}
        atomic_json(output / "results.json", result)
        atomic_json(output / "status.json", {"status": result["status"], "fingerprint": fingerprint,
                                             "numerical_errors": errors})
        for row in rows:
            logger.log({"pc_consistency/" + key: value for key, value in row.items()})
        for row in support:
            logger.log({"pc_velocity_support/" + key: value for key, value in row.items()})
        if source_nll is not None:
            logger.log({"pc_consistency/source_validation_nll_per_dimension": source_nll})
        if logger.run is not None:
            import wandb

            with logger.remote("PC consistency figure"):
                logger.run.log({"pc_consistency/figure": wandb.Image(str(figure))})
        return result
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "failed", "fingerprint": fingerprint, "error": str(error)})
        raise
    finally:
        logger.close()


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    import math

    if (args.samples < 2 or args.batch_size < 1 or args.tail_count < 0 or args.ode_steps < 1 or args.max_nfe < 1
            or (args.metric_samples is not None and not 2 <= args.metric_samples <= args.samples)
            or not 0 <= args.seed < 2**31 - 3000 or not math.isfinite(args.tolerance) or args.tolerance <= 0):
        cli.error("Invalid samples, solver budgets, seed or tolerance")
    try:
        config = load_config(args.config)
        if not isinstance(config, dict) or config["stage"] != "distill":
            raise ValueError("Use an ALI-CFM/OTP-FM distill config")
        checkpoint = (args.checkpoint or Path(config["output"]["dir"]) / "checkpoints/best.pt").expanduser().resolve()
        spec = protocol_spec(args, config, checkpoint)
        if args.dry_run:
            print(json.dumps({"dry_run": True, "model_loaded": False, "training": False,
                              "checkpoint_exists": checkpoint.is_file(), "protocol": spec,
                              "controller_loaded_torch": "torch" in sys.modules}, indent=2))
            return 0
        if not checkpoint.is_file():
            raise ValueError(f"Missing student checkpoint: {checkpoint}")
        result = run(args, config, checkpoint, spec)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "complete" else 1
    except (ValueError, OSError, RuntimeError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

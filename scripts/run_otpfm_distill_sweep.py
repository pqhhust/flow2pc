#!/usr/bin/env python3
"""Tune CITE OTP-FM students through the existing distillation/GPU queue.

--dry-run reads configs and checkpoint metadata only. It imports no Torch,
writes no output files, queries no GPUs, and starts no worker or W&B run.
"""

import copy
import csv
import fcntl
import hashlib
import io
import itertools
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

import run_otpfm_distill_parallel as batch  # noqa: E402
from flow2pc.utils.config import load_config  # noqa: E402

DATASETS = ("cite50", "cite100")
DEFAULT_LATENTS = (64, 96, 128, 192, 256)
DEFAULT_FEATURES = (16, 32, 64, 128)
DEFAULT_LRS = (0.0001, 0.0003, 0.001, 0.002, 0.003)


def parser():
    result = batch.parser()
    result.description = __doc__
    result.set_defaults(
        datasets=list(DATASETS),
        output_root=Path("/data/lthung/flow2pc/distill/otpfm/cite-sweep-100"),
    )
    result.add_argument("--latents", nargs="+", type=int, default=list(DEFAULT_LATENTS))
    result.add_argument("--time-features", nargs="+", type=int, default=list(DEFAULT_FEATURES))
    result.add_argument("--learning-rates", nargs="+", type=float, default=list(DEFAULT_LRS))
    result.add_argument("--steps", type=int, help="Override every preset's training budget")
    return result


def build_runs(args):
    if not args.datasets or set(args.datasets) - set(DATASETS):
        raise ValueError("This sweep supports only cite50 and cite100")
    if len(args.datasets) != len(set(args.datasets)):
        raise ValueError("Dataset names must be unique")
    for values in (args.latents, args.time_features, args.learning_rates):
        if not values or len(values) != len(set(values)):
            raise ValueError("Sweep axes must be nonempty and contain no duplicates")
    if (any(v < 2 for v in args.latents)
            or any(v < 2 or v % 2 for v in args.time_features)
            or any(not math.isfinite(v) or v <= 0 for v in args.learning_rates)
            or (args.steps is not None and args.steps < 1)):
        raise ValueError("Invalid HCLT sizes, time features, learning rates or steps")
    combinations = list(itertools.product(args.latents, args.time_features, args.learning_rates))
    # Cover both datasets and different capacities early, even if interrupted.
    random.Random(2026).shuffle(combinations)
    baseline = (128, 32, 0.001)
    if baseline in combinations:
        combinations.remove(baseline)
        combinations.insert(0, baseline)
    presets = {
        name: load_config(ROOT / f"configs/distill/otpfm/otpfm_{name}_hclt.yaml")
        for name in args.datasets
    }
    runs = []
    for latents, features, lr in combinations:
        for dataset in args.datasets:
            config = copy.deepcopy(presets[dataset])
            if (config["stage"] != "distill" or config["teacher"].get("name") != "otpfm"
                    or config["distillation"]["sampler"]["name"] != "otpfm_conditional"):
                raise ValueError(f"Expected an OTP-FM distill preset: {dataset}")
            config["student"]["params"].update(
                num_latents=latents, time_embedding_dim=features,
            )
            config["distillation"].update(lr=lr, device="cuda:0", resume=True)
            if args.steps is not None:
                config["distillation"]["steps"] = args.steps
            signature = hashlib.sha256(json.dumps(
                batch.scientific_config(config), sort_keys=True, allow_nan=False,
            ).encode()).hexdigest()[:12]
            settings = config["distillation"]
            name = (f"{dataset}d-otpfm-hclt-h{latents}-f{features}-lr{lr:g}"
                    f"-b{settings['batch_size']}-n{settings['steps']}-s{config['seed']}-{signature}")
            output = args.output_root / dataset / name
            config["output"]["dir"] = str(output)
            config["logging"].update(project="flow2pc", mode=args.mode, name=name)
            runs.append(batch.Run(dataset, name, config, output))
    if len({run.output for run in runs}) != len(runs):
        raise ValueError("Sweep configurations produced duplicate output paths")
    return runs


def result_row(run):
    params, settings = run.config["student"]["params"], run.config["distillation"]
    by_time = run.metrics.get("by_time", {})
    last = by_time[max(by_time, key=int)] if by_time else {}
    return {
        "dataset": run.dataset, "run": run.name, "status": run.status,
        "num_latents": params["num_latents"], "time_features": params["time_embedding_dim"],
        "lr": settings["lr"], "steps": settings["steps"],
        "batch_size": settings["batch_size"], "seed": run.config["seed"],
        "best_step": run.state.get("best_step"),
        "validation_velocity_mse": run.state.get("best_selection_score"),
        "teacher_ode_emd_mean": run.metrics.get("teacher_ode_emd_mean"),
        "student_ode_emd_mean": run.metrics.get("student_ode_emd_mean"),
        "teacher_ode_emd_heldout": run.metrics.get("teacher_ode_emd_heldout"),
        "student_ode_emd_heldout": run.metrics.get("student_ode_emd_heldout"),
        "teacher_ode_emd_last": last.get("teacher_ode_emd"),
        "student_ode_emd_last": last.get("student_ode_emd"),
        "student_teacher_paired_mse_last": last.get("paired_ode_mse"),
        "teacher_ode_error": run.metrics.get("teacher_ode_error", ""),
        "student_ode_error": run.metrics.get("student_ode_error", ""),
        "checkpoint": str(run.output / "checkpoints/best.pt"),
        "media_manifest": str(run.output / "eval/media.json"),
        "tracking_status": run.tracking_status, "error": run.error,
    }


def summary(args, runs, active, state):
    rows = [result_row(run) for run in runs]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path = args.output_root / "summary.csv"
    temporary = path.with_suffix(".csv.tmp")
    try:
        temporary.write_text(stream.getvalue())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    for dataset in args.datasets:
        complete = [row for row in rows if row["dataset"] == dataset
                    and row["status"] == "complete"
                    and isinstance(row["validation_velocity_mse"], (int, float))
                    and math.isfinite(row["validation_velocity_mse"])]
        complete.sort(key=lambda row: (row["validation_velocity_mse"], row["run"]))
        batch.atomic_json(args.output_root / dataset / "best_models.json", {
            "dataset": dataset,
            "selection_metric": "validation_student_teacher_velocity_mse",
            "population_emd_used_for_selection": False,
            "completed": len(complete),
            "planned": sum(row["dataset"] == dataset for row in rows),
            "best": complete[:10],
        })
    batch.atomic_json(args.output_root / "_launcher/status.json", {
        "status": state, "total": len(runs),
        "counts": {name: sum(run.status == name for run in runs) for name in
                   ("pending", "running", "complete", "failed", "blocked", "interrupted")},
        "active": [{"run": worker["task"].name, "gpu": worker["device"],
                    "pid": worker["process"].pid} for worker in active],
    })


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if (min(args.workers_per_gpu, args.threads) < 1 or args.max_parallel < 2
            or args.worker_retries < 0
            or any(not math.isfinite(v) or v <= 0
                   for v in (args.min_free_gb, args.min_disk_free_gb))):
        cli.error("Invalid parallel resource settings")
    args.output_root = args.output_root.expanduser().resolve()
    try:
        runs = build_runs(args)
        for run in runs:
            batch.inspect_run(run, args.retry_incomplete)
        blocked = Counter((run.dataset, run.error) for run in runs if run.status == "blocked")
        print(json.dumps({
            "dry_run": args.dry_run, "stage": "distill", "teacher_training": False,
            "total": len(runs), "runs_per_dataset": dict(Counter(run.dataset for run in runs)),
            "grid": {"num_latents": args.latents, "time_features": args.time_features,
                     "learning_rates": args.learning_rates},
            "teachers": {run.dataset: run.config["teacher"]["checkpoint"] for run in runs},
            "selection": "validation_velocity_MSE_only",
            "gpus": args.gpus, "max_parallel": args.max_parallel,
            "workers_per_gpu": args.workers_per_gpu, "min_free_gb": args.min_free_gb,
            "output_root": str(args.output_root), "wandb_project": "flow2pc",
            "controller_loaded_torch": "torch" in sys.modules,
            "counts": {name: sum(run.status == name for run in runs)
                       for name in ("pending", "complete", "blocked")},
            "blocked": [{"dataset": dataset, "error": error, "runs": count}
                        for (dataset, error), count in blocked.items()],
            "preview": [run.name for run in runs[:6]],
        }, indent=2), flush=True)
        if args.dry_run:
            return 0
        args.output_root.mkdir(parents=True, exist_ok=True)
        with (args.output_root / ".distill.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                cli.error("Another distillation controller or its workers owns this sweep root")
            for run in runs:
                batch.inspect_run(run, args.retry_incomplete)
            if args.summarize_only or not any(run.status == "pending" for run in runs):
                failed = any(run.status == "blocked" for run in runs)
                summary(args, runs, [], "summary_only" if args.summarize_only else "failed" if failed else "complete")
                return int(failed)
            devices = batch.selected_devices(args.gpus, batch.gpu_inventory())
            args.max_parallel = min(args.max_parallel, len(devices) * args.workers_per_gpu)
            batch.atomic_json(args.output_root / "distillation_plan.json", {
                "teacher_training": False, "selection": "validation_velocity_MSE_only",
                "runs": [{"dataset": run.dataset, "name": run.name, "config": run.config}
                         for run in runs],
            })
            return batch.DistillationQueue(
                args, runs, repo=ROOT, inventory=batch.gpu_inventory,
                inspect=batch.inspect_run, report=summary, archive=batch.archive_incomplete,
            ).run(devices, lock)
    except (ValueError, OSError, RuntimeError, yaml.YAMLError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

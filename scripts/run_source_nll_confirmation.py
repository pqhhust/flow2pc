#!/usr/bin/env python3
"""Confirm four declared ST validation tradeoffs across training and sampling seeds.

Reuses completed seed-0 students, trains only seeds 1/2, then runs direct-PC
checks without ODE rollouts. --dry-run is metadata-only and writes nothing.
"""

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import run_source_nll_pilot as source
import run_pc_consistency_pilot as diagnostic

ROOT, batch, sweep = source.ROOT, source.batch, source.sweep
DEFAULT_CONFIGS = [ROOT / f"configs/distill/{method}/source_anchor/confirmation/{name}.yaml"
                   for method, name in (
                       ("alicfm", "alicfm_st_nll0p003_hclt"),
                       ("otpfm", "otpfm_st_nll0p01_hclt"),
                       ("otpfm", "otpfm_st_nll0p03_hclt"),
                       ("otpfm", "otpfm_st_nll0p1_hclt"))]


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--phase", choices=("all", "train", "diagnose"), default="all")
    cli.add_argument("--configs", nargs="+", type=Path, default=DEFAULT_CONFIGS)
    cli.add_argument("--reference-root", type=Path, default=Path(
        "/data/lthung/flow2pc/distill/source-nll-refinement-v1"))
    cli.add_argument("--output-root", type=Path, default=Path(
        "/data/lthung/flow2pc/distill/source-nll-confirmation-v1"))
    cli.add_argument("--diagnostic-root", type=Path,
                     help="Optional fresh diagnostic directory while retaining the same trained students")
    cli.add_argument("--training-seeds", nargs="+", type=int, default=[0, 1, 2])
    cli.add_argument("--mc-seeds", nargs="+", type=int, default=[101, 202, 303])
    cli.add_argument("--samples", type=int, default=4096)
    cli.add_argument("--metric-samples", type=int, default=1024)
    cli.add_argument("--batch-size", type=int, default=64)
    cli.add_argument("--tail-count", type=int, default=16)
    cli.add_argument("--gpus", default="0,1,2,4,5,6,7")
    cli.add_argument("--max-parallel", type=int, default=7)
    cli.add_argument("--threads", type=int, default=4)
    cli.add_argument("--min-free-gb", type=float, default=4.)
    cli.add_argument("--min-disk-free-gb", type=float, default=20.)
    cli.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    cli.add_argument("--summarize-only", action="store_true")
    cli.add_argument("--dry-run", action="store_true")
    return cli


def build_runs(args):
    plan = batch.read_json(args.reference_root / "distillation_plan.json")
    runs = []
    for path in args.configs:
        declared = sweep.load_config(path)
        method = declared["teacher"]["name"]
        weight, lr = declared["distillation"]["source_nll_weight"], declared["distillation"]["lr"]
        matches = [r for r in plan.get("runs", []) if
                   batch.scientific_config(r["config"]) == batch.scientific_config(declared)
                   and r["config"]["seed"] == 0]
        if len(matches) != 1:
            raise ValueError(f"Expected one completed seed-0 recipe for {method}, NLL={weight:g}, LR={lr:g}")
        original = Path(matches[0]["config"]["output"]["dir"])
        baseline = sweep.load_config(original / "resolved_config.yaml")
        if batch.scientific_config(baseline) != batch.scientific_config(matches[0]["config"]):
            raise ValueError(f"Reference config differs from its sweep plan: {original}")
        reference = batch.Run("st", original.name, baseline, original)
        batch.inspect_run(reference)
        if reference.status != "complete":
            raise ValueError(f"Seed-0 reference is not complete; refusing to retrain it: {original}")
        for seed in args.training_seeds:
            if seed == 0:
                runs.append(reference)
                continue
            config = copy.deepcopy(baseline)
            config["seed"] = seed
            signature = hashlib.sha256(json.dumps(batch.scientific_config(config),
                                                  sort_keys=True, allow_nan=False).encode()).hexdigest()[:12]
            name = f"st-{method}-hclt-confirm-nll{weight:g}-lr{lr:g}-s{seed}-{signature}"
            output = args.output_root / "students" / method / "st" / name
            config["output"]["dir"] = str(output)
            config["logging"].update(project="flow2pc", mode=args.mode, name=name)
            runs.append(batch.Run("st", name, config, output))
    if len({r.output for r in runs}) != len(runs):
        raise ValueError("Confirmation configs produced duplicate student outputs")
    return runs


def diagnostic_args(args):
    result = diagnostic.parser().parse_args([])
    for name in ("gpus", "max_parallel", "threads", "min_free_gb", "min_disk_free_gb",
                 "mode", "samples", "metric_samples", "batch_size", "tail_count",
                 "dry_run", "summarize_only"):
        setattr(result, name, getattr(args, name))
    result.output_root = args.diagnostic_root or args.output_root / "diagnostics"
    result.direct_only, result.workers_per_gpu, result.worker_retries = True, 1, 0
    return result


def build_diagnostics(args, runs):
    settings = diagnostic_args(args)
    tasks = []
    for run in runs:
        for seed in args.mc_seeds:
            settings.seed = seed
            checkpoint = run.output / "checkpoints/best.pt"
            if not args.dry_run and not args.summarize_only and run.status != "complete":
                raise ValueError(f"Finish training before diagnosis: {run.output}")
            expected = diagnostic.check.protocol_spec(settings, run.config, checkpoint)
            signature = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:12]
            method, weight = run.config["teacher"]["name"], run.config["distillation"]["source_nll_weight"]
            name = f"st-{method}-nll{weight:g}-s{run.config['seed']}-mc{seed}-{signature}"
            tasks.append(diagnostic.Task(name, method, weight, run.output / "resolved_config.yaml",
                                         checkpoint, expected, settings.output_root / method / name))
    return settings, tasks


def report(args, tasks, active, state):
    """Average MC seeds within each checkpoint before comparing training seeds."""
    diagnostic.report(args, tasks, active, state)
    groups = defaultdict(list)
    for task in tasks:
        groups[(task.method, task.weight, task.expected["student_learning_rate"],
                task.expected["student_training_seed"])].append(task)
    checkpoint_rows = []
    for (method, weight, lr, seed), group in groups.items():
        complete = [t for t in group if t.status == "complete"]
        row = {"method": method, "source_nll_weight": weight, "lr": lr, "training_seed": seed,
               "mc_planned": len(group), "mc_complete": len(complete),
               "status": "complete" if len(complete) == len(group) else "incomplete",
               "checkpoint": str(group[0].checkpoint)}
        metrics = defaultdict(list)
        for task in complete:
            values = [r["direct_vs_observed_w1_raw"] for r in task.result["rows"]
                      if r["heldout"] and r.get("direct_vs_observed_w1_raw") is not None]
            if values:
                metrics["heldout_direct_w1"].append(mean(values))
            velocity = [r["velocity_mse_model"] for r in task.result["velocity_support"]
                        if r["population"] == "direct_PC" and r["time_index"] != 0
                        and r.get("velocity_mse_model") is not None]
            if velocity:
                metrics["direct_velocity_mse"].append(mean(velocity))
            source_nll = task.result.get("source_validation_nll_per_dimension")
            if source_nll is not None:
                metrics["source_val_nll"].append(source_nll)
        for metric in ("heldout_direct_w1", "direct_velocity_mse", "source_val_nll"):
            values = metrics[metric]
            row[metric + "_mc_mean"] = mean(values) if values else None
            row[metric + "_mc_std"] = stdev(values) if len(values) > 1 else None
        extremes = [abs(v) for t in complete for r in t.result.get("direct_state_summaries", [])
                    for name in ("min_raw", "max_raw") for v in r[name]]
        row["max_abs_raw_coordinate"] = max(extremes) if extremes else None
        checkpoint_rows.append(row)
    recipes = defaultdict(list)
    for row in checkpoint_rows:
        recipes[(row["method"], row["source_nll_weight"], row["lr"])].append(row)
    recipe_rows = []
    for (method, weight, lr), group in recipes.items():
        complete = [r for r in group if r["status"] == "complete"]
        row = {"method": method, "source_nll_weight": weight, "lr": lr,
               "training_seeds_planned": len(group), "training_seeds_complete": len(complete)}
        for metric in ("heldout_direct_w1", "direct_velocity_mse", "source_val_nll"):
            values = [r[metric + "_mc_mean"] for r in complete if r[metric + "_mc_mean"] is not None]
            row[metric + "_training_mean"] = mean(values) if values else None
            row[metric + "_training_std"] = stdev(values) if len(values) > 1 else None
        recipe_rows.append(row)
    for name, rows in (("checkpoint_summary.csv", checkpoint_rows), ("recipe_summary.csv", recipe_rows)):
        path = args.output_root / name
        temporary = path.with_suffix(".csv.tmp")
        try:
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def diagnose(args, runs):
    settings, tasks = build_diagnostics(args, runs)
    for task in tasks:
        diagnostic.inspect(task)
    print(json.dumps({"stage": "direct_PC_diagnostics", "dry_run": args.dry_run,
                      "training": False, "ODE": False, "total": len(tasks),
                      "samples_per_time": args.samples, "W1_subset_cap": args.metric_samples,
                      "training_seeds": args.training_seeds, "sampling_seeds": args.mc_seeds,
                      "output_root": str(settings.output_root), "gpus": args.gpus,
                      "max_parallel": args.max_parallel, "wandb_project": "flow2pc",
                      "controller_loaded_torch": "torch" in sys.modules,
                      "checkpoints_waiting_for_training": sum(not (r.output / "checkpoints/best.pt").is_file() for r in runs),
                      "counts": {state: sum(t.status == state for t in tasks)
                                 for state in ("pending", "complete", "blocked", "failed")}}, indent=2), flush=True)
    if args.dry_run:
        return 0
    settings.output_root.mkdir(parents=True, exist_ok=True)
    with (settings.output_root / ".diagnostics.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another diagnostic controller owns this output root") from error
        for task in tasks:
            diagnostic.inspect(task)
        if args.summarize_only or not any(t.status == "pending" for t in tasks):
            report(settings, tasks, [], "summary_only")
            return int(any(t.status in ("blocked", "failed") for t in tasks))
        devices = batch.selected_devices(args.gpus, batch.gpu_inventory())
        return diagnostic.Queue(settings, tasks, repo=ROOT, inventory=batch.gpu_inventory,
                                inspect=diagnostic.inspect, report=report,
                                archive=lambda task: None).run(devices, lock)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    for seeds in (args.training_seeds, args.mc_seeds):
        if (not seeds or len(set(seeds)) != len(seeds)
                or any(not 0 <= s < 2**31 - 100000 for s in seeds)):
            cli.error("Use distinct nonnegative seeds within the supported RNG range")
    if (not 2 <= args.metric_samples <= args.samples or min(args.batch_size, args.threads) < 1
            or args.tail_count < 1 or args.max_parallel < 2
            or any(not math.isfinite(v) or v <= 0 for v in (args.min_free_gb, args.min_disk_free_gb))):
        cli.error("Invalid samples, tail count or resource settings")
    args.reference_root = args.reference_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.diagnostic_root is not None:
        args.diagnostic_root = args.diagnostic_root.expanduser().resolve()
    try:
        runs = build_runs(args)
        for run in runs:
            batch.inspect_run(run)
        if args.phase in ("all", "train"):
            training = copy.copy(args)
            training.output_root = args.output_root / "students"
            training.workers_per_gpu, training.worker_retries, training.retry_incomplete = 1, 1, False
            training.latents = sorted({r.config["student"]["params"]["num_latents"] for r in runs})
            training.time_features = sorted({r.config["student"]["params"]["time_embedding_dim"] for r in runs})
            training.learning_rates = sorted({r.config["distillation"]["lr"] for r in runs})
            code = sweep.execute(training, runs, report=source.summary, details={
                "experiment": "ST_NLL_confirmation", "training_seeds": args.training_seeds,
                "candidate_selection": "Declared validation velocity/source-NLL tradeoffs; no held-out W1 selection",
                "teachers": {r.config["teacher"]["name"] + "/st": r.config["teacher"]["checkpoint"] for r in runs},
                "reused_seed_zero": sum(r.config["seed"] == 0 for r in runs),
                "teacher_training": False, "training_pc_sampling": False,
            })
            if code:
                return code
        if args.phase in ("all", "diagnose"):
            return diagnose(args, runs)
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

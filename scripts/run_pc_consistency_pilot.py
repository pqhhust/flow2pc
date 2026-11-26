#!/usr/bin/env python3
"""Parallel frozen-PC diagnostics for explicit source-NLL pilot weights.

Uses the existing resource-aware worker queue. No training is scheduled.
--dry-run reads saved identities only, without Torch/GPU queries/output writes.
"""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import check_pc_density_consistency as check
import run_otpfm_distill_parallel as batch
from flow2pc.utils.parallel_sweep import ParallelSweep

ROOT = check.ROOT


@dataclass
class Task:
    name: str
    method: str
    weight: float
    config_path: Path
    checkpoint: Path
    expected: dict
    output: Path
    dataset: str = "st"
    status: str = "pending"
    action: str = "diagnose"
    error: str = ""
    retries: int = 0
    not_before: float = 0.
    result: dict = field(default_factory=dict)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--pilot-root", type=Path, default=Path("/data/lthung/flow2pc/distill/source-nll-pilot-v1"))
    cli.add_argument("--output-root", type=Path, default=Path("/data/lthung/flow2pc/diagnostics/source-nll-pilot-v1"))
    cli.add_argument("--methods", nargs="+", choices=("alicfm", "otpfm"), default=["alicfm", "otpfm"])
    cli.add_argument("--datasets", nargs="+", choices=("st", "cite5", "cite50", "cite100", "knot"), default=["st"])
    cli.add_argument("--weights", nargs="+", type=float, default=[0., .1])
    cli.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    cli.add_argument("--max-parallel", type=int, default=4)
    cli.add_argument("--threads", type=int, default=4)
    cli.add_argument("--min-free-gb", type=float, default=4.)
    cli.add_argument("--min-disk-free-gb", type=float, default=20.)
    cli.add_argument("--samples", type=int, default=512)
    cli.add_argument("--metric-samples", type=int)
    cli.add_argument("--direct-only", action="store_true")
    cli.add_argument("--query-checks", action="store_true",
                     help="Check E3 direct-query densities and conditional sampling against the original PC")
    cli.add_argument("--batch-size", type=int, default=64)
    cli.add_argument("--tail-count", type=int, default=0,
                     help="Record the worst scored states and velocities without changing any samples")
    cli.add_argument("--time-indices", nargs="+", type=int)
    cli.add_argument("--seed", type=int, default=20261125)
    cli.add_argument("--solver", choices=("dopri5", "rk4"), default="dopri5")
    cli.add_argument("--tolerance", type=float, default=1e-5)
    cli.add_argument("--ode-steps", type=int, default=400)
    cli.add_argument("--max-nfe", type=int, default=10000)
    cli.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument("--summarize-only", action="store_true")
    return cli


def build_tasks(args):
    plan = batch.read_json(args.pilot_root / "distillation_plan.json")
    tasks = []
    for dataset, method in ((dataset, method) for dataset in args.datasets for method in args.methods):
        for weight in args.weights:
            matches = [r for r in plan.get("runs", [])
                       if r["config"]["teacher"]["name"] == method
                       and r.get("dataset", "st") == dataset
                       and r["config"]["distillation"].get("source_nll_weight") == weight]
            if len(matches) != 1:
                raise ValueError(f"Expected one explicit {method}/{dataset} lambda={weight:g} pilot run")
            entry = matches[0]
            run = Path(entry["config"]["output"]["dir"])
            state = batch.read_json(run / "status.json")
            if state.get("status") != "complete":
                raise ValueError(f"Pilot student is not complete: {run}")
            config_path = run / "resolved_config.yaml"
            config = check.load_config(config_path)
            if batch.scientific_config(config) != batch.scientific_config(entry["config"]):
                raise ValueError("Saved pilot recipe differs from its declared scientific configuration")
            checkpoint = run / "checkpoints/best.pt"
            expected = check.protocol_spec(args, config, checkpoint)
            if expected["checkpoint_sha256"] is None:
                raise ValueError(f"Missing pilot best checkpoint: {checkpoint}")
            signature = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:12]
            name = f"{dataset}-{method}-source-nll{weight:g}-{signature}"
            folder = args.output_root / method
            if dataset != "st":
                folder = folder / dataset
            tasks.append(Task(name, method, weight, config_path, checkpoint, expected,
                              folder / name, dataset=dataset))
    return tasks


def inspect(task, retry=False):
    task.result = batch.read_json(task.output / "results.json")
    state = batch.read_json(task.output / "status.json")
    if task.result:
        protocol = task.result.get("protocol", {})
        if any(protocol.get(key) != value for key, value in task.expected.items()):
            task.status, task.error = "blocked", "Saved diagnostic inputs/code differ"
        elif task.result.get("status") == "complete" and Path(task.result.get("media", "")).is_file():
            required = []
            if task.expected.get("direct_only"):
                required.append("direct_sample_bank")
            if task.expected.get("tail_count", 0):
                required.append("tail_audit")
            missing = [key for key in required if not task.result.get(key)
                       or not Path(task.result[key]).is_file()]
            if task.expected.get("query_checks"):
                checks = task.result.get("query_consistency", [])
                if (len(checks) != len(task.result.get("rows", [])) or not checks
                        or not all(row.get("passed") for row in checks)):
                    missing.append("complete passing query-consistency checks")
            task.status = "blocked" if missing else "complete"
            task.error = "Saved diagnostic artifacts are missing: " + ", ".join(missing) if missing else ""
        else:
            task.status = "failed"
            task.error = json.dumps(task.result.get("numerical_errors", []))
    elif task.output.exists():
        task.status, task.error = "blocked", state.get("error", "Partial diagnostic output; choose a new --output-root")
    else:
        task.status, task.error = "pending", ""


def report(args, tasks, active, state):
    rows = []
    for task in tasks:
        values = task.result.get("rows", [])
        velocity = task.result.get("velocity_support", [])
        row = {"method": task.method, "dataset": task.dataset, "source_nll_weight": task.weight,
               "distance_role": "raw-coordinate W1 consistency diagnostic; not the native dataset benchmark",
               "run": task.name, "status": task.status, "checkpoint": str(task.checkpoint),
               "mc_seed": task.expected.get("seed"),
               "training_seed": task.expected.get("student_training_seed"),
               "student_lr": task.expected.get("student_learning_rate"),
               "source_validation_nll_per_dimension": task.result.get("source_validation_nll_per_dimension"),
               "output_dir": str(task.output), "figure": task.result.get("media"), "error": task.error}
        if task.expected.get("query_checks"):
            checks = task.result.get("query_consistency", [])
            row["query_consistency_passed"] = (all(r.get("passed") for r in checks)
                if checks and len(checks) == len(values) else None)
            row["full_state_dimension"] = checks[0]["full_state_dimension"] if checks else None
        for key in ("direct_vs_own_initial_ode_w1_raw", "independent_direct_vs_direct_w1_raw", "direct_vs_observed_w1_raw"):
            points = [r[key] for r in values if r.get(key) is not None and r["time_index"] != 0]
            row[key + "_mean"] = sum(points) / len(points) if points else None
        heldout = [r["direct_vs_observed_w1_raw"] for r in values
                   if r.get("heldout") and r.get("direct_vs_observed_w1_raw") is not None]
        row["direct_vs_observed_w1_raw_heldout"] = sum(heldout) / len(heldout) if heldout else None
        for population in ("direct_PC", "teacher_ODE_from_empirical_initial"):
            points = [r["velocity_mse_model"] for r in velocity
                      if r["population"] == population and r.get("velocity_mse_model") is not None
                      and r["time_index"] != 0]
            row[population + "_velocity_mse_mean"] = sum(points) / len(points) if points else None
        rows.append(row)
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
    batch.atomic_json(args.output_root / "_launcher/status.json", {
        "status": state, "training": False, "total": len(tasks),
        "counts": {name: sum(task.status == name for task in tasks)
                   for name in ("pending", "running", "complete", "failed", "blocked", "interrupted")},
        "active": [{"run": w["task"].name, "gpu": w["device"], "pid": w["process"].pid} for w in active],
    })


class Queue(ParallelSweep):
    def launch(self, task, device, inventory, lock):
        command = [sys.executable, str(ROOT / "scripts/check_pc_density_consistency.py"),
                   "--config", str(task.config_path), "--checkpoint", str(task.checkpoint),
                   "--output-dir", str(task.output), "--device", "cuda:0", "--mode", self.args.mode]
        for name in ("samples", "metric_samples", "batch_size", "tail_count", "seed", "solver", "tolerance", "ode_steps", "max_nfe"):
            value = task.expected.get(name, getattr(self.args, name, None))
            if value is not None:
                command += ["--" + name.replace("_", "-"), str(value)]
        if task.expected.get("direct_only", False):
            command.append("--direct-only")
        if task.expected.get("query_checks", False):
            command.append("--query-checks")
        if self.args.time_indices is not None:
            command += ["--time-indices", *map(str, self.args.time_indices)]
        path = self.args.output_root / "_launcher/logs" / f"{task.name}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a")
        try:
            process = subprocess.Popen(command, cwd=ROOT, env=self.environment(inventory[device]),
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                       pass_fds=(lock.fileno(),))
        except BaseException:
            stream.close()
            raise
        task.status, task.error = "running", ""
        self.active.append(dict(task=task, process=process, log=stream, device=device, started=time.monotonic()))
        print(f"Started diagnostic {task.name}, GPU {device}, pid {process.pid}", flush=True)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    args.pilot_root = args.pilot_root.expanduser().resolve()
    args.workers_per_gpu, args.worker_retries = 1, 0
    if (not args.weights or len(set(args.weights)) != len(args.weights)
            or len(set(args.methods)) != len(args.methods)
            or len(set(args.datasets)) != len(args.datasets)
            or any(not math.isfinite(v) or v < 0 for v in args.weights)
            or min(args.samples, args.batch_size, args.max_parallel, args.threads, args.ode_steps, args.max_nfe) < 1
            or args.samples < 2 or not 0 <= args.seed < 2**31 - 3000
            or args.tail_count < 0
            or (args.metric_samples is not None and not 2 <= args.metric_samples <= args.samples)
            or any(not math.isfinite(v) or v <= 0 for v in (args.tolerance, args.min_free_gb, args.min_disk_free_gb))):
        cli.error("Invalid diagnostic weights, seeds or resource/solver budgets")
    try:
        tasks = build_tasks(args)
        for task in tasks:
            inspect(task)
        print(json.dumps({"dry_run": args.dry_run, "training": False, "total": len(tasks),
                          "gpus": args.gpus, "max_parallel": args.max_parallel, "wandb_project": "flow2pc",
                          "controller_loaded_torch": "torch" in sys.modules,
                          "output_root": str(args.output_root),
                          "runs": [{"name": task.name, "status": task.status, "error": task.error,
                                    "checkpoint": str(task.checkpoint)} for task in tasks]}, indent=2), flush=True)
        if args.dry_run:
            return 0
        args.output_root.mkdir(parents=True, exist_ok=True)
        with (args.output_root / ".diagnostics.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Another diagnostic controller owns this output root") from error
            for task in tasks:
                inspect(task)
            if args.summarize_only or not any(task.status == "pending" for task in tasks):
                report(args, tasks, [], "summary_only")
                return int(any(task.status in ("blocked", "failed") for task in tasks))
            devices = batch.selected_devices(args.gpus, batch.gpu_inventory())
            queue = Queue(args, tasks, repo=ROOT, inventory=batch.gpu_inventory, inspect=inspect,
                          report=report, archive=lambda task: None)
            return queue.run(devices, lock)
    except (ValueError, OSError, RuntimeError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

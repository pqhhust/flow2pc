"""One shared multi-GPU queue for evaluation-only experiment configurations."""

import argparse
import copy
import csv
import fcntl
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

from flow2pc.experiments import EXPERIMENTS
from flow2pc.experiments.io import write_csv, write_json
from flow2pc.experiments.selection import digest, identity, implementation_identity, read_json, select_student
from flow2pc.utils.config import load_config
from flow2pc.utils.gpu_queue import gpu_inventory, selected_devices
from flow2pc.utils.parallel_sweep import ParallelSweep
from flow2pc.utils.paths import REPO_ROOT


@dataclass
class Task:
    dataset: str
    name: str
    config: dict
    output: Path
    action: str = "experiment"
    status: str = "pending"
    error: str = ""
    selection: dict = field(default_factory=dict)
    retries: int = 0
    not_before: float = 0.0


def scientific_config(config):
    return {key: value for key, value in config.items() if key not in ("device", "logging", "output")}


def inspect(task, retry_incomplete=False):
    try:
        task.selection = select_student(task.config)
        task.error = ""
        state = read_json(task.output / "status.json")
        binding = read_json(task.output / "binding.json")
        if binding:
            if (binding.get("config") != scientific_config(task.config)
                    or binding.get("teacher_sha256") != task.selection["teacher_sha256"]
                    or binding.get("student_sha256") != digest(task.selection["checkpoint"])):
                raise ValueError("Saved experiment uses different settings/checkpoints; use a new --output-root")
            implementation = implementation_identity()
            if binding.get("implementation") != implementation:
                raise ValueError("Experiment code changed; select a new --output-root")
        if state.get("status") == "complete":
            media = read_json(task.output / "media.json")
            result = read_json(task.output / "results.json")
            if (not binding or state.get("binding_sha256") != identity(binding)
                    or not (task.output / "summary.csv").is_file()
                    or result.get("selection", {}).get("student_sha256") != binding.get("student_sha256")
                    or not media or any(not Path(p).is_file() for p in media.values())):
                raise ValueError("Completed experiment artifacts are incomplete")
            task.status = "complete"
        else:
            task.status = "pending"
            task.error = state.get("error", "")
    except (ValueError, OSError, KeyError, TypeError) as error:
        task.status, task.error = "blocked", str(error)


class ExperimentQueue(ParallelSweep):
    def launch(self, task, device, inventory, lock):
        # Experiment configs dispatch evaluation themselves. Never append the
        # pretraining queue's teacher-evaluation flag, even on a retry.
        task.action = "experiment"
        super().launch(task, device, inventory, lock)


def report(args, tasks, active, state):
    summary, audits, comparisons = [], [], []
    for task in tasks:
        result = read_json(task.output / "results.json")
        if result.get("comparison"):
            comparisons.append(result["comparison"])
        for row in result.get("summary", []):
            summary.append({"method": task.config["teacher"]["name"], "dataset": task.dataset,
                            "status": task.status, **row,
                            "representation_sha256": result.get("representation_sha256"),
                            "selected_student": task.selection.get("checkpoint"),
                            "selection_emd": task.selection.get("score"),
                            "output_dir": str(task.output)})
        path = task.output / "dataset_audit.csv"
        if path.is_file():
            with path.open() as stream:
                audits.extend(csv.DictReader(stream))
    write_csv(args.output_root / "summary.csv", summary)
    write_csv(args.output_root / "comparisons.csv", comparisons)
    write_csv(args.output_root / "dataset_audit.csv", audits)
    write_json(args.output_root / "_launcher/status.json", {
        "status": state, "experiment": args.experiment, "total": len(tasks),
        "counts": {name: sum(task.status == name for task in tasks)
                   for name in ("pending", "running", "complete", "blocked", "failed", "interrupted")},
        "runs": [{"name": task.name, "status": task.status, "error": task.error,
                  "output": str(task.output)} for task in tasks],
        "active": [{"run": worker["task"].name, "gpu": worker["device"], "pid": worker["process"].pid}
                   for worker in active],
    })


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="e3_1")
    cli.add_argument("--methods", nargs="+", choices=("alicfm", "otpfm"), default=["alicfm", "otpfm"])
    cli.add_argument("--datasets", nargs="+", default=["beijing", "gom"],
                     help="Dataset profiles from the experiment's YAMLs, or all")
    cli.add_argument("--output-root", type=Path)
    cli.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    cli.add_argument("--workers-per-gpu", type=int, default=1)
    cli.add_argument("--max-parallel", type=int, default=8)
    cli.add_argument("--threads", type=int, default=4)
    cli.add_argument("--min-free-gb", type=float, default=4)
    cli.add_argument("--min-disk-free-gb", type=float, default=20)
    cli.add_argument("--worker-retries", type=int, default=1)
    cli.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    cli.add_argument("--dry-run", action="store_true", help="Metadata only; no files, GPUs, W&B or workers")
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if (min(args.workers_per_gpu, args.max_parallel, args.threads) < 1 or args.worker_retries < 0
            or any(not math.isfinite(v) or v <= 0 for v in (args.min_free_gb, args.min_disk_free_gb))
            or len(set(args.methods)) != len(args.methods)
            or len(set(args.datasets)) != len(args.datasets)
            or ("all" in args.datasets and args.datasets != ["all"])):
        cli.error("Invalid/duplicate experiment datasets or resource settings")
    args.output_root = (args.output_root or Path("/data/lthung/flow2pc/experiments") / args.experiment / "selected-emd-v1").expanduser().resolve()
    tasks = []
    try:
        presets = {}
        for method in args.methods:
            folder = REPO_ROOT / "configs/experiments" / args.experiment / method
            for path in sorted(folder.glob("*.yaml")):
                config = load_config(path)
                key = (method, config["dataset"])
                if key in presets:
                    raise ValueError(f"Duplicate experiment preset for {key}")
                presets[key] = config
        datasets = sorted({dataset for _, dataset in presets}) if args.datasets == ["all"] else args.datasets
        if not datasets:
            raise ValueError("No experiment presets found")
        for dataset in datasets:
            for method in args.methods:
                if (method, dataset) not in presets:
                    raise ValueError(f"No {args.experiment} preset for {method}/{dataset}")
                config = copy.deepcopy(presets[method, dataset])
                config["device"] = "cuda:0"
                config["logging"].update(project="flow2pc", mode=args.mode)
                output = args.output_root / method / dataset
                config["output"]["dir"] = str(output)
                task = Task(dataset, f"{args.experiment}-{method}-{dataset}-hclt", config, output)
                inspect(task)
                if task.selection:
                    # Freeze one EMD winner in the worker config. New candidates
                    # completing later cannot silently change an active task.
                    task.config["selection"]["pinned_checkpoint"] = task.selection["checkpoint"]
                    inspect(task)
                tasks.append(task)
        print(json.dumps({"dry_run": args.dry_run, "experiment": args.experiment,
            "training": False, "controller_loaded_torch": "torch" in sys.modules,
            "wandb_project": "flow2pc", "gpus": args.gpus, "max_parallel": args.max_parallel,
            "counts": {name: sum(t.status == name for t in tasks) for name in ("pending", "complete", "blocked")},
            "runs": [{"name": t.name, "status": t.status, "error": t.error,
                      "source_split": t.config["evaluation"]["source_split"],
                      "reference_split": t.config["evaluation"]["reference_split"],
                      "max_test_samples": t.config["evaluation"]["max_test_samples"],
                      "student": t.selection.get("checkpoint"), "selection_emd": t.selection.get("score"),
                      "output": str(t.output)} for t in tasks]}, indent=2), flush=True)
        if args.dry_run:
            return int(any(t.status == "blocked" for t in tasks))
        args.output_root.mkdir(parents=True, exist_ok=True)
        with (args.output_root / ".experiment-queue.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                cli.error("Another experiment queue or its workers owns this output root")
            for task in tasks:
                inspect(task)
            write_json(args.output_root / "experiment_plan.json", {
                "experiment": args.experiment, "training": False,
                "runs": [{"name": t.name, "config": t.config, "selection": t.selection} for t in tasks]})
            if not any(t.status == "pending" for t in tasks):
                failed = any(t.status == "blocked" for t in tasks)
                report(args, tasks, [], "blocked" if failed else "complete")
                return int(failed)
            devices = selected_devices(args.gpus, gpu_inventory())
            args.max_parallel = min(args.max_parallel, len(devices) * args.workers_per_gpu)
            return ExperimentQueue(args, tasks, repo=REPO_ROOT, inventory=gpu_inventory,
                                   inspect=inspect, report=report,
                                   archive=lambda task: None).run(devices, lock)
    except (ValueError, OSError, KeyError, TypeError) as error:
        cli.error(str(error))

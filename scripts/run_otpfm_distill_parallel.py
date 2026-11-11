#!/usr/bin/env python3
"""Distill existing OTP teachers in parallel; never pretrain a teacher.

The controller is Torch-free. --dry-run reads configs/metadata only, performs no
GPU query, writes no files and starts no training, evaluation or W&B session.
"""

import argparse
import copy
import csv
import fcntl
import hashlib
import io
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from flow2pc.utils.config import load_config  # noqa: E402
from flow2pc.utils.parallel_sweep import ParallelSweep  # noqa: E402
from flow2pc.distill.otpfm_compat import compatible_implementation  # noqa: E402
from flow2pc.data.restore import resolve_data_root  # noqa: E402
from run_alicfm_checks import atomic_json, gpu_inventory, selected_devices  # noqa: E402

DEFAULT_DATASETS = (
    "cite5", "cite50", "cite100", "multi5", "multi50", "multi100",
    "eb", "st", "knot", "cell_tracking",
)
DATASETS = (*DEFAULT_DATASETS, "gom_w2", "gom_w2inf", "beijing_w2", "beijing_w2inf",
            "arch", "sphere", "lidar")


@dataclass
class Run:
    dataset: str
    name: str
    config: dict
    output: Path
    action: str = "distill"
    status: str = "pending"
    error: str = ""
    metrics: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    tracking_status: str = "unknown"
    retries: int = 0
    not_before: float = 0.0


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def scientific_config(config):
    result = copy.deepcopy(config)
    result.pop("logging", None)
    result.pop("output", None)
    result["distillation"].pop("device", None)
    result["distillation"].pop("resume", None)
    return result


def build_runs(args):
    runs = []
    for dataset in args.datasets:
        preset = ROOT / f"configs/distill/otpfm/otpfm_{dataset}_hclt.yaml"
        config = load_config(preset)
        if config["stage"] != "distill" or config["teacher"].get("name") != "otpfm":
            raise ValueError(f"Expected an OTP distillation preset: {preset}")
        if config["distillation"]["sampler"]["name"] != "otpfm_conditional":
            raise ValueError(f"Expected the restored OTP sampler: {preset}")
        output = args.output_root / dataset / Path(config["output"]["dir"]).name
        config["output"]["dir"] = str(output)
        config["distillation"].update(device="cuda:0", resume=True)
        config["logging"].update(project="flow2pc", mode=args.mode)
        runs.append(Run(dataset, f"otpfm-{dataset}-hclt-s{config['seed']}", config, output))
    return runs


def inspect_run(run, retry_incomplete=False):
    try:
        teacher_path = Path(run.config["teacher"]["checkpoint"])
        teacher = read_json(teacher_path.parents[1] / "manifest.json")
        if not teacher_path.is_file() or teacher.get("method") != "otpfm":
            raise ValueError(f"Missing OTP teacher/checkpoint metadata: {teacher_path}")
        native = teacher["model_config"]["native_config"]
        if native["potential"] != "w2inf" or native.get("solver_type") not in (None, "direct"):
            raise ValueError("This batch requires the supported native direct W2Inf sampler")
        resolve_data_root(teacher["dataset"], run.config["teacher"].get("data_root"))
        if not run.output.exists():
            run.status, run.action, run.error = "pending", "distill", ""
            return
        saved = run.output / "resolved_config.yaml"
        if not saved.is_file():
            if (run.output / "checkpoints/last.pt").is_file():
                raise ValueError("Saved checkpoint has no resolved config; preserve it for inspection")
        elif scientific_config(load_config(saved)) != scientific_config(run.config):
            raise ValueError("Saved settings differ from the preset; use a new --output-root")
        run.state = read_json(run.output / "status.json")
        run.error = run.state.get("error", "")
        run.tracking_status = read_json(run.output / "tracking_status.json").get("status", "unknown")
        run.metrics = read_json(run.output / "eval/population/evaluation.json")
        checkpoint = run.output / "checkpoints/last.pt"
        if not checkpoint.is_file():
            run.action = "restart"
            run.status = "pending" if retry_incomplete else "blocked"
            run.error = run.error or "Partial directory has no checkpoint; --retry-incomplete preserves it before restarting"
            return
        binding = read_json(run.output / "manifest.json")
        if binding.get("teacher_sha256") != teacher["checkpoint_sha256"]:
            raise ValueError("Saved student belongs to another teacher; use a new output root")
        media = read_json(run.output / "eval/media.json")
        selected = run.state.get("best_step")
        score = run.state.get("best_selection_score")
        complete = (
            run.state.get("status") == "complete"
            and run.state.get("steps") == run.config["distillation"]["steps"]
            and isinstance(score, (int, float)) and math.isfinite(score)
            and (run.output / "checkpoints/best.pt").is_file()
            and read_json(run.output / "eval/metrics.json").get("checkpoint_step") == selected
            and run.metrics.get("checkpoint_step") == selected
            and bool(media) and all(Path(p).is_file() for p in media.values())
        )
        if complete:
            run.status, run.error = "complete", ""
            return
        expected = hashlib.sha256((ROOT / "src/flow2pc/distill/otpfm_sampling.py").read_bytes()).hexdigest()
        if not compatible_implementation(
            binding.get("sampler", {}).get("implementation_sha256"), expected
        ):
            raise ValueError("Sampler code changed since the saved run; use a fresh output root")
        run.status, run.action = "pending", "distill"
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as error:
        run.status, run.error = "blocked", str(error)


def archive_incomplete(run):
    if (run.output / "checkpoints/last.pt").is_file():
        raise ValueError("Refusing to archive a resumable student checkpoint")
    if run.output.exists():
        destination = run.output.parents[1] / "_attempts" / run.dataset / str(time.time_ns())
        destination.parent.mkdir(parents=True, exist_ok=True)
        run.output.rename(destination)
        print(f"Preserved incomplete output: {destination}", flush=True)


class DistillationQueue(ParallelSweep):
    def launch(self, task, device, inventory, lock):
        # The shared pretrain queue labels saved-checkpoint retries 'evaluate'.
        # A distill resume already reloads its checkpoint, finishes any remaining
        # updates and performs student evaluation; never dispatch teacher eval.
        if task.action == "restart":
            self.archive(task)
        task.action = "distill"
        super().launch(task, device, inventory, lock)


def summary(args, runs, active, state):
    rows = []
    for run in runs:
        row = {
            "dataset": run.dataset, "run": run.name, "status": run.status,
            "best_step": run.state.get("best_step"),
            "validation_velocity_mse": run.state.get("best_selection_score"),
            "teacher_ode_emd_mean": run.metrics.get("teacher_ode_emd_mean"),
            "student_ode_emd_mean": run.metrics.get("student_ode_emd_mean"),
            "teacher_ode_emd_heldout": run.metrics.get("teacher_ode_emd_heldout"),
            "student_ode_emd_heldout": run.metrics.get("student_ode_emd_heldout"),
            "checkpoint": str(run.output / "checkpoints/best.pt"),
            "media_manifest": str(run.output / "eval/media.json"),
            "tracking_status": run.tracking_status, "error": run.error,
        }
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
    atomic_json(args.output_root / "_launcher/status.json", {
        "status": state, "total": len(runs),
        "counts": {name: sum(r.status == name for r in runs) for name in
                   ("pending", "running", "complete", "failed", "blocked", "interrupted")},
        "active": [{"run": w["task"].name, "gpu": w["device"], "pid": w["process"].pid} for w in active],
    })


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DEFAULT_DATASETS))
    result.add_argument("--output-root", type=Path, default=Path("/data/lthung/flow2pc/distill/otpfm"))
    result.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    result.add_argument("--workers-per-gpu", type=int, default=2)
    result.add_argument("--max-parallel", type=int, default=16)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--min-free-gb", type=float, default=12)
    result.add_argument("--min-disk-free-gb", type=float, default=20)
    result.add_argument("--worker-retries", type=int, default=1)
    result.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    result.add_argument("--retry-incomplete", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--summarize-only", action="store_true")
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if (len(args.datasets) != len(set(args.datasets))
            or min(args.workers_per_gpu, args.threads) < 1 or args.max_parallel < 2
            or args.worker_retries < 0
            or any(not math.isfinite(v) or v <= 0 for v in (args.min_free_gb, args.min_disk_free_gb))):
        cli.error("Invalid datasets or parallel resource settings")
    args.output_root = args.output_root.expanduser().resolve()
    try:
        runs = build_runs(args)
        for run in runs:
            inspect_run(run, args.retry_incomplete)
        print(json.dumps({
            "dry_run": args.dry_run, "stage": "distill", "teacher_training": False,
            "gpus": args.gpus, "max_parallel": args.max_parallel,
            "workers_per_gpu": args.workers_per_gpu, "min_free_gb": args.min_free_gb,
            "output_root": str(args.output_root), "wandb_project": "flow2pc",
            "controller_loaded_torch": "torch" in sys.modules,
            "counts": {name: sum(r.status == name for r in runs) for name in
                       ("pending", "complete", "blocked")},
            "runs": [{"dataset": r.dataset, "status": r.status, "error": r.error,
                      "output": str(r.output)} for r in runs],
        }, indent=2), flush=True)
        if args.dry_run:
            return 0
        args.output_root.mkdir(parents=True, exist_ok=True)
        with (args.output_root / ".distill.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                cli.error("Another distillation controller or its workers owns this output root")
            # Recheck after acquiring ownership, before deciding what to launch.
            for run in runs:
                inspect_run(run, args.retry_incomplete)
            if args.summarize_only or not any(r.status == "pending" for r in runs):
                failed = any(r.status == "blocked" for r in runs)
                summary(args, runs, [], "summary_only" if args.summarize_only else "failed" if failed else "complete")
                return int(failed)
            devices = selected_devices(args.gpus, gpu_inventory())
            args.max_parallel = min(args.max_parallel, len(devices) * args.workers_per_gpu)
            atomic_json(args.output_root / "distillation_plan.json", {
                "teacher_training": False, "runs": [{"dataset": r.dataset, "name": r.name,
                    "config": r.config} for r in runs],
            })
            return DistillationQueue(args, runs, repo=ROOT, inventory=gpu_inventory,
                inspect=inspect_run, report=summary, archive=archive_incomplete).run(devices, lock)
    except (ValueError, OSError, RuntimeError, yaml.YAMLError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Distill existing OTP/ALI teachers in parallel; never pretrain a teacher.

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
from contextlib import ExitStack
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
            "arch", "sphere", "lidar", "gom", "beijing")
DEFAULT_ALICFM_DATASETS = (
    "cite5", "cite50", "cite100", "multi5", "multi50", "multi100", "cell_tracking",
)
ALL_DATASETS = (*DEFAULT_DATASETS, "gom", "beijing", "arch", "sphere", "lidar")
SELECTED_ROOT = Path("/data/lthung/flow2pc/distill")


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
    method = getattr(args, "method", "otpfm")
    if method == "both":
        datasets = args.datasets or ALL_DATASETS
        if set(datasets) - set(ALL_DATASETS):
            raise ValueError("--method both uses dataset names gom/beijing, without recipe suffixes")
        by_method = []
        for name in ("alicfm", "otpfm"):
            child = copy.copy(args)
            child.method = name
            child.datasets = [
                f"{dataset}_w2" if name == "otpfm" and dataset in ("gom", "beijing") else dataset
                for dataset in datasets
            ]
            child.output_root = selected_output_root(args, name)
            if args.teacher_root is not None:
                child.teacher_root = args.teacher_root / name
            by_method.append(build_runs(child))
        # One shared queue interleaves methods, with a single concurrency cap.
        return [run for pair in zip(*by_method) for run in pair]
    defaults = DEFAULT_ALICFM_DATASETS if method == "alicfm" else DEFAULT_DATASETS
    for dataset in args.datasets or defaults:
        preset = ROOT / f"configs/distill/{method}/{method}_{dataset}_hclt.yaml"
        config = load_config(preset)
        if config["stage"] != "distill" or config["teacher"].get("name", "alicfm") != method:
            raise ValueError(f"Expected a {method} distillation preset: {preset}")
        expected_sampler = "teacher_interpolant" if method == "alicfm" else "otpfm_conditional"
        if config["distillation"]["sampler"]["name"] != expected_sampler:
            raise ValueError(f"Expected sampler {expected_sampler}: {preset}")
        teacher_root = getattr(args, "teacher_root", None)
        if teacher_root is not None and config["teacher"].get("format") != "legacy":
            # Explicit launch-time teacher selection; keep every student and
            # dataset protocol setting from the existing YAML unchanged.
            directory = teacher_root.expanduser().resolve() / dataset
            manifest = read_json(directory / "manifest.json")
            if manifest.get("method") != method:
                raise ValueError(f"Missing/incompatible teacher manifest: {directory}")
            relative = Path(manifest["checkpoint_relpath"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Teacher checkpoint path escapes its run directory")
            config["teacher"].update(name=method, checkpoint=str(directory / relative))
        output = args.output_root / dataset / Path(config["output"]["dir"]).name
        config["output"]["dir"] = str(output)
        config["distillation"].update(device="cuda:0", resume=True)
        config["logging"].update(project="flow2pc", mode=args.mode)
        runs.append(Run(dataset, f"{method}-{dataset}-hclt-s{config['seed']}", config, output))
    return runs


def selected_output_root(args, method):
    return (getattr(args, f"{method}_output_root", None)
            or SELECTED_ROOT / method / "selected-students").expanduser().resolve()


def teacher_identity(config):
    """Read/check bindings only, including an explicitly declared legacy ALI model."""
    source = config["teacher"]
    path = Path(source["checkpoint"])
    method = source.get("name", "alicfm")
    if not path.is_file():
        raise ValueError(f"Missing frozen teacher checkpoint: {path}")
    metadata_path = source.get("legacy_metadata")
    if metadata_path is not None:
        if method != "alicfm":
            raise ValueError("Only ALI checkpoints support this explicit legacy metadata")
        metadata_path = Path(metadata_path)
        metadata = yaml.safe_load(metadata_path.read_text())
        if metadata.get("time_input") != "global" or metadata.get("velocity_time_units") != "model":
            raise ValueError("This legacy batch requires declared global model time")
        if not Path(metadata["data"]["root"]).is_dir():
            raise ValueError("Legacy teacher data root is missing")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if metadata.get("checkpoint_sha256") != digest:
            raise ValueError("Legacy teacher hash differs from its pinned metadata")
        generator = source.get("generator_checkpoint")
        generator_digest = hashlib.sha256(Path(generator).read_bytes()).hexdigest() if generator else None
        return method, digest, hashlib.sha256(metadata_path.read_bytes()).hexdigest(), generator_digest
    teacher = read_json(path.parents[1] / "manifest.json")
    if teacher.get("method") != method:
        raise ValueError(f"Missing/incompatible frozen teacher metadata: {path}")
    if method == "otpfm":
        native = teacher["model_config"]["native_config"]
        if native["potential"] != "w2inf" or native.get("solver_type") not in (None, "direct"):
            raise ValueError("This batch requires the supported native direct W2Inf sampler")
    resolve_data_root(teacher["dataset"], source.get("data_root"))
    return method, teacher["checkpoint_sha256"], None, None


def inspect_run(run, retry_incomplete=False):
    try:
        method, teacher_sha, metadata_sha, generator_sha = teacher_identity(run.config)
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
        if (binding.get("teacher_sha256") != teacher_sha
                or binding.get("legacy_metadata_sha256") != metadata_sha
                or binding.get("generator_sha256") != generator_sha):
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
        if method == "otpfm":
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
            "method": run.config["teacher"].get("name", "alicfm"),
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
    result.add_argument("--method", choices=("otpfm", "alicfm", "both"), default="otpfm")
    result.add_argument("--datasets", nargs="+", choices=DATASETS)
    result.add_argument("--output-root", type=Path)
    result.add_argument("--teacher-root", type=Path,
                        help="METHOD dataset-folder root, or parent of alicfm/otpfm for --method both; legacy presets retain their checkpoint")
    result.add_argument("--alicfm-output-root", type=Path,
                        help="ALI student output root for --method both (default: distill/alicfm/selected-students)")
    result.add_argument("--otpfm-output-root", type=Path,
                        help="OTP student output root for --method both (default: distill/otpfm/selected-students)")
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
    if args.datasets is None:
        args.datasets = list(ALL_DATASETS if args.method == "both" else
                             DEFAULT_ALICFM_DATASETS if args.method == "alicfm" else DEFAULT_DATASETS)
    if args.output_root is None:
        args.output_root = SELECTED_ROOT / ("all-students" if args.method == "both" else args.method)
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
            "dry_run": args.dry_run, "stage": "distill", "method": args.method, "teacher_training": False,
            "gpus": args.gpus, "max_parallel": args.max_parallel,
            "workers_per_gpu": args.workers_per_gpu, "min_free_gb": args.min_free_gb,
            "output_root": str(args.output_root), "wandb_project": "flow2pc",
            "controller_loaded_torch": "torch" in sys.modules,
            "counts": {name: sum(r.status == name for r in runs) for name in
                       ("pending", "complete", "blocked")},
            "runs": [{"method": r.config["teacher"].get("name", "alicfm"),
                      "dataset": r.dataset, "status": r.status, "error": r.error,
                      "output": str(r.output)} for r in runs],
        }, indent=2), flush=True)
        if args.dry_run:
            return 0
        roots = [args.output_root]
        if args.method == "both":
            roots.extend(selected_output_root(args, method) for method in ("alicfm", "otpfm"))
        with ExitStack() as stack:
            locks = []
            for root in sorted(set(roots)):
                root.mkdir(parents=True, exist_ok=True)
                handle = stack.enter_context((root / ".distill.lock").open("a"))
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    cli.error(f"Another distillation controller or its workers owns {root}; let it finish before launching this batch")
                locks.append(handle)
            lock = locks[0]
            # Workers retain ownership even if the controller exits unexpectedly.
            args.worker_lock_fds = tuple(handle.fileno() for handle in locks[1:])
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

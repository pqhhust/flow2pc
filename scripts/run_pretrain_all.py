#!/usr/bin/env python3
"""Train existing ALI-CFM/OTP-FM presets with the shared independent-GPU queue.

No hyperparameter changes or student jobs. --dry-run reads metadata only: no
models, GPU queries, training, evaluation, W&B sessions, or output writes.
"""

import argparse
import copy
import csv
import fcntl
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from flow2pc.utils.config import resolve_config  # noqa: E402
from flow2pc.evaluation.config import transport_w2_complete  # noqa: E402
from flow2pc.utils.parallel_sweep import ParallelSweep  # noqa: E402
from run_alicfm_checks import atomic_json, gpu_inventory, selected_devices  # noqa: E402

METHODS = ("alicfm", "otpfm")


@dataclass
class Run:
    method: str
    dataset: str
    name: str
    source_config: Path
    config: dict
    output: Path
    action: str = "pretrain"
    status: str = "pending"
    error: str = ""
    retries: int = 0
    not_before: float = 0.0


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def scientific_config(config):
    result = copy.deepcopy(config)
    for key in ("logging", "output"):
        result.pop(key, None)
    result["training"].pop("device", None)
    return result


def build_runs(args):
    runs = []
    for method in args.methods:
        paths = sorted((ROOT / "configs/pretrain" / method).glob(f"{method}_*.yaml"))
        if not paths:
            raise ValueError(f"No pretraining presets found for {method}")
        for path in paths:
            raw = yaml.safe_load(path.read_text())
            if raw["stage"] != "pretrain" or raw["model"]["name"] != method:
                raise ValueError(f"Unexpected teacher preset: {path}")
            tag = path.stem.removeprefix(method + "_")
            if args.presets and path.stem not in args.presets:
                continue
            output = args.output_root / method / tag
            raw["output"]["dir"] = str(output)
            raw["training"]["device"] = "cuda:0"
            raw["logging"].update(
                mode=args.mode, project="flow2pc", name=f"{path.stem}-s{raw['seed']}-pretrain"
            )
            # These prepared arrays were relocated; only their filesystem path
            # changes. Keep the YAML's experiment, saved splits, and all budgets.
            if raw["data"]["name"] in ("arch", "sphere", "lidar"):
                original = Path(raw["data"]["root"])
                if args.prepared_root is not None or not original.is_dir():
                    parent = args.prepared_root or Path("/data/pqhung_group/pdtung/flow2pc/data")
                    raw["data"]["root"] = str(parent / raw["data"]["name"] / "seed42")
                from flow2pc.data.mfm_snapshots import read_manifest

                manifest = read_manifest(raw["data"]["root"])
                if manifest["profile"] != raw["data"]["name"]:
                    raise ValueError(f"Prepared dataset profile differs: {path}")
            config = resolve_config(raw, base=ROOT)
            data_root = Path(config["data"]["root"])
            if not data_root.is_dir():
                raise ValueError(f"Dataset root missing for {path.stem}: {data_root}")
            if config["data"]["name"] in ("gom", "beijing"):
                bundle = data_root / config["data"]["name"] / "marginals.npz"
                if not bundle.is_file():
                    raise ValueError(f"Prepared transport bundle missing: {bundle}")
            if not config["evaluation"]["enabled"]:
                raise ValueError(f"Preset must retain enabled final evaluation: {path}")
            runs.append(Run(method, tag, path.stem, path, config, output))
    if args.presets:
        missing = set(args.presets) - {r.name for r in runs}
        if missing:
            raise ValueError(f"Unknown/excluded presets: {sorted(missing)}")
    # Alternate methods while both have pending work; one common pool of GPUs.
    groups = [[r for r in runs if r.method == m] for m in args.methods]
    return [group[i] for i in range(max(map(len, groups))) for group in groups if i < len(group)]


def checkpoint_for(run):
    manifest = read_json(run.output / "manifest.json")
    relative = Path(manifest.get("checkpoint_relpath", "checkpoints/last.pt"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid checkpoint path in {run.output / 'manifest.json'}")
    return run.output / relative


def inspect_run(run, retry_incomplete=False):
    if not run.output.exists():
        return
    try:
        saved = run.output / "resolved_config.yaml"
        if not saved.is_file() or scientific_config(yaml.safe_load(saved.read_text())) != scientific_config(run.config):
            raise ValueError("Existing configuration differs or is missing; use a fresh output root")
        state = read_json(run.output / "status.json")
        media = read_json(run.output / "eval/media.json")
        complete = (
            state.get("status") == "complete"
            and checkpoint_for(run).is_file()
            and (run.output / "preprocessing.json").is_file()
            and read_json(run.output / "manifest.json").get("method") == run.method
            and read_json(run.output / "eval/status.json").get("status") == "complete"
            and (run.output / "eval/metrics.json").is_file()
            and bool(media)
            and all(Path(p).is_file() for p in media.values())
        )
        if complete and run.method == "alicfm" and run.config["data"]["name"] in ("gom", "beijing"):
            manifest = read_json(run.output / "manifest.json")
            complete = transport_w2_complete(read_json(run.output / "eval/metrics.json"),
                                            manifest["dataset"]["heldout_indices"])
            if not complete:
                run.status, run.action, run.error = "pending", "evaluate", ""
                return
        run.status = "complete" if complete else "blocked"
        run.error = "" if complete else state.get("error", "Incomplete output preserved; use a fresh output root for this preset")
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        run.status, run.error = "blocked", str(error)


def report(args, runs, active, state):
    rows = []
    for run in runs:
        metrics = read_json(run.output / "eval/metrics.json")
        media = read_json(run.output / "eval/media.json")
        rows.append({
            "method": run.method, "preset": run.name, "status": run.status,
            "source_config": str(run.source_config), "output": str(run.output),
            "checkpoint": str(checkpoint_for(run)) if (run.output / "manifest.json").is_file() else "",
            "teacher_mean_emd": metrics.get("teacher_mean_emd"),
            "teacher_heldout_emd": metrics.get("teacher_heldout_emd"),
            "teacher_heldout_w2": metrics.get("teacher_heldout_w2"),
            "media_manifest": str(run.output / "eval/media.json") if media else "",
            "images": ";".join(str(p) for p in media.values()), "error": run.error,
        })
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    target = args.output_root / "summary.csv"
    temporary = target.with_suffix(".csv.tmp")
    try:
        temporary.write_text(stream.getvalue())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(args.output_root / "checkpoints.json", rows)
    atomic_json(args.output_root / "_launcher/status.json", {
        "status": state, "total": len(runs),
        "counts": {s: sum(r.status == s for r in runs) for s in
                   ("pending", "running", "complete", "failed", "blocked", "interrupted")},
        "active": [{"run": w["task"].name, "gpu": w["device"], "pid": w["process"].pid} for w in active],
    })


def no_archive(run):
    raise ValueError(f"Partial training output is preserved: {run.output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--presets", nargs="+", help="Optional exact YAML stems, e.g. alicfm_cite50 otpfm_cite50")
    parser.add_argument("--output-root", type=Path, default=Path("/data/lthung/flow2pc/pretrained-teachers"))
    parser.add_argument("--prepared-root", type=Path, help="Parent containing arch/sphere/lidar/seed42")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--min-free-gb", type=float, default=12)
    parser.add_argument("--min-disk-free-gb", type=float, default=20)
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (args.threads < 1 or len(set(args.methods)) != len(args.methods)
            or any(not math.isfinite(v) or v <= 0 for v in (args.min_free_gb, args.min_disk_free_gb))):
        parser.error("Invalid resource limits or duplicate methods")
    args.output_root = args.output_root.expanduser().resolve()
    if args.prepared_root is not None:
        args.prepared_root = args.prepared_root.expanduser().resolve()
    # Reuse the existing queue unchanged: one independent teacher per GPU.
    args.workers_per_gpu, args.max_parallel, args.worker_retries = 1, 8, 0
    try:
        runs = build_runs(args)
        for run in runs:
            inspect_run(run)
        print(json.dumps({
            "dry_run": args.dry_run, "runs": len(runs), "gpus": args.gpus,
            "max_parallel": 8, "workers_per_gpu": 1, "project": "flow2pc",
            "output_root": str(args.output_root),
            "presets": [{"name": r.name, "status": r.status, "data_root": r.config["data"]["root"],
                         "output": str(r.output), "error": r.error} for r in runs],
        }, indent=2), flush=True)
        if args.dry_run:
            return 0
        devices = selected_devices(args.gpus, gpu_inventory())
        args.max_parallel = min(8, len(devices))
        args.output_root.mkdir(parents=True, exist_ok=True)
        with (args.output_root / ".pretrain.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("An active teacher controller/worker owns this output root")
            # Recheck after obtaining the lock to avoid stale completion state.
            for run in runs:
                inspect_run(run)
            if any(r.status == "blocked" for r in runs):
                parser.error("Incomplete/conflicting output preserved; select a fresh --output-root (or --presets subset)")
            atomic_json(args.output_root / "pretrain_plan.json", {
                "runs": [{"source_config": str(r.source_config), "config": r.config} for r in runs]
            })
            return ParallelSweep(args, runs, repo=ROOT, inventory=gpu_inventory,
                                 inspect=inspect_run, report=report, archive=no_archive).run(devices, lock)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

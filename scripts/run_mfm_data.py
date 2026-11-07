#!/usr/bin/env python3
"""Prepare and run the non-AFHQ MFM catalog. --dry-run never writes or trains."""

import argparse
import fcntl
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flow2pc.data.mfm_catalog import (  # noqa: E402
    DATA_ROOT,
    PREPARED_ROOT,
    PROFILES,
    link_sources,
    prepared_path,
)
from flow2pc.utils.paths import MFM_PRETRAIN_ROOT  # noqa: E402


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("list", "prepare", "pretrain", "distill", "both"))
    result.add_argument("profile", nargs="?", default="all", choices=(*PROFILES, "all"))
    result.add_argument("--student", choices=("hclt", "gaussian", "both"), default="hclt")
    result.add_argument("--data-root", type=Path, default=PREPARED_ROOT)
    result.add_argument("--catalog-root", type=Path, default=DATA_ROOT)
    result.add_argument("--bundle", type=Path, default=ROOT / "data")
    result.add_argument("--source-root", type=Path, default=MFM_PRETRAIN_ROOT)
    result.add_argument("--native-data", type=Path, default=MFM_PRETRAIN_ROOT / "data")
    result.add_argument("--output-root", type=Path, default=ROOT / "outputs",
                        help="Root containing pretrain/mfm and distill/mfm outputs")
    result.add_argument("--run-name", default="default")
    result.add_argument("--teacher-run", type=Path, help="Existing teacher run, for distill of one profile")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="auto-free", help="Training device (default: wait for an available GPU)")
    result.add_argument("--gpus", default="auto", help="Physical GPU indices/UUIDs for auto-free, or auto")
    result.add_argument("--min-free-gb", type=float, default=8.0, help="Minimum free GPU memory for auto-free")
    result.add_argument("--gpu-poll-seconds", type=int, default=20)
    result.add_argument("--epochs", type=int, help="Override teacher epochs per phase")
    result.add_argument("--steps", type=int, help="Override student update count")
    result.add_argument("--batch-size", type=int, help="Override requested batch size for both stages")
    result.add_argument("--logging", choices=("disabled", "offline", "online"), default="online")
    result.add_argument("--python", default=os.environ.get("FLOW2PC_PYTHON", sys.executable))
    result.add_argument("--links-only", action="store_true", help="For prepare: only create source symlinks/catalog")
    result.add_argument("--dry-run", action="store_true")
    return result


def commands(args, name):
    output_root = args.output_root.expanduser().resolve()
    pretrain_output = output_root / "pretrain/mfm" / name / args.run_name
    distill_output = output_root / "distill/mfm" / name / args.run_name
    teacher = args.teacher_run.expanduser().resolve() if args.teacher_run else pretrain_output
    prepared = prepared_path(args.data_root, name, args.seed)
    if args.teacher_run and not args.dry_run:
        pair = json.loads((teacher / "teacher_pair.json").read_text())
        if pair["binding"]["profile"] != name:
            raise ValueError("--teacher-run belongs to a different data profile")
        prepared = Path(pair["binding"]["prepared_path"])
    common = {"seed": args.seed, "wandb.mode": args.logging,
              "wandb.project": "flow2pc", "wandb.group": f"{name}-mfm"}
    stages = []
    if args.action in ("pretrain", "both"):
        values = {**common, "data.root": str(prepared), "output.dir": str(pretrain_output),
                  "model.params.source_root": str(args.source_root.expanduser().resolve()),
                  "training.device": "cuda:0" if args.device == "auto-free" else args.device}
        if args.epochs is not None:
            values["training.epochs"] = args.epochs
        if args.batch_size is not None:
            values["training.batch_size"] = args.batch_size
        stages.append((ROOT / f"configs/pretrain/mfm/mfm_{name}.yaml", values))
    if args.action in ("distill", "both"):
        students = ("hclt", "gaussian") if args.student == "both" else (args.student,)
        for student in students:
            values = {**common, "data.root": str(prepared), "output.dir": str(distill_output / student),
                      "teacher.checkpoint": str(teacher / "checkpoints/flow.pt"),
                      "teacher.params.geopath_checkpoint": str(teacher / "checkpoints/geopath.pt"),
                      "teacher.params.repo_path": str(args.source_root.expanduser().resolve()),
                      "distillation.device": "cuda:0" if args.device == "auto-free" else args.device}
            if args.steps is not None:
                values["distillation.steps"] = args.steps
            if args.batch_size is not None:
                values["distillation.batch_size"] = args.batch_size
            stages.append((ROOT / f"configs/distill/mfm/mfm_{name}_{student}.yaml", values))
    result = []
    for config, overrides in stages:
        if not config.is_file():
            raise FileNotFoundError(f"Recipe missing: {config}")
        command = [args.python, str(ROOT / "main.py"), "--config", str(config)]
        for key, value in overrides.items():
            command.extend(("--set", f"{key}={json.dumps(value)}"))
        result.append(command)
    return result


def gpu_inventory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    inventory = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3 or not parts[0].isdigit() or not parts[2].isdigit():
            raise ValueError(f"Unexpected nvidia-smi row: {line!r}")
        inventory[parts[0]] = {"uuid": parts[1], "free_mib": int(parts[2])}
    if not inventory:
        raise ValueError("nvidia-smi reported no GPUs")
    return inventory


def eligible_gpus(requested, inventory):
    def resolve(token):
        token = token.strip()
        if token in inventory:
            return token
        matches = [index for index, gpu in inventory.items() if token and gpu["uuid"].startswith(token)]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous GPU: {token!r}")
        return matches[0]

    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = list(inventory) if inherited is None else [resolve(value) for value in inherited.split(",") if value.strip()]
    selected = visible if requested == "auto" else [resolve(value) for value in requested.split(",")]
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(visible):
        raise ValueError("GPU selection is empty, duplicated, or outside CUDA_VISIBLE_DEVICES")
    return selected


def reserve_gpu(args):
    locks = ROOT / "outputs/.gpu_locks/mfm"
    locks.mkdir(parents=True, exist_ok=True)
    last_notice = 0.0
    while True:
        inventory = gpu_inventory()
        for index in sorted(eligible_gpus(args.gpus, inventory),
                            key=lambda key: inventory[key]["free_mib"], reverse=True):
            if inventory[index]["free_mib"] < args.min_free_gb * 1024:
                continue
            uuid = inventory[index]["uuid"]
            lock_name = re.sub(r"[^A-Za-z0-9_-]", "_", uuid)
            lock = (locks / f"{lock_name}.lock").open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            refreshed = gpu_inventory()
            if (refreshed.get(index, {}).get("uuid") == uuid
                    and refreshed[index]["free_mib"] >= args.min_free_gb * 1024):
                return index, uuid, lock
            lock.close()
        if last_notice == 0 or time.monotonic() - last_notice >= 60:
            print(f"Waiting for a GPU with at least {args.min_free_gb:g} GiB free", flush=True)
            last_notice = time.monotonic()
        time.sleep(args.gpu_poll_seconds)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.action == "list":
        for name, spec in PROFILES.items():
            print(f"{name:25s} {spec['dim']:3d}D  holdouts={spec['holdouts']}")
        return 0
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        cli.error("--run-name must be a single directory name")
    if args.teacher_run and (args.action != "distill" or args.profile == "all"):
        cli.error("--teacher-run requires distill and one profile")
    if args.links_only and args.action != "prepare":
        cli.error("--links-only applies only to prepare")
    if (not 0 <= args.seed < 2**32
            or any(x is not None and x < 1 for x in (args.epochs, args.steps, args.batch_size))
            or not math.isfinite(args.min_free_gb) or args.min_free_gb <= 0
            or args.gpu_poll_seconds < 1):
        cli.error("Invalid seed or training count")
    names = list(PROFILES) if args.profile == "all" else [args.profile]
    try:
        plans = [(name, commands(args, name)) for name in names]
        if args.action in ("prepare", "both"):
            print(f"Prepare source links under {args.catalog_root / 'sources'}", flush=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "src"), env.get("PYTHONPATH"))))
        for name, jobs in plans:
            if args.links_only and name != names[0]:
                continue
            if args.action in ("prepare", "both") and not args.links_only:
                print(f"Prepare {name}: {prepared_path(args.data_root, name, args.seed)}", flush=True)
            if args.action in ("prepare", "both") and not args.dry_run:
                catalog_root = args.catalog_root.expanduser().resolve()
                catalog_root.mkdir(parents=True, exist_ok=True)
                with (catalog_root / ".mfm_prepare.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    link_sources(catalog_root, bundle=args.bundle, native_data=args.native_data)
                    if not args.links_only:
                        # One profile prepares at a time; source links stay stable.
                        subprocess.run([
                            args.python, "-m", "flow2pc.data.prepare_mfm", name,
                            "--data-root", str(args.data_root.expanduser().resolve()),
                            "--catalog-root", str(catalog_root), "--seed", str(args.seed),
                            "--source-root", str(args.source_root.expanduser().resolve()),
                        ], env=env, cwd=ROOT, check=True)
            gpu_lock = None
            run_env = env
            try:
                if jobs and args.device == "auto-free":
                    if args.dry_run:
                        print(f"{name}: would wait for a GPU with {args.min_free_gb:g} GiB free", flush=True)
                    else:
                        index, uuid, gpu_lock = reserve_gpu(args)
                        run_env = {**env, "CUDA_VISIBLE_DEVICES": uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
                        print(f"{name}: physical GPU {index} ({uuid}) selected as cuda:0", flush=True)
                for command in jobs:
                    print(shlex.join(command), flush=True)
                    if not args.dry_run:
                        subprocess.run(command, cwd=ROOT, env=run_env, check=True,
                                       pass_fds=(gpu_lock.fileno(),) if gpu_lock else ())
            finally:
                if gpu_lock:
                    gpu_lock.close()
    except subprocess.CalledProcessError as error:
        return error.returncode
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        cli.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

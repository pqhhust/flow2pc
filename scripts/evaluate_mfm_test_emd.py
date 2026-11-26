#!/usr/bin/env python3
"""Backfill held-out test_EMD for completed MFM snapshot teacher runs."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flow2pc.pretrain.models.mfm.test_emd import evaluate_run, publish_wandb  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--all", action="store_true", help="Find completed snapshot runs under outputs/pretrain/mfm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Velocity rollout batch size; EMD always uses full populations")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Also append diagnostics to the existing W&B run")
    args = parser.parse_args()
    if not args.all and not args.runs:
        parser.error("Provide run directories or --all")
    runs = list(args.runs)
    if args.all:
        for status in (ROOT / "outputs/pretrain/mfm").rglob("status.json"):
            if json.loads(status.read_text()).get("status") == "complete" and (status.parent / "teacher_pair.json").is_file():
                runs.append(status.parent)
    for run in sorted(set(runs)):
        report = evaluate_run(run, device=args.device, batch_size=args.batch_size,
                              steps=args.steps, force=args.force)
        if args.wandb:
            publish_wandb(run, report)
        value = "no temporal holdout" if report["test_EMD"] is None else f"test_EMD={report['test_EMD']:.6f}"
        print(f"{run}: {value} ({run / 'eval/test_emd.json'})", flush=True)


if __name__ == "__main__":
    main()

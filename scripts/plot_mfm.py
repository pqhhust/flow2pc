"""Export MFM student ODE figures from tuning checkpoints."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from flow2pc.distill.student import build_student_from_config
from flow2pc.evaluation.mfm_student_plots import export as export_plots


def export(checkpoint_path: Path, device: torch.device):
    return export_plots(checkpoint_path, device, build_student_from_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/tuning/mfm")
    parser.add_argument("--datasets", nargs="+", choices=("arch", "sphere", "lidar"), default=["arch", "sphere", "lidar"])
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoints = [
        path for name in args.datasets
        for path in sorted((args.root / name).glob(f"*/runs/*/checkpoints/{args.checkpoint}.pt"))
    ]
    if not checkpoints:
        parser.error(f"No matching checkpoints under {args.root}")
    print(f"Rendering {len(checkpoints)} runs on {args.device}", flush=True)
    total = 0
    for checkpoint in checkpoints:
        total += export(checkpoint, torch.device(args.device))
    print(f"Done: {total} images across {len(checkpoints)} runs", flush=True)


if __name__ == "__main__":
    main()

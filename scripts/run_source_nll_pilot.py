#!/usr/bin/env python3
"""Source-NLL ablations using existing dataset recipes and the student queue.

Metadata-only --dry-run does not import Torch or launch GPU/model work.
No automatic lambda winner is selected from population/test metrics.
"""

import argparse
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import run_otpfm_distill_sweep as sweep

batch, ROOT = sweep.batch, sweep.ROOT
DATASETS = ("st", "cite5", "cite50", "cite100", "knot")


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--method", choices=("both", "alicfm", "otpfm"), default="both")
    cli.add_argument("--datasets", nargs="+", choices=DATASETS, default=["st"])
    cli.add_argument("--weights", nargs="+", type=float, default=[0., .01, .1, 1.])
    cli.add_argument("--lr-factors", nargs="+", type=float, default=[1.],
                     help="Multiply each method's selected learning rate; default preserves the original pilot")
    cli.add_argument("--steps", type=int)
    cli.add_argument("--output-root", type=Path, default=Path("/data/lthung/flow2pc/distill/source-nll-pilot-v1"))
    cli.add_argument("--gpus", default="0,1,2,4,5,6,7")
    cli.add_argument("--workers-per-gpu", type=int, default=1)
    cli.add_argument("--max-parallel", type=int, default=7)
    cli.add_argument("--threads", type=int, default=4)
    cli.add_argument("--min-free-gb", type=float, default=4.)
    cli.add_argument("--min-disk-free-gb", type=float, default=20.)
    cli.add_argument("--worker-retries", type=int, default=1)
    cli.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    cli.add_argument("--retry-incomplete", action="store_true")
    cli.add_argument("--summarize-only", action="store_true")
    cli.add_argument("--dry-run", action="store_true")
    return cli


def build_runs(args):
    datasets = getattr(args, "datasets", ["st"])
    if not datasets or len(set(datasets)) != len(datasets) or set(datasets) - set(DATASETS):
        raise ValueError("Use distinct supported datasets: " + ", ".join(DATASETS))
    if (not args.weights or len(set(args.weights)) != len(args.weights)
            or any(not math.isfinite(w) or w < 0 for w in args.weights)
            or not args.lr_factors or len(set(args.lr_factors)) != len(args.lr_factors)
            or any(not math.isfinite(f) or f <= 0 for f in args.lr_factors)
            or (args.steps is not None and args.steps < 1)):
        raise ValueError("Use distinct finite nonnegative NLL weights, positive LR factors and positive steps")
    methods = ("alicfm", "otpfm") if args.method == "both" else (args.method,)
    runs = []
    for weight in args.weights:
        for factor in args.lr_factors:
            for dataset, method in ((dataset, method) for dataset in datasets for method in methods):
                config = copy.deepcopy(sweep.load_config(
                    ROOT / f"configs/distill/{method}/source_anchor/{method}_{dataset}_hclt.yaml"))
                cfg = config["distillation"]
                lr = cfg["lr"] * factor
                if not math.isfinite(lr) or lr <= 0:
                    raise ValueError("Scaled learning rate must be finite and positive")
                cfg.update(source_nll_weight=weight, lr=lr, resume=True, device="cuda:0")
                if args.steps is not None:
                    cfg["steps"] = args.steps
                signature = hashlib.sha256(json.dumps(batch.scientific_config(config), sort_keys=True,
                                                     allow_nan=False).encode()).hexdigest()[:12]
                # Preserve the eight original pilot identities with default arguments.
                lr_tag = "" if args.lr_factors == [1.] else f"-lr{lr:g}"
                name = f"{dataset}-{method}-hclt-source-nll{weight:g}{lr_tag}-s{config['seed']}-{signature}"
                output = args.output_root / method / dataset / name
                config["output"]["dir"] = str(output)
                config["logging"].update(project="flow2pc", mode=args.mode, name=name)
                runs.append(batch.Run(dataset, name, config, output))
    if len({run.output for run in runs}) != len(runs):
        raise ValueError("Sweep settings produced duplicate output paths")
    return runs


def summary(args, runs, active, state):
    rows = []
    for run in runs:
        row = sweep.result_row(run)
        row.update(method=run.config["teacher"]["name"],
                   source_nll_weight=run.config["distillation"]["source_nll_weight"],
                   source_validation_nll_at_velocity_best=run.state.get("best_source_nll_per_dimension"),
                   initial_source_validation_nll=batch.read_json(run.output / "source_density_initial.json").get("source_nll_per_dimension"))
        protocol = run.metrics.get("protocol", {})
        row.update(benchmark_metric=protocol.get("primary_distance"),
                   benchmark_coordinate_space=run.metrics.get("space"),
                   benchmark_rollout_origin=protocol.get("rollout_origin"),
                   benchmark_report=str(run.output / "eval/population/evaluation.json"))
        for key in ("direct_pc_emd_mean", "direct_pc_emd_heldout"):
            row[key] = run.metrics.get(key)
        heldout = [v for v in run.metrics.get("by_time", {}).values() if v.get("heldout")]
        teacher_distances = [v["direct_pc_teacher_emd"] for v in heldout
                             if v.get("direct_pc_teacher_emd") is not None]
        row["direct_pc_teacher_emd_heldout"] = (sum(teacher_distances) / len(teacher_distances)
                                               if teacher_distances else None)
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
        "status": state, "total": len(runs),
        "counts": {name: sum(run.status == name for run in runs)
                   for name in ("pending", "running", "complete", "failed", "blocked", "interrupted")},
        "active": [{"run": worker["task"].name, "gpu": worker["device"],
                    "pid": worker["process"].pid} for worker in active],
        "selection_note": "best.pt uses validation velocity MSE; lambda is not selected automatically",
    })


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    try:
        runs = build_runs(args)
        # Existing execute() owns inspection, locking, resources and workers.
        args.latents = sorted({r.config["student"]["params"]["num_latents"] for r in runs})
        args.time_features = sorted({r.config["student"]["params"]["time_embedding_dim"] for r in runs})
        args.learning_rates = sorted({r.config["distillation"]["lr"] for r in runs})
        return sweep.execute(args, runs, report=summary, details={
            "grid": {"source_nll_weight": args.weights, "lr_factors": args.lr_factors,
                     "learning_rates": args.learning_rates},
            "teachers": {r.config["teacher"]["name"] + "/" + r.dataset:
                         r.config["teacher"]["checkpoint"] for r in runs},
            "objective": "mean velocity MSE + lambda * initial NLL per coordinate",
            "training_pc_sampling": False,
            "source_density_training_split": "initial-time train cells only",
            "source_density_validation_split": "fixed initial-time val cells; no temporal test fallback",
            "lambda_selection": "manual validation velocity/density trade-off; no test-coverage selection",
        })
    except (ValueError, OSError, RuntimeError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())

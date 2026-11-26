"""Held-out snapshot W1 after MFM training, using the selected teacher pair."""

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from flow2pc.data.mfm_snapshots import load
from flow2pc.evaluation.ode import emd
from flow2pc.pretrain.models.mfm.snapshots import load_teacher
from flow2pc.utils.checkpoint import atomic_json
from flow2pc.utils.mfm_source import source_module


def _trajectory_plot(profile, teacher, source, target, *, left, next_time,
                     steps, run_dir, source_root, device, seed):
    """Render the native Arch/Sphere plot from a matched rollout."""
    import matplotlib.pyplot as plt

    count = min(100 if profile == "sphere" else 400, len(source), len(target))
    trajectory = [source[:count].clone()]
    state = source[:count].to(device)
    with torch.no_grad():
        for step in range(steps):
            time = left + (next_time - left) * step / steps
            state = state + (next_time - left) / steps * teacher.velocity(state, time)
            if not torch.isfinite(state).all():
                raise FloatingPointError("Nonfinite MFM sphere plot rollout")
            trajectory.append(state.cpu())
    # The native plotter indexes trajectories by indices into its source cloud.
    # Give it equally sized source/target clouds to preserve that correspondence.
    plot = getattr(source_module("utils", source_root), f"plot_{profile}")
    old_rng = np.random.get_state()
    try:
        np.random.seed(seed)
        figures = plot([source[:count], target[:count]], torch.stack(trajectory),
                       time_steps=[0, 1], fname=str(run_dir / f"{profile}_trajs.png"),
                       n_samples=count)
    finally:
        np.random.set_state(old_rng)
    if profile == "arch":
        figures = [figures]
    for figure in figures:
        plt.close(figure)
    return (["arch_trajs.png"] if profile == "arch" else
            ["sphere_trajs.png", *(f"sphere_trajs_view_{i}.png" for i in (2, 3, 4))])


def _lidar_plot(teacher, data, *, times, steps, run_dir, source_root, device):
    """Render the native LiDAR terrain and flow in physical scene coordinates."""
    import matplotlib.pyplot as plt

    source = data.validation_marginals[0]
    state = source.to(device)
    trajectory = [source.clone()]
    left, right = float(times[0]), float(times[-1])
    with torch.no_grad():
        for step in range(steps):
            time = left + (right - left) * step / steps
            state = state + (right - left) / steps * teacher.velocity(state, time)
            if not torch.isfinite(state).all():
                raise FloatingPointError("Nonfinite MFM LiDAR plot rollout")
            trajectory.append(state.cpu())
    mean = torch.as_tensor(data.normalization["mean"], dtype=torch.float64)
    scale = torch.as_tensor(data.normalization["scale"], dtype=torch.float64)
    raw_trajectory = torch.stack(trajectory).double() * scale + mean
    terrain = torch.cat(data.metric_marginals).double() * scale + mean
    fig = plt.figure(figsize=(5, 4))
    try:
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        ax.view_init(elev=30, azim=-115, roll=0)
        source_module("utils", source_root).plot_lidar(
            ax, terrain.float(), xs=raw_trajectory.transpose(0, 1).float()
        )
        fig.savefig(run_dir / "lidar_interpolant.png", dpi=300)
    finally:
        plt.close(fig)
    return ["lidar_interpolant.png"]


def evaluate_run(run_dir, *, device="cpu", batch_size=512, steps=100, force=False):
    """Evaluate each temporal holdout with upstream-style Euler and exact OT.

    Both empirical populations are complete. Only the velocity rollout is
    batched; transport is solved on the full pairwise cost matrix.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    destination = run_dir / "eval" / "test_emd.json"
    if destination.exists() and not force:
        cached = json.loads(destination.read_text())
        if cached.get("format") == "mfm_snapshot_test_emd_v2":
            if not (run_dir / "eval/status.json").exists():
                atomic_json(run_dir / "eval/status.json", {"status": "complete"})
            return cached
    if json.loads((run_dir / "status.json").read_text()).get("status") != "complete":
        raise ValueError(f"MFM run is not complete: {run_dir}")
    if batch_size < 1 or steps < 1:
        raise ValueError("batch_size and steps must be positive")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pair = json.loads((run_dir / "teacher_pair.json").read_text())
    binding = pair["binding"]
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    teacher = load_teacher({
        "checkpoint_path": str(run_dir / "checkpoints/flow.pt"),
        "geopath_checkpoint": str(run_dir / "checkpoints/geopath.pt"),
        "repo_path": config["model"]["params"]["source_root"],
    }, device)
    data = load({"data_dir": binding["prepared_path"]},
                {"teacher_metadata": {"snapshot_identity": binding["snapshot_identity"]}})
    times = binding["model_times"]
    results = {}
    images = []
    for index in data.heldout_indices:
        previous = max(i for i in binding["observed_indices"] if i < index)
        source = data.marginals[previous]
        target = data.test_marginals[index]
        left, right = float(times[previous]), float(times[index])
        predictions = []
        with torch.no_grad():
            for chunk in source.split(batch_size):
                state = chunk.to(device)
                for step in range(steps):
                    time = left + (right - left) * step / steps
                    state = state + (right - left) / steps * teacher.velocity(state, time)
                    if not torch.isfinite(state).all():
                        raise FloatingPointError(f"Nonfinite MFM rollout at holdout {index}")
                predictions.append(state.cpu())
        prediction = torch.cat(predictions)
        print(f"{binding['profile']} holdout {index}: exact EMD over {len(prediction)} × {len(target)} points", flush=True)
        value = emd(target, prediction, order=1)
        baseline = emd(target, source, order=1)
        mean = torch.as_tensor(data.normalization["mean"], dtype=torch.float64)
        scale = torch.as_tensor(data.normalization["scale"], dtype=torch.float64)
        raw_target = target.double() * scale + mean
        raw_source = source.double() * scale + mean
        raw_prediction = prediction.double() * scale + mean
        raw_value = emd(raw_target, raw_prediction, order=1)
        raw_baseline = emd(raw_target, raw_source, order=1)
        results[str(index)] = {"test_EMD": value, "source_index": previous,
                               "test_EMD_raw": raw_value, "baseline_EMD": baseline,
                               "baseline_EMD_raw": raw_baseline,
                               "source_count": len(source), "target_count": len(target),
                               "source_total": len(data.marginals[previous]),
                               "target_total": len(data.test_marginals[index])}
        if binding["profile"] in ("arch", "sphere"):
            profile = binding["profile"]
            following = min(i for i in binding["observed_indices"] if i > index)
            images = _trajectory_plot(
                profile, teacher, source, target, left=left,
                next_time=float(times[following]), steps=steps, run_dir=run_dir,
                source_root=config["model"]["params"]["source_root"],
                device=device, seed=binding["args"]["seed_current"],
            )
        if binding["profile"] == "sphere":
            results[str(index)]["mean_distance_from_sphere"] = float(
                (torch.linalg.vector_norm(prediction, dim=1) - 1).abs().mean()
            )
    if binding["profile"] == "lidar":
        images = _lidar_plot(
            teacher, data, times=times, steps=steps, run_dir=run_dir,
            source_root=config["model"]["params"]["source_root"],
            device=device,
        )
    report = {"format": "mfm_snapshot_test_emd_v2", "profile": binding["profile"],
              "snapshot_identity": binding["snapshot_identity"],
              "flow_sha256": pair["files"]["flow"]["sha256"],
              "metric": "uniform_mass_euclidean_w1", "solver": "euler",
              "steps_per_interval": steps, "rollout_batch_size": batch_size,
              "sampling": "full observed marginal and full held-out marginal; no subsampling",
              "coordinate_space": "prepared normalized features; raw suffix reverses saved mean/scale",
              "holdouts": results,
              "images": images,
              "mean_distance_from_sphere": float(np.mean([item["mean_distance_from_sphere"] for item in results.values()])) if binding["profile"] == "sphere" and results else None,
              "test_EMD": float(np.mean([item["test_EMD"] for item in results.values()])) if results else None,
              "test_EMD_raw": float(np.mean([item["test_EMD_raw"] for item in results.values()])) if results else None}
    atomic_json(destination, report)
    atomic_json(run_dir / "eval/status.json", {"status": "complete"})
    if (run_dir / "eval/wandb_status.json").exists():
        atomic_json(run_dir / "eval/wandb_status.json", {
            "status": "stale", "reason": "Local EMD was recomputed; publish to W&B again",
        })
    return report


def publish_wandb(run_dir, report):
    """Append local diagnostics to the W&B run created by snapshot training."""
    run_dir = Path(run_dir).expanduser().resolve()
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    tracking = config.get("wandb", {})
    mode = tracking.get("mode", "disabled")
    if mode == "disabled":
        return
    sessions = [p for p in (run_dir / "wandb").glob("run-*") if p.is_dir()]
    if len(sessions) != 1:
        raise ValueError("Expected exactly one W&B session for this MFM teacher")
    run_id = sessions[0].name.rsplit("-", 1)[-1]
    import wandb

    run = wandb.init(
        project=tracking.get("project", "flow2pc"), group=tracking.get("group"),
        id=run_id, resume="allow", mode=mode, dir=str(run_dir),
    )
    try:
        metrics = {"test_EMD": report["test_EMD"],
                   "test_EMD_raw": report["test_EMD_raw"]}
        if report["mean_distance_from_sphere"] is not None:
            metrics["mean_distance_from_sphere"] = report["mean_distance_from_sphere"]
        for index, entry in report["holdouts"].items():
            metrics[f"test_EMD/holdout_{index}"] = entry["test_EMD"]
        for number, filename in enumerate(report.get("images", []), start=1):
            key = ("arch_interpolant" if filename == "arch_trajs.png" else
                   "lidar_interpolant" if filename == "lidar_interpolant.png" else
                   f"sphere_interpolant_view_{number}")
            metrics[key] = wandb.Image(str(run_dir / filename))
        run.log({key: value for key, value in metrics.items() if value is not None})
    finally:
        run.finish()
    atomic_json(run_dir / "eval/wandb_status.json", {"status": "complete", "run_id": run_id})

"""Held-out snapshot Wasserstein metrics using the selected teacher pair."""

import json
from pathlib import Path
from types import SimpleNamespace

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


def _knot_plot(teacher, data, *, times, steps, run_dir, device):
    """Plot the observed marginals and trajectories rolled out from time zero."""
    import matplotlib.pyplot as plt

    state = data.marginals[0].to(device)
    trajectory = [state.cpu()]
    with torch.no_grad():
        for left, right in zip(times[:-1], times[1:], strict=True):
            left, right = float(left), float(right)
            dt = (right - left) / steps
            for step in range(steps):
                state = state + dt * teacher.velocity(state, left + step * dt)
                if not torch.isfinite(state).all():
                    raise FloatingPointError("Nonfinite MFM knot plot rollout")
            trajectory.append(state.cpu())

    mean = torch.as_tensor(data.normalization["mean"], dtype=torch.float64)
    scale = torch.as_tensor(data.normalization["scale"], dtype=torch.float64)
    paths = (torch.stack(trajectory).double() * scale + mean).numpy()
    observed = (torch.cat(data.marginals).double() * scale + mean).numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        ax.scatter(observed[:, 0], observed[:, 1], s=8, alpha=0.28,
                   color="tab:blue", label="Observed marginals")
        for index in range(paths.shape[1]):
            ax.plot(paths[:, index, 0], paths[:, index, 1], color="tab:orange",
                    alpha=0.65, linewidth=1,
                    label="Teacher rollout" if index == 0 else None)
        ax.scatter(paths[0, :, 0], paths[0, :, 1], color="tab:green", s=24,
                   label="Initial points", zorder=3)
        ax.set(xlabel="x", ylabel="y", title="Knot: observed marginals and teacher rollout")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "knot_trajectory.png", dpi=200)
    finally:
        plt.close(fig)
    return ["knot_trajectory.png"]


def _cell_tracking_plot(teacher, data, *, times, steps, batch_size, run_dir, device):
    """Render the same microscopy overlays and frames as ALI-CFM evaluation."""
    import matplotlib.pyplot as plt
    from PIL import Image

    from flow2pc.evaluation.figures import TRACKING_FRAMES

    if len(times) <= max(TRACKING_FRAMES):
        raise ValueError("Cell tracking requires all microscopy frames")
    mask_paths = [Path(item["path"]) for item in data.source_info["source_files"]
                  if Path(item["path"]).name == "mask_cell4_stack.npy"]
    if len(mask_paths) != 1:
        raise ValueError("Cell tracking manifest must identify the source mask stack")
    image_dir = mask_paths[0].parent.parent / "PhC-C2DH-U373" / "01"
    image_paths = [image_dir / f"t{index:03d}.tif" for index in TRACKING_FRAMES]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cell tracking microscopy images missing: {missing}")

    predictions = [[] for _ in TRACKING_FRAMES]
    with torch.no_grad():
        for chunk in data.marginals[0].split(batch_size):
            state = chunk.to(device)
            predictions[0].append(state.cpu())
            for pos, (start, end) in enumerate(zip(TRACKING_FRAMES[:-1], TRACKING_FRAMES[1:], strict=True), start=1):
                left, right = float(times[start]), float(times[end])
                dt = (right - left) / steps
                for step in range(steps):
                    state = state + dt * teacher.velocity(state, left + step * dt)
                    if not torch.isfinite(state).all():
                        raise FloatingPointError("Nonfinite MFM cell tracking rollout")
                predictions[pos].append(state.cpu())

    mean = torch.as_tensor(data.normalization["mean"], dtype=torch.float64)
    scale = torch.as_tensor(data.normalization["scale"], dtype=torch.float64)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
    try:
        for axis, index, chunks, image_path in zip(axes.flat, TRACKING_FRAMES, predictions, image_paths, strict=True):
            with Image.open(image_path) as handle:
                background = np.asarray(handle).copy()
            real = (data.marginals[index].double() * scale + mean).numpy()
            prediction = (torch.cat(chunks).double() * scale + mean).numpy()
            axis.imshow(background, cmap="gray", origin="upper")
            axis.scatter(real[:, 0], real[:, 1], s=1, alpha=0.12, color="cyan")
            axis.scatter(prediction[:, 0], prediction[:, 1], s=2, alpha=0.35, color="red")
            axis.set(xlim=(0, background.shape[1]), ylim=(background.shape[0], 0),
                     title=f"teacher | frame {index + 1}")
            axis.axis("off")
        fig.suptitle("Cell tracking | cyan: observed mask, red: velocity-driven ODE")
        fig.savefig(run_dir / "teacher_overlay.png", dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
    return ["teacher_overlay.png"]


def _transport_plots(profile, teacher, data, *, times, steps, batch_size, run_dir, device):
    """Plot MFM transport from the earliest population."""
    if profile == "gom":
        from flow2pc.evaluation.gom_trajectories import plot_gom_trajectories

        plot_gom_trajectories(teacher, data, times, run_dir, device, steps)
        return ["gom_ground_truth_simulated_trajectories.png"]

    from flow2pc.evaluation.figures import save_figures

    if profile == "beijing":
        from flow2pc.evaluation.ode import rollout

        class PlotTeacher(torch.nn.Module):
            def __init__(self, loaded):
                super().__init__()
                self.model = loaded.model

            def ode_velocity(self, time, values):
                return self.model(time, values)

        # Match the ALI-CFM Beijing figure protocol, independently of test EMD.
        settings = {"batch_size": 128, "max_nfe": 50000, "solver": "dopri5",
                    "tolerance": 1e-5, "ode_steps": 100}
        path, _ = rollout(PlotTeacher(teacher), data.marginals[0],
                          torch.tensor(times, dtype=torch.float32), settings)
    else:
        raise ValueError(f"Unsupported transport plot profile: {profile}")

    mean = torch.as_tensor(data.normalization["mean"], dtype=torch.float64)
    scale = torch.as_tensor(data.normalization["scale"], dtype=torch.float64)
    # Snapshot tensors use (x - mean) / scale; the shared plotting interface
    # expects a scale and shift for conversion back to physical coordinates.
    dataset = SimpleNamespace(
        metadata=SimpleNamespace(name=profile),
        clouds=data.marginals, labels=data.source_info["labels"],
        heldout_indices=data.heldout_indices,
        preprocessing={"scale": scale.tolist(), "shift": mean.tolist()},
    )
    files = save_figures(dataset, path, None, run_dir, {})
    return [file.name for file in files.values()]


def evaluate_run(run_dir, *, device="cpu", batch_size=512, steps=100, force=False, render_plots=True):
    """Evaluate each temporal holdout with upstream-style Euler and exact OT.

    Both empirical populations are complete. Only the velocity rollout is
    batched; transport is solved on the full pairwise cost matrix.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    destination = run_dir / "eval" / "test_emd.json"
    if destination.exists() and not force:
        cached = json.loads(destination.read_text())
        knot_image_missing = (cached.get("profile") == "knot" and
                              ("knot_trajectory.png" not in cached.get("images", []) or
                               not (run_dir / "knot_trajectory.png").is_file()))
        beijing_image_missing = (cached.get("profile") == "beijing" and (
            cached.get("plot_protocol", {}).get("solver") != "dopri5" or any(
                name not in cached.get("images", []) or not (run_dir / name).is_file()
                for name in ("teacher_ode_marginals.png", "teacher_quantiles.png"))))
        tracking_image_missing = (cached.get("profile") == "cell_tracking" and
                                  ("teacher_overlay.png" not in cached.get("images", []) or
                                   not (run_dir / "teacher_overlay.png").is_file()))
        gom_image_missing = (cached.get("profile") == "gom" and
                             ("gom_ground_truth_simulated_trajectories.png" not in cached.get("images", []) or
                              not (run_dir / "gom_ground_truth_simulated_trajectories.png").is_file()))
        if (cached.get("format") == "mfm_snapshot_test_emd_v2" and
                cached.get("metric") == ("uniform_mass_euclidean_w2" if cached.get("profile") == "beijing" else "uniform_mass_euclidean_w1") and
                cached.get("steps_per_interval") == steps and
                cached.get("rollout_batch_size") == batch_size and
                not knot_image_missing and not beijing_image_missing and
                not tracking_image_missing and not gom_image_missing):
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
    order = 2 if binding["profile"] == "beijing" else 1
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
        value = emd(target, prediction, order=order)
        baseline = emd(target, source, order=order)
        mean = torch.as_tensor(data.normalization["mean"], dtype=torch.float64)
        scale = torch.as_tensor(data.normalization["scale"], dtype=torch.float64)
        raw_target = target.double() * scale + mean
        raw_source = source.double() * scale + mean
        raw_prediction = prediction.double() * scale + mean
        raw_value = emd(raw_target, raw_prediction, order=order)
        raw_baseline = emd(raw_target, raw_source, order=order)
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
    if binding["profile"] == "knot":
        images = _knot_plot(teacher, data, times=times, steps=steps,
                            run_dir=run_dir, device=device)
    if binding["profile"] == "cell_tracking":
        images = _cell_tracking_plot(teacher, data, times=times, steps=steps,
                                     batch_size=batch_size, run_dir=run_dir, device=device)
    if binding["profile"] == "beijing" and not render_plots:
        images = [name for name in ("teacher_ode_marginals.png", "teacher_quantiles.png")
                  if (run_dir / name).is_file()]
    elif binding["profile"] in ("beijing", "gom"):
        images = _transport_plots(binding["profile"], teacher, data, times=times, steps=steps,
                                  batch_size=batch_size, run_dir=run_dir, device=device)
    report = {"format": "mfm_snapshot_test_emd_v2", "profile": binding["profile"],
              "snapshot_identity": binding["snapshot_identity"],
              "flow_sha256": pair["files"]["flow"]["sha256"],
              "metric": f"uniform_mass_euclidean_w{order}", "solver": "euler",
              "steps_per_interval": steps, "rollout_batch_size": batch_size,
              "sampling": "full observed marginal and full held-out marginal; no subsampling",
              "coordinate_space": "prepared normalized features; raw suffix reverses saved mean/scale",
              "holdouts": results,
              "images": images,
              "mean_distance_from_sphere": float(np.mean([item["mean_distance_from_sphere"] for item in results.values()])) if binding["profile"] == "sphere" and results else None,
              "test_EMD": float(np.mean([item["test_EMD"] for item in results.values()])) if results else None,
              "test_EMD_raw": float(np.mean([item["test_EMD_raw"] for item in results.values()])) if results else None}
    if binding["profile"] == "beijing":
        report["plot_protocol"] = {"solver": "dopri5", "tolerance": 1e-5,
                                   "batch_size": 128, "max_nfe": 50000,
                                   "source": "full earliest marginal", "metric_solver_unchanged": "euler"}
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
                   "knot_trajectory" if filename == "knot_trajectory.png" else
                   "gom_trajectories" if filename == "gom_ground_truth_simulated_trajectories.png" else
                   "teacher_marginals" if filename == "teacher_ode_marginals.png" else
                   "teacher_quantiles" if filename == "teacher_quantiles.png" else
                   "teacher_overlay" if filename == "teacher_overlay.png" else
                   f"sphere_interpolant_view_{number}")
            metrics[key] = wandb.Image(str(run_dir / filename))
        run.log({key: value for key, value in metrics.items() if value is not None})
    finally:
        run.finish()
    atomic_json(run_dir / "eval/wandb_status.json", {"status": "complete", "run_id": run_id})

"""GoM observations and MFM trajectories in one figure."""

def plot_gom_trajectories(teacher, data, times, run_dir, device, steps):
    """Overlay unpaired observations and simulated paths in original coordinates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    state = data.marginals[0].to(device)
    path = [state.cpu()]
    with torch.no_grad():
        for left, right in zip(times[:-1], times[1:], strict=True):
            dt = (right - left) / steps
            for step in range(steps):
                state = state + dt * teacher.velocity(state, left + step * dt)
                if not torch.isfinite(state).all():
                    raise FloatingPointError("Nonfinite GoM trajectory")
                if (step + 1) % max(1, steps // 20) == 0 or step == steps - 1:
                    path.append(state.cpu())
    mean = torch.as_tensor(data.normalization["mean"], dtype=torch.float64)
    scale = torch.as_tensor(data.normalization["scale"], dtype=torch.float64)
    trajectories = (torch.stack(path).double() * scale + mean).numpy()
    observations = (torch.cat(data.marginals).double() * scale + mean).numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(observations[:, 0], observations[:, 1], s=9, color="tab:blue",
               alpha=0.45, label="Ground truth (observations)", zorder=2)
    for index in range(trajectories.shape[1]):
        ax.plot(trajectories[:, index, 0], trajectories[:, index, 1],
                color="tab:orange", linewidth=0.7, alpha=0.45,
                label="Simulated trajectories (MFM)" if index == 0 else None)
    ax.set(title="GoM: ground truth and simulated trajectories", xlabel="Coordinate 1", ylabel="Coordinate 2")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(run_dir / "gom_ground_truth_simulated_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close(fig)



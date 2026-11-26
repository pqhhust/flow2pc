"""E3.1 figures saved locally before optional W&B upload."""

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"teacher": "#2563eb", "student": "#e87924"}


def save_summary(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    maximum = max(row["bank_size"] for row in rows)
    eligible = [row for row in rows if row["coverage"] is not None]
    figures = {}
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    for role, color in COLORS.items():
        groups = defaultdict(list)
        for row in eligible:
            if row["role"] == role and row["bank_size"] == maximum:
                groups[row["nominal_alpha"]].append(row["coverage"])
        if groups:
            x = sorted(groups)
            ax.plot(x, [np.mean(groups[a]) for a in x], "o-", color=color, label=role)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Nominal probability", ylabel="Real-observation coverage",
           title="E3.1 — mean across selected times and MC seeds")
    if eligible:
        ax.legend()
    figures["calibration"] = str(output / "calibration.png")
    fig.savefig(figures["calibration"], dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for role, color in COLORS.items():
        groups = defaultdict(list)
        scores = defaultdict(list)
        for row in rows:
            if row["role"] == role and row["nominal_alpha"] == 0.9:
                if row["hdr_ece"] is not None:
                    groups[row["bank_size"]].append(row["hdr_ece"])
                if row["bank_size"] == maximum:
                    scores[row["physical_time"]].append(row["energy_score"])
        if groups:
            x = sorted(groups)
            axes[0].plot(x, [np.mean(groups[n]) for n in x], "o-", color=color, label=role)
        if scores:
            x = sorted(scores)
            axes[1].plot(x, [np.mean(scores[t]) for t in x], "o-", color=color, label=role)
    axes[0].set(xlabel="Fit / threshold bank size", ylabel="HDR-ECE", title="Bank-budget sensitivity")
    axes[1].set(xlabel="Physical time (saved dataset units)", ylabel="Full-state Energy Score",
                title="Distribution quality")
    for ax in axes:
        if ax.lines:
            ax.legend()
    figures["convergence_and_energy"] = str(output / "convergence_and_energy.png")
    fig.savefig(figures["convergence_and_energy"], dpi=170, bbox_inches="tight")
    plt.close(fig)

    times = sorted({r["physical_time"] for r in eligible})
    if times:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for role, color in COLORS.items():
            for key, ax in (("coverage", axes[0]), ("hdr_ece", axes[1])):
                groups = defaultdict(list)
                for row in eligible:
                    if row["role"] == role and row["bank_size"] == maximum and row["nominal_alpha"] == 0.9:
                        groups[row["physical_time"]].append(row[key])
                x = sorted(groups)
                means = np.array([np.mean(groups[t]) for t in x])
                ax.plot(x, means, "o-", color=color, label=role)
                ax.fill_between(x, [min(groups[t]) for t in x], [max(groups[t]) for t in x], color=color, alpha=.15)
        axes[0].axhline(.9, color="black", ls="--", lw=1)
        axes[0].set(ylabel="90% region coverage", ylim=(0, 1), title="Shading: MC-seed range, not a test-data CI")
        axes[1].set(ylabel="HDR-ECE", title="Calibration error by time")
        for ax in axes:
            ax.set_xlabel("Physical time (saved dataset units)")
            ax.legend()
        figures["by_time"] = str(output / "by_time.png")
        fig.savefig(figures["by_time"], dpi=170, bbox_inches="tight")
        plt.close(fig)
    return figures


def save_regions(regions, output):
    """Independent real observations overlaid with estimated 90% model regions."""
    figures = {}
    for index, entries in regions.items():
        dimension = entries[0]["real"].shape[1]
        if dimension not in (1, 2):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
        for ax, entry in zip(axes, entries):
            real, model, estimator = entry["real"], entry["model"], entry["estimator"]
            combined = np.concatenate((real, model))
            low, high = np.quantile(combined, [.005, .995], axis=0)
            margin = np.maximum((high - low) * .15, .1)
            if dimension == 1:
                grid = np.linspace(low[0] - margin[0], high[0] + margin[0], 300)
                score = estimator.score(grid[:, None])
                density = np.exp(score)
                ax.hist(real[:, 0], bins=35, density=True, alpha=.3, color="gray", label="real observations")
                ax.plot(grid, density, color=COLORS[entry["role"]],
                        label="direct PC density" if entry["role"] == "student" else "teacher KDE")
                ax.fill_between(grid, 0, density, where=score >= entry["threshold"], alpha=.3,
                                color=COLORS[entry["role"]], label="estimated 90% region")
            else:
                # Resolve the model's region locally, including narrow clouds;
                # real/model scatter determines the shared displayed axes.
                model_low, model_high = np.quantile(model, [.001, .999], axis=0)
                model_margin = np.maximum((model_high - model_low) * .2, 4 * estimator.bandwidth)
                xs = np.linspace(model_low[0] - model_margin[0], model_high[0] + model_margin[0], 200)
                ys = np.linspace(model_low[1] - model_margin[1], model_high[1] + model_margin[1], 200)
                xx, yy = np.meshgrid(xs, ys)
                score = estimator.score(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
                ax.scatter(real[:, 0], real[:, 1], s=5, color="gray", alpha=.4, label="real observations")
                if score.min() <= entry["threshold"] <= score.max():
                    ax.contour(xx, yy, score, levels=[entry["threshold"]], colors=[COLORS[entry["role"]]])
                else:
                    ax.text(.02, .02, "Contour unresolved on plotting grid", transform=ax.transAxes, fontsize=7)
                ax.scatter(model[:500, 0], model[:500, 1], s=3, color=COLORS[entry["role"]], alpha=.2,
                           label="direct PC draws" if entry["role"] == "student" else "teacher ODE draws")
                ax.set_ylabel("Scoring coordinate 2")
            ax.set_xlabel("Scoring coordinate 1")
            ax.set_title(f"{entry['role']} — physical time {entry['time']:g}")
            ax.legend(fontsize=8)
        fig.suptitle("Teacher KDE and direct PC density regions; thresholds estimated from model draws")
        path = output / f"regions_time_{index}.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        figures[f"regions_time_{index}"] = str(path)
    return figures


def save_fidelity(summary, output):
    if not summary:
        return {}
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    levels = [row["nominal_alpha"] for row in summary]
    for ax, region in zip(axes, ("teacher", "student"), strict=True):
        for role in ("teacher", "student"):
            key = f"{region}_region_{role}_mass"
            ax.errorbar(levels, [row[key] for row in summary],
                        yerr=[row[key + "_mc_std"] for row in summary],
                        marker="o", color=COLORS[role], label=f"{role} draws")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Nominal region probability",
               title=f"Mass inside {region}-defined regions")
        ax.legend()
    axes[0].set_ylabel("Independent model-draw mass")
    fig.suptitle("Uncertainty fidelity — error bars are MC standard deviations, not test-data CIs")
    path = output / "fidelity.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {"fidelity": str(path)}


def save_knot_geometry(paths, references, times, dataset, fit_indices, output):
    """Raw-coordinate knot geometry alongside marginal uncertainty diagnostics."""
    from flow2pc.data.synthetic.knot import knot_means

    means, _ = knot_means(dataset.params["size"])
    chosen = np.linspace(0, len(times) - 1, min(12, len(times)), dtype=int)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    axes[0].plot(means[:, 0], means[:, 1], color="black", lw=1, label="true knot mean")
    for position in chosen:
        values = references[times[position]][:150]
        axes[0].scatter(values[:, 0], values[:, 1], s=3, alpha=.2, color="gray")
    axes[0].set_title("Independent analytic reference observations")
    for ax, role in zip(axes[1:], ("teacher", "student")):
        values = paths[role][:, fit_indices]
        centers = values.mean(axis=1)
        ax.plot(means[:, 0], means[:, 1], color="black", lw=1, label="true knot mean")
        ax.plot(centers[:, 0], centers[:, 1], color=COLORS[role], lw=1.5,
                label="direct PC marginal mean" if role == "student" else "teacher ODE mean")
        for position in chosen:
            cloud = values[position, :150]
            ax.scatter(cloud[:, 0], cloud[:, 1], s=3, alpha=.15, color=COLORS[role])
        ax.set_title(role)
    for ax in axes:
        ax.set_xlabel("Raw x")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Raw y")
    fig.suptitle("Lifted knot: geometry and generated marginal spread")
    path = output / "knot_geometry.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"knot_geometry": str(path)}

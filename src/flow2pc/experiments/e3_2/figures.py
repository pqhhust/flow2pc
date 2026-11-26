"""Conditional calibration, estimator sensitivity and ST missing-coordinate views."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"teacher": "#2563eb", "student": "#e87924"}


def save(rows, observations, settings, output):
    output.mkdir(parents=True, exist_ok=True)
    figures = {}
    maximum, draws = max(settings["bank_sizes"]), max(settings["conditional_draw_sizes"])
    primary = [r for r in rows if r["bank_size"] == maximum and r["conditional_draws"] == draws
               and r["bandwidth"] == settings["primary_bandwidth"]]
    masks = sorted({r["mask_id"] for r in rows})
    fig, axes = plt.subplots(1, len(masks), figsize=(5 * len(masks), 4), squeeze=False)
    for ax, mask in zip(axes[0], masks, strict=True):
        for role, color in COLORS.items():
            selected = [r for r in primary if r["role"] == role and r["mask_id"] == mask]
            levels = sorted({r["nominal_alpha"] for r in selected})
            ax.plot(levels, [np.mean([r["coverage"] for r in selected if r["nominal_alpha"] == level])
                             for level in levels], "o-", color=color, label=role)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Nominal conditional HDR mass",
               ylabel="Real hidden-value coverage", title=f"Observed coordinates: {mask}")
        ax.legend()
    fig.suptitle("E3.2 — aggregate over conditions, not a pointwise calibration guarantee")
    figures["calibration"] = str(output / "calibration.png")
    fig.savefig(figures["calibration"], dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for role, color in COLORS.items():
        for bandwidth in settings["condition_bandwidths"]:
            chosen = [r for r in rows if r["role"] == role and r["bandwidth"] == bandwidth
                      and r["nominal_alpha"] == .9 and r["conditional_draws"] == draws]
            sizes = sorted({r["bank_size"] for r in chosen})
            if role == "student":
                if chosen:
                    axes[0].axhline(np.mean([r["hdr_ece"] for r in chosen]), color=color,
                                    label="student direct PC (no KDE bandwidth)")
                continue
            axes[0].plot(sizes, [np.mean([r["hdr_ece"] for r in chosen if r["bank_size"] == n]) for n in sizes],
                         "o-", color=color, alpha=.9 if bandwidth == settings["primary_bandwidth"] else .4,
                         label=f"{role}, h={bandwidth:g}")
        chosen = [r for r in primary if r["role"] == role and r["nominal_alpha"] == .9]
        axes[1].plot(range(len(masks)), [np.mean([r["energy_score"] for r in chosen if r["mask_id"] == mask])
                                       for mask in masks], "o-", color=color, label=role)
        if role == "teacher":
            axes[2].plot(range(len(masks)), [np.mean([r["low_ess_fraction"] for r in chosen if r["mask_id"] == mask])
                                           for mask in masks], "o-", color=color, label=role)
    axes[0].set(xlabel="Joint KDE fit bank size (bandwidth held fixed)", ylabel="Conditional HDR-ECE")
    axes[1].set(ylabel="Conditional Energy Score", xlabel="Observed mask")
    axes[2].set(ylabel="Fraction below effective-component threshold", xlabel="Observed mask", ylim=(0, 1))
    for ax in axes[1:]:
        ax.set_xticks(range(len(masks)), masks)
    for ax in axes:
        ax.legend(fontsize=7)
    figures["sensitivity_and_energy"] = str(output / "sensitivity_and_energy.png")
    fig.savefig(figures["sensitivity_and_energy"], dpi=170, bbox_inches="tight")
    plt.close(fig)

    # First MC seed only; these bands are central intervals, not joint HDRs.
    if observations and all(r["observed_dim"] == r["hidden_dim"] == 1 for r in observations):
        fig, axes = plt.subplots(1, len(masks), figsize=(6 * len(masks), 4), squeeze=False)
        for ax, mask in zip(axes[0], masks, strict=True):
            for role, color in COLORS.items():
                selected = sorted([r for r in observations if r["role"] == role and r["mask_id"] == mask
                                   and r["seed"] == settings["seeds"][0] and r["nominal_alpha"] == .9],
                                  key=lambda r: r["observed_value"])
                x = [r["observed_value"] for r in selected]
                if role == "teacher":
                    ax.scatter(x, [r["hidden_value"] for r in selected], color="gray", s=8, alpha=.45,
                               label="Real held-out hidden coordinate")
                ax.plot(x, [r["conditional_mean"] for r in selected], color=color, lw=1, label=role)
                ax.fill_between(x, [r["lower90"] for r in selected], [r["upper90"] for r in selected],
                                color=color, alpha=.15)
            ax.set(xlabel="Observed coordinate (training-standardized)", ylabel="Hidden coordinate (training-standardized)",
                   title=f"Mask {mask}: mean and central 90% interval")
            ax.legend(fontsize=7)
        figures["missing_coordinates"] = str(output / "missing_coordinates.png")
        fig.savefig(figures["missing_coordinates"], dpi=170, bbox_inches="tight")
        plt.close(fig)
    return figures

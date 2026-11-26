"""Teacher-relative E4 curves; no density/ground-truth calibration claims."""

import numpy as np


def save(e41, e42, coverage, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    media = {}

    def finish(fig, name):
        fig.tight_layout()
        path = output / (name+".png")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        media[name] = str(path)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
    for row, kind in enumerate(("mean", "spread")):
        for ax, metric in zip(axes[row], ("shifted_mmd2", "mean_shift_error", "variance_shift_error")):
            selected = [r for r in e41 if r["shift_type"] == kind or r["shift_type"] == "clean"]
            selected.sort(key=lambda r: 1. if kind == "spread" and r["shift_type"] == "clean" else r["severity"])
            x = [1. if kind == "spread" and r["shift_type"] == "clean" else r["severity"] for r in selected]
            ax.errorbar(x, [r[metric] for r in selected],
                        yerr=[r[metric+"_mc_std"] or 0. for r in selected], marker="o", label="student–teacher")
            if metric == "shifted_mmd2":
                ax.plot(x, [r["teacher_teacher_mmd2_floor"] for r in selected], "--", label="independent teacher floor")
            ax.set(xlabel=f"{kind} severity (training-standardized)", ylabel=metric)
            ax.legend(fontsize=7)
    fig.suptitle("E4.1 teacher-relative shift propagation; error bars: MC SD")
    finish(fig, "e41_metric_vs_severity")

    fig, ax = plt.subplots(figsize=(8, 4))
    for condition in sorted({r["condition"] for r in coverage}):
        selected = [r for r in coverage if r["condition"] == condition and r["role"] == "student"]
        levels = sorted({r["nominal_alpha"] for r in selected})
        ax.plot(levels, [np.mean([r["coverage_sd"] for r in selected if r["nominal_alpha"] == a
                                 and r["time_index"] != 0]) for a in levels], marker="o", label=condition)
    ax.plot([.5, .95], [.5, .95], "k--", label="nominal")
    ax.set(xlabel="Nominal interval coverage", ylabel="Coverage against independent teacher draws",
           title="E4.2 teacher-relative marginal interval coverage (not HDR)")
    ax.legend(fontsize=7)
    finish(fig, "e42_coverage_curves")

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
    for row, kind in enumerate(("mean", "spread")):
        for ax, metric in zip(axes[row], ("sd_ece", "energy_score", "crps")):
            for role in ("teacher", "student"):
                selected = [r for r in e42 if r["role"] == role and r["shift_type"] in (kind, "clean")]
                def position(record):
                    return (1. if kind == "spread" and record["shift_type"] == "clean"
                            else record["severity"])
                selected.sort(key=position)
                ax.errorbar([position(r) for r in selected], [r[metric] for r in selected],
                            yerr=[r[metric+"_mc_std"] or 0. for r in selected], marker="o", label=role)
            ax.set(xlabel=f"{kind} severity", ylabel=metric)
            ax.legend()
    fig.suptitle("E4.2 teacher-relative scores; clean included; error bars: MC SD")
    finish(fig, "e42_metrics_vs_severity")
    return media


def time_curves(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for condition in sorted({r["condition"] for r in rows}):
        selected = [r for r in rows if r["condition"] == condition]
        times = sorted({r["physical_time"] for r in selected})
        ax.plot(times, [np.mean([r["trajectory_rmse"] for r in selected if r["physical_time"] == time])
                        for time in times], marker="o", label=condition)
    ax.set(xlabel="Physical observation time", ylabel="Paired RMSE (training-standardized)",
           title="E4.1 identical initial particles; time zero excluded from summary")
    ax.legend(fontsize=7)
    path = output / "e41_trajectory_rmse_vs_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"e41_trajectory_rmse_vs_time": str(path)}

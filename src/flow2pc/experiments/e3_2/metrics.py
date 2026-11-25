"""Conditional HDR coverage and proper scores; no SD-ECE or recalibration."""

import numpy as np

from flow2pc.experiments.e3_1.metrics import finite_matrix


def conditional_scores(target, score_a, score_b, threshold_log_density,
                       real_log_density, levels, *, validation_log_density=None):
    """Score ONE condition; aggregate coverage across real conditions later.

score_a, score_b and threshold_log_density must come from independent RNG
streams of the same conditional predictive law. Targets never tune that law.
Paired independent draws give an O(M), unbiased Monte Carlo Energy Score.
"""
    score_a, score_b = map(finite_matrix, (score_a, score_b))
    target = np.asarray(target, dtype=np.float64)
    threshold_log_density = np.asarray(threshold_log_density, dtype=np.float64)
    if (target.shape != (score_a.shape[1],) or score_a.shape != score_b.shape
            or threshold_log_density.ndim != 1 or len(threshold_log_density) < 2
            or not np.isfinite(target).all() or not np.isfinite(threshold_log_density).all()
            or not np.isscalar(real_log_density) or not np.isfinite(real_log_density)
            or not levels or any(not 0 < level < 1 for level in levels)):
        raise ValueError("Invalid conditional score arrays, density or probability levels")
    energy = float((.5 * (np.linalg.norm(score_a - target, axis=1)
                         + np.linalg.norm(score_b - target, axis=1))
                    - .5 * np.linalg.norm(score_a - score_b, axis=1)).mean())
    crps = float((.5 * (np.abs(score_a - target) + np.abs(score_b - target))
                  - .5 * np.abs(score_a - score_b)).mean())
    shared = {"energy_score": energy, "conditional_crps": crps,
              "estimated_conditional_nll_per_dim": float(-real_log_density / len(target))}
    if validation_log_density is not None:
        validation_log_density = np.asarray(validation_log_density, dtype=np.float64)
        if validation_log_density.ndim != 1 or not len(validation_log_density) or not np.isfinite(validation_log_density).all():
            raise ValueError("Invalid independent conditional model density bank")
    rows = []
    for level in levels:
        threshold = float(np.quantile(threshold_log_density, 1 - level, method="lower"))
        lower, upper = np.quantile(score_a, [(1 - level) / 2, (1 + level) / 2], axis=0)
        interval_score = upper - lower + 2 / (1 - level) * (
            np.maximum(lower - target, 0) + np.maximum(target - upper, 0))
        rows.append({"nominal_alpha": level, **shared,
                     "hdr_covered": float(real_log_density >= threshold),
                     "independent_model_mass": (float(np.mean(validation_log_density >= threshold))
                                                if validation_log_density is not None else None),
                     "threshold_bank_mass": float(np.mean(threshold_log_density >= threshold)),
                     "threshold_tie_fraction": float(np.mean(threshold_log_density == threshold)),
                     "interval_coverage": float(np.mean((target >= lower) & (target <= upper))),
                     "interval_width": float(np.mean(upper - lower)),
                     "mean_interval_score": float(interval_score.mean())})
    return rows


def aggregate(rows):
    """Never average per-observation |indicator-alpha|: that is not ECE."""
    if not rows:
        raise ValueError("No conditional observations to aggregate")
    result = []
    for level in sorted({r["nominal_alpha"] for r in rows}):
        selected = [r for r in rows if r["nominal_alpha"] == level]
        record = {"nominal_alpha": level, "n_conditions": len(selected)}
        for key in ("hdr_covered", "interval_coverage", "interval_width", "mean_interval_score",
                    "energy_score", "conditional_crps", "estimated_conditional_nll_per_dim",
                    "threshold_bank_mass", "threshold_tie_fraction", "independent_model_mass"):
            values = [r[key] for r in selected if r[key] is not None]
            record["coverage" if key == "hdr_covered" else key] = float(np.mean(values)) if values else None
        result.append(record)
    hdr_ece = float(np.mean([abs(row["coverage"] - row["nominal_alpha"]) for row in result]))
    interval_ece = float(np.mean([abs(row["interval_coverage"] - row["nominal_alpha"]) for row in result]))
    for row in result:
        row.update(hdr_ece=hdr_ece, interval_coverage_ece=interval_ece)
    return result

"""Shift-response metrics in fixed training-standardized full coordinates."""

import json

import numpy as np
from scipy.spatial.distance import cdist, pdist

from flow2pc.experiments.e3_1.metrics import energy_score, finite_matrix

HDR_REASON = "Finite empirical ODE law; no validated normalized density/HDR estimator. No PC-density or KDE substitution."


def kernel_widths(training, settings):
    training = finite_matrix(training)
    rng = np.random.default_rng(8137)
    fit = training[rng.permutation(len(training))[:settings["kernel_samples"]]]
    distances = pdist(fit)
    distances = distances[distances > 0]
    if not len(distances):
        raise ValueError("Clean training coordinates have no positive pair distances for MMD bandwidth")
    base = float(np.median(distances))
    return [base * m for m in settings["kernel_multipliers"]]


def mmd2(a, b, widths, chunk=256):
    """Biased empirical MMD², averaged over predeclared Gaussian kernels."""
    a, b = map(finite_matrix, (a, b))
    if a.shape[1] != b.shape[1] or not widths or any(h <= 0 or not np.isfinite(h) for h in widths):
        raise ValueError("Invalid MMD dimensions/bandwidths")

    # Exact duplicate compression preserves the empirical estimator, including
    # diagonal mass. Larger bootstrap budgets do not require repeated kernels.
    a, ca = np.unique(a, axis=0, return_counts=True)
    b, cb = np.unique(b, axis=0, return_counts=True)
    wa, wb = ca/ca.sum(), cb/cb.sum()

    def average(x, y, wx, wy):
        total = 0.
        for start in range(0, len(x), chunk):
            squared = cdist(x[start:start+chunk], y, "sqeuclidean")
            weights = wx[start:start+chunk, None] * wy[None, :]
            total += sum((np.exp(-squared / (2*h*h))*weights).sum()
                         for h in widths) / len(widths)
        return total

    value = average(a, a, wa, wa) + average(b, b, wb, wb) - 2*average(a, b, wa, wb)
    if value < -1e-8:
        raise FloatingPointError("Materially negative biased MMD²")
    return float(max(value, 0.))


def response(teacher, student, clean_teacher, clean_student):
    t, s, ct, cs = map(finite_matrix, (teacher, student, clean_teacher, clean_student))
    if len({x.shape for x in (t, s, ct, cs)}) != 1:
        raise ValueError("Response metrics require paired initial draws and equal shapes")
    mean_difference = (s.mean(0)-cs.mean(0)) - (t.mean(0)-ct.mean(0))
    variance_difference = (s.var(0)-cs.var(0)) - (t.var(0)-ct.var(0))
    return {"mean_shift_error": float(np.sqrt(np.mean(mean_difference**2))),
            "variance_shift_error": float(np.mean(np.abs(variance_difference))),
            "trajectory_rmse": float(np.sqrt(np.mean((s-t)**2)))}


def uncertainty(region, reference, score_a, score_b, levels, chunk=256):
    region, reference, a, b = map(finite_matrix, (region, reference, score_a, score_b))
    if len({x.shape[1] for x in (region, reference, a, b)}) != 1:
        raise ValueError("Uncertainty score dimensions differ")
    records = []
    for level in levels:
        bounds = np.quantile(region, [(1-level)/2, (1+level)/2], axis=0)
        per_feature = ((reference >= bounds[0]) & (reference <= bounds[1])).mean(0)
        records.append({"nominal_alpha": level, "coverage_sd": float(per_feature.mean()),
                        "coverage_sd_by_feature": json.dumps(per_feature.tolist()),
                        "interval_ece_at_level": float(np.abs(per_feature-level).mean()),
                        "mean_interval_width": float((bounds[1]-bounds[0]).mean()),
                        "region_unique_rows": len(np.unique(region, axis=0)),
                        "hdr_coverage": None, "hdr_ece": None, "hdr_reason": HDR_REASON})
    # Marginal CRPS uses the same independent score banks and held-out targets.
    marginal_crps = []
    for feature in range(reference.shape[1]):
        marginal_crps.append(energy_score(reference[:, feature:feature+1],
                                          a[:, feature:feature+1], b[:, feature:feature+1], chunk))
    return records, {"sd_ece": float(np.mean([r["interval_ece_at_level"] for r in records])),
                     "energy_score": energy_score(reference, a, b, chunk),
                     "crps": float(np.mean(marginal_crps)), "hdr_ece": None,
                     "hdr_reason": HDR_REASON}

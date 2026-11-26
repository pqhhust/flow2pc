"""Torch-free E4 protocol: explicit law, perturbations, budgets and HDR gate."""

import math

DEFAULTS = {
    "reference_mode": "teacher-relative", "source_split": "all",
    # Shared queue metadata keys; no real future observations are used by E4.
    "reference_split": "independent_teacher", "max_test_samples": 2000,
    "n_initial": 2000, "n_region": 2000, "n_es": 256,
    "seeds": [0, 1, 2], "levels": [.5, .7, .9, .95],
    "mean_shifts": [.25, .5, 1.], "spread_factors": [.75, 1.25, 1.5],
    "direction": None, "time_indices": None, "representation_samples": 10000,
    "kernel_samples": 512, "kernel_multipliers": [.5, 1., 2.],
    "chunk_size": 256, "hdr_method": "unavailable",
    "batch_size": 128, "solver": "dopri5", "tolerance": 1e-5,
    "max_nfe": 50000, "ode_steps": 200,
}


def resolve(value):
    from flow2pc.utils.config import mapping

    mapping(value, "E4 evaluation", set(DEFAULTS))
    result = {**DEFAULTS, **value}
    if result["reference_mode"] != "teacher-relative":
        raise ValueError("E4 has no independent shifted ground truth; only teacher-relative mode is implemented")
    if result["source_split"] not in ("all", "train") or result["reference_split"] != "independent_teacher":
        raise ValueError("E4 needs a declared empirical source and independent teacher references")
    if result["hdr_method"] != "unavailable":
        raise ValueError("E4 HDR requires a validated density estimator; PC density is not the ODE law")
    for key in ("max_test_samples", "n_initial", "n_region", "n_es", "representation_samples",
                "kernel_samples", "chunk_size", "batch_size", "max_nfe", "ode_steps"):
        if type(result[key]) is not int or result[key] < 2:
            raise ValueError(f"evaluation.{key} must be an integer >= 2")
    for key in ("seeds", "levels", "mean_shifts", "spread_factors", "kernel_multipliers"):
        items = result[key]
        if (not isinstance(items, list) or not items or items != sorted(set(items))
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in items)):
            raise ValueError(f"Invalid evaluation.{key}")
    if any(type(s) is not int or not 0 <= s < 2**32 for s in result["seeds"]):
        raise ValueError("seeds must be distinct unsigned 32-bit integers")
    if any(not 0 < a < 1 for a in result["levels"]) or .9 not in result["levels"]:
        raise ValueError("levels must include .9 and be inside (0,1)")
    if any(a == 0 for a in result["mean_shifts"]):
        raise ValueError("Zero mean shift is the separate clean condition")
    if any(c <= 0 or c == 1 for c in result["spread_factors"]):
        raise ValueError("Spread factors must be positive and nonidentity; clean represents c=1")
    if any(m <= 0 for m in result["kernel_multipliers"]):
        raise ValueError("Kernel multipliers must be positive")
    direction = result["direction"]
    if (not isinstance(direction, list) or not direction
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in direction)
            or not math.isclose(sum(v*v for v in direction), 1., rel_tol=1e-7, abs_tol=1e-7)):
        raise ValueError("Declare a finite unit shift direction in training-standardized coordinates")
    indices = result["time_indices"]
    if indices is not None and (not isinstance(indices, list) or not indices
                               or indices != sorted(set(indices))
                               or any(type(i) is not int or i <= 0 for i in indices)):
        raise ValueError("time_indices must be distinct sorted noninitial indices")
    if (result["solver"] not in ("dopri5", "rk4")
            or type(result["tolerance"]) not in (float, int)
            or not math.isfinite(result["tolerance"]) or result["tolerance"] <= 0):
        raise ValueError("Invalid E4 ODE solver/tolerance")
    return result

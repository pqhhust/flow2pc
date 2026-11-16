"""Explicit E3.1 budgets; no student optimization or post-hoc recalibration."""

import math

DEFAULTS = {
    "levels": [0.5, 0.7, 0.9, 0.95], "seeds": [0, 1, 2],
    "bank_sizes": [5000, 10000], "tune_samples": 2048, "energy_samples": 2048,
    "max_test_samples": 2048, "min_test_samples": 20, "representation_samples": 10000,
    "bandwidth_multipliers": [0.5, 1.0, 2.0], "kde_chunk_size": 256,
    "hdr_representation": "auto", "source_split": "all", "reference_split": "protocol",
    "analytic_seed": 20261113, "bandwidth_scale": "unit",
    "time_indices": None, "batch_size": 128, "tolerance": 1e-5,
    "max_nfe": 50000, "solver": "dopri5", "ode_steps": 200,
}


def resolve(value):
    from flow2pc.utils.config import mapping

    mapping(value, "E3.1 evaluation", set(DEFAULTS))
    result = {**DEFAULTS, **value}
    for key in ("tune_samples", "energy_samples", "max_test_samples", "min_test_samples",
                "representation_samples", "kde_chunk_size", "batch_size", "max_nfe", "ode_steps"):
        if type(result[key]) is not int or result[key] < 2:
            raise ValueError(f"evaluation.{key} must be an integer >= 2")
    for key in ("levels", "bandwidth_multipliers"):
        values = result[key]
        if (not isinstance(values, list) or not values
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)
                or (key == "levels" and (max(values) >= 1 or values != sorted(set(values))))):
            raise ValueError(f"Invalid evaluation.{key}")
    for key in ("seeds", "bank_sizes"):
        values = result[key]
        minimum = 0 if key == "seeds" else 2
        if (not isinstance(values, list) or not values or values != sorted(set(values))
                or any(type(v) is not int or not minimum <= v < 2**32 for v in values)):
            raise ValueError(f"Invalid evaluation.{key}")
    if result["hdr_representation"] not in ("auto", "full", "pca2"):
        raise ValueError("hdr_representation must be auto, full or pca2")
    if (result["source_split"] not in ("all", "train", "analytic_knot")
            or result["reference_split"] not in ("protocol", "test", "val", "all", "analytic_knot")):
        raise ValueError("Invalid source/reference split")
    if (result["source_split"] == "analytic_knot") != (result["reference_split"] == "analytic_knot"):
        raise ValueError("Analytic knot evaluation requires both source_split and reference_split: analytic_knot")
    if type(result["analytic_seed"]) is not int or not 0 <= result["analytic_seed"] < 2**32:
        raise ValueError("analytic_seed must be an unsigned 32-bit integer")
    if result["bandwidth_scale"] not in ("unit", "model_std"):
        raise ValueError("bandwidth_scale must be unit or model_std")
    indices = result["time_indices"]
    if indices is not None and (not isinstance(indices, list) or not indices
            or indices != sorted(set(indices)) or any(type(i) is not int or i < 0 for i in indices)):
        raise ValueError("time_indices must be sorted unique nonnegative indices")
    if (type(result["tolerance"]) not in (int, float)
            or not math.isfinite(result["tolerance"]) or result["tolerance"] <= 0
            or result["solver"] not in ("dopri5", "rk4")):
        raise ValueError("Invalid ODE settings")
    return result

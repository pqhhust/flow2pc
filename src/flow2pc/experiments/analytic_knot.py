"""Evaluation-only independent draws from the saved lifted-knot generator.

This explicitly replaces the finite empirical evaluation initial law with the
known analytic q0. It never changes a checkpoint, training sampler or data split.
"""

import hashlib

import numpy as np

from flow2pc.data.preprocessing import transform
from flow2pc.data.synthetic.knot import knot_means


def enabled(settings):
    return settings["source_split"] == "analytic_knot"


def parameters(dataset):
    if dataset.metadata.name != "knot" or dataset.dim != 2:
        raise ValueError("Analytic knot sampling requires the saved 2D knot dataset")
    size, noise = dataset.params["size"], dataset.params["noise_std"]
    means, times = knot_means(size)
    if (len(dataset.metadata.model_times) != size
            or not np.allclose(dataset.metadata.model_times, times, rtol=0, atol=1e-7)
            or not np.isfinite(noise) or noise <= 0):
        raise ValueError("Saved knot grid/noise differs from the analytic generator")
    return means, float(noise)


def array_sha256(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def contract(dataset, settings):
    means, noise = parameters(dataset)
    return {"name": "analytic_knot", "generator": "flow2pc.data.synthetic.knot:knot_means",
            "curve_size": len(means), "mean_curve_raw": means.tolist(),
            "noise_std_raw": noise, "noise_covariance_raw": (np.eye(2) * noise**2).tolist(),
            "vertical_shift": "middle branch sin(...)+1, from the existing shared generator",
            "analytic_seed": settings["analytic_seed"],
            "reference_samples_per_time": settings["max_test_samples"],
            "evaluation_initial_law": {"kind": "analytic Gaussian knot q0", "mean_raw": means[0].tolist(),
                                       "noise_std_raw": noise},
            "checkpoint_initial_law": dataset.contract()["initial_law"],
            "initial_law_override": "explicit evaluation-only analytic law; not the finite saved source pool",
            "reference_independence": "fresh generator noise; no training/validation/test snapshot rows reused",
            "stream_namespaces": {"reference": 0, "source_banks": 1},
            "time_coupling": "no paired real trajectories; independent marginal observations",
            "training_unchanged": True}


def reference(dataset, index, settings):
    means, noise = parameters(dataset)
    if not 0 <= index < len(means):
        raise ValueError("Analytic reference index is outside the saved knot grid")
    rng = np.random.default_rng(np.random.SeedSequence([settings["analytic_seed"], 0, index]))
    values = means[index] + rng.standard_normal((settings["max_test_samples"], 2)) * noise
    return values, "analytic_knot", len(values)


def source_banks(dataset, settings):
    """Disjoint source-draw blocks; identical blocks are used by teacher/student."""
    means, noise = parameters(dataset)
    maximum = max(settings["bank_sizes"])
    sizes = {"fit": maximum, "tune": settings["tune_samples"], "threshold": maximum,
             "energy_a": settings["energy_samples"], "energy_b": settings["energy_samples"]}
    arrays, plans, offset = [], {}, 0
    for seed in settings["seeds"]:
        children = np.random.SeedSequence([settings["analytic_seed"], 1, seed]).spawn(len(sizes))
        plan = {}
        for (name, count), child in zip(sizes.items(), children):
            values = means[0] + np.random.default_rng(child).standard_normal((count, 2)) * noise
            arrays.append(values)
            plan[name] = np.arange(offset, offset + count)
            offset += count
        plans[seed] = plan
    raw = np.concatenate(arrays)
    source = transform(raw, dataset.preprocessing)
    return source, plans, {"initial_law": "analytic Gaussian knot q0; fresh independent draws",
                           "source_split": "analytic_knot", "source_count": len(source),
                           "unique_source_rows_integrated": len(source),
                           "raw_source_sha256": array_sha256(raw),
                           "analytic_seed": settings["analytic_seed"],
                           "source_resampling_with_replacement": False,
                           "bank_independence": "disjoint independently drawn analytic source banks; shared across model roles"}

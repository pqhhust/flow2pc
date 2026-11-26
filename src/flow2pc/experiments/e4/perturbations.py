"""Perturb a finite empirical source law; preserve identity and draw separation."""

import hashlib

import numpy as np

from flow2pc.experiments.e3_1.metrics import finite_matrix


def conditions(settings):
    return ([{"condition": "clean", "shift_type": "clean", "severity": 0.}]
            + [{"condition": f"mean-{x:g}", "shift_type": "mean", "severity": x}
               for x in settings["mean_shifts"]]
            + [{"condition": f"spread-{x:g}", "shift_type": "spread", "severity": x}
               for x in settings["spread_factors"]])


def perturb(values, condition, direction):
    values = finite_matrix(values)
    direction = np.asarray(direction, dtype=np.float64)
    if direction.shape != (values.shape[1],) or not np.isclose(direction @ direction, 1.):
        raise ValueError("Shift direction must be a unit vector of the saved state dimension")
    kind, severity = condition["shift_type"], condition["severity"]
    if kind == "clean":
        return values.copy()
    if kind == "mean":
        return values + severity * direction
    if kind == "spread" and severity > 0:
        center = values.mean(0)
        return center + severity * (values - center)
    raise ValueError("Unsupported perturbation")


def bank_plans(count, settings):
    if count < 2:
        raise ValueError("E4 requires at least two source rows")
    sizes = {"initial": settings["n_initial"], "floor": settings["n_initial"],
             "region": settings["n_region"], "reference": settings["max_test_samples"],
             "energy_a": settings["n_es"], "energy_b": settings["n_es"]}
    plans, audit = {}, []
    for seed in settings["seeds"]:
        children = np.random.SeedSequence([20261125, seed]).spawn(len(sizes))
        plans[seed] = {}
        for (name, size), child in zip(sizes.items(), children):
            ids = np.random.default_rng(child).integers(count, size=size)
            plans[seed][name] = ids
            audit.append({"seed": seed, "bank": name, "draws": size,
                          "unique_source_rows": len(np.unique(ids)),
                          "indices_sha256": hashlib.sha256(ids.tobytes()).hexdigest()})
    return plans, audit

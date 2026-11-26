"""Evaluation configuration without importing model implementations."""

import math

TRANSPORT_W2_PROTOCOL = "transport_ode_w2_v1"
TRANSPORT_W2_POPULATIONS = "truncate both subsets to first min(N,M) rows after configured subset selection"


def transport_w2_complete(metrics, heldout_indices):
    """Torch-free guard against accepting an old W1 report as transport W2."""
    if not isinstance(metrics, dict) or not isinstance(metrics.get("protocol"), dict):
        return False
    protocol = metrics.get("protocol", {})
    if (protocol.get("metric_protocol_version") != TRANSPORT_W2_PROTOCOL
            or protocol.get("primary_distance") != "w2"
            or protocol.get("units") != "raw_feature_coordinates"
            or protocol.get("w2_population_policy") != TRANSPORT_W2_POPULATIONS
            or protocol.get("time_rule") != "exact observation times"):
        return False
    candidates = metrics.get("scores", [])
    if not isinstance(candidates, list) or any(not isinstance(r, dict) for r in candidates):
        return False
    rows = [r for r in candidates if r.get("held_out")]
    if any(type(r.get("index")) is not int for r in rows):
        return False
    if sorted(r.get("index", -1) for r in rows) != sorted(heldout_indices) or not rows:
        return False
    values = [r.get("teacher_w2") for r in rows]
    score = metrics.get("teacher_heldout_w2")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in [score, *values]):
        return False
    return math.isclose(sum(values) / len(values), score, rel_tol=1e-12, abs_tol=1e-12)


DEFAULTS = dict(
    enabled=False,
    num_samples=512,
    metric_samples=512,
    seed=2026,
    batch_size=64,
    tolerance=1e-5,
    max_nfe=50000,
    solver="dopri5",
    ode_steps=100,
    metric_space="historical",
    source_split="all",
    reference_split="all",
)


def resolve_evaluation(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS) - {"image_dir", "protocol", "time_indices"}:
        raise ValueError("Unknown evaluation settings")
    result = {**DEFAULTS, **value}
    if "protocol" in value and value["protocol"] not in ("shared_ode", "native_otpfm"):
        raise ValueError("Unknown evaluation.protocol")
    if "time_indices" in value:
        indices = value["time_indices"]
        if (not isinstance(indices, list) or len(indices) < 2
                or any(type(i) is not int or i < 0 for i in indices)
                or indices != sorted(set(indices))):
            raise ValueError("evaluation.time_indices must be sorted unique nonnegative marginal indices")
    if type(result["enabled"]) is not bool:
        raise ValueError("evaluation.enabled must be boolean")
    for key in (
        "num_samples",
        "metric_samples",
        "seed",
        "batch_size",
        "max_nfe",
        "ode_steps",
    ):
        if type(result[key]) is not int or result[key] < (
            0 if key in ("num_samples", "metric_samples", "seed") else 1
        ):
            raise ValueError(f"Invalid evaluation.{key}")
    if result["seed"] >= 2**32:
        raise ValueError("evaluation.seed must fit an unsigned 32-bit integer")
    if (
        type(result["tolerance"]) not in (int, float)
        or not math.isfinite(result["tolerance"])
        or result["tolerance"] <= 0
    ):
        raise ValueError("evaluation.tolerance must be finite and positive")
    if result["solver"] not in ("dopri5", "rk4") or result["metric_space"] not in (
        "historical",
        "raw",
    ):
        raise ValueError("Invalid evaluation solver or metric space")
    for key in ("source_split", "reference_split"):
        if result[key] not in ("all", "train", "val", "test"):
            raise ValueError(f"Invalid evaluation.{key}")
    return result

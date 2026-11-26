"""E3.2 settings: fixed bandwidth sensitivity, shared masks, no training."""

import math

from flow2pc.experiments.e3_1.config import DEFAULTS as BASE, resolve as resolve_base

DEFAULTS = {
    "observed_masks": [[0], [1]],
    "condition_bandwidths": [0.05, 0.1, 0.2],
    "primary_bandwidth": 0.1,
    "conditional_draw_sizes": [256, 512],
    "min_effective_components": 20,
}


def validate_dimension(masks, dimension):
    if type(dimension) is not int or dimension < 2:
        raise ValueError("E3.2 needs >=2 state coordinates; a 1D marginal has no nonempty observed/hidden split")
    for observed in masks:
        if (not isinstance(observed, list) or not observed
                or any(type(index) is not int or index < 0 or index >= dimension for index in observed)
                or observed != sorted(set(observed)) or len(observed) >= dimension):
            raise ValueError(f"Observed mask {observed} leaves no valid hidden coordinates for dimension {dimension}")


def resolve(value):
    from flow2pc.utils.config import mapping

    mapping(value, "E3.2 evaluation", set(BASE) | set(DEFAULTS))
    result = resolve_base({key: val for key, val in value.items() if key in BASE})
    result.update({**DEFAULTS, **{key: val for key, val in value.items() if key in DEFAULTS}})
    if result["source_split"] == "analytic_knot" or result["reference_split"] in ("all", "analytic_knot"):
        raise ValueError("E3.2 uses saved reserved test/val populations and empirical source laws")
    if .9 not in result["levels"]:
        raise ValueError("E3.2 requires level 0.9 for its headline coverage")
    masks = result["observed_masks"]
    if (not isinstance(masks, list) or not masks
            or any(not isinstance(mask, list) or not mask or mask != sorted(set(mask))
                   or any(type(i) is not int or i < 0 for i in mask) for mask in masks)
            or len({tuple(mask) for mask in masks}) != len(masks)):
        raise ValueError("observed_masks must contain distinct sorted nonempty coordinate lists")
    widths = result["condition_bandwidths"]
    if (not isinstance(widths, list) or not widths
            or any(type(h) not in (int, float) or not math.isfinite(h) or h <= 0 for h in widths)
            or widths != sorted(set(widths))):
        raise ValueError("condition_bandwidths must be sorted unique finite positive numbers")
    primary = result["primary_bandwidth"]
    if type(primary) not in (int, float) or primary not in widths:
        raise ValueError("primary_bandwidth must be declared in condition_bandwidths before evaluation")
    sizes = result["conditional_draw_sizes"]
    if (not isinstance(sizes, list) or not sizes or sizes != sorted(set(sizes))
            or any(type(n) is not int or n < 2 for n in sizes)):
        raise ValueError("conditional_draw_sizes must be sorted unique integers >= 2")
    if type(result["min_effective_components"]) is not int or result["min_effective_components"] < 2:
        raise ValueError("min_effective_components must be an integer >= 2")
    # These E3.1-only settings have no mathematical role in E3.2. Reject
    # explicit values rather than allowing users to believe they affect results.
    if set(value) & {"hdr_representation", "bandwidth_multipliers", "bandwidth_scale"}:
        raise ValueError("E3.2 conditions in full standardized state space; use condition_bandwidths")
    for key in ("hdr_representation", "bandwidth_multipliers", "bandwidth_scale"):
        result.pop(key)
    return result

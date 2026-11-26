"""Evaluation-only numerical checks of E3 queries against the original PC law."""

import numpy as np
import torch


@torch.no_grad()
def query_consistency(student, values, time, preprocessing, *, seed, batch_size=64):
    """Check full-dimensional densities, conditional identities, and CDF draws.

    Exact-density comparisons use the original student.log_prob as reference.
    Independent conditional samples are checked against circuit rectangle
    probabilities, with a conservative six-binomial-SE numerical check. This
    is an implementation diagnostic, not real-data calibration or a proof.
    """
    from flow2pc.experiments.pc_queries import DirectPC

    parameter = next(student.parameters())
    if student.training or values.ndim != 2 or values.shape[1] != student.dim or len(values) < 2:
        raise ValueError("Query audit requires an eval-mode PC and at least two full-dimensional states")
    adapter = DirectPC(student, preprocessing, np.zeros(student.dim), np.ones(student.dim), batch_size)
    law = adapter.at(time)
    x = values[:64].to(parameter)
    shift, scale = adapter.raw_shift, adapter.raw_scale
    raw = (x.double() * scale + shift).cpu().numpy()
    log_jacobian = float(scale.log().sum())

    def native_log_prob(points):
        scores = [student.log_prob(chunk, parameter.new_tensor(time)).double().cpu()
                  for chunk in points.to(parameter).split(batch_size)]
        return torch.cat(scores).numpy()

    def compare(actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise FloatingPointError("Nonfinite density in direct-query consistency audit")
        error = np.abs(actual - expected)
        allowed = 5e-4 + 2e-5 * np.abs(expected)
        return {"passed": bool(np.all(error <= allowed)),
                "max_abs_log_density_error": float(error.max()),
                "max_error_over_tolerance": float((error / allowed).max()),
                "states": len(actual), "absolute_tolerance": 5e-4, "relative_tolerance": 2e-5}

    joint = compare(law.log_prob(raw), native_log_prob(x) - log_jacobian)
    # Zero-dimensional marginal and unbounded rectangle must both have mass 1.
    normalization_error = abs(float(law.marginal_log_prob(np.empty((1, 0)), [])[0]))
    normalization_error = max(normalization_error, abs(law.rectangle_probability(
        np.full(student.dim, -np.inf), np.full(student.dim, np.inf)) - 1.))

    def sampling_check(distribution, namespace):
        count = 512
        a = distribution.sample(count, np.random.default_rng(np.random.SeedSequence([seed, namespace, 0])))
        b = distribution.sample(count, np.random.default_rng(np.random.SeedSequence([seed, namespace, 1])))
        # Test a coordinate event, integrating all other hidden dimensions out.
        # A joint interquartile box would have vanishing mass in 100D.
        lower, upper = np.full(a.shape[1], -np.inf), np.full(a.shape[1], np.inf)
        lower[0], upper[0] = np.quantile(a[:, 0], [.25, .75])
        probability = distribution.rectangle_probability(lower, upper)
        observed = float(np.mean((b[:, 0] >= lower[0]) & (b[:, 0] <= upper[0])))
        allowed = 6 * np.sqrt(probability * (1 - probability) / count) + 1 / count
        return {"passed": bool(abs(observed - probability) <= allowed),
                "circuit_probability": probability, "independent_sample_fraction": observed,
                "tolerance": float(allowed), "samples": count,
                "queried_coordinate": distribution.hidden[0]}

    unconditional_sampling = sampling_check(law, 0)
    candidates = [[0], [student.dim - 1], list(range(student.dim - 1))]
    masks = list(dict.fromkeys(tuple(mask) for mask in candidates if mask and len(mask) < student.dim))
    conditionals = []
    for number, mask in enumerate(masks, 1):
        observed = list(mask)
        # Fixed evidence from a generated full joint state, never a test target.
        conditional = law.condition(observed, raw[0, observed])
        full = x.clone()
        full[:, observed] = x[0, observed]
        full_raw = (full.double() * scale + shift).cpu().numpy()
        expected = native_log_prob(full) - log_jacobian - float(conditional.log_normalizer)
        density = compare(conditional.log_prob(full_raw[:, conditional.hidden]), expected)
        sampling = sampling_check(conditional, number)
        conditionals.append({"observed": observed, "hidden": conditional.hidden,
                             "density": density, "sampling": sampling,
                             "passed": density["passed"] and sampling["passed"]})
    return {"passed": bool(joint["passed"] and normalization_error < 1e-8
                             and unconditional_sampling["passed"]
                             and all(c["passed"] for c in conditionals)),
            "full_state_dimension": student.dim, "model_time": time,
            "joint_density": joint, "normalization_error": normalization_error,
            "unconditional_sampling": unconditional_sampling, "conditionals": conditionals,
            "scope": "numerical query implementation checks; no teacher KDE, no real-data calibration, no ODE"}

"""Read-only velocity-support diagnostics for frozen teacher/student pairs."""

import math

import torch


@torch.no_grad()
def velocity_support_metrics(student, teacher, values, model_time, batch_size, *, tail_count=0):
    """Score a declared state population; do not resample or drop its tails.

    Errors and speeds are in saved model-coordinate units per model-time unit.
    Relative RMSE is sqrt(mean squared error / mean squared teacher speed).
    A zero teacher energy gives N/A rather than a denominator floor.
    Optional tail records retain the exact scored rows; they never alter the
    population or the aggregate errors. Density is queried only when requested.
    """
    if (values.ndim != 2 or not len(values) or values.shape[1] != student.dim
            or not torch.isfinite(values).all() or batch_size < 1):
        raise ValueError("Velocity-support diagnostics need finite nonempty [N,D] states")
    if type(tail_count) is not int or tail_count < 0:
        raise ValueError("tail_count must be a nonnegative integer")
    parameter = next(student.parameters())
    errors, speeds, point_errors = [], [], []
    states, predictions, targets = [], [], []
    for chunk in values.split(batch_size):
        x = chunk.to(parameter)
        time = x.new_tensor(model_time)
        target = teacher.velocity(x, time)
        predicted = student.velocity(x, time)
        if target.shape != x.shape or predicted.shape != x.shape:
            raise ValueError("Teacher/student velocity shapes differ from diagnostic states")
        if not torch.isfinite(target).all() or not torch.isfinite(predicted).all():
            raise FloatingPointError("Nonfinite velocity on the declared diagnostic population")
        squared = (predicted.double() - target.double()).square()
        errors.append(squared.cpu())
        speeds.append(target.double().square().cpu())
        point_errors.append(squared.mean(-1).sqrt().cpu())
        if tail_count:
            states.append(x.detach().cpu())
            predictions.append(predicted.detach().cpu())
            targets.append(target.detach().cpu())
    mse = float(torch.cat(errors).mean())
    energy = float(torch.cat(speeds).mean())
    point_errors = torch.cat(point_errors)
    result = {
        "state_count": len(values), "velocity_mse_model": mse,
        "velocity_rmse_model": math.sqrt(mse),
        "teacher_velocity_energy_model": energy,
        "relative_velocity_rmse": math.sqrt(mse / energy) if energy > 0 else None,
        "point_velocity_rmse_median": float(point_errors.median()),
        "point_velocity_rmse_p90": float(torch.quantile(point_errors, .9)),
        "point_velocity_rmse_p99": float(torch.quantile(point_errors, .99)),
        "point_velocity_rmse_max": float(point_errors.max()),
    }
    if any(not math.isfinite(value) for value in result.values() if value is not None):
        raise FloatingPointError("Nonfinite aggregate velocity-support diagnostic")
    if tail_count:
        squared_per_point = torch.cat(errors).mean(-1)
        indices = torch.argsort(squared_per_point, descending=True, stable=True)[:tail_count]
        selected = torch.cat(states)[indices]
        density = student.log_prob(selected.to(parameter), parameter.new_tensor(model_time))
        if density.shape != (len(indices),) or not torch.isfinite(density).all():
            raise FloatingPointError("Nonfinite or invalid PC density on selected tail states")
        predictions, targets = torch.cat(predictions), torch.cat(targets)
        total = float(squared_per_point.sum())
        result["tail_samples"] = [
            {"rank": rank + 1, "population_index": int(index),
             "state_model": selected[rank].tolist(),
             "student_velocity_model": predictions[index].tolist(),
             "teacher_velocity_model": targets[index].tolist(),
             "point_velocity_mse_model": float(squared_per_point[index]),
             "population_mse_fraction": float(squared_per_point[index]) / total if total else None,
             "pc_log_probability_model": float(density[rank]),
             "pc_nll_per_dimension_model": -float(density[rank]) / student.dim}
            for rank, index in enumerate(indices.tolist())
        ]
    return result


def diagnostic_failure(error):
    """Only numerical failures can be recorded and isolated; contracts propagate."""
    from flow2pc.evaluation.metric_flow import _numerical_failure

    return _numerical_failure(error)

"""Sample-based estimated HDR coverage and full-state Energy Score.

Density fit, bandwidth selection and mass thresholds use separate MODEL banks.
No real evaluation observation is used to fit or tune a region.
"""

import numpy as np
from scipy.spatial.distance import cdist


def finite_matrix(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Expected a nonempty finite [N,D] sample matrix")
    return values


class Representation:
    def __init__(self, training, mode):
        training = finite_matrix(training)
        self.center = training.mean(0)
        self.scale = np.maximum(training.std(0), 1e-8)
        values = (training - self.center) / self.scale
        self.full_dim = training.shape[1]
        projected = mode == "pca2" or (mode == "auto" and self.full_dim > 3)
        if not projected and self.full_dim > 3:
            raise ValueError("Full-state KDE is restricted to <=3D; explicitly select pca2")
        if projected:
            _, _, vectors = np.linalg.svd(values, full_matrices=False)
            self.axes = vectors[:min(2, self.full_dim)].T
        else:
            self.axes = np.eye(self.full_dim)
        self.projected = projected
        self.project_scale = np.maximum((values @ self.axes).std(0), 1e-8)

    def energy(self, values):
        return (finite_matrix(values) - self.center) / self.scale

    def hdr(self, values):
        return self.energy(values) @ self.axes / self.project_scale

    def contract(self):
        return {"fit_scope": "observed training rows only", "full_dim": self.full_dim,
                "hdr_dim": self.axes.shape[1], "projected": self.projected,
                "center": self.center.tolist(), "scale": self.scale.tolist(),
                "axes": self.axes.tolist(), "project_scale": self.project_scale.tolist(),
                "energy_space": "full-state training-standardized coordinates",
                "hdr_space": "training-PCA projection" if self.projected else "full state, training-standardized coordinates",
                "density": "Gaussian-KDE ranking regions; estimated/smoothed model HDR, not exact flow density"}


def energy_score(real, generated, independent, chunk=256):
    """Unbiased independent-bank Monte Carlo ES; no quadratic arrays are saved."""
    real, generated, independent = map(finite_matrix, (real, generated, independent))
    if real.shape[1] != generated.shape[1] or generated.shape[1] != independent.shape[1]:
        raise ValueError("Energy Score dimensions differ")
    attraction = np.concatenate([cdist(real[i:i + chunk], generated).mean(1)
                                for i in range(0, len(real), chunk)])
    repulsion = sum(cdist(generated[i:i + chunk], independent).sum()
                    for i in range(0, len(generated), chunk)) / (len(generated) * len(independent))
    return float((attraction - 0.5 * repulsion).mean())


def distribution_fidelity(teacher, student, levels):
    """Full-state diagnostics on paired initial-particle pushforwards, not real trajectories."""
    teacher, student = map(finite_matrix, (teacher, student))
    if teacher.shape != student.shape:
        raise ValueError("Fidelity requires the same initial-particle bank and state dimension")
    teacher_centered, student_centered = teacher - teacher.mean(0), student - student.mean(0)
    teacher_cov = teacher_centered.T @ teacher_centered / len(teacher)
    student_cov = student_centered.T @ student_centered / len(student)
    covariance_norm = float(np.linalg.norm(teacher_cov))
    shared = {
        "paired_ode_rmse": float(np.sqrt(np.mean((student - teacher) ** 2))),
        "mean_shift_l2": float(np.linalg.norm(student.mean(0) - teacher.mean(0))),
        "relative_covariance_error": (float(np.linalg.norm(student_cov - teacher_cov)) / covariance_norm
                                      if covariance_norm > 0 else None),
    }
    rows = []
    for level in levels:
        quantiles = [(1 - level) / 2, (1 + level) / 2]
        teacher_bounds, student_bounds = np.quantile(teacher, quantiles, axis=0), np.quantile(student, quantiles, axis=0)
        teacher_width, student_width = teacher_bounds[1] - teacher_bounds[0], student_bounds[1] - student_bounds[0]
        total_width = float(teacher_width.sum())
        rows.append({"nominal_alpha": level, **shared,
                     "mean_interval_width_ratio": (float(np.mean(student_width / teacher_width))
                                                   if np.all(teacher_width > 0) else None),
                     "relative_interval_width_error": (float(np.abs(student_width - teacher_width).sum()) / total_width
                                                       if total_width > 0 else None)})
    return rows


def region_fidelity(teacher_estimator, student_estimator, teacher_draws, student_draws,
                    teacher_regions, student_regions):
    """Mass on independent model banks; separates teacher fidelity from truth coverage."""
    teacher_thresholds = {r["nominal_alpha"]: r["log_density_threshold"] for r in teacher_regions}
    student_thresholds = {r["nominal_alpha"]: r["log_density_threshold"] for r in student_regions}
    if teacher_thresholds.keys() != student_thresholds.keys():
        raise ValueError("Teacher and student region levels differ")
    tt, ts = teacher_estimator.score(teacher_draws), teacher_estimator.score(student_draws)
    st, ss = student_estimator.score(teacher_draws), student_estimator.score(student_draws)
    rows = []
    for level, threshold in teacher_thresholds.items():
        student_threshold = student_thresholds[level]
        teacher_self, student_in_teacher = float(np.mean(tt >= threshold)), float(np.mean(ts >= threshold))
        student_self, teacher_in_student = float(np.mean(ss >= student_threshold)), float(np.mean(st >= student_threshold))
        rows.append({"nominal_alpha": level,
                     "teacher_region_teacher_mass": teacher_self,
                     "teacher_region_student_mass": student_in_teacher,
                     "student_region_student_mass": student_self,
                     "student_region_teacher_mass": teacher_in_student,
                     "teacher_region_mass_gap": student_in_teacher - teacher_self,
                     "student_region_mass_gap": teacher_in_student - student_self})
    return rows


class EstimatedHDR:
    def __init__(self, fit, tune, multipliers, chunk=256, bandwidth_scale="unit"):
        from sklearn.neighbors import KernelDensity

        fit, tune = map(finite_matrix, (fit, tune))
        self.chunk = chunk
        base = len(fit) ** (-1 / (fit.shape[1] + 4))
        if bandwidth_scale not in ("unit", "model_std"):
            raise ValueError("Unknown model-only KDE bandwidth scaling")
        # Knot noise is narrow relative to the full curve. Scale the search to
        # model spread using fit draws only; never use real test observations.
        self.scale = (float(np.exp(np.mean(np.log(np.maximum(fit.std(0), 1e-8)))))
                      if bandwidth_scale == "model_std" else 1.0)
        base *= self.scale
        self.search = []
        models = []
        for multiplier in multipliers:
            bandwidth = base * multiplier
            model = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(fit)
            score = float(self._score(model, tune).mean())
            self.search.append({"bandwidth": bandwidth, "validation_model_log_score": score})
            models.append(model)
        index = max(range(len(models)), key=lambda i: self.search[i]["validation_model_log_score"])
        self.model = models[index]
        self.bandwidth = self.search[index]["bandwidth"]
        self.selected_index = index
        self.at_search_boundary = index in (0, len(models) - 1)

    def _score(self, model, values):
        scores = np.concatenate([model.score_samples(values[i:i + self.chunk])
                                 for i in range(0, len(values), self.chunk)])
        if not np.isfinite(scores).all():
            raise FloatingPointError("Nonfinite estimated density scores")
        return scores

    def score(self, values):
        return self._score(self.model, finite_matrix(values))

    def coverage(self, threshold_bank, real, levels):
        bank_scores, real_scores = self.score(threshold_bank), self.score(real)
        rows = []
        for level in levels:
            threshold = float(np.quantile(bank_scores, 1 - level, method="lower"))
            rows.append({"nominal_alpha": level, "coverage": float(np.mean(real_scores >= threshold)),
                         "log_density_threshold": threshold,
                         "threshold_bank_mass": float(np.mean(bank_scores >= threshold)),
                         "threshold_tie_fraction": float(np.mean(bank_scores == threshold))})
        ece = float(np.mean([abs(row["coverage"] - row["nominal_alpha"]) for row in rows]))
        return rows, ece

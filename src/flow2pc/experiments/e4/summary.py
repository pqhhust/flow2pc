"""Macro averages across noninitial times; uncertainty is MC only."""

import numpy as np
from scipy.stats import t

E41 = ("shifted_mmd2", "teacher_teacher_mmd2_floor", "clean_mmd2", "delta_mmd2",
       "mean_shift_error", "variance_shift_error", "trajectory_rmse")
E42 = ("sd_ece", "energy_score", "crps", "coverage90", "hdr_ece",
       "delta_sd_ece", "delta_energy_score", "delta_crps")


def summarize(rows, metrics, phase):
    keys = sorted({(r["condition"], r["role"]) for r in rows})
    result = []
    for condition, role in keys:
        group = [r for r in rows if r["condition"] == condition and r["role"] == role and r["time_index"] != 0]
        if not group:
            continue
        first = group[0]
        record = {k: first[k] for k in ("method", "dataset", "reference_mode", "eval_target",
                                       "condition", "shift_type", "severity", "role", "score_space")}
        record.update(phase=phase, evaluated_times=len({r["time_index"] for r in group}),
                      uncertainty_note="MC variability conditional on fixed checkpoints and finite empirical source; not training-seed or real-data uncertainty")
        for key in ("n_initial", "n_reference", "n_region", "n_es", "density_method", "hdr_reason"):
            if key in first:
                record[key] = first[key]
        for metric in metrics:
            per_seed = []
            for seed in sorted({r["seed"] for r in group}):
                values = [r.get(metric) for r in group if r["seed"] == seed and r.get(metric) is not None]
                if values:
                    per_seed.append(float(np.sqrt(np.mean(np.square(values))))
                                    if metric == "trajectory_rmse" else float(np.mean(values)))
            mean = float(np.mean(per_seed)) if per_seed else None
            std = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None
            se = std/np.sqrt(len(per_seed)) if std is not None else None
            half = float(t.ppf(.975, len(per_seed)-1)*se) if se is not None else None
            record.update({metric: mean, metric+"_mc_std": std, metric+"_mc_se": se,
                           metric+"_mc_ci95_low": mean-half if half is not None else None,
                           metric+"_mc_ci95_high": mean+half if half is not None else None})
        result.append(record)
    return result


def rectangular(rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    return [{k: row.get(k) for k in fields} for row in rows]

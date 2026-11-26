"""Explicit student-only knot resampling; frozen teacher data stay unchanged."""

from pathlib import Path

from flow2pc.utils.checkpoint import identity, sha256


def training_dataset(dataset, settings):
    """Return the student's training/val dataset and its optional provenance.

    No preprocessing is fitted here. The original dataset remains the teacher
    and benchmark contract; new independent marginal rows are used only for
    student initialization, teacher queries, and student validation/NLL.
    """
    specification = settings.get("student_data")
    if specification is None:
        return dataset, None
    if dataset.metadata.name != "knot":
        raise ValueError("student_data resampling currently supports the analytic knot only")
    if (specification.get("name") != "independent_knot"
            or set(specification) != {"name", "samples_per_time", "seed"}):
        raise ValueError("Use student_data: independent_knot with samples_per_time and seed")
    count, seed = specification["samples_per_time"], specification["seed"]
    if (type(count) is not int or count <= max(dataset.contract()["cells_per_time"])
            or type(seed) is not int or not 0 <= seed < 2**32 or seed == dataset.seed):
        raise ValueError("Student knot resampling needs more points and a distinct unsigned seed")
    from flow2pc.data.synthetic.knot import Knot

    expanded = Knot(name="knot", root=dataset.root, seed=seed,
                    params={**dataset.params, "samples_per_time": count},
                    preprocessing=dataset.preprocessing)
    if (expanded.metadata.model_times != dataset.metadata.model_times
            or expanded.metadata.physical_times != dataset.metadata.physical_times
            or expanded.metadata.preprocessing_id != dataset.metadata.preprocessing_id
            or expanded.observed != dataset.observed):
        raise ValueError("Student resampling changed coordinates, times, or observed marginals")
    return expanded, {
        "protocol": "independent_knot_student_marginals_v1",
        "teacher_dataset_sha256": identity(dataset.contract()),
        "dataset": expanded.contract(),
        "implementation_sha256": sha256(Path(__file__)),
        "roles": ["student_initialization", "teacher_query_populations", "velocity_validation", "source_NLL"],
        "evaluation": "teacher-bound benchmark data remain unchanged; fresh analytic evaluation is separate",
    }

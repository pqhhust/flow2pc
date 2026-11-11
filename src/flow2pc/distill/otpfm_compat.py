"""Torch-free compatibility for the conditional-v1 performance-only change.

Only the two explicitly listed source revisions may differ in a resumed
binding. Every query setting and the rest of the binding must still match.
Adding a future implementation here requires a separate equivalence review.
"""

CONDITIONAL_V1_IMPLEMENTATIONS = frozenset({
    "71c2e9a50a34310901bf277e7da1f3d9297e13ca83a21a812f66a8a9a3f84c14",
    "a34e312f479293e22adbdf0805f16a8944b5d0ef002d19681a32bac22f5e5f65",
})


def compatible_implementation(saved, current):
    if not isinstance(saved, str) or not isinstance(current, str):
        return False
    return saved == current or {saved, current} <= CONDITIONAL_V1_IMPLEMENTATIONS


def compatible_sampler(saved, current):
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    if saved.get("name") != "otpfm_conditional_v1" or current.get("name") != saved["name"]:
        return False
    if not compatible_implementation(
        saved.get("implementation_sha256"), current.get("implementation_sha256")
    ):
        return False
    return (
        {key: value for key, value in saved.items() if key != "implementation_sha256"}
        == {key: value for key, value in current.items() if key != "implementation_sha256"}
    )


def compatible_binding(saved, current):
    return (
        isinstance(saved, dict) and isinstance(current, dict)
        and compatible_sampler(saved.get("sampler"), current.get("sampler"))
        and {key: value for key, value in saved.items() if key != "sampler"}
        == {key: value for key, value in current.items() if key != "sampler"}
    )

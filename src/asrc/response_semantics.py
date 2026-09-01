from __future__ import annotations

from typing import Any


SUPPORTED_RESPONSE_KINDS = {
    "positive_strength",
    "nonnegative_magnitude",
    "signed_response",
}


def resolve_response_semantics(case_or_config: Any) -> dict[str, Any]:
    """Return response-aware constraint settings with backward-compatible defaults."""
    config = getattr(case_or_config, "config", case_or_config)
    config = config if isinstance(config, dict) else {}
    raw = config.get("response_semantics", {})
    if isinstance(raw, str):
        raw = {"kind": raw}
    if not isinstance(raw, dict):
        raise ValueError("response_semantics must be a mapping or a supported response kind.")

    kind = str(raw.get("kind", "positive_strength"))
    if kind not in SUPPORTED_RESPONSE_KINDS:
        raise ValueError(
            f"Unsupported response_semantics.kind {kind!r}; "
            f"choose from {sorted(SUPPORTED_RESPONSE_KINDS)}."
        )

    positive_default = kind in {"positive_strength", "nonnegative_magnitude"}
    dense_positive_default = kind == "positive_strength"
    return {
        "kind": kind,
        "unit": str(raw.get("unit", "")),
        "require_observed_positive": bool(
            raw.get("require_observed_positive", positive_default)
        ),
        "require_dense_positive": bool(
            raw.get("require_dense_positive", dense_positive_default)
        ),
        "boundary_max_correction_amplification": float(
            raw.get("boundary_max_correction_amplification", 5.0)
        ),
    }

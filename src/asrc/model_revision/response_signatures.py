from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping

import numpy as np
import pandas as pd

from asrc.model_revision.benchmarks import Gate1Task
from asrc.model_revision.evaluation import predict_repair
from asrc.model_revision.proposals import TypedRepair


@dataclass(frozen=True)
class ResponseSignatureConfig:
    axis_points: int = 41
    context_levels: tuple[float, ...] = (0.2, 0.5, 0.8)
    monotonic_fraction: float = 0.95
    near_linear_curvature: float = 0.05
    slope_ratio_lower: float = 0.5
    slope_ratio_upper: float = 2.0
    interaction_negligible: float = 0.02
    interaction_strong: float = 0.10

    def __post_init__(self) -> None:
        if self.axis_points < 9:
            raise ValueError("axis_points must be at least nine.")
        if not self.context_levels:
            raise ValueError("context_levels must not be empty.")
        if any(not 0.0 <= value <= 1.0 for value in self.context_levels):
            raise ValueError("context_levels must lie in [0, 1].")
        if not 0.5 < self.monotonic_fraction <= 1.0:
            raise ValueError("monotonic_fraction must lie in (0.5, 1].")
        if not 0.0 < self.slope_ratio_lower < 1.0:
            raise ValueError("slope_ratio_lower must lie in (0, 1).")
        if self.slope_ratio_upper <= 1.0:
            raise ValueError("slope_ratio_upper must exceed one.")


def axis_response_signature(
    coordinates: np.ndarray,
    responses: np.ndarray,
    *,
    config: ResponseSignatureConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ResponseSignatureConfig()
    x = np.asarray(coordinates, dtype=float)
    y = np.asarray(responses, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 5:
        raise ValueError("Axis signature requires equal one-dimensional arrays.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return {
            "finite": False,
            "monotonic_class": "nonfinite",
            "curvature_class": "nonfinite",
            "endpoint_slope_class": "nonfinite",
        }
    dx = np.diff(x)
    if np.any(dx <= 0.0):
        raise ValueError("Axis coordinates must be strictly increasing.")
    slopes = np.diff(y) / dx
    slope_scale = max(float(np.max(np.abs(slopes))), 1.0e-12)
    slope_tolerance = 1.0e-8 * slope_scale
    increasing_fraction = float(np.mean(slopes >= -slope_tolerance))
    decreasing_fraction = float(np.mean(slopes <= slope_tolerance))
    if increasing_fraction >= cfg.monotonic_fraction:
        monotonic_class = "increasing"
    elif decreasing_fraction >= cfg.monotonic_fraction:
        monotonic_class = "decreasing"
    else:
        monotonic_class = "mixed"

    midpoint_spacing = 0.5 * (dx[:-1] + dx[1:])
    curvature = np.diff(slopes) / midpoint_spacing
    response_span = max(float(np.ptp(y)), 1.0e-12)
    domain_span = max(float(x[-1] - x[0]), 1.0e-12)
    normalized_curvature = float(
        np.median(np.abs(curvature)) * domain_span**2 / response_span
    )
    curvature_scale = max(float(np.max(np.abs(curvature))), 1.0e-12)
    curvature_tolerance = 1.0e-8 * curvature_scale
    positive_curvature_fraction = float(
        np.mean(curvature >= -curvature_tolerance)
    )
    negative_curvature_fraction = float(
        np.mean(curvature <= curvature_tolerance)
    )
    if normalized_curvature <= cfg.near_linear_curvature:
        curvature_class = "near_linear"
    elif positive_curvature_fraction >= 0.75:
        curvature_class = "convex"
    elif negative_curvature_fraction >= 0.75:
        curvature_class = "concave"
    else:
        curvature_class = "mixed"

    segment = max(2, len(slopes) // 5)
    start_slope = float(np.median(np.abs(slopes[:segment])))
    end_slope = float(np.median(np.abs(slopes[-segment:])))
    endpoint_slope_ratio = end_slope / max(start_slope, 1.0e-12)
    if endpoint_slope_ratio <= cfg.slope_ratio_lower:
        endpoint_slope_class = "slope_decay"
    elif endpoint_slope_ratio >= cfg.slope_ratio_upper:
        endpoint_slope_class = "slope_growth"
    else:
        endpoint_slope_class = "slope_persistent"
    return {
        "finite": True,
        "minimum": float(np.min(y)),
        "maximum": float(np.max(y)),
        "response_span": response_span,
        "increasing_fraction": increasing_fraction,
        "decreasing_fraction": decreasing_fraction,
        "monotonic_class": monotonic_class,
        "normalized_curvature": normalized_curvature,
        "positive_curvature_fraction": positive_curvature_fraction,
        "negative_curvature_fraction": negative_curvature_fraction,
        "curvature_class": curvature_class,
        "endpoint_slope_ratio": endpoint_slope_ratio,
        "endpoint_slope_class": endpoint_slope_class,
    }


def _axis_contexts(
    task: Gate1Task,
    axis: str,
    config: ResponseSignatureConfig,
) -> list[dict[str, float]]:
    other = [name for name in task.variables if name != axis]
    if not other:
        return [{}]
    contexts = []
    for level in config.context_levels:
        context = {}
        for name in other:
            lower, upper = task.definition.locked_ranges[name]
            context[name] = float(lower + level * (upper - lower))
        contexts.append(context)
    return contexts


def _aggregate_axis_signatures(
    signatures: list[dict[str, Any]],
    config: ResponseSignatureConfig,
) -> dict[str, Any]:
    if not signatures or not all(item.get("finite", False) for item in signatures):
        return {
            "finite": False,
            "monotonic_class": "nonfinite",
            "curvature_class": "nonfinite",
            "endpoint_slope_class": "nonfinite",
        }
    increasing = float(np.median([item["increasing_fraction"] for item in signatures]))
    decreasing = float(np.median([item["decreasing_fraction"] for item in signatures]))
    if increasing >= config.monotonic_fraction:
        monotonic_class = "increasing"
    elif decreasing >= config.monotonic_fraction:
        monotonic_class = "decreasing"
    else:
        monotonic_class = "mixed"
    curvature_classes = [str(item["curvature_class"]) for item in signatures]
    slope_classes = [str(item["endpoint_slope_class"]) for item in signatures]

    def consensus(values: list[str]) -> str:
        unique, counts = np.unique(values, return_counts=True)
        index = int(np.argmax(counts))
        if counts[index] / len(values) >= 2.0 / 3.0:
            return str(unique[index])
        return "context_dependent"

    return {
        "finite": True,
        "minimum": float(min(item["minimum"] for item in signatures)),
        "maximum": float(max(item["maximum"] for item in signatures)),
        "increasing_fraction": increasing,
        "decreasing_fraction": decreasing,
        "monotonic_class": monotonic_class,
        "normalized_curvature": float(
            np.median([item["normalized_curvature"] for item in signatures])
        ),
        "curvature_class": consensus(curvature_classes),
        "endpoint_slope_ratio": float(
            np.median([item["endpoint_slope_ratio"] for item in signatures])
        ),
        "endpoint_slope_class": consensus(slope_classes),
    }


def _interaction_signature(
    task: Gate1Task,
    repair: TypedRepair,
    parameters: Mapping[str, float],
    left: str,
    right: str,
    config: ResponseSignatureConfig,
) -> dict[str, Any]:
    points = max(9, min(config.axis_points, 21))
    left_values = np.linspace(*task.definition.locked_ranges[left], points)
    right_values = np.linspace(*task.definition.locked_ranges[right], points)
    rows = []
    for left_value in left_values:
        for right_value in right_values:
            row = {
                name: float(np.mean(task.definition.locked_ranges[name]))
                for name in task.variables
            }
            row[left] = float(left_value)
            row[right] = float(right_value)
            rows.append(row)
    frame = pd.DataFrame(rows)
    prediction = predict_repair(task, repair, frame, parameters).reshape(points, points)
    if not np.all(np.isfinite(prediction)):
        return {"finite": False, "interaction_class": "nonfinite"}
    center = points // 2
    additive_reference = (
        prediction[:, [center]]
        + prediction[[center], :]
        - prediction[center, center]
    )
    nonadditive = prediction - additive_reference
    scale = max(float(np.ptp(prediction)), 1.0e-12)
    strength = float(np.sqrt(np.mean(nonadditive**2)) / scale)
    if strength <= config.interaction_negligible:
        interaction_class = "negligible"
    elif strength >= config.interaction_strong:
        interaction_class = "strong"
    else:
        interaction_class = "moderate"
    return {
        "finite": True,
        "normalized_nonadditivity": strength,
        "interaction_class": interaction_class,
    }


def candidate_response_signature(
    task: Gate1Task,
    repair: TypedRepair,
    parameters: Mapping[str, float],
    *,
    config: ResponseSignatureConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ResponseSignatureConfig()
    axes: dict[str, dict[str, Any]] = {}
    for axis in task.variables:
        lower, upper = task.definition.locked_ranges[axis]
        coordinates = np.linspace(lower, upper, cfg.axis_points)
        signatures = []
        for context in _axis_contexts(task, axis, cfg):
            frame = pd.DataFrame(
                {
                    name: (
                        coordinates
                        if name == axis
                        else np.full(cfg.axis_points, context[name], dtype=float)
                    )
                    for name in task.variables
                }
            )
            prediction = predict_repair(task, repair, frame, parameters)
            signatures.append(
                axis_response_signature(coordinates, prediction, config=cfg)
            )
        axes[axis] = _aggregate_axis_signatures(signatures, cfg)
    interactions = {
        f"{left}:{right}": _interaction_signature(
            task, repair, parameters, left, right, cfg
        )
        for left, right in combinations(task.variables, 2)
    }
    audit_prediction = predict_repair(task, repair, task.audit, parameters)
    return {
        "finite": bool(np.all(np.isfinite(audit_prediction))),
        "audit_prediction_rms": float(
            np.sqrt(np.mean(np.asarray(audit_prediction, dtype=float) ** 2))
        ),
        "audit_prediction_minimum": float(np.min(audit_prediction)),
        "audit_prediction_maximum": float(np.max(audit_prediction)),
        "axes": axes,
        "interactions": interactions,
    }


def response_signature_contrast(
    selected: Mapping[str, Any],
    alternative: Mapping[str, Any],
) -> dict[str, Any]:
    differences: list[str] = []
    axis_rows = []
    for axis in sorted(set(selected.get("axes", {})) & set(alternative.get("axes", {}))):
        left = selected["axes"][axis]
        right = alternative["axes"][axis]
        for contract_type, field in (
            ("monotonicity", "monotonic_class"),
            ("curvature", "curvature_class"),
            ("endpoint_slope", "endpoint_slope_class"),
        ):
            if left.get(field) != right.get(field):
                differences.append(f"{contract_type}:{axis}")
        axis_rows.append(
            {
                "axis": axis,
                "selected_monotonicity": left.get("monotonic_class"),
                "alternative_monotonicity": right.get("monotonic_class"),
                "selected_curvature": left.get("curvature_class"),
                "alternative_curvature": right.get("curvature_class"),
                "selected_endpoint_slope": left.get("endpoint_slope_class"),
                "alternative_endpoint_slope": right.get("endpoint_slope_class"),
                "selected_endpoint_slope_ratio": left.get("endpoint_slope_ratio"),
                "alternative_endpoint_slope_ratio": right.get("endpoint_slope_ratio"),
            }
        )
    interaction_rows = []
    for pair in sorted(
        set(selected.get("interactions", {}))
        & set(alternative.get("interactions", {}))
    ):
        left = selected["interactions"][pair]
        right = alternative["interactions"][pair]
        if left.get("interaction_class") != right.get("interaction_class"):
            differences.append(f"interaction:{pair}")
        interaction_rows.append(
            {
                "variable_pair": pair,
                "selected_interaction": left.get("interaction_class"),
                "alternative_interaction": right.get("interaction_class"),
                "selected_nonadditivity": left.get("normalized_nonadditivity"),
                "alternative_nonadditivity": right.get("normalized_nonadditivity"),
            }
        )
    return {
        "descriptive_separation_types": sorted(set(differences)),
        "descriptively_separable": bool(differences),
        "axis_contrasts": axis_rows,
        "interaction_contrasts": interaction_rows,
    }


def response_family_labels(signature: Mapping[str, Any]) -> dict[str, str]:
    """Flatten qualitative response behavior into a source-neutral family key.

    The labels describe total model behavior, not expression syntax. They are
    intentionally limited to generic response properties that can be evaluated
    for any scalar regression model over a declared input domain.
    """

    labels = {
        "global:finite": "finite" if bool(signature.get("finite", False)) else "nonfinite"
    }
    for axis, payload in sorted(signature.get("axes", {}).items()):
        labels[f"axis:{axis}:monotonicity"] = str(
            payload.get("monotonic_class", "missing")
        )
        labels[f"axis:{axis}:curvature"] = str(
            payload.get("curvature_class", "missing")
        )
        labels[f"axis:{axis}:endpoint_slope"] = str(
            payload.get("endpoint_slope_class", "missing")
        )
    for pair, payload in sorted(signature.get("interactions", {}).items()):
        labels[f"interaction:{pair}"] = str(
            payload.get("interaction_class", "missing")
        )
    return labels


def response_family_match(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two response signatures without using response magnitudes."""

    reference_labels = response_family_labels(reference)
    candidate_labels = response_family_labels(candidate)
    label_names = tuple(sorted(set(reference_labels) | set(candidate_labels)))
    mismatches = tuple(
        name
        for name in label_names
        if reference_labels.get(name, "missing")
        != candidate_labels.get(name, "missing")
    )
    compared_count = len(label_names)
    matching_count = compared_count - len(mismatches)
    match_fraction = (
        float(matching_count / compared_count) if compared_count else 1.0
    )
    component_matches: dict[str, dict[str, int]] = {}
    for name in label_names:
        component = name.split(":", 1)[0]
        counters = component_matches.setdefault(
            component,
            {"matching_count": 0, "compared_count": 0},
        )
        counters["compared_count"] += 1
        if name not in mismatches:
            counters["matching_count"] += 1
    return {
        "exact_family_match": not mismatches,
        "matching_label_count": matching_count,
        "compared_label_count": compared_count,
        "match_fraction": match_fraction,
        "mismatched_labels": list(mismatches),
        "reference_labels": reference_labels,
        "candidate_labels": candidate_labels,
        "component_matches": component_matches,
    }


def registered_contract_compliance(
    task: Gate1Task,
    signature: Mapping[str, Any],
    *,
    monotonic_fraction: float = 0.95,
) -> dict[str, bool]:
    compliance = {"finite_output": bool(signature.get("finite", False))}
    minimum_output = task.definition.minimum_output
    if minimum_output is not None:
        compliance["output_lower_bound"] = bool(
            float(signature.get("audit_prediction_minimum", float("-inf")))
            >= float(minimum_output) - 1.0e-8
        )
    for axis, direction in task.definition.monotonicity.items():
        axis_signature = signature["axes"][axis]
        fraction = (
            axis_signature["increasing_fraction"]
            if direction == "increasing"
            else axis_signature["decreasing_fraction"]
        )
        compliance[f"monotonicity:{axis}:{direction}"] = bool(
            float(fraction) >= monotonic_fraction
        )
    return compliance

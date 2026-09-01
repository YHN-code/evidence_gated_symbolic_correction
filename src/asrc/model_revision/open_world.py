from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import chi2, norm

from asrc.model_revision.ast import (
    ExpressionEvaluationError,
    evaluate_expression,
    expression_node_count,
    parameter_names,
)
from asrc.model_revision.benchmarks import Gate1Task, _residual_evidence
from asrc.model_revision.evaluation import evaluate_candidates, predict_repair
from asrc.model_revision.proposals import RepairRequest, TypedRepair


@dataclass(frozen=True)
class ValidationEvidenceSplit:
    selection: pd.DataFrame
    adequacy: pd.DataFrame


@dataclass(frozen=True)
class AdequacyEvidence:
    observation_count: int
    standardized_sum_of_squares: float
    normalized_rmse: float
    p_value: float
    significance_level: float
    adequate: bool


@dataclass(frozen=True)
class ArchiveUpdate:
    repairs: tuple[TypedRepair, ...]
    added: tuple[TypedRepair, ...]
    duplicate_count: int
    budget_rejected_count: int


@dataclass(frozen=True)
class PrequentialSurprise:
    observed: float
    predicted: float
    residual: float
    standardized_residual: float
    raw_two_sided_p_value: float
    adjusted_p_value: float
    familywise_significance_level: float
    comparison_count: int
    critical_standardized_residual: float
    surprising: bool


@dataclass(frozen=True)
class DiverseProposalSelection:
    selected: tuple[TypedRepair, ...]
    diagnostics: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ArchiveReplacement:
    repairs: tuple[TypedRepair, ...]
    added: tuple[TypedRepair, ...]
    evicted: tuple[TypedRepair, ...]
    diagnostics: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FunctionalFamilyAudit:
    rmse: float
    relative_rmse: float
    tolerance: float
    representable: bool
    evaluation_failed: bool
    parameter_values: dict[str, float]


_STRUCTURAL_OPERATORS = (
    "baseline",
    "variable",
    "parameter",
    "constant",
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "negate",
    "abs",
    "exp",
    "log",
    "sin",
    "cos",
    "tanh",
)
_EDIT_TYPES = (
    "add_term",
    "multiply_term",
    "replace_subtree",
    "add_state_dependence",
    "add_bounded_transition",
)


def split_validation_evidence(
    task: Gate1Task,
    *,
    selection_fraction: float,
    seed: int,
) -> ValidationEvidenceSplit:
    """Create model-selection and independent adequacy subsets once per task."""

    if not 0.25 <= selection_fraction <= 0.75:
        raise ValueError("selection_fraction must be in [0.25, 0.75].")
    validation = (
        task.observed.loc[task.observed["partition"].eq("validation")]
        .copy()
        .reset_index(drop=True)
    )
    if len(validation) < 8:
        raise ValueError("At least eight validation observations are required.")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(validation))
    selection_count = int(round(len(validation) * selection_fraction))
    selection_count = min(len(validation) - 4, max(4, selection_count))
    selection = validation.iloc[order[:selection_count]].copy().reset_index(drop=True)
    adequacy = validation.iloc[order[selection_count:]].copy().reset_index(drop=True)
    selection["partition"] = "validation"
    adequacy["partition"] = "adequacy"
    return ValidationEvidenceSplit(selection=selection, adequacy=adequacy)


def refresh_task_evidence(
    task: Gate1Task,
    fit: pd.DataFrame,
    selection_validation: pd.DataFrame,
    *,
    maximum_candidates: int,
    failed_structural_keys: Iterable[str] = (),
) -> Gate1Task:
    """Rebuild the public fit-only evidence after acquiring observations."""

    active_fit = fit.copy().reset_index(drop=True)
    active_fit["partition"] = "fit"
    validation = selection_validation.copy().reset_index(drop=True)
    validation["partition"] = "validation"
    observed = pd.concat([active_fit, validation], ignore_index=True)
    evidence = _residual_evidence(observed, task.variables)
    prior = task.request.residual_evidence
    public = {
        key: prior[key]
        for key in ("task_id", "target_kind", "variables")
        if key in prior
    }
    request = RepairRequest(
        baseline_expression=task.baseline_expression,
        residual_evidence={**public, **evidence},
        constraints=task.request.constraints,
        failed_structural_keys=frozenset(str(value) for value in failed_structural_keys),
        maximum_candidates=int(maximum_candidates),
    )
    return replace(task, observed=observed, request=request)


def known_noise_adequacy_test(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    noise_std: float,
    significance_level: float,
) -> AdequacyEvidence:
    """Test whether independent residual energy is compatible with known noise."""

    truth = np.asarray(target, dtype=float)
    estimate = np.asarray(prediction, dtype=float)
    if truth.ndim != 1 or truth.shape != estimate.shape or len(truth) < 1:
        raise ValueError("Target and prediction must be equal non-empty vectors.")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(estimate)):
        raise ValueError("Adequacy inputs must be finite.")
    if noise_std <= 0.0 or not np.isfinite(noise_std):
        raise ValueError("noise_std must be finite and positive.")
    if not 0.0 < significance_level < 1.0:
        raise ValueError("significance_level must lie in (0, 1).")
    standardized = (truth - estimate) / float(noise_std)
    statistic = float(np.sum(standardized**2))
    p_value = float(chi2.sf(statistic, df=len(standardized)))
    return AdequacyEvidence(
        observation_count=len(standardized),
        standardized_sum_of_squares=statistic,
        normalized_rmse=float(np.sqrt(np.mean(standardized**2))),
        p_value=p_value,
        significance_level=float(significance_level),
        adequate=bool(p_value >= float(significance_level)),
    )


def known_noise_prequential_surprise(
    observed: float,
    predicted: float,
    *,
    noise_std: float,
    familywise_significance_level: float,
    comparison_count: int,
) -> PrequentialSurprise:
    """Test a response revealed after its design point was selected.

    Bonferroni correction keeps the false-trigger probability bounded across the
    registered acquisition budget. The result is suitable for triggering a
    later search round; it must not be computed from locked evaluation data.
    """

    if not np.isfinite(observed) or not np.isfinite(predicted):
        raise ValueError("Prequential observations and predictions must be finite.")
    if noise_std <= 0.0 or not np.isfinite(noise_std):
        raise ValueError("noise_std must be finite and positive.")
    if not 0.0 < familywise_significance_level < 1.0:
        raise ValueError("familywise_significance_level must lie in (0, 1).")
    if int(comparison_count) < 1:
        raise ValueError("comparison_count must be positive.")
    residual = float(observed) - float(predicted)
    standardized = residual / float(noise_std)
    raw_p_value = float(2.0 * norm.sf(abs(standardized)))
    adjusted_p_value = min(1.0, raw_p_value * int(comparison_count))
    per_comparison_alpha = float(familywise_significance_level) / int(
        comparison_count
    )
    critical = float(norm.isf(per_comparison_alpha / 2.0))
    return PrequentialSurprise(
        observed=float(observed),
        predicted=float(predicted),
        residual=residual,
        standardized_residual=float(standardized),
        raw_two_sided_p_value=raw_p_value,
        adjusted_p_value=adjusted_p_value,
        familywise_significance_level=float(familywise_significance_level),
        comparison_count=int(comparison_count),
        critical_standardized_residual=critical,
        surprising=bool(adjusted_p_value < float(familywise_significance_level)),
    )


def _visit_expression(
    expression: Mapping[str, Any],
    visitor: Any,
    *,
    depth: int = 1,
) -> int:
    visitor(expression, depth)
    operation = str(expression["op"])
    children: list[Mapping[str, Any]] = []
    if operation in {"negate", "abs", "exp", "log", "sin", "cos", "tanh"}:
        children = [expression["argument"]]
    elif operation in {"subtract", "divide", "power"}:
        children = [expression["left"], expression["right"]]
    elif operation in {"add", "multiply"}:
        children = list(expression["arguments"])
    maximum_depth = depth
    for child in children:
        maximum_depth = max(
            maximum_depth,
            _visit_expression(child, visitor, depth=depth + 1),
        )
    return maximum_depth


def _structural_vector(repair: TypedRepair, variable_names: tuple[str, ...]) -> np.ndarray:
    operator_counts = {name: 0.0 for name in _STRUCTURAL_OPERATORS}
    variable_counts = {name: 0.0 for name in variable_names}

    def count(node: Mapping[str, Any], _depth: int) -> None:
        operation = str(node["op"])
        if operation in operator_counts:
            operator_counts[operation] += 1.0
        if operation == "variable" and str(node["name"]) in variable_counts:
            variable_counts[str(node["name"])] += 1.0

    depth = _visit_expression(repair.expression, count)
    values = [operator_counts[name] for name in _STRUCTURAL_OPERATORS]
    values.extend(variable_counts[name] for name in variable_names)
    values.extend(float(repair.edit_type == name) for name in _EDIT_TYPES)
    values.extend(
        [
            float(expression_node_count(repair.expression)) / 48.0,
            float(depth) / 8.0,
            float(len(parameter_names(repair.expression))) / 8.0,
        ]
    )
    vector = np.asarray(values, dtype=float)
    norm_value = float(np.linalg.norm(vector))
    return vector / norm_value if norm_value > 0.0 else vector


def _typed_path_fingerprint(
    repair: TypedRepair,
    variable_names: tuple[str, ...],
) -> Counter[str]:
    """Represent an expression by typed root paths and parent-child relations.

    Numeric constants and parameter names are intentionally anonymized. The
    fingerprint retains whether a child is fixed or learnable and where it
    occurs in the tree, which distinguishes structurally different model
    families without using response targets or fitted parameter values.
    """

    fingerprint: Counter[str] = Counter()
    variable_set = set(variable_names)

    def visit(
        node: Mapping[str, Any],
        path: tuple[str, ...],
        parent_operation: str | None,
        child_role: str,
    ) -> None:
        operation = str(node["op"])
        typed_operation = operation
        if operation == "variable":
            name = str(node["name"])
            typed_operation = f"variable:{name if name in variable_set else 'other'}"
        elif operation == "parameter":
            typed_operation = "parameter:free"
        elif operation == "constant":
            typed_operation = "constant:fixed"

        current_path = (*path, f"{child_role}:{typed_operation}")
        fingerprint[f"path:{'/'.join(current_path)}"] += 1
        if parent_operation is not None:
            fingerprint[
                f"edge:{parent_operation}:{child_role}:{typed_operation}"
            ] += 1

        if operation in {"negate", "abs", "exp", "log", "sin", "cos", "tanh"}:
            visit(node["argument"], current_path, operation, "argument")
        elif operation in {"subtract", "divide", "power"}:
            visit(node["left"], current_path, operation, "left")
            visit(node["right"], current_path, operation, "right")
        elif operation in {"add", "multiply"}:
            for child in node["arguments"]:
                visit(child, current_path, operation, "argument")

    visit(repair.expression, (), None, "root")
    fingerprint[f"edit_type:{repair.edit_type}"] += 1
    return fingerprint


def _counter_cosine_distance(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    shared = set(left) & set(right)
    dot = sum(float(left[key]) * float(right[key]) for key in shared)
    left_norm = float(np.sqrt(sum(float(value) ** 2 for value in left.values())))
    right_norm = float(np.sqrt(sum(float(value) ** 2 for value in right.values())))
    if left_norm <= 1.0e-12 and right_norm <= 1.0e-12:
        return 0.0
    if left_norm <= 1.0e-12 or right_norm <= 1.0e-12:
        return 1.0
    similarity = dot / (left_norm * right_norm)
    return float(np.clip(1.0 - similarity, 0.0, 1.0))


def _behavior_vector(task: Gate1Task, repair: TypedRepair) -> np.ndarray:
    frame = task.audit
    baseline = task.baseline_expression
    names = parameter_names(repair.expression)
    probes, _weights = np.polynomial.legendre.leggauss(4)
    signatures: list[np.ndarray] = []
    for probe_index in range(len(probes)):
        parameters = {
            name: float(probes[(probe_index + index) % len(probes)])
            for index, name in enumerate(names)
        }
        try:
            prediction = predict_repair(task, repair, frame, parameters)
            variables = {name: frame[name].to_numpy(float) for name in task.variables}
            base = evaluate_expression(baseline, variables)
            correction = np.asarray(prediction - base, dtype=float)
            scale = float(np.linalg.norm(correction))
            signatures.append(correction / scale if scale > 1.0e-12 else correction)
        except ExpressionEvaluationError:
            signatures.append(np.zeros(len(frame), dtype=float))
    vector = np.concatenate(signatures)
    norm_value = float(np.linalg.norm(vector))
    return vector / norm_value if norm_value > 0.0 else vector


def _cosine_distance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    sign_invariant: bool,
) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1.0e-12 and right_norm <= 1.0e-12:
        return 0.0
    if left_norm <= 1.0e-12 or right_norm <= 1.0e-12:
        return 1.0
    similarity = float(np.dot(left, right) / (left_norm * right_norm))
    if sign_invariant:
        similarity = abs(similarity)
    return float(np.clip(1.0 - similarity, 0.0, 2.0))


def select_diverse_repairs(
    task: Gate1Task,
    existing: Iterable[TypedRepair],
    proposed: Iterable[TypedRepair],
    *,
    count: int,
    diversity_weight: float,
    structural_weight: float,
) -> DiverseProposalSelection:
    """Select a source-neutral, target-free diverse proposal subset.

    Candidate quality is restricted to parsimony and typed-contract validity.
    Structural and output-shape fingerprints use no response values, so the
    proposal pool does not consume the fitted/validated candidate budget.
    """

    if int(count) < 0:
        raise ValueError("count must be non-negative.")
    if not 0.0 <= float(diversity_weight) <= 1.0:
        raise ValueError("diversity_weight must lie in [0, 1].")
    if not 0.0 <= float(structural_weight) <= 1.0:
        raise ValueError("structural_weight must lie in [0, 1].")
    existing_repairs = tuple(existing)
    seen = {repair.structural_key for repair in existing_repairs}
    candidates: list[TypedRepair] = []
    for repair in proposed:
        if repair.structural_key in seen:
            continue
        seen.add(repair.structural_key)
        candidates.append(repair)
    if not candidates or count == 0:
        return DiverseProposalSelection((), ())

    variable_names = tuple(task.variables)
    all_repairs = (*existing_repairs, *candidates)
    representations = {
        repair.structural_key: (
            _structural_vector(repair, variable_names),
            _behavior_vector(task, repair),
        )
        for repair in all_repairs
    }

    def distance(left: TypedRepair, right: TypedRepair) -> tuple[float, float, float]:
        left_structure, left_behavior = representations[left.structural_key]
        right_structure, right_behavior = representations[right.structural_key]
        structural = _cosine_distance(
            left_structure,
            right_structure,
            sign_invariant=False,
        )
        behavioral = _cosine_distance(
            left_behavior,
            right_behavior,
            sign_invariant=True,
        )
        combined = (
            float(structural_weight) * structural
            + (1.0 - float(structural_weight)) * behavioral
        )
        return combined, structural, behavioral

    selected: list[TypedRepair] = []
    rank_metadata: dict[str, dict[str, Any]] = {}
    remaining = list(candidates)
    maximum_nodes = max(expression_node_count(item.expression) for item in candidates)
    while remaining and len(selected) < min(int(count), len(candidates)):
        references = (*existing_repairs, *selected)
        scored: list[tuple[tuple[float, float, float, str], TypedRepair, dict[str, Any]]] = []
        for repair in remaining:
            if references:
                distances = [distance(repair, reference) for reference in references]
                nearest = min(distances, key=lambda item: item[0])
            else:
                nearest = (1.0, 1.0, 1.0)
            complexity = expression_node_count(repair.expression)
            parsimony = 1.0 - (float(complexity - 1) / max(1.0, maximum_nodes - 1.0))
            score = (
                float(diversity_weight) * nearest[0]
                + (1.0 - float(diversity_weight)) * parsimony
            )
            metadata = {
                "proposal_pool_score": float(score),
                "nearest_combined_distance": float(nearest[0]),
                "nearest_structural_distance": float(nearest[1]),
                "nearest_behavioral_distance": float(nearest[2]),
                "parsimony_score": float(parsimony),
            }
            key = (-score, -nearest[0], -parsimony, repair.structural_key)
            scored.append((key, repair, metadata))
        _key, chosen, metadata = min(scored, key=lambda item: item[0])
        selected.append(chosen)
        rank_metadata[chosen.structural_key] = {
            **metadata,
            "selected_rank": len(selected),
        }
        remaining = [item for item in remaining if item.structural_key != chosen.structural_key]

    final_references = (*existing_repairs, *selected)
    diagnostics: list[dict[str, Any]] = []
    for repair in candidates:
        metadata = rank_metadata.get(repair.structural_key)
        if metadata is None:
            distances = [
                distance(repair, reference)
                for reference in final_references
                if reference.structural_key != repair.structural_key
            ]
            nearest = min(distances, key=lambda item: item[0]) if distances else (1.0, 1.0, 1.0)
            complexity = expression_node_count(repair.expression)
            parsimony = 1.0 - (float(complexity - 1) / max(1.0, maximum_nodes - 1.0))
            metadata = {
                "proposal_pool_score": float(
                    float(diversity_weight) * nearest[0]
                    + (1.0 - float(diversity_weight)) * parsimony
                ),
                "nearest_combined_distance": float(nearest[0]),
                "nearest_structural_distance": float(nearest[1]),
                "nearest_behavioral_distance": float(nearest[2]),
                "parsimony_score": float(parsimony),
                "selected_rank": None,
            }
        diagnostics.append(
            {
                "proposal_id": repair.proposal_id,
                "structural_key": repair.structural_key,
                "source": repair.source,
                "selected": repair.structural_key in rank_metadata,
                **metadata,
            }
        )
    return DiverseProposalSelection(tuple(selected), tuple(diagnostics))


def select_facility_location_repairs(
    task: Gate1Task,
    existing: Iterable[TypedRepair],
    proposed: Iterable[TypedRepair],
    *,
    count: int,
    structural_weight: float,
    similarity_temperature: float,
    parsimony_exponent: float,
    source_mass: Mapping[str, float] | None = None,
    structural_fingerprint: str = "operator_counts",
    similarity_aggregation: str = "weighted_mean",
    coverage_utility: str = "linear",
) -> DiverseProposalSelection:
    """Select a target-free subset by weighted facility-location coverage.

    The clients and selectable facilities are the typed candidate structures.
    Similarity combines expression structure with parameter-probed correction
    shapes on the audit inputs; neither observed responses nor fitted candidate
    scores enter the objective. Optional ``source_mass`` values allocate prior
    probability mass to proposal sources without imposing hard source quotas.
    """

    if int(count) < 0:
        raise ValueError("count must be non-negative.")
    if not 0.0 <= float(structural_weight) <= 1.0:
        raise ValueError("structural_weight must lie in [0, 1].")
    if float(similarity_temperature) <= 0.0:
        raise ValueError("similarity_temperature must be positive.")
    if float(parsimony_exponent) < 0.0:
        raise ValueError("parsimony_exponent must be non-negative.")
    if structural_fingerprint not in {"operator_counts", "typed_paths"}:
        raise ValueError(
            "structural_fingerprint must be operator_counts or typed_paths."
        )
    if similarity_aggregation not in {"weighted_mean", "conjunctive"}:
        raise ValueError(
            "similarity_aggregation must be weighted_mean or conjunctive."
        )
    if coverage_utility not in {"linear", "square_root"}:
        raise ValueError("coverage_utility must be linear or square_root.")
    if source_mass is not None:
        if not source_mass:
            raise ValueError("source_mass must not be empty when provided.")
        if any(float(value) < 0.0 for value in source_mass.values()):
            raise ValueError("source_mass values must be non-negative.")
        if sum(float(value) for value in source_mass.values()) <= 0.0:
            raise ValueError("source_mass must contain positive total mass.")

    existing_repairs = tuple(existing)
    seen = {repair.structural_key for repair in existing_repairs}
    candidates: list[TypedRepair] = []
    for repair in proposed:
        if repair.structural_key in seen:
            continue
        seen.add(repair.structural_key)
        candidates.append(repair)
    if not candidates or count == 0:
        return DiverseProposalSelection((), ())

    source_counts: dict[str, int] = {}
    for repair in candidates:
        source_counts[repair.source] = source_counts.get(repair.source, 0) + 1
    if source_mass is None:
        base_weights = {
            repair.structural_key: 1.0 / float(len(candidates))
            for repair in candidates
        }
    else:
        active_mass = sum(
            float(source_mass.get(source, 0.0)) for source in source_counts
        )
        if active_mass <= 0.0:
            raise ValueError(
                "source_mass assigns no positive mass to available proposal sources."
            )
        base_weights = {
            repair.structural_key: (
                float(source_mass.get(repair.source, 0.0))
                / active_mass
                / float(source_counts[repair.source])
            )
            for repair in candidates
        }

    raw_client_weights = {
        repair.structural_key: base_weights[repair.structural_key]
        / (1.0 + float(expression_node_count(repair.expression)))
        ** float(parsimony_exponent)
        for repair in candidates
    }
    total_client_weight = sum(raw_client_weights.values())
    if total_client_weight <= 0.0:
        raise ValueError("facility-location client weights have zero total mass.")
    client_weights = {
        key: value / total_client_weight
        for key, value in raw_client_weights.items()
    }

    variable_names = tuple(task.variables)
    all_repairs = (*existing_repairs, *candidates)
    representations = {
        repair.structural_key: (
            _typed_path_fingerprint(repair, variable_names)
            if structural_fingerprint == "typed_paths"
            else _structural_vector(repair, variable_names),
            _behavior_vector(task, repair),
        )
        for repair in all_repairs
    }

    def similarity(left: TypedRepair, right: TypedRepair) -> float:
        left_structure, left_behavior = representations[left.structural_key]
        right_structure, right_behavior = representations[right.structural_key]
        structural = (
            _counter_cosine_distance(left_structure, right_structure)
            if structural_fingerprint == "typed_paths"
            else _cosine_distance(
                left_structure,
                right_structure,
                sign_invariant=False,
            )
        )
        behavioral = _cosine_distance(
            left_behavior,
            right_behavior,
            sign_invariant=True,
        )
        distance = (
            max(structural, behavioral)
            if similarity_aggregation == "conjunctive"
            else (
                float(structural_weight) * structural
                + (1.0 - float(structural_weight)) * behavioral
            )
        )
        return float(np.exp(-distance / float(similarity_temperature)))

    client_similarity = {
        client.structural_key: {
            facility.structural_key: similarity(client, facility)
            for facility in all_repairs
        }
        for client in candidates
    }

    def coverage_value(value: float) -> float:
        bounded = float(np.clip(value, 0.0, 1.0))
        return float(np.sqrt(bounded)) if coverage_utility == "square_root" else bounded

    current_coverage = {
        client.structural_key: max(
            (
                client_similarity[client.structural_key][repair.structural_key]
                for repair in existing_repairs
            ),
            default=0.0,
        )
        for client in candidates
    }

    selected: list[TypedRepair] = []
    rank_metadata: dict[str, dict[str, Any]] = {}
    remaining = list(candidates)
    while remaining and len(selected) < min(int(count), len(candidates)):
        scored: list[tuple[tuple[float, int, str], TypedRepair, float]] = []
        for facility in remaining:
            marginal_gain = sum(
                client_weights[client.structural_key]
                * (
                    coverage_value(
                        max(
                            current_coverage[client.structural_key],
                            client_similarity[client.structural_key][
                                facility.structural_key
                            ],
                        )
                    )
                    - coverage_value(current_coverage[client.structural_key])
                )
                for client in candidates
            )
            key = (
                -float(marginal_gain),
                expression_node_count(facility.expression),
                facility.structural_key,
            )
            scored.append((key, facility, float(marginal_gain)))
        _key, chosen, marginal_gain = min(scored, key=lambda item: item[0])
        selected.append(chosen)
        for client in candidates:
            current_coverage[client.structural_key] = max(
                current_coverage[client.structural_key],
                client_similarity[client.structural_key][chosen.structural_key],
            )
        rank_metadata[chosen.structural_key] = {
            "selected_rank": len(selected),
            "marginal_coverage_gain": marginal_gain,
            "coverage_after_selection": float(
                sum(
                    client_weights[client.structural_key]
                    * coverage_value(current_coverage[client.structural_key])
                    for client in candidates
                )
            ),
        }
        remaining = [
            repair
            for repair in remaining
            if repair.structural_key != chosen.structural_key
        ]

    selected_keys = {repair.structural_key for repair in selected}
    final_facilities = (*existing_repairs, *selected)
    diagnostics: list[dict[str, Any]] = []
    for repair in candidates:
        final_similarity = max(
            (
                client_similarity[repair.structural_key][facility.structural_key]
                for facility in final_facilities
            ),
            default=0.0,
        )
        diagnostics.append(
            {
                "proposal_id": repair.proposal_id,
                "structural_key": repair.structural_key,
                "source": repair.source,
                "selected": repair.structural_key in selected_keys,
                "selected_rank": rank_metadata.get(repair.structural_key, {}).get(
                    "selected_rank"
                ),
                "marginal_coverage_gain": rank_metadata.get(
                    repair.structural_key, {}
                ).get("marginal_coverage_gain"),
                "coverage_after_selection": rank_metadata.get(
                    repair.structural_key, {}
                ).get("coverage_after_selection"),
                "client_weight": float(client_weights[repair.structural_key]),
                "final_coverage_similarity": float(final_similarity),
                "complexity": expression_node_count(repair.expression),
                "structural_fingerprint": structural_fingerprint,
                "similarity_aggregation": similarity_aggregation,
                "coverage_utility": coverage_utility,
            }
        )
    return DiverseProposalSelection(tuple(selected), tuple(diagnostics))


def select_quality_diverse_repairs(
    task: Gate1Task,
    existing: Iterable[TypedRepair],
    proposed: Iterable[TypedRepair],
    *,
    count: int,
    quality_weight: float,
    structural_weight: float,
    elite_count: int,
    evaluation_config: Mapping[str, Any],
    seed: int,
) -> DiverseProposalSelection:
    """Select fitted candidates with quality elitism and diversity fill.

    Only the active fit and model-selection evidence in ``task.observed`` is
    used. Independent adequacy and locked responses are unavailable here.
    Every novel proposal is fitted and therefore counts toward the explicit
    screening budget, irrespective of whether it enters the active archive.
    """

    if int(count) < 0 or int(elite_count) < 0:
        raise ValueError("count and elite_count must be non-negative.")
    if not 0.0 <= float(quality_weight) <= 1.0:
        raise ValueError("quality_weight must lie in [0, 1].")
    if not 0.0 <= float(structural_weight) <= 1.0:
        raise ValueError("structural_weight must lie in [0, 1].")
    existing_repairs = tuple(existing)
    seen = {repair.structural_key for repair in existing_repairs}
    candidates: list[TypedRepair] = []
    for repair in proposed:
        if repair.structural_key in seen:
            continue
        seen.add(repair.structural_key)
        candidates.append(repair)
    if not candidates or count == 0:
        return DiverseProposalSelection((), ())

    evaluations = evaluate_candidates(
        task,
        candidates,
        method="open_world_quality_screen",
        seed=int(seed),
        evaluation_config=evaluation_config,
    )
    evaluation_by_key = {
        evaluation.structural_key: evaluation for evaluation in evaluations
    }
    valid = [
        repair
        for repair in candidates
        if evaluation_by_key[repair.structural_key].status == "valid"
    ]
    ordered = sorted(
        valid,
        key=lambda repair: (
            evaluation_by_key[repair.structural_key].selection_score,
            evaluation_by_key[repair.structural_key].complexity,
            repair.structural_key,
        ),
    )
    rank_quality: dict[str, float] = {}
    denominator = max(1, len(ordered) - 1)
    for rank, repair in enumerate(ordered):
        rank_quality[repair.structural_key] = 1.0 - float(rank) / denominator

    variable_names = tuple(task.variables)
    all_repairs = (*existing_repairs, *candidates)
    representations = {
        repair.structural_key: (
            _structural_vector(repair, variable_names),
            _behavior_vector(task, repair),
        )
        for repair in all_repairs
    }

    def distance(left: TypedRepair, right: TypedRepair) -> tuple[float, float, float]:
        left_structure, left_behavior = representations[left.structural_key]
        right_structure, right_behavior = representations[right.structural_key]
        structural = _cosine_distance(
            left_structure,
            right_structure,
            sign_invariant=False,
        )
        behavioral = _cosine_distance(
            left_behavior,
            right_behavior,
            sign_invariant=True,
        )
        combined = (
            float(structural_weight) * structural
            + (1.0 - float(structural_weight)) * behavioral
        )
        return combined, structural, behavioral

    selected = ordered[: min(int(elite_count), int(count), len(ordered))]
    metadata: dict[str, dict[str, Any]] = {}
    for rank, repair in enumerate(selected, start=1):
        metadata[repair.structural_key] = {
            "selection_phase": "quality_elite",
            "selected_rank": rank,
            "quality_rank_score": rank_quality[repair.structural_key],
            "nearest_combined_distance": None,
            "nearest_structural_distance": None,
            "nearest_behavioral_distance": None,
            "quality_diversity_score": rank_quality[repair.structural_key],
        }

    remaining = [item for item in ordered if item not in selected]
    while remaining and len(selected) < min(int(count), len(ordered)):
        references = (*existing_repairs, *selected)
        scored = []
        for repair in remaining:
            if references:
                nearest = min(
                    (distance(repair, reference) for reference in references),
                    key=lambda item: item[0],
                )
            else:
                nearest = (1.0, 1.0, 1.0)
            diversity = float(np.clip(nearest[0], 0.0, 1.0))
            quality = rank_quality[repair.structural_key]
            score = float(quality_weight) * quality + (
                1.0 - float(quality_weight)
            ) * diversity
            scored.append(
                (
                    (-score, -quality, -diversity, repair.structural_key),
                    repair,
                    nearest,
                    score,
                )
            )
        _key, chosen, nearest, score = min(scored, key=lambda item: item[0])
        selected.append(chosen)
        metadata[chosen.structural_key] = {
            "selection_phase": "quality_diversity_fill",
            "selected_rank": len(selected),
            "quality_rank_score": rank_quality[chosen.structural_key],
            "nearest_combined_distance": float(nearest[0]),
            "nearest_structural_distance": float(nearest[1]),
            "nearest_behavioral_distance": float(nearest[2]),
            "quality_diversity_score": float(score),
        }
        remaining = [item for item in remaining if item is not chosen]

    selected_keys = {item.structural_key for item in selected}
    diagnostics = []
    for repair in candidates:
        evaluation = evaluation_by_key[repair.structural_key]
        diagnostics.append(
            {
                "proposal_id": repair.proposal_id,
                "structural_key": repair.structural_key,
                "source": repair.source,
                "selected": repair.structural_key in selected_keys,
                "fit_status": evaluation.status,
                "validation_rmse": evaluation.validation_rmse,
                "selection_score": evaluation.selection_score,
                "complexity": evaluation.complexity,
                "parameter_count": evaluation.parameter_count,
                **metadata.get(
                    repair.structural_key,
                    {
                        "selection_phase": "not_selected",
                        "selected_rank": None,
                        "quality_rank_score": rank_quality.get(
                            repair.structural_key,
                            0.0,
                        ),
                        "nearest_combined_distance": None,
                        "nearest_structural_distance": None,
                        "nearest_behavioral_distance": None,
                        "quality_diversity_score": None,
                    },
                ),
            }
        )
    return DiverseProposalSelection(tuple(selected), tuple(diagnostics))


def replace_quality_diverse_archive(
    task: Gate1Task,
    existing: Iterable[TypedRepair],
    proposed: Iterable[TypedRepair],
    *,
    maximum_candidates: int,
    quality_weight: float,
    structural_weight: float,
    elite_count: int,
    evaluation_config: Mapping[str, Any],
    seed: int,
) -> ArchiveReplacement:
    """Maintain a bounded archive by replacing weak or redundant structures."""

    old = tuple(existing)
    union: list[TypedRepair] = []
    seen: set[str] = set()
    for repair in (*old, *tuple(proposed)):
        if repair.structural_key in seen:
            continue
        seen.add(repair.structural_key)
        union.append(repair)
    selection = select_quality_diverse_repairs(
        task,
        (),
        union,
        count=min(int(maximum_candidates), len(union)),
        quality_weight=float(quality_weight),
        structural_weight=float(structural_weight),
        elite_count=int(elite_count),
        evaluation_config=evaluation_config,
        seed=int(seed),
    )
    retained_keys = {item.structural_key for item in selection.selected}
    old_keys = {item.structural_key for item in old}
    return ArchiveReplacement(
        repairs=selection.selected,
        added=tuple(
            item for item in selection.selected if item.structural_key not in old_keys
        ),
        evicted=tuple(item for item in old if item.structural_key not in retained_keys),
        diagnostics=selection.diagnostics,
    )


def audit_functional_family(
    task: Gate1Task,
    repair: TypedRepair | None,
    *,
    parameter_lower_bound: float,
    parameter_upper_bound: float,
    multistart_count: int,
    maximum_function_evaluations: int,
    relative_rmse_tolerance: float,
    seed: int,
) -> FunctionalFamilyAudit:
    """Audit whether a selected family can represent the noise-free oracle.

    This deliberately uses the locked oracle response and therefore is an
    evaluation-only diagnostic. Its fitted parameters must never be returned to
    candidate selection, acquisition, stopping, or proposal generation.
    """

    if parameter_lower_bound >= parameter_upper_bound:
        raise ValueError("Functional-family parameter bounds are invalid.")
    if int(multistart_count) < 1 or int(maximum_function_evaluations) < 1:
        raise ValueError("Functional-family optimizer budgets must be positive.")
    if relative_rmse_tolerance <= 0.0:
        raise ValueError("relative_rmse_tolerance must be positive.")
    target = task.locked["target"].to_numpy(float)
    baseline = task.locked["baseline"].to_numpy(float)
    correction_scale = max(
        float(np.sqrt(np.mean((target - baseline) ** 2))),
        np.finfo(float).eps,
    )
    if repair is None:
        rmse = float(np.sqrt(np.mean((target - baseline) ** 2)))
        relative = rmse / correction_scale
        return FunctionalFamilyAudit(
            rmse=rmse,
            relative_rmse=relative,
            tolerance=float(relative_rmse_tolerance),
            representable=bool(relative <= float(relative_rmse_tolerance)),
            evaluation_failed=False,
            parameter_values={},
        )

    names = parameter_names(repair.expression)

    def residual(vector: np.ndarray) -> np.ndarray:
        parameters = {name: float(vector[index]) for index, name in enumerate(names)}
        try:
            return predict_repair(task, repair, task.locked, parameters) - target
        except ExpressionEvaluationError:
            return np.full(len(target), 1.0e6, dtype=float)

    best_vector = np.zeros(len(names), dtype=float)
    best_rmse = float("inf")
    if names:
        rng = np.random.default_rng(int(seed))
        starts = [np.zeros(len(names), dtype=float)]
        starts.extend(
            rng.uniform(
                max(float(parameter_lower_bound), -2.0),
                min(float(parameter_upper_bound), 2.0),
                len(names),
            )
            for _ in range(int(multistart_count) - 1)
        )
        for start in starts:
            try:
                result = least_squares(
                    residual,
                    np.clip(
                        start,
                        float(parameter_lower_bound) + 1.0e-10,
                        float(parameter_upper_bound) - 1.0e-10,
                    ),
                    bounds=(float(parameter_lower_bound), float(parameter_upper_bound)),
                    max_nfev=int(maximum_function_evaluations),
                    xtol=1.0e-11,
                    ftol=1.0e-11,
                    gtol=1.0e-11,
                )
            except (ValueError, FloatingPointError):
                continue
            candidate_rmse = float(np.sqrt(np.mean(residual(result.x) ** 2)))
            if np.isfinite(candidate_rmse) and candidate_rmse < best_rmse:
                best_rmse = candidate_rmse
                best_vector = np.asarray(result.x, dtype=float)
    else:
        try:
            best_rmse = float(np.sqrt(np.mean(residual(np.empty(0)) ** 2)))
        except (ValueError, FloatingPointError):
            best_rmse = float("inf")
    failed = not np.isfinite(best_rmse)
    relative = best_rmse / correction_scale if not failed else float("inf")
    return FunctionalFamilyAudit(
        rmse=best_rmse,
        relative_rmse=relative,
        tolerance=float(relative_rmse_tolerance),
        representable=bool(not failed and relative <= float(relative_rmse_tolerance)),
        evaluation_failed=failed,
        parameter_values={
            name: float(best_vector[index]) for index, name in enumerate(names)
        },
    )


def update_candidate_archive(
    existing: Iterable[TypedRepair],
    proposed: Iterable[TypedRepair],
    *,
    maximum_candidates: int,
) -> ArchiveUpdate:
    """Append structurally new repairs without exceeding the unique budget."""

    if maximum_candidates < 1:
        raise ValueError("maximum_candidates must be positive.")
    repairs = list(existing)
    if len(repairs) > maximum_candidates:
        raise ValueError("Existing archive already exceeds maximum_candidates.")
    seen = {repair.structural_key for repair in repairs}
    added: list[TypedRepair] = []
    duplicates = 0
    budget_rejected = 0
    for repair in proposed:
        if repair.structural_key in seen:
            duplicates += 1
            continue
        if len(repairs) >= maximum_candidates:
            budget_rejected += 1
            continue
        seen.add(repair.structural_key)
        repairs.append(repair)
        added.append(repair)
    return ArchiveUpdate(
        repairs=tuple(repairs),
        added=tuple(added),
        duplicate_count=duplicates,
        budget_rejected_count=budget_rejected,
    )


def identify_round_repairs(
    repairs: Iterable[TypedRepair],
    *,
    method: str,
    round_index: int,
) -> tuple[TypedRepair, ...]:
    return tuple(
        replace(
            repair,
            proposal_id=(
                f"{method}_r{int(round_index):02d}_{index:03d}"
            ),
        )
        for index, repair in enumerate(repairs, start=1)
    )

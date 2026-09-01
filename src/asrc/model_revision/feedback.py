from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from asrc.model_revision.ast import (
    ExpressionEvaluationError,
    apply_repair,
    evaluate_expression,
)
from asrc.model_revision.benchmarks import Gate1Task
from asrc.model_revision.evaluation import CandidateEvaluation
from asrc.model_revision.proposals import RepairRequest, TypedRepair
from asrc.model_revision.response_signatures import (
    ResponseSignatureConfig,
    candidate_response_signature,
    response_family_labels,
)


def _predict_repair(
    task: Gate1Task,
    repair: TypedRepair,
    frame: pd.DataFrame,
    parameters: Mapping[str, float],
) -> np.ndarray:
    variables = {name: frame[name].to_numpy(float) for name in task.variables}
    baseline = evaluate_expression(task.baseline_expression, variables)
    return apply_repair(repair, baseline, variables, parameters)


def _diagnostic_transforms(
    frame: pd.DataFrame,
    variable_names: Sequence[str],
) -> dict[str, np.ndarray]:
    transforms: dict[str, np.ndarray] = {}
    for name in variable_names:
        values = frame[name].to_numpy(float)
        transforms[name] = values
        transforms[f"{name}_squared"] = values**2
        transforms[f"abs_{name}"] = np.abs(values)
    for left_index, left in enumerate(variable_names):
        for right in variable_names[left_index + 1 :]:
            transforms[f"{left}_times_{right}"] = (
                frame[left].to_numpy(float) * frame[right].to_numpy(float)
            )
    return transforms


def summarize_remaining_fit_residual(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    variable_names: Sequence[str],
    *,
    probe_count: int = 12,
) -> dict[str, Any]:
    residual = frame["target"].to_numpy(float) - np.asarray(prediction, dtype=float)
    correlations: dict[str, float] = {}
    for name, values in _diagnostic_transforms(frame, variable_names).items():
        if float(np.std(values)) <= 1.0e-12 or float(np.std(residual)) <= 1.0e-12:
            correlations[name] = 0.0
        else:
            correlations[name] = float(np.corrcoef(values, residual)[0, 1])

    order = np.lexsort(
        tuple(frame[name].to_numpy(float) for name in reversed(variable_names))
    )
    count = min(max(1, int(probe_count)), len(frame))
    indices = np.unique(np.linspace(0, len(order) - 1, count, dtype=int))
    probes = []
    for ordered_index in indices:
        row_index = int(order[int(ordered_index)])
        row = frame.iloc[row_index]
        probes.append(
            {
                **{name: float(row[name]) for name in variable_names},
                "corrected_prediction": float(prediction[row_index]),
                "remaining_residual": float(residual[row_index]),
            }
        )
    return {
        "fit_count": int(len(frame)),
        "remaining_residual_mean": float(np.mean(residual)),
        "remaining_residual_std": float(np.std(residual)),
        "remaining_residual_quantiles": {
            "q10": float(np.quantile(residual, 0.10)),
            "q50": float(np.quantile(residual, 0.50)),
            "q90": float(np.quantile(residual, 0.90)),
        },
        "remaining_residual_correlations": correlations,
        "fit_only_remaining_residual_probes": probes,
    }


def build_fit_feedback(
    task: Gate1Task,
    evaluations: Sequence[CandidateEvaluation],
    repair_by_id: Mapping[str, TypedRepair],
    *,
    round_index: int,
    top_k: int = 3,
    complexity_penalty_fraction: float = 0.0,
) -> dict[str, Any]:
    fit = task.observed.loc[task.observed["partition"].eq("fit")].reset_index(drop=True)
    baseline_prediction = fit["baseline"].to_numpy(float)
    baseline_fit_rmse = float(
        np.sqrt(
            np.mean(
                (fit["target"].to_numpy(float) - baseline_prediction) ** 2
            )
        )
    )

    def fit_score(item: CandidateEvaluation) -> float:
        return float(
            item.train_rmse
            + float(complexity_penalty_fraction)
            * baseline_fit_rmse
            * item.complexity
        )

    valid = sorted(
        (item for item in evaluations if item.status == "valid"),
        key=lambda item: (fit_score(item), item.complexity, item.proposal_id),
    )
    top = valid[: max(1, int(top_k))]
    feedback: dict[str, Any] = {
        "evidence_scope": "fit_subset_only",
        "completed_round": int(round_index),
        "evaluated_candidate_count": int(len(evaluations)),
        "valid_candidate_count": int(len(valid)),
        "top_candidates_by_fit_score": [
            {
                "proposal_id": item.proposal_id,
                "formula": item.formula,
                "fit_rmse": float(item.train_rmse),
                "fit_selection_score": fit_score(item),
                "complexity": int(item.complexity),
                "parameter_values": dict(item.parameter_values),
                "physical_violation_count": int(item.stability_violations),
            }
            for item in top
        ],
    }
    prediction = baseline_prediction
    incumbent = {
        "source": "baseline",
        "proposal_id": "baseline_no_change",
        "formula": "M_base(x)",
        "fit_rmse": baseline_fit_rmse,
        "fit_selection_score": baseline_fit_rmse,
        "candidate_replaced_baseline": False,
    }
    if top and fit_score(top[0]) < baseline_fit_rmse:
        best = top[0]
        prediction = _predict_repair(
            task,
            repair_by_id[best.proposal_id],
            fit,
            best.parameter_values,
        )
        incumbent = {
            "source": "candidate",
            "proposal_id": best.proposal_id,
            "formula": best.formula,
            "fit_rmse": float(best.train_rmse),
            "fit_selection_score": fit_score(best),
            "candidate_replaced_baseline": True,
        }
    feedback["incumbent"] = incumbent
    feedback["remaining_residual_evidence"] = summarize_remaining_fit_residual(
        fit,
        prediction,
        task.variables,
    )
    feedback["revision_objective"] = (
        "Propose compact structures that explain systematic remaining fit residuals. "
        "The baseline remains the incumbent unless a candidate improves its fit score. "
        "Do not repeat an already seen structure and do not infer held-out evidence."
    )
    return feedback


def request_with_fit_feedback(
    request: RepairRequest,
    feedback: Mapping[str, Any],
    *,
    seen_structural_keys: set[str],
    maximum_candidates: int,
) -> RepairRequest:
    residual_evidence = dict(request.residual_evidence)
    residual_evidence["iterative_fit_feedback"] = dict(feedback)
    return replace(
        request,
        residual_evidence=residual_evidence,
        failed_structural_keys=frozenset(seen_structural_keys),
        maximum_candidates=max(1, int(maximum_candidates)),
    )


def _expression_operators(node: Mapping[str, Any]) -> set[str]:
    operators = {str(node["op"])}
    for value in node.values():
        if isinstance(value, Mapping):
            operators.update(_expression_operators(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    operators.update(_expression_operators(item))
    return operators


def _fit_domain_signature_task(task: Gate1Task) -> Gate1Task:
    """Restrict response-family probes to public fit-domain coordinates."""

    fit = task.observed.loc[task.observed["partition"].eq("fit")].reset_index(
        drop=True
    )
    definition = replace(
        task.definition,
        locked_ranges=dict(task.definition.observed_ranges),
    )
    return replace(task, definition=definition, audit=fit, locked=fit)


def build_coverage_feedback(
    task: Gate1Task,
    evaluations: Sequence[CandidateEvaluation],
    repair_by_id: Mapping[str, TypedRepair],
    *,
    round_index: int,
    top_k: int = 6,
    complexity_penalty_fraction: float = 0.0,
    signature_config: ResponseSignatureConfig | None = None,
) -> dict[str, Any]:
    """Build source-neutral candidate-coverage feedback from fit evidence only.

    The packet reports what the current search has tried, how those candidates
    behave over the observed input domain, and what residual structure remains.
    It never compares a candidate with the oracle or a held-out response.
    """

    feedback = build_fit_feedback(
        task,
        evaluations,
        repair_by_id,
        round_index=round_index,
        top_k=top_k,
        complexity_penalty_fraction=complexity_penalty_fraction,
    )
    status_counts = Counter(str(item.status) for item in evaluations)
    failure_counts = Counter(
        str(item.failure_reason)
        for item in evaluations
        if item.failure_reason
    )
    public_task = _fit_domain_signature_task(task)
    family_counts: Counter[str] = Counter()
    component_counts: dict[str, Counter[str]] = {}
    signature_failures = 0
    valid = [item for item in evaluations if item.status == "valid"]
    for item in valid:
        repair = repair_by_id[item.proposal_id]
        try:
            signature = candidate_response_signature(
                public_task,
                repair,
                item.parameter_values,
                config=signature_config,
            )
        except (ExpressionEvaluationError, ValueError, FloatingPointError):
            signature_failures += 1
            continue
        labels = response_family_labels(signature)
        family_key = json.dumps(labels, sort_keys=True, separators=(",", ":"))
        family_counts[family_key] += 1
        for component, label in labels.items():
            component_counts.setdefault(component, Counter())[str(label)] += 1

    used_operators: Counter[str] = Counter()
    for repair in repair_by_id.values():
        used_operators.update(_expression_operators(repair.expression))
    structural_operators = sorted(
        operator
        for operator in used_operators
        if operator not in {"variable", "parameter", "constant", "baseline"}
    )
    candidate_count = len(evaluations)
    unique_family_count = len(family_counts)
    feedback.update(
        {
            "coverage_feedback_version": "1.0",
            "coverage_evidence_scope": (
                "fit responses and candidate predictions over observed input ranges only"
            ),
            "candidate_status_counts": dict(sorted(status_counts.items())),
            "candidate_failure_counts": dict(sorted(failure_counts.items())),
            "response_family_inventory": {
                "valid_signature_count": int(sum(family_counts.values())),
                "signature_failure_count": int(signature_failures),
                "unique_family_count": int(unique_family_count),
                "family_diversity_fraction": (
                    float(unique_family_count / max(1, sum(family_counts.values())))
                ),
                "family_multiplicities": sorted(
                    family_counts.values(), reverse=True
                ),
                "component_label_counts": {
                    component: dict(sorted(counts.items()))
                    for component, counts in sorted(component_counts.items())
                },
            },
            "structural_search_inventory": {
                "evaluated_candidate_count": int(candidate_count),
                "unique_structural_key_count": int(
                    len({item.structural_key for item in evaluations})
                ),
                "used_structural_operators": structural_operators,
                "operator_usage_counts": {
                    key: int(value) for key, value in sorted(used_operators.items())
                },
            },
            "coverage_revision_objective": (
                "Propose compact structures that address the remaining fit residual "
                "while adding response behavior or operator combinations not already "
                "overrepresented. Do not assume which family is correct and do not "
                "infer held-out responses."
            ),
        }
    )
    return feedback


def derive_pysr_search_config(
    base_config: Mapping[str, Any],
    repairs: Sequence[TypedRepair],
    *,
    niterations: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile typed LLM structures into a bounded PySR operator subspace."""

    configured = dict(base_config)
    base_binary = [str(item) for item in configured.get("binary_operators", ())]
    base_unary = [str(item) for item in configured.get("unary_operators", ())]
    operator_map = {
        "add": "+",
        "subtract": "-",
        "multiply": "*",
        "divide": "/",
        "power": "^",
    }
    proposed = set()
    for repair in repairs:
        proposed.update(_expression_operators(repair.expression))
    if proposed:
        requested_binary = {operator_map[item] for item in proposed if item in operator_map}
        # Addition and multiplication provide the minimal algebraic closure for
        # combining terminals; all other operators must be proposed explicitly.
        requested_binary.update({"+", "*"})
        binary = [item for item in base_binary if item in requested_binary]
        unary = [item for item in base_unary if item in proposed]
        mode = "llm_typed_operator_control"
    else:
        binary = base_binary
        unary = base_unary
        mode = "fallback_full_operator_set"
    if not binary:
        binary = [item for item in base_binary if item in {"+", "*"}]
    configured["binary_operators"] = binary
    configured["unary_operators"] = unary
    configured["niterations"] = int(niterations)
    return configured, {
        "search_control_mode": mode,
        "proposed_ast_operators": sorted(proposed),
        "pysr_binary_operators": binary,
        "pysr_unary_operators": unary,
        "pysr_iterations": int(niterations),
    }


def derive_additive_pysr_search_config(
    base_config: Mapping[str, Any],
    *,
    niterations: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preserve the frozen grammar for feedback-guided candidate expansion.

    Feedback-generated typed repairs are added directly to the candidate pool.
    They cannot remove operators from the independent symbolic search, because
    absence from a small LLM batch is not evidence that an operator is invalid.
    """

    configured = dict(base_config)
    binary = [str(item) for item in configured.get("binary_operators", ())]
    unary = [str(item) for item in configured.get("unary_operators", ())]
    configured["binary_operators"] = binary
    configured["unary_operators"] = unary
    configured["niterations"] = int(niterations)
    return configured, {
        "search_control_mode": "additive_candidate_expansion",
        "base_grammar_preserved": True,
        "pysr_binary_operators": binary,
        "pysr_unary_operators": unary,
        "pysr_iterations": int(niterations),
    }

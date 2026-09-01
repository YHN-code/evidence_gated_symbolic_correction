from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from asrc.model_revision.ambiguity import (
    DeploymentAmbiguity,
    deployment_prediction_ambiguity,
    one_standard_error_equivalence,
)
from asrc.model_revision.ast import (
    ExpressionEvaluationError,
    expression_node_count,
    evaluate_expression,
)
from asrc.model_revision.benchmarks import Gate1Task
from asrc.model_revision.evaluation import CandidateEvaluation, predict_repair
from asrc.model_revision.patch_algebra import extract_structural_atom
from asrc.model_revision.proposals import (
    ProposalBatch,
    RepairContract,
    RepairProposer,
    RepairRequest,
    TypedRepair,
    validate_typed_repair,
)


EVIDENCE_PACKET_VERSION = "1.0"
REVISION_ACTIONS = frozenset(
    {"expand_concepts", "acquire_observation", "accept_current_best"}
)
ACQUISITION_POLICIES = frozenset(
    {
        "no_acquisition",
        "space_filling_design",
        "always_predictive_disagreement",
        "gated_predictive_disagreement",
    }
)


@dataclass(frozen=True)
class CounterexampleObservation:
    evidence_id: str
    variables: Mapping[str, float]
    observed_response: float
    baseline_prediction: float
    candidate_prediction: float
    signed_residual: float

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "variables": dict(self.variables),
            "observed_response": self.observed_response,
            "baseline_prediction": self.baseline_prediction,
            "candidate_prediction": self.candidate_prediction,
            "signed_residual": self.signed_residual,
        }


@dataclass(frozen=True)
class CounterexampleEvidencePacket:
    packet_version: str
    response_name: str
    variable_descriptions: Mapping[str, str]
    noise_scale: float
    baseline_validation_rmse: float
    best_validation_rmse: float
    relative_validation_improvement: float
    selected_candidate_id: str
    residual_statistics: Mapping[str, float]
    residual_variable_correlations: Mapping[str, float]
    counterexamples: tuple[CounterexampleObservation, ...]
    competing_candidates: tuple[Mapping[str, Any], ...]
    constraints: tuple[str, ...]
    unresolved_issue: str

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "packet_version": self.packet_version,
            "response_name": self.response_name,
            "variable_descriptions": dict(self.variable_descriptions),
            "noise_scale": self.noise_scale,
            "baseline_validation_rmse": self.baseline_validation_rmse,
            "best_validation_rmse": self.best_validation_rmse,
            "relative_validation_improvement": self.relative_validation_improvement,
            "selected_candidate_id": self.selected_candidate_id,
            "residual_statistics": dict(self.residual_statistics),
            "residual_variable_correlations": dict(
                self.residual_variable_correlations
            ),
            "counterexamples": [
                item.to_prompt_payload() for item in self.counterexamples
            ],
            "competing_candidates": [dict(item) for item in self.competing_candidates],
            "constraints": list(self.constraints),
            "unresolved_issue": self.unresolved_issue,
            "instructions": [
                "Infer reusable structural atoms from the evidence, not numerical coefficients.",
                "Use fitted parameter nodes for unknown constants.",
                "Do not claim that a proposed concept is accepted.",
            ],
        }

    @property
    def sha256(self) -> str:
        serialized = json.dumps(
            self.to_prompt_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RevisionGateDecision:
    action: str
    reason: str
    baseline_validation_rmse: float
    best_validation_rmse: float
    relative_validation_improvement: float
    valid_candidate_count: int
    stable_candidate_count: int
    ambiguity_model_count: int
    normalized_prediction_ambiguity: float

    def __post_init__(self) -> None:
        if self.action not in REVISION_ACTIONS:
            raise ValueError(f"Unsupported revision action {self.action!r}.")

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConceptExpansionResult:
    invoked: bool
    decision: RevisionGateDecision
    evidence_sha256: str
    proposed_concept_count: int
    compiled_candidate_count: int
    candidates: tuple[TypedRepair, ...]
    rejected_reasons: tuple[str, ...]

    def to_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("candidates")
        payload["rejected_reasons"] = list(self.rejected_reasons)
        return payload


@dataclass(frozen=True)
class AcquisitionPolicyDecision:
    policy: str
    acquire: bool
    scoring_strategy: str
    reason: str


@dataclass(frozen=True)
class CandidateAmbiguityAudit:
    ambiguity: DeploymentAmbiguity
    valid_candidate_count: int
    equivalent_candidate_ids: tuple[str, ...]
    equivalence_threshold: float
    excluded_candidate_ids: tuple[str, ...]

    def to_row(self) -> dict[str, Any]:
        return {
            "valid_candidate_count": self.valid_candidate_count,
            "equivalent_candidate_ids": list(self.equivalent_candidate_ids),
            "equivalent_candidate_count": len(self.equivalent_candidate_ids),
            "equivalence_threshold": self.equivalence_threshold,
            "excluded_candidate_ids": list(self.excluded_candidate_ids),
            "ambiguity_model_count": self.ambiguity.model_count,
            "maximum_pairwise_prediction_rmse": (
                self.ambiguity.maximum_pairwise_rmse
            ),
            "normalized_prediction_ambiguity": (
                self.ambiguity.normalized_maximum_pairwise_rmse
            ),
            "prediction_ambiguity_detected": self.ambiguity.ambiguous,
        }


def decide_acquisition_execution(
    policy: str,
    revision_decision: RevisionGateDecision,
) -> AcquisitionPolicyDecision:
    """Map a frozen acquisition arm to an executable existing score."""

    if policy not in ACQUISITION_POLICIES:
        raise ValueError(f"Unsupported acquisition policy {policy!r}.")
    if policy == "no_acquisition":
        return AcquisitionPolicyDecision(
            policy, False, "", "control_without_new_observations"
        )
    if policy == "space_filling_design":
        return AcquisitionPolicyDecision(
            policy, True, "space_filling_design", "fixed_budget_design_control"
        )
    if policy == "always_predictive_disagreement":
        return AcquisitionPolicyDecision(
            policy,
            True,
            "predictive_disagreement",
            "fixed_budget_disagreement_control",
        )
    acquire = revision_decision.action == "acquire_observation"
    return AcquisitionPolicyDecision(
        policy,
        acquire,
        "predictive_disagreement" if acquire else "",
        (
            "ambiguity_gate_triggered"
            if acquire
            else "ambiguity_gate_not_triggered"
        ),
    )


def audit_candidate_ambiguity(
    task: Gate1Task,
    evaluations: Sequence[CandidateEvaluation],
    repairs_by_id: Mapping[str, TypedRepair],
    query_pool: Any,
    *,
    bootstrap_samples: int,
    seed: int,
    standard_error_multiplier: float,
    noise_multiplier: float,
) -> CandidateAmbiguityAudit:
    """Audit extrapolative disagreement without accessing query responses."""

    if "target" in query_pool or "target_noise_free" in query_pool:
        raise ValueError("The ambiguity query pool must not contain response values.")
    validation = task.observed.loc[
        task.observed["partition"].eq("validation")
    ]
    if validation.empty:
        raise ValueError("A validation partition is required for ambiguity audit.")
    usable: list[CandidateEvaluation] = []
    validation_predictions: list[np.ndarray] = []
    query_predictions: list[np.ndarray] = []
    excluded: list[str] = []
    for evaluation in evaluations:
        repair = repairs_by_id.get(evaluation.proposal_id)
        if (
            evaluation.status != "valid"
            or evaluation.stability_violations != 0
            or repair is None
        ):
            excluded.append(evaluation.proposal_id)
            continue
        try:
            validation_prediction = predict_repair(
                task, repair, validation, evaluation.parameter_values
            )
            query_prediction = predict_repair(
                task, repair, query_pool, evaluation.parameter_values
            )
        except (
            ExpressionEvaluationError,
            KeyError,
            ValueError,
            FloatingPointError,
        ):
            excluded.append(evaluation.proposal_id)
            continue
        if not np.all(np.isfinite(validation_prediction)) or not np.all(
            np.isfinite(query_prediction)
        ):
            excluded.append(evaluation.proposal_id)
            continue
        usable.append(evaluation)
        validation_predictions.append(np.asarray(validation_prediction, dtype=float))
        query_predictions.append(np.asarray(query_prediction, dtype=float))

    if not usable:
        return CandidateAmbiguityAudit(
            ambiguity=DeploymentAmbiguity(0, 0.0, 0.0, False),
            valid_candidate_count=0,
            equivalent_candidate_ids=(),
            equivalence_threshold=float("nan"),
            excluded_candidate_ids=tuple(excluded),
        )
    equivalence = one_standard_error_equivalence(
        validation["target"].to_numpy(float),
        np.vstack(validation_predictions),
        np.asarray([item.complexity for item in usable], dtype=float),
        [item.proposal_id for item in usable],
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        standard_error_multiplier=standard_error_multiplier,
    )
    equivalent_ids = tuple(
        usable[index].proposal_id for index in equivalence.equivalent_indices
    )
    equivalent_query_predictions = np.vstack(
        [query_predictions[index] for index in equivalence.equivalent_indices]
    )
    ambiguity = deployment_prediction_ambiguity(
        equivalent_query_predictions,
        noise_std=task.definition.noise_std,
        noise_multiplier=noise_multiplier,
    )
    return CandidateAmbiguityAudit(
        ambiguity=ambiguity,
        valid_candidate_count=len(usable),
        equivalent_candidate_ids=equivalent_ids,
        equivalence_threshold=float(equivalence.equivalence_threshold),
        excluded_candidate_ids=tuple(excluded),
    )


def _finite_float(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Evidence values must be finite.")
    return result


def _rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((target - prediction) ** 2)))


def _baseline_prediction(task: Gate1Task, frame: Any) -> np.ndarray:
    variables = {name: frame[name].to_numpy(float) for name in task.variables}
    return np.asarray(
        evaluate_expression(task.baseline_expression, variables), dtype=float
    )


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= np.finfo(float).eps:
        return 0.0
    if np.std(right) <= np.finfo(float).eps:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def build_counterexample_evidence(
    task: Gate1Task,
    evaluations: Sequence[CandidateEvaluation],
    repairs_by_id: Mapping[str, TypedRepair],
    *,
    maximum_counterexamples: int = 6,
    maximum_competitors: int = 5,
    unresolved_issue: str = "current candidate family does not satisfy the acceptance contract",
) -> CounterexampleEvidencePacket:
    """Build a target-safe packet from fit/validation evidence only."""

    if maximum_counterexamples < 1 or maximum_competitors < 1:
        raise ValueError("Evidence packet limits must be positive.")
    validation = (
        task.observed.loc[task.observed["partition"].eq("validation")]
        .copy()
        .reset_index(drop=True)
    )
    if validation.empty:
        validation = (
            task.observed.loc[task.observed["partition"].eq("fit")]
            .copy()
            .reset_index(drop=True)
        )
    if validation.empty:
        raise ValueError("At least one fit or validation observation is required.")

    target = validation["target"].to_numpy(float)
    baseline = _baseline_prediction(task, validation)
    baseline_rmse = _rmse(target, baseline)
    valid = sorted(
        (
            item
            for item in evaluations
            if item.status == "valid" and item.proposal_id in repairs_by_id
        ),
        key=lambda item: (
            item.selection_score,
            item.complexity,
            item.proposal_id,
        ),
    )

    best = valid[0] if valid else None
    prompt_candidate_ids = {
        item.proposal_id: f"candidate_{index + 1:03d}"
        for index, item in enumerate(valid[:maximum_competitors])
    }
    prediction = baseline
    selected_id = "baseline_no_change"
    best_rmse = baseline_rmse
    if best is not None:
        candidate = predict_repair(
            task,
            repairs_by_id[best.proposal_id],
            validation,
            best.parameter_values,
        )
        if np.all(np.isfinite(candidate)):
            prediction = np.asarray(candidate, dtype=float)
            selected_id = prompt_candidate_ids.get(
                best.proposal_id, "candidate_selected"
            )
            best_rmse = _rmse(target, prediction)

    residual = target - prediction
    relative_improvement = (
        (baseline_rmse - best_rmse) / baseline_rmse
        if baseline_rmse > np.finfo(float).eps
        else 0.0
    )
    ranked_rows = np.argsort(-np.abs(residual), kind="stable")[
        : min(maximum_counterexamples, len(validation))
    ]
    counterexamples = tuple(
        CounterexampleObservation(
            evidence_id=f"residual_{rank + 1:03d}",
            variables={
                name: _finite_float(validation.iloc[index][name])
                for name in task.variables
            },
            observed_response=_finite_float(target[index]),
            baseline_prediction=_finite_float(baseline[index]),
            candidate_prediction=_finite_float(prediction[index]),
            signed_residual=_finite_float(residual[index]),
        )
        for rank, index in enumerate(ranked_rows)
    )
    correlations = {
        name: _safe_correlation(validation[name].to_numpy(float), residual)
        for name in task.variables
    }
    competitors = tuple(
        {
            "candidate_id": prompt_candidate_ids[item.proposal_id],
            "formula": item.formula,
            "validation_rmse": _finite_float(item.validation_rmse),
            "complexity": int(item.complexity),
            "stability_violations": int(item.stability_violations),
        }
        for item in valid[:maximum_competitors]
    )
    statistics = {
        "mean": _finite_float(np.mean(residual)),
        "standard_deviation": _finite_float(np.std(residual)),
        "maximum_absolute": _finite_float(np.max(np.abs(residual))),
        "lower_quartile": _finite_float(np.quantile(residual, 0.25)),
        "median": _finite_float(np.median(residual)),
        "upper_quartile": _finite_float(np.quantile(residual, 0.75)),
    }
    return CounterexampleEvidencePacket(
        packet_version=EVIDENCE_PACKET_VERSION,
        response_name=task.definition.target,
        variable_descriptions=dict(task.definition.variable_descriptions),
        noise_scale=_finite_float(task.definition.noise_std),
        baseline_validation_rmse=_finite_float(baseline_rmse),
        best_validation_rmse=_finite_float(best_rmse),
        relative_validation_improvement=_finite_float(relative_improvement),
        selected_candidate_id=selected_id,
        residual_statistics=statistics,
        residual_variable_correlations=correlations,
        counterexamples=counterexamples,
        competing_candidates=competitors,
        constraints=tuple(task.request.constraints),
        unresolved_issue=str(unresolved_issue).strip(),
    )


def decide_revision_action(
    evaluations: Sequence[CandidateEvaluation],
    *,
    baseline_validation_rmse: float,
    minimum_validation_improvement_fraction: float,
    noise_scale: float | None = None,
    maximum_rmse_noise_multiplier: float = 3.0,
    ambiguity: DeploymentAmbiguity | None = None,
    minimum_ambiguity_models: int = 2,
) -> RevisionGateDecision:
    """Choose between structure expansion, evidence acquisition, and acceptance."""

    baseline_rmse = _finite_float(baseline_validation_rmse)
    if baseline_rmse < 0.0:
        raise ValueError("baseline_validation_rmse must be non-negative.")
    if not 0.0 <= minimum_validation_improvement_fraction < 1.0:
        raise ValueError(
            "minimum_validation_improvement_fraction must lie in [0, 1)."
        )
    if minimum_ambiguity_models < 2:
        raise ValueError("minimum_ambiguity_models must be at least two.")
    if noise_scale is not None and noise_scale <= 0.0:
        raise ValueError("noise_scale must be positive when supplied.")
    if maximum_rmse_noise_multiplier <= 0.0:
        raise ValueError("maximum_rmse_noise_multiplier must be positive.")

    valid = [item for item in evaluations if item.status == "valid"]
    stable = [item for item in valid if item.stability_violations == 0]
    best = min(
        stable,
        key=lambda item: (
            item.selection_score,
            item.complexity,
            item.proposal_id,
        ),
        default=None,
    )
    best_rmse = float("nan") if best is None else float(best.validation_rmse)
    improvement = (
        float("-inf")
        if best is None
        else (
            (baseline_rmse - best_rmse) / baseline_rmse
            if baseline_rmse > np.finfo(float).eps
            else 0.0
        )
    )
    ambiguity_count = 0 if ambiguity is None else int(ambiguity.model_count)
    normalized_ambiguity = (
        0.0
        if ambiguity is None
        else float(ambiguity.normalized_maximum_pairwise_rmse)
    )

    if best is None:
        action = "expand_concepts"
        reason = "no_stable_valid_candidate"
    elif improvement < minimum_validation_improvement_fraction:
        action = "expand_concepts"
        reason = "best_candidate_fails_minimum_improvement"
    elif (
        noise_scale is not None
        and best_rmse > maximum_rmse_noise_multiplier * float(noise_scale)
    ):
        action = "expand_concepts"
        reason = "best_candidate_exceeds_noise_adequacy_limit"
    elif (
        ambiguity is not None
        and ambiguity.ambiguous
        and ambiguity.model_count >= minimum_ambiguity_models
    ):
        action = "acquire_observation"
        reason = "accepted_candidates_are_observationally_ambiguous"
    else:
        action = "accept_current_best"
        reason = "stable_candidate_satisfies_acceptance_contract"

    return RevisionGateDecision(
        action=action,
        reason=reason,
        baseline_validation_rmse=baseline_rmse,
        best_validation_rmse=best_rmse,
        relative_validation_improvement=improvement,
        valid_candidate_count=len(valid),
        stable_candidate_count=len(stable),
        ambiguity_model_count=ambiguity_count,
        normalized_prediction_ambiguity=normalized_ambiguity,
    )


def build_concept_expansion_request(
    original: RepairRequest,
    evidence: CounterexampleEvidencePacket,
    *,
    maximum_candidates: int,
    existing_structural_keys: Sequence[str] = (),
) -> RepairRequest:
    if maximum_candidates < 1:
        raise ValueError("maximum_candidates must be positive.")
    return RepairRequest(
        baseline_expression=original.baseline_expression,
        residual_evidence={
            "counterexample_packet": evidence.to_prompt_payload(),
        },
        constraints=tuple(original.constraints)
        + (
            "Propose compact structural atoms that address the cited counterexamples.",
            "Do not fit or guess numerical coefficients; use parameter nodes.",
            "Do not repeat a structure already represented by the competing candidates.",
        ),
        failed_structural_keys=frozenset(
            (*original.failed_structural_keys, *existing_structural_keys)
        ),
        maximum_candidates=maximum_candidates,
    )


def _expression_depth(node: Mapping[str, Any]) -> int:
    children = [
        child
        for value in node.values()
        for child in (
            value
            if isinstance(value, list)
            else [value]
            if isinstance(value, dict)
            else []
        )
    ]
    return 1 + max((_expression_depth(child) for child in children), default=0)


def compile_concept_batch(
    concepts: Sequence[TypedRepair],
    contract: RepairContract,
    *,
    id_prefix: str,
) -> tuple[TypedRepair, ...]:
    """Compile structural atoms into equal-form additive residual candidates."""

    compiled: list[TypedRepair] = []
    seen: set[str] = set()
    for index, concept in enumerate(concepts, start=1):
        atom = extract_structural_atom(concept, prefix=f"{id_prefix}_{index:03d}")
        expression = {
            "op": "multiply",
            "arguments": [
                {
                    "op": "parameter",
                    "name": f"{id_prefix}_{index:03d}_amplitude",
                },
                atom,
            ],
        }
        generated_contract = replace(
            contract,
            maximum_depth=max(contract.maximum_depth, _expression_depth(expression)),
            maximum_nodes=max(
                contract.maximum_nodes, expression_node_count(expression)
            ),
        )
        candidate = validate_typed_repair(
            {
                "proposal_id": f"{id_prefix}_{index:03d}",
                "source": concept.source,
                "edit_type": "add_term",
                "target": concept.target,
                "expression": expression,
                "rationale": (
                    f"Evidence-gated compilation of structural concept "
                    f"{concept.proposal_id}: {concept.rationale}"
                ),
                "expected_signature": concept.expected_signature,
            },
            generated_contract,
        )
        if candidate.structural_key in seen:
            continue
        seen.add(candidate.structural_key)
        compiled.append(candidate)
    return tuple(compiled)


def expand_concepts_if_needed(
    decision: RevisionGateDecision,
    evidence: CounterexampleEvidencePacket,
    original_request: RepairRequest,
    contract: RepairContract,
    proposer: RepairProposer,
    *,
    maximum_candidates: int,
    id_prefix: str = "concept",
    existing_structural_keys: Sequence[str] = (),
) -> ConceptExpansionResult:
    """Invoke a structural proposer only when evidence rejects the current family."""

    if decision.action != "expand_concepts":
        return ConceptExpansionResult(
            invoked=False,
            decision=decision,
            evidence_sha256=evidence.sha256,
            proposed_concept_count=0,
            compiled_candidate_count=0,
            candidates=(),
            rejected_reasons=(),
        )
    request = build_concept_expansion_request(
        original_request,
        evidence,
        maximum_candidates=maximum_candidates,
        existing_structural_keys=existing_structural_keys,
    )
    batch: ProposalBatch = proposer.propose(request, contract)
    compiled = compile_concept_batch(
        batch.accepted[:maximum_candidates],
        contract,
        id_prefix=id_prefix,
    )
    return ConceptExpansionResult(
        invoked=True,
        decision=decision,
        evidence_sha256=evidence.sha256,
        proposed_concept_count=len(batch.accepted),
        compiled_candidate_count=len(compiled),
        candidates=compiled,
        rejected_reasons=tuple(item.reason for item in batch.rejected),
    )

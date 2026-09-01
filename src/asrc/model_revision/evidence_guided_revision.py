from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from asrc.model_revision.acquisition import select_acquisition_index
from asrc.model_revision.ambiguity_active_design import (
    PrecompressionGate,
    build_precompression_committee,
    evaluate_precompression_gate,
    sample_acquisition_pool,
    score_precompression_acquisition,
    task_with_active_fit,
)
from asrc.model_revision.ast import evaluate_expression
from asrc.model_revision.benchmarks import Gate1Task
from asrc.model_revision.cost_aware_gate import (
    CostAwareEvidenceDecision,
    evaluate_cost_aware_evidence_value,
)
from asrc.model_revision.evaluation import finalize_selected_repair, predict_repair
from asrc.model_revision.evidence_equivalence import (
    EvidenceEquivalenceAssessment,
    EquivalentPredictionPair,
    evidence_equivalent_prediction_assessment,
)
from asrc.model_revision.proposals import TypedRepair
from asrc.model_revision.risk_aware_racing import (
    RiskAwareRacingResult,
    RiskAwarePromotionSnapshot,
    run_risk_aware_candidate_race,
)


ObservationProvider = Callable[
    [pd.Series, int],
    tuple[pd.DataFrame, float | None],
]
IterationCallback = Callable[["EvidenceGuidedIteration"], None]
SUPPORTED_EVIDENCE_GUIDED_ACQUISITIONS = frozenset(
    {"predictive_disagreement", "space_filling_design"}
)


@dataclass(frozen=True)
class EvidenceGuidedRevisionConfig:
    snapshot_stage_id: str = "parameter_refine"
    maximum_new_observations: int = 3
    acquisition_pool_count: int = 256
    outer_shell_only: bool = True
    ambiguity_noise_multiplier: float = 2.0
    minimum_equivalent_candidates: int = 2
    acquisition_strategy: str = "predictive_disagreement"
    joint_information_quadrature_order: int = 24
    require_ambiguity_gate: bool = True
    cost_aware_gate_enabled: bool = False
    observation_cost: float = 0.0
    cost_value_quadrature_order: int = 24
    cost_value_numerical_floor: float = 1.0e-12
    cost_value_committee_weighting: str = "bic"

    def __post_init__(self) -> None:
        if not self.snapshot_stage_id.strip():
            raise ValueError("snapshot_stage_id must not be empty.")
        if self.maximum_new_observations < 0:
            raise ValueError("maximum_new_observations must be non-negative.")
        if self.acquisition_pool_count < 1:
            raise ValueError("acquisition_pool_count must be positive.")
        if self.ambiguity_noise_multiplier < 0.0:
            raise ValueError("ambiguity_noise_multiplier must be non-negative.")
        if self.minimum_equivalent_candidates < 2:
            raise ValueError("minimum_equivalent_candidates must be at least two.")
        if self.observation_cost < 0.0:
            raise ValueError("observation_cost must be non-negative.")
        if self.cost_value_quadrature_order < 3:
            raise ValueError("cost_value_quadrature_order must be at least three.")
        if self.cost_value_numerical_floor <= 0.0:
            raise ValueError("cost_value_numerical_floor must be positive.")
        if self.cost_value_committee_weighting not in {"bic", "uniform_equivalent"}:
            raise ValueError(
                "cost_value_committee_weighting must be bic or uniform_equivalent."
            )
        if self.acquisition_strategy not in SUPPORTED_EVIDENCE_GUIDED_ACQUISITIONS:
            raise ValueError(
                "The unified workflow admits only the previously evaluated "
                "predictive_disagreement and space_filling_design strategies."
            )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> EvidenceGuidedRevisionConfig:
        gate = payload.get("ambiguity_gate", {})
        active = payload.get("active_design", {})
        cost = payload.get("cost_aware_gate", {})
        return cls(
            snapshot_stage_id=str(
                gate.get("snapshot_stage_id", "parameter_refine")
            ),
            maximum_new_observations=int(
                active.get("maximum_new_observations", 3)
            ),
            acquisition_pool_count=int(active.get("acquisition_pool_count", 256)),
            outer_shell_only=bool(active.get("outer_shell_only", True)),
            ambiguity_noise_multiplier=float(
                gate.get("minimum_prediction_difference_noise_multiplier", 2.0)
            ),
            minimum_equivalent_candidates=int(
                gate.get("minimum_equivalent_candidates", 2)
            ),
            acquisition_strategy=str(
                active.get("strategy", "predictive_disagreement")
            ),
            joint_information_quadrature_order=int(
                active.get("joint_information_quadrature_order", 24)
            ),
            require_ambiguity_gate=bool(
                active.get("require_ambiguity_gate", True)
            ),
            cost_aware_gate_enabled=bool(cost.get("enabled", False)),
            observation_cost=float(cost.get("observation_cost", 0.0)),
            cost_value_quadrature_order=int(
                cost.get("quadrature_order", 24)
            ),
            cost_value_numerical_floor=float(
                cost.get("numerical_floor", 1.0e-12)
            ),
            cost_value_committee_weighting=str(
                cost.get("committee_weighting", "bic")
            ),
        )


@dataclass(frozen=True)
class EvidenceGuidedIteration:
    observation_count: int
    fit_observation_count: int
    query_pool_count: int
    race: RiskAwareRacingResult
    precompression_gate: PrecompressionGate
    precompression_equivalent_ids: tuple[str, ...]
    committee_ids: tuple[str, ...]
    selected_summary: Mapping[str, Any]
    cost_aware_decision: CostAwareEvidenceDecision | None = None


@dataclass(frozen=True)
class EvidenceGuidedAcquisition:
    acquisition_number: int
    point_id: str
    coordinates: Mapping[str, float]
    baseline: float
    revealed_target: float
    target_noise_free: float | None
    score: float
    score_unit: str
    committee_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceGuidedRevisionResult:
    iterations: tuple[EvidenceGuidedIteration, ...]
    acquisitions: tuple[EvidenceGuidedAcquisition, ...]
    stop_reason: str
    final_equivalent_ids: tuple[str, ...]
    final_equivalence: EvidenceEquivalenceAssessment
    final_prediction_pairs: tuple[EquivalentPredictionPair, ...]
    recommended_query: Mapping[str, float]

    @property
    def initial_summary(self) -> Mapping[str, Any]:
        return self.iterations[0].selected_summary

    @property
    def final_summary(self) -> Mapping[str, Any]:
        return self.iterations[-1].selected_summary


def _snapshot(
    result: RiskAwareRacingResult,
    stage_id: str,
) -> RiskAwarePromotionSnapshot:
    matches = [
        snapshot
        for snapshot in result.promotion_snapshots
        if snapshot.stage_id == stage_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one pre-compression snapshot for {stage_id}; got "
            f"{len(matches)}."
        )
    return matches[0]


def _warm_starts(result: RiskAwareRacingResult) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for snapshot in result.promotion_snapshots:
        for evaluation in snapshot.evaluations:
            if evaluation.status == "valid":
                values[evaluation.proposal_id] = dict(evaluation.parameter_values)
    for evaluation in result.evaluations:
        if evaluation.status == "valid":
            values[evaluation.proposal_id] = dict(evaluation.parameter_values)
    return values


def _validate_observation(
    task: Gate1Task,
    observation: pd.DataFrame,
) -> None:
    required = {*task.variables, "target"}
    missing = sorted(required.difference(observation.columns))
    if len(observation) != 1 or missing:
        raise ValueError(
            "Observation provider must return one row containing variables and target; "
            f"missing={missing}."
        )
    values = observation[list(required)].to_numpy(float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Observation provider returned non-finite values.")


def _final_equivalence(
    task: Gate1Task,
    result: RiskAwareRacingResult,
    repairs_by_id: Mapping[str, TypedRepair],
    query_pool: pd.DataFrame,
    config: EvidenceGuidedRevisionConfig,
) -> tuple[
    tuple[str, ...],
    EvidenceEquivalenceAssessment,
    tuple[EquivalentPredictionPair, ...],
    dict[str, float],
]:
    evaluation_by_id = {
        item.proposal_id: item
        for item in result.evaluations
        if item.status == "valid"
    }
    admissible = {
        item.proposal_id for item in result.assessments if item.admissible
    }
    if result.baseline_assessment.admissible:
        admissible.add(result.baseline_assessment.proposal_id)
    selected_id = (
        result.selected.proposal_id
        if result.selected is not None
        else result.baseline_assessment.proposal_id
    )
    ordered_ids = tuple(
        sorted(admissible, key=lambda item: (item != selected_id, item))
    )
    if not ordered_ids:
        raise RuntimeError("The final verifier produced no admissible model.")

    predictions: list[np.ndarray] = []
    variables = {
        name: query_pool[name].to_numpy(float) for name in task.variables
    }
    for proposal_id in ordered_ids:
        if proposal_id == result.baseline_assessment.proposal_id:
            prediction = evaluate_expression(task.baseline_expression, variables)
        else:
            evaluation = evaluation_by_id[proposal_id]
            prediction = predict_repair(
                task,
                repairs_by_id[proposal_id],
                query_pool,
                evaluation.parameter_values,
            )
        predictions.append(np.asarray(prediction, dtype=float))

    assessment, pairs = evidence_equivalent_prediction_assessment(
        ordered_ids,
        np.vstack(predictions),
        noise_std=task.definition.noise_std,
        ambiguity_noise_multiplier=config.ambiguity_noise_multiplier,
        acquisition_available=not query_pool.empty,
    )
    recommended: dict[str, float] = {}
    if assessment.recommended_query_index is not None:
        row = query_pool.iloc[assessment.recommended_query_index]
        recommended = {name: float(row[name]) for name in task.variables}
    return ordered_ids, assessment, pairs, recommended


def run_evidence_guided_revision(
    task: Gate1Task,
    repairs: Iterable[TypedRepair],
    *,
    method: str,
    seed: int,
    evaluation_config: Mapping[str, Any],
    racing_config: Mapping[str, Any],
    selection_config: Mapping[str, Any],
    workflow_config: EvidenceGuidedRevisionConfig | Mapping[str, Any],
    recovery_noise_multiplier: float,
    observation_provider: ObservationProvider | None,
    iteration_callback: IterationCallback | None = None,
) -> EvidenceGuidedRevisionResult:
    """Run risk selection, conditional querying, and equivalence reporting.

    The acquisition score never sees a response at an unobserved point. The
    observation provider is invoked only after a query coordinate is selected.
    """

    config = (
        workflow_config
        if isinstance(workflow_config, EvidenceGuidedRevisionConfig)
        else EvidenceGuidedRevisionConfig.from_mapping(workflow_config)
    )
    repair_tuple = tuple(repairs)
    repairs_by_id = {repair.proposal_id: repair for repair in repair_tuple}
    if len(repairs_by_id) != len(repair_tuple):
        raise ValueError("Evidence-guided revision requires unique proposal ids.")

    fit = (
        task.observed.loc[task.observed["partition"].eq("fit")]
        .copy()
        .reset_index(drop=True)
    )
    query_pool = sample_acquisition_pool(
        task,
        count=config.acquisition_pool_count,
        seed=int(seed) + 7919,
        outer_shell_only=config.outer_shell_only,
    )
    iterations: list[EvidenceGuidedIteration] = []
    acquisitions: list[EvidenceGuidedAcquisition] = []
    warm_starts: dict[str, dict[str, float]] = {}
    stop_reason = "maximum_active_budget_reached"

    for observation_count in range(config.maximum_new_observations + 1):
        active_task = task_with_active_fit(task, fit)
        race = run_risk_aware_candidate_race(
            active_task,
            repair_tuple,
            method=method,
            seed=int(seed) + 104729 * observation_count,
            evaluation_config=evaluation_config,
            racing_config=racing_config,
            selection_config=selection_config,
            initial_parameters_by_proposal=warm_starts,
        )
        warm_starts = _warm_starts(race)
        summary = finalize_selected_repair(
            active_task,
            method,
            race.evaluations,
            race.selected,
            repairs_by_id,
            recovery_noise_multiplier=float(recovery_noise_multiplier),
            retain_baseline_if_unselected=True,
        ).to_row()
        snapshot = _snapshot(race, config.snapshot_stage_id)
        committee = build_precompression_committee(
            active_task,
            snapshot,
            repairs_by_id,
            query_pool,
        )
        gate = evaluate_precompression_gate(
            snapshot,
            committee,
            noise_std=task.definition.noise_std,
            noise_multiplier=config.ambiguity_noise_multiplier,
            minimum_equivalent_candidates=config.minimum_equivalent_candidates,
        )
        scores: np.ndarray | None = None
        score_components: dict[str, np.ndarray | str] | None = None
        selected_index: int | None = None
        cost_decision: CostAwareEvidenceDecision | None = None
        standard_gate_allows_query = bool(
            (not config.require_ambiguity_gate) or gate.triggered
        )
        query_preconditions_met = bool(
            standard_gate_allows_query
            and observation_count < config.maximum_new_observations
            and observation_provider is not None
            and (
                config.acquisition_strategy == "space_filling_design"
                or len(committee.evaluations) >= 2
            )
        )
        if query_preconditions_met:
            scores, score_components = score_precompression_acquisition(
                config.acquisition_strategy,
                task=active_task,
                committee=committee,
                pool=query_pool,
                parameter_lower_bound=float(
                    evaluation_config["parameter_lower_bound"]
                ),
                parameter_upper_bound=float(
                    evaluation_config["parameter_upper_bound"]
                ),
                quadrature_order=config.joint_information_quadrature_order,
            )
            selected_index = select_acquisition_index(
                scores,
                query_pool["point_id"].astype(str).tolist(),
            )
            if config.cost_aware_gate_enabled:
                value_weights = (
                    np.full(
                        len(committee.evaluations),
                        1.0 / len(committee.evaluations),
                        dtype=float,
                    )
                    if config.cost_value_committee_weighting
                    == "uniform_equivalent"
                    else committee.weights
                )
                cost_decision = evaluate_cost_aware_evidence_value(
                    committee.pool_predictions,
                    value_weights,
                    query_index=selected_index,
                    query_id=str(query_pool.iloc[selected_index]["point_id"]),
                    noise_std=task.definition.noise_std,
                    observation_cost=config.observation_cost,
                    quadrature_order=config.cost_value_quadrature_order,
                    numerical_floor=config.cost_value_numerical_floor,
                    committee_weighting=config.cost_value_committee_weighting,
                )
        iteration = EvidenceGuidedIteration(
            observation_count=observation_count,
            fit_observation_count=len(fit),
            query_pool_count=len(query_pool),
            race=race,
            precompression_gate=gate,
            precompression_equivalent_ids=snapshot.diagnostic_equivalent_ids,
            committee_ids=committee.proposal_ids,
            selected_summary=summary,
            cost_aware_decision=cost_decision,
        )
        iterations.append(iteration)
        if iteration_callback is not None:
            iteration_callback(iteration)

        if config.require_ambiguity_gate and not gate.triggered:
            stop_reason = gate.reason
            break
        if observation_count >= config.maximum_new_observations:
            stop_reason = "maximum_active_budget_reached"
            break
        if observation_provider is None:
            stop_reason = "observation_provider_unavailable"
            break
        if (
            config.acquisition_strategy != "space_filling_design"
            and len(committee.evaluations) < 2
        ):
            stop_reason = "insufficient_models_for_predictive_disagreement"
            break
        if cost_decision is not None and not cost_decision.should_query:
            stop_reason = cost_decision.reason
            break
        if scores is None or score_components is None or selected_index is None:
            raise RuntimeError("The active query was not scored before observation.")
        point = query_pool.iloc[selected_index]
        observation, noise_free = observation_provider(
            point,
            observation_count + 1,
        )
        _validate_observation(task, observation)
        acquisitions.append(
            EvidenceGuidedAcquisition(
                acquisition_number=observation_count + 1,
                point_id=str(point["point_id"]),
                coordinates={
                    name: float(point[name]) for name in task.variables
                },
                baseline=float(point["baseline"]),
                revealed_target=float(observation.iloc[0]["target"]),
                target_noise_free=(
                    None if noise_free is None else float(noise_free)
                ),
                score=float(scores[selected_index]),
                score_unit=str(score_components["score_unit"]),
                committee_ids=committee.proposal_ids,
            )
        )
        fit = pd.concat([fit, observation], ignore_index=True)
        query_pool = query_pool.drop(query_pool.index[selected_index]).reset_index(
            drop=True
        )

    final_race = iterations[-1].race
    equivalent_ids, equivalence, pairs, recommended = _final_equivalence(
        task,
        final_race,
        repairs_by_id,
        query_pool,
        config,
    )
    return EvidenceGuidedRevisionResult(
        iterations=tuple(iterations),
        acquisitions=tuple(acquisitions),
        stop_reason=stop_reason,
        final_equivalent_ids=equivalent_ids,
        final_equivalence=equivalence,
        final_prediction_pairs=pairs,
        recommended_query=recommended,
    )

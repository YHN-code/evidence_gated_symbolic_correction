from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Mapping

import numpy as np

from asrc.model_revision.ast import ExpressionEvaluationError, evaluate_expression
from asrc.model_revision.benchmarks import Gate1Task
from asrc.model_revision.candidate_racing import CandidateRacingConfig
from asrc.model_revision.evaluation import (
    CandidateEvaluation,
    build_extrapolation_guard_frame,
    evaluate_candidates,
    predict_repair,
    repair_parameter_jacobian,
)
from asrc.model_revision.proposals import TypedRepair


@dataclass(frozen=True)
class RiskAwareSelectionConfig:
    validation_environment_count: int = 4
    minimum_environment_rows: int = 8
    one_standard_error_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.validation_environment_count < 2:
            raise ValueError("validation_environment_count must be at least two.")
        if self.minimum_environment_rows < 2:
            raise ValueError("minimum_environment_rows must be at least two.")
        if self.one_standard_error_multiplier < 0.0:
            raise ValueError("one_standard_error_multiplier must be non-negative.")

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> RiskAwareSelectionConfig:
        return cls(
            validation_environment_count=int(
                payload.get("validation_environment_count", 4)
            ),
            minimum_environment_rows=int(payload.get("minimum_environment_rows", 8)),
            one_standard_error_multiplier=float(
                payload.get("one_standard_error_multiplier", 1.0)
            ),
        )


@dataclass(frozen=True)
class UncertaintyPreservingPromotionConfig:
    policy: str = "risk_rank"
    standard_error_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.policy not in {"risk_rank", "one_standard_error_simplicity"}:
            raise ValueError(f"Unknown risk-aware promotion policy: {self.policy}")
        if self.standard_error_multiplier < 0.0:
            raise ValueError(
                "Promotion standard_error_multiplier must be non-negative."
            )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> UncertaintyPreservingPromotionConfig:
        values = dict(payload or {})
        return cls(
            policy=str(values.get("policy", "risk_rank")),
            standard_error_multiplier=float(
                values.get("standard_error_multiplier", 1.0)
            ),
        )


@dataclass(frozen=True)
class ValidationEnvironment:
    environment_id: str
    positions: tuple[int, ...]
    minimum_radius: float
    maximum_radius: float


@dataclass(frozen=True)
class RiskAwareAssessment:
    task_id: str
    proposal_id: str
    source: str
    status: str
    environment_rmse: dict[str, float]
    environment_rmse_standard_error: dict[str, float]
    environment_count: int
    mean_environment_rmse: float
    environment_standard_error: float
    worst_environment_rmse: float
    frontier_environment_rmse: float
    robust_validation_risk: float
    extrapolation_amplification: float
    log10_parameter_condition: float
    complexity: int
    parameter_count: int
    admissible: bool = False
    selected: bool = False

    def to_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["environment_rmse"] = dict(self.environment_rmse)
        return payload


@dataclass(frozen=True)
class RiskAwareRacingTrace:
    task_id: str
    method: str
    stage_index: int
    stage_id: str
    proposal_id: str
    source: str
    status: str
    rank: int | None
    promoted: bool
    admissible: bool
    final_selected: bool
    selection_score: float
    validation_rmse: float
    mean_environment_rmse: float
    environment_standard_error: float
    worst_environment_rmse: float
    frontier_environment_rmse: float
    robust_validation_risk: float
    extrapolation_amplification: float
    log10_parameter_condition: float
    selection_threshold: float
    promotion_threshold: float
    promotion_equivalent: bool
    complexity: int
    parameter_count: int
    stage_optimizer_evaluations: int
    cumulative_optimizer_evaluations: int
    maximum_function_evaluations: int
    multistart_count: int

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskAwarePromotionSnapshot:
    task_id: str
    method: str
    stage_index: int
    stage_id: str
    evaluations: tuple[CandidateEvaluation, ...]
    assessments: tuple[RiskAwareAssessment, ...]
    survivor_ids: tuple[str, ...]
    diagnostic_equivalence_threshold: float
    diagnostic_equivalent_ids: tuple[str, ...]


@dataclass(frozen=True)
class RiskAwareRacingResult:
    evaluations: tuple[CandidateEvaluation, ...]
    assessments: tuple[RiskAwareAssessment, ...]
    baseline_assessment: RiskAwareAssessment
    selected: CandidateEvaluation | None
    trace: tuple[RiskAwareRacingTrace, ...]
    initial_candidate_count: int
    completed_stage_count: int
    total_optimizer_evaluations: int
    selection_threshold: float
    promotion_snapshots: tuple[RiskAwarePromotionSnapshot, ...]


def validation_shell_environments(
    task: Gate1Task,
    selection_config: RiskAwareSelectionConfig | Mapping[str, Any],
) -> tuple[ValidationEnvironment, ...]:
    config = (
        selection_config
        if isinstance(selection_config, RiskAwareSelectionConfig)
        else RiskAwareSelectionConfig.from_mapping(selection_config)
    )
    fit = task.observed.loc[task.observed["partition"].eq("fit")]
    validation = task.observed.loc[
        task.observed["partition"].eq("validation")
    ].reset_index(drop=True)
    if len(validation) < 2 * config.minimum_environment_rows:
        raise ValueError(
            "Risk-aware selection requires enough validation rows for two environments."
        )
    variables = list(task.variables)
    fit_values = fit[variables].to_numpy(float)
    center = np.median(fit_values, axis=0)
    scale = np.quantile(fit_values, 0.75, axis=0) - np.quantile(
        fit_values, 0.25, axis=0
    )
    fallback = np.std(fit_values, axis=0)
    scale = np.where(scale > 1.0e-12, scale, fallback)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    radius = np.sqrt(
        np.mean(((validation[variables].to_numpy(float) - center) / scale) ** 2, axis=1)
    )
    environment_count = min(
        config.validation_environment_count,
        len(validation) // config.minimum_environment_rows,
    )
    environment_count = max(2, environment_count)
    ordered = np.argsort(radius, kind="stable")
    groups = np.array_split(ordered, environment_count)
    return tuple(
        ValidationEnvironment(
            environment_id=f"shell_{index:02d}",
            positions=tuple(int(item) for item in positions),
            minimum_radius=float(np.min(radius[positions])),
            maximum_radius=float(np.max(radius[positions])),
        )
        for index, positions in enumerate(groups, start=1)
    )


def _risk_statistics(
    observed: np.ndarray,
    predicted: np.ndarray,
    environments: tuple[ValidationEnvironment, ...],
) -> tuple[
    dict[str, float],
    dict[str, float],
    float,
    float,
    float,
    float,
    float,
]:
    environment_rmse: dict[str, float] = {}
    environment_standard_error: dict[str, float] = {}
    for environment in environments:
        positions = np.asarray(environment.positions, dtype=int)
        squared_error = (observed[positions] - predicted[positions]) ** 2
        rmse = float(np.sqrt(np.mean(squared_error)))
        mse_standard_error = (
            float(np.std(squared_error, ddof=1) / np.sqrt(len(squared_error)))
            if len(squared_error) > 1
            else 0.0
        )
        rmse_standard_error = (
            mse_standard_error / (2.0 * rmse) if rmse > 1.0e-12 else 0.0
        )
        environment_rmse[environment.environment_id] = rmse
        environment_standard_error[environment.environment_id] = float(
            rmse_standard_error
        )
    values = np.asarray(list(environment_rmse.values()), dtype=float)
    mean = float(np.mean(values))
    worst_index = int(np.argmax(values))
    worst = float(values[worst_index])
    standard_error = float(list(environment_standard_error.values())[worst_index])
    frontier = float(values[-1])
    robust_risk = worst
    return (
        environment_rmse,
        environment_standard_error,
        mean,
        standard_error,
        worst,
        frontier,
        robust_risk,
    )


def _prediction_amplification(
    task: Gate1Task,
    repair: TypedRepair | None,
    parameters: Mapping[str, float],
    evaluation_config: Mapping[str, Any],
) -> float:
    guard = build_extrapolation_guard_frame(task, evaluation_config)
    if repair is None:
        prediction = guard[list(task.variables)].copy()
        variables = {
            name: prediction[name].to_numpy(float) for name in task.variables
        }
        values = evaluate_expression(task.baseline_expression, variables)
    else:
        values = predict_repair(task, repair, guard, parameters)
    observed_rms = max(
        float(np.sqrt(np.mean(task.observed["target"].to_numpy(float) ** 2))),
        1.0e-12,
    )
    return float(np.max(np.abs(values)) / observed_rms)


def _parameter_condition(
    task: Gate1Task,
    repair: TypedRepair,
    parameters: Mapping[str, float],
) -> float:
    names, jacobian = repair_parameter_jacobian(
        task,
        repair,
        task.observed.loc[task.observed["partition"].eq("validation")],
        parameters,
    )
    if not names:
        return 0.0
    norms = np.linalg.norm(jacobian, axis=0)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1.0e-12):
        return float("inf")
    normalized = jacobian / norms
    condition = float(np.linalg.cond(normalized))
    return float(np.log10(max(condition, 1.0))) if np.isfinite(condition) else float("inf")


def assess_candidate_risk(
    task: Gate1Task,
    repair: TypedRepair,
    evaluation: CandidateEvaluation,
    *,
    evaluation_config: Mapping[str, Any],
    selection_config: RiskAwareSelectionConfig | Mapping[str, Any],
    environments: tuple[ValidationEnvironment, ...] | None = None,
) -> RiskAwareAssessment:
    config = (
        selection_config
        if isinstance(selection_config, RiskAwareSelectionConfig)
        else RiskAwareSelectionConfig.from_mapping(selection_config)
    )
    environments = environments or validation_shell_environments(task, config)
    if evaluation.status != "valid":
        return RiskAwareAssessment(
            task_id=task.task_id,
            proposal_id=evaluation.proposal_id,
            source=evaluation.source,
            status=evaluation.status,
            environment_rmse={},
            environment_rmse_standard_error={},
            environment_count=len(environments),
            mean_environment_rmse=float("inf"),
            environment_standard_error=float("inf"),
            worst_environment_rmse=float("inf"),
            frontier_environment_rmse=float("inf"),
            robust_validation_risk=float("inf"),
            extrapolation_amplification=float("inf"),
            log10_parameter_condition=float("inf"),
            complexity=evaluation.complexity,
            parameter_count=evaluation.parameter_count,
        )
    validation = task.observed.loc[
        task.observed["partition"].eq("validation")
    ].reset_index(drop=True)
    try:
        prediction = predict_repair(
            task,
            repair,
            validation,
            evaluation.parameter_values,
        )
        statistics = _risk_statistics(
            validation["target"].to_numpy(float),
            prediction,
            environments,
        )
        amplification = _prediction_amplification(
            task,
            repair,
            evaluation.parameter_values,
            evaluation_config,
        )
        condition = _parameter_condition(
            task,
            repair,
            evaluation.parameter_values,
        )
    except (
        ExpressionEvaluationError,
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ):
        return RiskAwareAssessment(
            task_id=task.task_id,
            proposal_id=evaluation.proposal_id,
            source=evaluation.source,
            status="risk_evaluation_failed",
            environment_rmse={},
            environment_rmse_standard_error={},
            environment_count=len(environments),
            mean_environment_rmse=float("inf"),
            environment_standard_error=float("inf"),
            worst_environment_rmse=float("inf"),
            frontier_environment_rmse=float("inf"),
            robust_validation_risk=float("inf"),
            extrapolation_amplification=float("inf"),
            log10_parameter_condition=float("inf"),
            complexity=evaluation.complexity,
            parameter_count=evaluation.parameter_count,
        )
    return RiskAwareAssessment(
        task_id=task.task_id,
        proposal_id=evaluation.proposal_id,
        source=evaluation.source,
        status="valid",
        environment_rmse=statistics[0],
        environment_rmse_standard_error=statistics[1],
        environment_count=len(environments),
        mean_environment_rmse=statistics[2],
        environment_standard_error=statistics[3],
        worst_environment_rmse=statistics[4],
        frontier_environment_rmse=statistics[5],
        robust_validation_risk=statistics[6],
        extrapolation_amplification=amplification,
        log10_parameter_condition=condition,
        complexity=evaluation.complexity,
        parameter_count=evaluation.parameter_count,
    )


def assess_baseline_risk(
    task: Gate1Task,
    *,
    evaluation_config: Mapping[str, Any],
    selection_config: RiskAwareSelectionConfig | Mapping[str, Any],
    environments: tuple[ValidationEnvironment, ...] | None = None,
) -> RiskAwareAssessment:
    config = (
        selection_config
        if isinstance(selection_config, RiskAwareSelectionConfig)
        else RiskAwareSelectionConfig.from_mapping(selection_config)
    )
    environments = environments or validation_shell_environments(task, config)
    validation = task.observed.loc[
        task.observed["partition"].eq("validation")
    ].reset_index(drop=True)
    statistics = _risk_statistics(
        validation["target"].to_numpy(float),
        validation["baseline"].to_numpy(float),
        environments,
    )
    return RiskAwareAssessment(
        task_id=task.task_id,
        proposal_id="baseline_no_change",
        source="baseline",
        status="valid",
        environment_rmse=statistics[0],
        environment_rmse_standard_error=statistics[1],
        environment_count=len(environments),
        mean_environment_rmse=statistics[2],
        environment_standard_error=statistics[3],
        worst_environment_rmse=statistics[4],
        frontier_environment_rmse=statistics[5],
        robust_validation_risk=statistics[6],
        extrapolation_amplification=_prediction_amplification(
            task,
            None,
            {},
            evaluation_config,
        ),
        log10_parameter_condition=0.0,
        complexity=0,
        parameter_count=0,
    )


def _assessment_rank_key(item: RiskAwareAssessment) -> tuple[Any, ...]:
    return (
        item.robust_validation_risk,
        item.complexity,
        item.parameter_count,
        item.extrapolation_amplification,
        item.proposal_id,
    )


def risk_equivalent_candidates(
    assessments: Iterable[RiskAwareAssessment],
    *,
    standard_error_multiplier: float,
) -> tuple[float, tuple[str, ...]]:
    if standard_error_multiplier < 0.0:
        raise ValueError("The standard-error multiplier must be non-negative.")
    valid = sorted(
        (item for item in assessments if item.status == "valid"),
        key=_assessment_rank_key,
    )
    if not valid:
        return float("nan"), ()
    best = valid[0]
    threshold = (
        best.robust_validation_risk
        + float(standard_error_multiplier) * best.environment_standard_error
    )
    tolerance = 1.0e-12 * max(1.0, abs(threshold))
    identifiers = tuple(
        item.proposal_id
        for item in valid
        if item.robust_validation_risk <= threshold + tolerance
    )
    return threshold, identifiers


def select_promotion_survivors(
    assessments: Iterable[RiskAwareAssessment],
    survivor_count: int,
    *,
    promotion_config: UncertaintyPreservingPromotionConfig | Mapping[str, Any] | None,
) -> tuple[list[RiskAwareAssessment], float, frozenset[str]]:
    config = (
        promotion_config
        if isinstance(promotion_config, UncertaintyPreservingPromotionConfig)
        else UncertaintyPreservingPromotionConfig.from_mapping(promotion_config)
    )
    valid = sorted(
        (item for item in assessments if item.status == "valid"),
        key=_assessment_rank_key,
    )
    count = min(len(valid), max(0, int(survivor_count)))
    if not valid or count == 0:
        return [], float("nan"), frozenset()
    if config.policy == "risk_rank":
        return valid[:count], float("nan"), frozenset()

    threshold, diagnostic_ids = risk_equivalent_candidates(
        valid,
        standard_error_multiplier=config.standard_error_multiplier,
    )
    equivalent_ids = frozenset(diagnostic_ids)
    equivalent = [item for item in valid if item.proposal_id in equivalent_ids]
    equivalent.sort(
        key=lambda item: (
            item.complexity,
            item.parameter_count,
            item.robust_validation_risk,
            item.extrapolation_amplification,
            item.proposal_id,
        )
    )
    outside = [item for item in valid if item.proposal_id not in equivalent_ids]
    return (equivalent + outside)[:count], threshold, equivalent_ids


def select_risk_aware_candidate(
    evaluations: Iterable[CandidateEvaluation],
    assessments: Iterable[RiskAwareAssessment],
    baseline_assessment: RiskAwareAssessment,
    *,
    selection_config: RiskAwareSelectionConfig | Mapping[str, Any],
) -> tuple[
    list[CandidateEvaluation],
    CandidateEvaluation | None,
    list[RiskAwareAssessment],
    RiskAwareAssessment,
    float,
]:
    config = (
        selection_config
        if isinstance(selection_config, RiskAwareSelectionConfig)
        else RiskAwareSelectionConfig.from_mapping(selection_config)
    )
    materialized = list(evaluations)
    assessment_list = list(assessments)
    valid = [item for item in assessment_list if item.status == "valid"]
    pool = [baseline_assessment, *valid]
    best = min(pool, key=_assessment_rank_key)
    threshold = (
        best.robust_validation_risk
        + config.one_standard_error_multiplier * best.environment_standard_error
    )
    tolerance = 1.0e-12 * max(1.0, abs(threshold))
    admissible = [
        item
        for item in pool
        if item.robust_validation_risk <= threshold + tolerance
    ]
    chosen = min(
        admissible,
        key=lambda item: (
            item.complexity,
            item.parameter_count,
            item.robust_validation_risk,
            item.extrapolation_amplification,
            item.proposal_id,
        ),
    )
    chosen_id = chosen.proposal_id
    selected_evaluations = [
        replace(item, selected=item.proposal_id == chosen_id)
        for item in materialized
    ]
    selected = next(
        (item for item in selected_evaluations if item.selected),
        None,
    )
    admissible_ids = {item.proposal_id for item in admissible}
    updated_assessments = [
        replace(
            item,
            admissible=item.proposal_id in admissible_ids,
            selected=item.proposal_id == chosen_id,
        )
        for item in assessment_list
    ]
    updated_baseline = replace(
        baseline_assessment,
        admissible=baseline_assessment.proposal_id in admissible_ids,
        selected=chosen_id == baseline_assessment.proposal_id,
    )
    return (
        selected_evaluations,
        selected,
        updated_assessments,
        updated_baseline,
        threshold,
    )


def run_risk_aware_candidate_race(
    task: Gate1Task,
    repairs: Iterable[TypedRepair],
    *,
    method: str,
    seed: int,
    evaluation_config: Mapping[str, Any],
    racing_config: CandidateRacingConfig | Mapping[str, Any],
    selection_config: RiskAwareSelectionConfig | Mapping[str, Any],
    initial_parameters_by_proposal: Mapping[str, Mapping[str, float]] | None = None,
    promotion_config: (
        UncertaintyPreservingPromotionConfig | Mapping[str, Any] | None
    ) = None,
) -> RiskAwareRacingResult:
    race = (
        racing_config
        if isinstance(racing_config, CandidateRacingConfig)
        else CandidateRacingConfig.from_mapping(racing_config)
    )
    risk = (
        selection_config
        if isinstance(selection_config, RiskAwareSelectionConfig)
        else RiskAwareSelectionConfig.from_mapping(selection_config)
    )
    promotion = (
        promotion_config
        if isinstance(promotion_config, UncertaintyPreservingPromotionConfig)
        else UncertaintyPreservingPromotionConfig.from_mapping(promotion_config)
    )
    active = list(repairs)
    if len({repair.proposal_id for repair in active}) != len(active):
        raise ValueError("Candidate racing requires unique proposal ids.")
    environments = validation_shell_environments(task, risk)
    baseline_assessment = assess_baseline_risk(
        task,
        evaluation_config=evaluation_config,
        selection_config=risk,
        environments=environments,
    )
    initial_count = len(active)
    if not active:
        return RiskAwareRacingResult(
            evaluations=(),
            assessments=(),
            baseline_assessment=baseline_assessment,
            selected=None,
            trace=(),
            initial_candidate_count=0,
            completed_stage_count=0,
            total_optimizer_evaluations=0,
            selection_threshold=baseline_assessment.robust_validation_risk,
            promotion_snapshots=(),
        )

    warm_starts = {
        proposal_id: dict(parameters)
        for proposal_id, parameters in (initial_parameters_by_proposal or {}).items()
    }
    cumulative_evaluations = {repair.proposal_id: 0 for repair in active}
    trace: list[RiskAwareRacingTrace] = []
    promotion_snapshots: list[RiskAwarePromotionSnapshot] = []
    final_evaluations: list[CandidateEvaluation] = []
    final_assessments: list[RiskAwareAssessment] = []
    selected: CandidateEvaluation | None = None
    threshold = float("nan")
    completed_stages = 0

    for stage_index, stage in enumerate(race.stages, start=1):
        stage_evaluation_config = {
            **dict(evaluation_config),
            "maximum_function_evaluations": stage.maximum_function_evaluations,
            "multistart_count": stage.multistart_count,
        }
        stage_evaluations = evaluate_candidates(
            task,
            active,
            method=method,
            seed=seed,
            evaluation_config=stage_evaluation_config,
            initial_parameters_by_proposal=warm_starts,
        )
        repair_by_id = {repair.proposal_id: repair for repair in active}
        stage_assessments = [
            assess_candidate_risk(
                task,
                repair_by_id[evaluation.proposal_id],
                evaluation,
                evaluation_config=stage_evaluation_config,
                selection_config=risk,
                environments=environments,
            )
            for evaluation in stage_evaluations
        ]
        completed_stages += 1
        ranked = sorted(
            (item for item in stage_assessments if item.status == "valid"),
            key=_assessment_rank_key,
        )
        rank_by_id = {
            assessment.proposal_id: rank
            for rank, assessment in enumerate(ranked, start=1)
        }
        evaluation_by_id = {
            evaluation.proposal_id: evaluation for evaluation in stage_evaluations
        }
        assessment_by_id = {
            assessment.proposal_id: assessment for assessment in stage_assessments
        }
        is_final_stage = stage_index == len(race.stages)
        if is_final_stage:
            (
                final_evaluations,
                selected,
                final_assessments,
                baseline_assessment,
                threshold,
            ) = select_risk_aware_candidate(
                stage_evaluations,
                stage_assessments,
                baseline_assessment,
                selection_config=risk,
            )
            promoted_ids: set[str] = set()
            final_assessment_by_id = {
                item.proposal_id: item for item in final_assessments
            }
            promotion_threshold = threshold
            equivalent_ids = frozenset(
                item.proposal_id for item in final_assessments if item.admissible
            )
        else:
            survivor_count = max(
                race.minimum_survivors,
                int(math.ceil(len(active) * stage.promote_fraction)),
            )
            survivors, promotion_threshold, equivalent_ids = (
                select_promotion_survivors(
                    stage_assessments,
                    survivor_count,
                    promotion_config=promotion,
                )
            )
            promoted_ids = {item.proposal_id for item in survivors}
            final_assessment_by_id = assessment_by_id
            diagnostic_threshold, diagnostic_equivalent_ids = (
                risk_equivalent_candidates(
                    stage_assessments,
                    standard_error_multiplier=risk.one_standard_error_multiplier,
                )
            )
            promotion_snapshots.append(
                RiskAwarePromotionSnapshot(
                    task_id=task.task_id,
                    method=method,
                    stage_index=stage_index,
                    stage_id=stage.stage_id,
                    evaluations=tuple(stage_evaluations),
                    assessments=tuple(stage_assessments),
                    survivor_ids=tuple(
                        item.proposal_id
                        for item in ranked
                        if item.proposal_id in promoted_ids
                    ),
                    diagnostic_equivalence_threshold=diagnostic_threshold,
                    diagnostic_equivalent_ids=diagnostic_equivalent_ids,
                )
            )

        for evaluation in stage_evaluations:
            cumulative_evaluations[evaluation.proposal_id] += int(
                evaluation.optimizer_evaluations
            )
            assessment = final_assessment_by_id[evaluation.proposal_id]
            trace.append(
                RiskAwareRacingTrace(
                    task_id=task.task_id,
                    method=method,
                    stage_index=stage_index,
                    stage_id=stage.stage_id,
                    proposal_id=evaluation.proposal_id,
                    source=evaluation.source,
                    status=assessment.status,
                    rank=rank_by_id.get(evaluation.proposal_id),
                    promoted=evaluation.proposal_id in promoted_ids,
                    admissible=assessment.admissible,
                    final_selected=assessment.selected,
                    selection_score=evaluation.selection_score,
                    validation_rmse=evaluation.validation_rmse,
                    mean_environment_rmse=assessment.mean_environment_rmse,
                    environment_standard_error=assessment.environment_standard_error,
                    worst_environment_rmse=assessment.worst_environment_rmse,
                    frontier_environment_rmse=assessment.frontier_environment_rmse,
                    robust_validation_risk=assessment.robust_validation_risk,
                    extrapolation_amplification=assessment.extrapolation_amplification,
                    log10_parameter_condition=assessment.log10_parameter_condition,
                    selection_threshold=threshold if is_final_stage else float("nan"),
                    promotion_threshold=promotion_threshold,
                    promotion_equivalent=evaluation.proposal_id in equivalent_ids,
                    complexity=evaluation.complexity,
                    parameter_count=evaluation.parameter_count,
                    stage_optimizer_evaluations=evaluation.optimizer_evaluations,
                    cumulative_optimizer_evaluations=cumulative_evaluations[
                        evaluation.proposal_id
                    ],
                    maximum_function_evaluations=stage.maximum_function_evaluations,
                    multistart_count=stage.multistart_count,
                )
            )

        if is_final_stage or not promoted_ids:
            if not is_final_stage:
                final_evaluations = stage_evaluations
                final_assessments = stage_assessments
                threshold = baseline_assessment.robust_validation_risk
            break
        warm_starts = {
            proposal_id: dict(evaluation_by_id[proposal_id].parameter_values)
            for proposal_id in promoted_ids
            if evaluation_by_id[proposal_id].status == "valid"
        }
        active = [
            repair_by_id[item.proposal_id]
            for item in ranked
            if item.proposal_id in promoted_ids
        ]

    return RiskAwareRacingResult(
        evaluations=tuple(final_evaluations),
        assessments=tuple(final_assessments),
        baseline_assessment=baseline_assessment,
        selected=selected,
        trace=tuple(trace),
        initial_candidate_count=initial_count,
        completed_stage_count=completed_stages,
        total_optimizer_evaluations=sum(cumulative_evaluations.values()),
        selection_threshold=threshold,
        promotion_snapshots=tuple(promotion_snapshots),
    )

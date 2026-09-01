from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from asrc.model_revision.benchmarks import Gate1Task
from asrc.model_revision.evaluation import (
    CandidateEvaluation,
    evaluate_candidates,
    select_candidate,
)
from asrc.model_revision.proposals import TypedRepair


@dataclass(frozen=True)
class RacingStage:
    stage_id: str
    maximum_function_evaluations: int
    multistart_count: int
    promote_fraction: float

    @property
    def resource_units(self) -> int:
        return self.maximum_function_evaluations * self.multistart_count

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> RacingStage:
        return cls(
            stage_id=str(payload["stage_id"]),
            maximum_function_evaluations=int(
                payload["maximum_function_evaluations"]
            ),
            multistart_count=int(payload["multistart_count"]),
            promote_fraction=float(payload.get("promote_fraction", 1.0)),
        )


@dataclass(frozen=True)
class CandidateRacingConfig:
    stages: tuple[RacingStage, ...]
    minimum_survivors: int = 4

    def __post_init__(self) -> None:
        if len(self.stages) < 2:
            raise ValueError("Candidate racing requires at least two fidelity stages.")
        if self.minimum_survivors < 1:
            raise ValueError("minimum_survivors must be positive.")
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("Candidate racing stage ids must be unique.")
        previous_resource = 0
        for index, stage in enumerate(self.stages):
            if not stage.stage_id.strip():
                raise ValueError("Candidate racing stage ids must not be empty.")
            if stage.maximum_function_evaluations < 1:
                raise ValueError("maximum_function_evaluations must be positive.")
            if stage.multistart_count < 1:
                raise ValueError("multistart_count must be positive.")
            if stage.resource_units <= previous_resource:
                raise ValueError(
                    "Candidate racing resource must increase strictly by stage."
                )
            if index < len(self.stages) - 1:
                if not 0.0 < stage.promote_fraction < 1.0:
                    raise ValueError(
                        "Non-final promote_fraction must lie strictly between zero and one."
                    )
            elif stage.promote_fraction != 1.0:
                raise ValueError("The final promote_fraction must equal one.")
            previous_resource = stage.resource_units

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CandidateRacingConfig:
        return cls(
            stages=tuple(
                RacingStage.from_mapping(stage) for stage in payload["stages"]
            ),
            minimum_survivors=int(payload.get("minimum_survivors", 4)),
        )


@dataclass(frozen=True)
class CandidateRacingTrace:
    task_id: str
    method: str
    stage_index: int
    stage_id: str
    proposal_id: str
    source: str
    status: str
    rank: int | None
    promoted: bool
    final_selected: bool
    selection_score: float
    validation_rmse: float
    complexity: int
    stage_optimizer_evaluations: int
    cumulative_optimizer_evaluations: int
    maximum_function_evaluations: int
    multistart_count: int

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateRacingResult:
    evaluations: tuple[CandidateEvaluation, ...]
    selected: CandidateEvaluation | None
    trace: tuple[CandidateRacingTrace, ...]
    initial_candidate_count: int
    completed_stage_count: int
    total_optimizer_evaluations: int


def _rank_valid(
    evaluations: Iterable[CandidateEvaluation],
) -> list[CandidateEvaluation]:
    return sorted(
        (item for item in evaluations if item.status == "valid"),
        key=lambda item: (
            item.selection_score,
            item.complexity,
            item.proposal_id,
        ),
    )


def run_candidate_race(
    task: Gate1Task,
    repairs: Iterable[TypedRepair],
    *,
    method: str,
    seed: int,
    evaluation_config: Mapping[str, Any],
    racing_config: CandidateRacingConfig | Mapping[str, Any],
    baseline_score: float | None = None,
    initial_parameters_by_proposal: Mapping[str, Mapping[str, float]] | None = None,
) -> CandidateRacingResult:
    config = (
        racing_config
        if isinstance(racing_config, CandidateRacingConfig)
        else CandidateRacingConfig.from_mapping(racing_config)
    )
    active = list(repairs)
    if len({repair.proposal_id for repair in active}) != len(active):
        raise ValueError("Candidate racing requires unique proposal ids.")
    initial_count = len(active)
    if not active:
        return CandidateRacingResult((), None, (), 0, 0, 0)

    warm_starts = {
        proposal_id: dict(parameters)
        for proposal_id, parameters in (initial_parameters_by_proposal or {}).items()
    }
    cumulative_evaluations = {repair.proposal_id: 0 for repair in active}
    trace: list[CandidateRacingTrace] = []
    final_evaluations: list[CandidateEvaluation] = []
    selected: CandidateEvaluation | None = None
    completed_stages = 0

    for stage_index, stage in enumerate(config.stages, start=1):
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
        completed_stages += 1
        ranked = _rank_valid(stage_evaluations)
        rank_by_id = {
            evaluation.proposal_id: rank
            for rank, evaluation in enumerate(ranked, start=1)
        }
        is_final_stage = stage_index == len(config.stages)
        if is_final_stage:
            final_evaluations, selected = select_candidate(
                stage_evaluations,
                baseline_score=baseline_score,
            )
            promoted_ids: set[str] = set()
            final_by_id = {
                evaluation.proposal_id: evaluation
                for evaluation in final_evaluations
            }
        else:
            survivor_count = max(
                config.minimum_survivors,
                int(math.ceil(len(active) * stage.promote_fraction)),
            )
            survivors = ranked[: min(len(ranked), survivor_count)]
            promoted_ids = {evaluation.proposal_id for evaluation in survivors}
            final_by_id = {item.proposal_id: item for item in stage_evaluations}

        for evaluation in stage_evaluations:
            cumulative_evaluations[evaluation.proposal_id] += int(
                evaluation.optimizer_evaluations
            )
            final_evaluation = final_by_id[evaluation.proposal_id]
            trace.append(
                CandidateRacingTrace(
                    task_id=task.task_id,
                    method=method,
                    stage_index=stage_index,
                    stage_id=stage.stage_id,
                    proposal_id=evaluation.proposal_id,
                    source=evaluation.source,
                    status=evaluation.status,
                    rank=rank_by_id.get(evaluation.proposal_id),
                    promoted=evaluation.proposal_id in promoted_ids,
                    final_selected=bool(final_evaluation.selected),
                    selection_score=evaluation.selection_score,
                    validation_rmse=evaluation.validation_rmse,
                    complexity=evaluation.complexity,
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
            break
        warm_starts = {
            evaluation.proposal_id: dict(evaluation.parameter_values)
            for evaluation in stage_evaluations
            if evaluation.proposal_id in promoted_ids
            and evaluation.status == "valid"
        }
        repair_by_id = {repair.proposal_id: repair for repair in active}
        active = [repair_by_id[item.proposal_id] for item in ranked if item.proposal_id in promoted_ids]

    return CandidateRacingResult(
        evaluations=tuple(final_evaluations),
        selected=selected,
        trace=tuple(trace),
        initial_candidate_count=initial_count,
        completed_stage_count=completed_stages,
        total_optimizer_evaluations=sum(cumulative_evaluations.values()),
    )

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import qmc

from asrc.constitutive.active_design import bic_model_weights
from asrc.model_revision.acquisition import (
    joint_structure_parameter_information_gain,
    local_parameter_information_gain,
    local_parameter_posterior_covariance,
    normalized_design_distance,
    weighted_point_disagreement,
)
from asrc.model_revision.ambiguity import (
    DeploymentAmbiguity,
    deployment_prediction_ambiguity,
)
from asrc.model_revision.ast import ExpressionEvaluationError, evaluate_expression
from asrc.model_revision.benchmarks import Gate1Task
from asrc.model_revision.evaluation import (
    CandidateEvaluation,
    predict_repair,
    repair_parameter_jacobian,
)
from asrc.model_revision.proposals import TypedRepair
from asrc.model_revision.risk_aware_racing import RiskAwarePromotionSnapshot


SUPPORTED_ACTIVE_STRATEGIES = frozenset(
    {
        "space_filling_design",
        "predictive_disagreement",
        "joint_structure_parameter_information_gain",
    }
)


@dataclass(frozen=True)
class PrecompressionCommittee:
    repairs: tuple[TypedRepair, ...]
    evaluations: tuple[CandidateEvaluation, ...]
    weights: np.ndarray
    bic: np.ndarray
    fit_predictions: np.ndarray
    pool_predictions: np.ndarray
    excluded_proposal_ids: tuple[str, ...]

    @property
    def proposal_ids(self) -> tuple[str, ...]:
        return tuple(item.proposal_id for item in self.evaluations)


@dataclass(frozen=True)
class PrecompressionGate:
    stage_id: str
    evaluated_candidate_count: int
    equivalent_candidate_count: int
    usable_committee_count: int
    equivalence_threshold: float
    ambiguity: DeploymentAmbiguity
    minimum_equivalent_candidates: int
    triggered: bool
    reason: str

    def to_row(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "evaluated_candidate_count": self.evaluated_candidate_count,
            "equivalent_candidate_count": self.equivalent_candidate_count,
            "usable_committee_count": self.usable_committee_count,
            "equivalence_threshold": self.equivalence_threshold,
            "maximum_pairwise_rmse": self.ambiguity.maximum_pairwise_rmse,
            "normalized_maximum_pairwise_rmse": (
                self.ambiguity.normalized_maximum_pairwise_rmse
            ),
            "ambiguity_model_count": self.ambiguity.model_count,
            "minimum_equivalent_candidates": self.minimum_equivalent_candidates,
            "triggered": self.triggered,
            "reason": self.reason,
        }


def sample_acquisition_pool(
    task: Gate1Task,
    *,
    count: int,
    seed: int,
    outer_shell_only: bool,
) -> pd.DataFrame:
    if count < 1:
        raise ValueError("Acquisition pool count must be positive.")
    names = list(task.variables)
    lower = np.asarray(
        [task.definition.locked_ranges[name][0] for name in names], dtype=float
    )
    upper = np.asarray(
        [task.definition.locked_ranges[name][1] for name in names], dtype=float
    )
    observed_lower = np.asarray(
        [task.definition.observed_ranges[name][0] for name in names], dtype=float
    )
    observed_upper = np.asarray(
        [task.definition.observed_ranges[name][1] for name in names], dtype=float
    )
    chunks: list[np.ndarray] = []
    generated = 0
    attempt = 0
    while generated < count:
        sampler = qmc.LatinHypercube(d=len(names), seed=int(seed) + attempt)
        values = qmc.scale(sampler.random(max(2 * count, 64)), lower, upper)
        if outer_shell_only:
            outside = np.any(
                (values < observed_lower[None, :])
                | (values > observed_upper[None, :]),
                axis=1,
            )
            values = values[outside]
        chunks.append(values)
        generated += len(values)
        attempt += 1
        if attempt > 100:
            raise RuntimeError("Unable to sample the locked-domain acquisition pool.")
    design = np.vstack(chunks)[:count]
    frame = pd.DataFrame(design, columns=names)
    frame.insert(0, "point_id", [f"q{index:04d}" for index in range(count)])
    variables = {name: frame[name].to_numpy(float) for name in names}
    frame["baseline"] = evaluate_expression(task.baseline_expression, variables)
    return frame


def task_with_active_fit(task: Gate1Task, fit: pd.DataFrame) -> Gate1Task:
    active_fit = fit.copy()
    active_fit["partition"] = "fit"
    validation = task.observed.loc[
        task.observed["partition"].eq("validation")
    ].copy()
    return replace(
        task,
        observed=pd.concat([active_fit, validation], ignore_index=True),
    )


def reveal_acquisition_observation(
    task: Gate1Task,
    point: pd.Series,
    *,
    data_seed: int,
) -> tuple[pd.DataFrame, float]:
    point_id = str(point["point_id"])
    design = pd.DataFrame(
        [{name: float(point[name]) for name in task.variables}]
    )
    variables = {name: design[name].to_numpy(float) for name in task.variables}
    baseline = evaluate_expression(task.baseline_expression, variables)
    noise_free = float(
        predict_repair(
            task,
            task.oracle_repair,
            design,
            task.definition.oracle_parameters,
        )[0]
    )
    digest = hashlib.sha256(
        f"{task.task_id}|{int(data_seed)}|{point_id}".encode("utf-8")
    ).digest()
    noise_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    noise = float(
        np.random.default_rng(noise_seed).normal(0.0, task.definition.noise_std)
    )
    design.insert(0, "point_id", point_id)
    design["baseline"] = baseline
    design["target"] = noise_free + noise
    design["partition"] = "fit"
    design["active_observation"] = True
    return design, noise_free


def build_precompression_committee(
    task: Gate1Task,
    snapshot: RiskAwarePromotionSnapshot,
    repairs_by_id: Mapping[str, TypedRepair],
    pool: pd.DataFrame,
) -> PrecompressionCommittee:
    evaluation_by_id = {
        item.proposal_id: item
        for item in snapshot.evaluations
        if item.status == "valid"
    }
    fit = task.observed.loc[task.observed["partition"].eq("fit")].copy()
    repairs: list[TypedRepair] = []
    evaluations: list[CandidateEvaluation] = []
    fit_predictions: list[np.ndarray] = []
    pool_predictions: list[np.ndarray] = []
    excluded: list[str] = []
    for proposal_id in snapshot.diagnostic_equivalent_ids:
        evaluation = evaluation_by_id.get(proposal_id)
        repair = repairs_by_id.get(proposal_id)
        if evaluation is None or repair is None:
            excluded.append(proposal_id)
            continue
        try:
            fit_prediction = predict_repair(
                task, repair, fit, evaluation.parameter_values
            )
            candidate_prediction = predict_repair(
                task, repair, pool, evaluation.parameter_values
            )
        except (ExpressionEvaluationError, KeyError, ValueError, FloatingPointError):
            excluded.append(proposal_id)
            continue
        if not np.all(np.isfinite(fit_prediction)) or not np.all(
            np.isfinite(candidate_prediction)
        ):
            excluded.append(proposal_id)
            continue
        repairs.append(repair)
        evaluations.append(evaluation)
        fit_predictions.append(np.asarray(fit_prediction, dtype=float))
        pool_predictions.append(np.asarray(candidate_prediction, dtype=float))

    if not evaluations:
        return PrecompressionCommittee(
            repairs=(),
            evaluations=(),
            weights=np.empty(0, dtype=float),
            bic=np.empty(0, dtype=float),
            fit_predictions=np.empty((0, len(fit)), dtype=float),
            pool_predictions=np.empty((0, len(pool)), dtype=float),
            excluded_proposal_ids=tuple(excluded),
        )
    fit_matrix = np.vstack(fit_predictions)
    pool_matrix = np.vstack(pool_predictions)
    weights, bic = bic_model_weights(
        fit["target"].to_numpy(float),
        fit_matrix,
        np.asarray([item.parameter_count for item in evaluations], dtype=float),
    )
    return PrecompressionCommittee(
        repairs=tuple(repairs),
        evaluations=tuple(evaluations),
        weights=weights,
        bic=bic,
        fit_predictions=fit_matrix,
        pool_predictions=pool_matrix,
        excluded_proposal_ids=tuple(excluded),
    )


def evaluate_precompression_gate(
    snapshot: RiskAwarePromotionSnapshot,
    committee: PrecompressionCommittee,
    *,
    noise_std: float,
    noise_multiplier: float,
    minimum_equivalent_candidates: int,
) -> PrecompressionGate:
    if minimum_equivalent_candidates < 2:
        raise ValueError("At least two equivalent candidates are required.")
    if committee.pool_predictions.shape[0] > 0:
        ambiguity = deployment_prediction_ambiguity(
            committee.pool_predictions,
            noise_std=noise_std,
            noise_multiplier=noise_multiplier,
        )
    else:
        ambiguity = DeploymentAmbiguity(0, 0.0, 0.0, False)
    enough = len(committee.evaluations) >= minimum_equivalent_candidates
    triggered = bool(enough and ambiguity.ambiguous)
    if not enough:
        reason = "insufficient_validation_equivalent_candidates"
    elif not ambiguity.ambiguous:
        reason = "equivalent_candidates_predict_similarly"
    else:
        reason = "validation_equivalent_candidates_diverge_in_query_domain"
    return PrecompressionGate(
        stage_id=snapshot.stage_id,
        evaluated_candidate_count=len(snapshot.evaluations),
        equivalent_candidate_count=len(snapshot.diagnostic_equivalent_ids),
        usable_committee_count=len(committee.evaluations),
        equivalence_threshold=snapshot.diagnostic_equivalence_threshold,
        ambiguity=ambiguity,
        minimum_equivalent_candidates=minimum_equivalent_candidates,
        triggered=triggered,
        reason=reason,
    )


def score_precompression_acquisition(
    strategy: str,
    *,
    task: Gate1Task,
    committee: PrecompressionCommittee,
    pool: pd.DataFrame,
    parameter_lower_bound: float,
    parameter_upper_bound: float,
    quadrature_order: int,
) -> tuple[np.ndarray, dict[str, np.ndarray | str]]:
    if strategy not in SUPPORTED_ACTIVE_STRATEGIES:
        raise ValueError(f"Unsupported pre-compression strategy: {strategy}")
    variable_names = list(task.variables)
    lower = np.asarray(
        [task.definition.locked_ranges[name][0] for name in variable_names],
        dtype=float,
    )
    upper = np.asarray(
        [task.definition.locked_ranges[name][1] for name in variable_names],
        dtype=float,
    )
    fit = task.observed.loc[task.observed["partition"].eq("fit")].copy()
    if strategy == "space_filling_design":
        scores = normalized_design_distance(
            pool[variable_names].to_numpy(float),
            fit[variable_names].to_numpy(float),
            lower,
            upper,
        )
        return scores, {"score_unit": "normalized_distance"}
    if len(committee.evaluations) < 2:
        raise ValueError("Active ambiguity resolution requires at least two models.")
    if strategy == "predictive_disagreement":
        scores = weighted_point_disagreement(
            committee.pool_predictions, committee.weights
        )
        return scores, {"score_unit": "response"}

    prior_variance = (
        (float(parameter_upper_bound) - float(parameter_lower_bound)) ** 2 / 12.0
    )
    parameter_information = np.zeros_like(committee.pool_predictions)
    predictive_std = np.full_like(
        committee.pool_predictions, float(task.definition.noise_std)
    )
    for index, (repair, evaluation) in enumerate(
        zip(committee.repairs, committee.evaluations)
    ):
        fit_names, fit_jacobian = repair_parameter_jacobian(
            task, repair, fit, evaluation.parameter_values
        )
        pool_names, pool_jacobian = repair_parameter_jacobian(
            task, repair, pool, evaluation.parameter_values
        )
        if fit_names != pool_names:
            raise RuntimeError("Parameter Jacobian order changed between designs.")
        covariance = local_parameter_posterior_covariance(
            fit_jacobian,
            task.definition.noise_std,
            prior_variance,
        )
        information, standard_deviation = local_parameter_information_gain(
            pool_jacobian,
            covariance,
            task.definition.noise_std,
        )
        parameter_information[index] = information
        predictive_std[index] = standard_deviation
    structural, parameter, joint = joint_structure_parameter_information_gain(
        committee.pool_predictions,
        predictive_std,
        parameter_information,
        committee.weights,
        quadrature_order=quadrature_order,
    )
    return joint, {
        "score_unit": "nat",
        "model_information_nats": structural,
        "parameter_information_nats": parameter,
        "joint_information_nats": joint,
    }

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CostAwareEvidenceDecision:
    query_index: int
    query_id: str
    committee_size: int
    current_bayes_risk: float
    expected_post_query_risk: float
    expected_value_of_evidence: float
    observation_cost: float
    net_value: float
    should_query: bool
    reason: str
    quadrature_order: int
    committee_weighting: str

    def to_row(self) -> dict[str, Any]:
        return {
            "cost_query_index": self.query_index,
            "cost_query_id": self.query_id,
            "cost_committee_size": self.committee_size,
            "cost_current_bayes_risk": self.current_bayes_risk,
            "cost_expected_post_query_risk": self.expected_post_query_risk,
            "cost_expected_value_of_evidence": self.expected_value_of_evidence,
            "cost_observation_cost": self.observation_cost,
            "cost_net_value": self.net_value,
            "cost_should_query": self.should_query,
            "cost_reason": self.reason,
            "cost_quadrature_order": self.quadrature_order,
            "cost_committee_weighting": self.committee_weighting,
        }


def _normalized_weights(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Cost-aware evidence value requires at least two weights.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Committee weights must be finite and non-negative.")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("Committee weights must have positive mass.")
    return values / total


def _decision_loss_matrix(
    predictions: np.ndarray,
    scale: float,
) -> np.ndarray:
    # Rows are possible reported candidates; columns are candidate states.
    differences = predictions[:, None, :] - predictions[None, :, :]
    return np.mean(np.square(differences), axis=2) / (scale * scale)


def _posterior_weights(
    prior: np.ndarray,
    query_predictions: np.ndarray,
    response: float,
    noise_scale: float,
    numerical_floor: float,
) -> np.ndarray:
    log_prior = np.log(np.maximum(prior, numerical_floor))
    standardized = (response - query_predictions) / noise_scale
    log_values = log_prior - 0.5 * np.square(standardized)
    log_values -= float(np.max(log_values))
    values = np.exp(log_values)
    return values / float(values.sum())


def evaluate_cost_aware_evidence_value(
    pool_predictions: np.ndarray,
    weights: np.ndarray,
    *,
    query_index: int,
    query_id: str,
    noise_std: float,
    observation_cost: float,
    quadrature_order: int = 24,
    numerical_floor: float = 1.0e-12,
    committee_weighting: str = "provided",
) -> CostAwareEvidenceDecision:
    """Estimate one-step value of a proposed observation without revealing it.

    Candidate expressions and fitted parameters are held fixed inside the
    preposterior calculation. A hypothetical response updates only the BIC
    committee weights. The result is therefore an auditable one-step
    approximation, not a dynamic-programming optimum.
    """

    predictions = np.asarray(pool_predictions, dtype=float)
    if predictions.ndim != 2 or predictions.shape[0] < 2:
        raise ValueError(
            "Cost-aware evidence value requires predictions from at least two models."
        )
    if predictions.shape[1] < 1 or not np.all(np.isfinite(predictions)):
        raise ValueError("Pool predictions must be a finite non-empty matrix.")
    if query_index < 0 or query_index >= predictions.shape[1]:
        raise IndexError("query_index is outside the prediction pool.")
    if quadrature_order < 3:
        raise ValueError("quadrature_order must be at least three.")
    if numerical_floor <= 0.0:
        raise ValueError("numerical_floor must be positive.")
    if not np.isfinite(observation_cost) or observation_cost < 0.0:
        raise ValueError("observation_cost must be finite and non-negative.")

    prior = _normalized_weights(weights)
    if len(prior) != predictions.shape[0]:
        raise ValueError("Committee weights and predictions have different sizes.")
    noise_scale = max(abs(float(noise_std)), float(numerical_floor))
    loss = _decision_loss_matrix(predictions, noise_scale)
    current_risk = float(np.min(loss @ prior))

    nodes, quadrature_weights = np.polynomial.hermite.hermgauss(quadrature_order)
    normal_weights = quadrature_weights / np.sqrt(np.pi)
    query_predictions = predictions[:, query_index]
    expected_post_risk = 0.0
    for state_index, state_weight in enumerate(prior):
        if state_weight <= 0.0:
            continue
        responses = query_predictions[state_index] + (
            np.sqrt(2.0) * noise_scale * nodes
        )
        for response, integration_weight in zip(responses, normal_weights):
            posterior = _posterior_weights(
                prior,
                query_predictions,
                float(response),
                noise_scale,
                numerical_floor,
            )
            posterior_risk = float(np.min(loss @ posterior))
            expected_post_risk += (
                float(state_weight) * float(integration_weight) * posterior_risk
            )

    # Conditioning cannot increase Bayes risk in exact arithmetic because the
    # current action remains available after observing the response.
    expected_post_risk = min(max(expected_post_risk, 0.0), current_risk)
    evidence_value = max(0.0, current_risk - expected_post_risk)
    net_value = evidence_value - float(observation_cost)
    should_query = bool(net_value > 0.0)
    reason = (
        "expected_value_exceeds_observation_cost"
        if should_query
        else "expected_value_not_above_observation_cost"
    )
    return CostAwareEvidenceDecision(
        query_index=int(query_index),
        query_id=str(query_id),
        committee_size=len(prior),
        current_bayes_risk=current_risk,
        expected_post_query_risk=expected_post_risk,
        expected_value_of_evidence=evidence_value,
        observation_cost=float(observation_cost),
        net_value=net_value,
        should_query=should_query,
        reason=reason,
        quadrature_order=int(quadrature_order),
        committee_weighting=str(committee_weighting),
    )

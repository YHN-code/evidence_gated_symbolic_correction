from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp


@dataclass(frozen=True)
class InformationGainEstimate:
    value_nats: float
    standard_error_nats: float
    prior_entropy_nats: float


@dataclass(frozen=True)
class HierarchicalInformationGainEstimate:
    structural_information_nats: float
    parameter_information_nats: float
    joint_information_nats: float
    structural_standard_error_nats: float
    parameter_standard_error_nats: float
    joint_standard_error_nats: float
    prior_model_entropy_nats: float


def gower_distance_matrix(
    frame: pd.DataFrame,
    continuous_features: list[str],
    categorical_features: list[str],
) -> np.ndarray:
    """Return a mixed-type Gower distance matrix without using responses."""
    features = [*continuous_features, *categorical_features]
    if not features:
        raise ValueError("At least one design feature is required.")
    missing = [column for column in features if column not in frame]
    if missing:
        raise ValueError(f"Missing Gower design features: {missing}")
    if frame.empty:
        raise ValueError("Gower design requires at least one row.")

    distance = np.zeros((len(frame), len(frame)), dtype=float)
    contribution_count = 0
    for column in continuous_features:
        values = frame[column].to_numpy(float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Continuous Gower feature {column!r} must be finite.")
        value_range = float(np.max(values) - np.min(values))
        if value_range > 0.0:
            distance += np.abs(values[:, None] - values[None, :]) / value_range
        contribution_count += 1
    for column in categorical_features:
        values = frame[column].astype(str).to_numpy()
        distance += (values[:, None] != values[None, :]).astype(float)
        contribution_count += 1
    return distance / float(contribution_count)


def gower_medoid_farthest_order(
    frame: pd.DataFrame,
    id_column: str,
    continuous_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    """Build a deterministic medoid-first, maximin Gower design order."""
    if id_column not in frame:
        raise ValueError(f"Missing design identifier column: {id_column}")
    identifiers = frame[id_column].astype(str).tolist()
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Gower design identifiers must be unique.")
    distance = gower_distance_matrix(
        frame,
        continuous_features,
        categorical_features,
    )
    mean_distance = np.mean(distance, axis=1)
    first = min(
        range(len(frame)),
        key=lambda index: (float(mean_distance[index]), identifiers[index]),
    )
    selected = [first]
    remaining = set(range(len(frame))).difference(selected)
    while remaining:
        next_index = min(
            remaining,
            key=lambda index: (
                -float(np.min(distance[index, selected])),
                identifiers[index],
            ),
        )
        selected.append(next_index)
        remaining.remove(next_index)
    return [identifiers[index] for index in selected]


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Model weights must be a one-dimensional array of length >= 2.")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Model weights must be finite and non-negative.")
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("At least one model weight must be positive.")
    return values / total


def entropy_nats(weights: np.ndarray) -> float:
    probabilities = normalize_weights(weights)
    positive = probabilities > 0.0
    return float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))


def bic_model_weights(
    observations: np.ndarray,
    predictions: np.ndarray,
    complexities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observations, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    model_complexities = np.asarray(complexities, dtype=float)
    if observed.ndim != 1:
        raise ValueError("Observations must be one-dimensional.")
    if predicted.ndim != 2 or predicted.shape[1] != len(observed):
        raise ValueError("Predictions must have shape (models, observations).")
    if model_complexities.shape != (predicted.shape[0],):
        raise ValueError("One complexity value is required per model.")
    if len(observed) < 2 or not np.all(np.isfinite(observed)):
        raise ValueError("At least two finite observations are required.")
    if not np.all(np.isfinite(predicted)):
        raise ValueError("Predictions must be finite.")

    residuals = predicted - observed[None, :]
    rss = np.sum(residuals**2, axis=1)
    mean_square = np.maximum(rss / len(observed), np.finfo(float).tiny)
    bic = len(observed) * np.log(mean_square) + model_complexities * np.log(
        len(observed)
    )
    log_weights = -0.5 * (bic - float(np.min(bic)))
    weights = np.exp(log_weights - logsumexp(log_weights))
    return weights, bic


def posterior_model_weights(
    observations: np.ndarray,
    predictions: np.ndarray,
    prior_weights: np.ndarray,
    noise_std: float,
) -> np.ndarray:
    observed = np.asarray(observations, dtype=float)
    predicted = np.asarray(predictions, dtype=float)
    prior = normalize_weights(prior_weights)
    if noise_std <= 0.0 or not np.isfinite(noise_std):
        raise ValueError("noise_std must be finite and positive.")
    if observed.ndim != 1:
        raise ValueError("Observations must be one-dimensional.")
    if predicted.shape != (len(prior), len(observed)):
        raise ValueError("Predictions and prior weights have incompatible shapes.")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
        raise ValueError("Observations and predictions must be finite.")

    log_likelihood = -0.5 * np.sum(
        ((predicted - observed[None, :]) / float(noise_std)) ** 2,
        axis=1,
    )
    log_prior = np.full(len(prior), -np.inf, dtype=float)
    positive = prior > 0.0
    log_prior[positive] = np.log(prior[positive])
    log_posterior = log_prior + log_likelihood
    return np.exp(log_posterior - logsumexp(log_posterior))


def expected_model_information_gain(
    predictions: np.ndarray,
    prior_weights: np.ndarray,
    noise_std: float,
    *,
    sample_count: int = 512,
    seed: int = 0,
) -> InformationGainEstimate:
    predicted = np.asarray(predictions, dtype=float)
    prior = normalize_weights(prior_weights)
    if predicted.ndim != 2 or predicted.shape[0] != len(prior):
        raise ValueError("Predictions must have shape (models, observations).")
    if predicted.shape[1] < 1 or not np.all(np.isfinite(predicted)):
        raise ValueError("Predictions require at least one finite observation.")
    if noise_std <= 0.0 or not np.isfinite(noise_std):
        raise ValueError("noise_std must be finite and positive.")
    if sample_count < 2:
        raise ValueError("sample_count must be at least two.")

    rng = np.random.default_rng(int(seed))
    sampled_models = rng.choice(len(prior), size=int(sample_count), p=prior)
    samples = predicted[sampled_models] + rng.normal(
        0.0,
        float(noise_std),
        size=(int(sample_count), predicted.shape[1]),
    )
    scaled = (samples[:, None, :] - predicted[None, :, :]) / float(noise_std)
    log_likelihood = -0.5 * np.sum(scaled**2, axis=2)
    log_prior = np.full(len(prior), -np.inf, dtype=float)
    positive = prior > 0.0
    log_prior[positive] = np.log(prior[positive])
    log_evidence = logsumexp(log_likelihood + log_prior[None, :], axis=1)
    information = (
        log_likelihood[np.arange(int(sample_count)), sampled_models]
        - log_evidence
    )
    return InformationGainEstimate(
        value_nats=float(np.mean(information)),
        standard_error_nats=float(
            np.std(information, ddof=1) / np.sqrt(int(sample_count))
        ),
        prior_entropy_nats=entropy_nats(prior),
    )


def expected_hierarchical_information_gain(
    predictions: np.ndarray,
    model_weights: np.ndarray,
    noise_std: float,
    *,
    sample_count: int = 512,
    seed: int = 0,
) -> HierarchicalInformationGainEstimate:
    """Estimate joint model-family and within-family parameter information.

    ``predictions`` has shape ``(models, parameter_draws, observations)``.
    Parameter draws are treated as an equally weighted empirical posterior
    within each model family. The returned decomposition follows the chain
    rule ``I(Y; M, theta) = I(Y; M) + E_M[I(Y; theta | M)]``.
    """

    predicted = np.asarray(predictions, dtype=float)
    weights = normalize_weights(model_weights)
    if predicted.ndim != 3 or predicted.shape[0] != len(weights):
        raise ValueError(
            "Hierarchical predictions must have shape "
            "(models, parameter_draws, observations)."
        )
    if predicted.shape[1] < 2 or predicted.shape[2] < 1:
        raise ValueError(
            "Hierarchical information requires at least two parameter draws "
            "and one observation."
        )
    if not np.all(np.isfinite(predicted)):
        raise ValueError("Hierarchical predictions must be finite.")
    if noise_std <= 0.0 or not np.isfinite(noise_std):
        raise ValueError("noise_std must be finite and positive.")
    if sample_count < 2:
        raise ValueError("sample_count must be at least two.")

    rng = np.random.default_rng(int(seed))
    model_count, draw_count, observation_count = predicted.shape
    sampled_models = rng.choice(model_count, size=int(sample_count), p=weights)
    sampled_draws = rng.integers(0, draw_count, size=int(sample_count))
    samples = predicted[sampled_models, sampled_draws] + rng.normal(
        0.0,
        float(noise_std),
        size=(int(sample_count), observation_count),
    )

    scaled = (
        samples[:, None, None, :] - predicted[None, :, :, :]
    ) / float(noise_std)
    log_likelihood = -0.5 * np.sum(scaled**2, axis=3)
    log_predictive_by_model = logsumexp(log_likelihood, axis=2) - np.log(
        float(draw_count)
    )
    log_model_weights = np.full(model_count, -np.inf, dtype=float)
    positive = weights > 0.0
    log_model_weights[positive] = np.log(weights[positive])
    log_evidence = logsumexp(
        log_predictive_by_model + log_model_weights[None, :],
        axis=1,
    )
    row = np.arange(int(sample_count))
    generating_model_predictive = log_predictive_by_model[row, sampled_models]
    generating_draw_likelihood = log_likelihood[
        row,
        sampled_models,
        sampled_draws,
    ]
    structural_samples = generating_model_predictive - log_evidence
    parameter_samples = generating_draw_likelihood - generating_model_predictive
    joint_samples = generating_draw_likelihood - log_evidence

    def estimate(values: np.ndarray) -> tuple[float, float]:
        return (
            float(np.mean(values)),
            float(np.std(values, ddof=1) / np.sqrt(len(values))),
        )

    structural, structural_se = estimate(structural_samples)
    parameter, parameter_se = estimate(parameter_samples)
    joint, joint_se = estimate(joint_samples)
    return HierarchicalInformationGainEstimate(
        structural_information_nats=structural,
        parameter_information_nats=parameter,
        joint_information_nats=joint,
        structural_standard_error_nats=structural_se,
        parameter_standard_error_nats=parameter_se,
        joint_standard_error_nats=joint_se,
        prior_model_entropy_nats=entropy_nats(weights),
    )


def expected_scalar_model_information_gain(
    predictions: np.ndarray,
    prior_weights: np.ndarray,
    noise_std: float,
    *,
    quadrature_order: int = 32,
) -> InformationGainEstimate:
    """Evaluate scalar Gaussian model-discrimination information deterministically."""
    predicted = np.asarray(predictions, dtype=float)
    if predicted.ndim == 2 and predicted.shape[1] == 1:
        predicted = predicted[:, 0]
    return expected_scalar_gaussian_model_information_gain(
        predicted,
        np.full(len(predicted), float(noise_std), dtype=float),
        prior_weights,
        quadrature_order=quadrature_order,
    )


def expected_scalar_gaussian_model_information_gain(
    predictions: np.ndarray,
    predictive_std: np.ndarray,
    prior_weights: np.ndarray,
    *,
    quadrature_order: int = 32,
) -> InformationGainEstimate:
    """Return mutual information between model identity and one Gaussian response."""
    predicted = np.asarray(predictions, dtype=float)
    standard_deviations = np.asarray(predictive_std, dtype=float)
    prior = normalize_weights(prior_weights)
    if predicted.shape != (len(prior),) or not np.all(np.isfinite(predicted)):
        raise ValueError("Predictions require one finite scalar per model.")
    if standard_deviations.shape != predicted.shape or np.any(
        ~np.isfinite(standard_deviations) | (standard_deviations <= 0.0)
    ):
        raise ValueError("predictive_std requires one finite positive value per model.")
    if quadrature_order < 4:
        raise ValueError("quadrature_order must be at least four.")

    nodes, quadrature_weights = np.polynomial.hermite.hermgauss(
        int(quadrature_order)
    )
    samples = (
        predicted[:, None]
        + np.sqrt(2.0) * standard_deviations[:, None] * nodes[None, :]
    )
    scaled = (
        samples[:, :, None] - predicted[None, None, :]
    ) / standard_deviations[None, None, :]
    log_likelihood = -0.5 * scaled**2 - np.log(
        standard_deviations[None, None, :]
    )
    log_prior = np.full(len(prior), -np.inf, dtype=float)
    positive = prior > 0.0
    log_prior[positive] = np.log(prior[positive])
    log_evidence = logsumexp(log_likelihood + log_prior[None, None, :], axis=2)
    generating_log_likelihood = -(
        nodes[None, :] ** 2
    ) - np.log(standard_deviations[:, None])
    point_information = generating_log_likelihood - log_evidence
    expectation_by_model = np.sum(
        point_information * quadrature_weights[None, :] / np.sqrt(np.pi),
        axis=1,
    )
    information = float(np.sum(prior * expectation_by_model))
    prior_entropy = entropy_nats(prior)
    return InformationGainEstimate(
        value_nats=float(np.clip(information, 0.0, prior_entropy)),
        standard_error_nats=0.0,
        prior_entropy_nats=prior_entropy,
    )


def realized_model_information_gain(
    observations: np.ndarray,
    predictions: np.ndarray,
    prior_weights: np.ndarray,
    noise_std: float,
) -> tuple[float, np.ndarray]:
    prior = normalize_weights(prior_weights)
    posterior = posterior_model_weights(
        observations,
        predictions,
        prior,
        noise_std,
    )
    return entropy_nats(prior) - entropy_nats(posterior), posterior


def pairwise_rms_disagreement(
    predictions: np.ndarray,
    weights: np.ndarray,
) -> float:
    predicted = np.asarray(predictions, dtype=float)
    probabilities = normalize_weights(weights)
    if predicted.ndim != 2 or predicted.shape[0] != len(probabilities):
        raise ValueError("Predictions must have shape (models, observations).")
    mean = np.sum(probabilities[:, None] * predicted, axis=0)
    variance = np.sum(probabilities[:, None] * (predicted - mean) ** 2, axis=0)
    return float(np.sqrt(np.mean(variance)))

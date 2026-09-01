from __future__ import annotations

from typing import Any

import numpy as np

from asrc.constitutive.active_design import (
    expected_scalar_gaussian_model_information_gain,
    expected_scalar_model_information_gain,
    normalize_weights,
)


def normalized_design_distance(
    candidates: np.ndarray,
    observed: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    candidate_values = np.asarray(candidates, dtype=float)
    observed_values = np.asarray(observed, dtype=float)
    lower = np.asarray(lower_bounds, dtype=float)
    upper = np.asarray(upper_bounds, dtype=float)
    if candidate_values.ndim != 2 or observed_values.ndim != 2:
        raise ValueError("Candidate and observed designs must be two-dimensional.")
    if candidate_values.shape[1] != observed_values.shape[1]:
        raise ValueError("Candidate and observed designs require the same columns.")
    if lower.shape != upper.shape or lower.shape != (candidate_values.shape[1],):
        raise ValueError("One lower and upper bound is required per design variable.")
    scale = upper - lower
    if np.any(scale <= 0.0):
        raise ValueError("Design upper bounds must exceed lower bounds.")
    normalized_candidates = (candidate_values - lower) / scale
    normalized_observed = (observed_values - lower) / scale
    distances = np.sqrt(
        np.sum(
            (
                normalized_candidates[:, None, :]
                - normalized_observed[None, :, :]
            )
            ** 2,
            axis=2,
        )
    )
    return np.min(distances, axis=1)


def weighted_point_disagreement(
    predictions: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    predicted = np.asarray(predictions, dtype=float)
    probabilities = normalize_weights(np.asarray(weights, dtype=float))
    if predicted.ndim != 2 or predicted.shape[0] != len(probabilities):
        raise ValueError("Predictions must have shape (models, candidate points).")
    mean = np.sum(probabilities[:, None] * predicted, axis=0)
    variance = np.sum(
        probabilities[:, None] * (predicted - mean[None, :]) ** 2,
        axis=0,
    )
    return np.sqrt(np.maximum(variance, 0.0))


def local_parameter_posterior_covariance(
    fit_jacobian: np.ndarray,
    noise_std: float,
    prior_variance: float,
) -> np.ndarray:
    jacobian = np.asarray(fit_jacobian, dtype=float)
    if jacobian.ndim != 2 or not np.all(np.isfinite(jacobian)):
        raise ValueError("fit_jacobian must be a finite two-dimensional array.")
    if noise_std <= 0.0 or not np.isfinite(noise_std):
        raise ValueError("noise_std must be finite and positive.")
    if prior_variance <= 0.0 or not np.isfinite(prior_variance):
        raise ValueError("prior_variance must be finite and positive.")
    parameter_count = jacobian.shape[1]
    if parameter_count == 0:
        return np.empty((0, 0), dtype=float)
    precision = (
        jacobian.T @ jacobian / float(noise_std) ** 2
        + np.eye(parameter_count) / float(prior_variance)
    )
    return np.linalg.inv(precision)


def local_parameter_information_gain(
    candidate_jacobian: np.ndarray,
    posterior_covariance: np.ndarray,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    jacobian = np.asarray(candidate_jacobian, dtype=float)
    covariance = np.asarray(posterior_covariance, dtype=float)
    if jacobian.ndim != 2 or covariance.shape != (
        jacobian.shape[1],
        jacobian.shape[1],
    ):
        raise ValueError("Candidate Jacobian and covariance have incompatible shapes.")
    if not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(covariance)):
        raise ValueError("Parameter-information inputs must be finite.")
    if noise_std <= 0.0 or not np.isfinite(noise_std):
        raise ValueError("noise_std must be finite and positive.")
    parameter_variance = np.einsum(
        "ij,jk,ik->i", jacobian, covariance, jacobian
    )
    parameter_variance = np.maximum(parameter_variance, 0.0)
    information = 0.5 * np.log1p(parameter_variance / float(noise_std) ** 2)
    predictive_std = np.sqrt(float(noise_std) ** 2 + parameter_variance)
    return information, predictive_std


def joint_structure_parameter_information_gain(
    prediction_matrix: np.ndarray,
    predictive_std_matrix: np.ndarray,
    parameter_information_matrix: np.ndarray,
    model_weights: np.ndarray,
    *,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.asarray(prediction_matrix, dtype=float)
    predictive_std = np.asarray(predictive_std_matrix, dtype=float)
    parameter_information = np.asarray(parameter_information_matrix, dtype=float)
    weights = normalize_weights(np.asarray(model_weights, dtype=float))
    if predictions.ndim != 2 or predictions.shape[0] != len(weights):
        raise ValueError("Predictions must have shape (models, candidate points).")
    if predictive_std.shape != predictions.shape:
        raise ValueError("Predictive standard deviations must match predictions.")
    if parameter_information.shape != predictions.shape:
        raise ValueError("Parameter information must match predictions.")
    structural = np.empty(predictions.shape[1], dtype=float)
    for index in range(predictions.shape[1]):
        structural[index] = expected_scalar_gaussian_model_information_gain(
            predictions[:, index],
            predictive_std[:, index],
            weights,
            quadrature_order=quadrature_order,
        ).value_nats
    parameter = np.sum(weights[:, None] * parameter_information, axis=0)
    return structural, parameter, structural + parameter


def score_acquisition_pool(
    strategy: str,
    *,
    candidate_design: np.ndarray,
    observed_design: np.ndarray,
    prediction_matrix: np.ndarray,
    model_weights: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    noise_std: float,
    seed: int,
    information_gain_quadrature_order: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    candidate_count = len(candidate_design)
    if candidate_count < 1:
        raise ValueError("Acquisition pool must not be empty.")
    if prediction_matrix.shape[1] != candidate_count:
        raise ValueError("Prediction matrix does not match the acquisition pool.")
    if strategy == "space_filling_design":
        scores = normalized_design_distance(
            candidate_design,
            observed_design,
            lower_bounds,
            upper_bounds,
        )
        return scores, {"score_unit": "normalized_distance"}
    if strategy == "predictive_disagreement":
        return weighted_point_disagreement(prediction_matrix, model_weights), {
            "score_unit": "response"
        }
    if strategy == "expected_information_gain":
        scores = []
        errors = []
        for index in range(candidate_count):
            estimate = expected_scalar_model_information_gain(
                prediction_matrix[:, index],
                model_weights,
                noise_std,
                quadrature_order=information_gain_quadrature_order,
            )
            scores.append(estimate.value_nats)
            errors.append(estimate.standard_error_nats)
        return np.asarray(scores, dtype=float), {
            "score_unit": "nat",
            "estimator": "gauss_hermite_scalar_gaussian",
            "score_standard_errors": errors,
        }
    if strategy == "random_design":
        rng = np.random.default_rng(int(seed))
        return rng.random(candidate_count), {"score_unit": "random_priority"}
    raise ValueError(f"Unsupported acquisition strategy: {strategy}")


def select_acquisition_index(scores: np.ndarray, identifiers: list[str]) -> int:
    values = np.asarray(scores, dtype=float)
    if values.shape != (len(identifiers),) or not np.all(np.isfinite(values)):
        raise ValueError("Acquisition scores must be finite and match identifiers.")
    return min(
        range(len(identifiers)),
        key=lambda index: (-float(values[index]), str(identifiers[index])),
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(target, dtype=float)
    estimate = np.asarray(prediction, dtype=float)
    if truth.shape != estimate.shape or truth.ndim != 1:
        raise ValueError("Target and prediction must be equal-length vectors.")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(estimate)):
        raise ValueError("Target and prediction must be finite.")
    return float(np.sqrt(np.mean((truth - estimate) ** 2)))


@dataclass(frozen=True)
class ValidationEquivalence:
    best_index: int
    selected_index: int
    equivalent_indices: tuple[int, ...]
    validation_rmse: np.ndarray
    best_rmse_standard_error: float
    equivalence_threshold: float


def one_standard_error_equivalence(
    target: np.ndarray,
    prediction_matrix: np.ndarray,
    complexities: np.ndarray,
    identifiers: Sequence[str],
    *,
    bootstrap_samples: int,
    seed: int,
    standard_error_multiplier: float = 1.0,
) -> ValidationEquivalence:
    truth = np.asarray(target, dtype=float)
    predictions = np.asarray(prediction_matrix, dtype=float)
    complexity = np.asarray(complexities, dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] != len(truth):
        raise ValueError("Predictions must have shape (models, observations).")
    if predictions.shape[0] != len(complexity) or len(identifiers) != len(complexity):
        raise ValueError("One complexity and identifier are required per model.")
    if bootstrap_samples < 2:
        raise ValueError("At least two bootstrap samples are required.")
    if standard_error_multiplier < 0.0:
        raise ValueError("The standard-error multiplier must be non-negative.")
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(complexity)):
        raise ValueError("Predictions and complexities must be finite.")

    errors = np.asarray([_rmse(truth, row) for row in predictions], dtype=float)
    best_index = min(
        range(len(errors)),
        key=lambda index: (errors[index], complexity[index], str(identifiers[index])),
    )
    rng = np.random.default_rng(int(seed))
    bootstrap_indices = rng.integers(
        0,
        len(truth),
        size=(int(bootstrap_samples), len(truth)),
    )
    best_residual = truth - predictions[best_index]
    bootstrap_rmse = np.sqrt(
        np.mean(best_residual[bootstrap_indices] ** 2, axis=1)
    )
    standard_error = float(np.std(bootstrap_rmse, ddof=1))
    threshold = float(
        errors[best_index] + float(standard_error_multiplier) * standard_error
    )
    tolerance = 10.0 * np.finfo(float).eps * max(1.0, abs(threshold))
    equivalent = tuple(
        index for index, value in enumerate(errors) if value <= threshold + tolerance
    )
    selected_index = min(
        equivalent,
        key=lambda index: (complexity[index], errors[index], str(identifiers[index])),
    )
    return ValidationEquivalence(
        best_index=best_index,
        selected_index=selected_index,
        equivalent_indices=equivalent,
        validation_rmse=errors,
        best_rmse_standard_error=standard_error,
        equivalence_threshold=threshold,
    )


@dataclass(frozen=True)
class DeploymentAmbiguity:
    model_count: int
    maximum_pairwise_rmse: float
    normalized_maximum_pairwise_rmse: float
    ambiguous: bool


def deployment_prediction_ambiguity(
    prediction_matrix: np.ndarray,
    *,
    noise_std: float,
    noise_multiplier: float,
) -> DeploymentAmbiguity:
    predictions = np.asarray(prediction_matrix, dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] < 1:
        raise ValueError("Deployment predictions must be two-dimensional and non-empty.")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Deployment predictions must be finite.")
    if noise_std <= 0.0 or noise_multiplier < 0.0:
        raise ValueError("Noise scale must be positive and multiplier non-negative.")
    maximum = 0.0
    for first in range(len(predictions)):
        for second in range(first + 1, len(predictions)):
            maximum = max(
                maximum,
                float(
                    np.sqrt(
                        np.mean((predictions[first] - predictions[second]) ** 2)
                    )
                ),
            )
    normalized = maximum / float(noise_std)
    return DeploymentAmbiguity(
        model_count=len(predictions),
        maximum_pairwise_rmse=maximum,
        normalized_maximum_pairwise_rmse=normalized,
        ambiguous=bool(
            len(predictions) > 1 and maximum > noise_multiplier * float(noise_std)
        ),
    )

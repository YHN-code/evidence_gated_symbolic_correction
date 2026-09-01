from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Sequence

import numpy as np

from asrc.model_revision.ambiguity import deployment_prediction_ambiguity


@dataclass(frozen=True)
class EquivalentPredictionPair:
    left_id: str
    right_id: str
    prediction_rmse: float
    normalized_prediction_rmse: float

    def to_row(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceEquivalenceAssessment:
    candidate_count: int
    maximum_pairwise_rmse: float
    normalized_maximum_pairwise_rmse: float
    mean_pointwise_range: float
    normalized_mean_pointwise_range: float
    maximum_pointwise_range: float
    normalized_maximum_pointwise_range: float
    ambiguity_class: str
    active_query_indicated: bool
    active_query_reason: str
    recommended_query_index: int | None

    def to_row(self) -> dict[str, float | int | str | bool | None]:
        return asdict(self)


def evidence_equivalent_prediction_assessment(
    identifiers: Sequence[str],
    prediction_matrix: np.ndarray,
    *,
    noise_std: float,
    ambiguity_noise_multiplier: float,
    acquisition_available: bool,
) -> tuple[EvidenceEquivalenceAssessment, tuple[EquivalentPredictionPair, ...]]:
    predictions = np.asarray(prediction_matrix, dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] < 1:
        raise ValueError("Prediction matrix must have shape (candidates, points).")
    if len(identifiers) != predictions.shape[0]:
        raise ValueError("One identifier is required per prediction row.")
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Candidate identifiers must be non-empty and unique.")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Equivalent candidate predictions must be finite.")
    if noise_std <= 0.0 or ambiguity_noise_multiplier < 0.0:
        raise ValueError("Noise scale must be positive and multiplier non-negative.")

    ambiguity = deployment_prediction_ambiguity(
        predictions,
        noise_std=noise_std,
        noise_multiplier=ambiguity_noise_multiplier,
    )
    pointwise_range = np.ptp(predictions, axis=0)
    maximum_index = int(np.argmax(pointwise_range))
    maximum_range = float(pointwise_range[maximum_index])
    mean_range = float(np.mean(pointwise_range))
    pairs = tuple(
        EquivalentPredictionPair(
            left_id=str(identifiers[left]),
            right_id=str(identifiers[right]),
            prediction_rmse=float(
                np.sqrt(np.mean((predictions[left] - predictions[right]) ** 2))
            ),
            normalized_prediction_rmse=float(
                np.sqrt(np.mean((predictions[left] - predictions[right]) ** 2))
                / noise_std
            ),
        )
        for left, right in combinations(range(len(identifiers)), 2)
    )

    if len(identifiers) == 1:
        ambiguity_class = "identified_singleton"
        active = False
        reason = "one_validation_admissible_candidate"
        query_index: int | None = None
    elif not ambiguity.ambiguous:
        ambiguity_class = "prediction_equivalent"
        active = False
        reason = "admissible_candidates_agree_within_noise_threshold"
        query_index = None
    elif not acquisition_available:
        ambiguity_class = "structurally_ambiguous"
        active = False
        reason = "candidate_predictions_diverge_but_no_query_is_available"
        query_index = None
    else:
        ambiguity_class = "structurally_ambiguous"
        active = True
        reason = "validation_equivalent_candidates_diverge_in_query_domain"
        query_index = maximum_index

    assessment = EvidenceEquivalenceAssessment(
        candidate_count=len(identifiers),
        maximum_pairwise_rmse=ambiguity.maximum_pairwise_rmse,
        normalized_maximum_pairwise_rmse=(
            ambiguity.normalized_maximum_pairwise_rmse
        ),
        mean_pointwise_range=mean_range,
        normalized_mean_pointwise_range=mean_range / noise_std,
        maximum_pointwise_range=maximum_range,
        normalized_maximum_pointwise_range=maximum_range / noise_std,
        ambiguity_class=ambiguity_class,
        active_query_indicated=active,
        active_query_reason=reason,
        recommended_query_index=query_index,
    )
    return assessment, pairs

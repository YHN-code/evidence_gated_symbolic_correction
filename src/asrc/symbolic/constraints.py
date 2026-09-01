from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from asrc.features.angle_features import make_angle_features
from asrc.response_semantics import resolve_response_semantics
from asrc.symbolic.formula_utils import design_matrix, has_endpoint_singularity


@dataclass(frozen=True)
class ConstraintReport:
    positive_strength_violations: int
    endpoint_singularity: bool
    finite_prediction_violations: int
    max_abs_correction: float
    response_kind: str = "positive_strength"
    positivity_check_applicable: bool = True

    @property
    def total_violations(self) -> int:
        return (
            self.positive_strength_violations
            + int(self.endpoint_singularity)
            + self.finite_prediction_violations
        )

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "physical_violations": self.total_violations,
            "positive_strength_violations": self.positive_strength_violations,
            "endpoint_singularity": self.endpoint_singularity,
            "finite_prediction_violations": self.finite_prediction_violations,
            "max_abs_correction": self.max_abs_correction,
            "response_kind": self.response_kind,
            "positivity_check_applicable": self.positivity_check_applicable,
        }


@dataclass(frozen=True)
class ConstraintEvaluator:
    observed_features: pd.DataFrame
    endpoint_features: pd.DataFrame
    y_base: np.ndarray
    response_kind: str
    require_observed_positive: bool

    def evaluate(
        self,
        terms: list[str],
        coefficients: dict[str, float],
    ) -> ConstraintReport:
        coef = np.asarray([coefficients.get(term, 0.0) for term in terms], dtype=float)
        correction = design_matrix(self.observed_features, terms) @ coef
        corrected = self.y_base + correction
        finite_bad = int(
            (~np.isfinite(corrected)).sum() + (~np.isfinite(correction)).sum()
        )
        positive_bad = (
            int((corrected <= 0.0).sum())
            if self.require_observed_positive
            else 0
        )

        endpoint_bad = has_endpoint_singularity(terms)
        endpoint_correction = design_matrix(self.endpoint_features, terms) @ coef
        if not np.isfinite(endpoint_correction).all():
            endpoint_bad = True

        return ConstraintReport(
            positive_strength_violations=positive_bad,
            endpoint_singularity=endpoint_bad,
            finite_prediction_violations=finite_bad,
            max_abs_correction=(
                float(np.nanmax(np.abs(correction))) if len(correction) else 0.0
            ),
            response_kind=self.response_kind,
            positivity_check_applicable=self.require_observed_positive,
        )


def prepare_constraint_evaluator(case: Any) -> ConstraintEvaluator:
    semantics = resolve_response_semantics(case)
    beta_grid = [0.0, 90.0]
    psi_values = sorted(case.frame["psi_deg"].unique().tolist()) if "psi_deg" in case.frame.columns else [0.0]
    grid = pd.DataFrame({"beta_deg": np.repeat(beta_grid, len(psi_values)), "psi_deg": psi_values * len(beta_grid)})
    return ConstraintEvaluator(
        observed_features=make_angle_features(case.frame, include_forbidden=True),
        endpoint_features=make_angle_features(grid, include_forbidden=True),
        y_base=case.frame[case.y_base].to_numpy(float),
        response_kind=str(semantics["kind"]),
        require_observed_positive=bool(semantics["require_observed_positive"]),
    )


def check_physical_constraints(
    case: Any,
    terms: list[str],
    coefficients: dict[str, float],
) -> ConstraintReport:
    return prepare_constraint_evaluator(case).evaluate(terms, coefficients)

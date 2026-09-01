from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from asrc.symbolic.formula_utils import design_matrix, formula_complexity, formula_text


_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_LINEAR_PIECE = re.compile(rf"([+-]?{_NUMBER})(?:\*([A-Za-z][A-Za-z0-9_]*))?")


# Each source feature is expanded into one fixed, linearly independent reporting
# basis. These are exact identities following from sin(theta)^2 + cos(theta)^2 = 1.
_CANONICAL_EXPANSIONS: dict[str, tuple[str, dict[str, float]]] = {
    "cos2_beta": ("TRIG-BETA-001", {"1": 1.0, "sin2_beta": -1.0}),
    "cos2_psi": ("TRIG-PSI-001", {"1": 1.0, "sin2_psi": -1.0}),
    "sin2_beta_cos2_psi": (
        "TRIG-INTERACTION-001",
        {"sin2_beta": 1.0, "sin2_beta_sin2_psi": -1.0},
    ),
    "cos2_beta_sin2_psi": (
        "TRIG-INTERACTION-002",
        {"sin2_psi": 1.0, "sin2_beta_sin2_psi": -1.0},
    ),
    "cos2_beta_cos2_psi": (
        "TRIG-INTERACTION-003",
        {"1": 1.0, "sin2_beta": -1.0, "sin2_psi": -1.0, "sin2_beta_sin2_psi": 1.0},
    ),
    "m_beta_cos2_psi": (
        "TRIG-MODULATION-001",
        {"m_beta": 1.0, "m_beta_sin2_psi": -1.0},
    ),
    "m2_beta_cos2_psi": (
        "TRIG-MODULATION-002",
        {"m2_beta": 1.0, "m2_beta_sin2_psi": -1.0},
    ),
    "m_psi_cos2_beta": (
        "TRIG-MODULATION-003",
        {"m_psi": 1.0, "m_psi_sin2_beta": -1.0},
    ),
    "m2_psi_cos2_beta": (
        "TRIG-MODULATION-004",
        {"m2_psi": 1.0, "m2_psi_sin2_beta": -1.0},
    ),
}


@dataclass(frozen=True)
class CanonicalFormula:
    terms: tuple[str, ...]
    coefficients: dict[str, float]
    formula: str
    applied_identities: tuple[str, ...]
    original_complexity: int
    canonical_complexity: int


def parse_linear_formula(text: str) -> tuple[list[str], dict[str, float]]:
    """Parse formulas emitted by ``formula_text`` without evaluating code."""

    compact = re.sub(r"\s+", "", str(text))
    if not compact:
        raise ValueError("Formula text is empty.")
    matches = list(_LINEAR_PIECE.finditer(compact))
    if not matches or "".join(match.group(0) for match in matches) != compact:
        raise ValueError(f"Formula is not a supported linear feature expression: {text!r}")

    coefficients: dict[str, float] = {}
    order: list[str] = []
    for match in matches:
        term = match.group(2) or "1"
        if term not in order:
            order.append(term)
        coefficients[term] = coefficients.get(term, 0.0) + float(match.group(1))
    return order, coefficients


def canonicalize_linear_formula(
    terms: list[str] | tuple[str, ...],
    coefficients: Mapping[str, float],
    *,
    zero_atol: float = 1e-12,
) -> CanonicalFormula:
    """Rewrite a linear feature formula into the fixed reporting basis.

    Coefficients are transformed algebraically. No fitting, ranking, or data
    access occurs in this function.
    """

    canonical: dict[str, float] = {}
    identities: list[str] = []
    for term in terms:
        coefficient = float(coefficients.get(term, 0.0))
        if abs(coefficient) <= zero_atol:
            continue
        identity, expansion = _CANONICAL_EXPANSIONS.get(term, ("", {term: 1.0}))
        if identity:
            identities.append(identity)
        for target, multiplier in expansion.items():
            canonical[target] = canonical.get(target, 0.0) + coefficient * multiplier

    canonical = {term: value for term, value in canonical.items() if abs(value) > zero_atol}
    if not canonical:
        canonical = {"1": 0.0}
    ordered_terms = tuple(sorted(canonical, key=lambda term: (term != "1", term)))
    ordered_coefficients = {term: canonical[term] for term in ordered_terms}
    original_terms = list(dict.fromkeys(terms))
    return CanonicalFormula(
        terms=ordered_terms,
        coefficients=ordered_coefficients,
        formula=formula_text(list(ordered_terms), ordered_coefficients),
        applied_identities=tuple(dict.fromkeys(identities)),
        original_complexity=formula_complexity(original_terms, dict(coefficients)),
        canonical_complexity=formula_complexity(list(ordered_terms), ordered_coefficients),
    )


def verify_formula_equivalence(
    features: pd.DataFrame,
    original_terms: list[str] | tuple[str, ...],
    original_coefficients: Mapping[str, float],
    canonical: CanonicalFormula,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-12,
) -> float:
    """Return maximum prediction difference and fail if equivalence is lost."""

    original = design_matrix(features, list(original_terms)) @ np.asarray(
        [float(original_coefficients.get(term, 0.0)) for term in original_terms]
    )
    transformed = design_matrix(features, list(canonical.terms)) @ np.asarray(
        [canonical.coefficients[term] for term in canonical.terms]
    )
    maximum = float(np.max(np.abs(original - transformed))) if len(features) else 0.0
    if not np.allclose(original, transformed, atol=atol, rtol=rtol):
        raise ValueError(f"Canonical formula failed prediction-equivalence verification (max diff={maximum:.6g}).")
    return maximum

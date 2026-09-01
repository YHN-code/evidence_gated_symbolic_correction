from __future__ import annotations

import re

import numpy as np
import pandas as pd


TERM_COMPLEXITY = {
    "1": 1,
    "m_beta": 1,
    "m2_beta": 1,
    "m4_beta": 2,
    "sat_m2_beta_b3": 3,
    "sin2beta": 1,
    "cos2beta": 1,
    "sin2_beta": 1,
    "cos2_beta": 1,
    "m_psi": 1,
    "m2_psi": 1,
    "m4_psi": 2,
    "sin2psi": 1,
    "cos2psi": 1,
    "sin2_psi": 1,
    "cos2_psi": 1,
    "m_beta_m_psi": 2,
    "m2_beta_m2_psi": 3,
    "m2_beta_m_psi": 3,
    "m_beta_m2_psi": 3,
    "sin2_beta_sin2_psi": 2,
    "sin2_beta_cos2_psi": 2,
    "cos2_beta_sin2_psi": 2,
    "cos2_beta_cos2_psi": 2,
    "m_beta_sin2_psi": 2,
    "m_beta_cos2_psi": 2,
    "m_psi_sin2_beta": 2,
    "m_psi_cos2_beta": 2,
    "m2_beta_sin2_psi": 3,
    "m2_beta_cos2_psi": 3,
    "m2_psi_sin2_beta": 3,
    "m2_psi_cos2_beta": 3,
    "sin_beta": 1,
    "cos_beta": 1,
    "sin_psi": 1,
    "cos_psi": 1,
    "sin_2_beta_plus_psi": 2,
    "cos_2_beta_plus_psi": 2,
    "sin_2_beta_minus_psi": 2,
    "cos_2_beta_minus_psi": 2,
    "sin_4_beta_plus_psi": 3,
    "cos_4_beta_plus_psi": 3,
    "sin_4_beta_minus_psi": 3,
    "cos_4_beta_minus_psi": 3,
    "sin_6_beta_plus_psi": 3,
    "cos_6_beta_plus_psi": 3,
    "sin_6_beta_minus_psi": 3,
    "cos_6_beta_minus_psi": 3,
    "tan_beta": 4,
    "reciprocal_sin_beta": 4,
    "reciprocal_cos_beta": 4,
}

FORBIDDEN_PATTERNS = ("tan_", "reciprocal_", "/sin", "/cos")


def design_matrix(features: pd.DataFrame, terms: list[str]) -> np.ndarray:
    columns = []
    for term in terms:
        if term == "1":
            columns.append(np.ones(len(features), dtype=float))
        else:
            columns.append(features[term].to_numpy(float))
    matrix = np.column_stack(columns)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def formula_complexity(terms: list[str], coefficients: dict[str, float] | None = None) -> int:
    total = 0
    for term in terms:
        if coefficients is not None and abs(coefficients.get(term, 0.0)) < 1e-10:
            continue
        total += TERM_COMPLEXITY.get(term, 2)
    return int(total)


def formula_text(terms: list[str], coefficients: dict[str, float]) -> str:
    pieces = []
    for term in terms:
        coef = coefficients.get(term, 0.0)
        if abs(coef) < 1e-10:
            continue
        if term == "1":
            pieces.append(f"{coef:+.4g}")
        else:
            pieces.append(f"{coef:+.4g}*{term}")
    return " ".join(pieces).lstrip("+").strip() or "0"


def has_endpoint_singularity(terms: list[str]) -> bool:
    joined = " ".join(terms)
    return any(pattern in joined for pattern in FORBIDDEN_PATTERNS)


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")

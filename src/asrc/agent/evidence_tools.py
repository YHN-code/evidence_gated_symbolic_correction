from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from asrc.evaluation.validation_split import leave_one_group_indices
from asrc.features.angle_features import make_angle_features
from asrc.response_semantics import resolve_response_semantics
from asrc.symbolic.formula_utils import design_matrix, formula_complexity, formula_text
from asrc.symbolic.sparse_regression import fit_fixed_terms


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x.T @ x
    penalty = np.eye(xtx.shape[0], dtype=float) * float(alpha)
    if len(penalty):
        penalty[0, 0] = 0.0
    try:
        return np.linalg.solve(xtx + penalty, x.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx + penalty) @ (x.T @ y)


def _rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2))) if len(values) else float("nan")


def _formula_terms(result: Any) -> list[str]:
    return [str(term) for term in result.best.get("terms", [])]


def _formula_coefficients(result: Any) -> dict[str, float]:
    return {str(term): float(value) for term, value in result.best.get("coefficients", {}).items()}


def _post_correction_residual(result: Any) -> np.ndarray:
    return result.predictions["residual_after_correction_MPa"].to_numpy(float)


def _cv_fold_rows(case: Any, terms: list[str], alpha: float, split_col: str) -> list[dict[str, Any]]:
    if split_col not in case.frame.columns:
        return []
    features = make_angle_features(case.frame, include_forbidden=False)
    residual = case.frame[case.residual].to_numpy(float)
    y_true = case.frame[case.y_true].to_numpy(float)
    y_base = case.frame[case.y_base].to_numpy(float)
    rows: list[dict[str, Any]] = []
    for train_idx, test_idx, value in leave_one_group_indices(case.frame, split_col):
        coef = _fit_ridge(design_matrix(features.iloc[train_idx], terms), residual[train_idx], alpha)
        correction = design_matrix(features.iloc[test_idx], terms) @ coef
        error = y_true[test_idx] - (y_base[test_idx] + correction)
        rows.append(
            {
                "axis": split_col,
                "held_out_angle_deg": float(value),
                "n_test": int(len(test_idx)),
                "rmse": _rmse(error),
                "mean_error": float(np.mean(error)),
                "max_abs_error": float(np.max(np.abs(error))),
            }
        )
    return rows


def screen_candidate_families(
    case: Any,
    result: Any,
    grammar_index: list[dict[str, Any]],
    *,
    min_relative_improvement: float = 0.01,
) -> dict[str, Any]:
    residual = _post_correction_residual(result)
    current_rmse = _rmse(residual)
    features = make_angle_features(case.frame, include_forbidden=False)
    current_terms = set(_formula_terms(result))
    rows: list[dict[str, Any]] = []
    residual_std = float(np.std(residual, ddof=0))
    for family in grammar_index:
        terms = [term for term in family.get("terms", []) if term in features.columns and term not in current_terms]
        if not terms:
            rows.append(
                {
                    "family_id": str(family["id"]),
                    "terms": [],
                    "screening_rmse": current_rmse,
                    "relative_rmse_reduction": 0.0,
                    "max_abs_correlation": 0.0,
                    "status": "already_represented_or_unavailable",
                }
            )
            continue
        x = design_matrix(features, terms)
        coef = np.linalg.pinv(x) @ residual
        screened_residual = residual - x @ coef
        correlations = []
        for term in terms:
            values = features[term].to_numpy(float)
            if residual_std <= 1e-12 or float(np.std(values, ddof=0)) <= 1e-12:
                correlations.append(0.0)
            else:
                corr = float(np.corrcoef(values, residual)[0, 1])
                correlations.append(abs(corr) if np.isfinite(corr) else 0.0)
        screening_rmse = _rmse(screened_residual)
        rows.append(
            {
                "family_id": str(family["id"]),
                "terms": terms,
                "screening_rmse": screening_rmse,
                "relative_rmse_reduction": float((current_rmse - screening_rmse) / current_rmse) if current_rmse > 0 else 0.0,
                "max_abs_correlation": max(correlations, default=0.0),
                "status": "screened",
            }
        )
    rows.sort(key=lambda row: (-float(row["relative_rmse_reduction"]), -float(row["max_abs_correlation"]), row["family_id"]))
    threshold = max(0.0, float(min_relative_improvement))
    actionable = [
        row["family_id"]
        for row in rows
        if row["status"] == "screened" and float(row["relative_rmse_reduction"]) >= threshold
    ]
    return {
        "formula": str(result.best.get("formula", "")),
        "current_rmse": current_rmse,
        "ranked_families": rows,
        "recommended_family_id": rows[0]["family_id"] if rows else None,
        "min_relative_improvement": threshold,
        "actionable_family_ids": actionable,
        "resolved": not actionable,
    }


def inspect_validation_failures(case: Any, result: Any) -> dict[str, Any]:
    terms = _formula_terms(result)
    alpha = float(result.best.get("alpha", 0.0))
    folds = [
        *_cv_fold_rows(case, terms, alpha, "beta_deg"),
        *_cv_fold_rows(case, terms, alpha, "psi_deg"),
    ]
    ranked = sorted(folds, key=lambda row: float(row["rmse"]), reverse=True)
    return {
        "formula": str(result.best.get("formula", "")),
        "fold_count": len(folds),
        "folds": folds,
        "worst_folds": ranked[:5],
        "worst_fold_rmse": float(ranked[0]["rmse"]) if ranked else None,
    }


def test_term_stability(
    case: Any,
    result: Any,
    *,
    n_bootstrap: int = 64,
    seed: int = 20260713,
) -> dict[str, Any]:
    n_bootstrap = max(16, min(int(n_bootstrap), 256))
    rng = np.random.default_rng(int(seed))
    features = make_angle_features(case.frame, include_forbidden=False)
    terms = _formula_terms(result)
    alpha = float(result.best.get("alpha", 0.0))
    residual = case.frame[case.residual].to_numpy(float)
    x = design_matrix(features, terms)
    samples = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(x), size=len(x))
        samples.append(_fit_ridge(x[index], residual[index], alpha))
    coefficients = np.asarray(samples, dtype=float)
    reference = np.asarray([_formula_coefficients(result)[term] for term in terms], dtype=float)
    rows = []
    for idx, term in enumerate(terms):
        values = coefficients[:, idx]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        denominator = max(abs(mean), abs(float(reference[idx])), 1e-8)
        sign = np.sign(reference[idx])
        sign_consistency = float(np.mean(np.sign(values) == sign)) if sign != 0 else float(np.mean(np.abs(values) <= std))
        rows.append(
            {
                "term": term,
                "reference_coefficient": float(reference[idx]),
                "bootstrap_mean": mean,
                "bootstrap_std": std,
                "coefficient_cv": float(std / denominator),
                "sign_consistency": sign_consistency,
            }
        )
    tested = [row for row in rows if row["term"] != "1"]
    stable = bool(tested) and all(float(row["sign_consistency"]) >= 0.8 and float(row["coefficient_cv"]) <= 1.0 for row in tested)
    return {
        "formula": str(result.best.get("formula", "")),
        "n_bootstrap": n_bootstrap,
        "seed": int(seed),
        "terms": rows,
        "stable": stable,
        "minimum_sign_consistency": min((float(row["sign_consistency"]) for row in tested), default=None),
        "maximum_coefficient_cv": max((float(row["coefficient_cv"]) for row in tested), default=None),
    }


def _aggregate_cv_rmse(case: Any, terms: list[str], alpha: float) -> float:
    rows = [*_cv_fold_rows(case, terms, alpha, "beta_deg"), *_cv_fold_rows(case, terms, alpha, "psi_deg")]
    by_axis: dict[str, list[float]] = {}
    for row in rows:
        by_axis.setdefault(str(row["axis"]), []).append(float(row["rmse"]) ** 2)
    axis_rmse = [float(np.sqrt(np.mean(values))) for values in by_axis.values() if values]
    return float(np.mean(axis_rmse)) if axis_rmse else float("nan")


def challenge_formula_complexity(
    case: Any,
    result: Any,
    *,
    noninferiority_relative: float = 0.01,
    max_challenges: int | None = None,
) -> dict[str, Any]:
    terms = _formula_terms(result)
    coefficients = _formula_coefficients(result)
    removable = [term for term in terms if term != "1"]
    removable.sort(key=lambda term: (abs(float(coefficients.get(term, 0.0))), term))
    total_removable_terms = len(removable)
    if max_challenges is not None:
        removable = removable[: max(0, int(max_challenges))]
    alpha = float(result.best.get("alpha", 0.0))
    current_validation = float(result.best.get("selection_validation_rmse", float("nan")))
    features = make_angle_features(case.frame, include_forbidden=False)
    residual = case.frame[case.residual].to_numpy(float)
    rows = []
    for removed in removable:
        reduced_terms = [term for term in terms if term != removed]
        if not reduced_terms:
            continue
        coef = _fit_ridge(design_matrix(features, reduced_terms), residual, alpha)
        fitted = design_matrix(features, reduced_terms) @ coef
        reduced_coefficients = {term: float(value) for term, value in zip(reduced_terms, coef)}
        rows.append(
            {
                "removed_term": removed,
                "terms": reduced_terms,
                "formula": formula_text(reduced_terms, reduced_coefficients),
                "training_rmse": _rmse(residual - fitted),
                "validation_rmse": _aggregate_cv_rmse(case, reduced_terms, alpha),
                "complexity": formula_complexity(reduced_terms, reduced_coefficients),
            }
        )
    valid = [row for row in rows if np.isfinite(float(row["validation_rmse"]))]
    valid.sort(key=lambda row: (float(row["validation_rmse"]), int(row["complexity"])))
    best_simpler = valid[0] if valid else None
    threshold = current_validation * (1.0 + max(0.0, float(noninferiority_relative)))
    simplification_available = bool(best_simpler and np.isfinite(current_validation) and float(best_simpler["validation_rmse"]) <= threshold)
    return {
        "formula": str(result.best.get("formula", "")),
        "current_validation_rmse": current_validation if np.isfinite(current_validation) else None,
        "noninferiority_relative": float(noninferiority_relative),
        "best_simpler": best_simpler,
        "all_challenges": valid,
        "validation_candidates_evaluated": len(valid),
        "total_removable_terms": total_removable_terms,
        "completed": len(valid) == total_removable_terms,
        "simplification_available": simplification_available,
        "current_coefficients": coefficients,
    }


def stress_test_boundaries(case: Any, result: Any, *, dense_points: int = 91) -> dict[str, Any]:
    semantics = resolve_response_semantics(case)
    dense_points = max(21, min(int(dense_points), 181))
    beta = np.linspace(0.0, 90.0, dense_points)
    if "psi_deg" in case.frame.columns:
        beta_grid, psi_grid = np.meshgrid(beta, beta, indexing="ij")
        dense = pd.DataFrame({"beta_deg": beta_grid.ravel(), "psi_deg": psi_grid.ravel()})
    else:
        dense = pd.DataFrame({"beta_deg": beta})
    features = make_angle_features(dense, include_forbidden=False)
    terms = _formula_terms(result)
    coefficients = _formula_coefficients(result)
    correction = design_matrix(features, terms) @ np.asarray([coefficients[term] for term in terms], dtype=float)
    observed = result.predictions["correction_MPa"].to_numpy(float)
    observed_max = max(float(np.max(np.abs(observed))), 1e-12)
    dense_max = float(np.max(np.abs(correction)))
    amplification = dense_max / observed_max
    finite = bool(np.isfinite(correction).all())
    max_amplification = float(semantics["boundary_max_correction_amplification"])
    observed_corrected = case.frame[case.y_base].to_numpy(float) + observed
    observed_minimum = float(np.min(observed_corrected))
    dense_positive_applicable = bool(semantics["require_dense_positive"])
    conservative_lower_bound = (
        float(case.frame[case.y_base].astype(float).min() + np.min(correction))
        if dense_positive_applicable
        else None
    )
    dense_positive_passed = (
        bool(conservative_lower_bound is not None and conservative_lower_bound > 0.0)
        if dense_positive_applicable
        else None
    )
    observed_positive_passed = (
        observed_minimum > 0.0
        if semantics["require_observed_positive"]
        else None
    )
    passed = (
        finite
        and amplification <= max_amplification
        and dense_positive_passed is not False
        and observed_positive_passed is not False
    )
    return {
        "formula": str(result.best.get("formula", "")),
        "response_kind": semantics["kind"],
        "dense_points_per_axis": dense_points,
        "evaluated_points": int(len(dense)),
        "finite": finite,
        "max_abs_correction": dense_max,
        "observed_to_dense_amplification": float(amplification),
        "max_allowed_correction_amplification": max_amplification,
        "observed_corrected_response_minimum": observed_minimum,
        "observed_positive_check_applicable": bool(
            semantics["require_observed_positive"]
        ),
        "observed_positive_passed": observed_positive_passed,
        "dense_positive_check_applicable": dense_positive_applicable,
        "dense_positive_passed": dense_positive_passed,
        "conservative_response_lower_bound": conservative_lower_bound,
        # Kept for compatibility with existing audit readers.
        "conservative_strength_lower_bound": conservative_lower_bound,
        "passed": passed,
    }


def test_remaining_structure(
    case: Any,
    result: Any,
    candidate_terms: list[str],
    *,
    n_permutations: int = 199,
    seed: int = 20260713,
    min_effect: float = 0.2,
    min_validation_improvement_relative: float = 0.01,
    max_validation_candidates: int | None = 1,
) -> dict[str, Any]:
    n_permutations = max(49, min(int(n_permutations), 999))
    residual = _post_correction_residual(result)
    features = make_angle_features(case.frame, include_forbidden=False)
    current_terms = set(_formula_terms(result))
    terms = [term for term in dict.fromkeys(candidate_terms) if term in features.columns and term not in current_terms]
    standardized = []
    usable_terms = []
    residual_centered = residual - np.mean(residual)
    residual_norm = float(np.linalg.norm(residual_centered))
    for term in terms:
        values = features[term].to_numpy(float)
        centered = values - np.mean(values)
        norm = float(np.linalg.norm(centered))
        if norm <= 1e-12 or residual_norm <= 1e-12:
            continue
        standardized.append(centered / norm)
        usable_terms.append(term)
    if not standardized:
        return {
            "formula": str(result.best.get("formula", "")),
            "tested_terms": [],
            "max_abs_correlation": 0.0,
            "max_term": None,
            "permutation_p_value": 1.0,
            "n_permutations": n_permutations,
            "significant_structure": False,
            "predictively_actionable": False,
            "validation_candidates_evaluated": 0,
            "validation_check_completed": False,
            "resolved": True,
        }
    matrix = np.column_stack(standardized)
    residual_unit = residual_centered / residual_norm
    observed = np.abs(matrix.T @ residual_unit)
    observed_max = float(np.max(observed))
    rng = np.random.default_rng(int(seed))
    exceed = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(residual_unit)
        if float(np.max(np.abs(matrix.T @ permuted))) >= observed_max:
            exceed += 1
    p_value = float((exceed + 1) / (n_permutations + 1))
    significant = bool(p_value < 0.05 and observed_max >= float(min_effect))
    max_term = usable_terms[int(np.argmax(observed))]
    current_validation = float(
        result.best.get("selection_validation_rmse", float("nan"))
    )
    augmented = None
    relative_validation_improvement = None
    predictively_actionable = False
    validation_check_allowed = (
        max_validation_candidates is None
        or int(max_validation_candidates) > 0
    )
    if significant and validation_check_allowed:
        augmented = fit_fixed_terms(
            case,
            [*_formula_terms(result), max_term],
            alpha=float(result.best.get("alpha", 0.0)),
            mode=str(getattr(result, "mode", "asrc")),
            enforce_physical=True,
            candidate_id=f"remaining_structure_{max_term}",
        )
        augmented_validation = float(
            augmented.best.get("selection_validation_rmse", float("nan"))
        )
        if (
            np.isfinite(current_validation)
            and current_validation > 0.0
            and np.isfinite(augmented_validation)
        ):
            relative_validation_improvement = float(
                (current_validation - augmented_validation)
                / current_validation
            )
        predictively_actionable = bool(
            int(augmented.best.get("physical_violations", 0)) == 0
            and relative_validation_improvement is not None
            and relative_validation_improvement
            >= max(0.0, float(min_validation_improvement_relative))
        )
    return {
        "formula": str(result.best.get("formula", "")),
        "tested_terms": usable_terms,
        "max_abs_correlation": observed_max,
        "max_term": max_term,
        "permutation_p_value": p_value,
        "n_permutations": n_permutations,
        "min_effect": float(min_effect),
        "significant_structure": significant,
        "current_validation_rmse": (
            current_validation if np.isfinite(current_validation) else None
        ),
        "augmented_formula": (
            str(augmented.best.get("formula", "")) if augmented else None
        ),
        "augmented_validation_rmse": (
            float(augmented.best["selection_validation_rmse"])
            if augmented is not None
            else None
        ),
        "augmented_physical_violations": (
            int(augmented.best.get("physical_violations", 0))
            if augmented is not None
            else None
        ),
        "relative_validation_improvement": relative_validation_improvement,
        "min_validation_improvement_relative": float(
            min_validation_improvement_relative
        ),
        "predictively_actionable": predictively_actionable,
        "validation_candidates_evaluated": int(augmented is not None),
        "validation_check_completed": bool(augmented is not None),
        "resolved": bool(
            not significant
            or (augmented is not None and not predictively_actionable)
        ),
    }

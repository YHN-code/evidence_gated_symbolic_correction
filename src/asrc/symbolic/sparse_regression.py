from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from asrc.evaluation.metrics import regression_metrics
from asrc.evaluation.validation_split import leave_one_group_indices
from asrc.features.angle_features import make_angle_features
from asrc.symbolic.candidate_library import candidate_terms, enumerate_term_sets
from asrc.symbolic.constraints import (
    ConstraintReport,
    check_physical_constraints,
    prepare_constraint_evaluator,
)
from asrc.symbolic.formula_utils import design_matrix, formula_complexity, formula_text


@dataclass(frozen=True)
class SearchResult:
    mode: str
    best: dict[str, Any]
    candidates: pd.DataFrame
    predictions: pd.DataFrame


def fit_fixed_terms(
    case: Any,
    terms: list[str],
    *,
    alpha: float = 0.0,
    mode: str = "asrc",
    enforce_physical: bool = True,
    candidate_id: str = "fixed_terms",
) -> SearchResult:
    """Refit and fully evaluate one already-selected symbolic term set."""
    features = make_angle_features(case.frame, include_forbidden=mode == "generic")
    selected_terms = list(dict.fromkeys(str(term) for term in terms))
    if "1" not in selected_terms:
        selected_terms.insert(0, "1")
    unknown = [term for term in selected_terms if term != "1" and term not in features.columns]
    if unknown:
        raise ValueError(f"Unknown fixed formula terms: {unknown}")

    residual = case.frame[case.residual].to_numpy(float)
    y_true = case.frame[case.y_true].to_numpy(float)
    y_base = case.frame[case.y_base].to_numpy(float)
    x = design_matrix(features, selected_terms)
    coefficients_array = _fit_ridge(x, residual, float(alpha))
    coefficients = {
        term: float(value) for term, value in zip(selected_terms, coefficients_array)
    }
    correction = x @ coefficients_array
    y_pred = y_base + correction
    metrics = regression_metrics(y_true, y_pred)
    if enforce_physical:
        report = check_physical_constraints(case, selected_terms, coefficients)
    else:
        report = ConstraintReport(
            positive_strength_violations=0,
            endpoint_singularity=False,
            finite_prediction_violations=0,
            max_abs_correction=float(np.nanmax(np.abs(correction))) if len(correction) else 0.0,
        )
    beta_rmse, psi_rmse, aggregate = _validation_rmse(
        case, features, selected_terms, float(alpha)
    )
    accepted = (not enforce_physical) or report.total_violations == 0
    best = {
        "candidate_id": candidate_id,
        "mode": mode,
        "terms": selected_terms,
        "alpha": float(alpha),
        "formula": formula_text(selected_terms, coefficients),
        "coefficients": coefficients,
        "rmse": metrics.rmse,
        "mae": metrics.mae,
        "mape": metrics.mape,
        "r2": metrics.r2,
        "max_abs_error": metrics.max_abs_error,
        "leave_one_beta_rmse": beta_rmse,
        "leave_one_psi_rmse": psi_rmse,
        "selection_validation_rmse": aggregate,
        "selection_metric": "group_cv",
        "complexity": formula_complexity(selected_terms, coefficients),
        "accepted": accepted,
        "rejection_reason": "accepted" if accepted else "physical_constraint_violation",
        "validation_candidates_evaluated": 0,
        **report.to_dict(),
    }
    predictions = case.frame.copy()
    predictions["method"] = mode
    predictions["correction_MPa"] = correction
    predictions["corrected_strength_MPa"] = y_pred
    predictions["residual_after_correction_MPa"] = y_true - y_pred
    candidate_row = {
        key: value for key, value in best.items() if key not in {"terms", "coefficients"}
    }
    return SearchResult(
        mode=mode,
        best=best,
        candidates=pd.DataFrame([candidate_row]),
        predictions=predictions,
    )


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x.T @ x
    penalty = np.eye(xtx.shape[0]) * alpha
    if penalty.shape[0] > 0:
        penalty[0, 0] = 0.0
    rhs = x.T @ y
    try:
        return np.linalg.solve(xtx + penalty, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx + penalty) @ rhs


def _cv_rmse(case: Any, features: pd.DataFrame, terms: list[str], alpha: float, split_col: str) -> float:
    if split_col not in case.frame.columns:
        return float("nan")
    residual = case.frame[case.residual].to_numpy(float)
    y_true = case.frame[case.y_true].to_numpy(float)
    y_base = case.frame[case.y_base].to_numpy(float)
    errors = []
    for train_idx, test_idx, _ in leave_one_group_indices(case.frame, split_col):
        x_train = design_matrix(features.iloc[train_idx], terms)
        coef = _fit_ridge(x_train, residual[train_idx], alpha)
        pred_corr = design_matrix(features.iloc[test_idx], terms) @ coef
        y_pred = y_base[test_idx] + pred_corr
        errors.extend((y_true[test_idx] - y_pred).tolist())
    if not errors:
        return float("nan")
    err = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(err**2)))


def _validation_rmse(case: Any, features: pd.DataFrame, terms: list[str], alpha: float) -> tuple[float, float, float]:
    """Return beta, psi, and aggregate group-held-out errors for one formula."""
    beta_rmse = _cv_rmse(case, features, terms, alpha, "beta_deg")
    psi_rmse = _cv_rmse(case, features, terms, alpha, "psi_deg")
    available = [value for value in [beta_rmse, psi_rmse] if np.isfinite(value)]
    aggregate = float(np.mean(available)) if available else float("nan")
    return beta_rmse, psi_rmse, aggregate


def group_cv_fold_metrics(case: Any, terms: list[str], alpha: float) -> list[dict[str, Any]]:
    """Return aligned fold losses for paired structural-model comparisons."""
    features = make_angle_features(case.frame, include_forbidden=False)
    residual = case.frame[case.residual].to_numpy(float)
    y_true = case.frame[case.y_true].to_numpy(float)
    y_base = case.frame[case.y_base].to_numpy(float)
    rows: list[dict[str, Any]] = []
    for split_col in ["beta_deg", "psi_deg"]:
        if split_col not in case.frame.columns:
            continue
        for train_idx, test_idx, held_out in leave_one_group_indices(case.frame, split_col):
            x_train = design_matrix(features.iloc[train_idx], terms)
            coef = _fit_ridge(x_train, residual[train_idx], float(alpha))
            correction = design_matrix(features.iloc[test_idx], terms) @ coef
            errors = y_true[test_idx] - (y_base[test_idx] + correction)
            sse = float(np.sum(np.asarray(errors, dtype=float) ** 2))
            rows.append(
                {
                    "axis": split_col,
                    "held_out_group": str(held_out),
                    "n": int(len(test_idx)),
                    "sse": sse,
                    "rmse": float(np.sqrt(sse / len(test_idx))) if len(test_idx) else float("nan"),
                }
            )
    return rows


def aggregate_group_cv_folds(rows: list[dict[str, Any]]) -> float:
    """Aggregate fold SSE using the same two-axis definition as model selection."""
    axis_scores: list[float] = []
    for axis in sorted({str(row["axis"]) for row in rows}):
        selected = [row for row in rows if str(row["axis"]) == axis]
        n = sum(int(row["n"]) for row in selected)
        if n:
            axis_scores.append(float(np.sqrt(sum(float(row["sse"]) for row in selected) / n)))
    return float(np.mean(axis_scores)) if axis_scores else float("nan")


def run_candidate_search(
    case: Any,
    mode: str = "asrc",
    max_terms: int = 3,
    alpha_grid: list[float] | None = None,
    enforce_physical: bool = True,
    allowed_terms: list[str] | None = None,
    selection_metric: str = "training_rmse",
    validation_top_k: int = 32,
) -> SearchResult:
    """Search a bounded formula library.

    ``selection_metric='group_cv'`` refits only the most competitive training
    candidates on leave-one-angle folds before selecting the final formula.
    This keeps the research pipeline tractable while preventing an agent from
    accepting a formula solely because of its in-sample residual error.
    """
    if selection_metric not in {"training_rmse", "group_cv"}:
        raise ValueError("selection_metric must be 'training_rmse' or 'group_cv'.")
    alpha_grid = alpha_grid or [0.0, 0.001, 0.01, 0.1, 1.0]
    include_forbidden = mode == "generic"
    features = make_angle_features(case.frame, include_forbidden=include_forbidden)
    has_psi = "psi_deg" in case.frame.columns
    terms = candidate_terms(list(features.columns), mode, has_psi)
    if allowed_terms:
        allowed = set(allowed_terms)
        terms = [term for term in features.columns if term not in {"beta_deg", "psi_deg"} and term in allowed]
    term_sets = enumerate_term_sets(terms, max_terms=max_terms)
    residual = case.frame[case.residual].to_numpy(float)
    y_true = case.frame[case.y_true].to_numpy(float)
    y_base = case.frame[case.y_base].to_numpy(float)
    constraint_evaluator = prepare_constraint_evaluator(case) if enforce_physical else None

    rows = []
    payloads: list[dict[str, Any]] = []
    for term_set in term_sets:
        x = design_matrix(features, term_set)
        for alpha in alpha_grid:
            coef = _fit_ridge(x, residual, float(alpha))
            correction = x @ coef
            y_pred = y_base + correction
            metric = regression_metrics(y_true, y_pred)
            coefficients = {term: float(value) for term, value in zip(term_set, coef)}
            if enforce_physical:
                report = constraint_evaluator.evaluate(term_set, coefficients)
            else:
                report = ConstraintReport(
                    positive_strength_violations=0,
                    endpoint_singularity=False,
                    finite_prediction_violations=0,
                    max_abs_correction=float(np.nanmax(np.abs(correction))) if len(correction) else 0.0,
                )
            complexity = formula_complexity(term_set, coefficients)
            accepted = (not enforce_physical) or report.total_violations == 0
            reason = "accepted" if accepted else "physical_constraint_violation"
            candidate_id = f"{mode}_{len(rows) + 1:04d}"
            payload = {
                "candidate_id": candidate_id,
                "candidate_enumeration_order": len(rows) + 1,
                "validation_evaluation_order": float("nan"),
                "mode": mode,
                "terms": term_set,
                "alpha": float(alpha),
                "formula": formula_text(term_set, coefficients),
                "coefficients": coefficients,
                "rmse": metric.rmse,
                "mae": metric.mae,
                "mape": metric.mape,
                "r2": metric.r2,
                "max_abs_error": metric.max_abs_error,
                "leave_one_beta_rmse": float("nan"),
                "leave_one_psi_rmse": float("nan"),
                "selection_validation_rmse": float("nan"),
                "selection_metric": selection_metric,
                "complexity": complexity,
                "accepted": accepted,
                "rejection_reason": reason,
                **report.to_dict(),
            }
            rows.append({k: v for k, v in payload.items() if k not in {"terms", "coefficients"}})

            payloads.append(payload)

    if not payloads:
        raise RuntimeError(f"No candidates were generated for mode={mode}.")

    # Do not perform CV for every combinatorial candidate.  The pool is chosen
    # without looking at held-out groups, then the held-out error selects among it.
    validation_candidates_evaluated = 0
    eligible_payloads = payloads
    if selection_metric == "group_cv":
        pool = sorted(
            payloads,
            key=lambda item: (
                int(not item["accepted"]),
                int(item["physical_violations"]) if enforce_physical else 0,
                float(item["rmse"]),
                int(item["complexity"]),
                float(item["mae"]),
            ),
        )[: max(1, int(validation_top_k))]
        eligible_payloads = pool
        validation_candidates_evaluated = len(pool)
        for validation_order, payload in enumerate(pool, start=1):
            payload["validation_evaluation_order"] = validation_order
            beta_rmse, psi_rmse, aggregate = _validation_rmse(
                case,
                features,
                payload["terms"],
                float(payload["alpha"]),
            )
            payload["leave_one_beta_rmse"] = beta_rmse
            payload["leave_one_psi_rmse"] = psi_rmse
            payload["selection_validation_rmse"] = aggregate

    def ranking(payload: dict[str, Any]) -> tuple[int, int, float, int, float, float]:
        validation_error = float(payload["selection_validation_rmse"])
        score = validation_error if selection_metric == "group_cv" and np.isfinite(validation_error) else float(payload["rmse"])
        return (
            int(not payload["accepted"]),
            int(payload["physical_violations"]) if enforce_physical else 0,
            score,
            int(payload["complexity"]),
            float(payload["rmse"]),
            float(payload["mae"]),
        )

    # Candidates without group-CV evidence must not compete using their lower
    # in-sample RMSE against validated candidates. Doing so makes the selected
    # model depend paradoxically on validation_top_k and can reward overfit.
    best_payload = min(eligible_payloads, key=ranking)
    best_payload = {**best_payload, "_ranking": ranking(best_payload)}

    final_report = (
        constraint_evaluator.evaluate(
            best_payload["terms"],
            best_payload["coefficients"],
        )
        if constraint_evaluator is not None
        else check_physical_constraints(
            case,
            best_payload["terms"],
            best_payload["coefficients"],
        )
    )
    best_payload.update(final_report.to_dict())
    beta_rmse, psi_rmse, aggregate = _validation_rmse(case, features, best_payload["terms"], float(best_payload["alpha"]))
    best_payload["leave_one_beta_rmse"] = beta_rmse
    best_payload["leave_one_psi_rmse"] = psi_rmse
    best_payload["selection_validation_rmse"] = aggregate
    best_payload["validation_candidates_evaluated"] = validation_candidates_evaluated
    best_payload["candidate_terms_available"] = len(terms)
    best_payload["candidate_term_sets_enumerated"] = len(term_sets)
    best_payload["training_candidates_evaluated"] = len(payloads)
    best_payload["alpha_values_evaluated"] = len(alpha_grid)

    # Persist selection diagnostics for all candidates inspected by CV.
    row_by_id = {row["candidate_id"]: row for row in rows}
    for payload in payloads:
        row = row_by_id[payload["candidate_id"]]
        row["leave_one_beta_rmse"] = payload["leave_one_beta_rmse"]
        row["leave_one_psi_rmse"] = payload["leave_one_psi_rmse"]
        row["selection_validation_rmse"] = payload["selection_validation_rmse"]
        row["validation_evaluation_order"] = payload["validation_evaluation_order"]
        row["selection_metric"] = selection_metric

    best_terms = best_payload["terms"]
    best_coef = np.asarray([best_payload["coefficients"][term] for term in best_terms], dtype=float)
    best_correction = design_matrix(features, best_terms) @ best_coef
    predictions = case.frame.copy()
    predictions["method"] = mode
    predictions["correction_MPa"] = best_correction
    predictions["corrected_strength_MPa"] = y_base + best_correction
    predictions["residual_after_correction_MPa"] = y_true - predictions["corrected_strength_MPa"].to_numpy(float)

    best_payload = {k: v for k, v in best_payload.items() if k != "_ranking"}
    return SearchResult(mode=mode, best=best_payload, candidates=pd.DataFrame(rows), predictions=predictions)

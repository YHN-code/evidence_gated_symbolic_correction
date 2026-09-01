from __future__ import annotations

from typing import Any

import numpy as np

from asrc.symbolic.sparse_regression import (
    SearchResult,
    aggregate_group_cv_folds,
    group_cv_fold_metrics,
    run_candidate_search,
)


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(str(item) for item in items if str(item)))


def _bootstrap_difference(
    preferred: list[dict[str, Any]],
    alternative: list[dict[str, Any]],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    pref_by_axis = {
        axis: [row for row in preferred if str(row["axis"]) == axis]
        for axis in sorted({str(row["axis"]) for row in preferred})
    }
    alt_lookup = {
        (str(row["axis"]), str(row["held_out_group"])): row for row in alternative
    }
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(max(1, int(samples))):
        preferred_axes: list[float] = []
        alternative_axes: list[float] = []
        for axis, rows in pref_by_axis.items():
            indices = rng.integers(0, len(rows), size=len(rows))
            pref_sse = pref_n = alt_sse = alt_n = 0.0
            for index in indices:
                pref_row = rows[int(index)]
                alt_row = alt_lookup[(axis, str(pref_row["held_out_group"]))]
                pref_sse += float(pref_row["sse"])
                pref_n += int(pref_row["n"])
                alt_sse += float(alt_row["sse"])
                alt_n += int(alt_row["n"])
            preferred_axes.append(float(np.sqrt(pref_sse / pref_n)))
            alternative_axes.append(float(np.sqrt(alt_sse / alt_n)))
        differences.append(float(np.mean(alternative_axes) - np.mean(preferred_axes)))
    tail = (1.0 - float(confidence)) / 2.0
    return float(np.quantile(differences, tail)), float(np.quantile(differences, 1.0 - tail))


def compare_family_hypotheses(
    *,
    case: Any,
    current: SearchResult,
    grammar_index: list[dict[str, Any]],
    reference_terms: list[str],
    mode: str,
    max_terms: int,
    alpha_grid: list[float],
    enforce_physical: bool,
    validation_top_k: int = 8,
    bootstrap_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260713,
) -> dict[str, Any]:
    """Compare the current formula with registered family hypotheses on paired CV folds.

    The comparison never reads locked-test targets or synthetic hidden formulas. Each
    family receives the same candidate-validation budget and the same group splits.
    """
    families = [family for family in grammar_index if family.get("id") and family.get("terms")]
    if not families:
        return {
            "audit_type": "registered_family_hypothesis_comparison",
            "audit_version": "1.0",
            "structural_status": "not_applicable",
            "summary": "No registered candidate family is available for this case.",
            "coverage_complete": True,
            "hypotheses": [],
            "validation_candidates_evaluated": 0,
        }

    hypotheses: list[dict[str, Any]] = []
    fold_metrics: dict[str, list[dict[str, Any]]] = {}
    current_id = "current_formula"
    current_folds = group_cv_fold_metrics(case, list(current.best["terms"]), float(current.best["alpha"]))
    fold_metrics[current_id] = current_folds
    hypotheses.append(
        {
            "hypothesis_id": current_id,
            "family_id": "current",
            "formula": current.best["formula"],
            "terms": list(current.best["terms"]),
            "structural_signature": sorted(term for term in current.best["terms"] if term != "1"),
            "alpha": float(current.best["alpha"]),
            "complexity": int(current.best["complexity"]),
            "physical_violations": int(current.best.get("physical_violations", 0)),
            "group_cv_rmse": aggregate_group_cv_folds(current_folds),
            "validation_candidates_evaluated": 0,
        }
    )

    failures: list[dict[str, str]] = []
    validation_consumed = 0
    base_terms = _unique(reference_terms)
    for family in families:
        family_id = str(family["id"])
        hypothesis_id = f"family:{family_id}"
        try:
            result = run_candidate_search(
                case,
                mode=mode,
                max_terms=max(1, int(max_terms)),
                alpha_grid=alpha_grid,
                enforce_physical=enforce_physical,
                allowed_terms=_unique([*base_terms, *[str(term) for term in family["terms"]]]),
                selection_metric="group_cv",
                validation_top_k=max(1, int(validation_top_k)),
            )
            consumed = int(result.best.get("validation_candidates_evaluated", 0))
            validation_consumed += consumed
            folds = group_cv_fold_metrics(case, list(result.best["terms"]), float(result.best["alpha"]))
            fold_metrics[hypothesis_id] = folds
            hypotheses.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "family_id": family_id,
                    "formula": result.best["formula"],
                    "terms": list(result.best["terms"]),
                    "structural_signature": sorted(term for term in result.best["terms"] if term != "1"),
                    "alpha": float(result.best["alpha"]),
                    "complexity": int(result.best["complexity"]),
                    "physical_violations": int(result.best.get("physical_violations", 0)),
                    "group_cv_rmse": aggregate_group_cv_folds(folds),
                    "validation_candidates_evaluated": consumed,
                }
            )
        except Exception as exc:
            failures.append({"family_id": family_id, "error_type": type(exc).__name__, "error": str(exc)})

    coverage_complete = len(failures) == 0 and len(hypotheses) == len(families) + 1
    eligible = [row for row in hypotheses if int(row["physical_violations"]) == 0 and np.isfinite(row["group_cv_rmse"])]
    if not eligible:
        status = "unresolved"
        preferred_id = ""
        equivalent_ids: list[str] = []
    else:
        preferred = min(eligible, key=lambda row: (float(row["group_cv_rmse"]), int(row["complexity"])))
        preferred_id = str(preferred["hypothesis_id"])
        equivalent_ids = [preferred_id]
        for index, row in enumerate(hypotheses):
            hypothesis_id = str(row["hypothesis_id"])
            if hypothesis_id == preferred_id:
                row.update({"relationship_to_preferred": "preferred", "difference_ci_low": 0.0, "difference_ci_high": 0.0})
                continue
            if hypothesis_id not in fold_metrics or preferred_id not in fold_metrics:
                row["relationship_to_preferred"] = "unresolved"
                continue
            low, high = _bootstrap_difference(
                fold_metrics[preferred_id],
                fold_metrics[hypothesis_id],
                samples=bootstrap_samples,
                confidence=confidence,
                seed=int(seed) + index * 1009,
            )
            numerical_tolerance = np.finfo(float).eps * max(
                1.0,
                abs(float(preferred["group_cv_rmse"])),
                abs(float(row["group_cv_rmse"])),
            ) * 1000.0
            relationship = "inferior" if low > numerical_tolerance else "equivalent_under_validation_uncertainty"
            row.update({"relationship_to_preferred": relationship, "difference_ci_low": low, "difference_ci_high": high})
            if relationship.startswith("equivalent"):
                equivalent_ids.append(hypothesis_id)
        current_relation = next(row for row in hypotheses if row["hypothesis_id"] == current_id).get("relationship_to_preferred")
        if not coverage_complete:
            status = "unresolved"
        elif current_relation == "inferior" or current_relation == "unresolved":
            status = "unresolved"
        elif len(equivalent_ids) > 1:
            equivalent_signatures = {
                tuple(row.get("structural_signature", []))
                for row in hypotheses
                if row["hypothesis_id"] in equivalent_ids
            }
            status = "resolved" if len(equivalent_signatures) == 1 else "equivalent"
        else:
            status = "resolved"

    preferred_family = next(
        (str(row["family_id"]) for row in hypotheses if row["hypothesis_id"] == preferred_id), ""
    )
    summary_by_status = {
        "resolved": "One structural term set is uniquely preferred among the registered family hypotheses under paired group validation.",
        "equivalent": "Multiple registered hypotheses are indistinguishable under paired group-validation uncertainty.",
        "unresolved": "The current formula is not uniquely supported or the registered-family comparison is incomplete.",
        "not_applicable": "No registered family comparison applies.",
    }
    return {
        "audit_type": "registered_family_hypothesis_comparison",
        "audit_version": "1.0",
        "structural_status": status,
        "summary": summary_by_status[status],
        "coverage_complete": coverage_complete,
        "confidence": float(confidence),
        "bootstrap_samples": int(bootstrap_samples),
        "preferred_hypothesis_id": preferred_id,
        "preferred_family_id": preferred_family,
        "equivalent_hypothesis_ids": equivalent_ids,
        "recommended_family_ids": [preferred_family] if status == "unresolved" and preferred_family not in {"", "current"} else [],
        "hypotheses": hypotheses,
        "failed_families": failures,
        "validation_candidates_evaluated": validation_consumed,
        "validation_scope": "training_group_cv_only",
        "locked_test_used": False,
        "hidden_formula_used": False,
    }

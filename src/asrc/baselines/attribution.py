from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from asrc.agent.evidence_tools import screen_candidate_families
from asrc.features.angle_features import make_angle_features
from asrc.features.bounded_grammar import bounded_grammar_index, bounded_grammar_terms
from asrc.symbolic.sparse_regression import SearchResult, run_candidate_search


def accessible_feature_terms(
    case: Any,
    *,
    extra_terms: list[str] | None = None,
    include_grammar: bool = False,
) -> list[str]:
    """Return the feature space accessible to an ASRC search policy."""
    features = make_angle_features(case.frame, include_forbidden=False)
    configured = case.config.get("features", {}).get("include")
    if configured:
        terms = [str(name) for name in configured if str(name) in features.columns]
    else:
        terms = [
            str(name)
            for name in features.columns
            if str(name) not in {"beta_deg", "psi_deg"}
        ]
    for term in extra_terms or []:
        if term in features.columns and term not in terms:
            terms.append(str(term))
    if include_grammar:
        for term in bounded_grammar_terms(case):
            if term in features.columns and term not in terms:
                terms.append(term)
    return terms


def screen_feature_terms(
    case: Any,
    terms: list[str],
    *,
    max_variables: int = 12,
) -> tuple[list[str], list[dict[str, float | str]]]:
    """Rank a feature superset without using grouped-validation outcomes."""
    if max_variables < 1:
        raise ValueError("max_variables must be positive.")
    features = make_angle_features(case.frame, include_forbidden=False)
    residual = case.frame[case.residual].to_numpy(float)
    residual_std = float(np.std(residual))
    rows: list[dict[str, float | str | int]] = []
    for order, term in enumerate(dict.fromkeys(terms)):
        if term not in features.columns:
            continue
        values = features[term].to_numpy(float)
        if residual_std <= 0.0 or float(np.std(values)) <= 0.0:
            score = 0.0
        else:
            correlation = float(np.corrcoef(values, residual)[0, 1])
            score = abs(correlation) if np.isfinite(correlation) else 0.0
        rows.append({"term": term, "score": score, "source_order": order})
    rows.sort(key=lambda row: (-float(row["score"]), int(row["source_order"])))
    selected = [str(row["term"]) for row in rows[: int(max_variables)]]
    diagnostics = [
        {"term": str(row["term"]), "absolute_residual_correlation": float(row["score"])}
        for row in rows
    ]
    return selected, diagnostics


def run_deterministic_expanded_search(
    case: Any,
    *,
    knowledge_terms: list[str] | None,
    max_terms: int,
    max_variables: int,
    alpha_grid: list[float],
    validation_budget: int,
) -> SearchResult:
    """Run a one-shot deterministic search with access to the Full-ASRC term superset."""
    superset = accessible_feature_terms(
        case,
        extra_terms=knowledge_terms,
        include_grammar=True,
    )
    selected, screening = screen_feature_terms(
        case,
        superset,
        max_variables=max_variables,
    )
    result = run_candidate_search(
        case,
        mode="asrc",
        max_terms=max_terms,
        alpha_grid=alpha_grid,
        enforce_physical=True,
        allowed_terms=selected,
        selection_metric="group_cv",
        validation_top_k=validation_budget,
    )
    best = {
        **result.best,
        "accessible_term_superset_size": len(superset),
        "screened_variable_count": len(selected),
        "screened_variables": selected,
        "screening_method": "absolute_training_residual_correlation",
        "screening_ranking": screening,
    }
    return SearchResult(
        mode="deterministic_asrc_expanded",
        best=best,
        candidates=result.candidates,
        predictions=result.predictions,
    )


@dataclass(frozen=True)
class ScriptedPolicyResult:
    result: SearchResult
    trace: list[dict[str, Any]]
    candidates: pd.DataFrame


def _selection_rank(result: SearchResult) -> tuple[int, float, int, float]:
    best = result.best
    score = float(best.get("selection_validation_rmse", float("inf")))
    if not np.isfinite(score):
        score = float(best["rmse"])
    return (
        int(best.get("physical_violations", 0)),
        score,
        int(best["complexity"]),
        float(best["rmse"]),
    )


def run_scripted_policy_search(
    case: Any,
    *,
    knowledge_terms: list[str] | None,
    base_max_terms: int,
    max_terms_cap: int,
    max_variables: int,
    alpha_grid: list[float],
    validation_top_k: int,
    validation_budget_total: int,
) -> ScriptedPolicyResult:
    """Run a deterministic evidence-driven policy over the Full-ASRC tools."""
    if validation_budget_total < 1:
        raise ValueError("validation_budget_total must be positive.")
    if validation_top_k < 1:
        raise ValueError("validation_top_k must be positive.")

    initial_terms = accessible_feature_terms(
        case,
        extra_terms=knowledge_terms,
        include_grammar=False,
    )
    knowledge_set = set(knowledge_terms or [])
    initial_search_terms = [
        term for term in initial_terms if term in knowledge_set
    ] or initial_terms
    allocation = min(int(validation_top_k), int(validation_budget_total))
    incumbent = run_candidate_search(
        case,
        mode="asrc",
        max_terms=base_max_terms,
        alpha_grid=alpha_grid,
        enforce_physical=True,
        allowed_terms=initial_search_terms,
        selection_metric="group_cv",
        validation_top_k=allocation,
    )
    budget_used = int(incumbent.best["validation_candidates_evaluated"])
    training_candidates = int(incumbent.best["training_candidates_evaluated"])
    search_calls = 1
    trace: list[dict[str, Any]] = [
        {
            "round": 0,
            "action": "build_validation_incumbent",
            "family_id": "initial_knowledge_terms",
            "validation_budget_used": budget_used,
            "selection_validation_rmse": incumbent.best[
                "selection_validation_rmse"
            ],
            "accepted_as_best": True,
        }
    ]
    candidate_frames = [incumbent.candidates.assign(scripted_round=0)]

    grammar = bounded_grammar_index(case)
    base_remainder = [
        term for term in initial_terms if term not in set(initial_search_terms)
    ]
    if base_remainder:
        grammar = [
            {
                "id": "configured_base_remainder",
                "purpose": "Screen configured bounded descriptors omitted from the initial knowledge subset.",
                "terms": base_remainder,
            },
            *grammar,
        ]
    tested: set[str] = set()
    round_number = 0
    while budget_used < int(validation_budget_total):
        screen = screen_candidate_families(
            case,
            incumbent,
            grammar,
            min_relative_improvement=0.0,
        )
        ordered = [
            str(row["family_id"])
            for row in screen.get("ranked_families", [])
            if str(row["family_id"]) not in tested
        ]
        if not ordered:
            break
        family_id = ordered[0]
        tested.add(family_id)
        family = next(row for row in grammar if str(row["id"]) == family_id)
        current_terms = [
            str(term) for term in incumbent.best.get("terms", []) if term != "1"
        ]
        candidate_pool = list(
            dict.fromkeys([*current_terms, *[str(term) for term in family["terms"]]])
        )
        selected, _ = screen_feature_terms(
            case,
            candidate_pool,
            max_variables=max_variables,
        )
        remaining = int(validation_budget_total) - budget_used
        allocation = min(int(validation_top_k), remaining)
        round_number += 1
        proposal = run_candidate_search(
            case,
            mode="asrc",
            max_terms=min(max_terms_cap, max(base_max_terms, len(current_terms) + 1)),
            alpha_grid=alpha_grid,
            enforce_physical=True,
            allowed_terms=selected,
            selection_metric="group_cv",
            validation_top_k=allocation,
        )
        consumed = int(proposal.best["validation_candidates_evaluated"])
        budget_used += consumed
        training_candidates += int(proposal.best["training_candidates_evaluated"])
        search_calls += 1
        improved = _selection_rank(proposal) < _selection_rank(incumbent)
        if improved:
            incumbent = proposal
        trace.append(
            {
                "round": round_number,
                "action": "compare_candidate_family",
                "family_id": family_id,
                "screening_relative_rmse_reduction": next(
                    (
                        float(row["relative_rmse_reduction"])
                        for row in screen.get("ranked_families", [])
                        if str(row["family_id"]) == family_id
                    ),
                    float("nan"),
                ),
                "validation_budget_consumed": consumed,
                "validation_budget_used": budget_used,
                "selection_validation_rmse": proposal.best[
                    "selection_validation_rmse"
                ],
                "accepted_as_best": improved,
            }
        )
        candidate_frames.append(
            proposal.candidates.assign(
                scripted_round=round_number,
                scripted_family_id=family_id,
            )
        )

    best = {
        **incumbent.best,
        "validation_candidates_evaluated": budget_used,
        "training_candidates_evaluated": training_candidates,
        "search_calls": search_calls,
        "scripted_families_tested": list(tested),
        "accessible_term_superset_size": len(
            accessible_feature_terms(
                case,
                extra_terms=knowledge_terms,
                include_grammar=True,
            )
        ),
        "screening_method": "deterministic_residual_family_screening",
    }
    result = SearchResult(
        mode="scripted_policy_asrc",
        best=best,
        candidates=incumbent.candidates,
        predictions=incumbent.predictions,
    )
    return ScriptedPolicyResult(
        result=result,
        trace=trace,
        candidates=pd.concat(candidate_frames, ignore_index=True),
    )

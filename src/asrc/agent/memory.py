from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from asrc.features.angle_features import make_angle_features


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _residual_group_stats(df: pd.DataFrame, residual_col: str, group_col: str) -> list[dict[str, Any]]:
    if group_col not in df.columns:
        return []
    rows = []
    grouped = df.groupby(group_col, dropna=False)[residual_col]
    for value, series in grouped:
        arr = series.astype(float).to_numpy()
        rows.append(
            {
                group_col: float(value),
                "count": int(len(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=0)),
                "max_abs": float(np.max(np.abs(arr))),
            }
        )
    return sorted(rows, key=lambda row: row[group_col])


def residual_diagnostics(
    case: Any,
    allowed_variables: list[str],
    top_n: int = 8,
    residual_values: np.ndarray | list[float] | None = None,
    residual_source: str = "baseline_residual",
) -> dict[str, Any]:
    df = case.frame.copy()
    residual_col = case.residual
    if residual_values is not None:
        residual = np.asarray(residual_values, dtype=float)
        if residual.shape != (len(df),):
            raise ValueError("residual_values must contain exactly one value per case row.")
        residual_col = "__diagnostic_residual__"
        df[residual_col] = residual
    else:
        residual = df[residual_col].astype(float).to_numpy()
    diagnostics: dict[str, Any] = {
        "residual_source": residual_source,
        "residual_overall": {
            "mean": float(np.mean(residual)),
            "std": float(np.std(residual, ddof=0)),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "max_abs": float(np.max(np.abs(residual))),
        },
        "by_beta": _residual_group_stats(df, residual_col, "beta_deg"),
        "by_psi": _residual_group_stats(df, residual_col, "psi_deg"),
        "feature_correlations": [],
        "worst_angle_cells": [],
        "dominant_axis": "undetermined",
    }

    residual_std = float(np.std(residual, ddof=0))
    features = make_angle_features(df, include_forbidden=False)
    correlations = []
    for variable in allowed_variables:
        if variable not in features.columns:
            continue
        values = features[variable].astype(float).to_numpy()
        value_std = float(np.std(values, ddof=0))
        if residual_std <= 1e-12 or value_std <= 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(values, residual)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        correlations.append({"variable": variable, "correlation": corr, "abs_correlation": abs(corr)})
    diagnostics["feature_correlations"] = sorted(correlations, key=lambda row: row["abs_correlation"], reverse=True)[:top_n]

    angle_cols = [col for col in ["beta_deg", "psi_deg"] if col in df.columns]
    if angle_cols:
        worst = df.copy()
        worst["residual"] = worst[residual_col].astype(float)
        worst["abs_residual"] = worst["residual"].abs()
        cols = [*angle_cols, "residual", "abs_residual"]
        diagnostics["worst_angle_cells"] = worst.sort_values("abs_residual", ascending=False)[cols].head(top_n).to_dict(orient="records")

    beta_span = max((row["mean"] for row in diagnostics["by_beta"]), default=0.0) - min((row["mean"] for row in diagnostics["by_beta"]), default=0.0)
    psi_span = max((row["mean"] for row in diagnostics["by_psi"]), default=0.0) - min((row["mean"] for row in diagnostics["by_psi"]), default=0.0)
    if abs(beta_span) > 1e-12 or abs(psi_span) > 1e-12:
        if abs(beta_span) > abs(psi_span) * 1.25:
            diagnostics["dominant_axis"] = "beta"
        elif abs(psi_span) > abs(beta_span) * 1.25:
            diagnostics["dominant_axis"] = "psi"
        else:
            diagnostics["dominant_axis"] = "coupled_or_mixed"
    return diagnostics


def create_agent_memory(case: Any, allowed_variables: list[str], metadata: dict[str, Any], base_max_terms: int) -> dict[str, Any]:
    diagnostics = residual_diagnostics(case, allowed_variables)
    return {
        "memory_type": "structured_agent_memory_v2",
        "case": metadata.get("case", case.case_name),
        "rock": metadata.get("rock", metadata.get("dataset", "")),
        "allowed_variables": allowed_variables,
        "base_max_terms": int(base_max_terms),
        "residual_diagnostics": diagnostics,
        "case_memory": {
            "observation_memory": [
                {
                    "round": 0,
                    "observation": "initial_residual_diagnostics",
                    "dominant_axis": diagnostics.get("dominant_axis", "undetermined"),
                    "residual_overall": diagnostics.get("residual_overall", {}),
                    "worst_angle_cells": diagnostics.get("worst_angle_cells", [])[:5],
                }
            ],
            "hypothesis_memory": [],
            "action_memory": [],
            "outcome_memory": [],
            "acceptance_memory": [],
            "commitment_memory": [],
        },
        "cross_case_memory": {
            "supported_terms": [],
            "unstable_terms": [],
            "notes": [],
        },
        "rounds": [],
        "tried_variables": [],
        "failed_variables": [],
        "best_formula": None,
        "best_rmse": None,
        "best_complexity": None,
        "best_physical_violations": None,
    }


def compact_memory_view(memory: dict[str, Any], max_rounds: int = 2, max_candidates: int = 3) -> dict[str, Any]:
    diagnostics = memory.get("residual_diagnostics", {})
    case_memory = memory.get("case_memory", {})
    observations = case_memory.get("observation_memory", [])
    outcomes = case_memory.get("outcome_memory", [])

    def compact_observation(row: dict[str, Any]) -> dict[str, Any]:
        compact = dict(row)
        post = compact.get("post_correction_diagnostics") or {}
        if post:
            compact["post_correction_diagnostics"] = {
                "residual_source": post.get("residual_source"),
                "residual_overall": post.get("residual_overall", {}),
                "dominant_axis": post.get("dominant_axis", "undetermined"),
                "feature_correlations": post.get("feature_correlations", [])[:max_candidates],
                "worst_angle_cells": post.get("worst_angle_cells", [])[:max_candidates],
            }
        return compact

    current_formula = str(memory.get("best_formula") or "")
    current_best_inspections = [
        row
        for row in observations
        if row.get("observation") == "post_correction_residual_inspected"
        and str(row.get("inspected_formula", "")) == current_formula
    ]
    current_best_evidence = {
        str(row.get("evidence_type")): {
            "round": row.get("round"),
            "observation": row.get("observation"),
            "passed": row.get("evidence_passed"),
            "evidence": row.get("evidence", {}),
        }
        for row in observations
        if row.get("evidence_type") and str(row.get("inspected_formula", "")) == current_formula
    }
    recent_rounds = []
    for row in memory.get("rounds", [])[-max_rounds:]:
        recent_rounds.append(
            {
                "round": row.get("round"),
                "action_effective": row.get("action_effective"),
                "variables": row.get("variables", []),
                "search_max_terms": row.get("search_max_terms"),
                "formula": row.get("formula"),
                "rmse": row.get("rmse"),
                "best_rmse_so_far": row.get("best_rmse_so_far"),
                "relative_improvement": row.get("relative_improvement"),
                "complexity": row.get("complexity"),
                "physical_violations": row.get("physical_violations"),
            }
        )
    return {
        "memory_type": memory.get("memory_type"),
        "case": memory.get("case"),
        "rock": memory.get("rock"),
        "best_formula": memory.get("best_formula"),
        "best_rmse": memory.get("best_rmse"),
        "best_complexity": memory.get("best_complexity"),
        "best_physical_violations": memory.get("best_physical_violations"),
        "tried_variables": memory.get("tried_variables", []),
        "failed_variables": memory.get("failed_variables", []),
        "case_memory": {
            "observations": [compact_observation(row) for row in observations[-max_rounds:]],
            "active_hypotheses": case_memory.get("hypothesis_memory", [])[-max_rounds:],
            "recent_actions": case_memory.get("action_memory", [])[-max_rounds:],
            "recent_outcomes": outcomes[-max_rounds:],
            "acceptance": case_memory.get("acceptance_memory", [])[-max_rounds:],
            "pending_commitments": [
                item for item in case_memory.get("commitment_memory", []) if item.get("status") == "pending"
            ],
            "evidence_summary": {
                "agent_searches_performed": sum(bool(row.get("performed_search")) for row in outcomes),
                "current_best_post_residual_inspected": bool(current_best_inspections),
                "latest_current_best_inspection": (
                    compact_observation(current_best_inspections[-1]) if current_best_inspections else None
                ),
                "current_best_evidence": current_best_evidence,
            },
        },
        "cross_case_memory": memory.get("cross_case_memory", {}),
        "recent_rounds": recent_rounds,
        "residual_diagnostics": {
            "residual_overall": diagnostics.get("residual_overall", {}),
            "dominant_axis": diagnostics.get("dominant_axis", "undetermined"),
            "feature_correlations": diagnostics.get("feature_correlations", [])[:max_candidates],
            "worst_angle_cells": diagnostics.get("worst_angle_cells", [])[:max_candidates],
            "by_beta": diagnostics.get("by_beta", [])[:max_candidates],
            "by_psi": diagnostics.get("by_psi", [])[:max_candidates],
        },
    }


def update_agent_memory(
    memory: dict[str, Any],
    round_row: dict[str, Any],
    decision: dict[str, Any],
    variables: list[str],
    candidate_feedback: list[dict[str, Any]],
    search_max_terms: int,
    best_payload: dict[str, Any],
    direct_accept: bool = False,
    tool_output: dict[str, Any] | None = None,
    acceptance_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tool_output = tool_output or {}
    acceptance_report = acceptance_report or {}
    tried = list(dict.fromkeys([*memory.get("tried_variables", []), *variables]))
    memory["tried_variables"] = tried
    if (
        round_row.get("relative_improvement") is not None
        and _safe_float(round_row.get("relative_improvement")) < 0.0
        and str(round_row.get("action_effective", "")) not in {"remove_unstable_terms", "tighten_constraints"}
    ):
        memory["failed_variables"] = list(dict.fromkeys([*memory.get("failed_variables", []), *variables]))

    memory["best_formula"] = best_payload.get("formula")
    memory["best_rmse"] = _safe_float(best_payload.get("rmse"))
    memory["best_complexity"] = int(best_payload.get("complexity", 0))
    memory["best_physical_violations"] = int(best_payload.get("physical_violations", 0))
    case_memory = memory.setdefault(
        "case_memory",
        {
            "observation_memory": [],
            "hypothesis_memory": [],
            "action_memory": [],
            "outcome_memory": [],
            "acceptance_memory": [],
            "commitment_memory": [],
        },
    )
    commitments = case_memory.setdefault("commitment_memory", [])
    requested_action = str(decision.get("round_action", ""))
    effective_action = str(round_row.get("action_effective", requested_action))
    plan_status = str(decision.get("plan_status", "none"))
    plan_revision_reason = str(decision.get("plan_revision_reason", ""))
    tool_observation = str(tool_output.get("observation", ""))
    action_succeeded = (
        not effective_action.endswith("budget_exhausted")
        and "invalid" not in effective_action
        and tool_observation not in {"feature_family_not_resolved", "search_skipped_validation_budget_exhausted"}
    )
    for commitment in commitments:
        if commitment.get("status") != "pending":
            continue
        if requested_action == commitment.get("follow_up_action") and action_succeeded:
            commitment["status"] = "resolved"
            commitment["resolved_round"] = int(round_row["round"])
            commitment["resolution_action"] = effective_action
    if plan_status in {"revise", "complete"}:
        for commitment in commitments:
            if commitment.get("status") != "pending":
                continue
            commitment["status"] = "revised" if plan_status == "revise" else "completed_without_execution"
            commitment["closed_round"] = int(round_row["round"])
            commitment["closure_reason"] = plan_revision_reason
    if plan_status in {"new", "revise"}:
        immediately_resolved = decision.get("planned_action") == requested_action and action_succeeded
        commitments.append(
            {
                "id": f"round_{int(round_row['round']):03d}_follow_up",
                "created_round": int(round_row["round"]),
                "source_action": requested_action,
                "follow_up_action": str(decision.get("follow_up_action", "")),
                "reason": decision.get("next_action", ""),
                "expected_evidence": decision.get("expected_evidence", decision.get("expected_improvement", "")),
                "plan_status": plan_status,
                "status": "resolved" if immediately_resolved else "pending",
                **(
                    {"resolved_round": int(round_row["round"]), "resolution_action": effective_action}
                    if immediately_resolved
                    else {}
                ),
            }
        )
    case_memory.setdefault("observation_memory", []).append(
        {
            "round": int(round_row["round"]),
            "observation": tool_output.get("observation", "round_completed"),
            "inspected_formula": tool_output.get("inspected_formula", ""),
            "post_correction_diagnostics": tool_output.get("post_correction_diagnostics", {}),
            "evidence_type": tool_output.get("evidence_type", ""),
            "evidence_passed": tool_output.get("evidence_passed"),
            "evidence": tool_output.get("evidence", {}),
            "worst_error_regions": tool_output.get("worst_error_regions", []),
            "queried_knowledge_ids": tool_output.get("queried_knowledge_ids", []),
            "unresolved_issues": acceptance_report.get("unresolved_issues", []),
        }
    )
    relative_improvement = _safe_float(round_row.get("relative_improvement"))
    if bool(decision.get("follow_up_required")):
        hypothesis_status = "pending"
    elif bool(acceptance_report.get("accepted", False)):
        hypothesis_status = "accepted"
    elif bool(round_row.get("performed_search", False)):
        hypothesis_status = "supported" if relative_improvement > 0.0 else "rejected"
    else:
        hypothesis_status = "observed"
    case_memory.setdefault("hypothesis_memory", []).append(
        {
            "round": int(round_row["round"]),
            "hypothesis": decision.get("diagnosis", ""),
            "revision_reason": decision.get("revision_reason", ""),
            "expected_evidence": decision.get("expected_evidence", decision.get("expected_improvement", "")),
            "status": hypothesis_status,
        }
    )
    case_memory.setdefault("action_memory", []).append(
        {
            "round": int(round_row["round"]),
            "action_requested": decision.get("round_action", ""),
            "action_effective": round_row.get("action_effective", decision.get("round_action", "")),
            "executed_tool": tool_output.get("executed_tool", round_row.get("action_effective", "")),
            "query_reason": tool_output.get("query_reason", ""),
            "selected_feature_families": tool_output.get("selected_feature_families", []),
            "evidence_type": tool_output.get("evidence_type", ""),
            "evidence_passed": tool_output.get("evidence_passed"),
            "variables": variables,
            "search_max_terms": int(search_max_terms),
            "follow_up_required": bool(decision.get("follow_up_required", False)),
            "follow_up_action": decision.get("follow_up_action", ""),
            "plan_status": plan_status,
            "planned_action": decision.get("planned_action", ""),
            "plan_revision_reason": plan_revision_reason,
        }
    )
    case_memory.setdefault("outcome_memory", []).append(
        {
            "round": int(round_row["round"]),
            "formula": round_row.get("formula", ""),
            "rmse": _safe_float(round_row.get("rmse")),
            "best_rmse_so_far": _safe_float(round_row.get("best_rmse_so_far")),
            "relative_improvement": _safe_float(round_row.get("relative_improvement")),
            "complexity": int(round_row.get("complexity", 0)),
            "physical_violations": int(round_row.get("physical_violations", 0)),
            "performed_search": bool(round_row.get("performed_search", False)),
            "top_candidate_feedback": candidate_feedback,
        }
    )
    if acceptance_report:
        case_memory.setdefault("acceptance_memory", []).append(
            {
                "round": int(round_row["round"]),
                "requested": bool(acceptance_report.get("requested", False)),
                "accepted": bool(acceptance_report.get("accepted", False)),
                "blockers": acceptance_report.get("blockers", []),
                "warnings": acceptance_report.get("warnings", []),
                "unresolved_issues": acceptance_report.get("unresolved_issues", []),
            }
        )
    memory["rounds"].append(
        {
            "round": int(round_row["round"]),
            "action_requested": decision.get("round_action", ""),
            "action_effective": round_row.get("action_effective", decision.get("round_action", "")),
            "direct_accept": bool(direct_accept),
            "variables": variables,
            "search_max_terms": int(search_max_terms),
            "formula": round_row.get("formula", ""),
            "rmse": _safe_float(round_row.get("rmse")),
            "best_rmse_so_far": _safe_float(round_row.get("best_rmse_so_far")),
            "relative_improvement": _safe_float(round_row.get("relative_improvement")),
            "complexity": int(round_row.get("complexity", 0)),
            "physical_violations": int(round_row.get("physical_violations", 0)),
            "performed_search": bool(round_row.get("performed_search", False)),
            "follow_up_required": bool(decision.get("follow_up_required", False)),
            "follow_up_action": decision.get("follow_up_action", ""),
            "plan_status": plan_status,
            "planned_action": decision.get("planned_action", ""),
            "plan_revision_reason": plan_revision_reason,
            "revision_reason": decision.get("revision_reason", ""),
            "top_candidate_feedback": candidate_feedback,
        }
    )
    return memory

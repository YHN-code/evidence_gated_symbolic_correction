from __future__ import annotations

from typing import Any

import numpy as np

from asrc.agent.memory import _safe_float


def acceptance_contract(
    allow_accept_before_round: int = 2,
    requirements: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "allow_accept_before_round": int(allow_accept_before_round),
        "requirements": requirements
        or [
            "physical_violations_zero",
            "improves_over_baseline",
            "complexity_justified",
            "small_sample_risk_recorded",
            "no_unresolved_experiments",
        ],
    }


def verify_acceptance(
    *,
    best_payload: dict[str, Any] | None,
    baseline_rmse: float,
    round_number: int,
    allow_accept_before_round: int,
    requested: bool,
    memory: dict[str, Any] | None = None,
    requirements: list[str] | None = None,
    incumbent_payload: dict[str, Any] | None = None,
    noninferiority_tolerance: float = 0.0,
) -> dict[str, Any]:
    requirements = requirements or acceptance_contract(allow_accept_before_round)["requirements"]
    blockers: list[str] = []
    warnings: list[str] = []
    unresolved: list[str] = []
    best_payload = best_payload or {}
    rmse = _safe_float(best_payload.get("rmse"))
    complexity = int(best_payload.get("complexity", 0) or 0)
    violations = int(best_payload.get("physical_violations", 0) or 0)
    candidate_validation = _safe_float(best_payload.get("selection_validation_rmse"))
    incumbent_payload = incumbent_payload or {}
    incumbent_validation = _safe_float(incumbent_payload.get("selection_validation_rmse"))
    if not requested:
        return {
            "requested": False,
            "accepted": False,
            "blockers": [],
            "warnings": [],
            "unresolved_issues": [],
            "requirements": requirements,
            "baseline_rmse": float(baseline_rmse) if np.isfinite(baseline_rmse) else None,
            "candidate_rmse": rmse if np.isfinite(rmse) else None,
            "candidate_complexity": complexity,
            "candidate_physical_violations": violations,
            "candidate_validation_rmse": candidate_validation if np.isfinite(candidate_validation) else None,
            "incumbent_validation_rmse": incumbent_validation if np.isfinite(incumbent_validation) else None,
        }

    if not best_payload:
        blockers.append("no_current_best_formula")
    if round_number < int(allow_accept_before_round):
        blockers.append(f"acceptance_not_allowed_before_round_{allow_accept_before_round}")
    if "physical_violations_zero" in requirements and violations != 0:
        blockers.append("physical_violations_nonzero")
    if "improves_over_baseline" in requirements:
        if not np.isfinite(rmse) or not np.isfinite(baseline_rmse):
            blockers.append("rmse_not_finite")
        elif rmse >= baseline_rmse:
            blockers.append("does_not_improve_over_baseline")
    if "complexity_justified" in requirements and complexity > 6:
        warnings.append("formula_complexity_above_compact_default")
    if "small_sample_risk_recorded" in requirements:
        has_leave_one = any(
            str(key).startswith("leave_one") and np.isfinite(_safe_float(best_payload.get(key)))
            for key in best_payload
        )
        if not has_leave_one:
            warnings.append("leave_one_angle_metric_missing")
    if "validation_noninferior_to_incumbent" in requirements:
        if not np.isfinite(candidate_validation):
            blockers.append("candidate_validation_metric_missing")
        elif np.isfinite(incumbent_validation) and candidate_validation > incumbent_validation + float(noninferiority_tolerance):
            blockers.append("worse_than_validation_incumbent")
    if "validation_improves_over_baseline" in requirements:
        if not np.isfinite(candidate_validation) or not np.isfinite(baseline_rmse):
            blockers.append("baseline_validation_comparison_missing")
        elif candidate_validation >= baseline_rmse:
            blockers.append("does_not_improve_baseline_validation")

    case_memory = (memory or {}).get("case_memory", {})
    pending_commitments = [
        item for item in case_memory.get("commitment_memory", []) if item.get("status") == "pending"
    ]
    if "no_unresolved_experiments" in requirements and pending_commitments:
        blockers.append("unresolved_experiments")
        unresolved.extend(
            f"{item.get('id')}:{item.get('follow_up_action')}" for item in pending_commitments
        )
    if "agent_search_performed" in requirements:
        performed_search = any(bool(item.get("performed_search")) for item in case_memory.get("outcome_memory", []))
        if not performed_search:
            blockers.append("no_agent_search_evidence")
    if "current_best_post_residual_inspected" in requirements:
        current_formula = str(best_payload.get("formula", ""))
        inspected_current_best = any(
            item.get("observation") == "post_correction_residual_inspected"
            and str(item.get("inspected_formula", "")) == current_formula
            for item in case_memory.get("observation_memory", [])
        )
        if not inspected_current_best:
            blockers.append("current_best_post_residual_not_inspected")

    current_formula = str(best_payload.get("formula", ""))
    evidence_by_type = {
        str(item.get("evidence_type")): item
        for item in case_memory.get("observation_memory", [])
        if item.get("evidence_type") and str(item.get("inspected_formula", "")) == current_formula
    }
    family_comparison_row = evidence_by_type.get("family_hypothesis_comparison")
    family_comparison = (
        family_comparison_row.get("evidence", {})
        if isinstance(family_comparison_row, dict) and isinstance(family_comparison_row.get("evidence"), dict)
        else {}
    )
    current_preferred_by_family_comparison = bool(
        family_comparison_row
        and family_comparison_row.get("evidence_passed") is True
        and family_comparison.get("coverage_complete") is True
        and family_comparison.get("structural_status") in {"resolved", "equivalent"}
        and (
            family_comparison.get("preferred_hypothesis_id") == "current_formula"
            or "current_formula" in (family_comparison.get("equivalent_hypothesis_ids") or [])
        )
    )
    evidence_requirements = {
        "current_best_family_screened": ("family_screening", "current_best_family_not_screened"),
        "current_best_validation_failures_inspected": ("validation_failures", "current_best_validation_failures_not_inspected"),
        "current_best_term_stability_tested": ("term_stability", "current_best_term_stability_not_tested"),
        "current_best_complexity_challenged": ("complexity_challenge", "current_best_complexity_not_challenged"),
        "current_best_boundary_stress_tested": ("boundary_stress", "current_best_boundary_not_stress_tested"),
        "current_best_remaining_structure_tested": ("remaining_structure", "current_best_remaining_structure_not_tested"),
    }
    for requirement, (evidence_type, missing_blocker) in evidence_requirements.items():
        if requirement not in requirements:
            continue
        evidence_row = evidence_by_type.get(evidence_type)
        if evidence_row is None:
            blockers.append(missing_blocker)
            continue
        if evidence_row.get("evidence_passed") is False:
            evidence_payload = evidence_row.get("evidence", {}) if isinstance(evidence_row.get("evidence"), dict) else {}
            if evidence_type == "family_screening" and current_preferred_by_family_comparison:
                continue
            if evidence_type == "complexity_challenge" and not evidence_payload.get("completed", True):
                blockers.append("complexity_challenge_incomplete")
                continue
            failure_blockers = {
                "family_screening": "promising_candidate_family_unresolved",
                "term_stability": "term_stability_unacceptable",
                "complexity_challenge": "simpler_noninferior_formula_available",
                "boundary_stress": "boundary_stress_failed",
                "remaining_structure": "significant_remaining_structure",
            }
            blocker = failure_blockers.get(evidence_type)
            if blocker:
                blockers.append(blocker)
        elif (
            evidence_type == "remaining_structure"
            and evidence_row.get("evidence_passed") is True
        ):
            evidence_payload = (
                evidence_row.get("evidence", {})
                if isinstance(evidence_row.get("evidence"), dict)
                else {}
            )
            if (
                evidence_payload.get("significant_structure") is True
                and evidence_payload.get("predictively_actionable") is False
            ):
                warnings.append(
                    "statistically_significant_but_validation_falsified_structure"
                )
    if requested and not case_memory.get("hypothesis_memory"):
        unresolved.append("no_structured_hypothesis_memory")
    if requested and not case_memory.get("outcome_memory"):
        unresolved.append("no_outcome_memory")

    return {
        "requested": bool(requested),
        "accepted": bool(requested and not blockers),
        "blockers": blockers,
        "warnings": warnings,
        "unresolved_issues": unresolved,
        "requirements": requirements,
        "baseline_rmse": float(baseline_rmse) if np.isfinite(baseline_rmse) else None,
        "candidate_rmse": rmse if np.isfinite(rmse) else None,
        "candidate_complexity": complexity,
        "candidate_physical_violations": violations,
        "candidate_validation_rmse": candidate_validation if np.isfinite(candidate_validation) else None,
        "incumbent_validation_rmse": incumbent_validation if np.isfinite(incumbent_validation) else None,
        "pending_commitments": pending_commitments,
    }

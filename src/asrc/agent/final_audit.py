from __future__ import annotations

from typing import Any

from asrc.agent.evidence_tools import (
    challenge_formula_complexity,
    inspect_validation_failures,
    screen_candidate_families,
    stress_test_boundaries,
    test_remaining_structure,
    test_term_stability,
)
from asrc.agent.memory import residual_diagnostics
from asrc.agent.family_hypotheses import compare_family_hypotheses
from asrc.agent.verifier import verify_acceptance


EVIDENCE_REQUIREMENTS = {
    "current_best_family_screened": "screen_candidate_families",
    "current_best_validation_failures_inspected": "inspect_validation_failures",
    "current_best_term_stability_tested": "test_term_stability",
    "current_best_complexity_challenged": "challenge_formula_complexity",
    "current_best_boundary_stress_tested": "stress_test_boundaries",
    "current_best_remaining_structure_tested": "test_remaining_structure",
}

RECOVERY_ACTIONS = {
    "current_best_post_residual_not_inspected": "inspect_residual_pattern",
    "current_best_family_not_screened": "screen_candidate_families",
    "promising_candidate_family_unresolved": "compare_candidate_family",
    "current_best_validation_failures_not_inspected": "inspect_validation_failures",
    "current_best_term_stability_not_tested": "test_term_stability",
    "term_stability_unacceptable": "remove_unstable_terms",
    "current_best_complexity_not_challenged": "challenge_formula_complexity",
    "complexity_challenge_incomplete": "challenge_formula_complexity",
    "simpler_noninferior_formula_available": "remove_unstable_terms",
    "current_best_boundary_not_stress_tested": "stress_test_boundaries",
    "boundary_stress_failed": "tighten_constraints",
    "current_best_remaining_structure_not_tested": "test_remaining_structure",
    "significant_remaining_structure": "broaden_search",
    "worse_than_validation_incumbent": "revise_variables",
    "does_not_improve_baseline_validation": "broaden_search",
    "baseline_validation_comparison_missing": "inspect_validation_failures",
    "no_agent_search_evidence": "revise_variables",
    "unresolved_experiments": "resolve_pending_commitment",
}


def _observation(
    *,
    round_number: int,
    formula: str,
    observation: str,
    evidence_type: str = "",
    evidence_passed: bool | None = None,
    evidence: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "round": int(round_number),
        "audit": True,
        "observation": observation,
        "inspected_formula": formula,
        "post_correction_diagnostics": diagnostics or {},
        "evidence_type": evidence_type,
        "evidence_passed": evidence_passed,
        "evidence": evidence or {},
        "worst_error_regions": (diagnostics or {}).get("worst_angle_cells", [])[:5],
        "queried_knowledge_ids": [],
        "unresolved_issues": [],
    }


def _completed_formula_evidence(
    observations: list[dict[str, Any]],
    *,
    formula: str,
    evidence_type: str,
) -> dict[str, Any] | None:
    for item in reversed(observations):
        if (
            item.get("evidence_type") != evidence_type
            or str(item.get("inspected_formula", "")) != formula
        ):
            continue
        evidence = item.get("evidence")
        if isinstance(evidence, dict) and evidence.get("completed") is True:
            return evidence
    return None


def _resolve_audited_commitments(memory: dict[str, Any], executed_actions: set[str], round_number: int) -> None:
    commitments = memory.get("case_memory", {}).get("commitment_memory", [])
    for commitment in commitments:
        if commitment.get("status") != "pending":
            continue
        follow_up = str(commitment.get("follow_up_action", ""))
        if follow_up in executed_actions or follow_up == "accept_current_best":
            commitment["status"] = "resolved_by_final_audit"
            commitment["resolved_round"] = int(round_number)
            commitment["resolution_action"] = "deterministic_final_audit"


def _recommended_recovery(report: dict[str, Any], memory: dict[str, Any]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for blocker in report.get("blockers", []):
        action = RECOVERY_ACTIONS.get(str(blocker), "inspect_residual_pattern")
        if action == "resolve_pending_commitment":
            pending = [
                str(item.get("follow_up_action", ""))
                for item in memory.get("case_memory", {}).get("commitment_memory", [])
                if item.get("status") == "pending" and item.get("follow_up_action")
            ]
            action = pending[0] if pending else "inspect_residual_pattern"
        key = f"{blocker}:{action}"
        if key not in seen:
            recommendations.append({"blocker": str(blocker), "recommended_action": action})
            seen.add(key)
    return recommendations


def run_deterministic_final_audit(
    *,
    case: Any,
    result: Any,
    memory: dict[str, Any],
    round_number: int,
    attempt: int,
    trigger: str,
    baseline_rmse: float,
    allow_accept_before_round: int,
    requirements: list[str],
    incumbent_payload: dict[str, Any] | None,
    allowed_terms: list[str],
    grammar_index: list[dict[str, Any]],
    family_min_relative_improvement: float = 0.01,
    validation_budget_remaining: int | None = None,
    structural_audit_enabled: bool = True,
    structural_reference_terms: list[str] | None = None,
    structural_mode: str = "asrc",
    structural_max_terms: int = 3,
    structural_alpha_grid: list[float] | None = None,
    structural_validation_top_k: int = 8,
    structural_bootstrap_samples: int = 1000,
    structural_confidence: float = 0.95,
    seed: int = 20260713,
) -> dict[str, Any]:
    """Audit the current best deterministically without consuming an LLM round."""
    formula = str(result.best.get("formula", ""))
    observations = memory.setdefault("case_memory", {}).setdefault("observation_memory", [])
    evidence_records: list[dict[str, Any]] = []
    executed_actions = {"inspect_residual_pattern"}
    reused_evidence_types: list[str] = []
    validation_consumed = 0

    diagnostics = residual_diagnostics(
        case,
        allowed_terms,
        residual_values=result.predictions["residual_after_correction_MPa"].to_numpy(float),
        residual_source="final_audit_current_best",
    )
    residual_row = _observation(
        round_number=round_number,
        formula=formula,
        observation="post_correction_residual_inspected",
        diagnostics=diagnostics,
    )
    observations.append(residual_row)
    evidence_records.append(residual_row)

    if "current_best_family_screened" in requirements:
        evidence = screen_candidate_families(
            case,
            result,
            grammar_index,
            min_relative_improvement=family_min_relative_improvement,
        )
        row = _observation(
            round_number=round_number,
            formula=formula,
            observation="candidate_families_screened",
            evidence_type="family_screening",
            evidence_passed=bool(evidence["resolved"]),
            evidence=evidence,
        )
        observations.append(row)
        evidence_records.append(row)
        executed_actions.add("screen_candidate_families")

    if "current_best_validation_failures_inspected" in requirements:
        evidence = inspect_validation_failures(case, result)
        row = _observation(
            round_number=round_number,
            formula=formula,
            observation="validation_failures_inspected",
            evidence_type="validation_failures",
            evidence_passed=True,
            evidence=evidence,
        )
        observations.append(row)
        evidence_records.append(row)
        executed_actions.update({"inspect_validation_failures", "test_leave_one_angle"})

    if "current_best_term_stability_tested" in requirements:
        evidence = test_term_stability(case, result, n_bootstrap=64, seed=seed)
        row = _observation(
            round_number=round_number,
            formula=formula,
            observation="term_stability_tested",
            evidence_type="term_stability",
            evidence_passed=bool(evidence["stable"]),
            evidence=evidence,
        )
        observations.append(row)
        evidence_records.append(row)
        executed_actions.add("test_term_stability")

    if "current_best_complexity_challenged" in requirements:
        prior = _completed_formula_evidence(
            observations,
            formula=formula,
            evidence_type="complexity_challenge",
        )
        if prior is not None:
            evidence = {**prior, "reused_prior_evidence": True}
            reused_evidence_types.append("complexity_challenge")
        else:
            evidence = challenge_formula_complexity(
                case,
                result,
                noninferiority_relative=0.01,
                max_challenges=validation_budget_remaining,
            )
            validation_consumed = int(evidence["validation_candidates_evaluated"])
        row = _observation(
            round_number=round_number,
            formula=formula,
            observation="formula_complexity_challenged",
            evidence_type="complexity_challenge",
            evidence_passed=bool(evidence["completed"]) and not bool(evidence["simplification_available"]),
            evidence=evidence,
        )
        observations.append(row)
        evidence_records.append(row)
        executed_actions.add("challenge_formula_complexity")

    if "current_best_boundary_stress_tested" in requirements:
        evidence = stress_test_boundaries(case, result, dense_points=91)
        row = _observation(
            round_number=round_number,
            formula=formula,
            observation="boundary_stress_tested",
            evidence_type="boundary_stress",
            evidence_passed=bool(evidence["passed"]),
            evidence=evidence,
        )
        observations.append(row)
        evidence_records.append(row)
        executed_actions.add("stress_test_boundaries")

    if "current_best_remaining_structure_tested" in requirements:
        evidence = test_remaining_structure(
            case,
            result,
            allowed_terms,
            n_permutations=199,
            seed=seed,
            min_effect=0.2,
            max_validation_candidates=max(
                0,
                int(validation_budget_remaining) - validation_consumed,
            ),
        )
        validation_consumed += int(
            evidence.get("validation_candidates_evaluated", 0)
        )
        row = _observation(
            round_number=round_number,
            formula=formula,
            observation="remaining_structure_tested",
            evidence_type="remaining_structure",
            evidence_passed=bool(evidence["resolved"]),
            evidence=evidence,
        )
        observations.append(row)
        evidence_records.append(row)
        executed_actions.add("test_remaining_structure")

    if structural_audit_enabled:
        prior = next(
            (
                item.get("evidence")
                for item in reversed(observations)
                if item.get("evidence_type") == "family_hypothesis_comparison"
                and str(item.get("inspected_formula", "")) == formula
                and isinstance(item.get("evidence"), dict)
            ),
            None,
        )
        structural = prior or compare_family_hypotheses(
            case=case,
            current=result,
            grammar_index=grammar_index,
            reference_terms=list(structural_reference_terms or []),
            mode=structural_mode,
            max_terms=structural_max_terms,
            alpha_grid=list(structural_alpha_grid or [0.0, 0.001, 0.01, 0.1, 1.0]),
            enforce_physical=True,
            validation_top_k=structural_validation_top_k,
            bootstrap_samples=structural_bootstrap_samples,
            confidence=structural_confidence,
            seed=seed,
        )
        structural = {**structural, "reused_prior_evidence": bool(prior)}
        row = _observation(
            round_number=round_number,
            formula=formula,
            observation="registered_family_hypotheses_compared",
            evidence_type="family_hypothesis_comparison",
            evidence_passed=structural["structural_status"] in {"resolved", "equivalent", "not_applicable"},
            evidence=structural,
        )
        observations.append(row)
        evidence_records.append(row)
        executed_actions.add("compare_alternative_families")
    else:
        structural = {
            "structural_status": "not_applicable",
            "summary": "Structural family audit is disabled.",
            "coverage_complete": False,
            "hypotheses": [],
            "validation_candidates_evaluated": 0,
        }

    _resolve_audited_commitments(memory, executed_actions, round_number)
    acceptance = verify_acceptance(
        best_payload=result.best,
        baseline_rmse=baseline_rmse,
        round_number=round_number,
        allow_accept_before_round=allow_accept_before_round,
        requested=True,
        memory=memory,
        requirements=requirements,
        incumbent_payload=incumbent_payload,
    )
    recommended_recovery = _recommended_recovery(acceptance, memory)
    if structural["structural_status"] == "unresolved":
        recommended_recovery.append(
            {
                "blocker": "structural_hypothesis_unresolved",
                "recommended_action": "compare_candidate_family",
                "recommended_family_ids": structural.get("recommended_family_ids", []),
            }
        )
    audit = {
        "audit_type": "deterministic_final_audit",
        "audit_version": "1.0",
        "attempt": int(attempt),
        "trigger": trigger,
        "round": int(round_number),
        "formula": formula,
        "accepted": bool(acceptance["accepted"]),
        "predictive_accepted": bool(acceptance["accepted"]),
        "predictive_status": "accepted" if acceptance["accepted"] else "provisional_unverified",
        "structural_status": structural["structural_status"],
        "structural_summary": structural["summary"],
        "structural_report": structural,
        "result_status": "accepted" if acceptance["accepted"] else "provisional_pending_recovery",
        "blockers": acceptance.get("blockers", []),
        "warnings": acceptance.get("warnings", []),
        "unresolved_issues": acceptance.get("unresolved_issues", []),
        "recommended_recovery": recommended_recovery,
        "validation_candidates_evaluated": validation_consumed,
        "structural_validation_candidates_evaluated": (
            0 if structural.get("reused_prior_evidence") else int(structural.get("validation_candidates_evaluated", 0))
        ),
        "executed_actions": sorted(executed_actions),
        "reused_evidence_types": reused_evidence_types,
        "evidence_records": evidence_records,
        "acceptance_report": acceptance,
    }
    memory.setdefault("case_memory", {}).setdefault("acceptance_memory", []).append(
        {
            "round": int(round_number),
            "audit_attempt": int(attempt),
            "trigger": trigger,
            "requested": True,
            "accepted": bool(acceptance["accepted"]),
            "blockers": acceptance.get("blockers", []),
            "warnings": acceptance.get("warnings", []),
            "unresolved_issues": acceptance.get("unresolved_issues", []),
            "recommended_recovery": audit["recommended_recovery"],
        }
    )
    return audit

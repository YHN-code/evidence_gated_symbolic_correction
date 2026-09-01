from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asrc.agent.llm_client import LLMClientConfig, LLMResponseError, call_llm_json, load_llm_config
from asrc.agent.context import DEFAULT_AGENT_ACTIONS, build_agent_context, save_agent_context
from asrc.agent.evidence_tools import (
    challenge_formula_complexity,
    inspect_validation_failures,
    screen_candidate_families,
    stress_test_boundaries,
    test_remaining_structure,
    test_term_stability,
)
from asrc.agent.final_audit import run_deterministic_final_audit
from asrc.agent.family_hypotheses import compare_family_hypotheses
from asrc.agent.knowledge import (
    knowledge_index,
    load_knowledge_base,
    prompt_knowledge_summary,
    query_knowledge_blocks,
    select_knowledge_blocks,
)
from asrc.agent.memory import compact_memory_view, create_agent_memory, residual_diagnostics, update_agent_memory
from asrc.agent.prompts import build_decision_prompt, build_revision_prompt, build_schema_repair_prompt
from asrc.agent.rules import active_rule_ids, load_rule_pack, prompt_rule_summary, rule_pack_version
from asrc.agent.schema import AgentDecisionError, validate_agent_decision
from asrc.agent.verifier import acceptance_contract, verify_acceptance
from asrc.baselines.attribution import accessible_feature_terms
from asrc.features.angle_features import make_angle_features
from asrc.features.bounded_grammar import (
    bounded_grammar_index,
    bounded_grammar_terms,
    load_bounded_grammar,
    resolve_bounded_families,
)
from asrc.symbolic.sparse_regression import SearchResult, fit_fixed_terms, run_candidate_search
from asrc.utils.agent_logs import save_agent_decision
from asrc.utils.io import read_json, write_json, write_table_bundle
from asrc.utils.progress import progress_message


class AgentSearchError(RuntimeError):
    pass


SEARCH_ACTIONS = {
    "revise_variables",
    "revise_variables_no_prior_best",
    "propose_feature_family",
    "broaden_search",
    "remove_unstable_terms",
    "tighten_constraints",
    "compare_candidate_family",
}

EXTENDED_EVIDENCE_REQUIREMENTS = [
    "current_best_family_screened",
    "current_best_validation_failures_inspected",
    "current_best_term_stability_tested",
    "current_best_complexity_challenged",
    "current_best_boundary_stress_tested",
    "current_best_remaining_structure_tested",
]


class AgenticSearchResult:
    def __init__(
        self,
        result: SearchResult,
        decision: dict[str, Any],
        round_rows: list[dict[str, Any]],
        candidates: pd.DataFrame,
        *,
        result_status: str = "accepted",
        final_audit: dict[str, Any] | None = None,
    ):
        self.result = result
        self.decision = decision
        self.round_rows = round_rows
        self.candidates = candidates
        self.result_status = result_status
        self.final_audit = final_audit or {}
        self.predictive_status = str(self.final_audit.get("predictive_status", result_status))
        self.structural_status = str(self.final_audit.get("structural_status", "not_applicable"))


def residual_summary(case: Any) -> dict[str, Any]:
    df = case.frame
    residual = df[case.residual].to_numpy(float)
    summary = {
        "case_name": case.case_name,
        "data_type": case.data_type,
        "n_rows": int(len(df)),
        "x_columns": case.x_columns,
        "target": case.y_true,
        "baseline": case.y_base,
        "residual_mean": float(np.mean(residual)),
        "residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "residual_min": float(np.min(residual)),
        "residual_max": float(np.max(residual)),
    }
    if case.group and case.group in df.columns:
        summary["group_values"] = sorted(df[case.group].astype(str).unique().tolist())
    for col in ["beta_deg", "psi_deg"]:
        if col in df.columns:
            summary[col] = sorted(float(v) for v in df[col].unique().tolist())
    return summary


def allowed_feature_names(
    case: Any,
    extra_terms: list[str] | None = None,
    include_grammar: bool = False,
) -> list[str]:
    return accessible_feature_terms(
        case,
        extra_terms=extra_terms,
        include_grammar=include_grammar,
    )


def deterministic_decision(case: Any, knowledge_terms: list[str] | None = None, knowledge_constraints: list[str] | None = None) -> dict[str, Any]:
    allowed = allowed_feature_names(case, extra_terms=knowledge_terms)
    has_psi = "psi_deg" in case.frame.columns
    preferred = ["m_beta", "m2_beta", "sat_m2_beta_b3", "sin_beta", "cos_beta"]
    if has_psi:
        preferred = ["m_beta", "m2_beta", "m_psi", "m2_psi", "m_beta_m_psi", "sin_psi", "cos_psi", "sin2_psi"]
    recommended = [term for term in (knowledge_terms or []) if term in allowed]
    recommended += [term for term in preferred if term in allowed and term not in recommended]
    recommended = recommended[:14] or allowed[:4]
    constraints = ["positive_corrected_strength", "bounded_endpoints", "low_complexity"]
    for constraint in knowledge_constraints or []:
        if constraint not in constraints:
            constraints.append(constraint)
    return {
        "agent_type": "deterministic",
        "diagnosis": "Residuals are modeled with bounded angle features selected from the executable mechanics knowledge base and configured mechanics prior.",
        "recommended_variables": recommended,
        "forbidden_variables": ["tan_beta", "reciprocal_sin_beta", "reciprocal_cos_beta"],
        "formula_skeletons": ["a0 + sum(ai * bounded_angle_feature_i)"],
        "constraints": constraints,
        "critic_notes": "Candidate formulas must reduce RMSE while avoiding endpoint singularities and negative corrected strengths.",
        "next_action": "run_constrained_symbolic_search",
        "round_action": "revise_variables",
        "revision_reason": "Initial deterministic mechanics prior.",
        "expected_improvement": "Reduce systematic residual error while preserving physical admissibility.",
    }


def _attach_rule_metadata(
    decision: dict[str, Any],
    version: str | None,
    active_ids: list[str] | None,
) -> dict[str, Any]:
    decision["rule_pack_version"] = version or "unknown"
    decision["active_rule_ids"] = active_ids or []
    return decision


def _attach_knowledge_metadata(decision: dict[str, Any], selection: dict[str, Any] | None) -> dict[str, Any]:
    selection = selection or {}
    decision["knowledge_base_version"] = selection.get("knowledge_base_version", "unknown")
    decision["active_knowledge_ids"] = selection.get("active_knowledge_ids", [])
    decision["knowledge_recommended_terms"] = selection.get("recommended_terms", [])
    return decision


def build_agent_decision(
    case: Any,
    run_dir: Path,
    name: str,
    use_llm: bool = False,
    allow_fallback: bool = False,
    llm_config: str | Path | None = None,
    previous_rounds: list[dict[str, Any]] | None = None,
    candidate_feedback: list[dict[str, Any]] | None = None,
    optimization_memory: dict[str, Any] | None = None,
    rule_summary: list[dict[str, str]] | None = None,
    rules_version: str | None = None,
    active_rules: list[str] | None = None,
    knowledge_summary: list[dict[str, Any]] | None = None,
    knowledge_selection: dict[str, Any] | None = None,
    agent_context: dict[str, Any] | None = None,
    structured_output_repair_attempts: int = 2,
) -> dict[str, Any]:
    summary = residual_summary(case)
    knowledge_terms = (knowledge_selection or {}).get("recommended_terms", [])
    knowledge_constraints = (knowledge_selection or {}).get("constraints", [])
    # Dynamic grammar terms are activated through explicit family tools. They
    # are not ordinary variables that the model can enable by listing them.
    allowed = allowed_feature_names(case, extra_terms=knowledge_terms, include_grammar=False)
    if not use_llm:
        decision = deterministic_decision(case, knowledge_terms=knowledge_terms, knowledge_constraints=knowledge_constraints)
        decision["case_summary"] = summary
        _attach_knowledge_metadata(decision, knowledge_selection)
        return _attach_rule_metadata(decision, rules_version, active_rules)

    previous_rounds = previous_rounds or []
    candidate_feedback = candidate_feedback or []
    if previous_rounds:
        prompt = build_revision_prompt(
            summary,
            allowed,
            previous_rounds,
            candidate_feedback,
            optimization_memory=optimization_memory,
            rule_summary=rule_summary,
            knowledge_summary=knowledge_summary,
            agent_context=agent_context,
        )
    else:
        prompt = build_decision_prompt(summary, allowed, rule_summary=rule_summary, knowledge_summary=knowledge_summary, agent_context=agent_context)
    try:
        fallback_variables = deterministic_decision(case, knowledge_terms=knowledge_terms)["recommended_variables"]
        available_actions = list((agent_context or {}).get("available_actions", DEFAULT_AGENT_ACTIONS))
        repair_limit = max(0, int(structured_output_repair_attempts))
        repair_errors: list[str] = []
        raw: Any = None
        try:
            raw = call_llm_json(prompt, config_path=llm_config)
            decision = validate_agent_decision(
                deepcopy(raw),
                set(allowed),
                fallback_variables=fallback_variables,
                available_actions=set(available_actions),
            )
        except (AgentDecisionError, LLMResponseError) as initial_error:
            current_error: Exception = initial_error
            invalid_output: Any = raw
            if isinstance(initial_error, LLMResponseError):
                invalid_output = initial_error.content if initial_error.content is not None else ""
            for repair_index in range(1, repair_limit + 1):
                invalid_number = repair_index
                invalid_file = f"{name}_llm_invalid_{invalid_number:03d}.json"
                write_json(
                    run_dir / "agent_decisions" / invalid_file,
                    {"validation_error": str(current_error), "invalid_output": invalid_output},
                )
                repair_errors.append(str(current_error))
                repair_prompt = build_schema_repair_prompt(
                    invalid_output,
                    str(current_error),
                    allowed,
                    available_actions,
                    original_task_prompt=prompt if not isinstance(invalid_output, dict) else None,
                )
                repaired_raw: Any = None
                try:
                    repaired_raw = call_llm_json(repair_prompt, config_path=llm_config)
                    decision = validate_agent_decision(
                        deepcopy(repaired_raw),
                        set(allowed),
                        fallback_variables=fallback_variables,
                        available_actions=set(available_actions),
                    )
                except (AgentDecisionError, LLMResponseError) as repair_error:
                    current_error = repair_error
                    invalid_output = (
                        repair_error.content
                        if isinstance(repair_error, LLMResponseError) and repair_error.content is not None
                        else repaired_raw
                    )
                    continue
                raw = repaired_raw
                decision["schema_repair"] = {
                    "attempted": True,
                    "attempts_used": repair_index,
                    "attempts_allowed": repair_limit,
                    "errors": repair_errors,
                    "additional_llm_calls": repair_index,
                }
                break
            else:
                final_invalid_file = f"{name}_llm_invalid_{repair_limit + 1:03d}.json"
                write_json(
                    run_dir / "agent_decisions" / final_invalid_file,
                    {"validation_error": str(current_error), "invalid_output": invalid_output},
                )
                repair_errors.append(str(current_error))
                raise AgentDecisionError(
                    f"Structured LLM output remained invalid after {repair_limit} repair attempt(s). "
                    f"Last error: {current_error}"
                ) from current_error
        decision["agent_type"] = "llm"
        decision["case_summary"] = summary
        decision["llm_config_path"] = str(llm_config) if llm_config else None
        _attach_knowledge_metadata(decision, knowledge_selection)
        _attach_rule_metadata(decision, rules_version, active_rules)
        write_json(run_dir / "agent_decisions" / f"{name}_llm_raw.json", raw)
        return decision
    except Exception as exc:
        error_payload = {"agent_type": "llm", "error_type": type(exc).__name__, "error": str(exc), "case_summary": summary}
        if "raw" in locals():
            error_payload["raw_decision"] = raw
        if "repair_errors" in locals() and repair_errors:
            error_payload["structured_output_repair_errors"] = repair_errors
        if isinstance(exc, LLMResponseError):
            error_payload["raw_response"] = exc.payload
            error_payload["raw_content"] = exc.content
        write_json(run_dir / "agent_decisions" / f"{name}_llm_error.json", error_payload)
        if not allow_fallback:
            raise
        decision = deterministic_decision(case, knowledge_terms=knowledge_terms, knowledge_constraints=knowledge_constraints)
        decision["agent_type"] = "deterministic_fallback"
        decision["llm_error"] = str(exc)
        decision["case_summary"] = summary
        _attach_knowledge_metadata(decision, knowledge_selection)
        return _attach_rule_metadata(decision, rules_version, active_rules)


def _top_candidate_feedback(candidates: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    columns = ["candidate_id", "formula", "rmse", "selection_validation_rmse", "leave_one_beta_rmse", "leave_one_psi_rmse", "mae", "r2", "complexity", "physical_violations", "accepted", "rejection_reason"]
    available = [column for column in columns if column in candidates.columns]
    if not available:
        return []
    ranking = candidates.copy()
    ranking["_not_accepted"] = (~ranking["accepted"].astype(bool)).astype(int) if "accepted" in ranking.columns else 0
    if "selection_validation_rmse" in ranking.columns:
        validation = pd.to_numeric(ranking["selection_validation_rmse"], errors="coerce")
        validated = ranking.loc[validation.notna()].copy()
        if not validated.empty:
            ranking = validated
            validation = pd.to_numeric(ranking["selection_validation_rmse"], errors="coerce")
        ranking["_selection_score"] = validation.fillna(ranking["rmse"])
    sort_cols = [column for column in ["_not_accepted", "physical_violations", "_selection_score", "rmse", "complexity"] if column in ranking.columns]
    ranking = ranking.sort_values(sort_cols).head(top_k)
    return ranking[available].to_dict(orient="records")


def _optional_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _candidate_frontier_rows(
    candidates: pd.DataFrame,
    *,
    agent_name: str,
    round_number: int,
    phase: str,
    selected_candidate_id: str | None,
) -> list[dict[str, Any]]:
    """Extract genuine improvements from candidate enumeration and CV evaluation order."""
    if candidates.empty or "rmse" not in candidates.columns:
        return []

    work = candidates.copy()
    if "candidate_enumeration_order" not in work.columns:
        work["candidate_enumeration_order"] = np.arange(1, len(work) + 1)
    accepted = work["accepted"].astype(bool) if "accepted" in work.columns else pd.Series(True, index=work.index)
    violations = (
        pd.to_numeric(work["physical_violations"], errors="coerce").fillna(0)
        if "physical_violations" in work.columns
        else pd.Series(0, index=work.index)
    )
    eligible = work.loc[accepted & violations.eq(0)].copy()
    if eligible.empty:
        return []

    rows: list[dict[str, Any]] = []

    def append_frontier(source: pd.DataFrame, trace_type: str, order_column: str, metric_column: str) -> None:
        ordered = source.sort_values(order_column, kind="stable")
        running_best = float("inf")
        step = 0
        for _, candidate in ordered.iterrows():
            metric = _optional_float(candidate.get(metric_column))
            if metric is None or metric >= running_best - 1e-12:
                continue
            running_best = metric
            step += 1
            candidate_id = str(candidate.get("candidate_id", ""))
            rows.append(
                {
                    "agent_name": agent_name,
                    "round": int(round_number),
                    "phase": phase,
                    "trace_type": trace_type,
                    "frontier_step": step,
                    "candidate_id": candidate_id,
                    "candidate_enumeration_order": int(candidate.get("candidate_enumeration_order", 0)),
                    "validation_evaluation_order": _optional_float(candidate.get("validation_evaluation_order")),
                    "candidate_rmse": _optional_float(candidate.get("rmse")),
                    "candidate_validation_rmse": _optional_float(candidate.get("selection_validation_rmse")),
                    "frontier_rmse": running_best,
                    "complexity": int(candidate.get("complexity", 0)),
                    "physical_violations": int(candidate.get("physical_violations", 0)),
                    "formula": str(candidate.get("formula", "")),
                    "selected_proposal": bool(candidate_id == selected_candidate_id),
                }
            )

    append_frontier(eligible, "training_discovery", "candidate_enumeration_order", "rmse")
    if "selection_validation_rmse" in eligible.columns:
        validation = pd.to_numeric(eligible["selection_validation_rmse"], errors="coerce")
        validated = eligible.loc[validation.notna()].copy()
        if not validated.empty:
            if "validation_evaluation_order" not in validated.columns:
                validated["validation_evaluation_order"] = validated["candidate_enumeration_order"]
            append_frontier(
                validated,
                "validation_evaluation",
                "validation_evaluation_order",
                "selection_validation_rmse",
            )
    return rows


def _log_candidate_frontier(run_dir: Path, rows: list[dict[str, Any]], verbose: bool) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    for trace_type, trace in frame.groupby("trace_type", sort=False):
        trace = trace.sort_values("frontier_step")
        path = " -> ".join(f"{value:.6g}" for value in trace["frontier_rmse"].astype(float))
        first = trace.iloc[0]
        progress_message(
            run_dir,
            f"Candidate RMSE path for {first['agent_name']}",
            verbose,
            phase=first["phase"],
            trace=trace_type,
            points=len(trace),
            path=path,
        )


def _relative_improvement(previous_best_rmse: float | None, current_rmse: float) -> float | None:
    if previous_best_rmse is None or previous_best_rmse <= 0.0 or not np.isfinite(previous_best_rmse):
        return None
    return float((previous_best_rmse - current_rmse) / previous_best_rmse)


def _selection_score(payload: dict[str, Any]) -> float:
    validation = float(payload.get("selection_validation_rmse", float("nan")))
    return validation if np.isfinite(validation) else float(payload["rmse"])


def _ranking_tuple(result: SearchResult) -> tuple[int, float, int, float]:
    best = result.best
    return (int(best.get("physical_violations", 0)), _selection_score(best), int(best["complexity"]), float(best["mae"]))


def _simpler_noninferior(candidate: SearchResult, incumbent: SearchResult, tolerance: float = 0.01) -> bool:
    candidate_best = candidate.best
    incumbent_best = incumbent.best
    if int(candidate_best.get("physical_violations", 0)) > int(incumbent_best.get("physical_violations", 0)):
        return False
    if int(candidate_best.get("complexity", 0)) >= int(incumbent_best.get("complexity", 0)):
        return False
    candidate_score = _selection_score(candidate_best)
    incumbent_score = _selection_score(incumbent_best)
    return bool(
        np.isfinite(candidate_score)
        and np.isfinite(incumbent_score)
        and candidate_score <= incumbent_score * (1.0 + max(0.0, float(tolerance)))
    )


def _unique_ordered(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _evidence_guided_expansion_terms(
    memory: dict[str, Any] | None,
    best_result: SearchResult | None,
    allowed: list[str],
    *,
    limit: int = 6,
) -> list[str]:
    if not memory or best_result is None:
        return []
    current_formula = str(best_result.best.get("formula", ""))
    allowed_set = set(allowed)
    terms: list[str] = []
    observations = memory.get("case_memory", {}).get("observation_memory", [])
    for row in reversed(observations):
        inspected_formula = str(row.get("inspected_formula", ""))
        if inspected_formula and inspected_formula != current_formula:
            continue
        evidence = row.get("evidence", {}) if isinstance(row.get("evidence"), dict) else {}
        if row.get("evidence_type") == "remaining_structure":
            max_term = str(evidence.get("max_term", ""))
            if max_term in allowed_set:
                terms.append(max_term)
        diagnostics = row.get("post_correction_diagnostics", {})
        if isinstance(diagnostics, dict):
            for item in diagnostics.get("feature_correlations", []) or []:
                term = str(item.get("variable", ""))
                if term in allowed_set:
                    terms.append(term)
        if len(_unique_ordered(terms)) >= limit:
            break
    return _unique_ordered(terms)[: max(1, int(limit))]


def _action_variables(
    action: str,
    recommended: list[str],
    allowed: list[str],
    best_result: SearchResult | None,
    memory: dict[str, Any] | None = None,
    *,
    max_variables: int = 12,
) -> list[str]:
    if action == "broaden_search":
        current_terms = (
            [str(term) for term in best_result.best.get("terms", []) if term != "1"]
            if best_result is not None
            else []
        )
        evidence_terms = _evidence_guided_expansion_terms(memory, best_result, allowed)
        variables = [term for term in _unique_ordered([*current_terms, *recommended, *evidence_terms]) if term in set(allowed)]
        if not variables:
            variables = [term for term in allowed[:1]]
        if best_result is not None and set(variables) == set(current_terms):
            variables.extend(term for term in allowed if term not in set(variables))
        return _unique_ordered(variables)[: max(1, int(max_variables))]
    if action == "keep_best" and best_result is not None:
        best_terms = [term for term in best_result.best.get("terms", []) if term != "1"]
        return _unique_ordered([*recommended, *best_terms])
    if action == "tighten_constraints" and best_result is not None:
        return [term for term in best_result.best.get("terms", []) if term != "1"]
    return recommended


def _action_max_terms(action: str, current_max_terms: int, base_max_terms: int, max_terms_cap: int) -> int:
    if action == "broaden_search":
        return min(max(current_max_terms + 1, base_max_terms + 1), max_terms_cap)
    if action == "compare_candidate_family":
        return min(max(current_max_terms + 1, base_max_terms), max_terms_cap)
    if action == "tighten_constraints":
        return max(1, current_max_terms - 1)
    return current_max_terms


def _decision_topics(decision: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    topics = [str(item) for item in decision.get("knowledge_topics", []) if str(item).strip()]
    tool_inputs = decision.get("tool_inputs", {}) if isinstance(decision.get("tool_inputs", {}), dict) else {}
    for key in ["topic", "topics", "category"]:
        value = tool_inputs.get(key)
        if isinstance(value, list):
            topics.extend(str(item) for item in value)
        elif value:
            topics.append(str(value))
    if not topics:
        case_name = str(metadata.get("case", "")).lower()
        if "dinh" in case_name:
            topics.extend(["anisotropy", "weak_plane", "angle_features"])
        elif "ma2018" in case_name or "numerical" in case_name:
            topics.extend(["weak_plane", "brazilian_test", "angle_features"])
        else:
            topics.extend(["residual_correction", "angle_features"])
    return list(dict.fromkeys(topic for topic in topics if topic))


def _feature_family_terms(
    case: Any,
    decision: dict[str, Any],
    grammar: dict[str, Any],
    allowed: list[str],
) -> dict[str, Any]:
    tool_inputs = decision.get("tool_inputs", {}) if isinstance(decision.get("tool_inputs", {}), dict) else {}
    families = tool_inputs.get("feature_families") or tool_inputs.get("feature_family") or []
    if isinstance(families, str):
        families = [families]
    resolution = resolve_bounded_families(case, [str(family) for family in families], grammar)
    resolution["terms"] = [term for term in resolution["terms"] if term in allowed]
    return resolution


def _tool_output_base(action: str, decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "executed_tool": action,
        "query_reason": decision.get("action_reason", decision.get("revision_reason", "")),
        "queried_knowledge_ids": [],
        "returned_terms": [],
        "returned_constraints": [],
        "observation": action,
        "worst_error_regions": [],
    }


def _bounded_tool_int(tool_inputs: dict[str, Any], key: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(tool_inputs.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _bounded_tool_float(tool_inputs: dict[str, Any], key: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(tool_inputs.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _evidence_guided_removals(
    decision: dict[str, Any],
    memory: dict[str, Any],
    best_result: SearchResult | None,
) -> tuple[list[str], str]:
    if best_result is None:
        return [], "no_current_formula"
    current_terms = [str(term) for term in best_result.best.get("terms", []) if term != "1"]
    current_set = set(current_terms)
    tool_inputs = decision.get("tool_inputs", {}) if isinstance(decision.get("tool_inputs"), dict) else {}
    explicit = tool_inputs.get("remove_terms", [])
    if isinstance(explicit, str):
        explicit = [explicit]
    explicit_terms = [str(term) for term in explicit if str(term) in current_set]
    if explicit_terms:
        return _unique_ordered(explicit_terms), "tool_inputs.remove_terms"

    forbidden = [str(term) for term in decision.get("forbidden_variables", []) if str(term) in current_set]
    if forbidden:
        return _unique_ordered(forbidden), "forbidden_variables"

    observations = memory.get("case_memory", {}).get("observation_memory", [])
    current_formula = str(best_result.best.get("formula", ""))
    for row in reversed(observations):
        if str(row.get("inspected_formula", "")) != current_formula:
            continue
        evidence = row.get("evidence", {}) if isinstance(row.get("evidence"), dict) else {}
        if row.get("evidence_type") == "complexity_challenge" and evidence.get("simplification_available"):
            removed = str((evidence.get("best_simpler") or {}).get("removed_term", ""))
            if removed in current_set:
                return [removed], "complexity_challenge"
        if row.get("evidence_type") == "term_stability" and not evidence.get("stable", True):
            unstable = [
                str(item.get("term"))
                for item in evidence.get("terms", [])
                if str(item.get("term")) in current_set
                and (float(item.get("sign_consistency", 1.0)) < 0.8 or float(item.get("coefficient_cv", 0.0)) > 1.0)
            ]
            if unstable:
                return _unique_ordered(unstable), "term_stability"

    failed = [str(term) for term in memory.get("failed_variables", []) if str(term) in current_set]
    return (_unique_ordered(failed), "failed_variable_memory") if failed else ([], "no_supported_removal")


def _best_metrics_payload(best: dict[str, Any]) -> dict[str, Any]:
    return {key: best[key] for key in ["rmse", "mae", "r2", "physical_violations", "complexity"] if key in best}


def _append_table_bundle(run_dir: Path, output_base: Path, rows: list[dict[str, Any]], key_columns: list[str]) -> None:
    if not rows:
        return
    current = pd.DataFrame(rows)
    csv_path = output_base.with_suffix(".csv")
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        for row in rows:
            mask = pd.Series(True, index=existing.index)
            for column in key_columns:
                mask &= existing[column].astype(str) == str(row[column])
            existing = existing.loc[~mask].copy()
        current = pd.concat([existing, current], ignore_index=True)
    write_table_bundle(current, output_base)


def _append_best_formula(run_dir: Path, payload: dict[str, Any]) -> None:
    path = run_dir / "formulas" / "llm_iterative_best_formulas.json"
    existing = read_json(path) if path.exists() else []
    existing = [row for row in existing if row.get("agent_name") != payload["agent_name"]]
    existing.append(payload)
    write_json(path, existing)


def _audit_simpler_candidate(report: dict[str, Any]) -> dict[str, Any] | None:
    if "simpler_noninferior_formula_available" not in report.get("blockers", []):
        return None
    for record in report.get("evidence_records", []):
        if record.get("evidence_type") != "complexity_challenge":
            continue
        evidence = record.get("evidence", {})
        if evidence.get("completed") and evidence.get("simplification_available"):
            candidate = evidence.get("best_simpler")
            return candidate if isinstance(candidate, dict) else None
    return None


def run_agentic_search(
    case: Any,
    run_dir: Path,
    agent_name: str,
    mode: str,
    max_terms: int,
    alpha_grid: list[float],
    enforce_physical: bool = True,
    use_llm: bool = False,
    allow_fallback: bool = False,
    llm_config: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    verbose: bool = False,
    validation_top_k: int | None = None,
    validation_budget_total: int | None = None,
    validation_budget_reserve: int = 0,
    max_rounds_override: int | None = None,
    available_actions_override: list[str] | None = None,
    knowledge_retrieval_enabled: bool = True,
    planner_memory_enabled: bool = True,
    final_audit_enabled_override: bool | None = None,
) -> AgenticSearchResult:
    metadata = metadata or {}
    label = metadata.get("rock") or metadata.get("dataset") or metadata.get("case") or agent_name
    rule_pack = load_rule_pack()
    rules_version = rule_pack_version(rule_pack)
    rule_summary = prompt_rule_summary(rule_pack)
    active_rules = active_rule_ids(rule_summary)
    knowledge_base = load_knowledge_base()
    feature_grammar = load_bounded_grammar()
    feature_grammar_summary = bounded_grammar_index(case, feature_grammar)
    knowledge_selection = select_knowledge_blocks(case, metadata=metadata, knowledge_base=knowledge_base)
    selected_knowledge_summary = prompt_knowledge_summary(knowledge_selection)
    knowledge_index_summary = (
        knowledge_index(knowledge_base) if knowledge_retrieval_enabled else []
    )
    progress_message(
        run_dir,
        f"Rock-mechanics rule pack loaded for {agent_name}",
        verbose,
        version=rules_version,
        active_rules=len(active_rules),
    )
    progress_message(
        run_dir,
        f"Executable knowledge selected for {agent_name}",
        verbose,
        version=knowledge_selection["knowledge_base_version"],
        blocks=len(knowledge_selection["active_knowledge_ids"]),
        terms=len(knowledge_selection["recommended_terms"]),
    )
    llm_runtime: LLMClientConfig | None = None
    if use_llm:
        progress_message(run_dir, f"Preparing LLM agent for {agent_name}", verbose, case=metadata.get("case", case.case_name), target=label)
        try:
            llm_runtime = load_llm_config(llm_config)
            progress_message(
                run_dir,
                f"LLM config loaded for {agent_name}",
                verbose,
                model=llm_runtime.model,
                max_rounds=llm_runtime.max_rounds,
                thinking=llm_runtime.thinking_type or "default",
            )
        except Exception as exc:
            write_json(
                run_dir / "agent_decisions" / f"{agent_name}_llm_config_error.json",
                {"agent_type": "llm", "error_type": type(exc).__name__, "error": str(exc), **metadata},
            )
            progress_message(run_dir, f"LLM config failed for {agent_name}", verbose, error=type(exc).__name__)
            if not allow_fallback:
                raise

    max_rounds = llm_runtime.max_rounds if use_llm and llm_runtime else 1
    if max_rounds_override is not None:
        max_rounds = max(1, int(max_rounds_override))
    top_k = llm_runtime.top_k_feedback if llm_runtime else 5
    early_stop_relative_rmse = llm_runtime.early_stop_relative_rmse if llm_runtime else 0.01
    early_stop_patience = llm_runtime.early_stop_patience if llm_runtime else 2
    available_actions = (
        list(available_actions_override)
        if available_actions_override is not None
        else (
            llm_runtime.agent_available_actions
            if llm_runtime and llm_runtime.agent_available_actions
            else DEFAULT_AGENT_ACTIONS
        )
    )
    if not knowledge_retrieval_enabled:
        available_actions = [
            action for action in available_actions if action != "query_knowledge"
        ]
    context_recent_rounds = llm_runtime.agent_context_recent_rounds if llm_runtime else 2
    knowledge_top_k = llm_runtime.agent_knowledge_top_k if llm_runtime else 3
    allow_accept_before_round = llm_runtime.agent_allow_accept_before_round if llm_runtime else 2
    acceptance_requirements = list(llm_runtime.agent_acceptance_requirements or []) if llm_runtime else []
    final_audit_enabled = bool(
        use_llm and llm_runtime and llm_runtime.agent_final_audit_enabled
    )
    if final_audit_enabled_override is not None:
        final_audit_enabled = bool(final_audit_enabled_override)
    audit_recovery_rounds = llm_runtime.agent_audit_recovery_rounds if final_audit_enabled and llm_runtime else 0
    structural_audit_enabled = bool(use_llm and llm_runtime and llm_runtime.agent_structural_audit_enabled)
    structural_validation_top_k = llm_runtime.agent_structural_validation_top_k if llm_runtime else 8
    structural_bootstrap_samples = llm_runtime.agent_structural_bootstrap_samples if llm_runtime else 1000
    structural_confidence = llm_runtime.agent_structural_confidence if llm_runtime else 0.95
    soft_exploration_limit = max(1, max_rounds - audit_recovery_rounds)
    if final_audit_enabled:
        progress_message(
            run_dir,
            f"Final-audit budget configured for {agent_name}",
            verbose,
            llm_round_limit=max_rounds,
            first_scheduled_audit_round=soft_exploration_limit,
            recovery_rounds=audit_recovery_rounds,
        )
    if use_llm and "validation_noninferior_to_incumbent" not in acceptance_requirements:
        acceptance_requirements.append("validation_noninferior_to_incumbent")
    if use_llm and "no_unresolved_experiments" not in acceptance_requirements:
        acceptance_requirements.append("no_unresolved_experiments")
    if use_llm and "agent_search_performed" not in acceptance_requirements:
        acceptance_requirements.append("agent_search_performed")
    if use_llm and "current_best_post_residual_inspected" not in acceptance_requirements:
        acceptance_requirements.append("current_best_post_residual_inspected")
    evidence_profile = str(metadata.get("evidence_profile", case.config.get("agent", {}).get("evidence_profile", "standard")))
    if use_llm and evidence_profile == "extended":
        for requirement in EXTENDED_EVIDENCE_REQUIREMENTS:
            if requirement not in acceptance_requirements:
                acceptance_requirements.append(requirement)
    contract = acceptance_contract(allow_accept_before_round, acceptance_requirements)
    if validation_top_k is None:
        validation_top_k = llm_runtime.agent_validation_top_k if llm_runtime else 32
    if validation_budget_total is None and llm_runtime is not None:
        validation_budget_total = llm_runtime.agent_validation_budget_total
    validation_budget_reserve = int(validation_budget_reserve)
    if validation_budget_reserve < 0:
        raise ValueError("validation_budget_reserve must be non-negative.")
    if validation_budget_total is not None and validation_budget_reserve >= int(validation_budget_total):
        raise ValueError("validation_budget_reserve must be smaller than validation_budget_total.")
    base_allowed = allowed_feature_names(
        case,
        extra_terms=knowledge_selection["recommended_terms"],
        include_grammar=False,
    )
    allowed = _unique_ordered([*base_allowed, *(bounded_grammar_terms(case) if use_llm else [])])
    symbolic_max_terms = case.config.get("symbolic", {}).get("max_terms", {})
    max_terms_cap = int(symbolic_max_terms.get("llm_max", max_terms + 1 if use_llm else max_terms))
    max_terms_cap = max(max_terms, max_terms_cap)
    current_max_terms = int(max_terms)
    memory = create_agent_memory(case, allowed, metadata, base_max_terms=max_terms)
    baseline_residual = case.frame[case.residual].to_numpy(float)
    baseline_rmse = float(np.sqrt(np.mean(baseline_residual**2)))
    validation_budget_used = 0

    def remaining_validation_budget() -> int | None:
        if validation_budget_total is None:
            return None
        return max(0, int(validation_budget_total) - validation_budget_used)

    def remaining_exploration_validation_budget() -> int | None:
        remaining = remaining_validation_budget()
        if remaining is None:
            return None
        return max(0, remaining - validation_budget_reserve)

    def next_validation_allocation() -> int:
        remaining = remaining_exploration_validation_budget()
        return int(validation_top_k) if remaining is None else min(int(validation_top_k), remaining)

    # Full ASRC is intentionally a strict extension of Knowledge-ASRC.  The
    # reference is selected with the same group-held-out criterion that later
    # governs agent proposals, so an LLM action cannot silently degrade it.
    validation_incumbent: SearchResult | None = None
    best_result: SearchResult | None = None
    best_decision: dict[str, Any] | None = None
    candidate_frontier_rows: list[dict[str, Any]] = []
    if use_llm:
        progress_message(run_dir, f"Building validation incumbent for {agent_name}", verbose)
        incumbent_allocation = next_validation_allocation()
        if incumbent_allocation <= 0:
            raise AgentSearchError("Validation budget must allow at least one incumbent candidate.")
        validation_incumbent = run_candidate_search(
            case,
            mode=mode,
            max_terms=max_terms,
            alpha_grid=alpha_grid,
            enforce_physical=enforce_physical,
            allowed_terms=knowledge_selection["recommended_terms"],
            selection_metric="group_cv",
            validation_top_k=incumbent_allocation,
        )
        validation_budget_used += int(validation_incumbent.best.get("validation_candidates_evaluated", incumbent_allocation))
        incumbent_frontier = _candidate_frontier_rows(
            validation_incumbent.candidates,
            agent_name=agent_name,
            round_number=0,
            phase="validation_incumbent",
            selected_candidate_id=str(validation_incumbent.best.get("candidate_id", "")),
        )
        candidate_frontier_rows.extend(incumbent_frontier)
        _log_candidate_frontier(run_dir, incumbent_frontier, verbose)
        progress_message(
            run_dir,
            f"Validation incumbent ready for {agent_name}",
            verbose,
            train_rmse=f"{float(validation_incumbent.best['rmse']):.6g}",
            validation_rmse=f"{_selection_score(validation_incumbent.best):.6g}",
            complexity=validation_incumbent.best["complexity"],
            candidates_evaluated=validation_incumbent.best.get("validation_candidates_evaluated", 0),
        )
        best_result = validation_incumbent
        best_decision = deterministic_decision(
            case,
            knowledge_terms=knowledge_selection["recommended_terms"],
            knowledge_constraints=knowledge_selection["constraints"],
        )
        best_decision["round_action"] = "knowledge_validation_incumbent"
        memory["best_formula"] = validation_incumbent.best.get("formula")
        memory["best_rmse"] = float(validation_incumbent.best["rmse"])
        memory["best_complexity"] = int(validation_incumbent.best["complexity"])
        memory["best_physical_violations"] = int(validation_incumbent.best.get("physical_violations", 0))
    round_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    candidate_frames: list[pd.DataFrame] = []
    previous_rounds: list[dict[str, Any]] = []
    candidate_feedback: list[dict[str, Any]] = (
        _top_candidate_feedback(validation_incumbent.candidates, top_k)
        if validation_incumbent is not None
        else []
    )
    lowest_train_rmse_so_far = float(best_result.best["rmse"]) if best_result is not None else float("inf")
    lowest_validation_rmse_so_far = _selection_score(best_result.best) if best_result is not None else float("inf")
    no_improve_count = 0
    stop_reason = "max_rounds"
    result_status = "provisional_pending_recovery" if use_llm else "completed_deterministic"
    final_audit_report: dict[str, Any] | None = None
    final_audit_rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    audit_attempts = 0
    rejected_audit_signature: tuple[str, tuple[str, ...], str] | None = None
    repeated_rejected_audits = 0

    for round_number in range(1, max_rounds + 1):
        audit_invalidated = False
        audit_stagnated = False
        previous_best_rmse = float(best_result.best["rmse"]) if best_result is not None else None
        previous_best_selection_score = _selection_score(best_result.best) if best_result is not None else None
        previous_best_formula = str(best_result.best.get("formula", "")) if best_result is not None else ""
        progress_message(
            run_dir,
            f"Agent round {round_number}/{max_rounds} started for {agent_name}",
            verbose,
            llm=bool(use_llm and llm_runtime),
            previous_best_rmse="NA" if previous_best_rmse is None else f"{previous_best_rmse:.6g}",
            previous_best_validation_rmse=(
                "NA" if previous_best_selection_score is None else f"{previous_best_selection_score:.6g}"
            ),
        )
        memory_view = (
            compact_memory_view(memory, max_rounds=context_recent_rounds)
            if planner_memory_enabled
            else {"planner_memory_enabled": False}
        )
        round_available_actions = list(available_actions)
        if round_number < allow_accept_before_round:
            round_available_actions = [action for action in round_available_actions if action != "accept_current_best"]
        if remaining_exploration_validation_budget() == 0:
            round_available_actions = [
                action for action in round_available_actions if action not in SEARCH_ACTIONS
            ]
        if remaining_validation_budget() == 0:
            round_available_actions = [
                action for action in round_available_actions if action != "challenge_formula_complexity"
            ]
        current_best_context = {}
        if best_result is not None:
            best = best_result.best
            beta_validation = float(best.get("leave_one_beta_rmse", float("nan")))
            psi_validation = float(best.get("leave_one_psi_rmse", float("nan")))
            aggregate_validation = float(best.get("selection_validation_rmse", float("nan")))
            current_best_context = {
                "formula": best.get("formula"),
                "training_rmse": best.get("rmse"),
                "group_validation_rmse": aggregate_validation if np.isfinite(aggregate_validation) else None,
                "leave_one_beta_rmse": beta_validation if np.isfinite(beta_validation) else None,
                "leave_one_psi_rmse": psi_validation if np.isfinite(psi_validation) else None,
                "validation_performed": bool(np.isfinite(beta_validation) or np.isfinite(psi_validation)),
                "complexity": best.get("complexity"),
                "physical_violations": best.get("physical_violations"),
            }
        context_payload = build_agent_context(
            case_summary=residual_summary(case),
            allowed_variables=base_allowed,
            rule_summary=rule_summary,
            knowledge_index=knowledge_index_summary,
            feature_grammar_index=feature_grammar_summary,
            memory_view=memory_view,
            previous_rounds=previous_rounds[-context_recent_rounds:],
            candidate_feedback=candidate_feedback,
            available_actions=round_available_actions,
            acceptance_contract=contract,
            validation_budget={
                "total": validation_budget_total,
                "used": validation_budget_used,
                "remaining": remaining_validation_budget(),
                "exploration_remaining": remaining_exploration_validation_budget(),
                "acceptance_reserve": validation_budget_reserve,
                "per_search_limit": validation_top_k,
            },
            current_best=current_best_context,
        )
        save_agent_context(run_dir, f"{agent_name}_round_{round_number:03d}", context_payload)
        decision = build_agent_decision(
            case,
            run_dir,
            f"{agent_name}_round_{round_number:03d}",
            use_llm=bool(use_llm and llm_runtime),
            allow_fallback=allow_fallback,
            llm_config=llm_config,
            previous_rounds=previous_rounds,
            candidate_feedback=candidate_feedback,
            optimization_memory=memory_view,
            rule_summary=rule_summary,
            rules_version=rules_version,
            active_rules=active_rules,
            knowledge_summary=knowledge_index_summary,
            knowledge_selection=knowledge_selection,
            agent_context=context_payload,
            structured_output_repair_attempts=(
                llm_runtime.structured_output_repair_attempts if llm_runtime is not None else 2
            ),
        )
        schema_repair_calls = int((decision.get("schema_repair") or {}).get("additional_llm_calls", 0))
        if schema_repair_calls:
            progress_message(
                run_dir,
                f"Agent repaired an invalid structured decision for {agent_name}",
                verbose,
                round=round_number,
                additional_llm_calls=schema_repair_calls,
                initial_error=((decision.get("schema_repair") or {}).get("errors") or [""])[0],
            )
        if use_llm and llm_runtime is None:
            decision = deterministic_decision(
                case,
                knowledge_terms=knowledge_selection["recommended_terms"],
                knowledge_constraints=knowledge_selection["constraints"],
            )
            decision["agent_type"] = "deterministic_fallback"
            decision["llm_error"] = "LLM configuration unavailable; fallback enabled."
            decision["case_summary"] = residual_summary(case)
            _attach_knowledge_metadata(decision, knowledge_selection)
            _attach_rule_metadata(decision, rules_version, active_rules)

        requested_action = str(decision.get("round_action", "revise_variables"))
        effective_action = requested_action
        recommended_variables = list(decision.get("recommended_variables") or [])
        tool_output = _tool_output_base(requested_action, decision)
        tool_inputs = decision.get("tool_inputs", {}) if isinstance(decision.get("tool_inputs", {}), dict) else {}
        evidence_validation_consumed = 0
        acceptance_report = verify_acceptance(
            best_payload=best_result.best if best_result is not None else None,
            baseline_rmse=baseline_rmse,
            round_number=round_number,
            allow_accept_before_round=allow_accept_before_round,
            requested=requested_action == "accept_current_best" and not final_audit_enabled,
            memory=memory,
            requirements=acceptance_requirements,
            incumbent_payload=validation_incumbent.best if validation_incumbent is not None else None,
        )
        direct_accept = bool(acceptance_report["accepted"])
        if requested_action == "accept_current_best" and not direct_accept:
            effective_action = "keep_best_accept_rejected" if best_result is not None else "revise_variables_no_prior_best"
            if final_audit_enabled and best_result is not None:
                effective_action = "keep_best"
        if requested_action == "query_knowledge" and knowledge_retrieval_enabled:
            query = query_knowledge_blocks(
                case,
                topics=_decision_topics(decision, metadata),
                top_k=knowledge_top_k,
                knowledge_base=knowledge_base,
            )
            tool_output.update(
                {
                    "executed_tool": "query_knowledge",
                    "query_topics": query["query_topics"],
                    "queried_knowledge_ids": query["queried_knowledge_ids"],
                    "returned_terms": query["returned_terms"],
                    "returned_constraints": query["returned_constraints"],
                    "observation": "knowledge_query_completed",
                }
            )
            recommended_variables = _unique_ordered([*recommended_variables, *query["returned_terms"]])
        if requested_action in {"propose_feature_family", "compare_candidate_family"}:
            family_resolution = _feature_family_terms(case, decision, feature_grammar, allowed)
            family_terms = family_resolution["terms"]
            tool_output.update(
                {
                    "executed_tool": requested_action,
                    "grammar_version": family_resolution["grammar_version"],
                    "selected_feature_families": family_resolution["selected_family_ids"],
                    "returned_terms": family_terms,
                }
            )
            if family_terms:
                incumbent_terms = (
                    [term for term in best_result.best.get("terms", []) if term != "1"]
                    if best_result is not None
                    else recommended_variables
                )
                recommended_variables = _unique_ordered([*incumbent_terms, *family_terms])
            else:
                effective_action = "keep_best_invalid_feature_family"
                tool_output["observation"] = "feature_family_not_resolved"
        if requested_action == "remove_unstable_terms":
            current_terms = [term for term in (best_result.best.get("terms", []) if best_result is not None else recommended_variables) if term != "1"]
            removed_terms, removal_source = _evidence_guided_removals(decision, memory, best_result)
            recommended_variables = [term for term in current_terms if term not in set(removed_terms)]
            if not recommended_variables:
                recommended_variables = current_terms
                removed_terms = []
                removal_source = "removal_would_empty_formula"
            tool_output.update(
                {
                    "executed_tool": "remove_unstable_terms",
                    "removed_terms": removed_terms,
                    "removal_source": removal_source,
                    "observation": "evidence_guided_terms_removed" if removed_terms else "no_supported_term_removal",
                }
            )
        if requested_action == "inspect_residual_pattern":
            if best_result is None:
                diagnostics = residual_diagnostics(case, allowed)
                inspected_formula = ""
            else:
                diagnostics = residual_diagnostics(
                    case,
                    allowed,
                    residual_values=best_result.predictions["residual_after_correction_MPa"].to_numpy(float),
                    residual_source="current_best_post_correction",
                )
                inspected_formula = str(best_result.best.get("formula", ""))
            tool_output.update(
                {
                    "executed_tool": "inspect_residual_pattern",
                    "observation": "post_correction_residual_inspected",
                    "inspected_formula": inspected_formula,
                    "post_correction_diagnostics": diagnostics,
                    "worst_error_regions": diagnostics.get("worst_angle_cells", [])[:5],
                }
            )
        if best_result is not None and requested_action == "screen_candidate_families":
            evidence = screen_candidate_families(
                case,
                best_result,
                feature_grammar_summary,
                min_relative_improvement=float(
                    feature_grammar.get("screening", {}).get("min_relative_improvement", 0.01)
                ),
            )
            tool_output.update(
                {
                    "executed_tool": "screen_candidate_families",
                    "observation": "candidate_families_screened",
                    "inspected_formula": str(best_result.best.get("formula", "")),
                    "evidence_type": "family_screening",
                    "evidence_passed": bool(evidence["resolved"]),
                    "evidence": evidence,
                    "recommended_next_actions": (
                        []
                        if evidence["resolved"]
                        else ["propose_feature_family", "compare_candidate_family"]
                    ),
                }
            )
        if best_result is not None and requested_action in {"inspect_validation_failures", "test_leave_one_angle"}:
            evidence = inspect_validation_failures(case, best_result)
            tool_output.update(
                {
                    "executed_tool": requested_action,
                    "observation": "validation_failures_inspected",
                    "inspected_formula": str(best_result.best.get("formula", "")),
                    "evidence_type": "validation_failures",
                    "evidence_passed": True,
                    "evidence": evidence,
                }
            )
        if best_result is not None and requested_action == "test_term_stability":
            evidence = test_term_stability(
                case,
                best_result,
                n_bootstrap=_bounded_tool_int(tool_inputs, "n_bootstrap", 64, 16, 256),
                seed=_bounded_tool_int(tool_inputs, "seed", int(metadata.get("seed", 20260713)), 0, 2**31 - 1),
            )
            tool_output.update(
                {
                    "executed_tool": "test_term_stability",
                    "observation": "term_stability_tested",
                    "inspected_formula": str(best_result.best.get("formula", "")),
                    "evidence_type": "term_stability",
                    "evidence_passed": bool(evidence["stable"]),
                    "evidence": evidence,
                }
            )
        if best_result is not None and requested_action == "challenge_formula_complexity":
            complexity_budget = (
                remaining_validation_budget()
                if round_number >= soft_exploration_limit
                else remaining_exploration_validation_budget()
            )
            evidence = challenge_formula_complexity(
                case,
                best_result,
                noninferiority_relative=_bounded_tool_float(tool_inputs, "noninferiority_relative", 0.01, 0.0, 0.1),
                max_challenges=complexity_budget,
            )
            evidence_validation_consumed = int(evidence["validation_candidates_evaluated"])
            validation_budget_used += evidence_validation_consumed
            tool_output.update(
                {
                    "executed_tool": "challenge_formula_complexity",
                    "observation": "formula_complexity_challenged",
                    "inspected_formula": str(best_result.best.get("formula", "")),
                    "evidence_type": "complexity_challenge",
                    "evidence_passed": bool(evidence["completed"]) and not bool(evidence["simplification_available"]),
                    "evidence": evidence,
                }
            )
        if best_result is not None and requested_action == "stress_test_boundaries":
            evidence = stress_test_boundaries(
                case,
                best_result,
                dense_points=_bounded_tool_int(tool_inputs, "dense_points", 91, 21, 181),
            )
            tool_output.update(
                {
                    "executed_tool": "stress_test_boundaries",
                    "observation": "boundary_stress_tested",
                    "inspected_formula": str(best_result.best.get("formula", "")),
                    "evidence_type": "boundary_stress",
                    "evidence_passed": bool(evidence["passed"]),
                    "evidence": evidence,
                }
            )
        if best_result is not None and requested_action == "test_remaining_structure":
            evidence = test_remaining_structure(
                case,
                best_result,
                allowed,
                n_permutations=_bounded_tool_int(tool_inputs, "n_permutations", 199, 49, 999),
                seed=_bounded_tool_int(tool_inputs, "seed", int(metadata.get("seed", 20260713)), 0, 2**31 - 1),
                min_effect=_bounded_tool_float(tool_inputs, "min_effect", 0.2, 0.0, 1.0),
                min_validation_improvement_relative=_bounded_tool_float(
                    tool_inputs,
                    "min_validation_improvement_relative",
                    0.01,
                    0.0,
                    0.2,
                ),
                max_validation_candidates=(
                    remaining_validation_budget()
                    if round_number >= soft_exploration_limit
                    else remaining_exploration_validation_budget()
                ),
            )
            evidence_validation_consumed += int(
                evidence.get("validation_candidates_evaluated", 0)
            )
            validation_budget_used += int(
                evidence.get("validation_candidates_evaluated", 0)
            )
            tool_output.update(
                {
                    "executed_tool": "test_remaining_structure",
                    "observation": "remaining_structure_tested",
                    "inspected_formula": str(best_result.best.get("formula", "")),
                    "evidence_type": "remaining_structure",
                    "evidence_passed": bool(evidence["resolved"]),
                    "evidence": evidence,
                }
            )
        if best_result is not None and requested_action == "compare_alternative_families":
            evidence = compare_family_hypotheses(
                case=case,
                current=best_result,
                grammar_index=feature_grammar_summary,
                reference_terms=(list(validation_incumbent.best.get("terms", [])) if validation_incumbent else []),
                mode=mode,
                max_terms=current_max_terms,
                alpha_grid=alpha_grid,
                enforce_physical=enforce_physical,
                validation_top_k=structural_validation_top_k,
                bootstrap_samples=structural_bootstrap_samples,
                confidence=structural_confidence,
                seed=int(metadata.get("seed", 20260713)),
            )
            tool_output.update(
                {
                    "executed_tool": "compare_alternative_families",
                    "observation": "registered_family_hypotheses_compared",
                    "inspected_formula": str(best_result.best.get("formula", "")),
                    "evidence_type": "family_hypothesis_comparison",
                    "evidence_passed": evidence["structural_status"] in {"resolved", "equivalent", "not_applicable"},
                    "evidence": evidence,
                }
            )
            structural_rows.extend(
                {
                    "agent_name": agent_name,
                    "round": round_number,
                    "audit_attempt": 0,
                    "source": "agent_tool",
                    "structural_status": evidence["structural_status"],
                    "preferred_hypothesis_id": evidence.get("preferred_hypothesis_id", ""),
                    "equivalent_hypothesis_ids": ";".join(evidence.get("equivalent_hypothesis_ids", [])),
                    **hypothesis,
                    **metadata,
                }
                for hypothesis in evidence.get("hypotheses", [])
            )
        if requested_action == "accept_current_best" and not direct_accept:
            effective_action = "revise_variables_no_prior_best"
            if best_result is not None:
                effective_action = "keep_best"
            if final_audit_enabled:
                tool_output.update(
                    {
                        "executed_tool": "final_audit_request",
                        "observation": "deterministic_final_audit_requested",
                    }
                )
                progress_message(
                    run_dir,
                    f"Final audit requested for {agent_name}",
                    verbose,
                    round=round_number,
                )
            else:
                tool_output.update({"executed_tool": "acceptance_verifier", "acceptance_report": acceptance_report})
                progress_message(
                    run_dir,
                    f"Acceptance rejected for {agent_name}",
                    verbose,
                    round=round_number,
                    blockers=",".join(acceptance_report.get("blockers", [])) or "none",
                )
        variables = _action_variables(effective_action, recommended_variables, allowed, best_result, memory)
        if not variables:
            variables = deterministic_decision(case, knowledge_terms=knowledge_selection["recommended_terms"])["recommended_variables"]
        search_max_terms = _action_max_terms(effective_action, current_max_terms, max_terms, max_terms_cap)
        search_max_terms = min(search_max_terms, max(1, len(variables)))
        tool_output["search_space"] = {
            "variables": variables,
            "variable_count": len(variables),
            "max_terms": search_max_terms,
            "guard": "evidence_guided_max_12_variables" if effective_action == "broaden_search" else "action_scoped",
        }
        if effective_action in {"broaden_search", "tighten_constraints", "compare_candidate_family"}:
            current_max_terms = search_max_terms

        search_requested = effective_action in SEARCH_ACTIONS or (best_result is None and not direct_accept)
        allocation = next_validation_allocation() if search_requested else 0
        budget_exhausted = bool(search_requested and allocation <= 0)
        if budget_exhausted:
            search_requested = False
            effective_action = f"{effective_action}_budget_exhausted"
            tool_output.update(
                {
                    "executed_tool": "validation_budget_guard",
                    "observation": "search_skipped_validation_budget_exhausted",
                }
            )

        performed_search = False
        if direct_accept:
            progress_message(
                run_dir,
                f"Accepting historical best for {agent_name}",
                verbose,
                round=round_number,
                action=requested_action,
                best_rmse=f"{float(best_result.best['rmse']):.6g}",
            )
            result = best_result
            candidates = pd.DataFrame()
        elif not search_requested and best_result is not None:
            progress_message(
                run_dir,
                f"Executing evidence-only action for {agent_name}",
                verbose,
                round=round_number,
                action=effective_action,
                validation_budget_used=validation_budget_used,
            )
            result = best_result
            candidates = pd.DataFrame()
        else:
            progress_message(
                run_dir,
                f"Running symbolic search for {agent_name}",
                verbose,
                round=round_number,
                variables=len(variables),
                action=effective_action,
                max_terms=search_max_terms,
                validation_candidates=allocation,
            )
            result = run_candidate_search(
                case,
                mode=mode,
                max_terms=search_max_terms,
                alpha_grid=alpha_grid,
                enforce_physical=enforce_physical,
                allowed_terms=variables,
                selection_metric="group_cv" if use_llm else "training_rmse",
                validation_top_k=allocation,
            )
            performed_search = True
            validation_budget_used += int(result.best.get("validation_candidates_evaluated", allocation))
            candidates = result.candidates.copy()
            candidates["agent_name"] = agent_name
            candidates["round"] = round_number
            candidate_frames.append(candidates)
            round_frontier = _candidate_frontier_rows(
                result.candidates,
                agent_name=agent_name,
                round_number=round_number,
                phase=f"agent_search_round_{round_number}",
                selected_candidate_id=str(result.best.get("candidate_id", "")),
            )
            candidate_frontier_rows.extend(round_frontier)
            _log_candidate_frontier(run_dir, round_frontier, verbose)
            tool_output["candidate_frontier"] = round_frontier

        tool_output["validation_budget"] = {
            "total": validation_budget_total,
            "used": validation_budget_used,
            "remaining": remaining_validation_budget(),
            "exploration_remaining": remaining_exploration_validation_budget(),
            "acceptance_reserve": validation_budget_reserve,
            "round_consumed": (
                int(result.best.get("validation_candidates_evaluated", 0)) if performed_search else evidence_validation_consumed
            ),
        }

        current_rmse = float(result.best["rmse"])
        proposal_selection_score = _selection_score(result.best)
        proposal_relative_improvement = (
            _relative_improvement(previous_best_selection_score, proposal_selection_score)
            if performed_search
            else None
        )
        if performed_search:
            lowest_train_rmse_so_far = min(lowest_train_rmse_so_far, current_rmse)
            lowest_validation_rmse_so_far = min(lowest_validation_rmse_so_far, proposal_selection_score)
        promote_simpler = bool(
            best_result is not None
            and requested_action in {"remove_unstable_terms", "tighten_constraints"}
            and _simpler_noninferior(
                result,
                best_result,
                tolerance=_bounded_tool_float(tool_inputs, "noninferiority_relative", 0.01, 0.0, 0.1),
            )
        )
        proposal_promoted = bool(
            performed_search
            and (best_result is None or _ranking_tuple(result) < _ranking_tuple(best_result) or promote_simpler)
        )
        if proposal_promoted:
            best_result = result
            best_decision = decision
            if promote_simpler:
                tool_output["simpler_noninferior_promoted"] = True
        if (
            final_audit_report is not None
            and str(best_result.best.get("formula", "")) != str(final_audit_report.get("formula", ""))
        ):
            final_audit_report = None
            result_status = "provisional_pending_recovery"
            audit_invalidated = True
            tool_output["prior_final_audit_invalidated"] = True
        best_selection_score = _selection_score(best_result.best)
        relative_improvement = _relative_improvement(previous_best_selection_score, best_selection_score)
        tool_output["search_proposal"] = {
            "performed_search": bool(performed_search),
            "proposal_formula": result.best.get("formula") if performed_search else None,
            "proposal_train_rmse": current_rmse if performed_search else None,
            "proposal_validation_rmse": proposal_selection_score if performed_search else None,
            "incumbent_train_rmse_before": previous_best_rmse,
            "incumbent_validation_rmse_before": previous_best_selection_score,
            "proposal_promoted": proposal_promoted,
            "proposal_relative_validation_improvement": proposal_relative_improvement,
            "retained_incumbent_train_rmse": float(best_result.best["rmse"]),
            "retained_incumbent_validation_rmse": best_selection_score,
            "lowest_train_rmse_so_far": lowest_train_rmse_so_far,
            "lowest_validation_rmse_so_far": lowest_validation_rmse_so_far,
        }

        if requested_action == "compare_candidate_family":
            tool_output.update(
                {
                    "observation": "candidate_family_comparison_completed",
                    "family_comparison": {
                        "proposed_formula": result.best["formula"],
                        "proposed_validation_rmse": result.best.get("selection_validation_rmse"),
                        "incumbent_formula": validation_incumbent.best["formula"] if validation_incumbent else None,
                        "incumbent_validation_rmse": validation_incumbent.best.get("selection_validation_rmse") if validation_incumbent else None,
                        "selected_formula": best_result.best["formula"],
                    },
                }
            )

        round_row = {
            "agent_name": agent_name,
            "case": metadata.get("case", case.case_name),
            "rock": metadata.get("rock", metadata.get("dataset", "")),
            "method": "llm_iterative_asrc" if use_llm else "asrc",
            "round": round_number,
            "agent_type": decision.get("agent_type", ""),
            "round_action": requested_action,
            "action_effective": effective_action,
            "plan_status": decision.get("plan_status", "none"),
            "planned_action": decision.get("planned_action", ""),
            "plan_revision_reason": decision.get("plan_revision_reason", ""),
            "direct_accept": bool(direct_accept),
            "variables": ",".join(variables),
            "search_max_terms": int(search_max_terms),
            "formula": best_result.best["formula"],
            "rmse": float(best_result.best["rmse"]),
            "proposal_formula": result.best["formula"] if performed_search else "",
            "proposal_rmse": current_rmse if performed_search else np.nan,
            "proposal_validation_rmse": proposal_selection_score if performed_search else np.nan,
            "proposal_complexity": int(result.best["complexity"]) if performed_search else np.nan,
            "proposal_physical_violations": (
                int(result.best["physical_violations"]) if performed_search else np.nan
            ),
            "incumbent_rmse_before": np.nan if previous_best_rmse is None else previous_best_rmse,
            "incumbent_validation_rmse_before": (
                np.nan if previous_best_selection_score is None else previous_best_selection_score
            ),
            "proposal_promoted": proposal_promoted,
            "proposal_relative_validation_improvement": (
                np.nan if proposal_relative_improvement is None else proposal_relative_improvement
            ),
            "best_rmse_so_far": float(best_result.best["rmse"]),
            "lowest_train_rmse_so_far": lowest_train_rmse_so_far,
            "lowest_validation_rmse_so_far": lowest_validation_rmse_so_far,
            "selection_validation_rmse": result.best.get("selection_validation_rmse", float("nan")),
            "best_selection_validation_rmse": best_selection_score,
            "performed_search": bool(performed_search),
            "validation_budget_used": validation_budget_used,
            "validation_budget_remaining": remaining_validation_budget(),
            "relative_improvement": np.nan if relative_improvement is None else relative_improvement,
            "mae": float(best_result.best["mae"]),
            "r2": float(best_result.best["r2"]),
            "complexity": int(best_result.best["complexity"]),
            "physical_violations": int(best_result.best["physical_violations"]),
            "accepted_formula": bool(best_result.best.get("accepted", False)),
            "executed_tool": tool_output.get("executed_tool", effective_action),
            "queried_knowledge_ids": ",".join(tool_output.get("queried_knowledge_ids", [])),
            "returned_terms": ",".join(tool_output.get("returned_terms", [])),
            "selected_feature_families": ",".join(tool_output.get("selected_feature_families", [])),
            "evidence_type": tool_output.get("evidence_type", ""),
            "evidence_passed": tool_output.get("evidence_passed", ""),
            "schema_repair_calls": schema_repair_calls,
            "acceptance_requested": bool(acceptance_report.get("requested", False)),
            "acceptance_accepted": bool(acceptance_report.get("accepted", False)),
            "acceptance_blockers": ";".join(acceptance_report.get("blockers", [])),
            "unresolved_issues": ";".join(acceptance_report.get("unresolved_issues", [])),
            "rule_pack_version": rules_version,
            "active_rule_ids": ",".join(active_rules),
            "knowledge_base_version": knowledge_selection["knowledge_base_version"],
            "active_knowledge_ids": ",".join(knowledge_selection["active_knowledge_ids"]),
            "knowledge_terms": ",".join(knowledge_selection["recommended_terms"]),
            "stop_reason": "",
            **metadata,
        }
        round_rows.append(round_row)
        formula_changed = str(best_result.best.get("formula", "")) != previous_best_formula
        if performed_search:
            progress_message(
                run_dir,
                f"Agent search round {round_number} finished for {agent_name}",
                verbose,
                proposal_train_rmse=f"{current_rmse:.6g}",
                proposal_validation_rmse=f"{proposal_selection_score:.6g}",
                incumbent_train_rmse_before=(
                    "NA" if previous_best_rmse is None else f"{previous_best_rmse:.6g}"
                ),
                incumbent_validation_rmse_before=(
                    "NA" if previous_best_selection_score is None else f"{previous_best_selection_score:.6g}"
                ),
                proposal_promoted=proposal_promoted,
                retained_incumbent_train_rmse=f"{float(best_result.best['rmse']):.6g}",
                retained_incumbent_validation_rmse=f"{best_selection_score:.6g}",
                lowest_train_rmse_so_far=f"{lowest_train_rmse_so_far:.6g}",
                lowest_validation_rmse_so_far=f"{lowest_validation_rmse_so_far:.6g}",
                formula_changed=formula_changed,
                complexity=best_result.best["complexity"],
                violations=best_result.best["physical_violations"],
                validation_budget_used=validation_budget_used,
            )
        else:
            progress_message(
                run_dir,
                f"Agent evidence round {round_number} finished for {agent_name}",
                verbose,
                action=requested_action,
                evidence=tool_output.get("evidence_type", "acceptance" if requested_action == "accept_current_best" else "state_review"),
                passed=tool_output.get("evidence_passed", acceptance_report.get("accepted", "recorded")),
                formula_unchanged=not formula_changed,
                validation_budget_used=validation_budget_used,
            )

        decision.update(
            {
                "round": round_number,
                "case": metadata.get("case", case.case_name),
                "rock": metadata.get("rock", metadata.get("dataset", "")),
                "action_effective": effective_action,
                "direct_accept": bool(direct_accept),
                "search_max_terms": int(search_max_terms),
                "search_variables": variables,
                "selected_formula": result.best["formula"],
                "selected_metrics": _best_metrics_payload(result.best),
                "retained_incumbent_formula": best_result.best["formula"],
                "retained_incumbent_metrics": _best_metrics_payload(best_result.best),
                "proposal_promoted": proposal_promoted,
                "proposal_relative_validation_improvement": proposal_relative_improvement,
                "selection_validation_rmse": result.best.get("selection_validation_rmse", float("nan")),
                "best_selection_validation_rmse": best_selection_score,
                "performed_search": bool(performed_search),
                "validation_budget": tool_output["validation_budget"],
                "relative_improvement": relative_improvement,
                "top_candidate_feedback": _top_candidate_feedback(result.candidates, top_k),
                "tool_output": tool_output,
                "acceptance_report": acceptance_report,
                "rule_pack_version": rules_version,
                "active_rule_ids": active_rules,
                "knowledge_base_version": knowledge_selection["knowledge_base_version"],
                "active_knowledge_ids": knowledge_selection["active_knowledge_ids"],
                "knowledge_recommended_terms": knowledge_selection["recommended_terms"],
                "feature_grammar_version": feature_grammar["version"],
                "feature_grammar_families": [family["id"] for family in feature_grammar_summary],
                **metadata,
            }
        )
        save_agent_decision(run_dir, decision, f"{agent_name}_round_{round_number:03d}")
        memory = update_agent_memory(
            memory,
            round_row,
            decision,
            variables,
            decision["top_candidate_feedback"],
            search_max_terms,
            best_result.best,
            direct_accept=direct_accept,
            tool_output=tool_output,
            acceptance_report=acceptance_report,
        )
        if audit_invalidated:
            memory.setdefault("case_memory", {}).setdefault("acceptance_memory", []).append(
                {
                    "round": int(round_number),
                    "requested": False,
                    "accepted": False,
                    "status": "invalidated_by_formula_change",
                    "formula": str(best_result.best.get("formula", "")),
                    "blockers": ["current_formula_not_finally_audited"],
                    "warnings": [],
                    "unresolved_issues": [],
                }
            )
        write_json(run_dir / "agent_memory" / f"{agent_name}_memory.json", memory)
        write_json(run_dir / "agent_tools" / f"{agent_name}_round_{round_number:03d}.json", tool_output)
        action_rows.append(
            {
                "agent_name": agent_name,
                "round": round_number,
                "action_requested": requested_action,
                "action_effective": effective_action,
                "executed_tool": tool_output.get("executed_tool", effective_action),
                "plan_status": decision.get("plan_status", "none"),
                "planned_action": decision.get("planned_action", ""),
                "plan_revision_reason": decision.get("plan_revision_reason", ""),
                "direct_accept": bool(direct_accept),
                "variables": ",".join(variables),
                "search_max_terms": int(search_max_terms),
                "rmse": float(best_result.best["rmse"]),
                "proposal_formula": result.best["formula"] if performed_search else "",
                "proposal_rmse": current_rmse if performed_search else np.nan,
                "proposal_validation_rmse": proposal_selection_score if performed_search else np.nan,
                "proposal_complexity": int(result.best["complexity"]) if performed_search else np.nan,
                "proposal_physical_violations": (
                    int(result.best["physical_violations"]) if performed_search else np.nan
                ),
                "incumbent_rmse_before": np.nan if previous_best_rmse is None else previous_best_rmse,
                "incumbent_validation_rmse_before": (
                    np.nan if previous_best_selection_score is None else previous_best_selection_score
                ),
                "proposal_promoted": proposal_promoted,
                "proposal_relative_validation_improvement": (
                    np.nan if proposal_relative_improvement is None else proposal_relative_improvement
                ),
                "best_rmse_so_far": float(best_result.best["rmse"]),
                "lowest_train_rmse_so_far": lowest_train_rmse_so_far,
                "lowest_validation_rmse_so_far": lowest_validation_rmse_so_far,
                "selection_validation_rmse": result.best.get("selection_validation_rmse", float("nan")),
                "best_selection_validation_rmse": best_selection_score,
                "performed_search": bool(performed_search),
                "validation_budget_used": validation_budget_used,
                "validation_budget_remaining": remaining_validation_budget(),
                "relative_improvement": np.nan if relative_improvement is None else relative_improvement,
                "complexity": int(best_result.best["complexity"]),
                "physical_violations": int(best_result.best["physical_violations"]),
                "revision_reason": decision.get("revision_reason", ""),
                "query_reason": tool_output.get("query_reason", ""),
                "queried_knowledge_ids": ",".join(tool_output.get("queried_knowledge_ids", [])),
                "returned_terms": ",".join(tool_output.get("returned_terms", [])),
                "returned_constraints": ",".join(tool_output.get("returned_constraints", [])),
                "selected_feature_families": ",".join(tool_output.get("selected_feature_families", [])),
                "evidence_type": tool_output.get("evidence_type", ""),
                "evidence_passed": tool_output.get("evidence_passed", ""),
                "schema_repair_calls": schema_repair_calls,
                "acceptance_requested": bool(acceptance_report.get("requested", False)),
                "acceptance_accepted": bool(acceptance_report.get("accepted", False)),
                "acceptance_blockers": ";".join(acceptance_report.get("blockers", [])),
                "acceptance_warnings": ";".join(acceptance_report.get("warnings", [])),
                "unresolved_issues": ";".join(acceptance_report.get("unresolved_issues", [])),
                "rule_pack_version": rules_version,
                "active_rule_ids": ",".join(active_rules),
                "knowledge_base_version": knowledge_selection["knowledge_base_version"],
                "active_knowledge_ids": ",".join(knowledge_selection["active_knowledge_ids"]),
                "knowledge_terms": ",".join(knowledge_selection["recommended_terms"]),
                **metadata,
            }
        )

        if performed_search and relative_improvement is not None and int(result.best["physical_violations"]) == 0:
            if relative_improvement < early_stop_relative_rmse:
                no_improve_count += 1
            else:
                no_improve_count = 0

        audit_trigger = ""
        if final_audit_enabled:
            if requested_action == "accept_current_best":
                audit_trigger = "agent_acceptance_request"
            elif round_number == soft_exploration_limit:
                audit_trigger = "soft_exploration_limit"
            elif round_number > soft_exploration_limit:
                audit_trigger = "recovery_follow_up"
            elif performed_search and no_improve_count >= early_stop_patience:
                audit_trigger = "rmse_plateau"
            elif round_number == max_rounds:
                audit_trigger = "hard_round_limit"

        if audit_trigger and best_result is not None:
            audit_attempts += 1
            progress_message(
                run_dir,
                f"Deterministic final audit started for {agent_name}",
                verbose,
                round=round_number,
                attempt=audit_attempts,
                trigger=audit_trigger,
            )
            audit_remaining = remaining_validation_budget()
            final_audit_report = run_deterministic_final_audit(
                case=case,
                result=best_result,
                memory=memory,
                round_number=round_number,
                attempt=audit_attempts,
                trigger=audit_trigger,
                baseline_rmse=baseline_rmse,
                allow_accept_before_round=allow_accept_before_round,
                requirements=acceptance_requirements,
                incumbent_payload=validation_incumbent.best if validation_incumbent is not None else None,
                allowed_terms=allowed,
                grammar_index=feature_grammar_summary,
                family_min_relative_improvement=float(
                    feature_grammar.get("screening", {}).get("min_relative_improvement", 0.01)
                ),
                validation_budget_remaining=audit_remaining,
                structural_audit_enabled=structural_audit_enabled,
                structural_reference_terms=(
                    list(validation_incumbent.best.get("terms", [])) if validation_incumbent is not None else []
                ),
                structural_mode=mode,
                structural_max_terms=current_max_terms,
                structural_alpha_grid=alpha_grid,
                structural_validation_top_k=structural_validation_top_k,
                structural_bootstrap_samples=structural_bootstrap_samples,
                structural_confidence=structural_confidence,
                seed=int(metadata.get("seed", 20260713)),
            )
            audit_validation_consumed = int(final_audit_report.get("validation_candidates_evaluated", 0))
            validation_budget_used += audit_validation_consumed
            acceptance_report = final_audit_report["acceptance_report"]
            result_status = str(final_audit_report["result_status"])
            audit_path = run_dir / "agent_tools" / f"{agent_name}_final_audit_attempt_{audit_attempts:03d}.json"
            write_json(audit_path, final_audit_report)
            final_audit_rows.append(
                {
                    "agent_name": agent_name,
                    "attempt": audit_attempts,
                    "round": round_number,
                    "trigger": audit_trigger,
                    "formula": best_result.best["formula"],
                    "rmse": float(best_result.best["rmse"]),
                    "selection_validation_rmse": _selection_score(best_result.best),
                    "complexity": int(best_result.best["complexity"]),
                    "physical_violations": int(best_result.best["physical_violations"]),
                    "accepted": bool(final_audit_report["accepted"]),
                    "predictive_status": final_audit_report.get("predictive_status", ""),
                    "structural_status": final_audit_report.get("structural_status", "not_applicable"),
                    "result_status": result_status,
                    "blockers": ";".join(final_audit_report.get("blockers", [])),
                    "warnings": ";".join(final_audit_report.get("warnings", [])),
                    "recommended_recovery": ";".join(
                        item["recommended_action"] for item in final_audit_report.get("recommended_recovery", [])
                    ),
                    "validation_candidates_evaluated": audit_validation_consumed,
                    "structural_validation_candidates_evaluated": int(
                        final_audit_report.get("structural_validation_candidates_evaluated", 0)
                    ),
                    **metadata,
                }
            )
            structural_report = final_audit_report.get("structural_report", {})
            structural_rows.extend(
                {
                    "agent_name": agent_name,
                    "round": round_number,
                    "audit_attempt": audit_attempts,
                    "source": "final_audit",
                    "structural_status": structural_report.get("structural_status", "not_applicable"),
                    "preferred_hypothesis_id": structural_report.get("preferred_hypothesis_id", ""),
                    "equivalent_hypothesis_ids": ";".join(structural_report.get("equivalent_hypothesis_ids", [])),
                    **hypothesis,
                    **metadata,
                }
                for hypothesis in structural_report.get("hypotheses", [])
            )
            round_row.update(
                {
                    "acceptance_requested": True,
                    "acceptance_accepted": bool(final_audit_report["accepted"]),
                    "acceptance_blockers": ";".join(final_audit_report.get("blockers", [])),
                    "unresolved_issues": ";".join(final_audit_report.get("unresolved_issues", [])),
                    "final_audit_triggered": True,
                    "final_audit_attempt": audit_attempts,
                    "final_audit_trigger": audit_trigger,
                    "result_status": result_status,
                    "predictive_status": final_audit_report.get("predictive_status", ""),
                    "structural_status": final_audit_report.get("structural_status", "not_applicable"),
                    "validation_budget_used": validation_budget_used,
                    "validation_budget_remaining": remaining_validation_budget(),
                }
            )
            action_rows[-1].update(
                {
                    "acceptance_requested": True,
                    "acceptance_accepted": bool(final_audit_report["accepted"]),
                    "acceptance_blockers": ";".join(final_audit_report.get("blockers", [])),
                    "acceptance_warnings": ";".join(final_audit_report.get("warnings", [])),
                    "unresolved_issues": ";".join(final_audit_report.get("unresolved_issues", [])),
                    "final_audit_triggered": True,
                    "final_audit_attempt": audit_attempts,
                    "final_audit_trigger": audit_trigger,
                    "result_status": result_status,
                    "predictive_status": final_audit_report.get("predictive_status", ""),
                    "structural_status": final_audit_report.get("structural_status", "not_applicable"),
                    "validation_budget_used": validation_budget_used,
                    "validation_budget_remaining": remaining_validation_budget(),
                }
            )
            decision["final_audit"] = final_audit_report
            tool_output["final_audit"] = final_audit_report
            save_agent_decision(run_dir, decision, f"{agent_name}_round_{round_number:03d}")
            write_json(run_dir / "agent_tools" / f"{agent_name}_round_{round_number:03d}.json", tool_output)
            write_json(run_dir / "agent_memory" / f"{agent_name}_memory.json", memory)
            progress_message(
                run_dir,
                f"Deterministic final audit finished for {agent_name}",
                verbose,
                accepted=bool(final_audit_report["accepted"]),
                structural_status=final_audit_report.get("structural_status", "not_applicable"),
                blockers=",".join(final_audit_report.get("blockers", [])) or "none",
                recovery_rounds_remaining=max(0, max_rounds - round_number),
            )
            current_audit_signature = (
                str(best_result.best.get("formula", "")),
                tuple(sorted(str(item) for item in final_audit_report.get("blockers", []))),
                str(final_audit_report.get("structural_status", "not_applicable")),
            )
            acceptance_repeated_without_budget = bool(
                requested_action == "accept_current_best"
                and not final_audit_report.get("accepted")
                and audit_validation_consumed == 0
                and remaining_exploration_validation_budget() == 0
            )
            if acceptance_repeated_without_budget:
                if current_audit_signature == rejected_audit_signature:
                    repeated_rejected_audits += 1
                else:
                    rejected_audit_signature = current_audit_signature
                    repeated_rejected_audits = 1
                audit_stagnated = repeated_rejected_audits >= 2
            elif current_audit_signature != rejected_audit_signature:
                rejected_audit_signature = current_audit_signature
                repeated_rejected_audits = 0
            simpler_candidate = _audit_simpler_candidate(final_audit_report)
            if simpler_candidate is not None:
                simplified_result = fit_fixed_terms(
                    case,
                    list(simpler_candidate["terms"]),
                    alpha=float(best_result.best.get("alpha", 0.0)),
                    mode=mode,
                    enforce_physical=enforce_physical,
                    candidate_id=f"audit_simplified_round_{round_number:03d}",
                )
                if int(simplified_result.best.get("physical_violations", 0)) == 0:
                    best_result = simplified_result
                    memory["best_formula"] = simplified_result.best["formula"]
                    memory["best_rmse"] = float(simplified_result.best["rmse"])
                    memory["best_complexity"] = int(simplified_result.best["complexity"])
                    memory["best_physical_violations"] = 0
                    final_audit_report["recovery_candidate_promoted"] = {
                        "source": "completed_complexity_challenge",
                        "removed_term": simpler_candidate.get("removed_term"),
                        "formula": simplified_result.best["formula"],
                        "selection_validation_rmse": _selection_score(simplified_result.best),
                        "complexity": int(simplified_result.best["complexity"]),
                    }
                    simplified_candidates = simplified_result.candidates.copy()
                    simplified_candidates["agent_name"] = agent_name
                    simplified_candidates["round"] = round_number
                    candidate_frames.append(simplified_candidates)
                    decision["top_candidate_feedback"] = _top_candidate_feedback(
                        simplified_candidates, top_k
                    )
                    decision["final_audit"] = final_audit_report
                    tool_output["final_audit"] = final_audit_report
                    write_json(audit_path, final_audit_report)
                    save_agent_decision(run_dir, decision, f"{agent_name}_round_{round_number:03d}")
                    write_json(
                        run_dir / "agent_tools" / f"{agent_name}_round_{round_number:03d}.json",
                        tool_output,
                    )
                    write_json(run_dir / "agent_memory" / f"{agent_name}_memory.json", memory)
                    progress_message(
                        run_dir,
                        f"Audit simplification promoted for {agent_name}",
                        verbose,
                        removed_term=simpler_candidate.get("removed_term", ""),
                        validation_rmse=f"{_selection_score(simplified_result.best):.6g}",
                        complexity=simplified_result.best["complexity"],
                    )
        else:
            round_row.setdefault("final_audit_triggered", False)
            round_row.setdefault("final_audit_attempt", 0)
            round_row.setdefault("final_audit_trigger", "")
            round_row.setdefault("result_status", result_status)
            round_row.setdefault("predictive_status", "not_audited")
            round_row.setdefault("structural_status", "not_audited")
            action_rows[-1].setdefault("final_audit_triggered", False)
            action_rows[-1].setdefault("final_audit_attempt", 0)
            action_rows[-1].setdefault("final_audit_trigger", "")
            action_rows[-1].setdefault("result_status", result_status)
            action_rows[-1].setdefault("predictive_status", "not_audited")
            action_rows[-1].setdefault("structural_status", "not_audited")

        previous_rounds.append(
            {
                key: round_row[key]
                for key in [
                    "round",
                    "formula",
                    "rmse",
                    "proposal_rmse",
                    "proposal_validation_rmse",
                    "incumbent_rmse_before",
                    "incumbent_validation_rmse_before",
                    "proposal_promoted",
                    "proposal_relative_validation_improvement",
                    "best_rmse_so_far",
                    "lowest_train_rmse_so_far",
                    "lowest_validation_rmse_so_far",
                    "selection_validation_rmse",
                    "best_selection_validation_rmse",
                    "complexity",
                    "physical_violations",
                    "round_action",
                    "action_effective",
                    "plan_status",
                    "planned_action",
                    "plan_revision_reason",
                    "executed_tool",
                    "acceptance_requested",
                    "acceptance_accepted",
                    "acceptance_blockers",
                    "unresolved_issues",
                    "final_audit_triggered",
                    "final_audit_attempt",
                    "final_audit_trigger",
                    "result_status",
                ]
            }
        )
        candidate_feedback = decision["top_candidate_feedback"]

        if not use_llm or decision.get("agent_type") == "deterministic_fallback":
            stop_reason = "deterministic_single_round"
            if decision.get("agent_type") == "deterministic_fallback":
                result_status = "completed_deterministic_fallback"
            progress_message(run_dir, f"Agent stopped for {agent_name}", verbose, reason=stop_reason)
            break
        if audit_stagnated:
            stop_reason = "repeated_acceptance_without_new_evidence"
            result_status = "provisional_unverified"
            progress_message(
                run_dir,
                f"Agent stopped after repeated unchanged acceptance requests for {agent_name}",
                verbose,
                reason=stop_reason,
                blockers=",".join((final_audit_report or {}).get("blockers", [])) or "none",
            )
            break
        if (
            final_audit_enabled
            and final_audit_report
            and final_audit_report.get("accepted")
            and final_audit_report.get("structural_status") in {"resolved", "equivalent", "not_applicable"}
        ):
            stop_reason = "deterministic_final_audit_accept"
            result_status = "accepted"
            progress_message(run_dir, f"Agent stopped for {agent_name}", verbose, reason=stop_reason)
            break
        if final_audit_enabled and round_number >= max_rounds:
            predictive_accepted = bool((final_audit_report or {}).get("accepted"))
            stop_reason = "predictive_accept_structural_unresolved" if predictive_accepted else "budget_exhausted_provisional"
            result_status = "accepted" if predictive_accepted else "provisional_unverified"
            progress_message(
                run_dir,
                (
                    f"Agent budget exhausted; retaining predictively accepted but structurally unresolved result for {agent_name}"
                    if predictive_accepted
                    else f"Agent budget exhausted; retaining provisional result for {agent_name}"
                ),
                verbose,
                blockers=",".join((final_audit_report or {}).get("blockers", [])) or "final_audit_not_completed",
            )
            break
        if (
            not final_audit_enabled
            and decision.get("round_action") == "accept_current_best"
            and acceptance_report.get("accepted")
        ):
            stop_reason = "verifier_accept_current_best"
            result_status = "accepted"
            progress_message(run_dir, f"Agent stopped for {agent_name}", verbose, reason=stop_reason)
            break
        if not final_audit_enabled and performed_search and relative_improvement is not None and int(result.best["physical_violations"]) == 0 and relative_improvement < early_stop_relative_rmse:
            no_improve_count += 1
        elif not final_audit_enabled and performed_search:
            no_improve_count = 0
        if not final_audit_enabled and performed_search and no_improve_count >= early_stop_patience:
            plateau_report = verify_acceptance(
                best_payload=best_result.best,
                baseline_rmse=baseline_rmse,
                round_number=round_number,
                allow_accept_before_round=allow_accept_before_round,
                requested=True,
                memory=memory,
                requirements=acceptance_requirements,
                incumbent_payload=validation_incumbent.best if validation_incumbent is not None else None,
            )
            if plateau_report.get("blockers"):
                progress_message(
                    run_dir,
                    f"RMSE plateau detected but evidence is incomplete for {agent_name}",
                    verbose,
                    round=round_number,
                    blockers=",".join(plateau_report["blockers"]),
                )
                continue
            stop_reason = "early_stop_rmse_plateau"
            result_status = "accepted"
            progress_message(run_dir, f"Agent stopped for {agent_name}", verbose, reason=stop_reason, patience=no_improve_count)
            break

    if best_result is None or best_decision is None:
        raise AgentSearchError(f"No agentic search result was generated for {agent_name}.")

    if use_llm and result_status == "provisional_pending_recovery":
        result_status = "provisional_unverified"

    round_rows[-1]["stop_reason"] = stop_reason
    round_rows[-1]["result_status"] = result_status
    _append_table_bundle(run_dir, run_dir / "metrics" / "llm_revision_rounds", round_rows, ["agent_name", "round"])
    if action_rows:
        action_rows[-1]["stop_reason"] = stop_reason
        action_rows[-1]["result_status"] = result_status
        _append_table_bundle(run_dir, run_dir / "metrics" / "agent_action_trace", action_rows, ["agent_name", "round"])
    if final_audit_rows:
        final_audit_rows[-1]["result_status"] = result_status
        _append_table_bundle(
            run_dir,
            run_dir / "metrics" / "agent_final_audit",
            final_audit_rows,
            ["agent_name", "attempt"],
        )
    if structural_rows:
        _append_table_bundle(
            run_dir,
            run_dir / "metrics" / "agent_family_comparison",
            structural_rows,
            ["agent_name", "round", "audit_attempt", "source", "hypothesis_id"],
        )
    if candidate_frontier_rows:
        _append_table_bundle(
            run_dir,
            run_dir / "metrics" / "agent_candidate_frontier",
            candidate_frontier_rows,
            ["agent_name", "round", "phase", "trace_type", "frontier_step"],
        )
    best_payload = {
        "agent_name": agent_name,
        "stop_reason": stop_reason,
        "rounds_completed": len(round_rows),
        "rule_pack_version": rules_version,
        "active_rule_ids": active_rules,
        "knowledge_base_version": knowledge_selection["knowledge_base_version"],
        "active_knowledge_ids": knowledge_selection["active_knowledge_ids"],
        "knowledge_recommended_terms": knowledge_selection["recommended_terms"],
        "feature_grammar_version": feature_grammar["version"],
        "feature_grammar_families": [family["id"] for family in feature_grammar_summary],
        "acceptance_contract": contract,
        "result_status": result_status,
        "predictive_status": (
            (final_audit_report or {}).get("predictive_status")
            or ("accepted" if result_status in {"accepted", "completed_deterministic"} else "provisional_unverified")
        ),
        "structural_status": (final_audit_report or {}).get("structural_status", "not_applicable"),
        "structural_summary": (final_audit_report or {}).get("structural_summary", ""),
        "structural_preferred_hypothesis": (
            (final_audit_report or {}).get("structural_report", {}).get("preferred_hypothesis_id", "")
        ),
        "structural_equivalent_hypotheses": (
            (final_audit_report or {}).get("structural_report", {}).get("equivalent_hypothesis_ids", [])
        ),
        "structural_coverage_complete": bool(
            (final_audit_report or {}).get("structural_report", {}).get("coverage_complete", False)
        ),
        "accepted_for_reporting": result_status in {"accepted", "completed_deterministic"},
        "provisional": result_status == "provisional_unverified",
        "final_acceptance_accepted": bool(
            (final_audit_report or {}).get("accepted", round_rows[-1].get("acceptance_accepted", False))
        ),
        "final_acceptance_report": (
            (final_audit_report or {}).get("acceptance_report", {})
            or {"accepted": bool(round_rows[-1].get("acceptance_accepted", False))}
        ),
        "final_audit_attempts": audit_attempts,
        "final_audit_blockers": (final_audit_report or {}).get("blockers", []),
        "soft_exploration_limit": soft_exploration_limit,
        "audit_recovery_rounds": audit_recovery_rounds,
        "validation_budget_total": validation_budget_total,
        "validation_budget_used": validation_budget_used,
        "validation_budget_reserve": validation_budget_reserve,
        "structural_validation_candidates_evaluated": int(
            (final_audit_report or {}).get("structural_report", {}).get("validation_candidates_evaluated", 0)
        ),
        **metadata,
        **best_result.best,
    }
    _append_best_formula(run_dir, best_payload)
    progress_message(
        run_dir,
        f"Agent search complete for {agent_name}",
        verbose,
        rounds=len(round_rows),
        best_rmse=f"{float(best_result.best['rmse']):.6g}",
        stop_reason=stop_reason,
        result_status=result_status,
    )
    all_candidates = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    return AgenticSearchResult(
        best_result,
        best_decision,
        round_rows,
        all_candidates,
        result_status=result_status,
        final_audit=final_audit_report,
    )

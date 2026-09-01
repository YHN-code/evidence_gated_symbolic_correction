from __future__ import annotations

from typing import Any

REQUIRED_DECISION_FIELDS = {
    "diagnosis",
    "recommended_variables",
    "forbidden_variables",
    "formula_skeletons",
    "constraints",
    "critic_notes",
    "next_action",
    "round_action",
    "revision_reason",
    "expected_improvement",
    "plan_status",
    "planned_action",
    "plan_revision_reason",
}

FORBIDDEN_TOKENS = ("tan_", "reciprocal_", "1/sin", "1/cos")
ALLOWED_ROUND_ACTIONS = {
    "revise_variables",
    "tighten_constraints",
    "broaden_search",
    "keep_best",
    "inspect_residual_pattern",
    "query_knowledge",
    "propose_feature_family",
    "remove_unstable_terms",
    "compare_candidate_family",
    "compare_alternative_families",
    "screen_candidate_families",
    "inspect_validation_failures",
    "test_term_stability",
    "challenge_formula_complexity",
    "stress_test_boundaries",
    "test_remaining_structure",
    "test_leave_one_angle",
    "accept_current_best",
    "stop_accept",
}
ALLOWED_PLAN_STATUSES = {"none", "new", "continue", "revise", "complete"}


class AgentDecisionError(ValueError):
    pass


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_as_text(item) for item in value]
    if isinstance(value, tuple):
        return [_as_text(item) for item in value]
    if isinstance(value, dict):
        return [f"{key}: {val}" for key, val in value.items()]
    if isinstance(value, str):
        if "\n" in value:
            parts = [part.strip(" -\t") for part in value.splitlines()]
            return [part for part in parts if part]
        return [value]
    return [_as_text(value)]


def validate_agent_decision(
    payload: dict[str, Any],
    allowed_variables: set[str],
    fallback_variables: list[str] | None = None,
    available_actions: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AgentDecisionError("LLM decision must be a JSON object.")
    for field in REQUIRED_DECISION_FIELDS:
        if field not in payload:
            raise AgentDecisionError(f"LLM decision missing required field: {field}")

    payload["recommended_variables"] = _as_string_list(payload["recommended_variables"])
    payload["forbidden_variables"] = _as_string_list(payload["forbidden_variables"])
    payload["formula_skeletons"] = _as_string_list(payload["formula_skeletons"])
    payload["constraints"] = _as_string_list(payload["constraints"])
    for field in ["diagnosis", "critic_notes", "next_action", "round_action", "revision_reason", "expected_improvement", "plan_status", "planned_action", "plan_revision_reason"]:
        payload[field] = _as_text(payload[field])
    payload["plan_status"] = payload["plan_status"].strip().lower()
    payload["planned_action"] = payload["planned_action"].strip()
    payload["next_action"] = payload["next_action"].strip()
    if "selected_action" in payload and not payload.get("round_action"):
        payload["round_action"] = _as_text(payload["selected_action"])
    payload["action_reason"] = _as_text(payload.get("action_reason", payload.get("revision_reason", "")))
    payload["expected_evidence"] = _as_text(payload.get("expected_evidence", payload.get("expected_improvement", "")))
    payload["knowledge_topics"] = _as_string_list(payload.get("knowledge_topics", []))
    payload["acceptance_evidence"] = _as_string_list(payload.get("acceptance_evidence", []))
    tool_inputs = payload.get("tool_inputs", {})
    payload["tool_inputs"] = tool_inputs if isinstance(tool_inputs, dict) else {"value": tool_inputs}

    clean_variables = []
    rejected_variables = []
    for variable in payload["recommended_variables"]:
        if not isinstance(variable, str):
            raise AgentDecisionError("recommended_variables must contain strings.")
        if variable not in allowed_variables:
            rejected_variables.append(variable)
            continue
        if any(token in variable for token in FORBIDDEN_TOKENS):
            rejected_variables.append(variable)
            continue
        clean_variables.append(variable)
    if not clean_variables:
        fallback = [variable for variable in (fallback_variables or []) if variable in allowed_variables]
        if not fallback:
            raise AgentDecisionError("LLM decision did not recommend any allowed variables.")
        clean_variables = fallback
        payload["variable_fallback_reason"] = (
            "LLM did not recommend allowed variables; deterministic bounded feature defaults were used."
        )
        payload["rejected_recommended_variables"] = rejected_variables
    if payload["round_action"] not in ALLOWED_ROUND_ACTIONS:
        raise AgentDecisionError(f"LLM decision round_action must be one of {sorted(ALLOWED_ROUND_ACTIONS)}.")
    if payload["round_action"] == "stop_accept":
        payload["round_action"] = "accept_current_best"
        payload["legacy_round_action"] = "stop_accept"
    if available_actions is not None and payload["round_action"] not in available_actions:
        raise AgentDecisionError(
            "LLM decision round_action is not available in the current round: "
            f"{payload['round_action']}."
        )
    if payload["plan_status"] not in ALLOWED_PLAN_STATUSES:
        raise AgentDecisionError(f"plan_status must be one of {sorted(ALLOWED_PLAN_STATUSES)}.")
    if payload["plan_status"] in {"new", "continue", "revise"}:
        if payload["planned_action"] not in ALLOWED_ROUND_ACTIONS:
            raise AgentDecisionError("planned_action must be a valid round action for an active plan.")
        if available_actions is not None and payload["planned_action"] not in available_actions:
            raise AgentDecisionError(
                "planned_action is not available in the current round."
            )
        if payload["planned_action"] in {"accept_current_best", "stop_accept", "keep_best"}:
            raise AgentDecisionError("An active plan must gather evidence or execute a search, not accept or keep the current result.")
        if payload["next_action"] != payload["planned_action"]:
            raise AgentDecisionError("next_action must exactly match planned_action for an active plan.")
        if payload["round_action"] == "accept_current_best":
            raise AgentDecisionError("Cannot accept the current result while declaring an active experiment plan.")
        if payload["plan_status"] == "continue" and payload["round_action"] != payload["planned_action"]:
            raise AgentDecisionError("plan_status=continue must execute the planned_action in the current round.")
        if payload["plan_status"] == "revise" and not payload["plan_revision_reason"].strip():
            raise AgentDecisionError("plan_status=revise requires plan_revision_reason.")
    else:
        redundant_actions = {value for value in [payload["planned_action"], payload["next_action"]] if value}
        if redundant_actions and redundant_actions != {payload["round_action"]}:
            raise AgentDecisionError(
                "For plan_status none or complete, planned_action/next_action must be empty or repeat round_action."
            )
        if payload["plan_status"] == "complete" and not payload["plan_revision_reason"].strip():
            raise AgentDecisionError("plan_status=complete requires plan_revision_reason.")
        payload["planned_action"] = ""
        payload["next_action"] = ""

    # Keep legacy fields in persisted logs while the public schema moves to an
    # explicit, revisable plan state.
    payload["follow_up_required"] = payload["plan_status"] in {"new", "revise"}
    payload["follow_up_action"] = payload["planned_action"]

    payload["recommended_variables"] = clean_variables
    payload["forbidden_variables"] = [str(item) for item in payload["forbidden_variables"]]
    payload["formula_skeletons"] = [str(item) for item in payload["formula_skeletons"]]
    payload["constraints"] = [str(item) for item in payload["constraints"]]
    return payload

from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are an ASRC research agent for anisotropic rock strength residual correction.
Return only valid JSON. Do not include Markdown.
Your role is to choose auditable tool actions, diagnose residual patterns, query rock-mechanics knowledge when needed,
propose bounded residual variables, define formula skeletons, and critique candidates.
The deterministic backend will perform parameter fitting, metrics, and physical checks."""


REQUIRED_FIELDS_TEXT = (
    "Return a single JSON object with exactly these fields: diagnosis, recommended_variables, "
    "forbidden_variables, formula_skeletons, constraints, critic_notes, next_action, "
    "round_action, revision_reason, expected_improvement, plan_status, planned_action, plan_revision_reason. The fields recommended_variables, "
    "forbidden_variables, formula_skeletons, and constraints must be JSON arrays of strings. "
    "All other required fields must be strings. Optional fields are action_reason, expected_evidence, "
    "knowledge_topics, tool_inputs, and acceptance_evidence. round_action must be copied from available_actions. "
    "recommended_variables must contain at least one exact string copied from allowed_variables, even when requesting acceptance."
)

ACTION_TEXT = (
    "Action semantics: inspect_residual_pattern returns the current best formula's post-correction residual evidence without fitting formulas; query_knowledge retrieves "
    "audited knowledge blocks by topic without fitting formulas; revise_variables searches the recommended variables; propose_feature_family "
    "uses exact family ids from feature_grammar_index via tool_inputs.feature_families; compare_candidate_family also requires exact family ids and searches only the current incumbent terms plus those selected family terms; broaden_search expands the variable set and term budget; "
    "remove_unstable_terms drops terms supported by stability or complexity evidence; optionally pass exact current-formula names in tool_inputs.remove_terms. tighten_constraints favors compact admissible formulas; "
    "compare_candidate_family runs a budgeted search for selected family ids; compare_alternative_families formally compares every registered family hypothesis with equal group-validation budgets and paired uncertainty; screen_candidate_families only provides a cheap diagnostic ranking. If screening evidence is not passed, use a formal family comparison before acceptance; "
    "inspect_validation_failures returns per-angle held-out errors; test_term_stability bootstraps the current formula coefficients; "
    "challenge_formula_complexity tests whether dropping one term is validation-noninferior; stress_test_boundaries evaluates the current formula on a dense angle grid; "
    "test_remaining_structure combines a max-correlation permutation test with a fixed-term group-validation check, so statistically detectable but non-predictive structure is recorded without forcing expansion; test_leave_one_angle is a legacy alias of inspect_validation_failures; "
    "keep_best preserves the historical best without fitting; accept_current_best requests verifier acceptance. "
    "Only revise_variables, propose_feature_family, broaden_search, remove_unstable_terms, tighten_constraints, and compare_candidate_family consume search budget."
)

VALIDATION_TEXT = (
    "Read current_best exactly. If current_best.validation_performed is true, do not claim that leave-one-angle validation is missing. "
    "When requesting acceptance, cite the reported training RMSE, group-validation RMSE, complexity, and physical violations. "
    "Do not call a result maximal unless the available validation budget has been exhausted or competing families were tested."
    " Manage the experiment plan explicitly: plan_status must be none, new, continue, revise, or complete."
    " For new/revise, copy one exact available action into both planned_action and next_action. For continue, execute that exact action now."
    " For revise/complete, explain the change in plan_revision_reason. For none/complete, planned_action and next_action must be empty."
    " A pending plan blocks acceptance until it is executed or explicitly completed with a recorded reason."
    " Acceptance requires at least one agent-requested symbolic search and a fresh post-correction residual inspection of the final current-best formula."
    " Do not repeat inspect_residual_pattern when memory already contains post-correction diagnostics for the unchanged current-best formula."
    " If post-correction correlations or worst-angle cells indicate a missing bounded feature family, query or test that family before accepting."
    " Dynamic terms listed in feature_grammar_index are activated only through a family action; do not treat them as ordinary allowed_variables."
    " Read acceptance_contract.requirements and gather missing evidence for the current formula in any scientifically justified order. Evidence for an older formula becomes stale after the current best changes."
    " If complexity evidence finds a simpler noninferior formula, use remove_unstable_terms; if stability fails, remove only the reported unstable terms. If boundary or remaining-structure evidence fails, revise the formula or search hypothesis before requesting acceptance."
)


def _rule_block(rule_summary: list[dict[str, str]] | None) -> str:
    rules = rule_summary or []
    return (
        "literature_backed_rules: "
        f"{json.dumps(rules, ensure_ascii=False)}\n"
        "Use only these audited rules as domain background. Do not invent additional rock-mechanics laws.\n"
    )


def _knowledge_block(knowledge_index: list[dict[str, Any]] | None) -> str:
    knowledge = knowledge_index or []
    return (
        "knowledge_index: "
        f"{json.dumps(knowledge, ensure_ascii=False)}\n"
        "The index is not the full knowledge content. Use round_action=query_knowledge with knowledge_topics when more details are needed.\n"
    )


def build_decision_prompt(
    summary: dict[str, Any],
    allowed_variables: list[str],
    rule_summary: list[dict[str, str]] | None = None,
    knowledge_summary: list[dict[str, Any]] | None = None,
    agent_context: dict[str, Any] | None = None,
) -> str:
    return (
        "Create round 1 of an ASRC agent decision for this case.\n"
        "Use only variables from allowed_variables. Do not recommend tan or reciprocal trigonometric terms.\n"
        "Do not invent aliases, mathematical notation, or renamed variables; copy exact variable names.\n"
        f"{_rule_block(rule_summary)}"
        f"{_knowledge_block(knowledge_summary)}"
        f"{REQUIRED_FIELDS_TEXT}\n"
        f"{ACTION_TEXT}\n"
        f"{VALIDATION_TEXT}\n"
        "Choose the next useful tool action from available_actions; do not use a fixed workflow.\n\n"
        f"agent_context: {json.dumps(agent_context or {}, ensure_ascii=False)}\n"
        f"allowed_variables: {allowed_variables}\n"
        f"case_summary: {summary}\n"
    )


def build_revision_prompt(
    summary: dict[str, Any],
    allowed_variables: list[str],
    previous_rounds: list[dict[str, Any]],
    candidate_feedback: list[dict[str, Any]],
    optimization_memory: dict[str, Any] | None = None,
    rule_summary: list[dict[str, str]] | None = None,
    knowledge_summary: list[dict[str, Any]] | None = None,
    agent_context: dict[str, Any] | None = None,
) -> str:
    return (
        "Revise the ASRC agent decision for the next round.\n"
        "Use only variables from allowed_variables. Do not recommend tan or reciprocal trigonometric terms.\n"
        "Do not invent aliases, mathematical notation, or renamed variables; copy exact variable names.\n"
        "Review the previous round metrics and top candidate feedback. Prefer bounded variables, low complexity, "
        "zero physical violations, and improved RMSE. Use residual diagnostics and failed-variable memory "
        "to avoid repeating ineffective searches.\n"
        f"{_rule_block(rule_summary)}"
        f"{_knowledge_block(knowledge_summary)}"
        f"{REQUIRED_FIELDS_TEXT}\n"
        f"{ACTION_TEXT}\n"
        f"{VALIDATION_TEXT}\n"
        "Use accept_current_best only when the verifier evidence is likely to satisfy the acceptance contract.\n\n"
        f"agent_context: {json.dumps(agent_context or {}, ensure_ascii=False)}\n"
        f"allowed_variables: {allowed_variables}\n"
        f"case_summary: {summary}\n"
        f"optimization_memory: {optimization_memory or {}}\n"
        f"previous_rounds: {previous_rounds}\n"
        f"top_candidate_feedback: {candidate_feedback}\n"
    )


def build_schema_repair_prompt(
    invalid_decision: Any,
    validation_error: str,
    allowed_variables: list[str],
    available_actions: list[str],
    original_task_prompt: str | None = None,
) -> str:
    return (
        "Your previous JSON decision failed the agent schema. Repair the decision yourself and return only one corrected JSON object.\n"
        "Preserve the scientific action and rationale unless the validation error explicitly requires changing them. Do not add Markdown.\n"
        f"validation_error: {validation_error}\n"
        f"available_actions: {json.dumps(available_actions, ensure_ascii=False)}\n"
        f"allowed_variables: {json.dumps(allowed_variables, ensure_ascii=False)}\n"
        f"{REQUIRED_FIELDS_TEXT}\n"
        f"{VALIDATION_TEXT}\n"
        f"invalid_decision: {json.dumps(invalid_decision, ensure_ascii=False)}\n"
        + (f"original_task: {original_task_prompt}\n" if original_task_prompt else "")
    )

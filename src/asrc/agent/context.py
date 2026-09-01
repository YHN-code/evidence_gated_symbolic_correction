from __future__ import annotations

from pathlib import Path
from typing import Any

from asrc.utils.io import write_json


DEFAULT_AGENT_ACTIONS = [
    "inspect_residual_pattern",
    "query_knowledge",
    "revise_variables",
    "propose_feature_family",
    "broaden_search",
    "remove_unstable_terms",
    "tighten_constraints",
    "compare_candidate_family",
    "compare_alternative_families",
    "screen_candidate_families",
    "inspect_validation_failures",
    "test_term_stability",
    "challenge_formula_complexity",
    "stress_test_boundaries",
    "test_remaining_structure",
    "test_leave_one_angle",
    "keep_best",
    "accept_current_best",
]


def build_agent_context(
    *,
    case_summary: dict[str, Any],
    allowed_variables: list[str],
    rule_summary: list[dict[str, Any]],
    knowledge_index: list[dict[str, Any]],
    feature_grammar_index: list[dict[str, Any]] | None,
    memory_view: dict[str, Any],
    previous_rounds: list[dict[str, Any]],
    candidate_feedback: list[dict[str, Any]],
    available_actions: list[str] | None = None,
    acceptance_contract: dict[str, Any] | None = None,
    validation_budget: dict[str, Any] | None = None,
    current_best: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "case_summary": case_summary,
        "allowed_variables": allowed_variables,
        "available_actions": available_actions or DEFAULT_AGENT_ACTIONS,
        "literature_backed_rules": rule_summary,
        "knowledge_index": knowledge_index,
        "feature_grammar_index": feature_grammar_index or [],
        "memory_view": memory_view,
        "previous_rounds": previous_rounds,
        "top_candidate_feedback": candidate_feedback,
        "acceptance_contract": acceptance_contract or {},
        "validation_budget": validation_budget or {},
        "current_best": current_best or {},
        "acceptance_available": "accept_current_best" in (available_actions or DEFAULT_AGENT_ACTIONS),
    }


def save_agent_context(run_dir: Path, agent_round_name: str, context: dict[str, Any]) -> Path:
    return write_json(run_dir / "agent_context" / f"{agent_round_name}.json", context)

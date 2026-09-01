from __future__ import annotations

from pathlib import Path
from typing import Any

from asrc.features.angle_features import make_angle_features
from asrc.utils.io import read_yaml


DEFAULT_KNOWLEDGE_CONFIG = "configs/rock_mechanics_knowledge.yaml"
_REQUIRED_FIELDS = {"id", "category", "uses", "purpose", "evidence"}


class KnowledgeBaseError(ValueError):
    pass


def _uses(item: dict[str, Any], use: str) -> bool:
    uses = item.get("uses") or []
    if not isinstance(uses, list):
        raise KnowledgeBaseError(f"Knowledge block {item.get('id', '<missing>')} uses must be a list.")
    return use in {str(entry) for entry in uses}


def load_knowledge_base(config_path: str | Path | None = None) -> dict[str, Any]:
    payload = read_yaml(config_path or DEFAULT_KNOWLEDGE_CONFIG)
    if not isinstance(payload, dict):
        raise KnowledgeBaseError("Knowledge base must be a YAML mapping.")
    if not payload.get("version"):
        raise KnowledgeBaseError("Knowledge base missing version.")
    blocks = payload.get("knowledge_blocks")
    if not isinstance(blocks, list):
        raise KnowledgeBaseError("Knowledge base must contain knowledge_blocks.")
    for block in blocks:
        if not isinstance(block, dict):
            raise KnowledgeBaseError("Each knowledge block must be a mapping.")
        missing = sorted(field for field in _REQUIRED_FIELDS if not block.get(field))
        if missing:
            raise KnowledgeBaseError(f"Knowledge block {block.get('id', '<missing>')} missing fields: {missing}.")
        evidence = block.get("evidence") or {}
        if not evidence.get("sources") or not evidence.get("locator") or not evidence.get("evidence_level"):
            raise KnowledgeBaseError(f"Knowledge block {block.get('id')} missing evidence source, locator, or level.")
    return payload


def knowledge_base_version(knowledge_base: dict[str, Any]) -> str:
    return str(knowledge_base.get("version", "unknown"))


def knowledge_index(knowledge_base: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    kb = knowledge_base or load_knowledge_base()
    rows = []
    for block in kb.get("knowledge_blocks", []):
        if not _uses(block, "llm_prompt"):
            continue
        evidence = block.get("evidence") or {}
        rows.append(
            {
                "id": str(block["id"]),
                "category": str(block["category"]),
                "purpose": str(block["purpose"]),
                "feature_families": [str(item) for item in block.get("feature_families", []) or []],
                "constraints": [str(item) for item in block.get("constraints", []) or []],
                "evidence_level": str(evidence.get("evidence_level", "")),
            }
        )
    return rows


def _block_applies(block: dict[str, Any], case: Any, metadata: dict[str, Any] | None = None) -> bool:
    applies = block.get("applies_to") or {}
    data_types = {str(item) for item in applies.get("data_types", [])}
    if data_types and str(case.data_type) not in data_types:
        return False
    required_columns = set(applies.get("required_columns", []))
    if required_columns and not required_columns.issubset(set(case.frame.columns)):
        return False
    required_case_fields = set(applies.get("required_case_fields", []))
    if required_case_fields and not all(getattr(case, field, None) for field in required_case_fields):
        return False
    max_rows = applies.get("max_rows")
    if max_rows is not None and len(case.frame) > int(max_rows):
        return False
    return True


def available_feature_terms(case: Any) -> list[str]:
    features = make_angle_features(case.frame, include_forbidden=False)
    return [name for name in features.columns if name not in {"beta_deg", "psi_deg"}]


def _prioritized_terms(case: Any, terms: list[str]) -> list[str]:
    if "psi_deg" in case.frame.columns:
        priority = [
            "m_beta",
            "m2_beta",
            "sin_beta",
            "cos_beta",
            "m_psi",
            "m2_psi",
            "sin_psi",
            "cos_psi",
            "sin2_beta",
            "cos2_beta",
            "sin2_psi",
            "cos2_psi",
            "m_beta_m_psi",
            "sin2_beta_sin2_psi",
        ]
        cap = 13
    else:
        priority = [
            "m_beta",
            "m2_beta",
            "m4_beta",
            "sat_m2_beta_b3",
            "sin_beta",
            "cos_beta",
            "sin2_beta",
            "cos2_beta",
            "sin2beta",
            "cos2beta",
        ]
        cap = 10
    ordered = [term for term in priority if term in terms]
    ordered.extend(term for term in terms if term not in ordered)
    return ordered[:cap]


def select_knowledge_blocks(
    case: Any,
    metadata: dict[str, Any] | None = None,
    knowledge_base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kb = knowledge_base or load_knowledge_base()
    feature_terms = set(available_feature_terms(case))
    selected = [
        block
        for block in kb.get("knowledge_blocks", [])
        if _uses(block, "executable_search") and _block_applies(block, case, metadata)
    ]
    terms: list[str] = []
    constraints: list[str] = []
    actions: list[str] = []
    for block in selected:
        for term in block.get("recommended_terms", []) or []:
            if term in feature_terms and term not in terms:
                terms.append(term)
        for constraint in block.get("constraints", []) or []:
            if constraint not in constraints:
                constraints.append(str(constraint))
        for action in block.get("search_actions", []) or []:
            if action not in actions:
                actions.append(str(action))
    terms = _prioritized_terms(case, terms)
    return {
        "knowledge_base_version": knowledge_base_version(kb),
        "active_knowledge_ids": [str(block["id"]) for block in selected],
        "recommended_terms": terms,
        "constraints": constraints,
        "search_actions": actions,
        "blocks": selected,
    }


def _topic_score(block: dict[str, Any], topics: list[str]) -> int:
    if not topics:
        return 1
    text = " ".join(
        [
            str(block.get("id", "")),
            str(block.get("category", "")),
            str(block.get("purpose", "")),
            " ".join(str(item) for item in block.get("feature_families", []) or []),
            " ".join(str(item) for item in block.get("constraints", []) or []),
        ]
    ).lower()
    aliases = {
        "anisotropy": ["anisotropic", "anisotropy", "orientation", "surface"],
        "weak_plane": ["weak", "plane", "mobilization", "foliation"],
        "brazilian_test": ["brazilian", "tensile", "strength"],
        "angle_features": ["angle", "orientation", "beta", "psi", "feature"],
        "small_sample_validation": ["small", "validation", "leave", "complexity"],
        "residual_correction": ["residual", "baseline", "additive", "correction"],
    }
    score = 0
    for topic in topics:
        words = aliases.get(topic.lower(), [topic.lower()])
        score += sum(1 for word in words if word in text)
    return score


def query_knowledge_blocks(
    case: Any,
    topics: list[str] | None = None,
    top_k: int = 3,
    knowledge_base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kb = knowledge_base or load_knowledge_base()
    feature_terms = set(available_feature_terms(case))
    topics = [str(topic) for topic in (topics or []) if str(topic).strip()]
    candidates = [
        block
        for block in kb.get("knowledge_blocks", [])
        if _uses(block, "llm_prompt") and _block_applies(block, case)
    ]
    ranked = sorted(candidates, key=lambda block: (_topic_score(block, topics), str(block.get("id", ""))), reverse=True)
    selected = [block for block in ranked if _topic_score(block, topics) > 0][: max(1, int(top_k))]
    terms: list[str] = []
    constraints: list[str] = []
    for block in selected:
        for term in block.get("recommended_terms", []) or []:
            if term in feature_terms and term not in terms:
                terms.append(term)
        for constraint in block.get("constraints", []) or []:
            if constraint not in constraints:
                constraints.append(str(constraint))
    terms = _prioritized_terms(case, terms)
    return {
        "knowledge_base_version": knowledge_base_version(kb),
        "query_topics": topics,
        "queried_knowledge_ids": [str(block["id"]) for block in selected],
        "returned_terms": terms,
        "returned_constraints": constraints,
        "blocks": selected,
    }


def prompt_knowledge_summary(selection: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = []
    for block in selection.get("blocks", []):
        if not _uses(block, "llm_prompt"):
            continue
        evidence = block.get("evidence") or {}
        blocks.append(
            {
                "id": str(block["id"]),
                "category": str(block["category"]),
                "purpose": str(block["purpose"]),
                "recommended_terms": [term for term in block.get("recommended_terms", []) if term in selection.get("recommended_terms", [])],
                "constraints": [str(item) for item in block.get("constraints", []) or []],
                "evidence_level": str(evidence.get("evidence_level", "")),
            }
        )
    return blocks

from __future__ import annotations

from pathlib import Path
from typing import Any

from asrc.utils.io import read_yaml


DEFAULT_RULE_CONFIG = "configs/rock_mechanics_rules.yaml"
_REQUIRED_PROMPT_FIELDS = {"id", "category", "uses", "statement", "source", "locator", "evidence_level"}


class RulePackError(ValueError):
    pass


def _uses(item: dict[str, Any], use: str) -> bool:
    uses = item.get("uses") or []
    if not isinstance(uses, list):
        raise RulePackError(f"Rule {item.get('id', '<missing>')} uses must be a list.")
    return use in {str(entry) for entry in uses}


def load_rule_pack(config_path: str | Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_RULE_CONFIG
    payload = read_yaml(path)
    if not isinstance(payload, dict):
        raise RulePackError("Rule pack must be a YAML mapping.")
    if not payload.get("version"):
        raise RulePackError("Rule pack missing version.")
    rules = payload.get("rules")
    if not isinstance(rules, list):
        raise RulePackError("Rule pack must contain a rules list.")
    for rule in rules:
        if not isinstance(rule, dict):
            raise RulePackError("Each rule must be a mapping.")
        if "applies_to" in rule:
            raise RulePackError(f"Rule {rule.get('id', '<missing>')} must not use case-routing applies_to.")
        if _uses(rule, "llm_prompt"):
            missing = sorted(field for field in _REQUIRED_PROMPT_FIELDS if not rule.get(field))
            if missing:
                raise RulePackError(f"Prompt-allowed rule {rule.get('id', '<missing>')} missing fields: {missing}.")
            if not rule.get("evidence_level"):
                raise RulePackError(f"Prompt-allowed rule {rule.get('id')} missing evidence level.")
    return payload


def rule_pack_version(rule_pack: dict[str, Any]) -> str:
    return str(rule_pack.get("version", "unknown"))


def prompt_rule_summary(rule_pack: dict[str, Any]) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    for rule in rule_pack.get("rules", []):
        if not _uses(rule, "llm_prompt"):
            continue
        summary.append(
            {
                "id": str(rule["id"]),
                "category": str(rule["category"]),
                "rule": str(rule.get("prompt_text") or rule["statement"]),
                "evidence_level": str(rule["evidence_level"]),
            }
        )
    return summary


def active_rule_ids(rule_summary: list[dict[str, str]]) -> list[str]:
    return [rule["id"] for rule in rule_summary]

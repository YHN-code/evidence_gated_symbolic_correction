from __future__ import annotations

from pathlib import Path
from typing import Any

from asrc.features.angle_features import make_angle_features
from asrc.utils.io import read_yaml


DEFAULT_GRAMMAR_CONFIG = "configs/bounded_feature_grammar.yaml"


class BoundedGrammarError(ValueError):
    pass


def load_bounded_grammar(config_path: str | Path | None = None) -> dict[str, Any]:
    payload = read_yaml(config_path or DEFAULT_GRAMMAR_CONFIG)
    if not isinstance(payload, dict) or not payload.get("version"):
        raise BoundedGrammarError("Bounded feature grammar must be a versioned mapping.")
    families = payload.get("families")
    if not isinstance(families, list) or not families:
        raise BoundedGrammarError("Bounded feature grammar must define at least one family.")
    seen: set[str] = set()
    for family in families:
        if not isinstance(family, dict):
            raise BoundedGrammarError("Each bounded feature family must be a mapping.")
        family_id = str(family.get("id", "")).strip()
        if not family_id or family_id in seen:
            raise BoundedGrammarError(f"Invalid or duplicate bounded feature family id: {family_id!r}.")
        seen.add(family_id)
        if not family.get("purpose") or not family.get("terms"):
            raise BoundedGrammarError(f"Bounded feature family {family_id} needs purpose and terms.")
    return payload


def bounded_grammar_index(case: Any, grammar: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = grammar or load_bounded_grammar()
    available = set(make_angle_features(case.frame, include_forbidden=False).columns)
    columns = set(case.frame.columns)
    rows = []
    for family in payload["families"]:
        required = {str(item) for item in family.get("required_columns", [])}
        if not required.issubset(columns):
            continue
        terms = [str(term) for term in family.get("terms", []) if str(term) in available]
        if terms:
            rows.append(
                {
                    "id": str(family["id"]),
                    "purpose": str(family["purpose"]),
                    "terms": terms,
                }
            )
    return rows


def bounded_grammar_terms(case: Any, grammar: dict[str, Any] | None = None) -> list[str]:
    return list(dict.fromkeys(term for family in bounded_grammar_index(case, grammar) for term in family["terms"]))


def resolve_bounded_families(
    case: Any,
    family_ids: list[str],
    grammar: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = grammar or load_bounded_grammar()
    index = {family["id"]: family for family in bounded_grammar_index(case, payload)}
    selected = [index[family_id] for family_id in family_ids if family_id in index]
    return {
        "grammar_version": str(payload["version"]),
        "selected_family_ids": [family["id"] for family in selected],
        "terms": list(dict.fromkeys(term for family in selected for term in family["terms"])),
    }

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


FORBIDDEN_ANSWER_FIELDS = frozenset(
    {
        "applies_to",
        "candidate",
        "candidates",
        "equation",
        "expression",
        "feature_families",
        "formula",
        "operators",
        "parameters",
        "recommended_terms",
        "required_columns",
        "search_actions",
        "target_formula",
        "task_id",
    }
)


class SemanticKnowledgeError(ValueError):
    """Raised when a semantic attribution knowledge file can leak answers."""


@dataclass(frozen=True)
class SemanticKnowledgeBlock:
    block_id: str
    topics: tuple[str, ...]
    statement: str
    scope: str
    evidence_sources: tuple[str, ...]
    evidence_locator: str
    evidence_level: str

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "id": self.block_id,
            "topics": list(self.topics),
            "statement": self.statement,
            "scope": self.scope,
            "evidence_level": self.evidence_level,
        }


@dataclass(frozen=True)
class SemanticKnowledgeBase:
    version: str
    description: str
    blocks: tuple[SemanticKnowledgeBlock, ...]
    sha256: str


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _find_forbidden_field(value: Any, path: str = "root") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_ANSWER_FIELDS:
                return f"{path}.{key}"
            found = _find_forbidden_field(child, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _find_forbidden_field(child, f"{path}[{index}]")
            if found:
                return found
    return None


def _require_nonempty_text(value: Any, name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise SemanticKnowledgeError(f"{name} must be non-empty.")
    return text


def load_semantic_knowledge(path: str | Path) -> SemanticKnowledgeBase:
    source_path = Path(path)
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise SemanticKnowledgeError("Semantic knowledge root must be a mapping.")
    forbidden = _find_forbidden_field(payload)
    if forbidden:
        raise SemanticKnowledgeError(
            f"Answer-bearing field is forbidden in semantic knowledge: {forbidden}."
        )
    allowed_root = {"version", "description", "knowledge_blocks"}
    unexpected_root = set(payload) - allowed_root
    if unexpected_root:
        raise SemanticKnowledgeError(
            f"Unexpected semantic knowledge root fields: {sorted(unexpected_root)}."
        )
    raw_blocks = payload.get("knowledge_blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise SemanticKnowledgeError("knowledge_blocks must be a non-empty list.")

    blocks: list[SemanticKnowledgeBlock] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_blocks):
        if not isinstance(raw, Mapping):
            raise SemanticKnowledgeError(
                f"knowledge_blocks[{index}] must be a mapping."
            )
        required = {"id", "topics", "statement", "scope", "evidence"}
        if set(raw) != required:
            raise SemanticKnowledgeError(
                f"knowledge_blocks[{index}] requires exactly {sorted(required)}."
            )
        block_id = _require_nonempty_text(raw["id"], f"block[{index}].id")
        if block_id in seen_ids:
            raise SemanticKnowledgeError(f"Duplicate knowledge id {block_id!r}.")
        seen_ids.add(block_id)
        raw_topics = raw["topics"]
        if not isinstance(raw_topics, list) or not raw_topics:
            raise SemanticKnowledgeError(f"{block_id}.topics must be non-empty.")
        topics = tuple(
            dict.fromkeys(
                _require_nonempty_text(item, f"{block_id}.topics")
                for item in raw_topics
            )
        )
        evidence = raw["evidence"]
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "sources",
            "locator",
            "evidence_level",
        }:
            raise SemanticKnowledgeError(
                f"{block_id}.evidence requires sources, locator, and evidence_level."
            )
        raw_sources = evidence["sources"]
        if not isinstance(raw_sources, list) or not raw_sources:
            raise SemanticKnowledgeError(
                f"{block_id}.evidence.sources must be non-empty."
            )
        blocks.append(
            SemanticKnowledgeBlock(
                block_id=block_id,
                topics=topics,
                statement=_require_nonempty_text(
                    raw["statement"], f"{block_id}.statement"
                ),
                scope=_require_nonempty_text(raw["scope"], f"{block_id}.scope"),
                evidence_sources=tuple(
                    _require_nonempty_text(item, f"{block_id}.evidence.sources")
                    for item in raw_sources
                ),
                evidence_locator=_require_nonempty_text(
                    evidence["locator"], f"{block_id}.evidence.locator"
                ),
                evidence_level=_require_nonempty_text(
                    evidence["evidence_level"],
                    f"{block_id}.evidence.evidence_level",
                ),
            )
        )
    return SemanticKnowledgeBase(
        version=_require_nonempty_text(payload.get("version"), "version"),
        description=_require_nonempty_text(
            payload.get("description"), "description"
        ),
        blocks=tuple(blocks),
        sha256=_canonical_sha256(payload),
    )


def retrieve_semantic_knowledge(
    knowledge: SemanticKnowledgeBase,
    topics: Iterable[str],
    *,
    top_k: int,
) -> tuple[SemanticKnowledgeBlock, ...]:
    query = {str(item).strip().lower() for item in topics if str(item).strip()}
    if not query:
        return ()
    scored = []
    for block in knowledge.blocks:
        block_topics = {item.lower() for item in block.topics}
        overlap = len(query & block_topics)
        if overlap:
            scored.append((-overlap, block.block_id, block))
    scored.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in scored[: max(1, int(top_k))])


def knowledge_prompt_payload(
    knowledge: SemanticKnowledgeBase,
    blocks: Iterable[SemanticKnowledgeBlock],
) -> dict[str, Any]:
    selected = tuple(blocks)
    return {
        "knowledge_version": knowledge.version,
        "knowledge_sha256": knowledge.sha256,
        "retrieved_blocks": [block.prompt_payload() for block in selected],
        "knowledge_boundary": (
            "These blocks contain qualitative scientific constraints only. "
            "They do not define the target expression or grant additional operators."
        ),
    }


def assert_generator_source_disjointness(
    knowledge: SemanticKnowledgeBase,
    generator_sources: Iterable[str],
) -> None:
    generator = {str(item).strip().lower() for item in generator_sources}
    evidence = {
        source.strip().lower()
        for block in knowledge.blocks
        for source in block.evidence_sources
    }
    overlap = sorted(generator & evidence)
    if overlap:
        raise SemanticKnowledgeError(
            "Benchmark generator sources overlap the prompt knowledge evidence: "
            + ", ".join(overlap)
        )


def semantic_knowledge_audit(
    knowledge: SemanticKnowledgeBase,
) -> dict[str, Any]:
    return {
        "status": "passed",
        "version": knowledge.version,
        "sha256": knowledge.sha256,
        "block_count": len(knowledge.blocks),
        "prompt_fields": [
            "id",
            "topics",
            "statement",
            "scope",
            "evidence_level",
        ],
        "forbidden_answer_fields": sorted(FORBIDDEN_ANSWER_FIELDS),
        "sources_are_excluded_from_prompt": True,
    }

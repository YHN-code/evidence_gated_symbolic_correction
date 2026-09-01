from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from asrc.model_revision.candidate_racing import CandidateRacingConfig
from asrc.model_revision.proposals import (
    ProposalBatch,
    RejectedRepair,
    RepairContract,
    TypedRepair,
    validate_typed_repair,
)
from asrc.utils.io import read_json


@dataclass(frozen=True)
class PortfolioProvenance:
    proposal_id: str
    structural_key: str
    source_labels: tuple[str, ...]
    source_proposal_ids: tuple[str, ...]

    def to_row(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "structural_key": self.structural_key,
            "source_labels": list(self.source_labels),
            "source_proposal_ids": list(self.source_proposal_ids),
            "source_count": len(self.source_labels),
        }


@dataclass(frozen=True)
class SourceNeutralPortfolio:
    repairs: tuple[TypedRepair, ...]
    provenance: tuple[PortfolioProvenance, ...]
    input_candidate_count: int
    duplicate_candidate_count: int


def load_proposal_batch(
    path: str | Path,
    contract: RepairContract,
) -> ProposalBatch:
    payload = read_json(Path(path))
    return ProposalBatch(
        accepted=tuple(
            validate_typed_repair(raw, contract)
            for raw in payload.get("accepted", [])
        ),
        rejected=tuple(
            RejectedRepair(
                str(raw["proposal_id"]),
                str(raw["source"]),
                str(raw["reason"]),
            )
            for raw in payload.get("rejected", [])
        ),
    )


def _portfolio_id(structural_key: str) -> str:
    digest = hashlib.sha256(structural_key.encode("utf-8")).hexdigest()[:16]
    return f"portfolio_{digest}"


def _representative_key(repair: TypedRepair) -> str:
    return json.dumps(
        {
            "edit_type": repair.edit_type,
            "target": repair.target,
            "expression": repair.expression,
            "rationale": repair.rationale,
            "expected_signature": repair.expected_signature,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_source_neutral_portfolio(
    batches: Mapping[str, ProposalBatch],
    *,
    maximum_initial_candidates: int,
) -> SourceNeutralPortfolio:
    if not batches:
        raise ValueError("A source-neutral portfolio requires at least one source.")
    if maximum_initial_candidates < 1:
        raise ValueError("maximum_initial_candidates must be positive.")

    grouped: dict[str, list[tuple[str, TypedRepair]]] = {}
    input_count = 0
    for source_label, batch in sorted(batches.items()):
        label = str(source_label).strip()
        if not label:
            raise ValueError("Portfolio source labels must not be empty.")
        for repair in batch.accepted:
            input_count += 1
            grouped.setdefault(repair.structural_key, []).append((label, repair))

    if len(grouped) > maximum_initial_candidates:
        raise ValueError(
            "The deduplicated source-neutral portfolio exceeds the frozen "
            f"maximum_initial_candidates={maximum_initial_candidates}; got {len(grouped)}. "
            "Source-specific truncation is prohibited."
        )

    repairs: list[TypedRepair] = []
    provenance: list[PortfolioProvenance] = []
    seen_ids: set[str] = set()
    for structural_key in sorted(grouped):
        members = grouped[structural_key]
        representative = min((item[1] for item in members), key=_representative_key)
        proposal_id = _portfolio_id(structural_key)
        if proposal_id in seen_ids:
            raise ValueError("Portfolio proposal hash collision detected.")
        seen_ids.add(proposal_id)
        repairs.append(
            replace(
                representative,
                proposal_id=proposal_id,
                source="replay",
            )
        )
        provenance.append(
            PortfolioProvenance(
                proposal_id=proposal_id,
                structural_key=structural_key,
                source_labels=tuple(sorted({item[0] for item in members})),
                source_proposal_ids=tuple(
                    sorted(f"{item[0]}:{item[1].proposal_id}" for item in members)
                ),
            )
        )

    return SourceNeutralPortfolio(
        repairs=tuple(repairs),
        provenance=tuple(provenance),
        input_candidate_count=input_count,
        duplicate_candidate_count=input_count - len(repairs),
    )


def nominal_racing_resource_upper_bound(
    initial_candidate_count: int,
    racing_config: CandidateRacingConfig | Mapping[str, Any],
) -> int:
    config = (
        racing_config
        if isinstance(racing_config, CandidateRacingConfig)
        else CandidateRacingConfig.from_mapping(racing_config)
    )
    active = max(0, int(initial_candidate_count))
    total = 0
    for index, stage in enumerate(config.stages):
        if active == 0:
            break
        total += active * stage.resource_units
        if index < len(config.stages) - 1:
            active = min(
                active,
                max(
                    config.minimum_survivors,
                    int(math.ceil(active * stage.promote_fraction)),
                ),
            )
    return total

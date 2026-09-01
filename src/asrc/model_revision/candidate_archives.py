from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from asrc.model_revision.benchmarks import Gate1Task
from asrc.model_revision.proposals import (
    TypedRepair,
    canonical_expression_key,
    validate_typed_repair,
)
from asrc.utils.io import read_json, runs_root


def _archive_fingerprint(
    candidates: Mapping[str, tuple[TypedRepair, ...]],
) -> str:
    payload = {
        task_id: [canonical_expression_key(item.expression) for item in repairs]
        for task_id, repairs in sorted(candidates.items())
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_development_matrix_candidates(
    archive: Mapping[str, Any],
    tasks: Mapping[str, Gate1Task],
) -> tuple[dict[str, tuple[TypedRepair, ...]], pd.DataFrame, dict[str, Any]]:
    """Load a confirmation-response-independent committee from a Gate 1A archive."""
    source_run_id = str(archive["source_run_id"])
    source_method = str(archive["source_method"])
    source_dir = runs_root() / source_run_id
    report_path = source_dir / "reports" / "gate1_pilot_summary.json"
    candidate_path = source_dir / "metrics" / "gate1_candidate_results.csv"
    if not report_path.exists() or not candidate_path.exists():
        raise FileNotFoundError("The Gate 1A development-matrix archive is incomplete.")

    report = read_json(report_path)
    source_seed = int(archive["source_data_seed"])
    if int(report.get("seed", -1)) != source_seed:
        raise ValueError("Development-matrix archive seed does not match the protocol.")
    if source_method not in report.get("methods", []):
        raise ValueError("Development-matrix archive method is unavailable.")

    task_ids = tuple(str(value) for value in archive["task_ids"])
    if set(task_ids) != set(tasks):
        raise ValueError("Development-matrix task set does not match the protocol.")
    frame = pd.read_csv(candidate_path)
    frame = frame.loc[
        frame["task_id"].isin(task_ids)
        & frame["method"].eq(source_method)
        & frame["status"].eq("valid")
    ].copy()

    maximum = int(archive["maximum_structures_per_task"])
    selected: dict[str, tuple[TypedRepair, ...]] = {}
    audit_rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        task = tasks[task_id]
        proposal_path = (
            source_dir
            / "formulas"
            / "gate1"
            / f"{task_id}_{source_method}_proposals.json"
        )
        if not proposal_path.exists():
            raise FileNotFoundError(f"Missing frozen proposal archive: {proposal_path}")
        proposal_payload = read_json(proposal_path)
        proposal_map = {
            repair.proposal_id: repair
            for repair in (
                validate_typed_repair(item, task.contract)
                for item in proposal_payload.get("accepted", [])
            )
        }
        task_rows = frame.loc[frame["task_id"].eq(task_id)].sort_values(
            ["selection_score", "complexity", "proposal_id"]
        )
        repairs: list[TypedRepair] = []
        seen: set[str] = set()
        for _, row in task_rows.iterrows():
            source_id = str(row["proposal_id"])
            repair = proposal_map.get(source_id)
            if repair is None:
                raise ValueError(
                    f"Candidate row has no archived proposal: {(task_id, source_id)}"
                )
            expression_key = canonical_expression_key(repair.expression)
            if expression_key in seen:
                continue
            seen.add(expression_key)
            repair = replace(
                repair,
                proposal_id=f"gate1d_{task_id.lower()}_{len(repairs) + 1:02d}",
            )
            repairs.append(repair)
            audit_rows.append(
                {
                    "task_id": task_id,
                    "candidate_order": len(repairs),
                    "gate1d_proposal_id": repair.proposal_id,
                    "archive_source": "gate1a_development_matrix",
                    "source_run_id": source_run_id,
                    "source_method": source_method,
                    "source_proposal_id": source_id,
                    "source": str(row["source"]),
                    "formula": str(row["formula"]),
                    "expression_key": expression_key,
                    "source_validation_rmse": float(row["validation_rmse"]),
                    "source_selection_score": float(row["selection_score"]),
                    "complexity": int(row["complexity"]),
                }
            )
            if len(repairs) >= maximum:
                break
        if len(repairs) < 2:
            raise ValueError(f"Task {task_id} has fewer than two frozen candidates.")
        selected[task_id] = tuple(repairs)

    return selected, pd.DataFrame(audit_rows), {
        "source_run_id": source_run_id,
        "source_method": source_method,
        "source_data_seed": source_seed,
        "candidate_set_sha256": _archive_fingerprint(selected),
    }


def combined_candidate_fingerprint(
    candidates: Mapping[str, tuple[TypedRepair, ...]],
) -> str:
    return _archive_fingerprint(candidates)

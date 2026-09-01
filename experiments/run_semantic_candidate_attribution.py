from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from asrc.model_revision.evaluation import finalize_selected_repair
from asrc.model_revision.llm_proposer import LLMRepairProposer
from asrc.model_revision.proposals import (
    ProposalBatch,
    RejectedRepair,
    TypedRepair,
    validate_typed_repair,
)
from asrc.model_revision.pysr_proposer import (
    PySRRepairProposer,
    PySRRepairUnavailableError,
)
from asrc.model_revision.risk_aware_racing import (
    run_risk_aware_candidate_race,
)
from asrc.model_revision.semantic_benchmarks import (
    SemanticBenchmarkTask,
    build_semantic_benchmark_suite,
)
from asrc.model_revision.semantic_knowledge import (
    assert_generator_source_disjointness,
    knowledge_prompt_payload,
    load_semantic_knowledge,
    retrieve_semantic_knowledge,
    semantic_knowledge_audit,
)
from asrc.utils.io import (
    ensure_run_dir,
    project_root,
    read_json,
    read_yaml,
    write_json_atomic,
    write_table_bundle,
)
from asrc.utils.progress import mark_done, progress_message, should_skip


EXPECTED_METHODS = (
    "original_baseline",
    "pysr",
    "llm_guided_sr",
    "full_asrc",
)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _repair_payload(repair: TypedRepair) -> dict[str, Any]:
    return {
        "proposal_id": repair.proposal_id,
        "source": repair.source,
        "edit_type": repair.edit_type,
        "target": repair.target,
        "expression": repair.expression,
        "rationale": repair.rationale,
        "expected_signature": repair.expected_signature,
    }


def _batch_payload(
    batch: ProposalBatch,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "accepted": [_repair_payload(item) for item in batch.accepted],
        "rejected": [
            {
                "proposal_id": item.proposal_id,
                "source": item.source,
                "reason": item.reason,
            }
            for item in batch.rejected
        ],
        "metadata": dict(metadata or {}),
    }


def _load_batch(path: Path, item: SemanticBenchmarkTask) -> ProposalBatch:
    payload = read_json(path)
    return ProposalBatch(
        accepted=tuple(
            validate_typed_repair(raw, item.task.contract)
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


def _resolve_project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else project_root() / value


def _validate_config(config: Mapping[str, Any]) -> None:
    methods = tuple(str(item) for item in config.get("methods", []))
    if methods != EXPECTED_METHODS:
        raise ValueError(
            "Semantic attribution methods must be exactly "
            + ", ".join(EXPECTED_METHODS)
        )
    if bool(config.get("confirmation", {}).get("enabled", False)):
        raise ValueError(
            "This configuration is development-only; confirmation requires a new frozen task registry."
        )
    if not bool(config["candidate_budget"].get("equal_evaluated_candidate_cap")):
        raise ValueError("Equal candidate evaluation caps are required.")
    if int(config["candidate_budget"]["maximum_unique_candidates"]) < 2:
        raise ValueError("Candidate budget is too small for attribution.")


def _task_registration(item: SemanticBenchmarkTask) -> dict[str, Any]:
    definition = item.task.definition
    return {
        "task_id": item.task.task_id,
        "mechanism_class": item.mechanism_class,
        "task_family": definition.task_family,
        "variables": dict(definition.variable_descriptions),
        "observed_ranges": {
            key: list(value) for key, value in definition.observed_ranges.items()
        },
        "locked_ranges": {
            key: list(value) for key, value in definition.locked_ranges.items()
        },
        "public_semantics": dict(item.public_semantics),
        "knowledge_topics": list(item.knowledge_topics),
        "generator_sources_private": list(item.generator_sources),
        "oracle_fingerprint_private": _sha256(
            {
                "payload": definition.oracle_payload,
                "parameters": dict(definition.oracle_parameters),
            }
        ),
    }


def _preflight(
    config: Mapping[str, Any],
    tasks: tuple[SemanticBenchmarkTask, ...],
    *,
    run_dir: Path,
) -> dict[str, Any]:
    knowledge_path = _resolve_project_path(config["knowledge"]["path"])
    knowledge = load_semantic_knowledge(knowledge_path)
    registrations = []
    retrievals = []
    top_k = int(config["knowledge"]["top_k"])
    for item in tasks:
        if bool(config["knowledge"].get("prohibit_generator_source_overlap", True)):
            assert_generator_source_disjointness(
                knowledge,
                item.generator_sources,
            )
        selected = retrieve_semantic_knowledge(
            knowledge,
            item.knowledge_topics,
            top_k=top_k,
        )
        if not selected:
            raise ValueError(
                f"No answer-safe knowledge was retrieved for {item.task.task_id}."
            )
        prompt_view = knowledge_prompt_payload(knowledge, selected)
        serialized_prompt_view = json.dumps(prompt_view, ensure_ascii=False).lower()
        if "oracle" in serialized_prompt_view or any(
            source.lower() in serialized_prompt_view
            for source in item.generator_sources
        ):
            raise ValueError(
                f"Private generator evidence leaked into {item.task.task_id} prompt view."
            )
        registrations.append(_task_registration(item))
        retrievals.append(
            {
                "task_id": item.task.task_id,
                "query_topics": list(item.knowledge_topics),
                "retrieved_ids": [block.block_id for block in selected],
                "prompt_view_sha256": _sha256(prompt_view),
            }
        )
    report = {
        "status": "passed",
        "protocol": dict(config["protocol"]),
        "development_only": True,
        "methods": list(EXPECTED_METHODS),
        "shared_operator_set": sorted(tasks[0].task.contract.allowed_operators),
        "shared_candidate_cap": int(
            config["candidate_budget"]["maximum_unique_candidates"]
        ),
        "knowledge_audit": semantic_knowledge_audit(knowledge),
        "task_registration_sha256": _sha256(registrations),
        "task_registrations": registrations,
        "retrieval_audit": retrievals,
        "new_llm_calls": 0,
        "new_pysr_searches": 0,
        "confirmation_unlocked": False,
    }
    write_json_atomic(
        run_dir / "reports" / "semantic_candidate_attribution_preflight.json",
        report,
    )
    return report


def _proposal_path(run_dir: Path, task_id: str, method: str) -> Path:
    return run_dir / "formulas" / "semantic_attribution" / f"{task_id}_{method}.json"


def _llm_batch(
    item: SemanticBenchmarkTask,
    *,
    method: str,
    context: Mapping[str, Any],
    config: Mapping[str, Any],
    run_dir: Path,
    llm_config: str,
    resume: bool,
    force: bool,
    verbose: bool,
) -> tuple[ProposalBatch, dict[str, Any]]:
    path = _proposal_path(run_dir, item.task.task_id, method)
    if resume and not force and path.exists():
        return _load_batch(path, item), dict(read_json(path).get("metadata", {}))

    checkpoint = (
        run_dir
        / "formulas"
        / "semantic_attribution"
        / "checkpoints"
        / f"{item.task.task_id}_{method}.json"
    )

    def report(event: str, fields: dict[str, Any]) -> None:
        progress_message(
            run_dir,
            f"Semantic attribution {event}",
            verbose,
            task=item.task.task_id,
            method=method,
            **fields,
        )

    llm_cfg = config["llm"]
    proposer = LLMRepairProposer(
        config_path=llm_config,
        mode=str(llm_cfg.get("mode", "typed_repair")),
        candidates_per_call=int(llm_cfg["candidates_per_call"]),
        maximum_generation_calls=int(llm_cfg["maximum_generation_calls"]),
        transport_retry_attempts=int(llm_cfg.get("transport_retry_attempts", 5)),
        transport_retry_wait_seconds=float(
            llm_cfg.get("transport_retry_wait_seconds", 25.0)
        ),
        reasoning_effort_override=llm_cfg.get("reasoning_effort_override"),
        max_tokens_override=llm_cfg.get("max_tokens_override"),
        proposal_context=context,
        checkpoint_path=checkpoint,
        resume=resume and not force,
        on_progress=report,
    )
    batch = proposer.propose(item.task.request, item.task.contract)
    metadata = {
        "llm_calls": proposer.llm_calls,
        "structured_repair_calls": proposer.repair_calls,
        "transport_failures": proposer.transport_failures,
        "truncation_retries": proposer.truncation_retries,
        "wall_time_seconds": proposer.llm_elapsed_seconds,
        "proposal_context_sha256": proposer.proposal_context_sha256,
        "checkpoint": str(checkpoint),
    }
    write_json_atomic(path, _batch_payload(batch, metadata))
    return batch, metadata


def _pysr_batch(
    item: SemanticBenchmarkTask,
    *,
    config: Mapping[str, Any],
    run_dir: Path,
    seed: int,
    mode: str,
    resume: bool,
    force: bool,
    verbose: bool,
) -> tuple[ProposalBatch | None, dict[str, Any]]:
    path = _proposal_path(run_dir, item.task.task_id, "pysr")
    if resume and not force and path.exists():
        return _load_batch(path, item), dict(read_json(path).get("metadata", {}))
    progress_message(
        run_dir,
        "Semantic attribution PySR search started",
        verbose,
        task=item.task.task_id,
    )
    proposer = PySRRepairProposer(
        observed=item.task.observed,
        variable_names=item.task.variables,
        seed=seed,
        output_directory=run_dir / "formulas" / "pysr" / item.task.task_id,
        config=config["pysr"],
    )
    try:
        batch = proposer.propose(item.task.request, item.task.contract)
    except PySRRepairUnavailableError:
        if mode == "required":
            raise
        progress_message(
            run_dir,
            "Semantic attribution optional PySR arm skipped",
            verbose,
            task=item.task.task_id,
        )
        return None, {"status": "unavailable"}
    metadata = {
        "wall_time_seconds": proposer.wall_time_seconds,
        "internal_iterations": proposer.internal_iterations,
    }
    write_json_atomic(path, _batch_payload(batch, metadata))
    return batch, metadata


def _evaluate_method(
    item: SemanticBenchmarkTask,
    batch: ProposalBatch,
    *,
    method: str,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    result = run_risk_aware_candidate_race(
        item.task,
        batch.accepted,
        method=method,
        seed=seed,
        evaluation_config=config["evaluation"],
        racing_config=config["candidate_racing"],
        selection_config=config["risk_selection"],
    )
    repair_by_id = {repair.proposal_id: repair for repair in batch.accepted}
    summary = finalize_selected_repair(
        item.task,
        method,
        result.evaluations,
        result.selected,
        repair_by_id,
        recovery_noise_multiplier=float(
            config["evaluation"]["recovery_noise_multiplier"]
        ),
        retain_baseline_if_unselected=True,
    )
    summary = replace(summary, candidate_count=result.initial_candidate_count)
    method_row = {
        **summary.to_row(),
        "mechanism_class": item.mechanism_class,
        "total_optimizer_evaluations": result.total_optimizer_evaluations,
        "completed_racing_stages": result.completed_stage_count,
        "risk_selection_threshold": result.selection_threshold,
    }
    candidate_rows = []
    for evaluation in result.evaluations:
        row = evaluation.to_row()
        row["parameter_values"] = json.dumps(
            row["parameter_values"], sort_keys=True
        )
        row["mechanism_class"] = item.mechanism_class
        candidate_rows.append(row)
    risk_rows = []
    for assessment in (*result.assessments, result.baseline_assessment):
        row = assessment.to_row()
        row["environment_rmse"] = json.dumps(
            row["environment_rmse"], sort_keys=True
        )
        row["environment_rmse_standard_error"] = json.dumps(
            assessment.environment_rmse_standard_error, sort_keys=True
        )
        row["method"] = method
        row["mechanism_class"] = item.mechanism_class
        risk_rows.append(row)
    trace_rows = []
    for trace in result.trace:
        row = trace.to_row()
        row["mechanism_class"] = item.mechanism_class
        trace_rows.append(row)
    return method_row, candidate_rows, risk_rows, trace_rows


def _baseline_row(item: SemanticBenchmarkTask, config: Mapping[str, Any]) -> dict[str, Any]:
    summary = finalize_selected_repair(
        item.task,
        "original_baseline",
        (),
        None,
        {},
        recovery_noise_multiplier=float(
            config["evaluation"]["recovery_noise_multiplier"]
        ),
        retain_baseline_if_unselected=True,
    )
    return {
        **summary.to_row(),
        "mechanism_class": item.mechanism_class,
        "total_optimizer_evaluations": 0,
        "completed_racing_stages": 0,
        "risk_selection_threshold": float("nan"),
    }


def _inventory_rows(
    item: SemanticBenchmarkTask,
    method: str,
    batch: ProposalBatch,
) -> list[dict[str, Any]]:
    return [
        {
            "task_id": item.task.task_id,
            "mechanism_class": item.mechanism_class,
            "method": method,
            "proposal_id": repair.proposal_id,
            "source": repair.source,
            "structural_key": repair.structural_key,
            "edit_type": repair.edit_type,
            "rationale": repair.rationale,
            "expected_signature": repair.expected_signature,
        }
        for repair in batch.accepted
    ]


def _candidate_attribution(
    methods: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "task_id",
        "mechanism_class",
        "method",
        "generated_count",
        "unique_generated_count",
        "selected_proposal_id",
        "selected_structural_key",
        "selected_unique_structure",
        "selected_locked_improves_baseline",
        "unique_supported_selected",
    ]
    if methods.empty or inventory.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    symbolic_methods = ("pysr", "llm_guided_sr", "full_asrc")
    for task_id, task_methods in methods.groupby("task_id", sort=False):
        task_inventory = inventory.loc[inventory["task_id"].eq(task_id)]
        keys = {
            method: set(
                task_inventory.loc[
                    task_inventory["method"].eq(method), "structural_key"
                ]
            )
            for method in symbolic_methods
        }
        for method in symbolic_methods:
            method_row = task_methods.loc[task_methods["method"].eq(method)]
            if method_row.empty:
                continue
            method_row = method_row.iloc[0]
            selected_id = str(method_row["selected_proposal_id"])
            selected_inventory = task_inventory.loc[
                task_inventory["method"].eq(method)
                & task_inventory["proposal_id"].eq(selected_id)
            ]
            selected_key = (
                str(selected_inventory.iloc[0]["structural_key"])
                if not selected_inventory.empty
                else ""
            )
            other_keys = set().union(
                *(keys[other] for other in symbolic_methods if other != method)
            )
            selected_unique = bool(selected_key and selected_key not in other_keys)
            locked_improved = bool(
                float(method_row["locked_rmse"])
                < float(method_row["baseline_locked_rmse"])
            )
            rows.append(
                {
                    "task_id": task_id,
                    "mechanism_class": method_row["mechanism_class"],
                    "method": method,
                    "generated_count": len(keys[method]),
                    "unique_generated_count": len(keys[method] - other_keys),
                    "selected_proposal_id": selected_id,
                    "selected_structural_key": selected_key,
                    "selected_unique_structure": selected_unique,
                    "selected_locked_improves_baseline": locked_improved,
                    "unique_supported_selected": selected_unique and locked_improved,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _method_summary(
    methods: pd.DataFrame,
    attribution: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for method, group in methods.groupby("method", sort=False):
        method_attribution = attribution.loc[attribution["method"].eq(method)]
        rows.append(
            {
                "method": method,
                "task_count": int(len(group)),
                "median_locked_rmse": float(group["locked_rmse"].median()),
                "mean_locked_rmse": float(group["locked_rmse"].mean()),
                "behavior_recovery_count": int(
                    group["corrected_behavior_recovered"].astype(bool).sum()
                ),
                "baseline_retained_count": int(
                    group["selected_source"].eq("baseline").sum()
                ),
                "mean_complexity": float(group["complexity"].mean()),
                "physical_violation_count": int(
                    group["stability_violations"].sum()
                ),
                "optimizer_evaluations": int(
                    group["total_optimizer_evaluations"].sum()
                ),
                "unique_supported_structure_count": int(
                    method_attribution["unique_supported_selected"].astype(bool).sum()
                )
                if not method_attribution.empty
                else 0,
            }
        )
    return pd.DataFrame(rows)


def _paired_comparisons(methods: pd.DataFrame) -> pd.DataFrame:
    pivot = methods.pivot(index="task_id", columns="method", values="locked_rmse")
    rows = []
    for comparator in ("pysr", "llm_guided_sr"):
        if "full_asrc" not in pivot or comparator not in pivot:
            continue
        paired = pivot[["full_asrc", comparator]].dropna()
        full = paired["full_asrc"].to_numpy(float)
        other = paired[comparator].to_numpy(float)
        tolerance = 1.0e-12 * np.maximum(1.0, np.maximum(np.abs(full), np.abs(other)))
        wins = int(np.sum(full < other - tolerance))
        losses = int(np.sum(full > other + tolerance))
        ties = int(len(paired) - wins - losses)
        relative = (other - full) / np.maximum(np.abs(other), 1.0e-12)
        nonzero = full - other
        nonzero = nonzero[np.abs(nonzero) > 1.0e-15]
        p_value = (
            float(wilcoxon(nonzero, alternative="less").pvalue)
            if len(nonzero) >= 2
            else float("nan")
        )
        rows.append(
            {
                "treatment": "full_asrc",
                "comparator": comparator,
                "paired_tasks": int(len(paired)),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "median_relative_locked_improvement": float(np.median(relative)),
                "one_sided_wilcoxon_p": p_value,
                "development_only": True,
            }
        )
    return pd.DataFrame(rows)


def run(
    config_path: str,
    run_id: str,
    *,
    seed: int | None,
    llm: bool,
    llm_config: str | None,
    pysr_mode: str,
    resume: bool,
    force: bool,
    verbose: bool,
    preflight_only: bool,
    limit: int | None,
    task_ids: list[str] | None,
) -> Path:
    config = read_yaml(config_path)
    _validate_config(config)
    actual_seed = int(seed if seed is not None else config["development_seed"])
    run_dir = ensure_run_dir(run_id)
    tasks = build_semantic_benchmark_suite(config, seed=actual_seed)
    if task_ids:
        requested = set(task_ids)
        available = {item.task.task_id for item in tasks}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"Unknown semantic task ids: {unknown}")
        tasks = tuple(item for item in tasks if item.task.task_id in requested)
    if limit is not None:
        tasks = tasks[: max(0, int(limit))]
    if not tasks:
        raise ValueError("No semantic attribution tasks were selected.")

    preflight = _preflight(config, tasks, run_dir=run_dir)
    progress_message(
        run_dir,
        "Semantic attribution preflight passed",
        verbose,
        tasks=len(tasks),
        knowledge=preflight["knowledge_audit"]["version"],
    )
    if preflight_only:
        progress_message(
            run_dir,
            "Semantic attribution preflight complete; no model search executed",
            verbose,
            output=run_dir,
        )
        return run_dir
    if not llm or not llm_config:
        raise ValueError(
            "The registered four-method run requires --llm and --llm-config."
        )
    if pysr_mode == "off":
        raise ValueError("The registered four-method run requires PySR.")

    stage = "semantic_candidate_attribution_v1"
    method_output = run_dir / "metrics" / "semantic_attribution_method_results.csv"
    candidate_output = run_dir / "metrics" / "semantic_attribution_candidate_results.csv"
    if should_skip(
        run_dir,
        stage,
        [method_output, candidate_output],
        resume,
        force,
    ):
        progress_message(run_dir, "Semantic attribution run already complete", verbose)
        return run_dir

    knowledge = load_semantic_knowledge(
        _resolve_project_path(config["knowledge"]["path"])
    )
    method_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    generation_metadata: dict[str, Any] = {}
    knowledge_logs: list[dict[str, Any]] = []
    progress_message(
        run_dir,
        "Semantic candidate attribution started",
        verbose,
        tasks=len(tasks),
        seed=actual_seed,
    )

    for index, item in enumerate(tasks, start=1):
        task_id = item.task.task_id
        checkpoint = (
            run_dir / "reports" / "semantic_attribution_checkpoints" / f"{task_id}.json"
        )
        if resume and not force and checkpoint.exists():
            saved = read_json(checkpoint)
            method_rows.extend(saved["method_rows"])
            candidate_rows.extend(saved["candidate_rows"])
            risk_rows.extend(saved["risk_rows"])
            trace_rows.extend(saved["trace_rows"])
            inventory_rows.extend(saved["inventory_rows"])
            generation_metadata[task_id] = saved["generation_metadata"]
            knowledge_logs.append(saved["knowledge_log"])
            progress_message(
                run_dir,
                "Semantic attribution task restored",
                verbose,
                task=f"{task_id} ({index}/{len(tasks)})",
            )
            continue

        progress_message(
            run_dir,
            "Semantic attribution task started",
            verbose,
            task=f"{task_id} ({index}/{len(tasks)})",
            mechanism=item.mechanism_class,
        )
        task_method_rows = [_baseline_row(item, config)]
        task_candidate_rows: list[dict[str, Any]] = []
        task_risk_rows: list[dict[str, Any]] = []
        task_trace_rows: list[dict[str, Any]] = []
        task_inventory_rows: list[dict[str, Any]] = []
        task_generation: dict[str, Any] = {}

        pysr_batch, pysr_metadata = _pysr_batch(
            item,
            config=config,
            run_dir=run_dir,
            seed=actual_seed + index,
            mode=pysr_mode,
            resume=resume,
            force=force,
            verbose=verbose,
        )
        if pysr_batch is None:
            raise RuntimeError("PySR is required by the registered method matrix.")
        task_generation["pysr"] = pysr_metadata

        selected_blocks = retrieve_semantic_knowledge(
            knowledge,
            item.knowledge_topics,
            top_k=int(config["knowledge"]["top_k"]),
        )
        retrieval_payload = knowledge_prompt_payload(knowledge, selected_blocks)
        no_retrieval_context = {
            "retrieval_mode": "disabled",
            "knowledge_boundary": "No explicit domain knowledge was retrieved.",
        }
        full_context = {
            "retrieval_mode": "answer_safe_literature_retrieval",
            **retrieval_payload,
        }
        llm_plain, plain_metadata = _llm_batch(
            item,
            method="llm_guided_sr",
            context=no_retrieval_context,
            config=config,
            run_dir=run_dir,
            llm_config=llm_config,
            resume=resume,
            force=force,
            verbose=verbose,
        )
        llm_full, full_metadata = _llm_batch(
            item,
            method="full_asrc",
            context=full_context,
            config=config,
            run_dir=run_dir,
            llm_config=llm_config,
            resume=resume,
            force=force,
            verbose=verbose,
        )
        task_generation["llm_guided_sr"] = plain_metadata
        task_generation["full_asrc"] = full_metadata
        batches = {
            "pysr": pysr_batch,
            "llm_guided_sr": llm_plain,
            "full_asrc": llm_full,
        }

        for method, batch in batches.items():
            progress_message(
                run_dir,
                "Semantic attribution candidate race started",
                verbose,
                task=task_id,
                method=method,
                candidates=len(batch.accepted),
            )
            method_row, candidates, risks, traces = _evaluate_method(
                item,
                batch,
                method=method,
                config=config,
                seed=actual_seed + index,
            )
            task_method_rows.append(method_row)
            task_candidate_rows.extend(candidates)
            task_risk_rows.extend(risks)
            task_trace_rows.extend(traces)
            task_inventory_rows.extend(_inventory_rows(item, method, batch))
            progress_message(
                run_dir,
                "Semantic attribution candidate race finished",
                verbose,
                task=task_id,
                method=method,
                locked_rmse=f"{float(method_row['locked_rmse']):.6g}",
                selected_source=method_row["selected_source"],
            )

        knowledge_log = {
            "task_id": task_id,
            "query_topics": list(item.knowledge_topics),
            "retrieved_ids": [block.block_id for block in selected_blocks],
            "knowledge_version": knowledge.version,
            "knowledge_sha256": knowledge.sha256,
            "prompt_view_sha256": _sha256(retrieval_payload),
            "generator_sources_excluded_from_prompt": True,
        }
        payload = {
            "task_id": task_id,
            "method_rows": task_method_rows,
            "candidate_rows": task_candidate_rows,
            "risk_rows": task_risk_rows,
            "trace_rows": task_trace_rows,
            "inventory_rows": task_inventory_rows,
            "generation_metadata": task_generation,
            "knowledge_log": knowledge_log,
        }
        write_json_atomic(checkpoint, payload)
        method_rows.extend(task_method_rows)
        candidate_rows.extend(task_candidate_rows)
        risk_rows.extend(task_risk_rows)
        trace_rows.extend(task_trace_rows)
        inventory_rows.extend(task_inventory_rows)
        generation_metadata[task_id] = task_generation
        knowledge_logs.append(knowledge_log)

    method_frame = pd.DataFrame(method_rows)
    candidate_frame = pd.DataFrame(candidate_rows)
    risk_frame = pd.DataFrame(risk_rows)
    trace_frame = pd.DataFrame(trace_rows)
    inventory_frame = pd.DataFrame(inventory_rows)
    attribution_frame = _candidate_attribution(method_frame, inventory_frame)
    summary_frame = _method_summary(method_frame, attribution_frame)
    paired_frame = _paired_comparisons(method_frame)
    write_table_bundle(method_frame, method_output.with_suffix(""))
    write_table_bundle(candidate_frame, candidate_output.with_suffix(""))
    write_table_bundle(
        risk_frame,
        run_dir / "metrics" / "semantic_attribution_risk_assessments",
    )
    write_table_bundle(
        trace_frame,
        run_dir / "metrics" / "semantic_attribution_racing_trace",
    )
    write_table_bundle(
        inventory_frame,
        run_dir / "metrics" / "semantic_attribution_candidate_inventory",
    )
    write_table_bundle(
        attribution_frame,
        run_dir / "metrics" / "semantic_attribution_source_attribution",
    )
    write_table_bundle(
        summary_frame,
        run_dir / "tables" / "semantic_attribution_method_summary",
    )
    write_table_bundle(
        paired_frame,
        run_dir / "tables" / "semantic_attribution_paired_comparisons",
    )
    write_json_atomic(
        run_dir / "reports" / "semantic_candidate_attribution_report.json",
        {
            "protocol": config["protocol"],
            "development_only": True,
            "seed": actual_seed,
            "task_count": len(tasks),
            "methods": list(EXPECTED_METHODS),
            "knowledge_audit": preflight["knowledge_audit"],
            "knowledge_logs": knowledge_logs,
            "generation_metadata": generation_metadata,
            "result_interpretation": (
                "Development evidence only. A new frozen mechanism registry and "
                "unseen seeds are required before confirmatory claims."
            ),
        },
    )
    mark_done(
        run_dir,
        stage,
        {"task_count": len(tasks), "method_rows": len(method_frame)},
    )
    progress_message(
        run_dir,
        "Semantic candidate attribution complete",
        verbose,
        output=run_dir,
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the answer-safe semantic candidate-source attribution study."
    )
    parser.add_argument(
        "--config",
        default="configs/semantic_candidate_attribution_v1.yaml",
    )
    parser.add_argument("--run-id", default="semantic_candidate_attribution_v1")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--llm-config")
    parser.add_argument(
        "--pysr",
        choices=("required", "optional", "off"),
        default="required",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-id", dest="task_ids", action="append")
    args = parser.parse_args()
    run(
        args.config,
        args.run_id,
        seed=args.seed,
        llm=args.llm,
        llm_config=args.llm_config,
        pysr_mode=args.pysr,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
        preflight_only=args.preflight_only,
        limit=args.limit,
        task_ids=args.task_ids,
    )


if __name__ == "__main__":
    main()

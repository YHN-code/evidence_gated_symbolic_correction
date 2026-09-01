from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from experiments._bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from _bootstrap import ROOT  # type: ignore  # noqa: F401

if __package__ in {None, ""}:
    from run_semantic_candidate_attribution import (  # type: ignore  # noqa: E402
        _llm_batch,
        _pysr_batch,
    )
else:
    from experiments.run_semantic_candidate_attribution import (  # noqa: E402
        _llm_batch,
        _pysr_batch,
    )

from asrc.model_revision.ambiguity_active_design import (  # noqa: E402
    reveal_acquisition_observation,
)
from asrc.model_revision.candidate_portfolio import (  # noqa: E402
    build_source_neutral_portfolio,
)
from asrc.model_revision.confirmation import paired_cluster_comparisons  # noqa: E402
from asrc.model_revision.evidence_guided_revision import (  # noqa: E402
    EvidenceGuidedIteration,
    EvidenceGuidedRevisionConfig,
    EvidenceGuidedRevisionResult,
    run_evidence_guided_revision,
)
from asrc.model_revision.mechanism_transfer_benchmarks import (  # noqa: E402
    MechanismTransferTask,
    build_mechanism_transfer_suite,
)
from asrc.model_revision.semantic_knowledge import (  # noqa: E402
    assert_generator_source_disjointness,
    knowledge_prompt_payload,
    load_semantic_knowledge,
    retrieve_semantic_knowledge,
    semantic_knowledge_audit,
)
from asrc.utils.io import (  # noqa: E402
    ensure_run_dir,
    project_root,
    read_json,
    read_yaml,
    write_json_atomic,
    write_table_bundle,
)
from asrc.utils.progress import mark_done, progress_message, should_skip  # noqa: E402


STAGE_NAME = "semantic_evidence_guided_revision_confirmation_v1"
NO_ACQUISITION = "no_acquisition"
SPACE_FILLING = "conditional_space_filling"
PREDICTIVE = "evidence_guided_predictive_disagreement"
EXPECTED_METHODS = (NO_ACQUISITION, SPACE_FILLING, PREDICTIVE)


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


def _validate_protocol(
    config: Mapping[str, Any],
    tasks: Sequence[MechanismTransferTask],
) -> None:
    if tuple(str(item) for item in config["methods"]) != EXPECTED_METHODS:
        raise ValueError("Confirmation requires the frozen three-arm matrix.")
    if str(config["protocol"].get("status")) != "registered_before_execution":
        raise ValueError("Confirmation protocol must be registered before execution.")
    if not bool(config["confirmation"].get("enabled", False)):
        raise ValueError("Confirmation must be explicitly enabled.")
    if not bool(config["confirmation"].get("require_fresh_candidates_per_cell")):
        raise ValueError("Fresh candidates are required for every cell.")
    if not bool(config["confirmation"].get("prohibit_candidate_replay")):
        raise ValueError("Candidate replay must be prohibited.")
    seeds = [int(item) for item in config["confirmation"]["generation_data_seeds"]]
    prior = {int(item) for item in config["prior_development_data_seeds"]}
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Confirmation seeds must be unique and non-empty.")
    if prior.intersection(seeds):
        raise ValueError("Confirmation seeds overlap prior development seeds.")
    if len(tasks) != int(config["confirmation"]["task_count"]):
        raise ValueError("Task count differs from the registered protocol.")
    expected = len(seeds) * len(tasks)
    if expected != int(config["confirmation"]["expected_paired_cells"]):
        raise ValueError("Expected paired-cell count is inconsistent.")
    if bool(config["reporting"].get("locked_target_used_for_generation_or_selection")):
        raise ValueError("Locked targets cannot be used before final reporting.")
    if not bool(config["candidate_budget"].get("equal_source_generation_cap")):
        raise ValueError("Equal PySR and LLM candidate caps are required.")
    llm_cap = int(config["llm"]["candidates_per_call"]) * int(
        config["llm"]["maximum_generation_calls"]
    )
    if llm_cap != int(config["candidate_budget"]["maximum_source_candidates"]):
        raise ValueError("The LLM source cap differs from the frozen source budget.")


def _task_registration(item: MechanismTransferTask) -> dict[str, Any]:
    definition = item.task.definition
    return {
        "task_id": item.task.task_id,
        "mechanism_class": item.mechanism_class,
        "task_family": definition.task_family,
        "variables": dict(definition.variable_descriptions),
        "observed_ranges": {
            name: list(bounds)
            for name, bounds in definition.observed_ranges.items()
        },
        "locked_ranges": {
            name: list(bounds) for name, bounds in definition.locked_ranges.items()
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
    tasks: Sequence[MechanismTransferTask],
    run_dir: Path,
) -> dict[str, Any]:
    knowledge = load_semantic_knowledge(_project_path(config["knowledge"]["path"]))
    registrations: list[dict[str, Any]] = []
    retrievals: list[dict[str, Any]] = []
    for item in tasks:
        if bool(config["knowledge"].get("prohibit_generator_source_overlap", True)):
            assert_generator_source_disjointness(knowledge, item.generator_sources)
        blocks = retrieve_semantic_knowledge(
            knowledge,
            item.knowledge_topics,
            top_k=int(config["knowledge"]["top_k"]),
        )
        if not blocks:
            raise ValueError(f"No answer-safe knowledge for {item.task.task_id}.")
        prompt_view = knowledge_prompt_payload(knowledge, blocks)
        serialized = json.dumps(prompt_view, ensure_ascii=False).lower()
        if "oracle" in serialized or any(
            source.lower() in serialized for source in item.generator_sources
        ):
            raise ValueError(f"Generator evidence leaked into {item.task.task_id}.")
        registrations.append(_task_registration(item))
        retrievals.append(
            {
                "task_id": item.task.task_id,
                "retrieved_ids": [block.block_id for block in blocks],
                "prompt_view_sha256": _sha256(prompt_view),
            }
        )
    report = {
        "status": "passed",
        "protocol": dict(config["protocol"]),
        "confirmation_config_sha256": _sha256(config),
        "methods": list(EXPECTED_METHODS),
        "task_registrations": registrations,
        "task_registration_sha256": _sha256(registrations),
        "retrieval_audit": retrievals,
        "knowledge_audit": semantic_knowledge_audit(knowledge),
        "confirmation_seeds": list(
            config["confirmation"]["generation_data_seeds"]
        ),
        "candidate_replay_allowed": False,
        "locked_target_used_before_reporting": False,
    }
    write_json_atomic(
        run_dir / "reports" / "evidence_guided_confirmation_preflight.json",
        report,
    )
    return report


def _workflow_config(
    config: Mapping[str, Any],
    strategy: str,
) -> EvidenceGuidedRevisionConfig:
    base = EvidenceGuidedRevisionConfig.from_mapping(
        {
            "ambiguity_gate": config["ambiguity_gate"],
            "active_design": {
                **config["active_design"],
                "strategy": "predictive_disagreement",
            },
        }
    )
    if strategy == NO_ACQUISITION:
        return replace(base, maximum_new_observations=0)
    if strategy == SPACE_FILLING:
        return replace(base, acquisition_strategy="space_filling_design")
    if strategy == PREDICTIVE:
        return replace(base, acquisition_strategy="predictive_disagreement")
    raise ValueError(f"Unknown confirmation strategy: {strategy}")


def _source_labels(portfolio: Any) -> dict[str, str]:
    return {
        item.proposal_id: "+".join(item.source_labels)
        for item in portfolio.provenance
    }


def _label(labels: Mapping[str, str], proposal_id: str) -> str:
    return "baseline" if proposal_id == "baseline_no_change" else labels.get(
        proposal_id,
        "unknown",
    )


def _result_rows(
    item: MechanismTransferTask,
    data_seed: int,
    strategy: str,
    result: EvidenceGuidedRevisionResult,
    labels: Mapping[str, str],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    trajectory_rows: list[dict[str, Any]] = []
    for iteration in result.iterations:
        summary = iteration.selected_summary
        selected_id = str(summary["selected_proposal_id"])
        trajectory_rows.append(
            {
                "task_id": item.task.task_id,
                "mechanism_class": item.mechanism_class,
                "data_seed": int(data_seed),
                "strategy": strategy,
                "observation_count": iteration.observation_count,
                "selected_proposal_id": selected_id,
                "selected_source_labels": _label(labels, selected_id),
                "validation_rmse": summary["validation_rmse"],
                "locked_rmse": summary["locked_rmse"],
                "complexity": summary["complexity"],
                "stability_violations": summary["stability_violations"],
                "gate_triggered": iteration.precompression_gate.triggered,
                "gate_reason": iteration.precompression_gate.reason,
                "equivalent_candidate_count": (
                    iteration.precompression_gate.equivalent_candidate_count
                ),
                "normalized_prediction_ambiguity": (
                    iteration.precompression_gate.ambiguity
                    .normalized_maximum_pairwise_rmse
                ),
                "committee_ids": json.dumps(iteration.committee_ids),
                "optimizer_evaluations": iteration.race.total_optimizer_evaluations,
            }
        )
    acquisition_rows = [
        {
            "task_id": item.task.task_id,
            "mechanism_class": item.mechanism_class,
            "data_seed": int(data_seed),
            "strategy": strategy,
            "acquisition_number": acquisition.acquisition_number,
            "point_id": acquisition.point_id,
            **dict(acquisition.coordinates),
            "baseline": acquisition.baseline,
            "revealed_target": acquisition.revealed_target,
            "target_noise_free": acquisition.target_noise_free,
            "acquisition_score": acquisition.score,
            "score_unit": acquisition.score_unit,
        }
        for acquisition in result.acquisitions
    ]

    final_race = result.iterations[-1].race
    evaluation_by_id = {
        evaluation.proposal_id: evaluation
        for evaluation in final_race.evaluations
    }
    assessment_by_id = {
        assessment.proposal_id: assessment
        for assessment in final_race.assessments
    }
    selected_id = str(result.final_summary["selected_proposal_id"])
    equivalent_rows: list[dict[str, Any]] = []
    for proposal_id in result.final_equivalent_ids:
        if proposal_id == "baseline_no_change":
            formula = "M_base(x)"
            parameters: Mapping[str, float] = {}
            complexity = 0
            assessment = final_race.baseline_assessment
        else:
            evaluation = evaluation_by_id[proposal_id]
            formula = evaluation.formula
            parameters = evaluation.parameter_values
            complexity = evaluation.complexity
            assessment = assessment_by_id[proposal_id]
        equivalent_rows.append(
            {
                "task_id": item.task.task_id,
                "mechanism_class": item.mechanism_class,
                "data_seed": int(data_seed),
                "strategy": strategy,
                "proposal_id": proposal_id,
                "source_labels": _label(labels, proposal_id),
                "formula": formula,
                "parameter_values": json.dumps(parameters, sort_keys=True),
                "complexity": complexity,
                "robust_validation_risk": assessment.robust_validation_risk,
                "selected_representative": proposal_id == selected_id,
            }
        )
    pair_rows = [
        {
            "task_id": item.task.task_id,
            "mechanism_class": item.mechanism_class,
            "data_seed": int(data_seed),
            "strategy": strategy,
            **pair.to_row(),
        }
        for pair in result.final_prediction_pairs
    ]
    final = result.final_summary
    initial_gate = result.iterations[0].precompression_gate.triggered
    final_row = {
        "task_id": item.task.task_id,
        "mechanism_class": item.mechanism_class,
        "data_seed": int(data_seed),
        "strategy": strategy,
        "selected_proposal_id": selected_id,
        "selected_source_labels": _label(labels, selected_id),
        "selected_formula": final["selected_formula"],
        "validation_rmse": final["validation_rmse"],
        "locked_rmse": final["locked_rmse"],
        "baseline_locked_rmse": final["baseline_locked_rmse"],
        "normalized_locked_rmse": float(final["locked_rmse"])
        / max(abs(float(final["baseline_locked_rmse"])), 1.0e-12),
        "complexity": final["complexity"],
        "stability_violations": final["stability_violations"],
        "corrected_behavior_recovered": final["corrected_behavior_recovered"],
        "acquisitions_used": len(result.acquisitions),
        "initial_gate_triggered": initial_gate,
        "final_ambiguity_class": result.final_equivalence.ambiguity_class,
        "final_equivalent_candidate_count": len(result.final_equivalent_ids),
        "final_equivalent_candidate_ids": json.dumps(result.final_equivalent_ids),
        "final_normalized_maximum_pairwise_rmse": (
            result.final_equivalence.normalized_maximum_pairwise_rmse
        ),
        "ambiguity_resolved_or_prediction_equivalent": bool(
            initial_gate
            and result.final_equivalence.ambiguity_class
            != "structurally_ambiguous"
        ),
        "further_active_query_indicated": (
            result.final_equivalence.active_query_indicated
        ),
        "recommended_query": json.dumps(result.recommended_query),
        "stop_reason": result.stop_reason,
    }
    return final_row, trajectory_rows, acquisition_rows, equivalent_rows, pair_rows


def _method_summary(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("strategy", as_index=False, sort=False)
        .agg(
            cell_count=("task_id", "size"),
            median_locked_rmse=("locked_rmse", "median"),
            mean_locked_rmse=("locked_rmse", "mean"),
            median_normalized_locked_rmse=("normalized_locked_rmse", "median"),
            mean_normalized_locked_rmse=("normalized_locked_rmse", "mean"),
            behavior_recovery_count=("corrected_behavior_recovered", "sum"),
            mean_acquisitions=("acquisitions_used", "mean"),
            total_acquisitions=("acquisitions_used", "sum"),
            structural_ambiguity_count=(
                "final_ambiguity_class",
                lambda values: int((values == "structurally_ambiguous").sum()),
            ),
            physical_violation_count=("stability_violations", "sum"),
        )
    )


def _cell_contrasts(results: pd.DataFrame) -> pd.DataFrame:
    values = results.pivot(
        index=["task_id", "mechanism_class", "data_seed"],
        columns="strategy",
        values="normalized_locked_rmse",
    ).reset_index()
    values["predictive_minus_no_acquisition"] = (
        values[PREDICTIVE] - values[NO_ACQUISITION]
    )
    values["predictive_minus_space_filling"] = (
        values[PREDICTIVE] - values[SPACE_FILLING]
    )
    predictive = results.loc[
        results["strategy"].eq(PREDICTIVE),
        [
            "task_id",
            "data_seed",
            "initial_gate_triggered",
            "acquisitions_used",
            "ambiguity_resolved_or_prediction_equivalent",
            "final_ambiguity_class",
        ],
    ]
    return values.merge(predictive, on=["task_id", "data_seed"], validate="one_to_one")


def _verdict(
    comparisons: pd.DataFrame,
    results: pd.DataFrame,
    contrasts: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    primary = config["primary_analysis"]
    main = comparisons.loc[
        comparisons["comparator"].eq(str(primary["primary_comparator"]))
    ]
    if len(main) != 1:
        raise ValueError("Expected one primary paired comparison.")
    row = main.iloc[0]
    triggered = contrasts.loc[contrasts["initial_gate_triggered"]].copy()
    worsened_fraction = float(
        (triggered["predictive_minus_no_acquisition"] > 0.0).mean()
    ) if len(triggered) else 0.0
    resolved_fraction = float(
        triggered["ambiguity_resolved_or_prediction_equivalent"].mean()
    ) if len(triggered) else 1.0
    complete = len(contrasts) == int(
        config["confirmation"]["expected_paired_cells"]
    )
    mean_pass = float(row["mean_difference"]) < 0.0
    interval_pass = float(row["bootstrap_ci_upper"]) < 0.0
    pvalue_pass = float(row["one_sided_sign_flip_pvalue"]) < float(
        primary["alpha"]
    )
    worsened_pass = worsened_fraction <= float(
        primary["maximum_worsened_triggered_fraction"]
    )
    resolved_pass = resolved_fraction >= float(
        primary["minimum_resolved_triggered_fraction"]
    )
    physical_pass = int(
        results.loc[results["strategy"].eq(PREDICTIVE), "stability_violations"].sum()
    ) == 0
    success = bool(
        complete
        and mean_pass
        and interval_pass
        and pvalue_pass
        and worsened_pass
        and resolved_pass
        and physical_pass
    )
    return {
        "confirmation_success": success,
        "complete_cell_count_pass": complete,
        "negative_mean_difference_pass": mean_pass,
        "negative_cluster_ci_upper_pass": interval_pass,
        "one_sided_sign_flip_pass": pvalue_pass,
        "maximum_worsened_triggered_fraction_pass": worsened_pass,
        "minimum_resolved_triggered_fraction_pass": resolved_pass,
        "zero_physical_violations_pass": physical_pass,
        "triggered_cell_count": len(triggered),
        "worsened_triggered_fraction": worsened_fraction,
        "resolved_triggered_fraction": resolved_fraction,
    }


def run(
    config_path: str,
    run_id: str,
    *,
    llm: bool,
    llm_config: str | None,
    pysr_mode: str,
    resume: bool,
    force: bool,
    verbose: bool,
    preflight_only: bool,
) -> Path:
    config = read_yaml(config_path)
    seeds = [int(item) for item in config["confirmation"]["generation_data_seeds"]]
    registry = build_mechanism_transfer_suite(config, seed=seeds[0])
    _validate_protocol(config, registry)
    run_dir = ensure_run_dir(run_id)
    preflight = _preflight(config, registry, run_dir)
    progress_message(
        run_dir,
        "Evidence-guided confirmation preflight passed",
        verbose,
        tasks=len(registry),
        seeds=len(seeds),
        cells=len(registry) * len(seeds),
    )
    if preflight_only:
        return run_dir
    if not llm or not llm_config:
        raise ValueError("Confirmation requires --llm and --llm-config.")
    if pysr_mode != "required":
        raise ValueError("Confirmation requires --pysr required.")

    all_cells = [
        (data_seed, task_index, item)
        for data_seed in seeds
        for task_index, item in enumerate(
            build_mechanism_transfer_suite(config, seed=data_seed),
            start=1,
        )
    ]
    selected_cells = all_cells
    complete_matrix = True
    primary_output = run_dir / "metrics" / "evidence_guided_confirmation_results.csv"
    if complete_matrix and should_skip(
        run_dir,
        STAGE_NAME,
        [primary_output],
        resume,
        force,
    ):
        progress_message(run_dir, "Evidence-guided confirmation complete", verbose)
        return run_dir

    knowledge = load_semantic_knowledge(_project_path(config["knowledge"]["path"]))
    payloads: list[dict[str, Any]] = []
    for cell_index, (data_seed, task_index, item) in enumerate(
        selected_cells,
        start=1,
    ):
        checkpoint = (
            run_dir
            / "reports"
            / "confirmation_checkpoints"
            / f"seed_{data_seed}"
            / f"{item.task.task_id}.json"
        )
        if resume and not force and checkpoint.is_file():
            payload = read_json(checkpoint)
            progress_message(
                run_dir,
                "Confirmation cell restored",
                verbose,
                cell=f"{cell_index}/{len(selected_cells)}",
                task=item.task.task_id,
                seed=data_seed,
            )
            payloads.append(payload)
            continue

        progress_message(
            run_dir,
            "Fresh confirmation cell started",
            verbose,
            cell=f"{cell_index}/{len(selected_cells)}",
            task=item.task.task_id,
            seed=data_seed,
        )
        cell_dir = run_dir / "generation" / f"seed_{data_seed}"
        pysr_batch, pysr_metadata = _pysr_batch(
            item,
            config=config,
            run_dir=cell_dir,
            seed=data_seed + 1009 * task_index,
            mode=pysr_mode,
            resume=resume,
            force=force,
            verbose=verbose,
        )
        if pysr_batch is None:
            raise RuntimeError("PySR is required by the confirmation protocol.")
        blocks = retrieve_semantic_knowledge(
            knowledge,
            item.knowledge_topics,
            top_k=int(config["knowledge"]["top_k"]),
        )
        context = {
            "retrieval_mode": "answer_safe_literature_retrieval",
            **knowledge_prompt_payload(knowledge, blocks),
        }
        llm_batch, llm_metadata = _llm_batch(
            item,
            method="semantic_llm",
            context=context,
            config=config,
            run_dir=cell_dir,
            llm_config=str(llm_config),
            resume=resume,
            force=force,
            verbose=verbose,
        )
        portfolio = build_source_neutral_portfolio(
            {"pysr": pysr_batch, "semantic_llm": llm_batch},
            maximum_initial_candidates=int(
                config["candidate_budget"]["maximum_hybrid_candidates"]
            ),
        )
        labels = _source_labels(portfolio)

        def observe(
            point: pd.Series,
            _: int,
        ) -> tuple[pd.DataFrame, float | None]:
            return reveal_acquisition_observation(
                item.task,
                point,
                data_seed=data_seed,
            )

        final_rows: list[dict[str, Any]] = []
        trajectory_rows: list[dict[str, Any]] = []
        acquisition_rows: list[dict[str, Any]] = []
        equivalent_rows: list[dict[str, Any]] = []
        pair_rows: list[dict[str, Any]] = []
        optimizer_seed = data_seed + 100_003 * task_index + 31
        for strategy in EXPECTED_METHODS:
            def report_iteration(
                iteration: EvidenceGuidedIteration,
                strategy_name: str = strategy,
            ) -> None:
                progress_message(
                    run_dir,
                    "Confirmation strategy iteration complete",
                    verbose,
                    task=item.task.task_id,
                    seed=data_seed,
                    strategy=strategy_name,
                    observations=iteration.observation_count,
                    validation_rmse=(
                        f"{float(iteration.selected_summary['validation_rmse']):.6g}"
                    ),
                    equivalent=(
                        iteration.precompression_gate.equivalent_candidate_count
                    ),
                    query=iteration.precompression_gate.triggered,
                )

            result = run_evidence_guided_revision(
                item.task,
                portfolio.repairs,
                method=strategy,
                seed=optimizer_seed,
                evaluation_config=config["evaluation"],
                racing_config=config["hybrid_candidate_racing"],
                selection_config=config["risk_selection"],
                workflow_config=_workflow_config(config, strategy),
                recovery_noise_multiplier=float(
                    config["evaluation"]["recovery_noise_multiplier"]
                ),
                observation_provider=(None if strategy == NO_ACQUISITION else observe),
                iteration_callback=report_iteration,
            )
            rows = _result_rows(item, data_seed, strategy, result, labels)
            final_rows.append(rows[0])
            trajectory_rows.extend(rows[1])
            acquisition_rows.extend(rows[2])
            equivalent_rows.extend(rows[3])
            pair_rows.extend(rows[4])

        initial_frame = pd.DataFrame(trajectory_rows).loc[
            lambda frame: frame["observation_count"].eq(0)
        ]
        selected_match = initial_frame["selected_proposal_id"].nunique() == 1
        validation_spread = float(initial_frame["validation_rmse"].max()) - float(
            initial_frame["validation_rmse"].min()
        )
        locked_spread = float(initial_frame["locked_rmse"].max()) - float(
            initial_frame["locked_rmse"].min()
        )
        if not (
            selected_match
            and validation_spread <= 1.0e-12
            and locked_spread <= 1.0e-12
        ):
            raise RuntimeError(
                "Confirmation strategies do not share an identical zero-observation "
                "starting point."
            )

        payload = {
            "final_rows": final_rows,
            "trajectory_rows": trajectory_rows,
            "acquisition_rows": acquisition_rows,
            "equivalent_rows": equivalent_rows,
            "pair_rows": pair_rows,
            "portfolio_rows": [
                {
                    "task_id": item.task.task_id,
                    "mechanism_class": item.mechanism_class,
                    "data_seed": data_seed,
                    **record.to_row(),
                }
                for record in portfolio.provenance
            ],
            "generation_rows": [
                {
                    "task_id": item.task.task_id,
                    "mechanism_class": item.mechanism_class,
                    "data_seed": data_seed,
                    "source": source,
                    **metadata,
                }
                for source, metadata in (
                    ("pysr", pysr_metadata),
                    ("semantic_llm", llm_metadata),
                )
            ],
            "knowledge_log": {
                "retrieved_ids": [block.block_id for block in blocks],
                "prompt_view_sha256": _sha256(context),
                "generator_sources_excluded_from_prompt": True,
            },
            "initial_strategy_fidelity": {
                "status": "passed",
                "selected_proposal_match": selected_match,
                "maximum_validation_rmse_spread": validation_spread,
                "maximum_locked_rmse_spread": locked_spread,
            },
        }
        write_json_atomic(checkpoint, payload)
        payloads.append(payload)

    def collect(key: str) -> list[dict[str, Any]]:
        return [row for payload in payloads for row in payload[key]]

    results = pd.DataFrame(collect("final_rows"))
    trajectories = pd.DataFrame(collect("trajectory_rows"))
    acquisitions = pd.DataFrame(collect("acquisition_rows"))
    equivalents = pd.DataFrame(collect("equivalent_rows"))
    pairs = pd.DataFrame(collect("pair_rows"))
    portfolios = pd.DataFrame(collect("portfolio_rows"))
    generation = pd.DataFrame(collect("generation_rows"))
    summary = _method_summary(results)
    primary = config["primary_analysis"]
    comparisons = paired_cluster_comparisons(
        results.loc[:, ["task_id", "data_seed", "strategy", primary["endpoint"]]],
        treatment=str(primary["treatment"]),
        comparators=(
            str(primary["primary_comparator"]),
            str(primary["descriptive_comparator"]),
        ),
        value_column=str(primary["endpoint"]),
        bootstrap_replicates=int(primary["bootstrap_replicates"]),
        bootstrap_seed=int(primary["bootstrap_seed"]),
    )
    contrasts = _cell_contrasts(results)
    verdict = _verdict(comparisons, results, contrasts, config)
    predictive_equivalents = equivalents.loc[
        equivalents["strategy"].eq(PREDICTIVE)
    ]
    predictive_selected = predictive_equivalents.loc[
        predictive_equivalents["selected_representative"]
    ]
    llm_selected_count = int(
        predictive_selected["source_labels"].astype(str).str.contains(
            "semantic_llm"
        ).sum()
    )
    llm_equivalent_cell_count = int(
        predictive_equivalents.loc[
            predictive_equivalents["source_labels"].astype(str).str.contains(
                "semantic_llm"
            ),
            ["task_id", "data_seed"],
        ].drop_duplicates().shape[0]
    )

    write_table_bundle(results, primary_output.with_suffix(""))
    write_table_bundle(
        trajectories,
        run_dir / "metrics" / "evidence_guided_confirmation_trajectory",
    )
    write_table_bundle(
        acquisitions,
        run_dir / "metrics" / "evidence_guided_confirmation_acquisitions",
    )
    write_table_bundle(
        equivalents,
        run_dir / "metrics" / "evidence_guided_confirmation_equivalent_candidates",
    )
    write_table_bundle(
        pairs,
        run_dir / "metrics" / "evidence_guided_confirmation_prediction_pairs",
    )
    write_table_bundle(
        portfolios,
        run_dir / "metrics" / "evidence_guided_confirmation_portfolio",
    )
    write_table_bundle(
        generation,
        run_dir / "metrics" / "evidence_guided_confirmation_generation_costs",
    )
    write_table_bundle(
        summary,
        run_dir / "tables" / "evidence_guided_confirmation_method_summary",
    )
    write_table_bundle(
        comparisons,
        run_dir / "tables" / "evidence_guided_confirmation_paired_comparisons",
    )
    write_table_bundle(
        contrasts,
        run_dir / "tables" / "evidence_guided_confirmation_cell_contrasts",
    )

    report = {
        "status": (
            "confirmation_complete"
        ),
        "protocol": config["protocol"],
        "complete_matrix": complete_matrix,
        "fresh_candidates_generated_per_cell": True,
        "candidate_replay_used": False,
        "initial_strategy_fidelity_pass": all(
            payload["initial_strategy_fidelity"]["status"] == "passed"
            for payload in payloads
        ),
        "locked_target_used_before_reporting": False,
        "space_filling_comparison_is_descriptive": True,
        "knowledge_audit": preflight["knowledge_audit"],
        "confirmation_config_sha256": preflight[
            "confirmation_config_sha256"
        ],
        "llm_selected_representative_cell_count": llm_selected_count,
        "llm_present_in_equivalent_set_cell_count": llm_equivalent_cell_count,
        **verdict,
        "method_summary": summary.to_dict(orient="records"),
        "paired_comparisons": comparisons.to_dict(orient="records"),
        "interpretation_boundary": (
            "This confirmation tests the registered active-revision policy on fresh "
            "candidate generations and data seeds for four synthetic mechanisms. It "
            "does not establish universal discovery or field-scale constitutive validity."
        ),
    }
    write_json_atomic(
        run_dir / "reports" / "evidence_guided_confirmation_report.json",
        report,
    )
    if complete_matrix:
        mark_done(run_dir, STAGE_NAME, verdict)
    progress_message(
        run_dir,
        "Evidence-guided independent confirmation complete",
        verbose,
        cells=len(contrasts),
        success=verdict["confirmation_success"],
        output=run_dir,
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run independent confirmation of evidence-guided revision."
    )
    parser.add_argument(
        "--config",
        default=(
            "configs/semantic_evidence_guided_revision_confirmation_v1.yaml"
        ),
    )
    parser.add_argument(
        "--run-id",
        default="semantic_evidence_guided_revision_confirmation_v1",
    )
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--llm-config")
    parser.add_argument(
        "--pysr",
        choices=("auto", "required", "off"),
        default="required",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(
        args.config,
        args.run_id,
        llm=args.llm,
        llm_config=args.llm_config,
        pysr_mode=args.pysr,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
        preflight_only=args.preflight_only,
    )


if __name__ == "__main__":
    main()

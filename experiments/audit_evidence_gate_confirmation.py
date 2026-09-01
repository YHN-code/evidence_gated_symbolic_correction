from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

try:
    from experiments._bootstrap import ROOT  # noqa: F401
except ModuleNotFoundError:
    from _bootstrap import ROOT  # type: ignore  # noqa: F401

if __package__ in {None, ""}:
    from run_semantic_candidate_attribution import _load_batch  # type: ignore  # noqa: E402
    from run_semantic_evidence_guided_confirmation import (  # type: ignore  # noqa: E402
        NO_ACQUISITION,
        SPACE_FILLING,
        _result_rows,
        _source_labels,
        _workflow_config,
    )
else:
    from experiments.run_semantic_candidate_attribution import _load_batch  # noqa: E402
    from experiments.run_semantic_evidence_guided_confirmation import (  # noqa: E402
        NO_ACQUISITION,
        SPACE_FILLING,
        _result_rows,
        _source_labels,
        _workflow_config,
    )

from asrc.model_revision import (  # noqa: E402
    build_source_neutral_portfolio,
    canonical_expression_key,
    run_evidence_guided_revision,
)
from asrc.model_revision.ambiguity_active_design import (  # noqa: E402
    reveal_acquisition_observation,
)
from asrc.model_revision.confirmation import (  # noqa: E402
    paired_cluster_comparisons,
)
from asrc.model_revision.mechanism_transfer_benchmarks import (  # noqa: E402
    build_mechanism_transfer_suite,
)
from asrc.utils.io import (  # noqa: E402
    ensure_run_dir,
    project_root,
    read_json,
    read_yaml,
    runs_root,
    write_json_atomic,
    write_table_bundle,
)
from asrc.utils.progress import mark_done, progress_message, should_skip  # noqa: E402


STAGE_NAME = "evidence_gate_confirmation_audit_v1"
UNCONDITIONAL = "unconditional_space_filling"
ORACLE_AVAILABLE = "oracle_available_predictive_disagreement"


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


def _validate_config(config: Mapping[str, Any]) -> None:
    if str(config["protocol"].get("status")) != "post_confirmation_diagnostic":
        raise ValueError("This audit must remain labelled post-confirmation.")
    if not bool(config["source"].get("prohibit_candidate_regeneration")):
        raise ValueError("Frozen candidate replay is required.")
    if bool(
        config["reporting"].get(
            "locked_target_used_for_generation_fitting_acquisition_or_stopping"
        )
    ):
        raise ValueError("Locked targets cannot guide the replay.")
    if bool(config["coverage_audit"].get("report_as_algorithm_performance")):
        raise ValueError("Oracle-available results are diagnostic only.")


def _candidate_paths(source_dir: Path, seed: int, task_id: str) -> tuple[Path, Path]:
    root = source_dir / "generation" / f"seed_{seed}" / "formulas" / "semantic_attribution"
    return root / f"{task_id}_pysr.json", root / f"{task_id}_semantic_llm.json"


def _portfolio(source_dir: Path, item: Any, seed: int, source_config: Mapping[str, Any]) -> Any:
    pysr_path, llm_path = _candidate_paths(source_dir, seed, item.task.task_id)
    if not pysr_path.is_file() or not llm_path.is_file():
        raise FileNotFoundError(
            f"Frozen candidate files are missing for seed={seed}, task={item.task.task_id}."
        )
    pysr = _load_batch(pysr_path, item)
    llm = _load_batch(llm_path, item)
    return build_source_neutral_portfolio(
        {"pysr": pysr, "semantic_llm": llm},
        maximum_initial_candidates=int(
            source_config["candidate_budget"]["maximum_hybrid_candidates"]
        ),
    )


def _exact_candidate_ids(item: Any, repairs: tuple[Any, ...]) -> tuple[str, ...]:
    reference_key = canonical_expression_key(item.task.oracle_repair.expression)
    return tuple(
        repair.proposal_id
        for repair in repairs
        if canonical_expression_key(repair.expression) == reference_key
    )


def _run_one(
    item: Any,
    repairs: tuple[Any, ...],
    *,
    method: str,
    strategy: str,
    data_seed: int,
    task_index: int,
    source_config: Mapping[str, Any],
    require_gate: bool,
) -> Any:
    workflow = replace(
        _workflow_config(source_config, strategy),
        require_ambiguity_gate=require_gate,
    )

    def observe(point: pd.Series, _: int) -> tuple[pd.DataFrame, float | None]:
        return reveal_acquisition_observation(item.task, point, data_seed=data_seed)

    return run_evidence_guided_revision(
        item.task,
        repairs,
        method=method,
        seed=data_seed + 100_003 * task_index + 31,
        evaluation_config=source_config["evaluation"],
        racing_config=source_config["hybrid_candidate_racing"],
        selection_config=source_config["risk_selection"],
        workflow_config=workflow,
        recovery_noise_multiplier=float(
            source_config["evaluation"]["recovery_noise_multiplier"]
        ),
        observation_provider=observe,
    )


def _gate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("strategy", sort=False, as_index=False)
        .agg(
            cell_count=("task_id", "size"),
            mean_normalized_locked_rmse=("normalized_locked_rmse", "mean"),
            median_normalized_locked_rmse=("normalized_locked_rmse", "median"),
            mean_acquisitions=("acquisitions_used", "mean"),
            total_acquisitions=("acquisitions_used", "sum"),
            behavior_recovery_count=("corrected_behavior_recovered", "sum"),
        )
    )


def run(
    config_path: str,
    run_id: str,
    *,
    source_run_id: str | None,
    resume: bool,
    force: bool,
    verbose: bool,
) -> Path:
    config = read_yaml(config_path)
    _validate_config(config)
    source_config = read_yaml(config["source"]["config_path"])
    source_id = str(source_run_id or config["source"]["run_id"])
    source_dir = runs_root() / source_id
    source_results_path = source_dir / "metrics" / "evidence_guided_confirmation_results.csv"
    if not source_results_path.is_file():
        raise FileNotFoundError(f"Frozen confirmation results are missing: {source_results_path}")
    source_results = pd.read_csv(source_results_path)
    run_dir = ensure_run_dir(run_id)
    primary_output = run_dir / "metrics" / "gate_audit_results.csv"
    if should_skip(run_dir, STAGE_NAME, [primary_output], resume, force):
        progress_message(run_dir, "Gate and coverage audit complete", verbose)
        return run_dir

    seeds = [int(value) for value in source_config["confirmation"]["generation_data_seeds"]]
    replay_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    checkpoint_root = run_dir / "reports" / "checkpoints"
    cell_total = len(seeds) * int(source_config["confirmation"]["task_count"])
    cell_number = 0
    for data_seed in seeds:
        tasks = build_mechanism_transfer_suite(source_config, seed=data_seed)
        for task_index, item in enumerate(tasks, start=1):
            cell_number += 1
            checkpoint = checkpoint_root / f"seed_{data_seed}" / f"{item.task.task_id}.json"
            if resume and not force and checkpoint.is_file():
                payload = read_json(checkpoint)
                replay_rows.append(payload["unconditional"])
                oracle_rows.append(payload["oracle_available"])
                continue

            progress_message(
                run_dir,
                "Frozen confirmation audit cell started",
                verbose,
                cell=f"{cell_number}/{cell_total}",
                task=item.task.task_id,
                seed=data_seed,
            )
            portfolio = _portfolio(source_dir, item, data_seed, source_config)
            labels = _source_labels(portfolio)
            initial_exact = _exact_candidate_ids(item, tuple(portfolio.repairs))

            unconditional = _run_one(
                item,
                tuple(portfolio.repairs),
                method=UNCONDITIONAL,
                strategy=SPACE_FILLING,
                data_seed=data_seed,
                task_index=task_index,
                source_config=source_config,
                require_gate=False,
            )
            unconditional_row = _result_rows(
                item,
                data_seed,
                UNCONDITIONAL,
                unconditional,
                labels,
            )[0]

            oracle_repairs = tuple(portfolio.repairs)
            oracle_injected = False
            if not initial_exact:
                oracle_repairs = (*oracle_repairs, item.task.oracle_repair)
                oracle_injected = True
            oracle_labels = dict(labels)
            oracle_labels[item.task.oracle_repair.proposal_id] = "reference_audit"
            oracle_result = _run_one(
                item,
                oracle_repairs,
                method=ORACLE_AVAILABLE,
                strategy="evidence_guided_predictive_disagreement",
                data_seed=data_seed,
                task_index=task_index,
                source_config=source_config,
                require_gate=True,
            )
            oracle_selected = str(oracle_result.final_summary["selected_proposal_id"])
            exact_ids = _exact_candidate_ids(item, oracle_repairs)
            oracle_row = {
                **_result_rows(
                    item,
                    data_seed,
                    ORACLE_AVAILABLE,
                    oracle_result,
                    oracle_labels,
                )[0],
                "reference_present_initially": bool(initial_exact),
                "reference_injected": oracle_injected,
                "exact_candidate_ids": json.dumps(exact_ids),
                "exact_selected_representative": oracle_selected in exact_ids,
                "exact_retained_final": bool(
                    set(exact_ids).intersection(oracle_result.final_equivalent_ids)
                ),
            }
            source_initial = source_results.loc[
                source_results["task_id"].eq(item.task.task_id)
                & source_results["data_seed"].eq(data_seed)
                & source_results["strategy"].eq(NO_ACQUISITION)
            ]
            if len(source_initial) != 1:
                raise RuntimeError("Frozen source cell is incomplete.")
            initial_locked = float(unconditional.initial_summary["locked_rmse"])
            if abs(initial_locked - float(source_initial.iloc[0]["locked_rmse"])) > 1.0e-10:
                raise RuntimeError("Frozen replay does not reproduce the source starting point.")

            payload = {
                "unconditional": unconditional_row,
                "oracle_available": oracle_row,
                "candidate_regeneration_used": False,
                "locked_target_used_before_final_reporting": False,
            }
            write_json_atomic(checkpoint, payload)
            replay_rows.append(unconditional_row)
            oracle_rows.append(oracle_row)

    unconditional = pd.DataFrame(replay_rows)
    oracle = pd.DataFrame(oracle_rows)
    conditional = source_results.loc[
        source_results["strategy"].eq(str(config["gate_audit"]["conditional_source_method"]))
    ].copy()
    gate = pd.concat([conditional, unconditional], ignore_index=True)
    gate_summary = _gate_summary(gate)
    gate_comparison = paired_cluster_comparisons(
        gate.loc[:, ["task_id", "data_seed", "strategy", "normalized_locked_rmse"]],
        treatment=SPACE_FILLING,
        comparators=(UNCONDITIONAL,),
        value_column="normalized_locked_rmse",
        bootstrap_replicates=int(config["statistics"]["bootstrap_replicates"]),
        bootstrap_seed=int(config["statistics"]["bootstrap_seed"]),
    )
    oracle_summary = pd.DataFrame(
        [
            {
                "cell_count": len(oracle),
                "reference_present_initially_count": int(
                    oracle["reference_present_initially"].sum()
                ),
                "reference_injected_count": int(oracle["reference_injected"].sum()),
                "exact_selected_representative_count": int(
                    oracle["exact_selected_representative"].sum()
                ),
                "exact_retained_final_count": int(oracle["exact_retained_final"].sum()),
                "mean_normalized_locked_rmse": float(
                    oracle["normalized_locked_rmse"].mean()
                ),
                "mean_acquisitions": float(oracle["acquisitions_used"].mean()),
            }
        ]
    )

    write_table_bundle(gate, primary_output.with_suffix(""))
    write_table_bundle(gate_summary, run_dir / "tables" / "gate_audit_summary")
    write_table_bundle(gate_comparison, run_dir / "tables" / "gate_audit_comparison")
    write_table_bundle(oracle, run_dir / "metrics" / "oracle_coverage_audit")
    write_table_bundle(oracle_summary, run_dir / "tables" / "oracle_coverage_summary")
    report = {
        "status": "post_confirmation_diagnostic_complete",
        "source_run_id": source_id,
        "candidate_regeneration_used": False,
        "locked_target_used_before_final_reporting": False,
        "gate_summary": gate_summary.to_dict(orient="records"),
        "gate_comparison": gate_comparison.to_dict(orient="records"),
        "oracle_coverage_summary": oracle_summary.to_dict(orient="records"),
        "interpretation_boundary": (
            "The gate and oracle-available analyses replay frozen confirmation "
            "candidate pools after the primary results were known. They diagnose "
            "query efficiency and failure attribution; they are not independent "
            "confirmation results or deployable oracle-assisted performance."
        ),
    }
    write_json_atomic(run_dir / "reports" / "gate_and_coverage_audit.json", report)
    mark_done(run_dir, STAGE_NAME, report)
    progress_message(
        run_dir,
        "Gate and coverage audit complete",
        verbose,
        cells=len(unconditional),
        output=run_dir,
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay frozen confirmation pools for gate and coverage audits."
    )
    parser.add_argument(
        "--config",
        default="configs/evidence_gate_confirmation_audit_v1.yaml",
    )
    parser.add_argument(
        "--run-id",
        default="evidence_gate_confirmation_audit_v1",
    )
    parser.add_argument("--source-run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(
        args.config,
        args.run_id,
        source_run_id=args.source_run_id,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

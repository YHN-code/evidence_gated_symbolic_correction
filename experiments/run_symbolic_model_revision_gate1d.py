from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401

from asrc.model_revision.benchmarks import Gate1Task, build_gate1_suite
from asrc.model_revision.candidate_archives import (
    combined_candidate_fingerprint,
    load_development_matrix_candidates,
)
from asrc.model_revision.confirmation import (
    integrated_error_by_cell,
    paired_cluster_comparisons,
)
from asrc.model_revision.proposals import TypedRepair
from asrc.utils.io import (
    ensure_run_dir,
    read_json,
    read_yaml,
    write_json,
    write_table_bundle,
)
from asrc.utils.progress import mark_done, progress_message, should_skip

try:
    from experiments.run_symbolic_model_revision_gate1b import (
        _load_frozen_candidates,
        _sample_acquisition_pool,
        _summarize,
        _summarize_budgets,
    )
    from experiments.run_symbolic_model_revision_gate1c import (
        _maximin_initial_fit,
        _run_strategy,
    )
except ModuleNotFoundError:
    from run_symbolic_model_revision_gate1b import (
        _load_frozen_candidates,
        _sample_acquisition_pool,
        _summarize,
        _summarize_budgets,
    )
    from run_symbolic_model_revision_gate1c import (
        _maximin_initial_fit,
        _run_strategy,
    )


def _fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_protocol_config(config: dict[str, Any]) -> None:
    if config["protocol"] != {
        "name": "symbolic_model_revision_gate1d_confirmation",
        "version": "1.0.0",
        "status": "frozen_confirmation",
    }:
        raise ValueError("Gate 1D requires the frozen version 1.0.0 protocol.")
    if config["study"].get("confirmatory") is not True:
        raise ValueError("Gate 1D must be declared confirmatory.")
    if int(config["study"].get("llm_calls", -1)) != 0:
        raise ValueError("Gate 1D forbids new LLM calls.")

    seeds = [int(value) for value in config["data_seeds"]]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("Gate 1D requires at least three unique data seeds.")
    archives = config["candidate_archives"]
    source_seeds = {
        int(archives["gate1c"]["source_data_seed"]),
        int(archives["development_matrix"]["source_data_seed"]),
    }
    if source_seeds.intersection(seeds):
        raise ValueError("Gate 1D data seeds must be unseen by candidate generation.")
    for archive in archives.values():
        if archive.get("allow_new_llm_calls") is not False:
            raise ValueError("Every Gate 1D candidate archive must be frozen.")
        if archive.get("ranking_criterion") != "observed_validation_selection_score":
            raise ValueError("Gate 1D archive ranking must remain frozen.")

    task_ids = [
        *(str(value) for value in archives["gate1c"]["task_ids"]),
        *(str(value) for value in archives["development_matrix"]["task_ids"]),
    ]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Gate 1D candidate archives contain overlapping tasks.")
    transfer = {str(value) for value in config["confirmation"]["transfer_task_ids"]}
    if not transfer or not transfer.issubset(task_ids):
        raise ValueError("Gate 1D transfer tasks must be a non-empty task subset.")

    frozen = read_yaml(config["frozen_algorithm_source"])
    design_keys = (
        "initial_observation_count",
        "acquisition_pool_count",
        "maximum_new_observations",
        "outer_shell_only",
        "strategies",
    )
    for key in design_keys:
        if config["design"][key] != frozen["design"][key]:
            raise ValueError(f"Gate 1D changed frozen design setting: {key}")
    for section in ("model_evidence", "joint_information", "random_design"):
        if config[section] != frozen[section]:
            raise ValueError(f"Gate 1D changed frozen algorithm section: {section}")
    evaluation_keys = (
        "recovery_noise_multiplier",
        "primary_budget",
        "report_budgets",
    )
    for key in evaluation_keys:
        if config["evaluation"][key] != frozen["evaluation"][key]:
            raise ValueError(f"Gate 1D changed frozen evaluation setting: {key}")
    if int(config["evaluation"]["bootstrap_replicates"]) < 1000:
        raise ValueError("Gate 1D requires at least 1000 bootstrap replicates.")

    confirmation = config["confirmation"]
    strategies = set(str(value) for value in config["design"]["strategies"])
    treatment = str(confirmation["treatment"])
    if treatment not in strategies:
        raise ValueError("Gate 1D treatment is absent from the frozen strategies.")
    required_lists = (
        "primary_comparators",
        "required_mean_superiority",
        "required_final_mean_superiority",
        "required_recovery_noninferiority",
        "required_cluster_ci_superiority",
        "required_transfer_mean_superiority",
    )
    for name in required_lists:
        values = [str(value) for value in confirmation[name]]
        if not values or treatment in values or not set(values).issubset(strategies):
            raise ValueError(f"Gate 1D confirmation list is invalid: {name}")


def _load_candidate_committees(
    config: dict[str, Any],
    tasks: dict[str, Gate1Task],
) -> tuple[dict[str, tuple[TypedRepair, ...]], pd.DataFrame, dict[str, Any]]:
    archives = config["candidate_archives"]
    gate1c_ids = set(str(value) for value in archives["gate1c"]["task_ids"])
    development_ids = set(
        str(value) for value in archives["development_matrix"]["task_ids"]
    )
    gate1c_config = {
        "candidate_archive": archives["gate1c"],
        "data_seed": config["data_seeds"][0],
    }
    gate1c, gate1c_audit, gate1c_metadata = _load_frozen_candidates(
        gate1c_config,
        {task_id: tasks[task_id] for task_id in gate1c_ids},
    )
    gate1c_audit = gate1c_audit.copy()
    gate1c_audit["archive_source"] = "gate1c_development_committee"
    development, development_audit, development_metadata = (
        load_development_matrix_candidates(
            archives["development_matrix"],
            {task_id: tasks[task_id] for task_id in development_ids},
        )
    )
    combined = {**gate1c, **development}
    if set(combined) != set(tasks):
        raise ValueError("Gate 1D candidate committees do not cover every task.")
    audit = pd.concat([gate1c_audit, development_audit], ignore_index=True)
    metadata = {
        "gate1c": gate1c_metadata,
        "development_matrix": development_metadata,
        "combined_candidate_set_sha256": combined_candidate_fingerprint(combined),
    }
    return combined, audit, metadata


def _cell_checkpoint_path(run_dir: Path, task_id: str, data_seed: int) -> Path:
    return (
        run_dir
        / "checkpoints"
        / "gate1d_cells"
        / f"{task_id}_seed_{int(data_seed)}.json"
    )


def _run_cell(
    task: Gate1Task,
    repairs: tuple[TypedRepair, ...],
    *,
    task_index: int,
    data_seed: int,
    config: dict[str, Any],
    gate1_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cell_config = deepcopy(config)
    cell_config["data_seed"] = int(data_seed)
    initial_fit = _maximin_initial_fit(
        task,
        int(config["design"]["initial_observation_count"]),
    )
    pool = _sample_acquisition_pool(
        task,
        count=int(config["design"]["acquisition_pool_count"]),
        seed=int(data_seed) + 2003 * task_index,
        outer_shell_only=bool(config["design"]["outer_shell_only"]),
    )
    rng = np.random.default_rng(int(data_seed) + 5003 * task_index + 1)
    potential_targets = pool["target_noise_free"].to_numpy(float) + rng.normal(
        0.0,
        task.definition.noise_std,
        len(pool),
    )
    trajectory_rows: list[dict[str, Any]] = []
    acquisition_rows: list[dict[str, Any]] = []
    for strategy in config["design"]["strategies"]:
        trajectory, acquisitions = _run_strategy(
            task,
            repairs,
            pool,
            initial_fit,
            potential_targets,
            strategy=str(strategy),
            repetition=1,
            config=cell_config,
            gate1_config=gate1_config,
        )
        for row in trajectory:
            row["data_seed"] = int(data_seed)
        for row in acquisitions:
            row["data_seed"] = int(data_seed)
        trajectory_rows.extend(trajectory)
        acquisition_rows.extend(acquisitions)
    return trajectory_rows, acquisition_rows


def _first_recovery_table(trajectory: pd.DataFrame, maximum_budget: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["task_id", "data_seed", "strategy"]
    for key, group in trajectory.groupby(keys, sort=True):
        recovered = group.loc[group["behavior_recovered"].astype(bool)]
        first = (
            int(recovered["new_observation_count"].min())
            if not recovered.empty
            else int(maximum_budget) + 1
        )
        rows.append({**dict(zip(keys, key)), "first_recovery_budget": first})
    return pd.DataFrame(rows)


def _strategy_means(frame: pd.DataFrame, value_column: str) -> dict[str, float]:
    return {
        str(strategy): float(group[value_column].mean())
        for strategy, group in frame.groupby("strategy", sort=True)
    }


def _confirmation_decision(
    config: dict[str, Any],
    integrated: pd.DataFrame,
    final: pd.DataFrame,
    primary_comparisons: pd.DataFrame,
) -> dict[str, Any]:
    contract = config["confirmation"]
    treatment = str(contract["treatment"])
    integrated_means = _strategy_means(integrated, "integrated_locked_rmse")
    final_means = _strategy_means(final, "locked_rmse")
    final_recovery = {
        str(strategy): int(group["behavior_recovered"].astype(bool).sum())
        for strategy, group in final.groupby("strategy", sort=True)
    }
    transfer_ids = set(str(value) for value in contract["transfer_task_ids"])
    transfer_means = _strategy_means(
        integrated.loc[integrated["task_id"].isin(transfer_ids)],
        "integrated_locked_rmse",
    )

    checks: dict[str, bool] = {}
    for comparator in contract["required_mean_superiority"]:
        checks[f"primary_mean_below_{comparator}"] = (
            integrated_means[treatment] < integrated_means[str(comparator)]
        )
    for comparator in contract["required_final_mean_superiority"]:
        checks[f"final_mean_below_{comparator}"] = (
            final_means[treatment] < final_means[str(comparator)]
        )
    for comparator in contract["required_recovery_noninferiority"]:
        checks[f"recovery_not_below_{comparator}"] = (
            final_recovery[treatment] >= final_recovery[str(comparator)]
        )
    comparison_index = primary_comparisons.set_index("comparator")
    for comparator in contract["required_cluster_ci_superiority"]:
        checks[f"cluster_ci_below_zero_vs_{comparator}"] = (
            float(comparison_index.loc[str(comparator), "bootstrap_ci_upper"]) < 0.0
        )
    for comparator in contract["required_transfer_mean_superiority"]:
        checks[f"transfer_mean_below_{comparator}"] = (
            transfer_means[treatment] < transfer_means[str(comparator)]
        )
    return {
        "gate1d_confirmation_success": bool(all(checks.values())),
        "checks": checks,
        "integrated_mean_by_strategy": integrated_means,
        "final_mean_by_strategy": final_means,
        "final_recovery_count_by_strategy": final_recovery,
        "transfer_integrated_mean_by_strategy": transfer_means,
    }


def run(
    config_path: str,
    run_id: str,
    *,
    force: bool,
    resume: bool,
    verbose: bool,
) -> Path:
    run_dir = ensure_run_dir(run_id)
    trajectory_output = run_dir / "metrics" / "gate1d_confirmation_trajectory.csv"
    summary_output = run_dir / "metrics" / "gate1d_confirmation_summary.csv"
    stage = "symbolic_model_revision_gate1d_confirmation"
    if should_skip(
        run_dir,
        stage,
        [trajectory_output, summary_output],
        resume,
        force,
    ):
        progress_message(run_dir, "Gate 1D confirmation already complete", verbose)
        return run_dir

    config = read_yaml(config_path)
    _validate_protocol_config(config)
    protocol_fingerprint = _fingerprint(config)
    gate1_config = read_yaml(config["gate1_config"])
    task_ids = {
        *(str(value) for value in config["candidate_archives"]["gate1c"]["task_ids"]),
        *(str(value) for value in config["candidate_archives"]["development_matrix"]["task_ids"]),
    }
    first_seed = int(config["data_seeds"][0])
    archive_tasks = {
        task.task_id: task
        for task in build_gate1_suite(gate1_config, seed=first_seed)
        if task.task_id in task_ids
    }
    if set(archive_tasks) != task_ids:
        raise ValueError("Gate 1D could not build every frozen task.")
    candidates, candidate_audit, candidate_metadata = _load_candidate_committees(
        config,
        archive_tasks,
    )
    write_table_bundle(
        candidate_audit,
        run_dir / "metrics" / "gate1d_frozen_candidate_archive",
    )
    write_json(
        run_dir / "reports" / "gate1d_candidate_archive.json",
        candidate_metadata,
    )
    write_json(run_dir / "reports" / "gate1d_protocol_snapshot.json", config)
    progress_message(
        run_dir,
        "Gate 1D frozen confirmation started",
        verbose,
        tasks=len(task_ids),
        seeds=len(config["data_seeds"]),
        strategies=len(config["design"]["strategies"]),
        llm_calls=0,
    )

    trajectory_rows: list[dict[str, Any]] = []
    acquisition_rows: list[dict[str, Any]] = []
    for seed_index, data_seed in enumerate(config["data_seeds"], start=1):
        seed = int(data_seed)
        tasks = {
            task.task_id: task
            for task in build_gate1_suite(gate1_config, seed=seed)
            if task.task_id in task_ids
        }
        for task_index, task_id in enumerate(sorted(task_ids), start=1):
            checkpoint_path = _cell_checkpoint_path(run_dir, task_id, seed)
            if resume and not force and checkpoint_path.exists():
                checkpoint = read_json(checkpoint_path)
                if checkpoint.get("protocol_fingerprint") != protocol_fingerprint:
                    raise ValueError("Gate 1D checkpoint belongs to another protocol.")
                trajectory_rows.extend(checkpoint["trajectory"])
                acquisition_rows.extend(checkpoint["acquisitions"])
                progress_message(
                    run_dir,
                    "Gate 1D cell resumed",
                    verbose,
                    task=task_id,
                    data_seed=seed,
                )
                continue
            progress_message(
                run_dir,
                "Gate 1D cell started",
                verbose,
                task=task_id,
                data_seed=seed,
                seed_index=f"{seed_index}/{len(config['data_seeds'])}",
            )
            trajectory, acquisitions = _run_cell(
                tasks[task_id],
                candidates[task_id],
                task_index=task_index,
                data_seed=seed,
                config=config,
                gate1_config=gate1_config,
            )
            checkpoint = {
                "protocol_fingerprint": protocol_fingerprint,
                "candidate_set_sha256": candidate_metadata[
                    "combined_candidate_set_sha256"
                ],
                "task_id": task_id,
                "data_seed": seed,
                "trajectory": trajectory,
                "acquisitions": acquisitions,
            }
            write_json(checkpoint_path, checkpoint)
            trajectory_rows.extend(trajectory)
            acquisition_rows.extend(acquisitions)
            progress_message(
                run_dir,
                "Gate 1D cell complete",
                verbose,
                task=task_id,
                data_seed=seed,
            )

    trajectory = pd.DataFrame(trajectory_rows)
    acquisitions = pd.DataFrame(acquisition_rows)
    primary_budget = int(config["evaluation"]["primary_budget"])
    final = trajectory.loc[
        trajectory["new_observation_count"].eq(primary_budget)
    ].copy()
    integrated = integrated_error_by_cell(trajectory)
    summary = _summarize(trajectory, primary_budget, by_task=False)
    task_summary = _summarize(trajectory, primary_budget, by_task=True)
    budget_summary = _summarize_budgets(
        trajectory,
        [int(value) for value in config["evaluation"]["report_budgets"]],
    )
    integrated_summary = (
        integrated.groupby("strategy", sort=True)["integrated_locked_rmse"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_integrated_locked_rmse",
                "median": "median_integrated_locked_rmse",
                "std": "std_integrated_locked_rmse",
                "count": "cell_count",
            }
        )
    )
    treatment = str(config["confirmation"]["treatment"])
    comparators = [
        strategy
        for strategy in config["design"]["strategies"]
        if str(strategy) != treatment
    ]
    primary_comparisons = paired_cluster_comparisons(
        integrated,
        treatment=treatment,
        comparators=comparators,
        value_column="integrated_locked_rmse",
        bootstrap_replicates=int(config["evaluation"]["bootstrap_replicates"]),
        bootstrap_seed=int(config["evaluation"]["bootstrap_seed"]),
    )
    final_comparisons = paired_cluster_comparisons(
        final,
        treatment=treatment,
        comparators=comparators,
        value_column="locked_rmse",
        bootstrap_replicates=int(config["evaluation"]["bootstrap_replicates"]),
        bootstrap_seed=int(config["evaluation"]["bootstrap_seed"]) + 100,
    )
    first_recovery = _first_recovery_table(
        trajectory,
        int(config["design"]["maximum_new_observations"]),
    )
    decision = _confirmation_decision(
        config,
        integrated,
        final,
        primary_comparisons,
    )

    write_table_bundle(trajectory, trajectory_output.with_suffix(""))
    write_table_bundle(
        acquisitions,
        run_dir / "metrics" / "gate1d_acquisition_trace",
    )
    write_table_bundle(summary, summary_output.with_suffix(""))
    write_table_bundle(
        task_summary,
        run_dir / "metrics" / "gate1d_confirmation_summary_by_task",
    )
    write_table_bundle(
        budget_summary,
        run_dir / "metrics" / "gate1d_confirmation_budget_summary",
    )
    write_table_bundle(
        integrated,
        run_dir / "metrics" / "gate1d_integrated_error_by_cell",
    )
    write_table_bundle(
        integrated_summary,
        run_dir / "metrics" / "gate1d_integrated_error_summary",
    )
    write_table_bundle(
        primary_comparisons,
        run_dir / "metrics" / "gate1d_primary_paired_comparisons",
    )
    write_table_bundle(
        final_comparisons,
        run_dir / "metrics" / "gate1d_final_paired_comparisons",
    )
    write_table_bundle(
        first_recovery,
        run_dir / "metrics" / "gate1d_first_recovery_budget",
    )
    report = {
        "protocol": config["protocol"],
        "protocol_fingerprint": protocol_fingerprint,
        "candidate_archive": candidate_metadata,
        "data_seeds": [int(value) for value in config["data_seeds"]],
        "task_ids": sorted(task_ids),
        "llm_calls": 0,
        "primary_endpoint": config["confirmation"]["primary_endpoint"],
        "final_endpoint": config["confirmation"]["final_endpoint"],
        **decision,
    }
    write_json(run_dir / "reports" / "gate1d_confirmation_report.json", report)
    mark_done(run_dir, stage, decision)
    progress_message(
        run_dir,
        "Gate 1D frozen confirmation complete",
        verbose,
        success=decision["gate1d_confirmation_success"],
        output=run_dir,
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen Gate 1D multi-seed confirmation."
    )
    parser.add_argument(
        "--config", default="configs/symbolic_model_revision_gate1d.yaml"
    )
    parser.add_argument(
        "--run-id", default="gate1d_joint_information_confirmation_v1"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(
        args.config,
        args.run_id,
        force=args.force,
        resume=args.resume,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

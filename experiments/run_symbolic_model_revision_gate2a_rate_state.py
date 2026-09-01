from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401

from asrc.model_revision.candidate_archives import combined_candidate_fingerprint
from asrc.model_revision.confirmation import (
    integrated_error_by_cell,
    paired_cluster_comparisons,
)
from asrc.model_revision.external_rate_state import (
    build_rate_state_candidates,
    build_rate_state_suite,
)
from asrc.model_revision.proposals import (
    TypedRepair,
    canonical_expression_key,
)
from asrc.model_revision.ast import repair_to_text
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
        _summarize,
        _summarize_budgets,
    )
    from experiments.run_symbolic_model_revision_gate1d import (
        _fingerprint,
        _first_recovery_table,
        _run_cell,
        _strategy_means,
    )
except ModuleNotFoundError:
    from run_symbolic_model_revision_gate1b import _summarize, _summarize_budgets
    from run_symbolic_model_revision_gate1d import (
        _fingerprint,
        _first_recovery_table,
        _run_cell,
        _strategy_means,
    )


def _validate_protocol_config(config: dict[str, Any]) -> None:
    if config["protocol"] != {
        "name": "symbolic_model_revision_gate2a_rate_state_external_confirmation",
        "version": "1.0.0",
        "status": "frozen_confirmation",
    }:
        raise ValueError("Gate 2A requires the frozen version 1.0.0 protocol.")
    if config["study"].get("confirmatory") is not True:
        raise ValueError("Gate 2A must be declared confirmatory.")
    if int(config["study"].get("llm_calls", -1)) != 0:
        raise ValueError("Gate 2A forbids LLM calls.")
    committee = config["candidate_committee"]
    if committee != {
        "source": "a_priori_literature_and_generic_competitors",
        "response_independent": True,
        "allow_llm_calls": False,
        "families_per_task": 5,
    }:
        raise ValueError("Gate 2A candidate committee contract changed.")
    seeds = [int(value) for value in config["data_seeds"]]
    if len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError("Gate 2A requires at least three unique data seeds.")

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
            raise ValueError(f"Gate 2A changed frozen design setting: {key}")
    for section in ("model_evidence", "joint_information", "random_design"):
        if config[section] != frozen[section]:
            raise ValueError(f"Gate 2A changed frozen algorithm section: {section}")
    for key in ("recovery_noise_multiplier", "primary_budget", "report_budgets"):
        if config["evaluation"][key] != frozen["evaluation"][key]:
            raise ValueError(f"Gate 2A changed frozen evaluation setting: {key}")
    if int(config["evaluation"]["bootstrap_replicates"]) < 1000:
        raise ValueError("Gate 2A requires at least 1000 bootstrap replicates.")

    mechanisms = config["mechanisms"]
    if mechanisms["state_evolution"]["reference_family"] != "ruina_slip":
        raise ValueError("Gate 2A state-evolution reference changed.")
    if mechanisms["state_evolution"]["baseline_family"] != "dieterich_aging":
        raise ValueError("Gate 2A state-evolution baseline changed.")
    if (
        mechanisms["friction_surface"]["reference_family"]
        != "dieterich_ruina_logarithmic"
    ):
        raise ValueError("Gate 2A friction reference changed.")
    if mechanisms["friction_surface"]["baseline_family"] != "direct_effect_only":
        raise ValueError("Gate 2A friction baseline changed.")

    strategies = set(str(value) for value in config["design"]["strategies"])
    contract = config["confirmation"]
    treatment = str(contract["treatment"])
    if treatment not in strategies:
        raise ValueError("Gate 2A treatment is absent from frozen strategies.")
    for name in (
        "required_mean_superiority",
        "required_final_mean_superiority",
        "required_recovery_noninferiority",
        "required_exact_recovery_noninferiority",
        "required_cluster_ci_superiority",
    ):
        values = [str(value) for value in contract[name]]
        if not values or treatment in values or not set(values).issubset(strategies):
            raise ValueError(f"Gate 2A confirmation list is invalid: {name}")


def _candidate_committees(
    tasks: dict[str, Any],
) -> tuple[dict[str, tuple[TypedRepair, ...]], pd.DataFrame, dict[str, Any]]:
    candidates = {
        task_id: build_rate_state_candidates(task)
        for task_id, task in sorted(tasks.items())
    }
    audit_rows = []
    for task_id, repairs in candidates.items():
        oracle_key = canonical_expression_key(tasks[task_id].oracle_repair.expression)
        for index, repair in enumerate(repairs, start=1):
            expression_key = canonical_expression_key(repair.expression)
            audit_rows.append(
                {
                    "task_id": task_id,
                    "candidate_order": index,
                    "proposal_id": repair.proposal_id,
                    "source": repair.source,
                    "formula": repair_to_text(repair),
                    "expression_key": expression_key,
                    "exact_oracle_expression": expression_key == oracle_key,
                    "response_independent": True,
                }
            )
    fingerprint = combined_candidate_fingerprint(candidates)
    return candidates, pd.DataFrame(audit_rows), {
        "source": "a_priori_literature_and_generic_competitors",
        "response_independent": True,
        "llm_calls": 0,
        "candidate_set_sha256": fingerprint,
        "candidate_count_by_task": {
            task_id: len(repairs) for task_id, repairs in sorted(candidates.items())
        },
    }


def _checkpoint_path(run_dir: Path, task_id: str, data_seed: int) -> Path:
    return (
        run_dir
        / "checkpoints"
        / "gate2a_rate_state_cells"
        / f"{task_id}_seed_{int(data_seed)}.json"
    )


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
    recovery = {
        str(strategy): int(group["behavior_recovered"].astype(bool).sum())
        for strategy, group in final.groupby("strategy", sort=True)
    }
    exact = {
        str(strategy): int(group["exact_expression_match"].astype(bool).sum())
        for strategy, group in final.groupby("strategy", sort=True)
    }
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
            recovery[treatment] >= recovery[str(comparator)]
        )
    for comparator in contract["required_exact_recovery_noninferiority"]:
        checks[f"exact_recovery_not_below_{comparator}"] = (
            exact[treatment] >= exact[str(comparator)]
        )
    comparison_index = primary_comparisons.set_index("comparator")
    for comparator in contract["required_cluster_ci_superiority"]:
        checks[f"cluster_ci_below_zero_vs_{comparator}"] = (
            float(comparison_index.loc[str(comparator), "bootstrap_ci_upper"]) < 0.0
        )
    return {
        "gate2a_confirmation_success": bool(all(checks.values())),
        "checks": checks,
        "integrated_mean_by_strategy": integrated_means,
        "final_mean_by_strategy": final_means,
        "final_recovery_count_by_strategy": recovery,
        "final_exact_recovery_count_by_strategy": exact,
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
    trajectory_output = run_dir / "metrics" / "gate2a_rate_state_trajectory.csv"
    summary_output = run_dir / "metrics" / "gate2a_rate_state_summary.csv"
    stage = "symbolic_model_revision_gate2a_rate_state_confirmation"
    if should_skip(
        run_dir,
        stage,
        [trajectory_output, summary_output],
        resume,
        force,
    ):
        progress_message(run_dir, "Gate 2A confirmation already complete", verbose)
        return run_dir

    config = read_yaml(config_path)
    _validate_protocol_config(config)
    protocol_fingerprint = _fingerprint(config)
    gate1_config = read_yaml(config["gate1_config"])
    first_seed = int(config["data_seeds"][0])
    archive_tasks = {
        task.task_id: task for task in build_rate_state_suite(config, seed=first_seed)
    }
    candidates, candidate_audit, candidate_metadata = _candidate_committees(
        archive_tasks
    )
    write_table_bundle(
        candidate_audit,
        run_dir / "metrics" / "gate2a_rate_state_candidate_committee",
    )
    write_json(
        run_dir / "reports" / "gate2a_rate_state_candidate_committee.json",
        candidate_metadata,
    )
    write_json(run_dir / "reports" / "gate2a_protocol_snapshot.json", config)
    progress_message(
        run_dir,
        "Gate 2A rate-state confirmation started",
        verbose,
        mechanisms=len(archive_tasks),
        seeds=len(config["data_seeds"]),
        strategies=len(config["design"]["strategies"]),
        llm_calls=0,
    )

    trajectory_rows: list[dict[str, Any]] = []
    acquisition_rows: list[dict[str, Any]] = []
    for seed_index, data_seed in enumerate(config["data_seeds"], start=1):
        seed = int(data_seed)
        tasks = {
            task.task_id: task for task in build_rate_state_suite(config, seed=seed)
        }
        for task_index, task_id in enumerate(sorted(tasks), start=1):
            checkpoint_path = _checkpoint_path(run_dir, task_id, seed)
            if resume and not force and checkpoint_path.exists():
                checkpoint = read_json(checkpoint_path)
                if checkpoint.get("protocol_fingerprint") != protocol_fingerprint:
                    raise ValueError("Gate 2A checkpoint belongs to another protocol.")
                trajectory_rows.extend(checkpoint["trajectory"])
                acquisition_rows.extend(checkpoint["acquisitions"])
                progress_message(
                    run_dir,
                    "Gate 2A cell resumed",
                    verbose,
                    task=task_id,
                    data_seed=seed,
                )
                continue
            progress_message(
                run_dir,
                "Gate 2A cell started",
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
            write_json(
                checkpoint_path,
                {
                    "protocol_fingerprint": protocol_fingerprint,
                    "candidate_set_sha256": candidate_metadata[
                        "candidate_set_sha256"
                    ],
                    "task_id": task_id,
                    "data_seed": seed,
                    "trajectory": trajectory,
                    "acquisitions": acquisitions,
                },
            )
            trajectory_rows.extend(trajectory)
            acquisition_rows.extend(acquisitions)
            progress_message(
                run_dir,
                "Gate 2A cell complete",
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
        str(strategy)
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
        run_dir / "metrics" / "gate2a_rate_state_acquisition_trace",
    )
    write_table_bundle(summary, summary_output.with_suffix(""))
    write_table_bundle(
        task_summary,
        run_dir / "metrics" / "gate2a_rate_state_summary_by_task",
    )
    write_table_bundle(
        budget_summary,
        run_dir / "metrics" / "gate2a_rate_state_budget_summary",
    )
    write_table_bundle(
        integrated,
        run_dir / "metrics" / "gate2a_rate_state_integrated_error_by_cell",
    )
    write_table_bundle(
        integrated_summary,
        run_dir / "metrics" / "gate2a_rate_state_integrated_error_summary",
    )
    write_table_bundle(
        primary_comparisons,
        run_dir / "metrics" / "gate2a_rate_state_primary_comparisons",
    )
    write_table_bundle(
        final_comparisons,
        run_dir / "metrics" / "gate2a_rate_state_final_comparisons",
    )
    write_table_bundle(
        first_recovery,
        run_dir / "metrics" / "gate2a_rate_state_first_recovery_budget",
    )
    report = {
        "protocol": config["protocol"],
        "protocol_fingerprint": protocol_fingerprint,
        "candidate_committee": candidate_metadata,
        "data_seeds": [int(value) for value in config["data_seeds"]],
        "task_ids": sorted(archive_tasks),
        "llm_calls": 0,
        "interpretation": config["study"]["interpretation"],
        **decision,
    }
    write_json(
        run_dir / "reports" / "gate2a_rate_state_confirmation_report.json",
        report,
    )
    mark_done(run_dir, stage, decision)
    progress_message(
        run_dir,
        "Gate 2A rate-state confirmation complete",
        verbose,
        success=decision["gate2a_confirmation_success"],
        output=run_dir,
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen Gate 2A rate-and-state friction confirmation."
    )
    parser.add_argument(
        "--config",
        default="configs/symbolic_model_revision_gate2a_rate_state.yaml",
    )
    parser.add_argument(
        "--run-id", default="gate2a_rate_state_external_confirmation_v1"
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

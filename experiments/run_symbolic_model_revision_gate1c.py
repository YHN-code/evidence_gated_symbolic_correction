from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401

from asrc.model_revision.acquisition import (
    joint_structure_parameter_information_gain,
    local_parameter_information_gain,
    local_parameter_posterior_covariance,
    score_acquisition_pool,
    select_acquisition_index,
)
from asrc.model_revision.benchmarks import Gate1Task, build_gate1_suite
from asrc.model_revision.evaluation import repair_parameter_jacobian
from asrc.utils.io import (
    ensure_run_dir,
    read_yaml,
    write_json,
    write_table_bundle,
)
from asrc.utils.progress import mark_done, progress_message, should_skip
try:
    from experiments.run_symbolic_model_revision_gate1b import (
        FittedCommittee,
        _committee_predictions,
        _evaluate_selected,
        _fit_committee,
        _load_frozen_candidates,
        _sample_acquisition_pool,
        _summarize,
        _summarize_budgets,
    )
except ModuleNotFoundError:
    from run_symbolic_model_revision_gate1b import (
        FittedCommittee,
        _committee_predictions,
        _evaluate_selected,
        _fit_committee,
        _load_frozen_candidates,
        _sample_acquisition_pool,
        _summarize,
        _summarize_budgets,
    )


STRATEGIES = {
    "space_filling_design",
    "random_design",
    "model_information_gain",
    "parameter_information_gain",
    "joint_structure_parameter_information_gain",
}


def _validate_protocol_config(config: dict[str, Any]) -> None:
    archive = config["candidate_archive"]
    design = config["design"]
    evidence = config["model_evidence"]
    joint = config["joint_information"]
    evaluation = config["evaluation"]
    if archive.get("allow_new_llm_calls") is not False:
        raise ValueError("Gate 1C forbids new LLM calls.")
    if int(archive["source_data_seed"]) == int(config["data_seed"]):
        raise ValueError("Gate 1C requires a data seed unseen by candidate generation.")
    if archive.get("ranking_criterion") != "observed_validation_selection_score":
        raise ValueError("Unsupported frozen-candidate ranking criterion.")
    if evidence != {
        "weight_criterion": "bayesian_information_criterion",
        "bic_complexity": "fitted_parameter_count",
        "retain_baseline_null_model": True,
    }:
        raise ValueError("Gate 1C requires the frozen BIC evidence contract.")
    expected_joint = {
        "decomposition": "chain_rule_model_identity_plus_conditional_parameters",
        "model_information_estimator": "gauss_hermite_scalar_gaussian",
        "quadrature_order": joint["quadrature_order"],
        "parameter_posterior": "local_gaussian_laplace",
        "parameter_prior": "uniform_variance_from_fit_bounds",
    }
    if joint != expected_joint or int(joint["quadrature_order"]) < 4:
        raise ValueError("Gate 1C joint-information contract is invalid.")
    strategies = [str(value) for value in design["strategies"]]
    if len(strategies) != len(set(strategies)) or set(strategies) != STRATEGIES:
        raise ValueError("Gate 1C requires each frozen acquisition strategy once.")
    initial_count = int(design["initial_observation_count"])
    if initial_count < 2:
        raise ValueError("Gate 1C requires at least two initial observations.")
    maximum_budget = int(design["maximum_new_observations"])
    primary_budget = int(evaluation["primary_budget"])
    report_budgets = [int(value) for value in evaluation["report_budgets"]]
    if primary_budget > maximum_budget or primary_budget not in report_budgets:
        raise ValueError("The primary budget must be a reported attainable budget.")
    if any(value < 0 or value > maximum_budget for value in report_budgets):
        raise ValueError("All report budgets must lie within the acquisition budget.")


def _maximin_initial_fit(task: Gate1Task, count: int) -> pd.DataFrame:
    fit = (
        task.observed.loc[task.observed["partition"].eq("fit")]
        .copy()
        .reset_index(drop=True)
    )
    if count > len(fit):
        raise ValueError("Initial observation count exceeds the available fit data.")
    names = list(task.variables)
    lower = np.asarray([task.definition.observed_ranges[name][0] for name in names])
    upper = np.asarray([task.definition.observed_ranges[name][1] for name in names])
    normalized = (fit[names].to_numpy(float) - lower) / (upper - lower)
    center_distance = np.linalg.norm(normalized - 0.5, axis=1)
    selected = [int(np.argmin(center_distance))]
    remaining = set(range(len(fit))).difference(selected)
    while len(selected) < count:
        next_index = min(
            remaining,
            key=lambda index: (
                -float(
                    np.min(
                        np.linalg.norm(
                            normalized[index] - normalized[selected], axis=1
                        )
                    )
                ),
                index,
            ),
        )
        selected.append(next_index)
        remaining.remove(next_index)
    return fit.iloc[selected].copy().reset_index(drop=True)


def _information_components(
    task: Gate1Task,
    committee: FittedCommittee,
    fit: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    noise_std: float,
    prior_variance: float,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions = _committee_predictions(task, committee, pool)
    parameter_information = np.zeros_like(predictions)
    predictive_std = np.full_like(predictions, float(noise_std))
    for index, (repair, parameters) in enumerate(
        zip(committee.repairs, committee.parameters)
    ):
        if repair is None:
            continue
        fit_names, fit_jacobian = repair_parameter_jacobian(
            task, repair, fit, parameters
        )
        pool_names, pool_jacobian = repair_parameter_jacobian(
            task, repair, pool, parameters
        )
        if fit_names != pool_names:
            raise RuntimeError("Parameter Jacobian order changed between designs.")
        covariance = local_parameter_posterior_covariance(
            fit_jacobian,
            noise_std,
            prior_variance,
        )
        information, standard_deviation = local_parameter_information_gain(
            pool_jacobian,
            covariance,
            noise_std,
        )
        parameter_information[index] = information
        predictive_std[index] = standard_deviation
    structural, parameter, joint = joint_structure_parameter_information_gain(
        predictions,
        predictive_std,
        parameter_information,
        committee.weights,
        quadrature_order=quadrature_order,
    )
    return (
        predictions,
        predictive_std,
        parameter_information,
        structural,
        parameter,
        joint,
    )


def _strategy_scores(
    strategy: str,
    *,
    task: Gate1Task,
    committee: FittedCommittee,
    fit: pd.DataFrame,
    pool: pd.DataFrame,
    config: dict[str, Any],
    gate1_config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    lower_bound = float(gate1_config["evaluation"]["parameter_lower_bound"])
    upper_bound = float(gate1_config["evaluation"]["parameter_upper_bound"])
    prior_variance = (upper_bound - lower_bound) ** 2 / 12.0
    quadrature_order = int(config["joint_information"]["quadrature_order"])
    (
        predictions,
        predictive_std,
        parameter_information,
        structural,
        parameter,
        joint,
    ) = _information_components(
        task,
        committee,
        fit,
        pool,
        noise_std=task.definition.noise_std,
        prior_variance=prior_variance,
        quadrature_order=quadrature_order,
    )
    components = {
        "model_information_nats": structural,
        "parameter_information_nats": parameter,
        "joint_information_nats": joint,
    }
    if strategy == "model_information_gain":
        return structural, {"score_unit": "nat"}, components
    if strategy == "parameter_information_gain":
        return parameter, {"score_unit": "nat"}, components
    if strategy == "joint_structure_parameter_information_gain":
        return joint, {"score_unit": "nat"}, components

    variable_names = list(task.variables)
    lower = np.asarray(
        [task.definition.locked_ranges[name][0] for name in variable_names]
    )
    upper = np.asarray(
        [task.definition.locked_ranges[name][1] for name in variable_names]
    )
    scores, metadata = score_acquisition_pool(
        strategy,
        candidate_design=pool[variable_names].to_numpy(float),
        observed_design=fit[variable_names].to_numpy(float),
        prediction_matrix=predictions,
        model_weights=committee.weights,
        lower_bounds=lower,
        upper_bounds=upper,
        noise_std=task.definition.noise_std,
        seed=seed,
        information_gain_quadrature_order=quadrature_order,
    )
    return scores, metadata, components


def _run_strategy(
    task: Gate1Task,
    repairs: tuple[Any, ...],
    pool: pd.DataFrame,
    initial_fit: pd.DataFrame,
    potential_targets: np.ndarray,
    *,
    strategy: str,
    repetition: int,
    config: dict[str, Any],
    gate1_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variable_names = list(task.variables)
    fit = initial_fit.copy().reset_index(drop=True)
    target_by_point = dict(
        zip(pool["point_id"].astype(str), np.asarray(potential_targets, dtype=float))
    )
    remaining = pool.drop(columns=["target_noise_free"]).copy().reset_index(drop=True)
    trajectory_rows: list[dict[str, Any]] = []
    acquisition_rows: list[dict[str, Any]] = []
    maximum_budget = int(config["design"]["maximum_new_observations"])
    for budget in range(maximum_budget + 1):
        active_task, committee = _fit_committee(
            task,
            repairs,
            fit,
            gate1_config,
            seed=int(config["data_seed"]) + 1000 * repetition + budget,
        )
        result = _evaluate_selected(
            task,
            committee,
            float(config["evaluation"]["recovery_noise_multiplier"]),
        )
        trajectory_rows.append(
            {
                "task_id": task.task_id,
                "strategy": strategy,
                "repetition": repetition,
                "new_observation_count": budget,
                "fit_observation_count": len(fit),
                "committee_model_count": len(committee.repairs),
                "bic_weight_entropy_nats": float(
                    -np.sum(
                        committee.weights[committee.weights > 0.0]
                        * np.log(committee.weights[committee.weights > 0.0])
                    )
                ),
                **result,
            }
        )
        if budget >= maximum_budget:
            break

        seed = int(config["data_seed"]) + 10000 * repetition + 100 * budget
        if strategy == "random_design":
            seed += int(config["random_design"]["seed_offset"])
        scores, metadata, components = _strategy_scores(
            strategy,
            task=active_task,
            committee=committee,
            fit=fit,
            pool=remaining,
            config=config,
            gate1_config=gate1_config,
            seed=seed,
        )
        identifiers = remaining["point_id"].astype(str).tolist()
        selected_index = select_acquisition_index(scores, identifiers)
        selected = remaining.iloc[selected_index].copy()
        point_id = str(selected["point_id"])
        selected_target = target_by_point[point_id]
        acquisition_rows.append(
            {
                "task_id": task.task_id,
                "strategy": strategy,
                "repetition": repetition,
                "acquisition_round": budget + 1,
                "point_id": point_id,
                **{name: float(selected[name]) for name in variable_names},
                "acquisition_score": float(scores[selected_index]),
                "score_unit": metadata["score_unit"],
                "model_information_nats": float(
                    components["model_information_nats"][selected_index]
                ),
                "parameter_information_nats": float(
                    components["parameter_information_nats"][selected_index]
                ),
                "joint_information_nats": float(
                    components["joint_information_nats"][selected_index]
                ),
                "observed_target_after_selection": float(selected_target),
            }
        )
        fit_row = selected.drop(labels=["point_id"])
        fit_row["target"] = float(selected_target)
        fit_row["partition"] = "fit"
        fit = pd.concat([fit, fit_row.to_frame().T], ignore_index=True)
        remaining = remaining.drop(index=selected_index).reset_index(drop=True)
    return trajectory_rows, acquisition_rows


def _success_decision(summary: pd.DataFrame) -> dict[str, Any]:
    indexed = summary.set_index("strategy")
    joint = indexed.loc["joint_structure_parameter_information_gain"]
    random = indexed.loc["random_design"]
    model = indexed.loc["model_information_gain"]
    recovery_floor = max(
        int(random["behavior_recovery_count"]),
        int(model["behavior_recovery_count"]),
    )
    success = (
        float(joint["mean_locked_rmse"]) < float(random["mean_locked_rmse"])
        and float(joint["mean_locked_rmse"]) < float(model["mean_locked_rmse"])
        and int(joint["behavior_recovery_count"]) >= recovery_floor
    )
    return {
        "gate1c_development_success": bool(success),
        "joint_minus_random_mean_locked_rmse": float(
            joint["mean_locked_rmse"] - random["mean_locked_rmse"]
        ),
        "joint_minus_model_information_mean_locked_rmse": float(
            joint["mean_locked_rmse"] - model["mean_locked_rmse"]
        ),
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
    trajectory_output = run_dir / "metrics" / "gate1c_joint_trajectory.csv"
    summary_output = run_dir / "metrics" / "gate1c_joint_summary.csv"
    stage = "symbolic_model_revision_gate1c_joint_information"
    if should_skip(
        run_dir,
        stage,
        [trajectory_output, summary_output],
        resume,
        force,
    ):
        progress_message(run_dir, "Gate 1C joint pilot already complete", verbose)
        return run_dir

    config = read_yaml(config_path)
    _validate_protocol_config(config)
    gate1_config = read_yaml(config["gate1_config"])
    selected_task_ids = set(config["candidate_archive"]["task_ids"])
    tasks = {
        task.task_id: task
        for task in build_gate1_suite(gate1_config, seed=int(config["data_seed"]))
        if task.task_id in selected_task_ids
    }
    if set(tasks) != selected_task_ids:
        raise ValueError("Gate 1C could not build every frozen task.")
    frozen, archive_frame, archive_metadata = _load_frozen_candidates(config, tasks)
    write_table_bundle(
        archive_frame,
        run_dir / "metrics" / "gate1c_frozen_candidate_archive",
    )
    write_json(run_dir / "reports" / "gate1c_candidate_archive.json", archive_metadata)
    write_json(run_dir / "reports" / "gate1c_protocol_snapshot.json", config)
    progress_message(
        run_dir,
        "Gate 1C joint-information pilot started",
        verbose,
        tasks=len(tasks),
        strategies=len(config["design"]["strategies"]),
        repetitions=config["design"]["repetitions"],
        llm_calls=0,
    )

    trajectory_rows: list[dict[str, Any]] = []
    acquisition_rows: list[dict[str, Any]] = []
    for task_index, (task_id, task) in enumerate(tasks.items(), start=1):
        initial_fit = _maximin_initial_fit(
            task,
            int(config["design"]["initial_observation_count"]),
        )
        pool = _sample_acquisition_pool(
            task,
            count=int(config["design"]["acquisition_pool_count"]),
            seed=int(config["data_seed"]) + 2003 * task_index,
            outer_shell_only=bool(config["design"]["outer_shell_only"]),
        )
        for repetition in range(1, int(config["design"]["repetitions"]) + 1):
            rng = np.random.default_rng(
                int(config["data_seed"]) + 5003 * task_index + repetition
            )
            potential_targets = pool["target_noise_free"].to_numpy(float) + rng.normal(
                0.0, task.definition.noise_std, len(pool)
            )
            for strategy in config["design"]["strategies"]:
                progress_message(
                    run_dir,
                    "Gate 1C strategy started",
                    verbose,
                    task=task_id,
                    repetition=repetition,
                    strategy=strategy,
                )
                trajectory, acquisitions = _run_strategy(
                    task,
                    frozen[task_id],
                    pool,
                    initial_fit,
                    potential_targets,
                    strategy=str(strategy),
                    repetition=repetition,
                    config=config,
                    gate1_config=gate1_config,
                )
                trajectory_rows.extend(trajectory)
                acquisition_rows.extend(acquisitions)

    trajectory_frame = pd.DataFrame(trajectory_rows)
    acquisition_frame = pd.DataFrame(acquisition_rows)
    primary_budget = int(config["evaluation"]["primary_budget"])
    summary_frame = _summarize(
        trajectory_frame,
        primary_budget,
        by_task=False,
    )
    task_summary_frame = _summarize(
        trajectory_frame,
        primary_budget,
        by_task=True,
    )
    budget_summary_frame = _summarize_budgets(
        trajectory_frame,
        [int(value) for value in config["evaluation"]["report_budgets"]],
    )
    decision = _success_decision(summary_frame)
    write_table_bundle(trajectory_frame, trajectory_output.with_suffix(""))
    write_table_bundle(
        acquisition_frame,
        run_dir / "metrics" / "gate1c_acquisition_trace",
    )
    write_table_bundle(summary_frame, summary_output.with_suffix(""))
    write_table_bundle(
        task_summary_frame,
        run_dir / "metrics" / "gate1c_joint_summary_by_task",
    )
    write_table_bundle(
        budget_summary_frame,
        run_dir / "metrics" / "gate1c_joint_budget_summary",
    )
    write_json(
        run_dir / "reports" / "gate1c_joint_report.json",
        {
            "protocol": config["protocol"],
            "candidate_archive": archive_metadata,
            "data_seed": config["data_seed"],
            "llm_calls": 0,
            "initial_observation_count": config["design"][
                "initial_observation_count"
            ],
            "primary_budget": primary_budget,
            "study_contract": config["study"],
            **decision,
        },
    )
    mark_done(run_dir, stage, decision)
    progress_message(
        run_dir,
        "Gate 1C joint-information pilot complete",
        verbose,
        success=decision["gate1c_development_success"],
        output=run_dir,
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gate 1C joint structure-parameter information design."
    )
    parser.add_argument(
        "--config", default="configs/symbolic_model_revision_gate1c.yaml"
    )
    parser.add_argument("--run-id", default="gate1c_joint_information_pilot_v1")
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

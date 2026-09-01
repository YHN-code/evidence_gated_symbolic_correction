from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401

from asrc.constitutive.active_design import bic_model_weights
from asrc.model_revision.acquisition import (
    score_acquisition_pool,
    select_acquisition_index,
)
from asrc.model_revision.ast import evaluate_expression
from asrc.model_revision.benchmarks import Gate1Task, build_gate1_suite
from asrc.model_revision.evaluation import evaluate_candidates, predict_repair
from asrc.model_revision.proposals import (
    TypedRepair,
    canonical_expression_key,
    validate_typed_repair,
)
from asrc.utils.io import (
    ensure_run_dir,
    read_json,
    read_yaml,
    runs_root,
    write_json,
    write_table_bundle,
)
from asrc.utils.progress import mark_done, progress_message, should_skip


@dataclass(frozen=True)
class FittedCommittee:
    repairs: tuple[TypedRepair | None, ...]
    formulas: tuple[str, ...]
    parameters: tuple[dict[str, float], ...]
    complexities: np.ndarray
    weights: np.ndarray
    bic: np.ndarray
    selected_index: int


def _validate_protocol_config(config: dict[str, Any]) -> None:
    archive = config["candidate_archive"]
    design = config["design"]
    evidence = config["model_evidence"]
    expected = config["expected_information_gain"]
    evaluation = config["evaluation"]
    if archive.get("allow_new_llm_calls") is not False:
        raise ValueError("Gate 1B forbids new LLM calls.")
    if archive.get("ranking_criterion") != "observed_validation_selection_score":
        raise ValueError("Unsupported frozen-candidate ranking criterion.")
    if evidence.get("weight_criterion") != "bayesian_information_criterion":
        raise ValueError("Gate 1B requires BIC model weights.")
    if evidence.get("bic_complexity") != "fitted_parameter_count":
        raise ValueError("Gate 1B BIC complexity must be fitted_parameter_count.")
    if evidence.get("retain_baseline_null_model") is not True:
        raise ValueError("Gate 1B requires the baseline null model.")
    if expected.get("estimator") != "gauss_hermite_scalar_gaussian":
        raise ValueError("Gate 1B requires deterministic scalar Gaussian EIG.")
    supported = {
        "space_filling_design",
        "random_design",
        "predictive_disagreement",
        "expected_information_gain",
    }
    strategies = [str(value) for value in design["strategies"]]
    if len(strategies) != len(set(strategies)) or set(strategies) != supported:
        raise ValueError("Gate 1B requires each frozen acquisition strategy once.")
    maximum_budget = int(design["maximum_new_observations"])
    primary_budget = int(evaluation["primary_budget"])
    report_budgets = [int(value) for value in evaluation["report_budgets"]]
    if primary_budget > maximum_budget or primary_budget not in report_budgets:
        raise ValueError("The primary budget must be a reported attainable budget.")
    if any(value < 0 or value > maximum_budget for value in report_budgets):
        raise ValueError("All report budgets must lie within the acquisition budget.")


def _proposal_payloads(path: Path, task: Gate1Task) -> dict[str, TypedRepair]:
    payload = read_json(path)
    return {
        repair.proposal_id: repair
        for repair in (
            validate_typed_repair(item, task.contract)
            for item in payload.get("accepted", [])
        )
    }


def _load_frozen_candidates(
    config: dict[str, Any],
    tasks: dict[str, Gate1Task],
) -> tuple[dict[str, tuple[TypedRepair, ...]], pd.DataFrame, dict[str, Any]]:
    archive = config["candidate_archive"]
    control_id = str(archive["control_run_id"])
    feedback_id = str(archive["feedback_run_id"])
    feedback_dir = runs_root() / feedback_id
    candidate_path = feedback_dir / "metrics" / "gate1_feedback_candidate_results.csv"
    report_path = feedback_dir / "reports" / "gate1_feedback_comparison_report.json"
    if not candidate_path.exists() or not report_path.exists():
        raise FileNotFoundError("Gate 1B frozen feedback archive is incomplete.")
    source_report = read_json(report_path)
    if str(source_report.get("control_run_id")) != control_id:
        raise ValueError("Gate 1B archive references a different control run.")
    source_seed = int(archive.get("source_data_seed", config["data_seed"]))
    if int(source_report.get("data_seed", -1)) != source_seed:
        raise ValueError("Gate 1B archive data seed does not match the protocol.")

    frame = pd.read_csv(candidate_path)
    frame = frame.loc[
        frame["task_id"].isin(archive["task_ids"])
        & frame["arm"].isin(archive["arms"])
        & frame["status"].eq("valid")
    ].copy()
    proposal_map: dict[tuple[str, str, int, str], TypedRepair] = {}
    control_dir = runs_root() / control_id
    for task_id, task in tasks.items():
        for replicate in sorted(frame["replicate"].astype(int).unique()):
            control_path = (
                control_dir
                / "formulas"
                / "gate1_stability"
                / f"{task_id}_r{replicate:02d}_llm_typed_proposals.json"
            )
            feedback_path = (
                feedback_dir
                / "formulas"
                / "gate1_feedback"
                / f"{task_id}_r{replicate:02d}_all_proposals.json"
            )
            for arm, path in (
                ("independent_resampling", control_path),
                ("fit_feedback", feedback_path),
            ):
                if not path.exists():
                    raise FileNotFoundError(f"Missing frozen proposal archive: {path}")
                for proposal_id, repair in _proposal_payloads(path, task).items():
                    proposal_map[(task_id, arm, replicate, proposal_id)] = repair

    selected: dict[str, tuple[TypedRepair, ...]] = {}
    audit_rows: list[dict[str, Any]] = []
    maximum = int(archive["maximum_structures_per_task"])
    for task_id in archive["task_ids"]:
        task_rows = frame.loc[frame["task_id"].eq(task_id)].sort_values(
            ["selection_score", "complexity", "arm", "replicate", "proposal_id"]
        )
        repairs: list[TypedRepair] = []
        seen_expressions: set[str] = set()
        for _, row in task_rows.iterrows():
            key = (
                task_id,
                str(row["arm"]),
                int(row["replicate"]),
                str(row["proposal_id"]),
            )
            repair = proposal_map.get(key)
            if repair is None:
                raise ValueError(f"Candidate row has no archived proposal: {key}")
            expression_key = canonical_expression_key(repair.expression)
            if expression_key in seen_expressions:
                continue
            seen_expressions.add(expression_key)
            source_proposal_id = repair.proposal_id
            repair = replace(
                repair,
                proposal_id=f"gate1b_{task_id.lower()}_{len(repairs) + 1:02d}",
            )
            repairs.append(repair)
            audit_rows.append(
                {
                    "task_id": task_id,
                    "candidate_order": len(repairs),
                    "gate1b_proposal_id": repair.proposal_id,
                    "source_arm": str(row["arm"]),
                    "source_replicate": int(row["replicate"]),
                    "source_proposal_id": source_proposal_id,
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

    fingerprint = hashlib.sha256(
        json.dumps(
            {
                task_id: [canonical_expression_key(item.expression) for item in repairs]
                for task_id, repairs in sorted(selected.items())
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return selected, pd.DataFrame(audit_rows), {
        "control_run_id": control_id,
        "feedback_run_id": feedback_id,
        "candidate_set_sha256": fingerprint,
        "source_development_revision": source_report.get("development_revision"),
        "source_data_seed": source_seed,
    }


def _initial_fit(task: Gate1Task, mode: str) -> pd.DataFrame:
    if mode != "all_source_fit":
        raise ValueError(
            "Gate 1B requires initial_observation_mode=all_source_fit because the "
            "frozen candidates were generated from the complete Gate 1A evidence."
        )
    return (
        task.observed.loc[task.observed["partition"].eq("fit")]
        .copy()
        .reset_index(drop=True)
    )


def _sample_acquisition_pool(
    task: Gate1Task,
    *,
    count: int,
    seed: int,
    outer_shell_only: bool,
) -> pd.DataFrame:
    names = list(task.variables)
    lower = np.asarray([task.definition.locked_ranges[name][0] for name in names])
    upper = np.asarray([task.definition.locked_ranges[name][1] for name in names])
    observed_lower = np.asarray(
        [task.definition.observed_ranges[name][0] for name in names]
    )
    observed_upper = np.asarray(
        [task.definition.observed_ranges[name][1] for name in names]
    )
    chunks = []
    generated = 0
    attempt = 0
    while generated < count:
        sampler = qmc.LatinHypercube(d=len(names), seed=int(seed) + attempt)
        unit = sampler.random(max(count * 2, 64))
        values = qmc.scale(unit, lower, upper)
        if outer_shell_only:
            outside = np.any(
                (values < observed_lower[None, :])
                | (values > observed_upper[None, :]),
                axis=1,
            )
            values = values[outside]
        chunks.append(values)
        generated += len(values)
        attempt += 1
        if attempt > 100:
            raise RuntimeError("Unable to sample the Gate 1B acquisition shell.")
    values = np.vstack(chunks)[:count]
    frame = pd.DataFrame(values, columns=names)
    frame.insert(0, "point_id", [f"q{index:04d}" for index in range(count)])
    variables = {name: frame[name].to_numpy(float) for name in names}
    frame["baseline"] = evaluate_expression(task.baseline_expression, variables)
    frame["target_noise_free"] = predict_repair(
        task,
        task.oracle_repair,
        frame,
        task.definition.oracle_parameters,
    )
    return frame


def _task_with_active_fit(
    task: Gate1Task,
    fit: pd.DataFrame,
) -> Gate1Task:
    active_fit = fit.copy()
    active_fit["partition"] = "fit"
    validation = task.observed.loc[
        task.observed["partition"].eq("validation")
    ].copy()
    return replace(
        task,
        observed=pd.concat([active_fit, validation], ignore_index=True),
    )


def _fit_committee(
    task: Gate1Task,
    repairs: tuple[TypedRepair, ...],
    fit: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
) -> tuple[Gate1Task, FittedCommittee]:
    active_task = _task_with_active_fit(task, fit)
    evaluations = evaluate_candidates(
        active_task,
        repairs,
        method="gate1b_frozen_committee",
        seed=seed,
        evaluation_config=config["evaluation"],
    )
    repair_by_id = {repair.proposal_id: repair for repair in repairs}
    valid = [item for item in evaluations if item.status == "valid"]
    if len(valid) < 1:
        raise RuntimeError(f"No valid frozen candidate remains for {task.task_id}.")

    models: list[TypedRepair | None] = [None]
    formulas = ["M_base(x)"]
    parameters: list[dict[str, float]] = [{}]
    complexities = [0.0]
    predictions = [fit["baseline"].to_numpy(float)]
    for item in valid:
        repair = repair_by_id[item.proposal_id]
        models.append(repair)
        formulas.append(item.formula)
        parameters.append(dict(item.parameter_values))
        complexities.append(float(item.parameter_count))
        predictions.append(
            predict_repair(active_task, repair, fit, item.parameter_values)
        )
    matrix = np.vstack(predictions)
    weights, bic = bic_model_weights(
        fit["target"].to_numpy(float),
        matrix,
        np.asarray(complexities),
    )
    selected_index = min(
        range(len(models)),
        key=lambda index: (float(bic[index]), complexities[index], formulas[index]),
    )
    return active_task, FittedCommittee(
        repairs=tuple(models),
        formulas=tuple(formulas),
        parameters=tuple(parameters),
        complexities=np.asarray(complexities),
        weights=weights,
        bic=bic,
        selected_index=selected_index,
    )


def _committee_predictions(
    task: Gate1Task,
    committee: FittedCommittee,
    frame: pd.DataFrame,
) -> np.ndarray:
    predictions = []
    for repair, parameters in zip(committee.repairs, committee.parameters):
        if repair is None:
            predictions.append(frame["baseline"].to_numpy(float))
        else:
            predictions.append(predict_repair(task, repair, frame, parameters))
    return np.vstack(predictions)


def _evaluate_selected(
    task: Gate1Task,
    committee: FittedCommittee,
    recovery_noise_multiplier: float,
) -> dict[str, Any]:
    index = committee.selected_index
    repair = committee.repairs[index]
    if repair is None:
        prediction = task.locked["baseline"].to_numpy(float)
        expression_match = False
        proposal_id = "baseline_no_change"
    else:
        prediction = predict_repair(
            task,
            repair,
            task.locked,
            committee.parameters[index],
        )
        expression_match = (
            canonical_expression_key(repair.expression)
            == canonical_expression_key(task.oracle_repair.expression)
        )
        proposal_id = repair.proposal_id
    locked_rmse = float(
        np.sqrt(
            np.mean((task.locked["target"].to_numpy(float) - prediction) ** 2)
        )
    )
    threshold = max(
        float(recovery_noise_multiplier) * task.definition.noise_std,
        0.03 * float(np.std(task.locked["target"].to_numpy(float))),
    )
    return {
        "selected_proposal_id": proposal_id,
        "selected_formula": committee.formulas[index],
        "selected_model_weight": float(committee.weights[index]),
        "selected_bic": float(committee.bic[index]),
        "locked_rmse": locked_rmse,
        "behavior_recovered": bool(locked_rmse <= threshold),
        "exact_expression_match": bool(expression_match),
        "recovery_threshold": threshold,
    }


def _run_strategy(
    task: Gate1Task,
    repairs: tuple[TypedRepair, ...],
    pool: pd.DataFrame,
    initial_fit: pd.DataFrame,
    potential_targets: np.ndarray,
    *,
    strategy: str,
    repetition: int,
    config: dict[str, Any],
    gate1_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    design = config["design"]
    variable_names = list(task.variables)
    lower = np.asarray(
        [task.definition.locked_ranges[name][0] for name in variable_names]
    )
    upper = np.asarray(
        [task.definition.locked_ranges[name][1] for name in variable_names]
    )
    fit = initial_fit.copy().reset_index(drop=True)
    target_by_point = dict(
        zip(pool["point_id"].astype(str), np.asarray(potential_targets, dtype=float))
    )
    remaining = pool.drop(columns=["target_noise_free"]).copy().reset_index(drop=True)
    trajectory_rows = []
    acquisition_rows = []
    max_budget = int(design["maximum_new_observations"])
    for budget in range(max_budget + 1):
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
                        committee.weights[committee.weights > 0]
                        * np.log(committee.weights[committee.weights > 0])
                    )
                ),
                **result,
            }
        )
        if budget >= max_budget:
            break

        prediction_matrix = _committee_predictions(active_task, committee, remaining)
        seed = int(config["data_seed"]) + 10000 * repetition + 100 * budget
        if strategy == "random_design":
            seed += int(config["random_design"]["seed_offset"])
        scores, metadata = score_acquisition_pool(
            strategy,
            candidate_design=remaining[variable_names].to_numpy(float),
            observed_design=fit[variable_names].to_numpy(float),
            prediction_matrix=prediction_matrix,
            model_weights=committee.weights,
            lower_bounds=lower,
            upper_bounds=upper,
            noise_std=task.definition.noise_std,
            seed=seed,
            information_gain_quadrature_order=int(
                config["expected_information_gain"]["quadrature_order"]
            ),
        )
        identifiers = remaining["point_id"].astype(str).tolist()
        selected_index = select_acquisition_index(scores, identifiers)
        selected = remaining.iloc[selected_index].copy()
        selected_target = target_by_point[str(selected["point_id"])]
        standard_errors = metadata.get("score_standard_errors")
        acquisition_rows.append(
            {
                "task_id": task.task_id,
                "strategy": strategy,
                "repetition": repetition,
                "acquisition_round": budget + 1,
                "point_id": str(selected["point_id"]),
                **{name: float(selected[name]) for name in variable_names},
                "acquisition_score": float(scores[selected_index]),
                "acquisition_score_standard_error": (
                    float(standard_errors[selected_index])
                    if standard_errors is not None
                    else float("nan")
                ),
                "score_unit": metadata["score_unit"],
                "observed_target_after_selection": float(selected_target),
            }
        )
        fit_row = selected.drop(labels=["point_id"])
        fit_row["target"] = float(selected_target)
        fit_row["partition"] = "fit"
        fit = pd.concat([fit, fit_row.to_frame().T], ignore_index=True)
        remaining = remaining.drop(index=selected_index).reset_index(drop=True)
    return trajectory_rows, acquisition_rows


def _summarize(
    trajectory: pd.DataFrame,
    primary_budget: int,
    *,
    by_task: bool,
) -> pd.DataFrame:
    final = trajectory.loc[trajectory["new_observation_count"].eq(primary_budget)]
    rows = []
    grouped = (
        final.groupby(["task_id", "strategy"], sort=True)
        if by_task
        else final.groupby("strategy", sort=True)
    )
    for group_key, group in grouped:
        if by_task:
            task_id, strategy = group_key
        else:
            task_id, strategy = "all", group_key
        rows.append(
            {
                "task_id": task_id,
                "strategy": strategy,
                "cell_count": len(group),
                "mean_locked_rmse": float(group["locked_rmse"].mean()),
                "median_locked_rmse": float(group["locked_rmse"].median()),
                "behavior_recovery_count": int(group["behavior_recovered"].sum()),
                "exact_expression_match_count": int(
                    group["exact_expression_match"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _summarize_budgets(
    trajectory: pd.DataFrame,
    report_budgets: list[int],
) -> pd.DataFrame:
    selected = trajectory.loc[
        trajectory["new_observation_count"].isin(report_budgets)
    ]
    return (
        selected.groupby(
            ["task_id", "strategy", "new_observation_count"], sort=True
        )
        .agg(
            replicate_count=("repetition", "size"),
            mean_locked_rmse=("locked_rmse", "mean"),
            median_locked_rmse=("locked_rmse", "median"),
            behavior_recovery_fraction=("behavior_recovered", "mean"),
            exact_expression_match_fraction=("exact_expression_match", "mean"),
            mean_bic_weight_entropy_nats=("bic_weight_entropy_nats", "mean"),
        )
        .reset_index()
    )


def _success_decision(summary: pd.DataFrame) -> dict[str, Any]:
    indexed = summary.set_index("strategy")
    nonadaptive = indexed.loc[["space_filling_design", "random_design"]]
    adaptive_names = ["predictive_disagreement", "expected_information_gain"]
    successful = []
    for name in adaptive_names:
        row = indexed.loc[name]
        if (
            float(row["mean_locked_rmse"])
            < float(nonadaptive["mean_locked_rmse"].min())
            and int(row["behavior_recovery_count"])
            >= int(nonadaptive["behavior_recovery_count"].max())
        ):
            successful.append(name)
    return {
        "successful_adaptive_strategies": successful,
        "gate1b_development_success": bool(successful),
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
    trajectory_output = run_dir / "metrics" / "gate1b_active_trajectory.csv"
    summary_output = run_dir / "metrics" / "gate1b_active_summary.csv"
    stage = "symbolic_model_revision_gate1b_active_discrimination"
    if should_skip(
        run_dir,
        stage,
        [trajectory_output, summary_output],
        resume,
        force,
    ):
        progress_message(run_dir, "Gate 1B active pilot already complete", verbose)
        return run_dir

    config = read_yaml(config_path)
    _validate_protocol_config(config)
    gate1_config = read_yaml(config["gate1_config"])
    tasks = {
        task.task_id: task
        for task in build_gate1_suite(gate1_config, seed=int(config["data_seed"]))
        if task.task_id in set(config["candidate_archive"]["task_ids"])
    }
    frozen, archive_frame, archive_metadata = _load_frozen_candidates(config, tasks)
    write_table_bundle(
        archive_frame,
        run_dir / "metrics" / "gate1b_frozen_candidate_archive",
    )
    write_json(
        run_dir / "reports" / "gate1b_candidate_archive.json",
        archive_metadata,
    )
    write_json(run_dir / "reports" / "gate1b_protocol_snapshot.json", config)
    progress_message(
        run_dir,
        "Gate 1B active pilot started",
        verbose,
        tasks=len(tasks),
        strategies=len(config["design"]["strategies"]),
        repetitions=config["design"]["repetitions"],
        llm_calls=0,
    )

    trajectory_rows: list[dict[str, Any]] = []
    acquisition_rows: list[dict[str, Any]] = []
    for task_index, (task_id, task) in enumerate(tasks.items(), start=1):
        initial_fit = _initial_fit(
            task, str(config["design"]["initial_observation_mode"])
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
                    "Gate 1B strategy started",
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
    summary_frame = _summarize(
        trajectory_frame,
        int(config["evaluation"]["primary_budget"]),
        by_task=False,
    )
    task_summary_frame = _summarize(
        trajectory_frame,
        int(config["evaluation"]["primary_budget"]),
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
        run_dir / "metrics" / "gate1b_acquisition_trace",
    )
    write_table_bundle(summary_frame, summary_output.with_suffix(""))
    write_table_bundle(
        task_summary_frame,
        run_dir / "metrics" / "gate1b_active_summary_by_task",
    )
    write_table_bundle(
        budget_summary_frame,
        run_dir / "metrics" / "gate1b_active_budget_summary",
    )
    write_json(
        run_dir / "reports" / "gate1b_active_report.json",
        {
            "protocol": config["protocol"],
            "candidate_archive": archive_metadata,
            "data_seed": config["data_seed"],
            "llm_calls": 0,
            "initial_observation_mode": config["design"][
                "initial_observation_mode"
            ],
            "primary_budget": config["evaluation"]["primary_budget"],
            "study_contract": config["study"],
            **decision,
        },
    )
    mark_done(run_dir, stage, decision)
    progress_message(
        run_dir,
        "Gate 1B active pilot complete",
        verbose,
        success=decision["gate1b_development_success"],
        output=run_dir,
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gate 1B active discrimination on a frozen repair committee."
    )
    parser.add_argument(
        "--config", default="configs/symbolic_model_revision_gate1b.yaml"
    )
    parser.add_argument("--run-id", default="gate1b_active_pilot_v2")
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

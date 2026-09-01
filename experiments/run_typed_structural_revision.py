from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401

from asrc.model_revision.typed_structural_revision import (
    StructuralMethodResult,
    baseline_method_result,
    build_structural_revision_suite,
    evaluate_structural_method,
    generate_structural_candidates,
)
from asrc.utils.io import (
    ensure_run_dir,
    read_json,
    read_yaml,
    write_json,
    write_json_atomic,
    write_table_bundle,
)
from asrc.utils.progress import mark_done, progress_message, should_skip


class StructuralPySRUnavailableError(RuntimeError):
    """Raised when the optional independent-output PySR comparator is unavailable."""


def _supported_kwargs(regressor: type[Any], configured: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(regressor.__init__).parameters
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return configured
    return {key: value for key, value in configured.items() if key in parameters}


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(observed) - np.asarray(predicted)) ** 2)))


def _selected_pysr_metadata(model: Any) -> tuple[str, int]:
    best = model.get_best() if callable(getattr(model, "get_best", None)) else None
    if isinstance(best, (pd.Series, dict)):
        return str(best.get("equation", model.sympy())), int(best.get("complexity", 0))
    return str(model.sympy()), 0


def _fit_pysr_residual(
    task: Any,
    target: str,
    baseline: str,
    *,
    seed: int,
    output_directory: Path,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], str, int, pd.DataFrame]:
    try:
        from pysr import PySRRegressor
    except Exception as exc:
        raise StructuralPySRUnavailableError(
            "PySR is unavailable in the active Python environment."
        ) from exc
    fit = task.data.loc[task.data["partition"].eq("fit")]
    x_fit = fit.loc[:, list(task.variables)].copy()
    rename = {name: f"x{index}" for index, name in enumerate(task.variables)}
    x_fit = x_fit.rename(columns=rename)
    configured = {
        "niterations": int(config.get("niterations", 80)),
        "populations": int(config.get("populations", 6)),
        "population_size": int(config.get("population_size", 30)),
        "binary_operators": list(config.get("binary_operators", ["+", "-", "*", "/"])),
        "unary_operators": list(config.get("unary_operators", ["sin", "cos", "tanh", "exp"])),
        "model_selection": "best",
        "maxsize": int(config.get("maxsize", 22)),
        "random_state": int(seed),
        "deterministic": True,
        "parallelism": "serial",
        "progress": bool(config.get("progress", False)),
        "verbosity": int(config.get("verbosity", 0)),
        "warm_start": False,
        "output_directory": str(output_directory),
    }
    model = PySRRegressor(**_supported_kwargs(PySRRegressor, configured))
    model.fit(x_fit, fit[target].to_numpy(float) - fit[baseline].to_numpy(float))
    predictions: dict[str, np.ndarray] = {}
    for partition in ("validation", "locked"):
        frame = task.data.loc[task.data["partition"].eq(partition)]
        x = frame.loc[:, list(task.variables)].rename(columns=rename)
        predictions[partition] = frame[baseline].to_numpy(float) + np.asarray(
            model.predict(x), dtype=float
        )
    formula, complexity = _selected_pysr_metadata(model)
    equations = getattr(model, "equations_", pd.DataFrame())
    return predictions, formula, complexity, equations.copy()


def _pysr_method_result(
    task: Any,
    *,
    seed: int,
    output_directory: Path,
    config: dict[str, Any],
    acceptance_noise_multiplier: float,
) -> StructuralMethodResult:
    started = perf_counter()
    primary, primary_formula, primary_complexity, primary_equations = _fit_pysr_residual(
        task,
        "target",
        "baseline",
        seed=seed,
        output_directory=output_directory / "primary",
        config=config,
    )
    diagnostic, diagnostic_formula, diagnostic_complexity, diagnostic_equations = _fit_pysr_residual(
        task,
        "diagnostic_target",
        "diagnostic_baseline",
        seed=seed + 1,
        output_directory=output_directory / "diagnostic",
        config=config,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    primary_equations.to_csv(output_directory / "primary_equations.csv", index=False)
    diagnostic_equations.to_csv(output_directory / "diagnostic_equations.csv", index=False)
    validation = task.data.loc[task.data["partition"].eq("validation")]
    locked = task.data.loc[task.data["partition"].eq("locked")]
    validation_primary = _rmse(validation["target"], primary["validation"])
    validation_diagnostic = _rmse(
        validation["diagnostic_target"], diagnostic["validation"]
    )
    locked_primary = _rmse(locked["target"], primary["locked"])
    locked_diagnostic = _rmse(locked["diagnostic_target"], diagnostic["locked"])
    baseline_primary = _rmse(locked["target"], locked["baseline"])
    baseline_diagnostic = _rmse(
        locked["diagnostic_target"], locked["diagnostic_baseline"]
    )
    primary_scale = max(float(np.std(locked["target"])), 0.1)
    diagnostic_scale = max(float(np.std(locked["diagnostic_target"])), 0.1)
    adequacy_accepted = bool(
        validation_primary
        <= acceptance_noise_multiplier * task.noise_standard_deviation
        and validation_diagnostic
        <= acceptance_noise_multiplier * task.diagnostic_noise_standard_deviation
        and not np.any(primary["locked"] <= 0.0)
    )
    return StructuralMethodResult(
        task_id=task.task_id,
        mechanism=task.mechanism,
        method="pysr_independent_discrepancies",
        candidate_count=len(primary_equations) + len(diagnostic_equations),
        wall_time_seconds=perf_counter() - started,
        selected_proposal_id="pysr_pair",
        selected_location="model_outputs",
        selected_edit_type="independent_additive",
        selected_family_id="pysr",
        selected_expression=(
            f"primary: M_base + ({primary_formula}); diagnostic: D_base + ({diagnostic_formula})"
        ),
        complexity=primary_complexity + diagnostic_complexity,
        validation_primary_rmse=validation_primary,
        validation_diagnostic_rmse=validation_diagnostic,
        locked_primary_rmse=locked_primary,
        locked_diagnostic_rmse=locked_diagnostic,
        baseline_locked_primary_rmse=baseline_primary,
        baseline_locked_diagnostic_rmse=baseline_diagnostic,
        locked_primary_improvement_fraction=(baseline_primary - locked_primary) / baseline_primary,
        locked_joint_normalized_rmse=float(
            np.sqrt(
                0.5
                * (
                    (locked_primary / primary_scale) ** 2
                    + (locked_diagnostic / diagnostic_scale) ** 2
                )
            )
        ),
        location_recovered=False,
        edit_type_recovered=False,
        joint_structure_recovered=False,
        family_recovered=False,
        physical_violations=int(np.any(primary["locked"] <= 0.0)),
        adequacy_accepted=adequacy_accepted,
        expected_library_coverage=task.expected_library_coverage,
        parameter_values={},
    )


def _aggregate(methods: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, frame in methods.groupby("method", sort=False):
        covered = frame.loc[frame["expected_library_coverage"].astype(bool)]
        uncovered = frame.loc[~frame["expected_library_coverage"].astype(bool)]
        rows.append(
            {
                "method": method,
                "task_count": len(frame),
                "mean_locked_primary_rmse": frame["locked_primary_rmse"].mean(),
                "mean_locked_diagnostic_rmse": frame["locked_diagnostic_rmse"].mean(),
                "mean_locked_joint_normalized_rmse": frame[
                    "locked_joint_normalized_rmse"
                ].mean(),
                "joint_structure_recovery_rate": frame[
                    "joint_structure_recovered"
                ].mean(),
                "covered_joint_structure_recovery_rate": (
                    covered["joint_structure_recovered"].mean()
                    if len(covered)
                    else float("nan")
                ),
                "family_recovery_rate": frame["family_recovered"].mean(),
                "covered_family_recovery_rate": (
                    covered["family_recovered"].mean()
                    if len(covered)
                    else float("nan")
                ),
                "acceptance_rate": frame["adequacy_accepted"].mean(),
                "covered_acceptance_rate": (
                    covered["adequacy_accepted"].mean()
                    if len(covered)
                    else float("nan")
                ),
                "uncovered_acceptance_rate": (
                    uncovered["adequacy_accepted"].mean()
                    if len(uncovered)
                    else float("nan")
                ),
                "physical_violation_rate": (frame["physical_violations"] > 0).mean(),
                "mean_complexity": frame["complexity"].mean(),
                "mean_candidate_count": frame["candidate_count"].mean(),
                "mean_wall_time_seconds": frame["wall_time_seconds"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _gate_decision(aggregate: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    gate = config["confirmation_gate"]
    development_only = str(config["protocol"].get("status", "development_only")) != "frozen_confirmation"
    indexed = aggregate.set_index("method")
    main = indexed.loc["typed_structural_revision"]
    output = indexed.loc["output_discrepancy"]
    checks = {
        "joint_structure_recovery": bool(
            main["covered_joint_structure_recovery_rate"]
            >= float(gate["minimum_joint_structure_recovery_rate"])
        ),
        "locked_joint_error": bool(
            main["mean_locked_joint_normalized_rmse"]
            <= float(gate["maximum_mean_locked_joint_normalized_rmse"])
        ),
        "physical_violations": bool(
            main["physical_violation_rate"]
            <= float(gate["maximum_physical_violation_rate"])
        ),
        "better_than_output_discrepancy": bool(
            main["mean_locked_joint_normalized_rmse"]
            < output["mean_locked_joint_normalized_rmse"]
        ),
        "rejects_uncovered_family": bool(
            main["uncovered_acceptance_rate"]
            <= float(gate.get("maximum_uncovered_acceptance_rate", 0.0))
        ),
    }
    if not bool(gate.get("require_improvement_over_output_discrepancy", True)):
        checks["better_than_output_discrepancy"] = True
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "development_only": development_only,
        "interpretation": (
            (
                "A pass permits an unseen-seed confirmation; it does not by itself justify "
                "renaming the manuscript as structural model revision."
            )
            if development_only
            else (
                "This cell is one part of the frozen confirmation matrix; manuscript-level "
                "claims require the aggregate confirmation gate."
            )
        ),
    }


def _checkpoint_path(run_dir: Path, task_id: str, method: str) -> Path:
    return (
        run_dir
        / "checkpoints"
        / "typed_structural_revision"
        / task_id
        / f"{method}.json"
    )


def _load_method_checkpoint(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    if payload.get("schema_version") != "typed-structural-revision-checkpoint-v2":
        return None
    method_row = payload.get("method_row")
    candidate_rows = payload.get("candidate_rows", [])
    if not isinstance(method_row, dict) or not isinstance(candidate_rows, list):
        return None
    return dict(method_row), [dict(row) for row in candidate_rows]


def _save_method_checkpoint(
    path: Path,
    method_row: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": "typed-structural-revision-checkpoint-v2",
            "method_row": method_row,
            "candidate_rows": candidate_rows,
        },
    )


def run(
    config_path: str,
    run_id: str,
    *,
    seed: int,
    pysr_mode: str,
    resume: bool,
    force: bool,
    verbose: bool,
    task_ids: list[str] | None = None,
    noise_multiplier: float = 1.0,
) -> Path:
    config = read_yaml(config_path)
    run_dir = ensure_run_dir(run_id)
    method_output = run_dir / "metrics" / "typed_structural_revision_methods.csv"
    candidate_output = run_dir / "metrics" / "typed_structural_revision_candidates.csv"
    aggregate_output = run_dir / "metrics" / "typed_structural_revision_aggregate.csv"
    report_output = run_dir / "reports" / "typed_structural_revision_gate.json"
    outputs = [method_output, candidate_output, aggregate_output, report_output]
    if should_skip(run_dir, "typed_structural_revision", outputs, resume, force):
        progress_message(run_dir, "Typed structural-revision pilot already complete", verbose)
        return run_dir

    data_config = config["data"]
    evaluation_config = config["evaluation"]
    tasks = build_structural_revision_suite(
        seed=seed,
        fit_size=int(data_config["fit_size"]),
        validation_size=int(data_config["validation_size"]),
        locked_size=int(data_config["locked_size"]),
        noise_multiplier=noise_multiplier,
    )
    if task_ids:
        selected = set(task_ids)
        tasks = tuple(task for task in tasks if task.task_id in selected)
        missing = selected - {task.task_id for task in tasks}
        if missing:
            raise ValueError(f"Unknown structural-revision tasks: {sorted(missing)}")
    progress_message(
        run_dir,
        "Typed structural-revision pilot started",
        verbose,
        tasks=len(tasks),
        seed=seed,
        pysr=pysr_mode,
        noise_multiplier=noise_multiplier,
    )
    candidate_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks, start=1):
        progress_message(
            run_dir,
            "Structural task started",
            verbose,
            task=f"{task.task_id} ({task_index}/{len(tasks)})",
            mechanism=task.mechanism,
        )
        task.data.loc[task.data["partition"].ne("locked")].to_csv(
            run_dir / "data" / f"{task.task_id}_observed.csv", index=False
        )
        task.data.loc[task.data["partition"].eq("locked")].to_csv(
            run_dir / "data" / f"{task.task_id}_locked.csv", index=False
        )
        write_json(
            run_dir / "formulas" / f"{task.task_id}_contract.json",
            {
                "task_id": task.task_id,
                "mechanism": task.mechanism,
                "variables": list(task.variables),
                "eligible_locations": list(task.locations),
                "supported_edit_types": [
                    "insert_additive",
                    "scale_component",
                    "replace_component",
                ],
                "baseline_response": task.baseline_response,
                "baseline_diagnostic": task.baseline_diagnostic,
                "true_structure_audit_only": {
                    "location": task.true_location,
                    "edit_type": task.true_edit_type,
                    "family": task.true_family_id,
                    "expected_library_coverage": task.expected_library_coverage,
                },
            },
        )
        baseline_checkpoint = _checkpoint_path(run_dir, task.task_id, "baseline")
        restored_baseline = (
            _load_method_checkpoint(baseline_checkpoint)
            if resume and not force
            else None
        )
        if restored_baseline is not None:
            method_rows.append(restored_baseline[0])
        else:
            baseline_row = baseline_method_result(task).to_row()
            baseline_row["parameter_values"] = "{}"
            method_rows.append(baseline_row)
            _save_method_checkpoint(baseline_checkpoint, baseline_row, [])
        structural_candidates = generate_structural_candidates(task)
        arms = (
            (
                "output_discrepancy",
                generate_structural_candidates(task, output_only=True),
                float(evaluation_config["diagnostic_weight"]),
            ),
            (
                "structural_response_only",
                structural_candidates,
                0.0,
            ),
            (
                "typed_structural_revision",
                structural_candidates,
                float(evaluation_config["diagnostic_weight"]),
            ),
            (
                "oracle_location",
                generate_structural_candidates(task, locations=(task.true_location,)),
                float(evaluation_config["diagnostic_weight"]),
            ),
        )
        for method, candidates, diagnostic_weight in arms:
            checkpoint = _checkpoint_path(run_dir, task.task_id, method)
            restored = (
                _load_method_checkpoint(checkpoint) if resume and not force else None
            )
            if restored is not None:
                method_row, restored_candidates = restored
                method_rows.append(method_row)
                candidate_rows.extend(restored_candidates)
                progress_message(
                    run_dir,
                    "Candidate evaluation restored",
                    verbose,
                    task=task.task_id,
                    method=method,
                    candidates=len(restored_candidates),
                )
                continue
            progress_message(
                run_dir,
                "Candidate evaluation started",
                verbose,
                task=task.task_id,
                method=method,
                candidates=len(candidates),
                diagnostic_weight=diagnostic_weight,
            )
            evaluations, result = evaluate_structural_method(
                task,
                candidates,
                method=method,
                seed=seed + task_index,
                diagnostic_weight=diagnostic_weight,
                complexity_penalty=float(evaluation_config["complexity_penalty"]),
                acceptance_noise_multiplier=float(
                    evaluation_config["acceptance_noise_multiplier"]
                ),
                restarts=int(evaluation_config["optimizer_restarts"]),
                max_nfev=int(evaluation_config["optimizer_max_nfev"]),
            )
            arm_candidate_rows: list[dict[str, Any]] = []
            for evaluation in evaluations:
                row = evaluation.to_row()
                row["parameter_values"] = json.dumps(
                    row["parameter_values"], sort_keys=True
                )
                arm_candidate_rows.append(row)
            candidate_rows.extend(arm_candidate_rows)
            row = result.to_row()
            row["parameter_values"] = json.dumps(row["parameter_values"], sort_keys=True)
            method_rows.append(row)
            _save_method_checkpoint(checkpoint, row, arm_candidate_rows)
            progress_message(
                run_dir,
                "Candidate evaluation finished",
                verbose,
                task=task.task_id,
                method=method,
                selected=f"{result.selected_location}/{result.selected_edit_type}/{result.selected_family_id}",
                locked_primary=f"{result.locked_primary_rmse:.6g}",
                locked_joint=f"{result.locked_joint_normalized_rmse:.6g}",
                structure_recovered=result.joint_structure_recovered,
            )
        if pysr_mode != "off":
            pysr_checkpoint = _checkpoint_path(
                run_dir, task.task_id, "pysr_independent_discrepancies"
            )
            restored_pysr = (
                _load_method_checkpoint(pysr_checkpoint)
                if resume and not force
                else None
            )
            if restored_pysr is not None:
                method_rows.append(restored_pysr[0])
                progress_message(
                    run_dir,
                    "Independent-output PySR restored",
                    verbose,
                    task=task.task_id,
                )
                continue
            progress_message(run_dir, "Independent-output PySR started", verbose, task=task.task_id)
            try:
                pysr_result = _pysr_method_result(
                    task,
                    seed=seed + 100 * task_index,
                    output_directory=run_dir / "formulas" / "pysr" / task.task_id,
                    config=config["pysr"],
                    acceptance_noise_multiplier=float(
                        evaluation_config["acceptance_noise_multiplier"]
                    ),
                )
            except StructuralPySRUnavailableError:
                if pysr_mode == "required":
                    raise
                progress_message(run_dir, "Optional PySR arm skipped", verbose, task=task.task_id)
            else:
                row = pysr_result.to_row()
                row["parameter_values"] = "{}"
                method_rows.append(row)
                _save_method_checkpoint(pysr_checkpoint, row, [])
                progress_message(
                    run_dir,
                    "Independent-output PySR finished",
                    verbose,
                    task=task.task_id,
                    locked_joint=f"{pysr_result.locked_joint_normalized_rmse:.6g}",
                )

    candidates = pd.DataFrame(candidate_rows)
    methods = pd.DataFrame(method_rows)
    aggregate = _aggregate(methods)
    gate = _gate_decision(aggregate, config)
    write_table_bundle(candidates, candidate_output.with_suffix(""))
    write_table_bundle(methods, method_output.with_suffix(""))
    write_table_bundle(aggregate, aggregate_output.with_suffix(""))
    write_json(
        report_output,
        {
            "protocol": config["protocol"],
            "seed": seed,
            "noise_multiplier": noise_multiplier,
            "task_count": len(tasks),
            "candidate_library_is_crossed_with_all_eligible_locations": True,
            "locked_data_used_for_selection": False,
            "gate": gate,
        },
    )
    mark_done(
        run_dir,
        "typed_structural_revision",
        {"task_count": len(tasks), "gate_status": gate["status"]},
    )
    progress_message(
        run_dir,
        "Typed structural-revision pilot complete",
        verbose,
        output=run_dir,
        gate=gate["status"],
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the development-only joint location/type/expression structural-revision pilot."
        )
    )
    parser.add_argument(
        "--config", default="configs/typed_structural_revision_pilot.yaml"
    )
    parser.add_argument("--run-id", default="typed_structural_revision_pilot_v1")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--pysr", choices=["off", "optional", "required"], default="off"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--task-id",
        action="append",
        choices=["SR01", "SR02", "SR03", "SR04"],
        dest="task_ids",
        help="Run one selected task; repeat to select multiple tasks.",
    )
    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    run(
        args.config,
        args.run_id,
        seed=args.seed,
        pysr_mode=args.pysr,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
        task_ids=args.task_ids,
        noise_multiplier=args.noise_multiplier,
    )


if __name__ == "__main__":
    main()

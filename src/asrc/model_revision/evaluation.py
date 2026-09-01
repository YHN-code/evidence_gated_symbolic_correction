from __future__ import annotations

import hashlib
import itertools
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from asrc.model_revision.ast import (
    ExpressionEvaluationError,
    apply_repair,
    evaluate_expression,
    expression_node_count,
    parameter_names,
    repair_to_text,
)
from asrc.model_revision.benchmarks import Gate1Task
from asrc.model_revision.proposals import TypedRepair, canonical_expression_key


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(observed) - np.asarray(predicted)) ** 2)))


def _frame_variables(task: Gate1Task, frame: Any) -> dict[str, np.ndarray]:
    return {name: frame[name].to_numpy(float) for name in task.variables}


def _predict(
    task: Gate1Task,
    repair: TypedRepair,
    frame: Any,
    parameters: Mapping[str, float],
) -> np.ndarray:
    variables = _frame_variables(task, frame)
    baseline = evaluate_expression(task.baseline_expression, variables)
    return apply_repair(repair, baseline, variables, parameters)


def predict_repair(
    task: Gate1Task,
    repair: TypedRepair,
    frame: Any,
    parameters: Mapping[str, float],
) -> np.ndarray:
    return _predict(task, repair, frame, parameters)


def repair_parameter_jacobian(
    task: Gate1Task,
    repair: TypedRepair,
    frame: Any,
    parameters: Mapping[str, float],
) -> tuple[tuple[str, ...], np.ndarray]:
    names = tuple(parameter_names(repair.expression))
    jacobian = np.empty((len(frame), len(names)), dtype=float)
    relative_step = float(np.cbrt(np.finfo(float).eps))
    for index, name in enumerate(names):
        center = float(parameters[name])
        step = relative_step * max(1.0, abs(center))
        plus = dict(parameters)
        minus = dict(parameters)
        plus[name] = center + step
        minus[name] = center - step
        jacobian[:, index] = (
            _predict(task, repair, frame, plus)
            - _predict(task, repair, frame, minus)
        ) / (2.0 * step)
    return names, jacobian


@dataclass(frozen=True)
class CandidateEvaluation:
    task_id: str
    method: str
    proposal_id: str
    source: str
    structural_key: str
    expression_key: str
    edit_type: str
    formula: str
    status: str
    failure_reason: str
    parameter_values: dict[str, float]
    train_rmse: float
    validation_rmse: float
    selection_score: float
    complexity: int
    parameter_count: int
    stability_violations: int
    optimizer_evaluations: int
    selected: bool = False

    def to_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parameter_values"] = dict(self.parameter_values)
        return payload


@dataclass(frozen=True)
class MethodEvaluation:
    task_id: str
    task_family: str
    method: str
    candidate_count: int
    valid_candidate_count: int
    selected_proposal_id: str
    selected_source: str
    selected_formula: str
    train_rmse: float
    validation_rmse: float
    locked_rmse: float
    baseline_train_rmse: float
    baseline_validation_rmse: float
    baseline_locked_rmse: float
    validation_improvement_fraction: float
    locked_improvement_fraction: float
    complexity: int
    stability_violations: int
    oracle_patch_in_candidate_pool: bool
    oracle_patch_best_rank: int | None
    exact_patch_expression_match: bool
    corrected_behavior_recovered: bool
    result_status: str

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _baseline_metrics(task: Gate1Task) -> dict[str, float]:
    fit = task.observed.loc[task.observed["partition"].eq("fit")]
    validation = task.observed.loc[task.observed["partition"].eq("validation")]
    return {
        "train": _rmse(fit["target"].to_numpy(float), fit["baseline"].to_numpy(float)),
        "validation": _rmse(
            validation["target"].to_numpy(float),
            validation["baseline"].to_numpy(float),
        ),
        "locked": _rmse(
            task.locked["target"].to_numpy(float),
            task.locked["baseline"].to_numpy(float),
        ),
    }


def _candidate_seed(seed: int, structural_key: str) -> int:
    digest = hashlib.sha256(structural_key.encode("utf-8")).digest()
    return int(seed) ^ int.from_bytes(digest[:4], "little")


_PROJECTED_LINEAR_PARAMETERS = (
    "baseline_scale",
    "intercept",
    "atom_amplitude",
)


def _project_linear_parameters(
    task: Gate1Task,
    repair: TypedRepair,
    frame: Any,
    nonlinear_parameters: Mapping[str, float],
) -> tuple[dict[str, float], np.ndarray]:
    parameters = dict(nonlinear_parameters)
    parameters.update({name: 0.0 for name in _PROJECTED_LINEAR_PARAMETERS})
    parameters["atom_amplitude"] = 1.0
    atom = _predict(task, repair, frame, parameters)
    variables = _frame_variables(task, frame)
    baseline = evaluate_expression(task.baseline_expression, variables)
    design = np.column_stack((baseline, np.ones(len(frame), dtype=float), atom))
    if not np.all(np.isfinite(design)):
        raise ExpressionEvaluationError("Variable-projection design is non-finite.")
    coefficients, *_ = np.linalg.lstsq(
        design,
        frame["target"].to_numpy(float),
        rcond=None,
    )
    parameters.update(
        {
            name: float(coefficients[index])
            for index, name in enumerate(_PROJECTED_LINEAR_PARAMETERS)
        }
    )
    return parameters, design @ coefficients


def _nonlinear_parameter_bounds(
    task: Gate1Task,
    fit: Any,
    names: tuple[str, ...],
    evaluation_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    lower = float(evaluation_config["parameter_lower_bound"])
    upper = float(evaluation_config["parameter_upper_bound"])
    lower_bounds = np.full(len(names), lower, dtype=float)
    upper_bounds = np.full(len(names), upper, dtype=float)
    multiplier = float(evaluation_config.get("data_scaled_bound_multiplier", 2.0))
    input_magnitude = max(
        1.0,
        *(float(np.max(np.abs(fit[name].to_numpy(float)))) for name in task.variables),
    )
    scaled_limit = max(abs(lower), abs(upper), multiplier * input_magnitude)
    for index, name in enumerate(names):
        if "_literal_" in name or name.endswith("_input_shift"):
            lower_bounds[index] = -scaled_limit
            upper_bounds[index] = scaled_limit
    return lower_bounds, upper_bounds


def _location_parameter_start(
    expression: Mapping[str, Any],
    fit: Any,
    nonlinear_names: tuple[str, ...],
) -> np.ndarray | None:
    """Build a data-scale start for explicit affine-location parameters."""

    anchors: dict[str, float] = {}

    def visit(node: Any) -> None:
        if not isinstance(node, Mapping):
            return
        operation = node.get("op")
        if operation == "subtract":
            left = node.get("left")
            right = node.get("right")
            if (
                isinstance(left, Mapping)
                and left.get("op") == "variable"
                and isinstance(right, Mapping)
                and right.get("op") == "parameter"
            ):
                name = str(right.get("name"))
                if "_literal_" in name:
                    anchors[name] = float(fit[str(left.get("name"))].median())
        if operation == "add":
            arguments = node.get("arguments", ())
            variables = [
                item
                for item in arguments
                if isinstance(item, Mapping) and item.get("op") == "variable"
            ]
            parameters = [
                item
                for item in arguments
                if isinstance(item, Mapping)
                and item.get("op") == "parameter"
                and "_literal_" in str(item.get("name"))
            ]
            if len(variables) == 1 and len(parameters) == 1:
                anchors[str(parameters[0]["name"])] = -float(
                    fit[str(variables[0]["name"])].median()
                )
        for value in node.values():
            if isinstance(value, Mapping):
                visit(value)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

    visit(expression)
    if not anchors:
        return None
    return np.asarray([anchors.get(name, 0.0) for name in nonlinear_names], dtype=float)


def _fit_with_variable_projection(
    task: Gate1Task,
    repair: TypedRepair,
    fit: Any,
    *,
    seed: int,
    evaluation_config: Mapping[str, Any],
    initial_parameters: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], int, str]:
    all_names = tuple(parameter_names(repair.expression))
    if not set(_PROJECTED_LINEAR_PARAMETERS).issubset(all_names):
        raise ValueError("Variable projection requires a compiled baseline-aware repair.")
    nonlinear_names = tuple(
        name for name in all_names if name not in _PROJECTED_LINEAR_PARAMETERS
    )
    if not nonlinear_names:
        parameters, _ = _project_linear_parameters(task, repair, fit, {})
        return parameters, 1, ""

    lower, upper = _nonlinear_parameter_bounds(
        task,
        fit,
        nonlinear_names,
        evaluation_config,
    )
    maximum_evaluations = int(evaluation_config["maximum_function_evaluations"])
    multistart_count = max(1, int(evaluation_config["multistart_count"]))
    rng = np.random.default_rng(_candidate_seed(seed, repair.structural_key))
    starts = [np.zeros(len(nonlinear_names), dtype=float)]
    if initial_parameters and len(starts) < multistart_count:
        warm = np.asarray(
            [float(initial_parameters.get(name, 0.0)) for name in nonlinear_names],
            dtype=float,
        )
        if np.all(np.isfinite(warm)):
            starts.append(np.clip(warm, lower, upper))
    location_start = _location_parameter_start(
        repair.expression,
        fit,
        nonlinear_names,
    )
    if location_start is not None and len(starts) < multistart_count:
        starts.append(np.clip(location_start, lower, upper))
    remaining_starts = multistart_count - len(starts)
    local_fraction = float(evaluation_config.get("local_initialization_fraction", 0.75))
    if not 0.0 <= local_fraction <= 1.0:
        raise ValueError("local_initialization_fraction must lie in [0, 1].")
    local_count = int(round(remaining_starts * local_fraction))
    global_count = remaining_starts - local_count
    local_limit = float(evaluation_config.get("local_initialization_bound", 1.5))
    if local_limit <= 0.0:
        raise ValueError("local_initialization_bound must be positive.")
    local_lower = np.maximum(lower, -local_limit)
    local_upper = np.minimum(upper, local_limit)
    starts.extend(
        rng.uniform(local_lower, local_upper)
        for _ in range(local_count)
    )
    starts.extend(
        rng.uniform(lower, upper)
        for _ in range(global_count)
    )
    optimizer_evaluations = 0
    best_vector: np.ndarray | None = None
    best_loss = float("inf")
    failure_reason = ""

    def residual(vector: np.ndarray) -> np.ndarray:
        nonlinear = {
            name: float(vector[index])
            for index, name in enumerate(nonlinear_names)
        }
        try:
            _, prediction = _project_linear_parameters(
                task,
                repair,
                fit,
                nonlinear,
            )
        except (ExpressionEvaluationError, np.linalg.LinAlgError):
            return np.full(len(fit), 1.0e6, dtype=float)
        return prediction - fit["target"].to_numpy(float)

    for start in starts:
        try:
            result = least_squares(
                residual,
                np.clip(start, lower + 1.0e-10, upper - 1.0e-10),
                bounds=(lower, upper),
                max_nfev=maximum_evaluations,
                x_scale="jac",
                xtol=1.0e-10,
                ftol=1.0e-10,
                gtol=1.0e-10,
            )
        except (ValueError, FloatingPointError) as exc:
            failure_reason = str(exc)
            continue
        optimizer_evaluations += int(result.nfev)
        loss = float(np.mean(residual(result.x) ** 2))
        if np.isfinite(loss) and loss < best_loss:
            best_vector = np.asarray(result.x, dtype=float)
            best_loss = loss
    if best_vector is None:
        return {}, optimizer_evaluations, failure_reason or "Variable projection failed."
    nonlinear = {
        name: float(best_vector[index])
        for index, name in enumerate(nonlinear_names)
    }
    parameters, _ = _project_linear_parameters(task, repair, fit, nonlinear)
    return parameters, optimizer_evaluations, ""


def build_extrapolation_guard_frame(
    task: Gate1Task,
    evaluation_config: Mapping[str, Any],
) -> pd.DataFrame:
    expansion = float(
        evaluation_config.get("extrapolation_guard_expansion_factor", 1.5)
    )
    sample_count = int(evaluation_config.get("extrapolation_guard_sample_count", 256))
    if expansion <= 1.0:
        raise ValueError("extrapolation_guard_expansion_factor must exceed one.")
    if sample_count < 2 * len(task.variables):
        raise ValueError(
            "extrapolation_guard_sample_count must cover both ends of every variable."
        )
    observed = task.observed
    lower = observed[list(task.variables)].min(axis=0).to_numpy(float)
    upper = observed[list(task.variables)].max(axis=0).to_numpy(float)
    center = 0.5 * (lower + upper)
    half_span = 0.5 * (upper - lower)
    half_span = np.where(half_span > 1.0e-12, half_span, 1.0)
    guard_lower = center - expansion * half_span
    guard_upper = center + expansion * half_span
    if bool(
        evaluation_config.get(
            "extrapolation_guard_clip_to_locked_domain", False
        )
    ):
        domain_lower = np.asarray(
            [task.definition.locked_ranges[name][0] for name in task.variables],
            dtype=float,
        )
        domain_upper = np.asarray(
            [task.definition.locked_ranges[name][1] for name in task.variables],
            dtype=float,
        )
        guard_lower = np.maximum(guard_lower, domain_lower)
        guard_upper = np.minimum(guard_upper, domain_upper)
        if np.any(guard_lower >= guard_upper):
            raise ValueError(
                "The extrapolation guard does not overlap the locked task domain."
            )
    rng = np.random.default_rng(
        _candidate_seed(0, f"{task.task_id}:extrapolation_guard")
    )
    unit = np.empty((sample_count, len(task.variables)), dtype=float)
    for index in range(len(task.variables)):
        unit[:, index] = (
            rng.permutation(sample_count) + rng.random(sample_count)
        ) / sample_count
    values = guard_lower + unit * (guard_upper - guard_lower)
    for index in range(len(task.variables)):
        values[2 * index] = center
        values[2 * index, index] = guard_lower[index]
        values[2 * index + 1] = center
        values[2 * index + 1, index] = guard_upper[index]
    frame = pd.DataFrame(
        {name: values[:, index] for index, name in enumerate(task.variables)}
    )
    if bool(evaluation_config.get("extrapolation_guard_include_corners", False)):
        maximum_dimensions = int(
            evaluation_config.get("extrapolation_guard_max_corner_dimensions", 8)
        )
        if maximum_dimensions < 1:
            raise ValueError(
                "extrapolation_guard_max_corner_dimensions must be positive."
            )
        if len(task.variables) <= maximum_dimensions:
            corners = np.asarray(
                list(itertools.product(*zip(guard_lower, guard_upper))),
                dtype=float,
            )
            corner_frame = pd.DataFrame(
                {
                    name: corners[:, index]
                    for index, name in enumerate(task.variables)
                }
            )
            frame = pd.concat((frame, corner_frame), ignore_index=True).drop_duplicates(
                ignore_index=True
            )
    return frame


def _extrapolation_guard_frame(
    task: Gate1Task,
    evaluation_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Compatibility wrapper for the original private helper."""

    return build_extrapolation_guard_frame(task, evaluation_config)


def _stability_violations(
    task: Gate1Task,
    repair: TypedRepair,
    parameters: Mapping[str, float],
    evaluation_config: Mapping[str, Any],
) -> int:
    try:
        prediction = _predict(task, repair, task.audit, parameters)
    except ExpressionEvaluationError:
        return 1
    violations = 0
    magnitude_multiplier = float(
        evaluation_config.get("stability_magnitude_multiplier", 20.0)
    )
    if magnitude_multiplier <= 0.0:
        raise ValueError("stability_magnitude_multiplier must be positive.")
    magnitude_limit = max(
        10.0,
        magnitude_multiplier
        * float(np.max(np.abs(task.observed["target"].to_numpy(float)))),
    )
    if np.any(np.abs(prediction) > magnitude_limit):
        violations += 1
    minimum = task.definition.minimum_output
    if minimum is not None and np.any(prediction < float(minimum) - 1.0e-8):
        violations += 1
    monotonicity = dict(task.definition.monotonicity)
    if monotonicity:
        axis_points = int(evaluation_config.get("shape_guard_axis_points", 24))
        context_count = int(evaluation_config.get("shape_guard_context_count", 8))
        relative_tolerance = float(
            evaluation_config.get("shape_guard_relative_tolerance", 1.0e-8)
        )
        if axis_points < 3 or context_count < 1 or relative_tolerance < 0.0:
            raise ValueError("Shape-guard settings are invalid.")
        variables = tuple(task.variables)
        for constrained_variable, direction in monotonicity.items():
            axis = np.linspace(
                task.definition.locked_ranges[constrained_variable][0],
                task.definition.locked_ranges[constrained_variable][1],
                axis_points,
            )
            rng = np.random.default_rng(
                _candidate_seed(
                    0,
                    f"{task.task_id}:shape_guard:{constrained_variable}",
                )
            )
            rows: list[dict[str, float]] = []
            for _ in range(context_count):
                context = {
                    variable: float(
                        rng.uniform(*task.definition.locked_ranges[variable])
                    )
                    for variable in variables
                    if variable != constrained_variable
                }
                rows.extend(
                    {
                        **context,
                        constrained_variable: float(value),
                    }
                    for value in axis
                )
            guard = pd.DataFrame(rows, columns=list(variables))
            try:
                shape_prediction = _predict(task, repair, guard, parameters).reshape(
                    context_count, axis_points
                )
            except ExpressionEvaluationError:
                violations += 1
                continue
            scale = max(1.0, float(np.max(np.abs(shape_prediction))))
            tolerance = relative_tolerance * scale
            differences = np.diff(shape_prediction, axis=1)
            violates_direction = (
                np.any(differences < -tolerance)
                if direction == "increasing"
                else np.any(differences > tolerance)
            )
            if violates_direction:
                violations += 1
    if bool(evaluation_config.get("extrapolation_guard_enabled", False)):
        try:
            guard_prediction = _predict(
                task,
                repair,
                build_extrapolation_guard_frame(task, evaluation_config),
                parameters,
            )
        except ExpressionEvaluationError:
            violations += 1
        else:
            if (
                not np.all(np.isfinite(guard_prediction))
                or np.any(np.abs(guard_prediction) > magnitude_limit)
            ):
                violations += 1
            if minimum is not None and np.any(
                guard_prediction < float(minimum) - 1.0e-8
            ):
                violations += 1
    return violations


def fit_candidate(
    task: Gate1Task,
    repair: TypedRepair,
    *,
    method: str,
    seed: int,
    evaluation_config: Mapping[str, Any],
    initial_parameters: Mapping[str, float] | None = None,
) -> CandidateEvaluation:
    fit = task.observed.loc[task.observed["partition"].eq("fit")]
    validation = task.observed.loc[task.observed["partition"].eq("validation")]
    names = parameter_names(repair.expression)
    lower = float(evaluation_config["parameter_lower_bound"])
    upper = float(evaluation_config["parameter_upper_bound"])
    maximum_evaluations = int(evaluation_config["maximum_function_evaluations"])
    multistart_count = max(1, int(evaluation_config["multistart_count"]))
    parameter_values: dict[str, float] = {}
    optimizer_evaluations = 0
    failure_reason = ""
    status = "valid"

    def residual(vector: np.ndarray) -> np.ndarray:
        parameters = {name: float(vector[index]) for index, name in enumerate(names)}
        try:
            prediction = _predict(task, repair, fit, parameters)
        except ExpressionEvaluationError:
            return np.full(len(fit), 1.0e6, dtype=float)
        return prediction - fit["target"].to_numpy(float)

    optimizer_mode = str(evaluation_config.get("parameter_optimizer", "joint_nls"))
    if names and optimizer_mode == "variable_projection":
        try:
            parameter_values, optimizer_evaluations, failure_reason = (
                _fit_with_variable_projection(
                    task,
                    repair,
                    fit,
                    seed=seed,
                    evaluation_config=evaluation_config,
                    initial_parameters=initial_parameters,
                )
            )
        except (ValueError, ExpressionEvaluationError, np.linalg.LinAlgError) as exc:
            failure_reason = str(exc)
            parameter_values = {}
        if not parameter_values:
            status = "fit_failed"
            parameter_values = {name: 0.0 for name in names}
    elif names:
        rng = np.random.default_rng(_candidate_seed(seed, repair.structural_key))
        starts = [np.zeros(len(names), dtype=float)]
        starts.extend(
            rng.uniform(-1.5, 1.5, len(names))
            for _ in range(multistart_count - 1)
        )
        best_result = None
        best_loss = float("inf")
        for start in starts:
            try:
                result = least_squares(
                    residual,
                    np.clip(start, lower + 1.0e-10, upper - 1.0e-10),
                    bounds=(lower, upper),
                    max_nfev=maximum_evaluations,
                    xtol=1.0e-10,
                    ftol=1.0e-10,
                    gtol=1.0e-10,
                )
            except (ValueError, FloatingPointError) as exc:
                failure_reason = str(exc)
                continue
            optimizer_evaluations += int(result.nfev)
            loss = float(np.mean(residual(result.x) ** 2))
            if np.isfinite(loss) and loss < best_loss:
                best_result = result
                best_loss = loss
        if best_result is None:
            status = "fit_failed"
            parameter_values = {name: 0.0 for name in names}
        else:
            parameter_values = {
                name: float(best_result.x[index]) for index, name in enumerate(names)
            }

    train_rmse = float("inf")
    validation_rmse = float("inf")
    violations = 1
    if status == "valid":
        try:
            train_rmse = _rmse(
                fit["target"].to_numpy(float),
                _predict(task, repair, fit, parameter_values),
            )
            validation_rmse = _rmse(
                validation["target"].to_numpy(float),
                _predict(task, repair, validation, parameter_values),
            )
            violations = _stability_violations(
                task,
                repair,
                parameter_values,
                evaluation_config,
            )
            if violations:
                status = "stability_rejected"
                failure_reason = "Candidate violated an observed-domain stability constraint."
        except ExpressionEvaluationError as exc:
            status = "evaluation_failed"
            failure_reason = str(exc)

    complexity = expression_node_count(repair.expression)
    baseline_validation = _baseline_metrics(task)["validation"]
    complexity_penalty = (
        float(evaluation_config["complexity_penalty_fraction"])
        * baseline_validation
        * complexity
    )
    selection_score = (
        validation_rmse + complexity_penalty if status == "valid" else float("inf")
    )
    return CandidateEvaluation(
        task_id=task.task_id,
        method=method,
        proposal_id=repair.proposal_id,
        source=repair.source,
        structural_key=repair.structural_key,
        expression_key=canonical_expression_key(repair.expression),
        edit_type=repair.edit_type,
        formula=repair_to_text(repair),
        status=status,
        failure_reason=failure_reason,
        parameter_values=parameter_values,
        train_rmse=train_rmse,
        validation_rmse=validation_rmse,
        selection_score=selection_score,
        complexity=complexity,
        parameter_count=len(names),
        stability_violations=violations,
        optimizer_evaluations=optimizer_evaluations,
    )


def evaluate_candidates(
    task: Gate1Task,
    repairs: Iterable[TypedRepair],
    *,
    method: str,
    seed: int,
    evaluation_config: Mapping[str, Any],
    initial_parameters_by_proposal: Mapping[str, Mapping[str, float]] | None = None,
) -> list[CandidateEvaluation]:
    initial_parameters_by_proposal = initial_parameters_by_proposal or {}
    return [
        fit_candidate(
            task,
            repair,
            method=method,
            seed=seed,
            evaluation_config=evaluation_config,
            initial_parameters=initial_parameters_by_proposal.get(repair.proposal_id),
        )
        for repair in repairs
    ]


def select_candidate(
    evaluations: Iterable[CandidateEvaluation],
    *,
    baseline_score: float | None = None,
) -> tuple[list[CandidateEvaluation], CandidateEvaluation | None]:
    materialized = list(evaluations)
    valid = [item for item in materialized if item.status == "valid"]
    if not valid:
        return materialized, None
    best = min(valid, key=lambda item: (item.selection_score, item.complexity, item.proposal_id))
    if baseline_score is not None and best.selection_score >= float(baseline_score):
        return [
            CandidateEvaluation(**{**asdict(item), "selected": False})
            for item in materialized
        ], None
    selected = [
        CandidateEvaluation(**{**asdict(item), "selected": item.proposal_id == best.proposal_id})
        for item in materialized
    ]
    return selected, next(item for item in selected if item.selected)


def _empty_method_evaluation(
    task: Gate1Task,
    method: str,
    evaluations: Iterable[CandidateEvaluation],
) -> MethodEvaluation:
    materialized = list(evaluations)
    baseline = _baseline_metrics(task)
    return MethodEvaluation(
        task_id=task.task_id,
        task_family=task.definition.task_family,
        method=method,
        candidate_count=len(materialized),
        valid_candidate_count=0,
        selected_proposal_id="",
        selected_source="",
        selected_formula="",
        train_rmse=float("nan"),
        validation_rmse=float("nan"),
        locked_rmse=float("nan"),
        baseline_train_rmse=baseline["train"],
        baseline_validation_rmse=baseline["validation"],
        baseline_locked_rmse=baseline["locked"],
        validation_improvement_fraction=float("nan"),
        locked_improvement_fraction=float("nan"),
        complexity=0,
        stability_violations=0,
        oracle_patch_in_candidate_pool=False,
        oracle_patch_best_rank=None,
        exact_patch_expression_match=False,
        corrected_behavior_recovered=False,
        result_status="no_valid_candidate",
    )


def finalize_selected_repair(
    task: Gate1Task,
    method: str,
    evaluations: Iterable[CandidateEvaluation],
    selected: CandidateEvaluation | None,
    repair_by_id: Mapping[str, TypedRepair],
    *,
    recovery_noise_multiplier: float,
    retain_baseline_if_unselected: bool = False,
    locked_failure_penalty_multiplier: float | None = None,
) -> MethodEvaluation:
    materialized = list(evaluations)
    baseline = _baseline_metrics(task)
    if selected is None:
        if retain_baseline_if_unselected:
            recovery_threshold = max(
                float(recovery_noise_multiplier) * task.definition.noise_std,
                0.03 * float(np.std(task.locked["target"].to_numpy(float))),
            )
            return MethodEvaluation(
                task_id=task.task_id,
                task_family=task.definition.task_family,
                method=method,
                candidate_count=len(materialized),
                valid_candidate_count=sum(
                    item.status == "valid" for item in materialized
                ),
                selected_proposal_id="baseline_no_change",
                selected_source="baseline",
                selected_formula="M_base(x)",
                train_rmse=baseline["train"],
                validation_rmse=baseline["validation"],
                locked_rmse=baseline["locked"],
                baseline_train_rmse=baseline["train"],
                baseline_validation_rmse=baseline["validation"],
                baseline_locked_rmse=baseline["locked"],
                validation_improvement_fraction=0.0,
                locked_improvement_fraction=0.0,
                complexity=0,
                stability_violations=0,
                oracle_patch_in_candidate_pool=False,
                oracle_patch_best_rank=None,
                exact_patch_expression_match=False,
                corrected_behavior_recovered=baseline["locked"] <= recovery_threshold,
                result_status="baseline_retained",
            )
        return _empty_method_evaluation(task, method, materialized)
    repair = repair_by_id[selected.proposal_id]
    locked_evaluation_failed = False
    try:
        locked_prediction = _predict(
            task,
            repair,
            task.locked,
            selected.parameter_values,
        )
        locked_rmse = _rmse(
            task.locked["target"].to_numpy(float),
            locked_prediction,
        )
    except ExpressionEvaluationError:
        if locked_failure_penalty_multiplier is None:
            raise
        if locked_failure_penalty_multiplier < 1.0:
            raise ValueError("locked_failure_penalty_multiplier must be at least one.")
        locked_evaluation_failed = True
        locked_rmse = (
            float(locked_failure_penalty_multiplier) * baseline["locked"]
        )
    validation_improvement = (
        (baseline["validation"] - selected.validation_rmse) / baseline["validation"]
        if baseline["validation"] > 0.0
        else 0.0
    )
    locked_improvement = (
        (baseline["locked"] - locked_rmse) / baseline["locked"]
        if baseline["locked"] > 0.0
        else 0.0
    )
    recovery_threshold = max(
        float(recovery_noise_multiplier) * task.definition.noise_std,
        0.03 * float(np.std(task.locked["target"].to_numpy(float))),
    )
    oracle_expression_key = canonical_expression_key(task.oracle_repair.expression)
    ranked_valid = sorted(
        (item for item in materialized if item.status == "valid"),
        key=lambda item: (item.selection_score, item.complexity, item.proposal_id),
    )
    oracle_ranks = [
        index
        for index, item in enumerate(ranked_valid, start=1)
        if item.edit_type != "replace_subtree"
        and item.expression_key == oracle_expression_key
    ]
    return MethodEvaluation(
        task_id=task.task_id,
        task_family=task.definition.task_family,
        method=method,
        candidate_count=len(materialized),
        valid_candidate_count=sum(item.status == "valid" for item in materialized),
        selected_proposal_id=selected.proposal_id,
        selected_source=selected.source,
        selected_formula=selected.formula,
        train_rmse=selected.train_rmse,
        validation_rmse=selected.validation_rmse,
        locked_rmse=locked_rmse,
        baseline_train_rmse=baseline["train"],
        baseline_validation_rmse=baseline["validation"],
        baseline_locked_rmse=baseline["locked"],
        validation_improvement_fraction=validation_improvement,
        locked_improvement_fraction=locked_improvement,
        complexity=selected.complexity,
        stability_violations=(
            selected.stability_violations + int(locked_evaluation_failed)
        ),
        oracle_patch_in_candidate_pool=bool(oracle_ranks),
        oracle_patch_best_rank=min(oracle_ranks) if oracle_ranks else None,
        exact_patch_expression_match=(
            repair.edit_type != "replace_subtree"
            and canonical_expression_key(repair.expression) == oracle_expression_key
        ),
        corrected_behavior_recovered=(
            not locked_evaluation_failed and locked_rmse <= recovery_threshold
        ),
        result_status=(
            "locked_evaluation_failed" if locked_evaluation_failed else "completed"
        ),
    )

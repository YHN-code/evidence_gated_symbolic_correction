from __future__ import annotations

import copy
import hashlib
import itertools
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import qmc

from asrc.model_revision.ast import (
    ExpressionEvaluationError,
    evaluate_expression,
    expression_node_count,
    expression_to_text,
    parameter_names,
)


PathToken = str | int
ExpressionPath = tuple[PathToken, ...]
SUPPORTED_STRUCTURAL_EDITS = frozenset(
    {"insert_additive", "scale_component", "replace_component"}
)


class StructuralRevisionError(ValueError):
    """Raised when a structural edit or benchmark definition is invalid."""


def get_expression_subtree(
    expression: Mapping[str, Any], path: Sequence[PathToken]
) -> dict[str, Any]:
    """Return a defensive copy of the AST node at ``path``."""

    node: Any = expression
    for token in path:
        if isinstance(token, int):
            if not isinstance(node, list) or token < 0 or token >= len(node):
                raise StructuralRevisionError(f"Invalid expression path token {token!r}.")
            node = node[token]
        else:
            if not isinstance(node, Mapping) or token not in node:
                raise StructuralRevisionError(f"Invalid expression path token {token!r}.")
            node = node[token]
    if not isinstance(node, Mapping) or "op" not in node:
        raise StructuralRevisionError("Expression path does not identify an AST node.")
    return copy.deepcopy(dict(node))


def replace_expression_subtree(
    expression: Mapping[str, Any],
    path: Sequence[PathToken],
    replacement: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a copied expression with one path-local subtree replaced."""

    if "op" not in replacement:
        raise StructuralRevisionError("Replacement must be an expression AST node.")
    result: Any = copy.deepcopy(dict(expression))
    if not path:
        return copy.deepcopy(dict(replacement))
    parent = result
    for token in path[:-1]:
        try:
            parent = parent[token]
        except (KeyError, IndexError, TypeError) as exc:
            raise StructuralRevisionError(f"Invalid expression path token {token!r}.") from exc
    final = path[-1]
    try:
        parent[final] = copy.deepcopy(dict(replacement))
    except (KeyError, IndexError, TypeError) as exc:
        raise StructuralRevisionError(f"Invalid expression path token {final!r}.") from exc
    return result


def apply_local_structural_edit(
    expression: Mapping[str, Any],
    path: Sequence[PathToken],
    edit_type: str,
    patch: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply an additive, scaling, or replacement edit to one AST location."""

    if edit_type not in SUPPORTED_STRUCTURAL_EDITS:
        raise StructuralRevisionError(f"Unsupported structural edit {edit_type!r}.")
    current = get_expression_subtree(expression, path)
    if edit_type == "insert_additive":
        replacement = {"op": "add", "arguments": [current, copy.deepcopy(dict(patch))]}
    elif edit_type == "scale_component":
        replacement = {
            "op": "multiply",
            "arguments": [current, copy.deepcopy(dict(patch))],
        }
    else:
        replacement = copy.deepcopy(dict(patch))
    return replace_expression_subtree(expression, path, replacement)


def _constant(value: float) -> dict[str, Any]:
    return {"op": "constant", "value": float(value)}


def _variable(name: str) -> dict[str, Any]:
    return {"op": "variable", "name": str(name)}


def _parameter(name: str) -> dict[str, Any]:
    return {"op": "parameter", "name": str(name)}


def _add(*arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {"op": "add", "arguments": [copy.deepcopy(dict(item)) for item in arguments]}


def _multiply(*arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "op": "multiply",
        "arguments": [copy.deepcopy(dict(item)) for item in arguments],
    }


def _power(left: Mapping[str, Any], exponent: float) -> dict[str, Any]:
    return {"op": "power", "left": copy.deepcopy(dict(left)), "right": _constant(exponent)}


def _unary(operation: str, argument: Mapping[str, Any]) -> dict[str, Any]:
    return {"op": operation, "argument": copy.deepcopy(dict(argument))}


@dataclass(frozen=True)
class StructuralCandidate:
    proposal_id: str
    source: str
    location: str
    edit_type: str
    family_id: str
    expression: dict[str, Any]
    parameter_bounds: dict[str, tuple[float, float]]
    variable_signature: tuple[str, ...]

    @property
    def complexity(self) -> int:
        edit_cost = {
            "insert_additive": 1,
            "scale_component": 1,
            "replace_component": 2,
        }[self.edit_type]
        return expression_node_count(self.expression) + edit_cost

    @property
    def structural_key(self) -> str:
        return "|".join(
            [self.location, self.edit_type, self.family_id, expression_to_text(self.expression)]
        )


@dataclass(frozen=True)
class StructuralRevisionTask:
    task_id: str
    mechanism: str
    variables: tuple[str, ...]
    baseline_response: dict[str, Any]
    baseline_diagnostic: dict[str, Any]
    response_paths: dict[str, ExpressionPath]
    diagnostic_paths: dict[str, ExpressionPath]
    true_location: str
    true_edit_type: str
    true_family_id: str
    true_patch: dict[str, Any]
    true_parameters: dict[str, float]
    expected_library_coverage: bool
    fit_ranges: dict[str, tuple[float, float]]
    locked_ranges: dict[str, tuple[float, float]]
    noise_standard_deviation: float
    diagnostic_noise_standard_deviation: float
    data: pd.DataFrame

    @property
    def locations(self) -> tuple[str, ...]:
        return tuple(sorted(self.response_paths))


@dataclass(frozen=True)
class StructuralCandidateEvaluation:
    task_id: str
    method: str
    proposal_id: str
    source: str
    location: str
    edit_type: str
    family_id: str
    expression_text: str
    complexity: int
    parameter_values: dict[str, float]
    fit_primary_rmse: float
    fit_diagnostic_rmse: float
    validation_primary_rmse: float
    validation_diagnostic_rmse: float
    locked_primary_rmse: float
    locked_diagnostic_rmse: float
    selection_score: float
    physical_violations: int
    adequacy_accepted: bool
    fit_success: bool

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructuralMethodResult:
    task_id: str
    mechanism: str
    method: str
    candidate_count: int
    wall_time_seconds: float
    selected_proposal_id: str
    selected_location: str
    selected_edit_type: str
    selected_family_id: str
    selected_expression: str
    complexity: int
    validation_primary_rmse: float
    validation_diagnostic_rmse: float
    locked_primary_rmse: float
    locked_diagnostic_rmse: float
    baseline_locked_primary_rmse: float
    baseline_locked_diagnostic_rmse: float
    locked_primary_improvement_fraction: float
    locked_joint_normalized_rmse: float
    location_recovered: bool
    edit_type_recovered: bool
    joint_structure_recovered: bool
    family_recovered: bool
    physical_violations: int
    adequacy_accepted: bool
    expected_library_coverage: bool
    parameter_values: dict[str, float]

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _model_expression(
    components: Mapping[str, Mapping[str, Any]],
    weights: Mapping[str, float],
    intercept: float,
) -> tuple[dict[str, Any], dict[str, ExpressionPath]]:
    arguments: list[dict[str, Any]] = [_constant(intercept)]
    paths: dict[str, ExpressionPath] = {}
    for index, name in enumerate(sorted(components), start=1):
        arguments.append(_multiply(_constant(float(weights[name])), components[name]))
        paths[name] = ("arguments", index, "arguments", 1)
    return {"op": "add", "arguments": arguments}, paths


def _family_specs(variables: Sequence[str]) -> list[StructuralCandidate]:
    """Build a task-neutral, low-order family library over declared variables."""

    templates: list[tuple[str, str, tuple[str, ...], dict[str, Any], dict[str, tuple[float, float]]]] = []
    amplitude = (-12.0, 12.0)
    slope = (-8.0, 8.0)
    positive_rate = (0.05, 8.0)

    templates.append(
        ("constant", "insert_additive", (), _parameter("a0"), {"a0": amplitude})
    )
    templates.append(
        ("constant", "replace_component", (), _parameter("a0"), {"a0": amplitude})
    )
    for name in variables:
        variable = _variable(name)
        templates.extend(
            [
                (
                    "affine",
                    "insert_additive",
                    (name,),
                    _add(_parameter("a0"), _multiply(_parameter("a1"), variable)),
                    {"a0": amplitude, "a1": amplitude},
                ),
                (
                    "affine",
                    "replace_component",
                    (name,),
                    _add(_parameter("a0"), _multiply(_parameter("a1"), variable)),
                    {"a0": amplitude, "a1": amplitude},
                ),
                (
                    "quadratic",
                    "insert_additive",
                    (name,),
                    _add(
                        _parameter("a0"),
                        _multiply(_parameter("a1"), variable),
                        _multiply(_parameter("a2"), _power(variable, 2.0)),
                    ),
                    {"a0": amplitude, "a1": amplitude, "a2": amplitude},
                ),
                (
                    "quadratic",
                    "replace_component",
                    (name,),
                    _add(
                        _parameter("a0"),
                        _multiply(_parameter("a1"), variable),
                        _multiply(_parameter("a2"), _power(variable, 2.0)),
                    ),
                    {"a0": amplitude, "a1": amplitude, "a2": amplitude},
                ),
                (
                    "tanh_transition",
                    "insert_additive",
                    (name,),
                    _add(
                        _parameter("a0"),
                        _multiply(
                            _parameter("a1"),
                            _unary("tanh", _multiply(_parameter("a2"), variable)),
                        ),
                    ),
                    {"a0": amplitude, "a1": amplitude, "a2": slope},
                ),
                (
                    "tanh_transition",
                    "replace_component",
                    (name,),
                    _add(
                        _parameter("a0"),
                        _multiply(
                            _parameter("a1"),
                            _unary("tanh", _multiply(_parameter("a2"), variable)),
                        ),
                    ),
                    {"a0": amplitude, "a1": amplitude, "a2": slope},
                ),
                (
                    "exponential_decay",
                    "replace_component",
                    (name,),
                    _multiply(
                        _parameter("a0"),
                        _unary(
                            "exp",
                            {"op": "negate", "argument": _multiply(_parameter("a1"), variable)},
                        ),
                    ),
                    {"a0": (0.0, 20.0), "a1": positive_rate},
                ),
                (
                    "unit_linear",
                    "scale_component",
                    (name,),
                    _add(_constant(1.0), _multiply(_parameter("a0"), variable)),
                    {"a0": slope},
                ),
                (
                    "unit_tanh",
                    "scale_component",
                    (name,),
                    _add(
                        _constant(1.0),
                        _multiply(
                            _parameter("a0"),
                            _unary("tanh", _multiply(_parameter("a1"), variable)),
                        ),
                    ),
                    {"a0": (-2.0, 2.0), "a1": positive_rate},
                ),
                (
                    "exponential_scale",
                    "scale_component",
                    (name,),
                    _unary("exp", _multiply(_parameter("a0"), variable)),
                    {"a0": (-4.0, 4.0)},
                ),
            ]
        )
    for left, right in itertools.combinations(variables, 2):
        left_node = _variable(left)
        right_node = _variable(right)
        templates.extend(
            [
                (
                    "bilinear",
                    "insert_additive",
                    (left, right),
                    _add(
                        _parameter("a0"),
                        _multiply(_parameter("a1"), left_node),
                        _multiply(_parameter("a2"), right_node),
                        _multiply(_parameter("a3"), left_node, right_node),
                    ),
                    {"a0": amplitude, "a1": amplitude, "a2": amplitude, "a3": amplitude},
                ),
                (
                    "bilinear",
                    "replace_component",
                    (left, right),
                    _add(
                        _parameter("a0"),
                        _multiply(_parameter("a1"), left_node),
                        _multiply(_parameter("a2"), right_node),
                        _multiply(_parameter("a3"), left_node, right_node),
                    ),
                    {"a0": amplitude, "a1": amplitude, "a2": amplitude, "a3": amplitude},
                ),
                (
                    "trigonometric_interaction",
                    "insert_additive",
                    (left, right),
                    _multiply(
                        _parameter("a0"),
                        right_node,
                        _power(
                            _unary("sin", _multiply(_parameter("a1"), left_node)),
                            2.0,
                        ),
                    ),
                    {"a0": amplitude, "a1": positive_rate},
                ),
                (
                    "unit_bilinear",
                    "scale_component",
                    (left, right),
                    _add(
                        _constant(1.0),
                        _multiply(_parameter("a0"), left_node, right_node),
                    ),
                    {"a0": slope},
                ),
            ]
        )

    candidates: list[StructuralCandidate] = []
    for index, (family_id, edit_type, signature, expression, bounds) in enumerate(
        templates, start=1
    ):
        candidates.append(
            StructuralCandidate(
                proposal_id=f"family_{index:03d}",
                source="bounded_generic_library",
                location="",
                edit_type=edit_type,
                family_id=family_id,
                expression=expression,
                parameter_bounds=bounds,
                variable_signature=signature,
            )
        )
    return candidates


def generate_structural_candidates(
    task: StructuralRevisionTask,
    *,
    locations: Iterable[str] | None = None,
    output_only: bool = False,
) -> tuple[StructuralCandidate, ...]:
    """Cross a task-neutral family library with edit location and edit type."""

    selected_locations = ("model_output",) if output_only else tuple(locations or task.locations)
    unknown = set(selected_locations) - set(task.locations) - {"model_output"}
    if unknown:
        raise StructuralRevisionError(f"Unknown edit locations: {sorted(unknown)}")
    output: list[StructuralCandidate] = []
    for location in selected_locations:
        for template in _family_specs(task.variables):
            digest = hashlib.sha1(
                f"{location}|{template.edit_type}|{template.family_id}|{template.variable_signature}".encode(
                    "utf-8"
                )
            ).hexdigest()[:10]
            output.append(
                StructuralCandidate(
                    proposal_id=f"edit_{digest}",
                    source=template.source,
                    location=location,
                    edit_type=template.edit_type,
                    family_id=template.family_id,
                    expression=template.expression,
                    parameter_bounds=template.parameter_bounds,
                    variable_signature=template.variable_signature,
                )
            )
    return tuple(output)


def _edited_expressions(
    task: StructuralRevisionTask,
    candidate: StructuralCandidate,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if candidate.location == "model_output":
        response = apply_local_structural_edit(
            task.baseline_response, (), candidate.edit_type, candidate.expression
        )
        return response, copy.deepcopy(task.baseline_diagnostic)
    return (
        apply_local_structural_edit(
            task.baseline_response,
            task.response_paths[candidate.location],
            candidate.edit_type,
            candidate.expression,
        ),
        apply_local_structural_edit(
            task.baseline_diagnostic,
            task.diagnostic_paths[candidate.location],
            candidate.edit_type,
            candidate.expression,
        ),
    )


def predict_structural_candidate(
    task: StructuralRevisionTask,
    candidate: StructuralCandidate,
    frame: pd.DataFrame,
    parameters: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    variables = {name: frame[name].to_numpy(float) for name in task.variables}
    response, diagnostic = _edited_expressions(task, candidate)
    return (
        evaluate_expression(response, variables, parameters),
        evaluate_expression(diagnostic, variables, parameters),
    )


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(observed) - np.asarray(predicted)) ** 2)))


def _candidate_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def _audit_frame(task: StructuralRevisionTask, points_per_axis: int = 13) -> pd.DataFrame:
    axes = [
        np.linspace(*task.locked_ranges[name], points_per_axis)
        for name in task.variables
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return pd.DataFrame(
        {name: values.reshape(-1) for name, values in zip(task.variables, mesh)}
    )


def fit_structural_candidate(
    task: StructuralRevisionTask,
    candidate: StructuralCandidate,
    *,
    method: str,
    seed: int,
    diagnostic_weight: float,
    complexity_penalty: float = 0.002,
    acceptance_noise_multiplier: float = 1.75,
    restarts: int = 5,
    max_nfev: int = 2500,
) -> StructuralCandidateEvaluation:
    fit = task.data.loc[task.data["partition"].eq("fit")]
    validation = task.data.loc[task.data["partition"].eq("validation")]
    locked = task.data.loc[task.data["partition"].eq("locked")]
    names = parameter_names(candidate.expression)
    lower = np.asarray([candidate.parameter_bounds[name][0] for name in names], dtype=float)
    upper = np.asarray([candidate.parameter_bounds[name][1] for name in names], dtype=float)
    primary_scale = max(float(np.std(fit["target"])), task.noise_standard_deviation, 0.1)
    diagnostic_scale = max(
        float(np.std(fit["diagnostic_target"])),
        task.diagnostic_noise_standard_deviation,
        0.1,
    )

    def residual(vector: np.ndarray) -> np.ndarray:
        parameters = dict(zip(names, vector))
        try:
            primary, diagnostic = predict_structural_candidate(
                task, candidate, fit, parameters
            )
        except (ExpressionEvaluationError, FloatingPointError, OverflowError):
            return np.full(len(fit) * 2, 1.0e6, dtype=float)
        pieces = [(primary - fit["target"].to_numpy(float)) / primary_scale]
        if diagnostic_weight > 0.0:
            pieces.append(
                np.sqrt(diagnostic_weight)
                * (diagnostic - fit["diagnostic_target"].to_numpy(float))
                / diagnostic_scale
            )
        return np.concatenate(pieces)

    starts = [0.5 * (lower + upper)]
    if restarts > 1:
        sampler = qmc.LatinHypercube(
            d=len(names),
            seed=_candidate_seed(seed, candidate.structural_key),
        )
        starts.extend(
            qmc.scale(sampler.random(n=restarts - 1), lower, upper)
        )
    best: Any = None
    for start in starts:
        try:
            result = least_squares(
                residual,
                start,
                bounds=(lower, upper),
                max_nfev=max_nfev,
                method="trf",
            )
        except (ValueError, FloatingPointError, OverflowError):
            continue
        if best is None or float(np.dot(result.fun, result.fun)) < float(
            np.dot(best.fun, best.fun)
        ):
            best = result
    if best is None:
        parameters = {name: float(0.5 * (lo + hi)) for name, lo, hi in zip(names, lower, upper)}
        fit_success = False
    else:
        parameters = {name: float(value) for name, value in zip(names, best.x)}
        fit_success = bool(best.success)

    def metrics(frame: pd.DataFrame) -> tuple[float, float]:
        try:
            primary, diagnostic = predict_structural_candidate(
                task, candidate, frame, parameters
            )
        except (ExpressionEvaluationError, FloatingPointError, OverflowError):
            return 1.0e6, 1.0e6
        return (
            _rmse(frame["target"].to_numpy(float), primary),
            _rmse(frame["diagnostic_target"].to_numpy(float), diagnostic),
        )

    fit_primary, fit_diagnostic = metrics(fit)
    validation_primary, validation_diagnostic = metrics(validation)
    locked_primary, locked_diagnostic = metrics(locked)
    violations = 0
    try:
        audit_primary, audit_diagnostic = predict_structural_candidate(
            task, candidate, _audit_frame(task), parameters
        )
        violations += int(np.any(~np.isfinite(audit_primary)))
        violations += int(np.any(~np.isfinite(audit_diagnostic)))
        violations += int(np.any(audit_primary <= 0.0))
    except (ExpressionEvaluationError, FloatingPointError, OverflowError):
        violations += 3
    selection_score = (
        validation_primary / primary_scale
        + diagnostic_weight * validation_diagnostic / diagnostic_scale
        + complexity_penalty * candidate.complexity
        + 100.0 * violations
    )
    adequacy_accepted = bool(
        validation_primary
        <= acceptance_noise_multiplier * task.noise_standard_deviation
        and validation_diagnostic
        <= acceptance_noise_multiplier * task.diagnostic_noise_standard_deviation
        and violations == 0
    )
    return StructuralCandidateEvaluation(
        task_id=task.task_id,
        method=method,
        proposal_id=candidate.proposal_id,
        source=candidate.source,
        location=candidate.location,
        edit_type=candidate.edit_type,
        family_id=candidate.family_id,
        expression_text=expression_to_text(candidate.expression),
        complexity=candidate.complexity,
        parameter_values=parameters,
        fit_primary_rmse=fit_primary,
        fit_diagnostic_rmse=fit_diagnostic,
        validation_primary_rmse=validation_primary,
        validation_diagnostic_rmse=validation_diagnostic,
        locked_primary_rmse=locked_primary,
        locked_diagnostic_rmse=locked_diagnostic,
        selection_score=selection_score,
        physical_violations=violations,
        adequacy_accepted=adequacy_accepted,
        fit_success=fit_success,
    )


def evaluate_structural_method(
    task: StructuralRevisionTask,
    candidates: Sequence[StructuralCandidate],
    *,
    method: str,
    seed: int,
    diagnostic_weight: float,
    complexity_penalty: float = 0.002,
    acceptance_noise_multiplier: float = 1.75,
    restarts: int = 5,
    max_nfev: int = 2500,
) -> tuple[tuple[StructuralCandidateEvaluation, ...], StructuralMethodResult]:
    started = perf_counter()
    evaluations = tuple(
        fit_structural_candidate(
            task,
            candidate,
            method=method,
            seed=seed,
            diagnostic_weight=diagnostic_weight,
            complexity_penalty=complexity_penalty,
            acceptance_noise_multiplier=acceptance_noise_multiplier,
            restarts=restarts,
            max_nfev=max_nfev,
        )
        for candidate in candidates
    )
    if not evaluations:
        raise StructuralRevisionError("At least one candidate is required.")
    selected = min(
        evaluations,
        key=lambda item: (
            item.selection_score,
            item.complexity,
            item.proposal_id,
        ),
    )
    locked = task.data.loc[task.data["partition"].eq("locked")]
    variables = {name: locked[name].to_numpy(float) for name in task.variables}
    baseline_primary = evaluate_expression(task.baseline_response, variables)
    baseline_diagnostic = evaluate_expression(task.baseline_diagnostic, variables)
    baseline_primary_rmse = _rmse(locked["target"].to_numpy(float), baseline_primary)
    baseline_diagnostic_rmse = _rmse(
        locked["diagnostic_target"].to_numpy(float), baseline_diagnostic
    )
    primary_scale = max(float(np.std(locked["target"])), 0.1)
    diagnostic_scale = max(float(np.std(locked["diagnostic_target"])), 0.1)
    result = StructuralMethodResult(
        task_id=task.task_id,
        mechanism=task.mechanism,
        method=method,
        candidate_count=len(candidates),
        wall_time_seconds=perf_counter() - started,
        selected_proposal_id=selected.proposal_id,
        selected_location=selected.location,
        selected_edit_type=selected.edit_type,
        selected_family_id=selected.family_id,
        selected_expression=selected.expression_text,
        complexity=selected.complexity,
        validation_primary_rmse=selected.validation_primary_rmse,
        validation_diagnostic_rmse=selected.validation_diagnostic_rmse,
        locked_primary_rmse=selected.locked_primary_rmse,
        locked_diagnostic_rmse=selected.locked_diagnostic_rmse,
        baseline_locked_primary_rmse=baseline_primary_rmse,
        baseline_locked_diagnostic_rmse=baseline_diagnostic_rmse,
        locked_primary_improvement_fraction=(
            (baseline_primary_rmse - selected.locked_primary_rmse)
            / baseline_primary_rmse
            if baseline_primary_rmse > 0.0
            else 0.0
        ),
        locked_joint_normalized_rmse=float(
            np.sqrt(
                0.5
                * (
                    (selected.locked_primary_rmse / primary_scale) ** 2
                    + (selected.locked_diagnostic_rmse / diagnostic_scale) ** 2
                )
            )
        ),
        location_recovered=selected.location == task.true_location,
        edit_type_recovered=selected.edit_type == task.true_edit_type,
        joint_structure_recovered=(
            selected.location == task.true_location
            and selected.edit_type == task.true_edit_type
        ),
        family_recovered=selected.family_id == task.true_family_id,
        physical_violations=selected.physical_violations,
        adequacy_accepted=selected.adequacy_accepted,
        expected_library_coverage=task.expected_library_coverage,
        parameter_values=selected.parameter_values,
    )
    return evaluations, result


def baseline_method_result(task: StructuralRevisionTask) -> StructuralMethodResult:
    locked = task.data.loc[task.data["partition"].eq("locked")]
    validation = task.data.loc[task.data["partition"].eq("validation")]

    def metrics(frame: pd.DataFrame) -> tuple[float, float]:
        variables = {name: frame[name].to_numpy(float) for name in task.variables}
        return (
            _rmse(
                frame["target"].to_numpy(float),
                evaluate_expression(task.baseline_response, variables),
            ),
            _rmse(
                frame["diagnostic_target"].to_numpy(float),
                evaluate_expression(task.baseline_diagnostic, variables),
            ),
        )

    validation_primary, validation_diagnostic = metrics(validation)
    locked_primary, locked_diagnostic = metrics(locked)
    primary_scale = max(float(np.std(locked["target"])), 0.1)
    diagnostic_scale = max(float(np.std(locked["diagnostic_target"])), 0.1)
    return StructuralMethodResult(
        task_id=task.task_id,
        mechanism=task.mechanism,
        method="baseline",
        candidate_count=0,
        wall_time_seconds=0.0,
        selected_proposal_id="none",
        selected_location="none",
        selected_edit_type="none",
        selected_family_id="none",
        selected_expression=expression_to_text(task.baseline_response),
        complexity=expression_node_count(task.baseline_response),
        validation_primary_rmse=validation_primary,
        validation_diagnostic_rmse=validation_diagnostic,
        locked_primary_rmse=locked_primary,
        locked_diagnostic_rmse=locked_diagnostic,
        baseline_locked_primary_rmse=locked_primary,
        baseline_locked_diagnostic_rmse=locked_diagnostic,
        locked_primary_improvement_fraction=0.0,
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
        physical_violations=0,
        adequacy_accepted=False,
        expected_library_coverage=task.expected_library_coverage,
        parameter_values={},
    )


def _sample_partition(
    rng: np.random.Generator,
    variables: Sequence[str],
    ranges: Mapping[str, tuple[float, float]],
    size: int,
) -> pd.DataFrame:
    # Independent stratification retains broad coverage without encoding a hidden law.
    output: dict[str, np.ndarray] = {}
    for name in variables:
        low, high = ranges[name]
        strata = (np.arange(size, dtype=float) + rng.random(size)) / size
        rng.shuffle(strata)
        output[name] = low + (high - low) * strata
    return pd.DataFrame(output)


def sample_structural_domain(
    task: StructuralRevisionTask,
    *,
    size: int,
    seed: int,
) -> pd.DataFrame:
    """Sample a new design pool independently over the declared audit domain."""

    if size < 1:
        raise ValueError("size must be positive.")
    return _sample_partition(
        np.random.default_rng(seed), task.variables, task.locked_ranges, size
    )


def _attach_targets(
    frame: pd.DataFrame,
    *,
    task_stub: StructuralRevisionTask,
    partition: str,
    rng: np.random.Generator,
    noisy: bool,
) -> pd.DataFrame:
    variables = {name: frame[name].to_numpy(float) for name in task_stub.variables}
    true_response_ast = apply_local_structural_edit(
        task_stub.baseline_response,
        task_stub.response_paths[task_stub.true_location],
        task_stub.true_edit_type,
        task_stub.true_patch,
    )
    true_diagnostic_ast = apply_local_structural_edit(
        task_stub.baseline_diagnostic,
        task_stub.diagnostic_paths[task_stub.true_location],
        task_stub.true_edit_type,
        task_stub.true_patch,
    )
    result = frame.copy()
    result["baseline"] = evaluate_expression(task_stub.baseline_response, variables)
    result["diagnostic_baseline"] = evaluate_expression(
        task_stub.baseline_diagnostic, variables
    )
    result["target"] = evaluate_expression(
        true_response_ast, variables, task_stub.true_parameters
    )
    result["diagnostic_target"] = evaluate_expression(
        true_diagnostic_ast, variables, task_stub.true_parameters
    )
    if noisy:
        result["target"] += rng.normal(
            0.0, task_stub.noise_standard_deviation, len(result)
        )
        result["diagnostic_target"] += rng.normal(
            0.0, task_stub.diagnostic_noise_standard_deviation, len(result)
        )
    result["partition"] = partition
    return result


def reveal_structural_observations(
    task: StructuralRevisionTask,
    design: pd.DataFrame,
    *,
    seed: int,
    noisy: bool = True,
    partition: str = "validation",
) -> pd.DataFrame:
    """Query the controlled oracle at newly designed points.

    In an application this call is replaced by a laboratory test, high-fidelity
    simulation, or another declared evidence source.
    """

    missing = set(task.variables) - set(design.columns)
    if missing:
        raise StructuralRevisionError(
            f"Design frame is missing variables: {sorted(missing)}"
        )
    return _attach_targets(
        design.loc[:, list(task.variables)].copy(),
        task_stub=task,
        partition=partition,
        rng=np.random.default_rng(seed),
        noisy=noisy,
    )


def _task_stub(
    *,
    task_id: str,
    mechanism: str,
    variables: tuple[str, ...],
    components: Mapping[str, Mapping[str, Any]],
    diagnostic_weights: Mapping[str, float],
    true_location: str,
    true_edit_type: str,
    true_family_id: str,
    true_patch: Mapping[str, Any],
    true_parameters: Mapping[str, float],
    expected_library_coverage: bool = True,
    fit_ranges: Mapping[str, tuple[float, float]],
    locked_ranges: Mapping[str, tuple[float, float]],
    noise: float,
    diagnostic_noise: float,
) -> StructuralRevisionTask:
    response, response_paths = _model_expression(
        components, {name: 1.0 for name in components}, intercept=0.0
    )
    diagnostic, diagnostic_paths = _model_expression(
        components, diagnostic_weights, intercept=0.0
    )
    return StructuralRevisionTask(
        task_id=task_id,
        mechanism=mechanism,
        variables=variables,
        baseline_response=response,
        baseline_diagnostic=diagnostic,
        response_paths=response_paths,
        diagnostic_paths=diagnostic_paths,
        true_location=true_location,
        true_edit_type=true_edit_type,
        true_family_id=true_family_id,
        true_patch=copy.deepcopy(dict(true_patch)),
        true_parameters={str(name): float(value) for name, value in true_parameters.items()},
        expected_library_coverage=bool(expected_library_coverage),
        fit_ranges={str(name): tuple(value) for name, value in fit_ranges.items()},
        locked_ranges={str(name): tuple(value) for name, value in locked_ranges.items()},
        noise_standard_deviation=float(noise),
        diagnostic_noise_standard_deviation=float(diagnostic_noise),
        data=pd.DataFrame(),
    )


def structural_revision_task_definitions() -> tuple[StructuralRevisionTask, ...]:
    """Return three mechanism-level tasks spanning the supported edit types."""

    beta = _variable("beta")
    pressure = _variable("pressure")
    state = _variable("state")
    damage = _variable("damage")
    return (
        _task_stub(
            task_id="SR01",
            mechanism="orientation-pressure coupling",
            variables=("beta", "pressure"),
            components={
                "matrix_strength": _constant(10.0),
                "pressure_resistance": _multiply(_constant(2.2), pressure),
                "orientation_effect": _multiply(
                    _constant(1.1), _unary("cos", _multiply(_constant(2.0), beta))
                ),
            },
            diagnostic_weights={
                "matrix_strength": 0.25,
                "pressure_resistance": 0.8,
                "orientation_effect": 1.7,
            },
            true_location="orientation_effect",
            true_edit_type="insert_additive",
            true_family_id="trigonometric_interaction",
            true_patch=_multiply(
                _parameter("a0"),
                pressure,
                _power(
                    _unary("sin", _multiply(_parameter("a1"), beta)), 2.0
                ),
            ),
            true_parameters={"a0": 2.6, "a1": 2.0},
            fit_ranges={"beta": (0.0, np.pi / 2.0), "pressure": (0.0, 0.65)},
            locked_ranges={"beta": (0.0, np.pi / 2.0), "pressure": (0.0, 1.0)},
            noise=0.045,
            diagnostic_noise=0.045,
        ),
        _task_stub(
            task_id="SR02",
            mechanism="pressure-dependent friction mobilization",
            variables=("pressure", "state"),
            components={
                "matrix_strength": _constant(6.0),
                "pressure_resistance": _multiply(_constant(3.0), pressure),
                "state_effect": _multiply(_constant(0.9), state),
            },
            diagnostic_weights={
                "matrix_strength": 0.3,
                "pressure_resistance": 1.8,
                "state_effect": 0.65,
            },
            true_location="pressure_resistance",
            true_edit_type="scale_component",
            true_family_id="unit_tanh",
            true_patch=_add(
                _constant(1.0),
                _multiply(
                    _parameter("a0"),
                    _unary("tanh", _multiply(_parameter("a1"), pressure)),
                ),
            ),
            true_parameters={"a0": 0.7, "a1": 2.4},
            fit_ranges={"pressure": (0.0, 0.7), "state": (0.0, 1.0)},
            locked_ranges={"pressure": (0.0, 1.2), "state": (0.0, 1.0)},
            noise=0.035,
            diagnostic_noise=0.035,
        ),
        _task_stub(
            task_id="SR03",
            mechanism="nonlinear cohesion softening",
            variables=("damage", "pressure"),
            components={
                "cohesion": _multiply(
                    _constant(4.0),
                    {"op": "subtract", "left": _constant(1.0), "right": damage},
                ),
                "matrix_strength": _constant(5.0),
                "pressure_resistance": _multiply(_constant(2.4), pressure),
            },
            diagnostic_weights={
                "cohesion": 1.9,
                "matrix_strength": 0.25,
                "pressure_resistance": 0.7,
            },
            true_location="cohesion",
            true_edit_type="replace_component",
            true_family_id="exponential_decay",
            true_patch=_multiply(
                _parameter("a0"),
                _unary(
                    "exp",
                    {"op": "negate", "argument": _multiply(_parameter("a1"), damage)},
                ),
            ),
            true_parameters={"a0": 4.0, "a1": 2.2},
            fit_ranges={"damage": (0.0, 0.6), "pressure": (0.0, 0.8)},
            locked_ranges={"damage": (0.0, 1.0), "pressure": (0.0, 1.2)},
            noise=0.04,
            diagnostic_noise=0.04,
        ),
        _task_stub(
            task_id="SR04",
            mechanism="piecewise-linear cohesion softening outside the frozen library",
            variables=("damage", "pressure"),
            components={
                "cohesion": _multiply(
                    _constant(4.0),
                    {"op": "subtract", "left": _constant(1.0), "right": damage},
                ),
                "matrix_strength": _constant(5.0),
                "pressure_resistance": _multiply(_constant(2.4), pressure),
            },
            diagnostic_weights={
                "cohesion": 1.9,
                "matrix_strength": 0.25,
                "pressure_resistance": 0.7,
            },
            true_location="cohesion",
            true_edit_type="replace_component",
            true_family_id="piecewise_linear_unavailable",
            true_patch=_add(
                _parameter("a0"),
                _multiply(_parameter("a1"), damage),
                _multiply(
                    _parameter("a2"),
                    _unary(
                        "abs",
                        {"op": "subtract", "left": damage, "right": _parameter("a3")},
                    ),
                ),
            ),
            true_parameters={"a0": 4.35, "a1": -2.0, "a2": -0.8, "a3": 0.45},
            expected_library_coverage=False,
            fit_ranges={"damage": (0.0, 0.75), "pressure": (0.0, 0.8)},
            locked_ranges={"damage": (0.0, 1.0), "pressure": (0.0, 1.2)},
            noise=0.015,
            diagnostic_noise=0.015,
        ),
    )


def build_structural_revision_suite(
    *,
    seed: int,
    fit_size: int = 48,
    validation_size: int = 32,
    locked_size: int = 128,
    noise_multiplier: float = 1.0,
) -> tuple[StructuralRevisionTask, ...]:
    if noise_multiplier <= 0.0:
        raise ValueError("noise_multiplier must be positive.")
    tasks: list[StructuralRevisionTask] = []
    for index, definition in enumerate(structural_revision_task_definitions()):
        definition = replace(
            definition,
            noise_standard_deviation=(
                definition.noise_standard_deviation * float(noise_multiplier)
            ),
            diagnostic_noise_standard_deviation=(
                definition.diagnostic_noise_standard_deviation
                * float(noise_multiplier)
            ),
        )
        rng = np.random.default_rng(seed + 1009 * (index + 1))
        fit = _attach_targets(
            _sample_partition(rng, definition.variables, definition.fit_ranges, fit_size),
            task_stub=definition,
            partition="fit",
            rng=rng,
            noisy=True,
        )
        validation = _attach_targets(
            _sample_partition(
                rng, definition.variables, definition.fit_ranges, validation_size
            ),
            task_stub=definition,
            partition="validation",
            rng=rng,
            noisy=True,
        )
        locked = _attach_targets(
            _sample_partition(
                rng, definition.variables, definition.locked_ranges, locked_size
            ),
            task_stub=definition,
            partition="locked",
            rng=rng,
            noisy=False,
        )
        tasks.append(
            StructuralRevisionTask(
                **{
                    **definition.__dict__,
                    "data": pd.concat([fit, validation, locked], ignore_index=True),
                }
            )
        )
    return tuple(tasks)

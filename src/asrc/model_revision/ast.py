from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from asrc.model_revision.proposals import TypedRepair


class ExpressionEvaluationError(ValueError):
    """Raised when a validated expression is numerically undefined."""


def parameter_names(expression: dict[str, Any]) -> tuple[str, ...]:
    names: set[str] = set()

    def visit(node: dict[str, Any]) -> None:
        operation = str(node["op"])
        if operation == "parameter":
            names.add(str(node["name"]))
        elif operation in {"negate", "abs", "exp", "log", "sin", "cos", "tanh"}:
            visit(node["argument"])
        elif operation in {"subtract", "divide", "power"}:
            visit(node["left"])
            visit(node["right"])
        elif operation in {"add", "multiply"}:
            for argument in node["arguments"]:
                visit(argument)

    visit(expression)
    return tuple(sorted(names))


def expression_node_count(expression: dict[str, Any]) -> int:
    operation = str(expression["op"])
    if operation in {"variable", "parameter", "constant", "baseline"}:
        return 1
    if operation in {"negate", "abs", "exp", "log", "sin", "cos", "tanh"}:
        return 1 + expression_node_count(expression["argument"])
    if operation in {"subtract", "divide", "power"}:
        return (
            1
            + expression_node_count(expression["left"])
            + expression_node_count(expression["right"])
        )
    return 1 + sum(expression_node_count(item) for item in expression["arguments"])


def _as_array(value: np.ndarray | float, size: int) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        return np.full(size, float(result), dtype=float)
    return result


def evaluate_expression(
    expression: dict[str, Any],
    variables: Mapping[str, np.ndarray],
    parameters: Mapping[str, float] | None = None,
    *,
    baseline: np.ndarray | None = None,
) -> np.ndarray:
    if not variables:
        raise ExpressionEvaluationError("At least one variable array is required.")
    sizes = {np.asarray(value).size for value in variables.values()}
    if len(sizes) != 1:
        raise ExpressionEvaluationError("Variable arrays must have equal length.")
    size = sizes.pop()
    parameter_values = parameters or {}

    def evaluate(node: dict[str, Any]) -> np.ndarray:
        operation = str(node["op"])
        if operation == "variable":
            name = str(node["name"])
            if name not in variables:
                raise ExpressionEvaluationError(f"Variable {name!r} is unavailable.")
            return _as_array(variables[name], size)
        if operation == "parameter":
            name = str(node["name"])
            if name not in parameter_values:
                raise ExpressionEvaluationError(f"Parameter {name!r} is unavailable.")
            return np.full(size, float(parameter_values[name]), dtype=float)
        if operation == "constant":
            return np.full(size, float(node["value"]), dtype=float)
        if operation == "baseline":
            if baseline is None:
                raise ExpressionEvaluationError(
                    "The baseline terminal is unavailable in this context."
                )
            baseline_values = _as_array(baseline, size)
            if baseline_values.shape != (size,):
                raise ExpressionEvaluationError(
                    "The baseline terminal must match the variable arrays."
                )
            return baseline_values
        if operation == "negate":
            return -evaluate(node["argument"])
        if operation == "abs":
            return np.abs(evaluate(node["argument"]))
        if operation == "sin":
            return np.sin(evaluate(node["argument"]))
        if operation == "cos":
            return np.cos(evaluate(node["argument"]))
        if operation == "tanh":
            return np.tanh(evaluate(node["argument"]))
        if operation == "exp":
            argument = evaluate(node["argument"])
            if np.any(np.abs(argument) > 60.0):
                raise ExpressionEvaluationError("exp argument exceeded the stability limit.")
            return np.exp(argument)
        if operation == "log":
            argument = evaluate(node["argument"])
            if np.any(argument <= 0.0):
                raise ExpressionEvaluationError("log argument must be positive.")
            return np.log(argument)
        if operation == "subtract":
            return evaluate(node["left"]) - evaluate(node["right"])
        if operation == "divide":
            denominator = evaluate(node["right"])
            if np.any(np.abs(denominator) < 1.0e-10):
                raise ExpressionEvaluationError("Division denominator approached zero.")
            return evaluate(node["left"]) / denominator
        if operation == "power":
            left = evaluate(node["left"])
            right = evaluate(node["right"])
            with np.errstate(all="ignore"):
                result = np.power(left, right)
            if np.iscomplexobj(result) or not np.all(np.isfinite(result)):
                raise ExpressionEvaluationError("power produced a non-real or non-finite value.")
            return np.asarray(result, dtype=float)
        if operation == "add":
            result = np.zeros(size, dtype=float)
            for argument in node["arguments"]:
                result = result + evaluate(argument)
            return result
        if operation == "multiply":
            result = np.ones(size, dtype=float)
            for argument in node["arguments"]:
                result = result * evaluate(argument)
            return result
        raise ExpressionEvaluationError(f"Unsupported operation {operation!r}.")

    output = evaluate(expression)
    if output.shape != (size,) or not np.all(np.isfinite(output)):
        raise ExpressionEvaluationError("Expression produced invalid output.")
    return output


def apply_repair(
    repair: TypedRepair,
    baseline: np.ndarray,
    variables: Mapping[str, np.ndarray],
    parameters: Mapping[str, float] | None = None,
) -> np.ndarray:
    patch = evaluate_expression(
        repair.expression,
        variables,
        parameters,
        baseline=np.asarray(baseline, dtype=float),
    )
    baseline_values = np.asarray(baseline, dtype=float)
    if repair.edit_type in {
        "add_term",
        "add_state_dependence",
        "add_bounded_transition",
    }:
        result = baseline_values + patch
    elif repair.edit_type == "multiply_term":
        result = baseline_values * patch
    elif repair.edit_type == "replace_subtree":
        result = patch
    else:
        raise ExpressionEvaluationError(
            f"Unsupported repair edit type {repair.edit_type!r}."
        )
    if not np.all(np.isfinite(result)):
        raise ExpressionEvaluationError("Corrected model produced non-finite output.")
    return result


def expression_to_text(expression: dict[str, Any]) -> str:
    operation = str(expression["op"])
    if operation == "variable":
        return str(expression["name"])
    if operation == "parameter":
        return str(expression["name"])
    if operation == "constant":
        return f"{float(expression['value']):g}"
    if operation == "baseline":
        return "M_base(x)"
    if operation in {"abs", "exp", "log", "sin", "cos", "tanh"}:
        return f"{operation}({expression_to_text(expression['argument'])})"
    if operation == "negate":
        return f"-({expression_to_text(expression['argument'])})"
    if operation in {"subtract", "divide", "power"}:
        symbol = {"subtract": "-", "divide": "/", "power": "^"}[operation]
        return (
            f"({expression_to_text(expression['left'])} {symbol} "
            f"{expression_to_text(expression['right'])})"
        )
    symbol = " + " if operation == "add" else " * "
    return "(" + symbol.join(expression_to_text(item) for item in expression["arguments"]) + ")"


def repair_to_text(repair: TypedRepair, baseline_text: str = "M_base(x)") -> str:
    patch = expression_to_text(repair.expression)
    if repair.edit_type in {
        "add_term",
        "add_state_dependence",
        "add_bounded_transition",
    }:
        return f"{baseline_text} + {patch}"
    if repair.edit_type == "multiply_term":
        return f"{baseline_text} * {patch}"
    return patch

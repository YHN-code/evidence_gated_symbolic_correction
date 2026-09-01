from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from asrc.model_revision.ast import ExpressionEvaluationError
from asrc.model_revision.benchmarks import Gate1Task
from asrc.model_revision.evaluation import predict_repair
from asrc.model_revision.patch_algebra import extract_structural_atom
from asrc.model_revision.proposals import (
    TypedRepair,
    canonical_expression_key,
)


@dataclass(frozen=True)
class ExpandedDomainAuditConfig:
    sample_count: int = 2048
    expansion_factor: float = 1.5
    minimum_truth_finite_fraction: float = 0.75
    accuracy_nmse_threshold: float = 0.01

    def __post_init__(self) -> None:
        if self.sample_count < 64:
            raise ValueError("Expanded-domain audit requires at least 64 samples.")
        if self.expansion_factor <= 1.0:
            raise ValueError("expansion_factor must exceed one.")
        if not 0.0 < self.minimum_truth_finite_fraction <= 1.0:
            raise ValueError("minimum_truth_finite_fraction must lie in (0, 1].")
        if self.accuracy_nmse_threshold <= 0.0:
            raise ValueError("accuracy_nmse_threshold must be positive.")


def _audit_repair(
    task: Gate1Task,
    expression: Mapping[str, Any],
    *,
    proposal_id: str,
) -> TypedRepair:
    return TypedRepair(
        proposal_id=proposal_id,
        source="replay",
        edit_type="add_term",
        target=task.definition.target,
        expression=dict(expression),
        rationale="Post-selection structural audit only.",
        expected_signature="Exact target or exact residual structure.",
        structural_key=canonical_expression_key(dict(expression)),
    )


def target_atom_keys(task: Gate1Task) -> frozenset[str]:
    """Return strict typed-AST keys for exact response and exact residual atoms.

    The comparison is conservative: it is invariant to parameter names,
    commutative argument order, and removable outer amplitudes, but does not
    claim general symbolic algebra equivalence.
    """

    exact = task.oracle_repair.expression
    residual = {
        "op": "subtract",
        "left": exact,
        "right": task.baseline_expression,
    }
    keys = set()
    for index, expression in enumerate((exact, residual), start=1):
        repair = _audit_repair(
            task,
            expression,
            proposal_id=f"audit_target_{index}",
        )
        atom = extract_structural_atom(repair, prefix=f"audit_target_{index}")
        keys.add(canonical_expression_key(atom))
    return frozenset(keys)


def candidate_target_structure_match(
    task: Gate1Task,
    repair: TypedRepair,
) -> bool:
    atom = extract_structural_atom(repair, prefix="audit_candidate")
    return canonical_expression_key(atom) in target_atom_keys(task)


def _numeric_constants(node: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    if node.get("op") == "constant":
        values.append(float(node["value"]))
    for value in node.values():
        if isinstance(value, Mapping):
            values.extend(_numeric_constants(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    values.extend(_numeric_constants(item))
    return values


def _snap_parameter(
    value: float,
    anchors: tuple[float, ...],
    *,
    relative_tolerance: float,
) -> tuple[float, bool]:
    candidates = (*anchors, -1.0, 0.0, 1.0)
    closest = min(candidates, key=lambda item: abs(float(value) - item))
    tolerance = relative_tolerance * max(1.0, abs(float(value)), abs(closest))
    if abs(float(value) - closest) <= tolerance:
        return float(closest), True
    return float(value), False


def fitted_algebraic_equivalence_audit(
    task: Gate1Task,
    repair: TypedRepair,
    parameters: Mapping[str, float],
    *,
    parameter_snap_relative_tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    """Check post-fit symbolic equivalence after conservative coefficient snapping.

    Fitted parameters are snapped only to constants already present in the exact
    response or baseline, plus the algebraic identities -1, 0, and 1.  The
    resulting expression is then compared with the exact response by SymPy.
    This is a post-selection oracle audit and must not guide the search.
    """

    try:
        import sympy
    except (ImportError, ModuleNotFoundError) as exc:
        return {
            "status": "sympy_unavailable",
            "equivalent": False,
            "reason": str(exc),
            "snapped_parameters": {},
        }
    anchors = tuple(
        dict.fromkeys(
            _numeric_constants(task.oracle_repair.expression)
            + _numeric_constants(task.baseline_expression)
        )
    )
    snapped: dict[str, float] = {}
    snap_flags: dict[str, bool] = {}
    for name, value in parameters.items():
        snapped[name], snap_flags[name] = _snap_parameter(
            float(value),
            anchors,
            relative_tolerance=parameter_snap_relative_tolerance,
        )
    symbols = {name: sympy.Symbol(name, real=True) for name in task.variables}

    def convert(node: Mapping[str, Any]) -> Any:
        operation = str(node["op"])
        if operation == "variable":
            return symbols[str(node["name"])]
        if operation == "parameter":
            name = str(node["name"])
            if name not in snapped:
                raise ValueError(f"Missing fitted parameter {name!r}.")
            return sympy.Rational(str(snapped[name]))
        if operation == "constant":
            return sympy.Rational(str(float(node["value"])))
        if operation == "baseline":
            return convert(task.baseline_expression)
        if operation == "negate":
            return -convert(node["argument"])
        if operation == "abs":
            return sympy.Abs(convert(node["argument"]))
        if operation in {"exp", "log", "sin", "cos", "tanh"}:
            return getattr(sympy, operation)(convert(node["argument"]))
        if operation == "subtract":
            return convert(node["left"]) - convert(node["right"])
        if operation == "divide":
            return convert(node["left"]) / convert(node["right"])
        if operation == "power":
            return convert(node["left"]) ** convert(node["right"])
        arguments = [convert(item) for item in node["arguments"]]
        if operation == "add":
            return sympy.Add(*arguments)
        if operation == "multiply":
            return sympy.Mul(*arguments)
        raise ValueError(f"Unsupported operation {operation!r}.")

    try:
        candidate = convert(repair.expression)
        exact = convert(task.oracle_repair.expression)
        difference = sympy.cancel(sympy.together(candidate - exact))
        equivalent = bool(difference == 0)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {
            "status": "symbolic_evaluation_failed",
            "equivalent": False,
            "reason": str(exc),
            "snapped_parameters": {
                name: value for name, value in snapped.items() if snap_flags[name]
            },
        }
    return {
        "status": "available",
        "equivalent": equivalent,
        "reason": "" if equivalent else "Simplified fitted expression differs from target.",
        "snapped_parameters": {
            name: value for name, value in snapped.items() if snap_flags[name]
        },
        "snapped_parameter_count": sum(snap_flags.values()),
        "parameter_count": len(snapped),
    }


def build_support_preserving_expanded_frame(
    task: Gate1Task,
    *,
    config: ExpandedDomainAuditConfig | None = None,
    seed: int,
) -> pd.DataFrame:
    """Sample a larger box without crossing an observed one-sided sign support."""

    cfg = config or ExpandedDomainAuditConfig()
    rng = np.random.default_rng(int(seed))
    values: dict[str, np.ndarray] = {}
    for name in task.variables:
        lower, upper = task.definition.observed_ranges[name]
        span = float(upper - lower)
        margin = 0.5 * (cfg.expansion_factor - 1.0) * span
        expanded_lower = float(lower - margin)
        expanded_upper = float(upper + margin)
        if lower > 0.0:
            expanded_lower = max(float(lower) * 0.5, np.finfo(float).tiny)
        elif upper < 0.0:
            expanded_upper = min(float(upper) * 0.5, -np.finfo(float).tiny)
        values[name] = rng.uniform(
            expanded_lower,
            expanded_upper,
            cfg.sample_count,
        )
    return pd.DataFrame(values)


def expanded_domain_recovery_audit(
    task: Gate1Task,
    repair: TypedRepair,
    parameters: Mapping[str, float],
    *,
    config: ExpandedDomainAuditConfig | None = None,
    seed: int,
) -> dict[str, Any]:
    """Evaluate post-selection recovery outside the observed coordinate box.

    Oracle responses are used only inside this audit.  They must never enter
    candidate generation, parameter fitting, or model selection.
    """

    cfg = config or ExpandedDomainAuditConfig()
    frame = build_support_preserving_expanded_frame(task, config=cfg, seed=seed)
    try:
        truth = predict_repair(
            task,
            task.oracle_repair,
            frame,
            task.definition.oracle_parameters,
        )
    except ExpressionEvaluationError as exc:
        return {
            "status": "oracle_evaluation_failed",
            "reason": str(exc),
            "sample_count": len(frame),
            "truth_finite_fraction": 0.0,
            "normalized_rmse": float("inf"),
            "acc_threshold": False,
        }
    truth = np.asarray(truth, dtype=float)
    truth_finite = np.isfinite(truth)
    finite_fraction = float(np.mean(truth_finite))
    if finite_fraction < cfg.minimum_truth_finite_fraction:
        return {
            "status": "insufficient_finite_oracle_support",
            "reason": "Expanded support is not valid for the exact response.",
            "sample_count": len(frame),
            "truth_finite_fraction": finite_fraction,
            "normalized_rmse": float("inf"),
            "acc_threshold": False,
        }
    valid_frame = frame.loc[truth_finite].reset_index(drop=True)
    truth = truth[truth_finite]
    try:
        prediction = predict_repair(task, repair, valid_frame, parameters)
    except ExpressionEvaluationError as exc:
        return {
            "status": "candidate_evaluation_failed",
            "reason": str(exc),
            "sample_count": len(valid_frame),
            "truth_finite_fraction": finite_fraction,
            "normalized_rmse": float("inf"),
            "acc_threshold": False,
        }
    prediction = np.asarray(prediction, dtype=float)
    if not np.all(np.isfinite(prediction)):
        return {
            "status": "candidate_nonfinite",
            "reason": "Candidate response is non-finite on expanded support.",
            "sample_count": len(valid_frame),
            "truth_finite_fraction": finite_fraction,
            "normalized_rmse": float("inf"),
            "acc_threshold": False,
        }
    variance = max(float(np.var(truth)), 1.0e-20)
    nmse = float(np.mean((truth - prediction) ** 2) / variance)
    return {
        "status": "available",
        "reason": "",
        "sample_count": len(valid_frame),
        "truth_finite_fraction": finite_fraction,
        "normalized_rmse": nmse,
        "acc_threshold": bool(nmse < cfg.accuracy_nmse_threshold),
        "accuracy_nmse_threshold": cfg.accuracy_nmse_threshold,
    }

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from asrc.model_revision.ast import expression_node_count
from asrc.model_revision.proposals import (
    RepairContract,
    TypedRepair,
    validate_typed_repair,
)


_AFFINE_INPUT_OPERATORS = frozenset({"abs", "sin", "cos", "tanh"})


def _expression_depth(node: dict[str, Any]) -> int:
    children = [
        child
        for value in node.values()
        for child in (
            value
            if isinstance(value, list)
            else [value]
            if isinstance(value, dict)
            else []
        )
    ]
    return 1 + max((_expression_depth(child) for child in children), default=0)


def _parameter(name: str) -> dict[str, Any]:
    return {"op": "parameter", "name": name}


def _constant(value: float) -> dict[str, Any]:
    return {"op": "constant", "value": float(value)}


def _multiply(*arguments: dict[str, Any]) -> dict[str, Any]:
    return {"op": "multiply", "arguments": list(arguments)}


def _add(*arguments: dict[str, Any]) -> dict[str, Any]:
    return {"op": "add", "arguments": list(arguments)}


def _contains_variable(node: dict[str, Any]) -> bool:
    if node["op"] == "variable":
        return True
    return any(
        _contains_variable(child)
        for value in node.values()
        for child in (
            value
            if isinstance(value, list)
            else [value]
            if isinstance(value, dict)
            else []
        )
    )


def _strip_outer_parameter_scale(node: dict[str, Any]) -> dict[str, Any]:
    """Remove a redundant free amplitude while retaining the proposed atom."""

    if node["op"] != "multiply":
        return node
    retained = [
        item for item in node["arguments"] if item.get("op") != "parameter"
    ]
    if not retained:
        return _constant(1.0)
    if len(retained) == 1:
        return retained[0]
    return _multiply(*retained)


def _strip_affine_parameters(node: dict[str, Any]) -> dict[str, Any]:
    """Recover the structural input of an already parameterized unary atom."""

    operation = node["op"]
    if operation == "multiply":
        retained = [
            _strip_affine_parameters(item)
            for item in node["arguments"]
            if _contains_variable(item)
        ]
        if not retained:
            return _constant(1.0)
        if len(retained) == 1:
            return retained[0]
        return _multiply(*retained)
    if operation == "add":
        retained = [
            _strip_affine_parameters(item)
            for item in node["arguments"]
            if _contains_variable(item)
        ]
        if not retained:
            return _constant(0.0)
        if len(retained) == 1:
            return retained[0]
        return _add(*retained)
    return node


def _rename_parameters(node: dict[str, Any], prefix: str) -> dict[str, Any]:
    names: dict[str, str] = {}

    def visit(value: dict[str, Any]) -> dict[str, Any]:
        operation = value["op"]
        if operation == "parameter":
            original = str(value["name"])
            names.setdefault(original, f"{prefix}_shape_{len(names)}")
            return {"op": "parameter", "name": names[original]}
        if operation in {"variable", "constant", "baseline"}:
            return dict(value)
        if operation in {"negate", "abs", "exp", "log", "sin", "cos", "tanh"}:
            return {"op": operation, "argument": visit(value["argument"])}
        if operation in {"subtract", "divide", "power"}:
            return {
                "op": operation,
                "left": visit(value["left"]),
                "right": visit(value["right"]),
            }
        return {
            "op": operation,
            "arguments": [visit(item) for item in value["arguments"]],
        }

    return visit(node)


def _lift_nonstructural_constants(
    node: dict[str, Any],
    prefix: str,
) -> dict[str, Any]:
    """Replace free numeric literals with fit parameters.

    Exponents and the algebraic identities -1, 0, and 1 remain structural.
    This prevents any proposer from bypassing shared constant optimization by
    embedding a task-scale literal directly in its expression tree.
    """

    parameter_index = 0

    def visit(value: dict[str, Any], *, exponent: bool = False) -> dict[str, Any]:
        nonlocal parameter_index
        operation = value["op"]
        if operation == "constant":
            number = float(value["value"])
            if exponent or number in {-1.0, 0.0, 1.0}:
                return {"op": "constant", "value": number}
            name = f"{prefix}_literal_{parameter_index}"
            parameter_index += 1
            return {"op": "parameter", "name": name}
        if operation in {"variable", "parameter", "baseline"}:
            return dict(value)
        if operation in {"negate", "abs", "exp", "log", "sin", "cos", "tanh"}:
            return {
                "op": operation,
                "argument": visit(value["argument"]),
            }
        if operation in {"subtract", "divide"}:
            return {
                "op": operation,
                "left": visit(value["left"]),
                "right": visit(value["right"]),
            }
        if operation == "power":
            return {
                "op": operation,
                "left": visit(value["left"]),
                "right": visit(value["right"], exponent=True),
            }
        return {
            "op": operation,
            "arguments": [visit(item) for item in value["arguments"]],
        }

    return visit(node)


def _normalize_unary_inputs(node: dict[str, Any], prefix: str) -> dict[str, Any]:
    operation = node["op"]
    if operation in _AFFINE_INPUT_OPERATORS:
        structural_argument = _strip_affine_parameters(node["argument"])
        normalized_argument = _normalize_unary_inputs(
            structural_argument,
            f"{prefix}_inner",
        )
        return {
            "op": operation,
            "argument": _add(
                _multiply(
                    _parameter(f"{prefix}_input_scale"),
                    normalized_argument,
                ),
                _parameter(f"{prefix}_input_shift"),
            ),
        }
    if operation == "exp":
        structural_argument = _strip_affine_parameters(node["argument"])
        return {
            "op": operation,
            "argument": _multiply(
                _parameter(f"{prefix}_input_scale"),
                _normalize_unary_inputs(
                    structural_argument,
                    f"{prefix}_inner",
                ),
            ),
        }
    if operation in {"negate", "log"}:
        return {
            "op": operation,
            "argument": _normalize_unary_inputs(
                node["argument"],
                f"{prefix}_inner",
            ),
        }
    if operation in {"subtract", "divide", "power"}:
        return {
            "op": operation,
            "left": _normalize_unary_inputs(node["left"], f"{prefix}_left"),
            "right": _normalize_unary_inputs(node["right"], f"{prefix}_right"),
        }
    if operation in {"add", "multiply"}:
        return {
            "op": operation,
            "arguments": [
                _normalize_unary_inputs(item, f"{prefix}_{index}")
                for index, item in enumerate(node["arguments"])
            ],
        }
    return dict(node)


def extract_structural_atom(repair: TypedRepair, *, prefix: str) -> dict[str, Any]:
    """Convert patch syntax into an amplitude-free structural atom."""

    expression = repair.expression
    if repair.edit_type == "multiply_term" and expression["op"] == "add":
        variable_terms = [
            item for item in expression["arguments"] if _contains_variable(item)
        ]
        if variable_terms:
            expression = (
                variable_terms[0]
                if len(variable_terms) == 1
                else _add(*variable_terms)
            )
    atom = _strip_outer_parameter_scale(expression)
    atom = _rename_parameters(atom, prefix)
    atom = _lift_nonstructural_constants(atom, prefix)
    atom = _strip_outer_parameter_scale(atom)
    return _normalize_unary_inputs(atom, prefix)


def compile_baseline_aware_patch(
    repair: TypedRepair,
    contract: RepairContract,
    *,
    proposal_id: str | None = None,
) -> TypedRepair:
    """Compile a proposed atom into a parameter-complete corrected model.

    The common algebra is ``c_b M_base(x) + c_0 + c_1 phi(x)``. Bounded,
    periodic, and absolute-value atoms additionally receive an affine input.
    This lets one generic model preserve, recalibrate, or cancel a baseline
    without exposing response targets or oracle expressions to the proposer.
    """

    identifier = str(proposal_id or f"compiled_{repair.proposal_id}")
    atom = extract_structural_atom(repair, prefix="patch")
    expression = _add(
        _multiply(_parameter("baseline_scale"), {"op": "baseline"}),
        _parameter("intercept"),
        _multiply(_parameter("atom_amplitude"), atom),
    )
    # The source repair has already passed the public contract. Compilation
    # deterministically adds calibration structure, so grant exactly the depth
    # and node capacity required by that generated wrapper.
    compiled_contract = replace(
        contract,
        allowed_operators=frozenset((*contract.allowed_operators, "baseline")),
        maximum_depth=max(contract.maximum_depth, _expression_depth(expression)),
        maximum_nodes=max(contract.maximum_nodes, expression_node_count(expression)),
    )
    return validate_typed_repair(
        {
            "proposal_id": identifier,
            "source": repair.source,
            "edit_type": "replace_subtree",
            "target": repair.target,
            "expression": expression,
            "rationale": (
                f"Baseline-aware typed compilation of {repair.proposal_id}: "
                f"{repair.rationale}"
            ),
            "expected_signature": repair.expected_signature,
        },
        compiled_contract,
    )


def compile_patch_batch(
    repairs: Iterable[TypedRepair],
    contract: RepairContract,
    *,
    id_prefix: str,
) -> tuple[TypedRepair, ...]:
    compiled: list[TypedRepair] = []
    seen: set[str] = set()
    for index, repair in enumerate(repairs, start=1):
        candidate = compile_baseline_aware_patch(
            repair,
            contract,
            proposal_id=f"{id_prefix}_{index:03d}",
        )
        if candidate.structural_key in seen:
            continue
        seen.add(candidate.structural_key)
        compiled.append(candidate)
    return tuple(compiled)

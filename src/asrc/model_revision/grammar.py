from __future__ import annotations

from typing import Any

from asrc.model_revision.proposals import (
    ProposalBatch,
    RepairContract,
    RepairRequest,
    StaticRepairProposer,
)


def _constant(value: float) -> dict[str, Any]:
    return {"op": "constant", "value": float(value)}


def _variable(name: str) -> dict[str, Any]:
    return {"op": "variable", "name": name}


def _parameter(name: str) -> dict[str, Any]:
    return {"op": "parameter", "name": name}


def _multiply(*arguments: dict[str, Any]) -> dict[str, Any]:
    return {"op": "multiply", "arguments": list(arguments)}


def _scaled(feature: dict[str, Any], suffix: str) -> dict[str, Any]:
    return _multiply(_parameter(f"coefficient_{suffix}"), feature)


def _payload(
    index: int,
    target: str,
    expression: dict[str, Any],
    *,
    edit_type: str = "add_term",
    signature: str,
) -> dict[str, Any]:
    return {
        "proposal_id": f"grammar_{index:03d}",
        "source": "grammar",
        "edit_type": edit_type,
        "target": target,
        "expression": expression,
        "rationale": "Frozen low-order generic repair grammar.",
        "expected_signature": signature,
    }


def grammar_repair_payloads(contract: RepairContract) -> list[dict[str, Any]]:
    """Build a small generic library without task-specific routing."""

    target = sorted(contract.allowed_targets)[0]
    variables = sorted(contract.allowed_variables)
    payloads: list[dict[str, Any]] = []

    def add(
        expression: dict[str, Any],
        signature: str,
        edit_type: str = "add_term",
    ) -> None:
        payloads.append(
            _payload(
                len(payloads) + 1,
                target,
                expression,
                edit_type=edit_type,
                signature=signature,
            )
        )

    for name in variables:
        variable = _variable(name)
        add(_scaled(variable, f"linear_{name}"), f"linear trend in {name}")
        add(
            _scaled(
                {"op": "power", "left": variable, "right": _constant(2.0)},
                f"quadratic_{name}",
            ),
            f"even curvature in {name}",
        )
        add(
            _scaled(
                {"op": "power", "left": variable, "right": _constant(3.0)},
                f"cubic_{name}",
            ),
            f"odd nonlinear trend in {name}",
        )
        add(
            _scaled({"op": "sin", "argument": variable}, f"sin_{name}"),
            f"bounded odd trend in {name}",
        )
        add(
            _scaled({"op": "cos", "argument": variable}, f"cos_{name}"),
            f"bounded even trend in {name}",
        )
        add(
            _scaled(
                {
                    "op": "exp",
                    "argument": _multiply(_parameter(f"rate_{name}"), variable),
                },
                f"exp_{name}",
            ),
            f"monotone exponential trend in {name}",
        )
        add(
            _scaled(
                {
                    "op": "tanh",
                    "argument": _multiply(_parameter(f"slope_{name}"), variable),
                },
                f"tanh_{name}",
            ),
            f"centered bounded transition in {name}",
            "add_bounded_transition",
        )

    for left_index, left_name in enumerate(variables):
        for right_name in variables[left_index + 1 :]:
            left = _variable(left_name)
            right = _variable(right_name)
            add(
                _scaled(_multiply(left, right), f"interaction_{left_name}_{right_name}"),
                f"bilinear interaction between {left_name} and {right_name}",
                "add_state_dependence",
            )
            add(
                _scaled(
                    _multiply(
                        {"op": "power", "left": left, "right": _constant(2.0)},
                        right,
                    ),
                    f"quadratic_interaction_{left_name}_{right_name}",
                ),
                f"low-order interaction between {left_name} and {right_name}",
                "add_state_dependence",
            )
            add(
                _scaled(
                    _multiply(
                        left,
                        {"op": "power", "left": right, "right": _constant(2.0)},
                    ),
                    f"interaction_quadratic_{left_name}_{right_name}",
                ),
                f"low-order interaction between {left_name} and {right_name}",
                "add_state_dependence",
            )

    for name in variables:
        add(
            {
                "op": "add",
                "arguments": [
                    _constant(1.0),
                    _scaled(_variable(name), f"factor_{name}"),
                ],
            },
            f"linear multiplicative factor in {name}",
            "multiply_term",
        )
    for left_index, left_name in enumerate(variables):
        for right_name in variables[left_index + 1 :]:
            add(
                {
                    "op": "add",
                    "arguments": [
                        _constant(1.0),
                        _scaled(
                            _multiply(_variable(left_name), _variable(right_name)),
                            f"factor_{left_name}_{right_name}",
                        ),
                    ],
                },
                f"bilinear multiplicative factor in {left_name} and {right_name}",
                "multiply_term",
            )
    return payloads


class GrammarRepairProposer:
    source = "grammar"

    def propose(
        self,
        request: RepairRequest,
        contract: RepairContract,
    ) -> ProposalBatch:
        return StaticRepairProposer(
            self.source,
            grammar_repair_payloads(contract),
        ).propose(request, contract)

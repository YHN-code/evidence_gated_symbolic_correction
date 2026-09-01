from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


DEFAULT_OPERATORS = frozenset(
    {
        "variable",
        "parameter",
        "constant",
        "add",
        "subtract",
        "multiply",
        "divide",
        "power",
        "negate",
        "abs",
        "exp",
        "log",
        "sin",
        "cos",
        "tanh",
    }
)
DEFAULT_EDIT_TYPES = frozenset(
    {
        "add_term",
        "multiply_term",
        "replace_subtree",
        "add_state_dependence",
        "add_bounded_transition",
    }
)
DEFAULT_SOURCES = frozenset({"grammar", "pysr", "llm", "replay"})

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORBIDDEN_CONTEXT_TOKENS = (
    "hidden_",
    "locked_",
    "reference_",
    "truth_",
)
_UNARY_OPERATORS = frozenset({"negate", "abs", "exp", "log", "sin", "cos", "tanh"})
_BINARY_OPERATORS = frozenset({"subtract", "divide", "power"})
_COMMUTATIVE_OPERATORS = frozenset({"add", "multiply"})
_CONTROLLED_TERMINAL_OPERATORS = frozenset({"baseline"})


class RepairValidationError(ValueError):
    """Raised when a proposed symbolic repair violates the public contract."""


@dataclass(frozen=True)
class RepairContract:
    allowed_variables: frozenset[str]
    allowed_targets: frozenset[str]
    allowed_operators: frozenset[str] = field(default_factory=lambda: DEFAULT_OPERATORS)
    allowed_edit_types: frozenset[str] = field(default_factory=lambda: DEFAULT_EDIT_TYPES)
    allowed_sources: frozenset[str] = field(default_factory=lambda: DEFAULT_SOURCES)
    maximum_depth: int = 8
    maximum_nodes: int = 48

    def __post_init__(self) -> None:
        if not self.allowed_variables:
            raise ValueError("allowed_variables must not be empty.")
        if not self.allowed_targets:
            raise ValueError("allowed_targets must not be empty.")
        if self.maximum_depth < 1 or self.maximum_nodes < 1:
            raise ValueError("Expression limits must be positive.")


@dataclass(frozen=True)
class RepairRequest:
    baseline_expression: dict[str, Any]
    residual_evidence: dict[str, Any]
    constraints: tuple[str, ...] = ()
    failed_structural_keys: frozenset[str] = field(default_factory=frozenset)
    maximum_candidates: int = 16

    def __post_init__(self) -> None:
        if self.maximum_candidates < 1:
            raise ValueError("maximum_candidates must be positive.")
        forbidden = _find_forbidden_context_key(
            {
                "baseline_expression": self.baseline_expression,
                "residual_evidence": self.residual_evidence,
            }
        )
        if forbidden is not None:
            raise RepairValidationError(
                f"Repair request contains forbidden evidence field {forbidden!r}."
            )


@dataclass(frozen=True)
class TypedRepair:
    proposal_id: str
    source: str
    edit_type: str
    target: str
    expression: dict[str, Any]
    rationale: str
    expected_signature: str
    structural_key: str


@dataclass(frozen=True)
class RejectedRepair:
    proposal_id: str
    source: str
    reason: str


@dataclass(frozen=True)
class ProposalBatch:
    accepted: tuple[TypedRepair, ...]
    rejected: tuple[RejectedRepair, ...]


class RepairProposer(Protocol):
    source: str

    def propose(
        self,
        request: RepairRequest,
        contract: RepairContract,
    ) -> ProposalBatch: ...


def _find_forbidden_context_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in _FORBIDDEN_CONTEXT_TOKENS):
                return str(key)
            nested = _find_forbidden_context_key(child)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for child in value:
            nested = _find_forbidden_context_key(child)
            if nested is not None:
                return nested
    return None


def _require_keys(node: dict[str, Any], required: set[str]) -> None:
    actual = set(node)
    if actual != required:
        raise RepairValidationError(
            f"AST node {node.get('op')!r} requires fields {sorted(required)}; "
            f"received {sorted(actual)}."
        )


def _validate_identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise RepairValidationError(f"{field_name} must be a valid identifier.")
    return value


def _validate_node(
    node: Any,
    contract: RepairContract,
    *,
    depth: int,
    node_counter: list[int],
) -> dict[str, Any]:
    if not isinstance(node, dict):
        raise RepairValidationError("Every expression node must be a JSON object.")
    node_counter[0] += 1
    if node_counter[0] > contract.maximum_nodes:
        raise RepairValidationError(
            f"Expression exceeds maximum_nodes={contract.maximum_nodes}."
        )
    if depth > contract.maximum_depth:
        raise RepairValidationError(
            f"Expression exceeds maximum_depth={contract.maximum_depth}."
        )

    op = node.get("op")
    if not isinstance(op, str) or op not in contract.allowed_operators:
        raise RepairValidationError(f"Unsupported expression operator {op!r}.")

    if op == "variable":
        _require_keys(node, {"op", "name"})
        name = _validate_identifier(node["name"], "variable name")
        if name not in contract.allowed_variables:
            raise RepairValidationError(f"Variable {name!r} is not allowed.")
        return {"op": op, "name": name}

    if op == "parameter":
        _require_keys(node, {"op", "name"})
        return {
            "op": op,
            "name": _validate_identifier(node["name"], "parameter name"),
        }

    if op == "constant":
        _require_keys(node, {"op", "value"})
        value = node["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RepairValidationError("constant value must be numeric.")
        if not math.isfinite(float(value)):
            raise RepairValidationError("constant value must be finite.")
        return {"op": op, "value": float(value)}

    if op in _CONTROLLED_TERMINAL_OPERATORS:
        _require_keys(node, {"op"})
        return {"op": op}

    if op in _UNARY_OPERATORS:
        _require_keys(node, {"op", "argument"})
        return {
            "op": op,
            "argument": _validate_node(
                node["argument"],
                contract,
                depth=depth + 1,
                node_counter=node_counter,
            ),
        }

    if op in _BINARY_OPERATORS:
        _require_keys(node, {"op", "left", "right"})
        return {
            "op": op,
            "left": _validate_node(
                node["left"],
                contract,
                depth=depth + 1,
                node_counter=node_counter,
            ),
            "right": _validate_node(
                node["right"],
                contract,
                depth=depth + 1,
                node_counter=node_counter,
            ),
        }

    if op in _COMMUTATIVE_OPERATORS:
        _require_keys(node, {"op", "arguments"})
        arguments = node["arguments"]
        if not isinstance(arguments, list) or len(arguments) < 2:
            raise RepairValidationError(f"{op} requires at least two arguments.")
        return {
            "op": op,
            "arguments": [
                _validate_node(
                    argument,
                    contract,
                    depth=depth + 1,
                    node_counter=node_counter,
                )
                for argument in arguments
            ],
        }

    raise RepairValidationError(f"Operator {op!r} has no validation contract.")


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _contains_parameter(node: dict[str, Any]) -> bool:
    if node["op"] == "parameter":
        return True
    return any(
        _contains_parameter(child)
        for value in node.values()
        for child in (
            value
            if isinstance(value, list)
            else [value]
            if isinstance(value, dict)
            else []
        )
    )


def _normalize_commutative(node: dict[str, Any]) -> dict[str, Any]:
    op = node["op"]
    if op in {"variable", "parameter", "constant", "baseline"}:
        return dict(node)
    if op in _UNARY_OPERATORS:
        return {
            "op": op,
            "argument": _normalize_commutative(node["argument"]),
        }
    if op in _BINARY_OPERATORS:
        return {
            "op": op,
            "left": _normalize_commutative(node["left"]),
            "right": _normalize_commutative(node["right"]),
        }
    arguments: list[dict[str, Any]] = []
    for item in node["arguments"]:
        normalized_item = _normalize_commutative(item)
        if normalized_item["op"] == op:
            arguments.extend(normalized_item["arguments"])
        else:
            arguments.append(normalized_item)
    if op == "multiply":
        grouped: dict[str, list[dict[str, Any]]] = {}
        for argument in arguments:
            grouped.setdefault(_json_key(argument), []).append(argument)
        arguments = []
        for group in grouped.values():
            factor = group[0]
            if len(group) > 1 and not _contains_parameter(factor):
                arguments.append(
                    {
                        "op": "power",
                        "left": factor,
                        "right": {"op": "constant", "value": float(len(group))},
                    }
                )
            else:
                arguments.extend(group)
    arguments.sort(key=_json_key)
    return {"op": op, "arguments": arguments}


def _alpha_rename_parameters(
    node: dict[str, Any],
    names: dict[str, str],
) -> dict[str, Any]:
    op = node["op"]
    if op == "parameter":
        original = node["name"]
        names.setdefault(original, f"p{len(names)}")
        return {"op": op, "name": names[original]}
    if op in {"variable", "constant", "baseline"}:
        return dict(node)
    if op in _UNARY_OPERATORS:
        return {
            "op": op,
            "argument": _alpha_rename_parameters(node["argument"], names),
        }
    if op in _BINARY_OPERATORS:
        return {
            "op": op,
            "left": _alpha_rename_parameters(node["left"], names),
            "right": _alpha_rename_parameters(node["right"], names),
        }
    return {
        "op": op,
        "arguments": [
            _alpha_rename_parameters(item, names) for item in node["arguments"]
        ],
    }


def _structural_key(edit_type: str, target: str, expression: dict[str, Any]) -> str:
    normalized = _normalize_commutative(expression)
    canonical = _alpha_rename_parameters(normalized, {})
    return _json_key(
        {
            "edit_type": edit_type,
            "target": target,
            "expression": canonical,
        }
    )


def canonical_expression_key(expression: dict[str, Any]) -> str:
    """Return a syntax key independent of coefficient names and edit metadata."""

    normalized = _normalize_commutative(expression)
    canonical = _alpha_rename_parameters(normalized, {})
    return _json_key(canonical)


def validate_expression_ast(
    expression: Any,
    contract: RepairContract,
) -> dict[str, Any]:
    """Validate and normalize a standalone expression AST."""

    return _validate_node(
        expression,
        contract,
        depth=1,
        node_counter=[0],
    )


def validate_typed_repair(
    payload: Any,
    contract: RepairContract,
) -> TypedRepair:
    if not isinstance(payload, dict):
        raise RepairValidationError("Repair proposal must be a JSON object.")
    required = {
        "proposal_id",
        "source",
        "edit_type",
        "target",
        "expression",
        "rationale",
        "expected_signature",
    }
    if set(payload) != required:
        raise RepairValidationError(
            f"Repair proposal requires exactly {sorted(required)}."
        )
    forbidden = _find_forbidden_context_key(payload)
    if forbidden is not None:
        raise RepairValidationError(
            f"Repair proposal contains forbidden evidence field {forbidden!r}."
        )

    proposal_id = _validate_identifier(payload["proposal_id"], "proposal_id")
    source = str(payload["source"])
    if source not in contract.allowed_sources:
        raise RepairValidationError(f"Unsupported proposal source {source!r}.")
    edit_type = str(payload["edit_type"])
    if edit_type not in contract.allowed_edit_types:
        raise RepairValidationError(f"Unsupported edit_type {edit_type!r}.")
    target = str(payload["target"])
    if target not in contract.allowed_targets:
        raise RepairValidationError(f"Unsupported repair target {target!r}.")
    rationale = str(payload["rationale"]).strip()
    expected_signature = str(payload["expected_signature"]).strip()
    if not rationale or not expected_signature:
        raise RepairValidationError(
            "rationale and expected_signature must be non-empty."
        )

    expression = validate_expression_ast(payload["expression"], contract)
    return TypedRepair(
        proposal_id=proposal_id,
        source=source,
        edit_type=edit_type,
        target=target,
        expression=expression,
        rationale=rationale,
        expected_signature=expected_signature,
        structural_key=_structural_key(edit_type, target, expression),
    )


class StaticRepairProposer:
    """Deterministic proposer used for grammar output and archived replay."""

    def __init__(self, source: str, payloads: Iterable[dict[str, Any]]) -> None:
        self.source = source
        self._payloads = tuple(payloads)

    def propose(
        self,
        request: RepairRequest,
        contract: RepairContract,
    ) -> ProposalBatch:
        if self.source not in contract.allowed_sources:
            raise RepairValidationError(
                f"Unsupported proposer source {self.source!r}."
            )
        accepted: list[TypedRepair] = []
        rejected: list[RejectedRepair] = []
        seen = set(request.failed_structural_keys)
        for index, raw_payload in enumerate(self._payloads):
            payload = dict(raw_payload)
            payload.setdefault("source", self.source)
            proposal_id = str(payload.get("proposal_id", f"proposal_{index:03d}"))
            try:
                proposal = validate_typed_repair(payload, contract)
            except RepairValidationError as exc:
                rejected.append(
                    RejectedRepair(proposal_id, self.source, str(exc))
                )
                continue
            if proposal.source != self.source:
                rejected.append(
                    RejectedRepair(
                        proposal.proposal_id,
                        self.source,
                        "Proposal source does not match the executing proposer.",
                    )
                )
                continue
            if proposal.structural_key in seen:
                rejected.append(
                    RejectedRepair(
                        proposal.proposal_id,
                        proposal.source,
                        "Duplicate or previously failed structural repair.",
                    )
                )
                continue
            seen.add(proposal.structural_key)
            if len(accepted) >= request.maximum_candidates:
                rejected.append(
                    RejectedRepair(
                        proposal.proposal_id,
                        proposal.source,
                        "Candidate evaluation budget is full.",
                    )
                )
                continue
            accepted.append(proposal)
        return ProposalBatch(tuple(accepted), tuple(rejected))


def merge_proposal_batches(
    batches: Iterable[ProposalBatch],
    *,
    maximum_candidates: int,
) -> ProposalBatch:
    if maximum_candidates < 1:
        raise ValueError("maximum_candidates must be positive.")
    accepted: list[TypedRepair] = []
    rejected: list[RejectedRepair] = []
    seen: set[str] = set()
    for batch in batches:
        rejected.extend(batch.rejected)
        for proposal in batch.accepted:
            if proposal.structural_key in seen:
                rejected.append(
                    RejectedRepair(
                        proposal.proposal_id,
                        proposal.source,
                        "Duplicate structural repair across proposers.",
                    )
                )
                continue
            if len(accepted) >= maximum_candidates:
                rejected.append(
                    RejectedRepair(
                        proposal.proposal_id,
                        proposal.source,
                        "Merged candidate evaluation budget is full.",
                    )
                )
                continue
            seen.add(proposal.structural_key)
            accepted.append(proposal)
    return ProposalBatch(tuple(accepted), tuple(rejected))


def round_robin_merge_proposal_batches(
    batches: Iterable[ProposalBatch],
    *,
    maximum_candidates: int,
) -> ProposalBatch:
    """Merge proposer outputs without allowing the first source to fill the budget."""

    if maximum_candidates < 1:
        raise ValueError("maximum_candidates must be positive.")
    materialized = tuple(batches)
    accepted: list[TypedRepair] = []
    rejected: list[RejectedRepair] = [
        item for batch in materialized for item in batch.rejected
    ]
    seen: set[str] = set()
    maximum_length = max((len(batch.accepted) for batch in materialized), default=0)
    for index in range(maximum_length):
        for batch in materialized:
            if index >= len(batch.accepted):
                continue
            proposal = batch.accepted[index]
            if proposal.structural_key in seen:
                rejected.append(
                    RejectedRepair(
                        proposal.proposal_id,
                        proposal.source,
                        "Duplicate structural repair across proposers.",
                    )
                )
                continue
            if len(accepted) >= maximum_candidates:
                rejected.append(
                    RejectedRepair(
                        proposal.proposal_id,
                        proposal.source,
                        "Merged candidate evaluation budget is full.",
                    )
                )
                continue
            seen.add(proposal.structural_key)
            accepted.append(proposal)
    return ProposalBatch(tuple(accepted), tuple(rejected))

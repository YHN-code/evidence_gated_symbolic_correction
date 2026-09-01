from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from asrc.constitutive.weak_plane import WeakPlaneState


PARAMETER_NAME = re.compile(r"^[a-z][a-z0-9_]{0,23}$")
CANDIDATE_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class OpenCandidateError(ValueError):
    pass


@dataclass(frozen=True)
class ShapeParameter:
    name: str
    role: str
    lower: float
    upper: float
    initial: float


@dataclass(frozen=True)
class OpenCandidateSpec:
    candidate_id: str
    rationale: str
    decay: dict[str, Any]
    parameters: tuple[ShapeParameter, ...]
    structural_key: str
    node_count: int
    complexity: int


@dataclass(frozen=True)
class OpenCohesionEvolution:
    spec: OpenCandidateSpec
    coefficients: tuple[float, ...]

    @property
    def family(self) -> str:
        return f"open::{self.spec.candidate_id}"

    @property
    def complexity(self) -> int:
        return self.spec.complexity

    @property
    def residual_cohesion_mpa(self) -> float:
        return float(self.coefficients[0])

    @property
    def peak_cohesion_mpa(self) -> float:
        return float(self.coefficients[0] + self.coefficients[1])

    @property
    def softening_scale(self) -> float:
        return float(self.coefficients[2])

    @property
    def shape_parameters(self) -> dict[str, float]:
        return {
            parameter.name: float(self.coefficients[index + 3])
            for index, parameter in enumerate(self.spec.parameters)
        }

    def cohesion(self, plastic_shear: float | np.ndarray) -> float | np.ndarray:
        kappa = np.maximum(np.asarray(plastic_shear, dtype=float), 0.0)
        z = kappa / max(self.softening_scale, np.finfo(float).tiny)
        decay = evaluate_decay(self.spec.decay, z, self.shape_parameters)
        value = self.residual_cohesion_mpa + float(self.coefficients[1]) * decay
        return float(value) if np.ndim(value) == 0 else value

    def strength(
        self,
        sigma_n_mpa: float,
        _mean_compression_mpa: float,
        _beta_deg: float,
        state: WeakPlaneState,
        friction_deg: float,
    ) -> float:
        return float(
            self.cohesion(state.accumulated_plastic_shear)
            - sigma_n_mpa * np.tan(np.deg2rad(friction_deg))
        )

    @property
    def formula(self) -> str:
        z_text = f"(kappa/{self.softening_scale:.8g})"
        shape = render_decay(
            self.spec.decay,
            parameter_values=self.shape_parameters,
            variable_text=z_text,
        )
        return (
            f"c(kappa) = {self.residual_cohesion_mpa:.8g} + "
            f"{self.coefficients[1]:.8g} * {shape}"
        )


def _require_exact_keys(
    node: dict[str, Any],
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    if not isinstance(node, dict):
        raise OpenCandidateError("Expression nodes must be JSON objects.")
    optional = optional or set()
    missing = required.difference(node)
    extra = set(node).difference(required | optional)
    if missing:
        raise OpenCandidateError(f"Expression node missing fields: {sorted(missing)}.")
    if extra:
        raise OpenCandidateError(
            f"Expression node contains unsupported fields: {sorted(extra)}."
        )


def _parameter_from_payload(
    payload: Any,
    grammar: dict[str, Any],
    registry: dict[str, ShapeParameter],
) -> dict[str, str]:
    _require_exact_keys(payload, {"name", "role"})
    name = str(payload["name"]).strip()
    role = str(payload["role"]).strip()
    if not PARAMETER_NAME.fullmatch(name):
        raise OpenCandidateError(
            f"Invalid shape parameter name {name!r}; use lower-case identifiers."
        )
    roles = grammar["parameter_roles"]
    if role not in roles:
        raise OpenCandidateError(f"Unknown shape parameter role: {role!r}.")
    bounds = roles[role]
    parameter = ShapeParameter(
        name=name,
        role=role,
        lower=float(bounds["lower"]),
        upper=float(bounds["upper"]),
        initial=float(bounds["initial"]),
    )
    if not (
        np.isfinite(parameter.lower)
        and np.isfinite(parameter.upper)
        and np.isfinite(parameter.initial)
        and 0.0 < parameter.lower < parameter.upper
        and parameter.lower <= parameter.initial <= parameter.upper
    ):
        raise OpenCandidateError(f"Invalid bounds for parameter role {role!r}.")
    previous = registry.get(name)
    if previous is not None and previous.role != role:
        raise OpenCandidateError(
            f"Shape parameter {name!r} is reused with inconsistent roles."
        )
    registry[name] = parameter
    return {"name": name, "role": role}


def _fixed_exponent(value: Any, grammar: dict[str, Any]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenCandidateError(
            "An exponent must be a parameter object or an allowed fixed number."
        )
    result = float(value)
    allowed = [
        float(item) for item in grammar["allowed_fixed_exponents"]
    ]
    if not any(np.isclose(result, item, rtol=0.0, atol=1.0e-12) for item in allowed):
        raise OpenCandidateError(
            f"Fixed exponent {result:g} is outside the pre-registered set {allowed}."
        )
    return result


def _validate_exponent(
    payload: Any,
    grammar: dict[str, Any],
    registry: dict[str, ShapeParameter],
) -> float | dict[str, str]:
    if isinstance(payload, dict):
        parameter = _parameter_from_payload(payload, grammar, registry)
        if parameter["role"] != "positive_exponent":
            raise OpenCandidateError(
                "Power exponents must use role 'positive_exponent'."
            )
        return parameter
    return _fixed_exponent(payload, grammar)


def _validate_progress(
    payload: Any,
    grammar: dict[str, Any],
    registry: dict[str, ShapeParameter],
    depth: int,
) -> tuple[dict[str, Any], int]:
    if depth > int(grammar["maximum_depth"]):
        raise OpenCandidateError("Expression exceeds the maximum grammar depth.")
    if not isinstance(payload, dict):
        raise OpenCandidateError("Progress expressions must be JSON objects.")
    operation = str(payload.get("op", ""))
    if operation not in set(grammar["progress_operators"]):
        raise OpenCandidateError(f"Unsupported progress operator: {operation!r}.")
    if operation == "z":
        _require_exact_keys(payload, {"op"})
        return {"op": "z"}, 1
    if operation == "power":
        _require_exact_keys(payload, {"op", "argument", "exponent"})
        argument, nodes = _validate_progress(
            payload["argument"],
            grammar,
            registry,
            depth + 1,
        )
        exponent = _validate_exponent(
            payload["exponent"],
            grammar,
            registry,
        )
        if isinstance(exponent, float) and np.isclose(exponent, 1.0):
            return argument, nodes
        return {
            "op": "power",
            "argument": argument,
            "exponent": exponent,
        }, nodes + 1 + int(isinstance(exponent, dict))
    if operation == "scale":
        _require_exact_keys(payload, {"op", "argument", "factor"})
        argument, nodes = _validate_progress(
            payload["argument"],
            grammar,
            registry,
            depth + 1,
        )
        factor = _parameter_from_payload(
            payload["factor"],
            grammar,
            registry,
        )
        if factor["role"] != "positive_weight":
            raise OpenCandidateError(
                "Scale factors must use role 'positive_weight'."
            )
        return {
            "op": "scale",
            "argument": argument,
            "factor": factor,
        }, nodes + 2
    if operation in {"add", "multiply"}:
        _require_exact_keys(payload, {"op", "arguments"})
        arguments = payload["arguments"]
        if not isinstance(arguments, list):
            raise OpenCandidateError(f"{operation} arguments must be a list.")
        maximum = int(
            grammar[
                "maximum_add_arguments"
                if operation == "add"
                else "maximum_multiply_arguments"
            ]
        )
        if not 2 <= len(arguments) <= maximum:
            raise OpenCandidateError(
                f"{operation} requires between 2 and {maximum} arguments."
            )
        normalized = []
        node_count = 1
        for argument_payload in arguments:
            argument, nodes = _validate_progress(
                argument_payload,
                grammar,
                registry,
                depth + 1,
            )
            normalized.append(argument)
            node_count += nodes
        normalized.sort(
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
        )
        return {"op": operation, "arguments": normalized}, node_count
    raise OpenCandidateError(f"Unsupported progress operator: {operation!r}.")


def _shape_signature(node: Any) -> Any:
    if isinstance(node, dict):
        if set(node) == {"name", "role"}:
            return {"parameter_role": node["role"]}
        return {
            key: _shape_signature(value)
            for key, value in sorted(node.items())
        }
    if isinstance(node, list):
        return [_shape_signature(value) for value in node]
    return node


def _alpha_normalized(node: Any) -> Any:
    parameter_names: dict[str, str] = {}
    role_counts: dict[str, int] = {}

    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"name", "role"}:
                original = str(value["name"])
                role = str(value["role"])
                if original not in parameter_names:
                    role_counts[role] = role_counts.get(role, 0) + 1
                    parameter_names[original] = f"{role}_{role_counts[role]}"
                return {
                    "name": parameter_names[original],
                    "role": role,
                }
            result = {}
            for key, child in sorted(value.items()):
                if isinstance(child, list):
                    ordered = sorted(
                        child,
                        key=lambda item: json.dumps(
                            _shape_signature(item),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                    result[key] = [visit(item) for item in ordered]
                else:
                    result[key] = visit(child)
            return result
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    return visit(node)


def validate_open_candidate(
    payload: dict[str, Any],
    config: dict[str, Any],
) -> OpenCandidateSpec:
    _require_exact_keys(payload, {"candidate_id", "rationale", "decay"})
    candidate_id = str(payload["candidate_id"]).strip()
    if not CANDIDATE_ID.fullmatch(candidate_id):
        raise OpenCandidateError(
            "candidate_id must be a lower-case identifier of at most 64 "
            "characters without punctuation."
        )
    rationale = str(payload["rationale"]).strip()
    if not rationale:
        raise OpenCandidateError("Candidate rationale must be non-empty.")
    grammar = config["grammar"]
    decay_payload = payload["decay"]
    if not isinstance(decay_payload, dict):
        raise OpenCandidateError("decay must be a JSON expression node.")
    root = str(decay_payload.get("op", ""))
    if root not in set(grammar["decay_roots"]):
        raise OpenCandidateError(f"Unsupported decay root: {root!r}.")
    registry: dict[str, ShapeParameter] = {}
    if root in {"exp_decay", "clipped_decay"}:
        _require_exact_keys(decay_payload, {"op", "argument"})
        argument, nodes = _validate_progress(
            decay_payload["argument"],
            grammar,
            registry,
            depth=2,
        )
        decay = {"op": root, "argument": argument}
        node_count = nodes + 1
    else:
        _require_exact_keys(
            decay_payload,
            {"op", "argument", "tail_exponent"},
        )
        argument, nodes = _validate_progress(
            decay_payload["argument"],
            grammar,
            registry,
            depth=2,
        )
        exponent = _validate_exponent(
            decay_payload["tail_exponent"],
            grammar,
            registry,
        )
        decay = {
            "op": root,
            "argument": argument,
            "tail_exponent": exponent,
        }
        node_count = nodes + 1 + int(isinstance(exponent, dict))
    if node_count > int(grammar["maximum_nodes"]):
        raise OpenCandidateError(
            f"Expression has {node_count} nodes, exceeding the grammar limit."
        )
    parameters = tuple(sorted(registry.values(), key=lambda item: item.name))
    if len(parameters) > int(grammar["maximum_free_shape_parameters"]):
        raise OpenCandidateError(
            "Expression exceeds the free shape-parameter budget."
        )
    structural_key = json.dumps(
        _alpha_normalized(decay),
        sort_keys=True,
        separators=(",", ":"),
    )
    spec = OpenCandidateSpec(
        candidate_id=candidate_id,
        rationale=rationale,
        decay=decay,
        parameters=parameters,
        structural_key=structural_key,
        node_count=node_count,
        complexity=3 + node_count + len(parameters),
    )
    audit = audit_open_candidate(spec, config)
    if audit["violation_count"]:
        raise OpenCandidateError(
            "Candidate failed structural physical audit: "
            + ", ".join(audit["violations"])
        )
    return spec


def _parameter_value(
    payload: float | dict[str, str],
    parameters: dict[str, float],
) -> float:
    if isinstance(payload, dict):
        name = str(payload["name"])
        if name not in parameters:
            raise OpenCandidateError(f"Missing fitted shape parameter: {name}.")
        value = float(parameters[name])
    else:
        value = float(payload)
    if not np.isfinite(value) or value <= 0.0:
        raise OpenCandidateError("Shape parameters and exponents must be positive.")
    return value


def evaluate_progress(
    node: dict[str, Any],
    z: np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray:
    operation = node["op"]
    if operation == "z":
        return np.asarray(z, dtype=float)
    if operation == "power":
        return np.power(
            evaluate_progress(node["argument"], z, parameters),
            _parameter_value(node["exponent"], parameters),
        )
    if operation == "scale":
        return _parameter_value(node["factor"], parameters) * evaluate_progress(
            node["argument"],
            z,
            parameters,
        )
    values = [
        evaluate_progress(argument, z, parameters)
        for argument in node["arguments"]
    ]
    if operation == "add":
        return np.sum(np.stack(values), axis=0)
    if operation == "multiply":
        return np.prod(np.stack(values), axis=0)
    raise OpenCandidateError(f"Unsupported normalized progress node: {operation}.")


def evaluate_decay(
    node: dict[str, Any],
    z: float | np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray:
    normalized = np.maximum(np.asarray(z, dtype=float), 0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        progress = evaluate_progress(node["argument"], normalized, parameters)
        if node["op"] == "exp_decay":
            return np.exp(-progress)
        if node["op"] == "rational_decay":
            exponent = _parameter_value(node["tail_exponent"], parameters)
            return np.power(1.0 + progress, -exponent)
        if node["op"] == "clipped_decay":
            return np.maximum(1.0 - progress, 0.0)
    raise OpenCandidateError(f"Unsupported normalized decay root: {node['op']}.")


def audit_open_candidate(
    spec: OpenCandidateSpec,
    config: dict[str, Any],
) -> dict[str, Any]:
    verification = config["verification"]
    z = np.linspace(
        0.0,
        float(verification["normalized_domain_max"]),
        int(verification["audit_points"]),
    )
    initial = {
        parameter.name: parameter.initial for parameter in spec.parameters
    }
    decay = np.asarray(evaluate_decay(spec.decay, z, initial), dtype=float)
    violations: list[str] = []
    if bool(verification["require_finite"]) and not np.all(np.isfinite(decay)):
        violations.append("non_finite_decay")
    finite = decay[np.isfinite(decay)]
    if bool(verification["require_nonnegative"]) and (
        not len(finite) or float(np.min(finite)) < -float(
            verification["monotonicity_tolerance"]
        )
    ):
        violations.append("negative_decay")
    if bool(verification["require_unit_initial_value"]) and (
        not len(decay)
        or not np.isfinite(decay[0])
        or abs(float(decay[0]) - 1.0)
        > float(verification["endpoint_tolerance"])
    ):
        violations.append("initial_decay_not_one")
    if bool(verification["require_nonincreasing"]) and len(finite) > 1:
        if float(np.max(np.diff(finite))) > float(
            verification["monotonicity_tolerance"]
        ):
            violations.append("decay_increases")
    if len(decay) and np.isfinite(decay[-1]):
        if float(decay[-1]) < float(
            verification["minimum_decay_at_domain_end"]
        ) - float(verification["monotonicity_tolerance"]):
            violations.append("terminal_decay_below_bound")
        if float(decay[-1]) > float(
            verification["maximum_decay_at_domain_end"]
        ) + float(verification["monotonicity_tolerance"]):
            violations.append("terminal_decay_above_bound")
    return {
        "violation_count": len(violations),
        "violations": violations,
        "normalized_domain_max": float(z[-1]),
        "minimum_decay": float(np.nanmin(decay)),
        "maximum_decay": float(np.nanmax(decay)),
        "terminal_decay": float(decay[-1]),
        "initial_parameter_values": initial,
    }


def _render_parameter(
    payload: float | dict[str, str],
    parameter_values: dict[str, float] | None,
) -> str:
    if isinstance(payload, dict):
        name = str(payload["name"])
        if parameter_values is not None and name in parameter_values:
            return f"{float(parameter_values[name]):.8g}"
        return name
    return f"{float(payload):g}"


def render_progress(
    node: dict[str, Any],
    parameter_values: dict[str, float] | None = None,
    variable_text: str = "z",
) -> str:
    operation = node["op"]
    if operation == "z":
        return variable_text
    if operation == "power":
        argument = render_progress(
            node["argument"],
            parameter_values,
            variable_text,
        )
        exponent = _render_parameter(node["exponent"], parameter_values)
        return f"({argument})^{exponent}"
    if operation == "scale":
        factor = _render_parameter(node["factor"], parameter_values)
        argument = render_progress(
            node["argument"],
            parameter_values,
            variable_text,
        )
        return f"{factor}*({argument})"
    separator = " + " if operation == "add" else " * "
    arguments = [
        render_progress(argument, parameter_values, variable_text)
        for argument in node["arguments"]
    ]
    return "(" + separator.join(arguments) + ")"


def render_decay(
    node: dict[str, Any],
    parameter_values: dict[str, float] | None = None,
    variable_text: str = "z",
) -> str:
    progress = render_progress(
        node["argument"],
        parameter_values,
        variable_text,
    )
    if node["op"] == "exp_decay":
        return f"exp(-({progress}))"
    if node["op"] == "rational_decay":
        exponent = _render_parameter(
            node["tail_exponent"],
            parameter_values,
        )
        return f"(1 + ({progress}))^(-{exponent})"
    if node["op"] == "clipped_decay":
        return f"max(1 - ({progress}), 0)"
    raise OpenCandidateError(f"Unsupported decay root: {node['op']}.")


def known_family_structural_keys(
    config: dict[str, Any],
) -> dict[str, str]:
    payloads = {
        "linear_clipped": {
            "candidate_id": "known_linear",
            "rationale": "Known clipped-linear reference.",
            "decay": {
                "op": "clipped_decay",
                "argument": {"op": "z"},
            },
        },
        "bilinear": {
            "candidate_id": "known_bilinear",
            "rationale": "Known algebraically equivalent clipped-linear reference.",
            "decay": {
                "op": "clipped_decay",
                "argument": {"op": "z"},
            },
        },
        "exponential": {
            "candidate_id": "known_exponential",
            "rationale": "Known exponential reference.",
            "decay": {
                "op": "exp_decay",
                "argument": {"op": "z"},
            },
        },
        "rational": {
            "candidate_id": "known_rational",
            "rationale": "Known rational reference.",
            "decay": {
                "op": "rational_decay",
                "argument": {"op": "z"},
                "tail_exponent": 1.0,
            },
        },
        "stretched_exponential": {
            "candidate_id": "known_stretched",
            "rationale": "Known stretched-exponential reference.",
            "decay": {
                "op": "exp_decay",
                "argument": {
                    "op": "power",
                    "argument": {"op": "z"},
                    "exponent": {
                        "name": "p",
                        "role": "positive_exponent",
                    },
                },
            },
        },
    }
    return {
        family: validate_open_candidate(payload, config).structural_key
        for family, payload in payloads.items()
    }


def validate_open_candidate_batch(
    payloads: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[OpenCandidateSpec], list[dict[str, str]]]:
    if not isinstance(payloads, list):
        raise OpenCandidateError("Candidate batch must be a JSON list.")
    accepted: list[OpenCandidateSpec] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    known = known_family_structural_keys(config)
    known_by_key: dict[str, list[str]] = {}
    for family, key in known.items():
        known_by_key.setdefault(key, []).append(family)
    generation = config["generation"]
    for index, payload in enumerate(payloads):
        candidate_id = (
            str(payload.get("candidate_id", f"candidate_{index}"))
            if isinstance(payload, dict)
            else f"candidate_{index}"
        )
        try:
            spec = validate_open_candidate(payload, config)
            if (
                bool(generation["reject_known_family_duplicates"])
                and spec.structural_key in known_by_key
            ):
                raise OpenCandidateError(
                    "Candidate duplicates known family: "
                    + ", ".join(sorted(known_by_key[spec.structural_key]))
                )
            if (
                bool(generation["reject_batch_duplicates"])
                and spec.structural_key in seen
            ):
                raise OpenCandidateError(
                    "Candidate duplicates an earlier accepted batch structure."
                )
            seen.add(spec.structural_key)
            accepted.append(spec)
        except (OpenCandidateError, TypeError, KeyError) as exc:
            rejected.append(
                {
                    "candidate_id": candidate_id,
                    "reason": str(exc),
                }
            )
    return accepted, rejected


def random_open_candidate_specs(
    config: dict[str, Any],
    count: int,
    seed: int,
) -> tuple[list[OpenCandidateSpec], list[dict[str, str]]]:
    if count < 1:
        raise ValueError("Random open-candidate count must be positive.")
    rng = np.random.default_rng(int(seed))
    grammar = config["grammar"]
    maximum_attempts = int(
        config["random_generation"]["maximum_generation_attempts"]
    )
    accepted: list[OpenCandidateSpec] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    parameter_index = 0

    def parameter(role: str) -> dict[str, str]:
        nonlocal parameter_index
        parameter_index += 1
        return {"name": f"r{parameter_index}", "role": role}

    def progress(depth: int) -> dict[str, Any]:
        if depth >= int(grammar["maximum_depth"]) - 1 or rng.random() < 0.28:
            return {"op": "z"}
        operation = str(
            rng.choice(["power", "scale", "add", "multiply"])
        )
        if operation == "power":
            exponent: float | dict[str, str]
            if rng.random() < 0.5:
                exponent = float(
                    rng.choice(grammar["allowed_fixed_exponents"])
                )
            else:
                exponent = parameter("positive_exponent")
            return {
                "op": "power",
                "argument": progress(depth + 1),
                "exponent": exponent,
            }
        if operation == "scale":
            return {
                "op": "scale",
                "argument": progress(depth + 1),
                "factor": parameter("positive_weight"),
            }
        maximum = int(
            grammar[
                "maximum_add_arguments"
                if operation == "add"
                else "maximum_multiply_arguments"
            ]
        )
        argument_count = int(rng.integers(2, maximum + 1))
        return {
            "op": operation,
            "arguments": [
                progress(depth + 1) for _ in range(argument_count)
            ],
        }

    for attempt in range(maximum_attempts):
        if len(accepted) >= count:
            break
        parameter_index = 0
        root = str(rng.choice(grammar["decay_roots"]))
        decay: dict[str, Any] = {
            "op": root,
            "argument": progress(2),
        }
        if root == "rational_decay":
            decay["tail_exponent"] = (
                parameter("positive_exponent")
                if rng.random() < 0.65
                else float(rng.choice(grammar["allowed_fixed_exponents"]))
            )
        payload = {
            "candidate_id": f"random_{attempt + 1}",
            "rationale": "Seeded random typed-grammar comparator.",
            "decay": decay,
        }
        batch, failures = validate_open_candidate_batch([payload], config)
        rejected.extend(failures)
        if not batch:
            continue
        spec = batch[0]
        if spec.structural_key in seen:
            rejected.append(
                {
                    "candidate_id": spec.candidate_id,
                    "reason": "Duplicate random structural key.",
                }
            )
            continue
        seen.add(spec.structural_key)
        accepted.append(spec)
    if len(accepted) < count:
        raise OpenCandidateError(
            f"Random grammar produced only {len(accepted)} unique valid "
            f"candidates after {maximum_attempts} attempts."
        )
    return accepted, rejected


def _initial_open_coefficients(
    spec: OpenCandidateSpec,
    config: dict[str, Any],
) -> np.ndarray:
    guess = config["search"]["initial_guess"]
    peak = float(guess["peak_cohesion_mpa"])
    residual = float(guess["residual_cohesion_mpa"])
    return np.asarray(
        [
            residual,
            max(peak - residual, 1.0e-6),
            float(guess["softening_scale"]),
            *[parameter.initial for parameter in spec.parameters],
        ],
        dtype=float,
    )


def _open_coefficient_bounds(
    spec: OpenCandidateSpec,
) -> tuple[np.ndarray, np.ndarray]:
    lower = [0.0, 0.0, 1.0e-7]
    upper = [10.0, 10.0, 0.1]
    lower.extend(parameter.lower for parameter in spec.parameters)
    upper.extend(parameter.upper for parameter in spec.parameters)
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def fit_open_c3_evolution(
    frame: pd.DataFrame,
    spec: OpenCandidateSpec,
    config: dict[str, Any],
) -> OpenCohesionEvolution:
    from asrc.constitutive.c3_search import (
        c3_known_parameters,
        observed_residual_vector,
        replay_c3_paths,
    )

    parameters = c3_known_parameters(config)
    initial = _initial_open_coefficients(spec, config)
    lower, upper = _open_coefficient_bounds(spec)
    state_weight = float(config["search"]["state_residual_weight_mpa"])

    def residuals(coefficients: np.ndarray) -> np.ndarray:
        model = OpenCohesionEvolution(
            spec,
            tuple(float(value) for value in coefficients),
        )
        predictions = replay_c3_paths(frame, model, parameters)
        return observed_residual_vector(
            frame,
            predictions,
            state_weight,
        )

    result = least_squares(
        residuals,
        np.clip(initial, lower + 1.0e-12, upper - 1.0e-12),
        bounds=(lower, upper),
        max_nfev=int(config["search"]["maximum_function_evaluations"]),
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
    )
    return OpenCohesionEvolution(
        spec,
        tuple(float(value) for value in result.x),
    )

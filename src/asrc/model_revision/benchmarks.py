from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from asrc.model_revision.ast import apply_repair, evaluate_expression
from asrc.model_revision.proposals import (
    RepairContract,
    RepairRequest,
    TypedRepair,
    validate_expression_ast,
    validate_typed_repair,
)


def _constant(value: float) -> dict[str, Any]:
    return {"op": "constant", "value": float(value)}


def _variable(name: str) -> dict[str, Any]:
    return {"op": "variable", "name": name}


def _parameter(name: str) -> dict[str, Any]:
    return {"op": "parameter", "name": name}


def _add(*arguments: dict[str, Any]) -> dict[str, Any]:
    return {"op": "add", "arguments": list(arguments)}


def _multiply(*arguments: dict[str, Any]) -> dict[str, Any]:
    return {"op": "multiply", "arguments": list(arguments)}


def _power(left: dict[str, Any], exponent: float) -> dict[str, Any]:
    return {"op": "power", "left": left, "right": _constant(exponent)}


@dataclass(frozen=True)
class Gate1TaskDefinition:
    task_id: str
    task_family: str
    target: str
    variable_descriptions: Mapping[str, str]
    observed_ranges: Mapping[str, tuple[float, float]]
    locked_ranges: Mapping[str, tuple[float, float]]
    baseline_expression: dict[str, Any]
    oracle_payload: dict[str, Any]
    oracle_parameters: Mapping[str, float]
    noise_std: float
    constraints: tuple[str, ...] = ()
    minimum_output: float | None = None
    monotonicity: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Gate1Task:
    definition: Gate1TaskDefinition
    contract: RepairContract
    baseline_expression: dict[str, Any]
    oracle_repair: TypedRepair
    observed: pd.DataFrame
    locked: pd.DataFrame
    audit: pd.DataFrame
    request: RepairRequest

    @property
    def task_id(self) -> str:
        return self.definition.task_id

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self.definition.variable_descriptions)


def gate1_task_definitions() -> tuple[Gate1TaskDefinition, ...]:
    x1 = _variable("x1")
    x2 = _variable("x2")
    return (
        Gate1TaskDefinition(
            task_id="T01",
            task_family="algebraic",
            target="model_output",
            variable_descriptions={"x1": "dimensionless input coordinate"},
            observed_ranges={"x1": (-1.0, 1.0)},
            locked_ranges={"x1": (-1.8, 1.8)},
            baseline_expression=_add(_constant(0.4), _multiply(_constant(1.1), x1)),
            oracle_payload={
                "proposal_id": "oracle_t01",
                "source": "replay",
                "edit_type": "add_term",
                "target": "model_output",
                "expression": _multiply(_parameter("amplitude"), _power(x1, 2.0)),
                "rationale": "Internal generator only.",
                "expected_signature": "Internal generator only.",
            },
            oracle_parameters={"amplitude": 0.65},
            noise_std=0.02,
        ),
        Gate1TaskDefinition(
            task_id="T02",
            task_family="algebraic",
            target="model_output",
            variable_descriptions={
                "x1": "dimensionless input coordinate 1",
                "x2": "dimensionless input coordinate 2",
            },
            observed_ranges={"x1": (-1.0, 1.0), "x2": (-1.0, 1.0)},
            locked_ranges={"x1": (-1.7, 1.7), "x2": (-1.7, 1.7)},
            baseline_expression=_add(
                _constant(1.5),
                _multiply(_constant(0.4), x1),
                _multiply(_constant(-0.2), x2),
            ),
            oracle_payload={
                "proposal_id": "oracle_t02",
                "source": "replay",
                "edit_type": "multiply_term",
                "target": "model_output",
                "expression": _add(
                    _constant(1.0),
                    _multiply(_parameter("coupling"), x1, x2),
                ),
                "rationale": "Internal generator only.",
                "expected_signature": "Internal generator only.",
            },
            oracle_parameters={"coupling": 0.30},
            noise_std=0.02,
            minimum_output=0.0,
        ),
        Gate1TaskDefinition(
            task_id="T03",
            task_family="dynamics_rhs",
            target="evolution_law",
            variable_descriptions={
                "x1": "dimensionless state coordinate",
                "x2": "dimensionless rate coordinate",
            },
            observed_ranges={"x1": (-1.2, 1.2), "x2": (-1.0, 1.0)},
            locked_ranges={"x1": (-2.0, 2.0), "x2": (-1.7, 1.7)},
            baseline_expression=_multiply(_constant(-0.8), x1),
            oracle_payload={
                "proposal_id": "oracle_t03",
                "source": "replay",
                "edit_type": "add_term",
                "target": "evolution_law",
                "expression": _multiply(_parameter("damping"), _power(x2, 3.0)),
                "rationale": "Internal generator only.",
                "expected_signature": "Internal generator only.",
            },
            oracle_parameters={"damping": -0.28},
            noise_std=0.025,
        ),
        Gate1TaskDefinition(
            task_id="T04",
            task_family="dynamics_rhs",
            target="evolution_law",
            variable_descriptions={
                "x1": "dimensionless state coordinate",
                "x2": "dimensionless rate coordinate",
            },
            observed_ranges={"x1": (-1.5, 1.5), "x2": (-1.0, 1.0)},
            locked_ranges={"x1": (-2.4, 2.4), "x2": (-1.6, 1.6)},
            baseline_expression=_add(
                _multiply(_constant(-0.5), x1),
                _multiply(_constant(-0.15), x2),
            ),
            oracle_payload={
                "proposal_id": "oracle_t04",
                "source": "replay",
                "edit_type": "add_state_dependence",
                "target": "evolution_law",
                "expression": _multiply(
                    _parameter("amplitude"),
                    {"op": "sin", "argument": x1},
                    x2,
                ),
                "rationale": "Internal generator only.",
                "expected_signature": "Internal generator only.",
            },
            oracle_parameters={"amplitude": 0.42},
            noise_std=0.025,
        ),
        Gate1TaskDefinition(
            task_id="T05",
            task_family="material_point",
            target="evolution_law",
            variable_descriptions={
                "x1": "normalized loading coordinate",
                "x2": "normalized internal-state coordinate",
            },
            observed_ranges={"x1": (0.0, 1.2), "x2": (0.0, 1.0)},
            locked_ranges={"x1": (0.0, 2.0), "x2": (0.0, 1.4)},
            baseline_expression=_add(
                _constant(0.9),
                _multiply(_constant(-0.25), x2),
            ),
            oracle_payload={
                "proposal_id": "oracle_t05",
                "source": "replay",
                "edit_type": "add_state_dependence",
                "target": "evolution_law",
                "expression": _multiply(
                    _parameter("amplitude"),
                    {
                        "op": "exp",
                        "argument": _multiply(_parameter("rate"), x1),
                    },
                ),
                "rationale": "Internal generator only.",
                "expected_signature": "Internal generator only.",
            },
            oracle_parameters={"amplitude": 0.55, "rate": -1.6},
            noise_std=0.015,
            constraints=("corrected output remains non-negative on the observed domain",),
            minimum_output=0.0,
        ),
        Gate1TaskDefinition(
            task_id="T06",
            task_family="material_point",
            target="model_output",
            variable_descriptions={
                "x1": "normalized loading coordinate",
                "x2": "normalized internal-state coordinate",
            },
            observed_ranges={"x1": (0.0, 1.0), "x2": (-0.6, 0.6)},
            locked_ranges={"x1": (0.0, 1.4), "x2": (-1.1, 1.1)},
            baseline_expression=_add(
                _constant(0.5),
                _multiply(_constant(1.5), x1),
            ),
            oracle_payload={
                "proposal_id": "oracle_t06",
                "source": "replay",
                "edit_type": "add_bounded_transition",
                "target": "model_output",
                "expression": _multiply(
                    _parameter("amplitude"),
                    {
                        "op": "tanh",
                        "argument": _multiply(
                            _parameter("slope"),
                            _add(x2, _parameter("shift")),
                        ),
                    },
                ),
                "rationale": "Internal generator only.",
                "expected_signature": "Internal generator only.",
            },
            oracle_parameters={"amplitude": -0.35, "slope": 3.0, "shift": -0.12},
            noise_std=0.015,
            constraints=("corrected output remains non-negative on the observed domain",),
            minimum_output=0.0,
        ),
    )


def _sample_uniform(
    rng: np.random.Generator,
    ranges: Mapping[str, tuple[float, float]],
    count: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            name: rng.uniform(float(bounds[0]), float(bounds[1]), int(count))
            for name, bounds in ranges.items()
        }
    )


def _sample_locked_shell(
    rng: np.random.Generator,
    observed_ranges: Mapping[str, tuple[float, float]],
    locked_ranges: Mapping[str, tuple[float, float]],
    count: int,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    collected = 0
    while collected < count:
        candidate = _sample_uniform(rng, locked_ranges, max(count, 32))
        outside = np.zeros(len(candidate), dtype=bool)
        for name, bounds in observed_ranges.items():
            outside |= (candidate[name].to_numpy(float) < float(bounds[0])) | (
                candidate[name].to_numpy(float) > float(bounds[1])
            )
        selected = candidate.loc[outside]
        chunks.append(selected)
        collected += len(selected)
        if len(chunks) > 100:
            raise RuntimeError("Unable to sample the locked outer shell.")
    return pd.concat(chunks, ignore_index=True).iloc[:count].reset_index(drop=True)


def _variables(frame: pd.DataFrame, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: frame[name].to_numpy(float) for name in names}


def _residual_evidence(
    observed: pd.DataFrame,
    variable_names: tuple[str, ...],
) -> dict[str, Any]:
    fit = observed.loc[observed["partition"].eq("fit")].copy()
    residual = fit["target"].to_numpy(float) - fit["baseline"].to_numpy(float)
    transforms: dict[str, np.ndarray] = {}
    for name in variable_names:
        values = fit[name].to_numpy(float)
        transforms[name] = values
        transforms[f"{name}_squared"] = values**2
        transforms[f"abs_{name}"] = np.abs(values)
    for left_index, left in enumerate(variable_names):
        for right in variable_names[left_index + 1 :]:
            transforms[f"{left}_times_{right}"] = (
                fit[left].to_numpy(float) * fit[right].to_numpy(float)
            )

    correlations = {}
    for name, values in transforms.items():
        if float(np.std(values)) <= 1.0e-12 or float(np.std(residual)) <= 1.0e-12:
            correlations[name] = 0.0
        else:
            correlations[name] = float(np.corrcoef(values, residual)[0, 1])
    sorted_fit = fit.sort_values(list(variable_names)).reset_index(drop=True)
    probe_count = min(16, len(sorted_fit))
    probe_indices = np.unique(
        np.linspace(0, len(sorted_fit) - 1, probe_count, dtype=int)
    )
    probes = []
    for index in probe_indices:
        row = sorted_fit.iloc[int(index)]
        probes.append(
            {
                **{name: float(row[name]) for name in variable_names},
                "baseline": float(row["baseline"]),
                "residual": float(row["target"] - row["baseline"]),
            }
        )
    return {
        "fit_count": int(len(fit)),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "residual_quantiles": {
            "q10": float(np.quantile(residual, 0.10)),
            "q50": float(np.quantile(residual, 0.50)),
            "q90": float(np.quantile(residual, 0.90)),
        },
        "diagnostic_correlations": correlations,
        "fit_only_residual_probes": probes,
    }


def build_gate1_task(
    definition: Gate1TaskDefinition,
    *,
    seed: int,
    observed_count: int,
    locked_count: int,
    audit_count: int,
    fit_fraction: float,
    maximum_candidates: int,
) -> Gate1Task:
    if not 0.5 <= fit_fraction < 1.0:
        raise ValueError("fit_fraction must be in [0.5, 1.0).")
    rng = np.random.default_rng(int(seed))
    variable_names = tuple(definition.variable_descriptions)
    unknown_monotonic_variables = set(definition.monotonicity) - set(variable_names)
    if unknown_monotonic_variables:
        raise ValueError(
            "Monotonicity references unknown variables: "
            f"{sorted(unknown_monotonic_variables)}"
        )
    invalid_directions = {
        str(direction)
        for direction in definition.monotonicity.values()
        if str(direction) not in {"increasing", "decreasing"}
    }
    if invalid_directions:
        raise ValueError(
            "Monotonicity directions must be increasing or decreasing: "
            f"{sorted(invalid_directions)}"
        )
    contract = RepairContract(
        allowed_variables=frozenset(variable_names),
        allowed_targets=frozenset({definition.target}),
    )
    baseline_expression = validate_expression_ast(
        definition.baseline_expression,
        contract,
    )
    oracle_repair = validate_typed_repair(definition.oracle_payload, contract)

    observed = _sample_uniform(rng, definition.observed_ranges, observed_count)
    permutation = rng.permutation(observed_count)
    fit_count = int(round(observed_count * fit_fraction))
    partition = np.full(observed_count, "validation", dtype=object)
    partition[permutation[:fit_count]] = "fit"
    observed["partition"] = partition
    locked = _sample_locked_shell(
        rng,
        definition.observed_ranges,
        definition.locked_ranges,
        locked_count,
    )
    locked["partition"] = "locked"
    audit = _sample_uniform(rng, definition.observed_ranges, audit_count)

    for frame, noisy in ((observed, True), (locked, False)):
        variables = _variables(frame, variable_names)
        baseline = evaluate_expression(baseline_expression, variables)
        target = apply_repair(
            oracle_repair,
            baseline,
            variables,
            definition.oracle_parameters,
        )
        if noisy:
            target = target + rng.normal(0.0, definition.noise_std, len(frame))
        frame["baseline"] = baseline
        frame["target"] = target

    public_context = {
        "task_id": definition.task_id,
        "target_kind": definition.target,
        "variables": dict(definition.variable_descriptions),
        "baseline_expression": baseline_expression,
        "residual_evidence": _residual_evidence(observed, variable_names),
        "constraints": list(definition.constraints),
    }
    request = RepairRequest(
        baseline_expression=public_context["baseline_expression"],
        residual_evidence={
            "task_id": public_context["task_id"],
            "target_kind": public_context["target_kind"],
            "variables": public_context["variables"],
            **public_context["residual_evidence"],
        },
        constraints=definition.constraints,
        maximum_candidates=maximum_candidates,
    )
    return Gate1Task(
        definition=definition,
        contract=contract,
        baseline_expression=baseline_expression,
        oracle_repair=oracle_repair,
        observed=observed,
        locked=locked,
        audit=audit,
        request=request,
    )


def build_gate1_suite(config: Mapping[str, Any], *, seed: int) -> tuple[Gate1Task, ...]:
    sampling = config["sampling"]
    budget = config["candidate_budget"]
    tasks = []
    for index, definition in enumerate(gate1_task_definitions()):
        tasks.append(
            build_gate1_task(
                definition,
                seed=int(seed) + 1009 * index,
                observed_count=int(sampling["observed_count"]),
                locked_count=int(sampling["locked_count"]),
                audit_count=int(sampling["audit_count"]),
                fit_fraction=float(sampling["fit_fraction"]),
                maximum_candidates=int(budget["maximum_unique_candidates"]),
            )
        )
    return tuple(tasks)

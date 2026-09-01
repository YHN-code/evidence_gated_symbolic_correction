from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from asrc.model_revision.benchmarks import (
    Gate1Task,
    Gate1TaskDefinition,
    build_gate1_task,
)
from asrc.model_revision.proposals import RepairRequest


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


def _subtract(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"op": "subtract", "left": left, "right": right}


def _divide(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"op": "divide", "left": left, "right": right}


def _power(left: dict[str, Any], exponent: float) -> dict[str, Any]:
    return {"op": "power", "left": left, "right": _constant(exponent)}


def _unary(operation: str, argument: dict[str, Any]) -> dict[str, Any]:
    return {"op": operation, "argument": argument}


@dataclass(frozen=True)
class MechanismTransferDefinition:
    task: Gate1TaskDefinition
    public_semantics: Mapping[str, Any]
    knowledge_topics: tuple[str, ...]
    generator_sources: tuple[str, ...]
    mechanism_class: str


@dataclass(frozen=True)
class MechanismTransferTask:
    task: Gate1Task
    public_semantics: Mapping[str, Any]
    knowledge_topics: tuple[str, ...]
    generator_sources: tuple[str, ...]
    mechanism_class: str


def mechanism_transfer_definitions() -> tuple[MechanismTransferDefinition, ...]:
    x1 = _variable("x1")
    x2 = _variable("x2")
    return (
        MechanismTransferDefinition(
            task=Gate1TaskDefinition(
                task_id="XFER-CREEP-01",
                task_family="time_dependent_creep",
                target="normalized_creep_strain",
                variable_descriptions={
                    "x1": "normalized time under sustained loading",
                    "x2": "positive normalized deviatoric stress",
                },
                observed_ranges={"x1": (0.02, 0.80), "x2": (0.55, 1.20)},
                locked_ranges={"x1": (0.0, 2.00), "x2": (0.35, 1.50)},
                baseline_expression=_add(
                    _multiply(_constant(0.02), x2),
                    _multiply(_constant(0.06), x1, x2),
                ),
                oracle_payload={
                    "proposal_id": "oracle_xfer_creep_01",
                    "source": "replay",
                    "edit_type": "add_state_dependence",
                    "target": "normalized_creep_strain",
                    "expression": _multiply(
                        _parameter("delayed_amplitude"),
                        x2,
                        _subtract(
                            _constant(1.0),
                            _unary(
                                "exp",
                                _multiply(_parameter("retardation_rate"), x1),
                            ),
                        ),
                    ),
                    "rationale": "Private benchmark generator.",
                    "expected_signature": "Private benchmark generator.",
                },
                oracle_parameters={
                    "delayed_amplitude": 0.22,
                    "retardation_rate": -2.40,
                },
                noise_std=0.002,
                constraints=(
                    "creep strain remains finite and non-negative on the declared domain",
                    "creep strain does not decrease with time under fixed positive stress",
                ),
                minimum_output=0.0,
                monotonicity={"x1": "increasing", "x2": "increasing"},
            ),
            public_semantics={
                "response": "normalized axial creep strain",
                "variables": {
                    "x1": "elapsed loading time normalized by a reference duration",
                    "x2": "deviatoric stress normalized by a positive reference stress",
                },
                "baseline_role": "instantaneous plus steady linear creep approximation",
                "loading_protocol": "constant-stress creep below rupture",
            },
            knowledge_topics=(
                "creep",
                "viscoelasticity",
                "time_dependence",
            ),
            generator_sources=("https://doi.org/10.1038/s41598-021-03539-7",),
            mechanism_class="creep",
        ),
        MechanismTransferDefinition(
            task=Gate1TaskDefinition(
                task_id="XFER-PERM-01",
                task_family="stress_sensitive_permeability",
                target="normalized_permeability",
                variable_descriptions={"x1": "positive normalized effective stress"},
                observed_ranges={"x1": (0.0, 0.85)},
                locked_ranges={"x1": (0.0, 2.20)},
                baseline_expression=_add(
                    _constant(1.0), _multiply(_constant(-0.12), x1)
                ),
                oracle_payload={
                    "proposal_id": "oracle_xfer_perm_01",
                    "source": "replay",
                    "edit_type": "add_term",
                    "target": "normalized_permeability",
                    "expression": _multiply(
                        _parameter("closure_amplitude"),
                        _subtract(
                            _unary(
                                "exp",
                                _multiply(_parameter("stress_sensitivity"), x1),
                            ),
                            _constant(1.0),
                        ),
                    ),
                    "rationale": "Private benchmark generator.",
                    "expected_signature": "Private benchmark generator.",
                },
                oracle_parameters={
                    "closure_amplitude": 0.32,
                    "stress_sensitivity": -1.10,
                },
                noise_std=0.006,
                constraints=(
                    "permeability remains finite and positive on the declared domain",
                    "permeability does not increase with effective stress",
                ),
                minimum_output=0.0,
                monotonicity={"x1": "decreasing"},
            ),
            public_semantics={
                "response": "permeability normalized by its zero-stress value",
                "variables": {
                    "x1": "effective confining stress normalized by a positive reference stress"
                },
                "baseline_role": "first-order stress-sensitivity approximation",
                "loading_protocol": "monotonic increase in effective stress",
            },
            knowledge_topics=(
                "permeability",
                "effective_stress",
                "crack_closure",
            ),
            generator_sources=("https://doi.org/10.1016/j.jngse.2016.01.034",),
            mechanism_class="permeability",
        ),
        MechanismTransferDefinition(
            task=Gate1TaskDefinition(
                task_id="XFER-CYCLIC-01",
                task_family="cyclic_accumulation",
                target="normalized_irrecoverable_strain",
                variable_descriptions={
                    "x1": "positive normalized cycle count",
                    "x2": "positive normalized cyclic stress amplitude",
                },
                observed_ranges={"x1": (0.02, 0.90), "x2": (0.30, 0.80)},
                locked_ranges={"x1": (0.0, 2.20), "x2": (0.20, 1.10)},
                baseline_expression=_multiply(_constant(0.015), x1, x2),
                oracle_payload={
                    "proposal_id": "oracle_xfer_cyclic_01",
                    "source": "replay",
                    "edit_type": "add_state_dependence",
                    "target": "normalized_irrecoverable_strain",
                    "expression": _multiply(
                        _parameter("damage_amplitude"),
                        _power(x2, 2.0),
                        _divide(
                            _power(x1, 2.0),
                            _add(
                                _constant(1.0),
                                _multiply(_parameter("saturation"), x1),
                            ),
                        ),
                    ),
                    "rationale": "Private benchmark generator.",
                    "expected_signature": "Private benchmark generator.",
                },
                oracle_parameters={"damage_amplitude": 0.08, "saturation": 1.40},
                noise_std=0.0015,
                constraints=(
                    "accumulated irrecoverable strain remains finite and non-negative",
                    "accumulation does not decrease with cycle count or stress amplitude",
                ),
                minimum_output=0.0,
                monotonicity={"x1": "increasing", "x2": "increasing"},
            ),
            public_semantics={
                "response": "normalized accumulated irrecoverable axial strain",
                "variables": {
                    "x1": "cycle count normalized by a reference count",
                    "x2": "cyclic stress amplitude normalized by a reference amplitude",
                },
                "baseline_role": "linear cycle-amplitude accumulation approximation",
                "loading_protocol": "constant-amplitude cyclic loading before rupture",
            },
            knowledge_topics=(
                "cyclic_loading",
                "damage_accumulation",
                "loading_amplitude",
            ),
            generator_sources=(
                "https://doi.org/10.1061/(ASCE)GM.1943-5622.0002202",
            ),
            mechanism_class="cyclic_damage",
        ),
        MechanismTransferDefinition(
            task=Gate1TaskDefinition(
                task_id="XFER-THERMAL-01",
                task_family="thermomechanical_stiffness",
                target="normalized_stiffness",
                variable_descriptions={
                    "x1": "positive normalized thermal exposure",
                    "x2": "positive normalized confining support",
                },
                observed_ranges={"x1": (0.0, 0.75), "x2": (0.20, 1.00)},
                locked_ranges={"x1": (0.0, 1.40), "x2": (0.0, 1.50)},
                baseline_expression=_add(
                    _constant(1.0),
                    _multiply(_constant(-0.10), x1),
                    _multiply(_constant(0.04), x2),
                ),
                oracle_payload={
                    "proposal_id": "oracle_xfer_thermal_01",
                    "source": "replay",
                    "edit_type": "add_term",
                    "target": "normalized_stiffness",
                    "expression": _multiply(
                        _parameter("thermal_damage"),
                        _divide(
                            _power(x1, 2.0),
                            _add(
                                _constant(1.0),
                                _multiply(_parameter("confinement_mitigation"), x2),
                            ),
                        ),
                    ),
                    "rationale": "Private benchmark generator.",
                    "expected_signature": "Private benchmark generator.",
                },
                oracle_parameters={
                    "thermal_damage": -0.28,
                    "confinement_mitigation": 0.90,
                },
                noise_std=0.006,
                constraints=(
                    "normalized stiffness remains finite and positive",
                    "stiffness does not increase with thermal exposure at fixed confinement",
                ),
                minimum_output=0.0,
                monotonicity={"x1": "decreasing", "x2": "increasing"},
            ),
            public_semantics={
                "response": "elastic stiffness normalized by its reference-temperature value",
                "variables": {
                    "x1": "temperature exposure normalized by a positive reference increment",
                    "x2": "confining pressure normalized by a positive reference pressure",
                },
                "baseline_role": "uncoupled first-order temperature and confinement approximation",
                "loading_protocol": "thermal exposure followed by mechanical characterization",
            },
            knowledge_topics=(
                "thermal_damage",
                "thermomechanical_coupling",
                "stiffness",
            ),
            generator_sources=("https://doi.org/10.1016/j.ijrmms.2018.01.030",),
            mechanism_class="thermal_damage",
        ),
    )


def build_mechanism_transfer_suite(
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[MechanismTransferTask, ...]:
    sampling = config["sampling"]
    budget = config["candidate_budget"]
    tasks: list[MechanismTransferTask] = []
    for index, definition in enumerate(mechanism_transfer_definitions()):
        task = build_gate1_task(
            definition.task,
            seed=int(seed) + 1009 * index,
            observed_count=int(sampling["observed_count"]),
            locked_count=int(sampling["locked_count"]),
            audit_count=int(sampling["audit_count"]),
            fit_fraction=float(sampling["fit_fraction"]),
            maximum_candidates=int(budget["maximum_source_candidates"]),
        )
        request = RepairRequest(
            baseline_expression=task.request.baseline_expression,
            residual_evidence={
                **task.request.residual_evidence,
                "public_semantics": dict(definition.public_semantics),
            },
            constraints=task.request.constraints,
            failed_structural_keys=task.request.failed_structural_keys,
            maximum_candidates=task.request.maximum_candidates,
        )
        tasks.append(
            MechanismTransferTask(
                task=replace(task, request=request),
                public_semantics=dict(definition.public_semantics),
                knowledge_topics=definition.knowledge_topics,
                generator_sources=definition.generator_sources,
                mechanism_class=definition.mechanism_class,
            )
        )
    return tuple(tasks)

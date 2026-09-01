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


def _power(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"op": "power", "left": left, "right": right}


def _unary(operation: str, argument: dict[str, Any]) -> dict[str, Any]:
    return {"op": operation, "argument": argument}


@dataclass(frozen=True)
class SemanticBenchmarkDefinition:
    task: Gate1TaskDefinition
    public_semantics: Mapping[str, Any]
    knowledge_topics: tuple[str, ...]
    generator_sources: tuple[str, ...]
    mechanism_class: str


@dataclass(frozen=True)
class SemanticBenchmarkTask:
    task: Gate1Task
    public_semantics: Mapping[str, Any]
    knowledge_topics: tuple[str, ...]
    generator_sources: tuple[str, ...]
    mechanism_class: str


def semantic_benchmark_definitions() -> tuple[SemanticBenchmarkDefinition, ...]:
    x1 = _variable("x1")
    x2 = _variable("x2")
    sin_x1 = _unary("sin", x1)
    cos_x1 = _unary("cos", x1)
    return (
        SemanticBenchmarkDefinition(
            task=Gate1TaskDefinition(
                task_id="SEM-ORIENT-01",
                task_family="orientation_response",
                target="normalized_strength",
                variable_descriptions={"x1": "orientation angle in radians"},
                observed_ranges={"x1": (0.14, 1.43)},
                locked_ranges={"x1": (0.0, 1.5707963267948966)},
                baseline_expression=_constant(1.0),
                oracle_payload={
                    "proposal_id": "oracle_sem_orient_01",
                    "source": "replay",
                    "edit_type": "add_term",
                    "target": "normalized_strength",
                    "expression": _multiply(
                        _parameter("amplitude"),
                        _unary("sin", _multiply(_constant(2.0), x1)),
                    ),
                    "rationale": "Private benchmark generator.",
                    "expected_signature": "Private benchmark generator.",
                },
                oracle_parameters={"amplitude": -0.24},
                noise_std=0.008,
                minimum_output=0.0,
            ),
            public_semantics={
                "response": "normalized peak strength",
                "variables": {"x1": "angle between loading and a material plane, rad"},
                "baseline_role": "orientation-independent reference strength",
            },
            knowledge_topics=("orientation", "anisotropy", "weak_plane", "identifiability"),
            generator_sources=("https://doi.org/10.1017/S0016756800061100",),
            mechanism_class="orientation",
        ),
        SemanticBenchmarkDefinition(
            task=Gate1TaskDefinition(
                task_id="SEM-ORIENT-02",
                task_family="directional_material_response",
                target="normalized_compliance",
                variable_descriptions={"x1": "orientation angle in radians"},
                observed_ranges={"x1": (0.12, 1.45)},
                locked_ranges={"x1": (0.0, 1.5707963267948966)},
                baseline_expression=_add(
                    _constant(0.90),
                    _multiply(_constant(0.12), _power(cos_x1, _constant(2.0))),
                ),
                oracle_payload={
                    "proposal_id": "oracle_sem_orient_02",
                    "source": "replay",
                    "edit_type": "add_term",
                    "target": "normalized_compliance",
                    "expression": _multiply(
                        _parameter("interaction"),
                        _power(sin_x1, _constant(2.0)),
                        _power(cos_x1, _constant(2.0)),
                    ),
                    "rationale": "Private benchmark generator.",
                    "expected_signature": "Private benchmark generator.",
                },
                oracle_parameters={"interaction": -0.38},
                noise_std=0.006,
                minimum_output=0.0,
            ),
            public_semantics={
                "response": "normalized directional compliance",
                "variables": {"x1": "angle to the material symmetry axis, rad"},
                "baseline_role": "lowest-order directional approximation",
            },
            knowledge_topics=("orientation", "transverse_isotropy", "invariance", "identifiability"),
            generator_sources=("https://web.stanford.edu/~borja/pub/ijnamg2018%281%29.pdf",),
            mechanism_class="orientation",
        ),
        SemanticBenchmarkDefinition(
            task=Gate1TaskDefinition(
                task_id="SEM-STRAIN-01",
                task_family="stress_strain_response",
                target="normalized_deviatoric_stress",
                variable_descriptions={"x1": "normalized axial strain"},
                observed_ranges={"x1": (0.02, 0.65)},
                locked_ranges={"x1": (0.0, 1.45)},
                baseline_expression=_multiply(_constant(0.90), x1),
                oracle_payload={
                    "proposal_id": "oracle_sem_strain_01",
                    "source": "replay",
                    "edit_type": "add_term",
                    "target": "normalized_deviatoric_stress",
                    "expression": _add(
                        _divide(
                            _multiply(_parameter("scale"), x1),
                            _add(
                                _constant(1.0),
                                _multiply(_parameter("curvature"), x1),
                            ),
                        ),
                        _multiply(_parameter("linear_offset"), x1),
                    ),
                    "rationale": "Private benchmark generator.",
                    "expected_signature": "Private benchmark generator.",
                },
                oracle_parameters={
                    "scale": 1.30,
                    "curvature": 1.50,
                    "linear_offset": -0.90,
                },
                noise_std=0.010,
                minimum_output=0.0,
            ),
            public_semantics={
                "response": "normalized deviatoric stress",
                "variables": {"x1": "normalized axial strain"},
                "baseline_role": "initial linear stress-strain approximation",
            },
            knowledge_topics=("stress_strain", "stiffness_evolution", "nonlinearity", "identifiability"),
            generator_sources=("https://trid.trb.org/View/127650",),
            mechanism_class="stress_strain",
        ),
        SemanticBenchmarkDefinition(
            task=Gate1TaskDefinition(
                task_id="SEM-PATH-01",
                task_family="path_dependent_softening",
                target="normalized_cohesion",
                variable_descriptions={"x1": "accumulated normalized plastic shear"},
                observed_ranges={"x1": (0.0, 0.75)},
                locked_ranges={"x1": (0.0, 1.60)},
                baseline_expression=_constant(1.0),
                oracle_payload={
                    "proposal_id": "oracle_sem_path_01",
                    "source": "replay",
                    "edit_type": "add_state_dependence",
                    "target": "normalized_cohesion",
                    "expression": _multiply(
                        _parameter("amplitude"),
                        _subtract(
                            _constant(1.0),
                            _unary(
                                "exp",
                                _multiply(_parameter("rate"), x1),
                            ),
                        ),
                    ),
                    "rationale": "Private benchmark generator.",
                    "expected_signature": "Private benchmark generator.",
                },
                oracle_parameters={"amplitude": -0.55, "rate": -2.30},
                noise_std=0.006,
                minimum_output=0.0,
            ),
            public_semantics={
                "response": "normalized weak-plane cohesion",
                "variables": {"x1": "accumulated normalized plastic shear"},
                "baseline_role": "constant pre-softening cohesion",
                "loading_protocol": "monotonic accumulated inelastic shear",
            },
            knowledge_topics=("path_dependence", "internal_state", "softening", "identifiability"),
            generator_sources=("https://doi.org/10.1016/j.compgeo.2022.104772",),
            mechanism_class="path_dependence",
        ),
        SemanticBenchmarkDefinition(
            task=Gate1TaskDefinition(
                task_id="SEM-RATE-01",
                task_family="rate_state_friction",
                target="friction_coefficient",
                variable_descriptions={
                    "x1": "positive normalized slip rate",
                    "x2": "positive normalized contact state",
                },
                observed_ranges={"x1": (0.70, 1.30), "x2": (0.70, 1.30)},
                locked_ranges={"x1": (0.20, 4.00), "x2": (0.20, 4.00)},
                baseline_expression=_add(
                    _constant(0.60),
                    _multiply(_constant(0.012), _unary("log", x1)),
                ),
                oracle_payload={
                    "proposal_id": "oracle_sem_rate_01",
                    "source": "replay",
                    "edit_type": "add_state_dependence",
                    "target": "friction_coefficient",
                    "expression": _multiply(
                        _parameter("state_effect"), _unary("log", x2)
                    ),
                    "rationale": "Private benchmark generator.",
                    "expected_signature": "Private benchmark generator.",
                },
                oracle_parameters={"state_effect": 0.018},
                noise_std=0.0005,
                minimum_output=0.0,
            ),
            public_semantics={
                "response": "friction coefficient",
                "variables": {
                    "x1": "slip rate normalized by a positive reference rate",
                    "x2": "contact state normalized by a positive reference state",
                },
                "baseline_role": "reference friction plus instantaneous rate effect",
            },
            knowledge_topics=("rate_dependence", "internal_state", "friction", "identifiability"),
            generator_sources=("https://doi.org/10.1029/JB088iB12p10359",),
            mechanism_class="rate_state",
        ),
        SemanticBenchmarkDefinition(
            task=Gate1TaskDefinition(
                task_id="SEM-CONFINE-01",
                task_family="confinement_strength",
                target="normalized_major_principal_strength",
                variable_descriptions={"x1": "normalized confining stress"},
                observed_ranges={"x1": (0.0, 0.70)},
                locked_ranges={"x1": (0.0, 2.00)},
                baseline_expression=_add(_constant(1.0), x1),
                oracle_payload={
                    "proposal_id": "oracle_sem_confine_01",
                    "source": "replay",
                    "edit_type": "add_term",
                    "target": "normalized_major_principal_strength",
                    "expression": _subtract(
                        _power(
                            _add(
                                _constant(1.0),
                                _multiply(_parameter("curvature"), x1),
                            ),
                            _parameter("exponent"),
                        ),
                        _constant(1.0),
                    ),
                    "rationale": "Private benchmark generator.",
                    "expected_signature": "Private benchmark generator.",
                },
                oracle_parameters={"curvature": 2.0, "exponent": 0.50},
                noise_std=0.008,
                minimum_output=0.0,
            ),
            public_semantics={
                "response": "normalized peak major principal stress",
                "variables": {"x1": "minor principal stress normalized by intact strength"},
                "baseline_role": "linear confinement-strength approximation",
            },
            knowledge_topics=("confinement", "strength", "nonlinearity", "identifiability"),
            generator_sources=("https://doi.org/10.1016/j.jrmge.2018.08.001",),
            mechanism_class="confinement",
        ),
    )


def build_semantic_benchmark_suite(
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[SemanticBenchmarkTask, ...]:
    sampling = config["sampling"]
    budget = config["candidate_budget"]
    tasks: list[SemanticBenchmarkTask] = []
    for index, definition in enumerate(semantic_benchmark_definitions()):
        task = build_gate1_task(
            definition.task,
            seed=int(seed) + 1009 * index,
            observed_count=int(sampling["observed_count"]),
            locked_count=int(sampling["locked_count"]),
            audit_count=int(sampling["audit_count"]),
            fit_fraction=float(sampling["fit_fraction"]),
            maximum_candidates=int(budget["maximum_unique_candidates"]),
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
            SemanticBenchmarkTask(
                task=replace(task, request=request),
                public_semantics=dict(definition.public_semantics),
                knowledge_topics=definition.knowledge_topics,
                generator_sources=definition.generator_sources,
                mechanism_class=definition.mechanism_class,
            )
        )
    return tuple(tasks)

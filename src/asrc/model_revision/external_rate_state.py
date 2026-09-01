from __future__ import annotations

from typing import Any, Mapping

from asrc.model_revision.benchmarks import (
    Gate1Task,
    Gate1TaskDefinition,
    build_gate1_task,
)
from asrc.model_revision.proposals import TypedRepair, validate_typed_repair


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


def _power(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {"op": "power", "left": left, "right": right}


def _log(argument: dict[str, Any]) -> dict[str, Any]:
    return {"op": "log", "argument": argument}


def _tanh(argument: dict[str, Any]) -> dict[str, Any]:
    return {"op": "tanh", "argument": argument}


def _proposal(
    proposal_id: str,
    *,
    edit_type: str,
    target: str,
    expression: dict[str, Any],
    source: str,
    rationale: str,
) -> dict[str, Any]:
    return {
        "proposal_id": proposal_id,
        "source": source,
        "edit_type": edit_type,
        "target": target,
        "expression": expression,
        "rationale": rationale,
        "expected_signature": "Competes under the frozen external-mechanism protocol.",
    }


def rate_state_task_definitions(
    config: Mapping[str, Any],
) -> tuple[Gate1TaskDefinition, ...]:
    evolution = config["mechanisms"]["state_evolution"]
    friction = config["mechanisms"]["friction_surface"]
    z = _variable("state_ratio")
    velocity = _variable("velocity_ratio")
    state = _variable("state_ratio")

    aging = _subtract(_constant(1.0), z)
    slip = _multiply(
        _constant(-1.0),
        _parameter("evolution_scale"),
        z,
        _log(z),
    )
    baseline_friction = _add(
        _constant(float(friction["reference_friction"])),
        _multiply(
            _constant(float(friction["direct_effect"])),
            _log(velocity),
        ),
    )
    state_correction = _multiply(_parameter("state_effect"), _log(state))
    return (
        Gate1TaskDefinition(
            task_id="RSF-EVOL",
            task_family="rate_state_friction",
            target="state_evolution_rate",
            variable_descriptions={
                "state_ratio": "dimensionless V*theta/Dc ratio",
            },
            observed_ranges={
                "state_ratio": tuple(float(value) for value in evolution["observed_range"]),
            },
            locked_ranges={
                "state_ratio": tuple(float(value) for value in evolution["locked_range"]),
            },
            baseline_expression=aging,
            oracle_payload=_proposal(
                "oracle_rsf_slip_law",
                edit_type="replace_subtree",
                target="state_evolution_rate",
                expression=slip,
                source="replay",
                rationale="Internal reference: normalized Ruina slip evolution law.",
            ),
            oracle_parameters={"evolution_scale": float(evolution["evolution_scale"])},
            noise_std=float(evolution["noise_std"]),
            constraints=(
                "state evolution vanishes at steady state ratio one",
                "state evolution restores perturbations toward steady state",
            ),
        ),
        Gate1TaskDefinition(
            task_id="RSF-FRIC",
            task_family="rate_state_friction",
            target="friction_coefficient",
            variable_descriptions={
                "velocity_ratio": "dimensionless V/V0 ratio",
                "state_ratio": "dimensionless V0*theta/Dc ratio",
            },
            observed_ranges={
                "velocity_ratio": tuple(
                    float(value) for value in friction["observed_velocity_range"]
                ),
                "state_ratio": tuple(
                    float(value) for value in friction["observed_state_range"]
                ),
            },
            locked_ranges={
                "velocity_ratio": tuple(
                    float(value) for value in friction["locked_velocity_range"]
                ),
                "state_ratio": tuple(
                    float(value) for value in friction["locked_state_range"]
                ),
            },
            baseline_expression=baseline_friction,
            oracle_payload=_proposal(
                "oracle_rsf_state_log",
                edit_type="add_state_dependence",
                target="friction_coefficient",
                expression=state_correction,
                source="replay",
                rationale="Internal reference: logarithmic rate-and-state friction term.",
            ),
            oracle_parameters={"state_effect": float(friction["state_effect"])},
            noise_std=float(friction["noise_std"]),
            constraints=(
                "friction coefficient remains positive",
                "state correction vanishes at the reference state ratio",
            ),
            minimum_output=0.0,
        ),
    )


def _state_evolution_candidates(target: str) -> tuple[dict[str, Any], ...]:
    z = _variable("state_ratio")
    one_minus_z = _subtract(_constant(1.0), z)
    return (
        _proposal(
            "rsf_evol_slip",
            edit_type="replace_subtree",
            target=target,
            expression=_multiply(
                _constant(-1.0), _parameter("p0"), z, _log(z)
            ),
            source="replay",
            rationale="Ruina slip-law family.",
        ),
        _proposal(
            "rsf_evol_scaled_aging",
            edit_type="replace_subtree",
            target=target,
            expression=_multiply(_parameter("p0"), one_minus_z),
            source="replay",
            rationale="Scaled Dieterich aging-law family.",
        ),
        _proposal(
            "rsf_evol_power",
            edit_type="replace_subtree",
            target=target,
            expression=_multiply(
                _parameter("p0"),
                _subtract(_constant(1.0), _power(z, _parameter("p1"))),
            ),
            source="grammar",
            rationale="Generalized power relaxation around the steady state.",
        ),
        _proposal(
            "rsf_evol_log",
            edit_type="replace_subtree",
            target=target,
            expression=_multiply(_constant(-1.0), _parameter("p0"), _log(z)),
            source="grammar",
            rationale="Logarithmic relaxation competitor.",
        ),
        _proposal(
            "rsf_evol_bounded",
            edit_type="replace_subtree",
            target=target,
            expression=_multiply(
                _parameter("p0"),
                _tanh(_multiply(_parameter("p1"), one_minus_z)),
            ),
            source="grammar",
            rationale="Bounded smooth relaxation competitor.",
        ),
    )


def _friction_candidates(target: str) -> tuple[dict[str, Any], ...]:
    velocity = _variable("velocity_ratio")
    state = _variable("state_ratio")
    log_velocity = _log(velocity)
    log_state = _log(state)
    return (
        _proposal(
            "rsf_fric_state_log",
            edit_type="add_state_dependence",
            target=target,
            expression=_multiply(_parameter("p0"), log_state),
            source="replay",
            rationale="Dieterich-Ruina logarithmic state term.",
        ),
        _proposal(
            "rsf_fric_state_linear",
            edit_type="add_state_dependence",
            target=target,
            expression=_multiply(
                _parameter("p0"), _subtract(state, _constant(1.0))
            ),
            source="grammar",
            rationale="Local linearization of the state effect.",
        ),
        _proposal(
            "rsf_fric_state_power",
            edit_type="add_state_dependence",
            target=target,
            expression=_multiply(
                _parameter("p0"),
                _subtract(_power(state, _parameter("p1")), _constant(1.0)),
            ),
            source="grammar",
            rationale="Generalized power state correction.",
        ),
        _proposal(
            "rsf_fric_velocity_modulated_log",
            edit_type="add_state_dependence",
            target=target,
            expression=_multiply(
                _parameter("p0"),
                log_state,
                _add(
                    _constant(1.0),
                    _multiply(_parameter("p1"), log_velocity),
                ),
            ),
            source="grammar",
            rationale="Velocity-modulated logarithmic state correction.",
        ),
        _proposal(
            "rsf_fric_state_log_quadratic",
            edit_type="add_state_dependence",
            target=target,
            expression=_add(
                _multiply(_parameter("p0"), log_state),
                _multiply(_parameter("p1"), _power(log_state, _constant(2.0))),
            ),
            source="grammar",
            rationale="Second-order logarithmic state correction.",
        ),
    )


def build_rate_state_suite(
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[Gate1Task, ...]:
    sampling = config["sampling"]
    tasks = []
    for index, definition in enumerate(rate_state_task_definitions(config)):
        tasks.append(
            build_gate1_task(
                definition,
                seed=int(seed) + 2027 * index,
                observed_count=int(sampling["observed_count"]),
                locked_count=int(sampling["locked_count"]),
                audit_count=int(sampling["audit_count"]),
                fit_fraction=float(sampling["fit_fraction"]),
                maximum_candidates=5,
            )
        )
    return tuple(tasks)


def build_rate_state_candidates(task: Gate1Task) -> tuple[TypedRepair, ...]:
    if task.task_id == "RSF-EVOL":
        payloads = _state_evolution_candidates(task.definition.target)
    elif task.task_id == "RSF-FRIC":
        payloads = _friction_candidates(task.definition.target)
    else:
        raise ValueError(f"Unsupported external rate-state task: {task.task_id}")
    return tuple(validate_typed_repair(payload, task.contract) for payload in payloads)


def rate_state_revision_task_definitions(
    config: Mapping[str, Any],
) -> tuple[Gate1TaskDefinition, ...]:
    """Build additive-revision forms for proposer attribution experiments."""

    evolution = config["mechanisms"]["state_evolution"]
    friction = config["mechanisms"]["friction_surface"]
    z = _variable("state_ratio")
    velocity = _variable("velocity_ratio")
    state = _variable("state_ratio")
    aging = _subtract(_constant(1.0), z)
    evolution_correction = _add(
        _parameter("evolution_offset"),
        _multiply(_parameter("aging_reversal"), z),
        _multiply(_parameter("slip_scale"), z, _log(z)),
    )
    baseline_friction = _add(
        _constant(float(friction["reference_friction"])),
        _multiply(
            _constant(float(friction["direct_effect"])),
            _log(velocity),
        ),
    )
    state_correction = _multiply(_parameter("state_effect"), _log(state))
    return (
        Gate1TaskDefinition(
            task_id="G2B-01",
            task_family="external_rate_state_revision",
            target="state_evolution_rate",
            variable_descriptions={
                "state_ratio": "dimensionless V*theta/Dc ratio",
            },
            observed_ranges={
                "state_ratio": tuple(
                    float(value) for value in evolution["observed_range"]
                ),
            },
            locked_ranges={
                "state_ratio": tuple(
                    float(value) for value in evolution["locked_range"]
                ),
            },
            baseline_expression=aging,
            oracle_payload=_proposal(
                "oracle_g2b_evolution_revision",
                edit_type="add_term",
                target="state_evolution_rate",
                expression=evolution_correction,
                source="replay",
                rationale="Internal equivalent additive revision only.",
            ),
            oracle_parameters={
                "evolution_offset": -1.0,
                "aging_reversal": 1.0,
                "slip_scale": -float(evolution["evolution_scale"]),
            },
            noise_std=float(evolution["noise_std"]),
            constraints=("corrected response remains finite",),
        ),
        Gate1TaskDefinition(
            task_id="G2B-02",
            task_family="external_rate_state_revision",
            target="friction_coefficient",
            variable_descriptions={
                "velocity_ratio": "dimensionless V/V0 ratio",
                "state_ratio": "dimensionless V0*theta/Dc ratio",
            },
            observed_ranges={
                "velocity_ratio": tuple(
                    float(value) for value in friction["observed_velocity_range"]
                ),
                "state_ratio": tuple(
                    float(value) for value in friction["observed_state_range"]
                ),
            },
            locked_ranges={
                "velocity_ratio": tuple(
                    float(value) for value in friction["locked_velocity_range"]
                ),
                "state_ratio": tuple(
                    float(value) for value in friction["locked_state_range"]
                ),
            },
            baseline_expression=baseline_friction,
            oracle_payload=_proposal(
                "oracle_g2b_friction_revision",
                edit_type="add_term",
                target="friction_coefficient",
                expression=state_correction,
                source="replay",
                rationale="Internal equivalent additive revision only.",
            ),
            oracle_parameters={"state_effect": float(friction["state_effect"])},
            noise_std=float(friction["noise_std"]),
            constraints=(
                "corrected response remains finite",
                "corrected response remains nonnegative",
            ),
            minimum_output=0.0,
        ),
    )


def build_rate_state_revision_suite(
    config: Mapping[str, Any],
    *,
    seed: int,
    maximum_candidates: int,
) -> tuple[Gate1Task, ...]:
    sampling = config["sampling"]
    tasks = []
    for index, definition in enumerate(rate_state_revision_task_definitions(config)):
        tasks.append(
            build_gate1_task(
                definition,
                seed=int(seed) + 2027 * index,
                observed_count=int(sampling["observed_count"]),
                locked_count=int(sampling["locked_count"]),
                audit_count=int(sampling["audit_count"]),
                fit_fraction=float(sampling["fit_fraction"]),
                maximum_candidates=int(maximum_candidates),
            )
        )
    return tuple(tasks)

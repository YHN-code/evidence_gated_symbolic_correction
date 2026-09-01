from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from asrc.model_revision.ast import apply_repair, evaluate_expression
from asrc.model_revision.external_rate_state import (
    build_rate_state_candidates,
    build_rate_state_suite,
)
from asrc.model_revision.proposals import canonical_expression_key
from asrc.utils.io import read_yaml
from experiments.run_symbolic_model_revision_gate2a_rate_state import (
    _validate_protocol_config,
)


def _config():
    return read_yaml("configs/symbolic_model_revision_gate2a_rate_state.yaml")


def test_gate2a_protocol_matches_frozen_information_design() -> None:
    config = _config()
    _validate_protocol_config(config)
    changed = deepcopy(config)
    changed["joint_information"]["quadrature_order"] = 16
    with pytest.raises(ValueError, match="joint_information"):
        _validate_protocol_config(changed)


def test_rate_state_external_tasks_remain_outside_gate1_ids() -> None:
    tasks = build_rate_state_suite(_config(), seed=101)
    assert {task.task_id for task in tasks} == {"RSF-EVOL", "RSF-FRIC"}
    for task in tasks:
        for name in task.variables:
            assert np.all(task.observed[name].to_numpy(float) > 0.0)
            assert np.all(task.locked[name].to_numpy(float) > 0.0)


def test_aging_and_slip_laws_share_the_steady_state() -> None:
    task = {task.task_id: task for task in build_rate_state_suite(_config(), seed=103)}[
        "RSF-EVOL"
    ]
    variables = {"state_ratio": np.asarray([1.0 - 1.0e-5, 1.0, 1.0 + 1.0e-5])}
    aging = evaluate_expression(task.baseline_expression, variables)
    slip = apply_repair(
        task.oracle_repair,
        aging,
        variables,
        task.definition.oracle_parameters,
    )
    assert aging[1] == pytest.approx(0.0, abs=1e-14)
    assert slip[1] == pytest.approx(0.0, abs=1e-14)
    aging_slope = (aging[2] - aging[0]) / 2.0e-5
    slip_slope = (slip[2] - slip[0]) / 2.0e-5
    assert aging_slope == pytest.approx(-1.0, rel=1e-6)
    assert slip_slope == pytest.approx(-1.0, rel=1e-6)


def test_each_rate_state_committee_contains_one_exact_reference() -> None:
    for task in build_rate_state_suite(_config(), seed=107):
        candidates = build_rate_state_candidates(task)
        oracle_key = canonical_expression_key(task.oracle_repair.expression)
        exact_count = sum(
            canonical_expression_key(candidate.expression) == oracle_key
            for candidate in candidates
        )
        assert len(candidates) == 5
        assert exact_count == 1


def test_candidate_structures_do_not_depend_on_response_seed() -> None:
    first = build_rate_state_suite(_config(), seed=109)
    second = build_rate_state_suite(_config(), seed=211)
    for first_task, second_task in zip(first, second):
        first_keys = [
            canonical_expression_key(item.expression)
            for item in build_rate_state_candidates(first_task)
        ]
        second_keys = [
            canonical_expression_key(item.expression)
            for item in build_rate_state_candidates(second_task)
        ]
        assert first_keys == second_keys

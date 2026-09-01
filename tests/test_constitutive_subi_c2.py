from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from asrc.agent.constitutive_tools import execute_c2_constitutive_tool
from asrc.constitutive.softening import (
    generate_subi_c2_dataset,
    reference_evolution,
    softening_parameters,
)
from asrc.constitutive.softening_search import (
    audit_evolution_model,
    run_evolution_search,
)
from asrc.utils.io import read_yaml
from experiments.run_constitutive_subi_c2 import run


ROOT = Path(__file__).resolve().parents[1]


def test_reference_softening_is_monotonic_and_bounded() -> None:
    config = read_yaml(ROOT / "configs" / "constitutive_subi_c2.yaml")
    model = reference_evolution(config)
    kappa = np.linspace(0.0, 0.003, 100)
    cohesion = np.asarray(model.cohesion(kappa))
    assert np.all(np.diff(cohesion) <= 0.0)
    assert np.isclose(cohesion[0], 1.5)
    assert cohesion[-1] >= 0.35


def test_c2_dataset_contains_path_state_and_locked_partitions() -> None:
    config = read_yaml(ROOT / "configs" / "constitutive_subi_c2.yaml")
    frame = generate_subi_c2_dataset(config)
    assert set(frame["partition"]) == {
        "calibration",
        "locked_angle",
        "locked_pressure",
        "locked_path",
        "locked_joint",
    }
    assert frame["reference_kappa_after"].max() > 0.0
    assert frame["reference_cohesion_before_mpa"].min() < 1.0


def test_c2_search_recovers_exponential_evolution() -> None:
    config = read_yaml(ROOT / "configs" / "constitutive_subi_c2.yaml")
    frame = generate_subi_c2_dataset(config)
    selected, candidates = run_evolution_search(frame, config)
    assert selected.model.family == "exponential"
    assert selected.audit["violation_count"] == 0
    assert len(candidates) == len(config["search"]["bounded_families"])
    assert np.isclose(selected.model.peak_cohesion_mpa, 1.5, rtol=0.05)
    assert np.isclose(selected.model.residual_cohesion_mpa, 0.35, rtol=0.1)
    assert np.isclose(selected.model.softening_scale, 0.00035, rtol=0.15)


def test_c2_agent_tools_return_auditable_evidence() -> None:
    config = read_yaml(ROOT / "configs" / "constitutive_subi_c2.yaml")
    frame = generate_subi_c2_dataset(config)
    result = execute_c2_constitutive_tool(
        "inspect_softening_state",
        dataset=frame,
    )
    assert result.action == "inspect_softening_state"
    assert result.evidence["maximum_reference_kappa"] > 0.0
    audit = execute_c2_constitutive_tool(
        "challenge_monotonicity",
        dataset=frame,
        candidate=reference_evolution(config),
    )
    assert audit.evidence["violation_count"] == 0


def test_c2_experiment_writes_passing_gate() -> None:
    run_id = "pytest_constitutive_subi_c2"
    fallback = ROOT / "outputs" / "runs" / run_id
    shutil.rmtree(fallback, ignore_errors=True)
    run_dir = run(
        str(ROOT / "configs" / "constitutive_subi_c2.yaml"),
        run_id,
        force=True,
    )
    report_path = run_dir / "reports" / "constitutive_subi_c2_gate.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert (
        report["locked_joint_rmse_mpa"]
        < report["baseline_locked_joint_rmse_mpa"]
    )

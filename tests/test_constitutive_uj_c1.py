from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asrc.constitutive.weak_plane import (
    WeakPlaneParameters,
    WeakPlaneState,
    generate_uj_c1_dataset,
    resolved_traction,
    update_weak_plane,
)
from asrc.constitutive.weak_plane_search import (
    fit_strength_model,
    replay_dataset,
    run_strength_family_search,
)
from asrc.utils.io import read_yaml
from experiments.run_constitutive_uj_c1 import run


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def c1_case() -> tuple[dict, pd.DataFrame]:
    config = read_yaml(ROOT / "configs" / "constitutive_uj_c1.yaml")
    return config, generate_uj_c1_dataset(config)


def test_joint_return_mapping_obeys_mohr_coulomb_strength() -> None:
    parameters = WeakPlaneParameters(20000.0, 0.24, 1.2, 28.0)
    state = WeakPlaneState(stress_mpa=-5.0 * np.eye(3))
    increment = np.zeros((3, 3))
    increment[0, 2] = increment[2, 0] = 0.002
    state = update_weak_plane(state, increment, 0.0, parameters)
    sigma_n, _, tau = resolved_traction(state.stress_mpa, 0.0)
    expected = parameters.cohesion_mpa - sigma_n * np.tan(
        np.deg2rad(parameters.friction_deg)
    )
    assert state.joint_shear_now
    assert np.isclose(tau, expected, atol=1.0e-9)
    assert state.accumulated_plastic_shear > 0.0
    assert state.plastic_dissipation_mpa >= 0.0


def test_c1_dataset_has_all_locked_partitions_and_path_history(
    c1_case: tuple[dict, pd.DataFrame],
) -> None:
    _, frame = c1_case
    assert set(frame["partition"]) == {
        "calibration",
        "locked_angle",
        "locked_pressure",
        "locked_path",
        "locked_joint",
    }
    assert {"reverse_shear", "cyclic_shear"}.issubset(set(frame["path"]))
    assert frame["reference_joint_shear_past"].max() == 1


def test_search_recovers_normal_stress_family(c1_case: tuple[dict, pd.DataFrame]) -> None:
    config, frame = c1_case
    selected, candidates = run_strength_family_search(frame, config)
    assert selected.model.family == "normal_linear"
    assert selected.audit["violation_count"] == 0
    assert len(candidates) == len(config["search"]["bounded_families"])
    assert np.isclose(selected.model.coefficients[0], 1.2, rtol=0.02)
    assert np.isclose(selected.model.coefficients[1], np.tan(np.deg2rad(28.0)), rtol=0.02)


def test_locked_replay_is_not_fit_on_locked_data(c1_case: tuple[dict, pd.DataFrame]) -> None:
    _, frame = c1_case
    calibration = frame.loc[frame["partition"] == "calibration"]
    locked = frame.loc[frame["partition"] == "locked_joint"]
    model = fit_strength_model(calibration, "normal_linear")
    predictions = replay_dataset(
        locked,
        model,
        WeakPlaneParameters(20000.0, 0.24, 1.2, 28.0),
    )
    assert len(predictions) == len(locked)
    assert predictions["predicted_plastic_dissipation_mpa"].min() >= 0.0


def test_c1_experiment_writes_gate_and_resumes() -> None:
    run_id = "pytest_constitutive_uj_c1"
    run_dir = ROOT / "outputs" / "runs" / run_id
    shutil.rmtree(run_dir, ignore_errors=True)
    actual_run_dir = run(
        str(ROOT / "configs" / "constitutive_uj_c1.yaml"),
        run_id,
        force=True,
        verbose=False,
    )
    report_path = actual_run_dir / "reports" / "constitutive_uj_c1_gate.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    modified = report_path.stat().st_mtime_ns
    run(
        str(ROOT / "configs" / "constitutive_uj_c1.yaml"),
        run_id,
        resume=True,
        verbose=False,
    )
    assert report_path.stat().st_mtime_ns == modified

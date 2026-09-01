from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from asrc.constitutive.flac3d_uj_validation import (
    FLAC3D_JOINT_SHEAR_NOW,
    ValidationCase,
    compare_flac3d_case,
    render_flac3d_data_file,
    validation_parameters,
)
from asrc.constitutive.weak_plane import (
    STRESS_COMPONENTS,
    WeakPlaneState,
    update_weak_plane,
)
from asrc.utils.io import read_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_rendered_flac3d_validation_uses_actual_strain_and_explicit_normal() -> None:
    config = read_yaml(ROOT / "configs" / "flac3d_uj_single_zone_validation.yaml")
    text = render_flac3d_data_file(config)
    assert "zone cmodel assign ubiquitous-joint" in text
    assert "zone.strain.inc(current_zone)" in text
    assert "zone.state(current_zone,1)" in text
    assert "zone property normal (" in text
    assert text.count("ASRC_CASE_COMPLETE") == len(config["cases"])


def test_python_replay_matches_synthetic_solver_trace() -> None:
    config = read_yaml(ROOT / "configs" / "flac3d_uj_single_zone_validation.yaml")
    parameters = validation_parameters(config)
    case = ValidationCase("synthetic", 30.0, 5.0, "monotonic_shear")
    state = WeakPlaneState(stress_mpa=-5.0 * np.eye(3))
    rows = []
    accumulated = np.zeros((3, 3))
    for step in range(1, 13):
        increment = np.zeros((3, 3))
        increment[0, 2] = increment[2, 0] = 0.00005
        accumulated += increment
        state = update_weak_plane(state, increment, case.beta_deg, parameters)
        row = {
            "step": step,
            "strain_inc_xx": accumulated[0, 0],
            "strain_inc_yy": accumulated[1, 1],
            "strain_inc_zz": accumulated[2, 2],
            "strain_inc_xy": accumulated[0, 1],
            "strain_inc_xz": accumulated[0, 2],
            "strain_inc_yz": accumulated[1, 2],
            "state_bits": FLAC3D_JOINT_SHEAR_NOW if state.joint_shear_now else 0,
        }
        indices = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
        for name, index in zip(STRESS_COMPONENTS, indices):
            row[f"stress_{name}_pa"] = state.stress_mpa[index] * 1.0e6
        rows.append(row)
    comparison, metrics = compare_flac3d_case(pd.DataFrame(rows), case, parameters)
    assert len(comparison) == 12
    assert metrics["stress_rmse_mpa"] < 1.0e-12
    assert metrics["yield_onset_error_steps"] == 0

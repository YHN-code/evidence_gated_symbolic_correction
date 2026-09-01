from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from asrc.constitutive.flac3d_subi_validation import (
    SUBIValidationCase,
    compare_subi_case,
    render_flac3d_subi_data_file,
    subi_parameters,
    subi_table,
)
from asrc.constitutive.weak_plane import (
    WeakPlaneState,
    update_weak_plane,
)
from asrc.utils.io import read_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_subi_data_file_uses_softening_model_and_joint_table() -> None:
    config = read_yaml(
        ROOT / "configs" / "flac3d_subi_single_zone_validation.yaml"
    )
    text = render_flac3d_subi_data_file(config)
    assert "zone cmodel assign softening-ubiquitous" in text
    assert "table-joint-cohesion 'asrc_joint_cohesion'" in text
    assert "strain-shear-plastic-joint" in text
    assert text.count("ASRC_CASE_COMPLETE") == len(config["cases"])


def test_subi_python_replay_matches_synthetic_tabulated_trace() -> None:
    config = read_yaml(
        ROOT / "configs" / "flac3d_subi_single_zone_validation.yaml"
    )
    parameters = subi_parameters(config)
    kappa_table, cohesion_table = subi_table(config)
    case = SUBIValidationCase("synthetic_subi", 0.0, 3.0, "monotonic")
    state = WeakPlaneState(stress_mpa=-3.0 * np.eye(3))
    accumulated = np.zeros((3, 3))
    rows = []
    for step in range(1, 31):
        increment = np.zeros((3, 3))
        increment[0, 2] = increment[2, 0] = 0.00005
        accumulated += increment
        state = update_weak_plane(
            state,
            increment,
            0.0,
            parameters,
            strength_law=lambda sn, _pm, _beta, current: float(
                np.interp(
                    current.accumulated_plastic_shear,
                    kappa_table,
                    cohesion_table,
                )
                - sn * np.tan(np.deg2rad(parameters.friction_deg))
            ),
        )
        rows.append(
            {
                "step": step,
                "strain_inc_xx": accumulated[0, 0],
                "strain_inc_yy": accumulated[1, 1],
                "strain_inc_zz": accumulated[2, 2],
                "strain_inc_xy": accumulated[0, 1],
                "strain_inc_xz": accumulated[0, 2],
                "strain_inc_yz": accumulated[1, 2],
                "stress_xx_pa": state.stress_mpa[0, 0] * 1.0e6,
                "stress_yy_pa": state.stress_mpa[1, 1] * 1.0e6,
                "stress_zz_pa": state.stress_mpa[2, 2] * 1.0e6,
                "stress_xy_pa": state.stress_mpa[0, 1] * 1.0e6,
                "stress_xz_pa": state.stress_mpa[0, 2] * 1.0e6,
                "stress_yz_pa": state.stress_mpa[1, 2] * 1.0e6,
                "state_bits": 16 if state.joint_shear_now else 0,
                "plastic_shear_joint": state.accumulated_plastic_shear,
                "joint_cohesion_pa": np.interp(
                    state.accumulated_plastic_shear,
                    kappa_table,
                    cohesion_table,
                )
                * 1.0e6,
            }
        )
    _, metrics = compare_subi_case(
        pd.DataFrame(rows),
        case,
        parameters,
        kappa_table,
        cohesion_table,
    )
    assert metrics["stress_rmse_mpa"] < 1.0e-12
    assert metrics["maximum_plastic_shear_error"] < 1.0e-14

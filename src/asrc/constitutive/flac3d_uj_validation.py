from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asrc.constitutive.weak_plane import (
    STRESS_COMPONENTS,
    WeakPlaneParameters,
    WeakPlaneState,
    resolved_traction,
    update_weak_plane,
    weak_plane_normal,
)


FLAC3D_JOINT_SHEAR_NOW = 0x010
FLAC3D_JOINT_SHEAR_PAST = 0x040


@dataclass(frozen=True)
class ValidationCase:
    case_id: str
    beta_deg: float
    pressure_mpa: float
    path: str


def elastic_moduli_pa(young_mpa: float, poisson: float) -> tuple[float, float]:
    young_pa = float(young_mpa) * 1.0e6
    bulk_pa = young_pa / (3.0 * (1.0 - 2.0 * float(poisson)))
    shear_pa = young_pa / (2.0 * (1.0 + float(poisson)))
    return bulk_pa, shear_pa


def validation_parameters(config: dict[str, Any]) -> WeakPlaneParameters:
    elastic = config["material"]["elastic"]
    joint = config["material"]["reference_joint"]
    return WeakPlaneParameters(
        young_mpa=float(elastic["young_mpa"]),
        poisson=float(elastic["poisson"]),
        cohesion_mpa=float(joint["cohesion_mpa"]),
        friction_deg=float(joint["friction_deg"]),
        dilation_deg=float(joint["dilation_deg"]),
        tension_mpa=float(joint["tension_mpa"]),
    )


def validation_cases(config: dict[str, Any]) -> list[ValidationCase]:
    return [
        ValidationCase(
            case_id=str(item["case_id"]),
            beta_deg=float(item["beta_deg"]),
            pressure_mpa=float(item["pressure_mpa"]),
            path=str(item["path"]),
        )
        for item in config["cases"]
    ]


def _velocity_schedule(
    path_name: str,
    path_config: dict[str, Any],
    timestep: float,
) -> list[tuple[float, float]]:
    steps = int(path_config["steps"])
    gamma_increment = float(path_config["engineering_shear_increment"])
    signs = np.ones(steps, dtype=float)
    if path_name == "reverse_shear":
        reversal_step = int(path_config["reversal_step"])
        if not 1 <= reversal_step < steps:
            raise ValueError("reverse_shear reversal_step must lie within the path.")
        signs[reversal_step:] = -1.0
    elif path_name not in {"monotonic_shear", "compression_shear"}:
        raise ValueError(f"Unknown FLAC3D validation path: {path_name}")
    axial_ratio = float(path_config.get("axial_strain_ratio", 0.0))
    return [
        (
            float(sign * gamma_increment / timestep),
            float(-axial_ratio * gamma_increment / timestep)
            if path_name == "compression_shear"
            else 0.0,
        )
        for sign in signs
    ]


def _fish_output_functions(case: ValidationCase, row_count: int) -> list[str]:
    result_name = f"{case.case_id}_flac3d.csv"
    return [
        "fish define asrc_case_initialize",
        f"    global asrc_output = array.create({row_count + 1})",
        "    global asrc_step = 0",
        "    asrc_output(1) = 'step,strain_inc_xx,strain_inc_yy,strain_inc_zz,strain_inc_xy,strain_inc_xz,strain_inc_yz,stress_xx_pa,stress_yy_pa,stress_zz_pa,stress_xy_pa,stress_xz_pa,stress_yz_pa,state_bits'",
        "end",
        "[asrc_case_initialize]",
        "fish define asrc_case_record",
        "    asrc_step += 1",
        "    local current_zone = zone.near(0.5,0.5,0.5)",
        "    local strain_increment = zone.strain.inc(current_zone)",
        "    local stress = zone.stress(current_zone)",
        "    local state_bits = zone.state(current_zone,1)",
        "    asrc_output(asrc_step + 1) = string(asrc_step) + ',' ...",
        "        + string(comp.xx(strain_increment)) + ',' + string(comp.yy(strain_increment)) + ',' ...",
        "        + string(comp.zz(strain_increment)) + ',' + string(comp.xy(strain_increment)) + ',' ...",
        "        + string(comp.xz(strain_increment)) + ',' + string(comp.yz(strain_increment)) + ',' ...",
        "        + string(comp.xx(stress)) + ',' + string(comp.yy(stress)) + ',' ...",
        "        + string(comp.zz(stress)) + ',' + string(comp.xy(stress)) + ',' ...",
        "        + string(comp.xz(stress)) + ',' + string(comp.yz(stress)) + ',' + string(state_bits)",
        "end",
        "fish define asrc_case_finalize",
        f"    local status = file.open('{result_name}',1,1)",
        f"    status = file.write(asrc_output,{row_count + 1})",
        "    status = file.close",
        "    asrc_output = array.delete(asrc_output)",
        f"    io.out('ASRC_CASE_COMPLETE case={case.case_id} rows={row_count}')",
        "end",
    ]


def render_flac3d_data_file(config: dict[str, Any]) -> str:
    elastic = config["material"]["elastic"]
    matrix = config["material"]["intact_matrix"]
    joint = config["material"]["reference_joint"]
    flac = config["flac3d"]
    bulk_pa, shear_pa = elastic_moduli_pa(elastic["young_mpa"], elastic["poisson"])
    timestep = float(flac["dynamic_timestep"])
    lines = [
        "; Generated ASRC FLAC3D single-zone ubiquitous-joint validation.",
        "model new",
        "model title 'ASRC UJ single-zone solver-semantic validation'",
        "model large-strain off",
        "model deterministic on",
        "model configure dynamic",
        "fish automatic-create off",
        f"model dynamic timestep fix {timestep:.12g}",
        "",
    ]
    for case in validation_cases(config):
        path_config = config["paths"][case.path]
        schedule = _velocity_schedule(case.path, path_config, timestep)
        normal = weak_plane_normal(case.beta_deg)
        pressure_pa = case.pressure_mpa * 1.0e6
        lines.extend(
            [
                f"; Case {case.case_id}: beta={case.beta_deg:g}, p={case.pressure_mpa:g} MPa, path={case.path}",
                "zone delete",
                "zone create brick size 1 1 1",
                "zone cmodel assign ubiquitous-joint",
                f"zone property density {float(flac['density_kg_m3']):.12g} bulk {bulk_pa:.12g} shear {shear_pa:.12g}",
                f"zone property cohesion {float(matrix['cohesion_mpa']) * 1.0e6:.12g} friction {float(matrix['friction_deg']):.12g} dilation {float(matrix['dilation_deg']):.12g} tension {float(matrix['tension_mpa']) * 1.0e6:.12g}",
                f"zone property joint-cohesion {float(joint['cohesion_mpa']) * 1.0e6:.12g} joint-friction {float(joint['friction_deg']):.12g} joint-dilation {float(joint['dilation_deg']):.12g} joint-tension {float(joint['tension_mpa']) * 1.0e6:.12g}",
                f"zone property normal ({normal[0]:.16g},{normal[1]:.16g},{normal[2]:.16g})",
                f"zone initialize stress xx {-pressure_pa:.12g} yy {-pressure_pa:.12g} zz {-pressure_pa:.12g}",
                "zone gridpoint fix velocity",
                *_fish_output_functions(case, len(schedule)),
            ]
        )
        previous_velocity: tuple[float, float] | None = None
        for velocity_x, velocity_z in schedule:
            velocity = (velocity_x, velocity_z)
            if velocity != previous_velocity:
                lines.append(
                    f"zone gridpoint initialize velocity-x {velocity_x:.16g} range position-z 1"
                )
                lines.append(
                    f"zone gridpoint initialize velocity-z {velocity_z:.16g} range position-z 1"
                )
                previous_velocity = velocity
            lines.extend(["model cycle 1", "[asrc_case_record]"])
        lines.extend(["[asrc_case_finalize]", ""])
    lines.extend(
        [
            "fish define asrc_validation_complete",
            f"    io.out('ASRC_RESULT cases={len(config['cases'])} status=complete')",
            "end",
            "[asrc_validation_complete]",
            "program return",
            "",
        ]
    )
    return "\n".join(lines)


def load_flac3d_case(path: str | Path, expected_steps: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "step",
        *{f"strain_inc_{name}" for name in STRESS_COMPONENTS},
        *{f"stress_{name}_pa" for name in STRESS_COMPONENTS},
        "state_bits",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"FLAC3D case output is missing columns: {sorted(missing)}")
    if len(frame) != int(expected_steps):
        raise ValueError(
            f"FLAC3D case output has {len(frame)} rows; expected {expected_steps}."
        )
    expected_index = np.arange(1, expected_steps + 1)
    if not np.array_equal(frame["step"].to_numpy(int), expected_index):
        raise ValueError("FLAC3D case output step sequence is incomplete.")
    return frame


def _strain_tensor(row: Any) -> np.ndarray:
    return np.asarray(
        [
            [row.strain_inc_xx, row.strain_inc_xy, row.strain_inc_xz],
            [row.strain_inc_xy, row.strain_inc_yy, row.strain_inc_yz],
            [row.strain_inc_xz, row.strain_inc_yz, row.strain_inc_zz],
        ],
        dtype=float,
    )


def _stress_tensor_mpa(row: Any) -> np.ndarray:
    scale = 1.0e-6
    return scale * np.asarray(
        [
            [row.stress_xx_pa, row.stress_xy_pa, row.stress_xz_pa],
            [row.stress_xy_pa, row.stress_yy_pa, row.stress_yz_pa],
            [row.stress_xz_pa, row.stress_yz_pa, row.stress_zz_pa],
        ],
        dtype=float,
    )


def compare_flac3d_case(
    frame: pd.DataFrame,
    case: ValidationCase,
    parameters: WeakPlaneParameters,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    state = WeakPlaneState(stress_mpa=-case.pressure_mpa * np.eye(3))
    rows: list[dict[str, Any]] = []
    stress_errors: list[float] = []
    traction_errors: list[float] = []
    previous_accumulated_strain = np.zeros((3, 3), dtype=float)
    for source in frame.itertuples(index=False):
        accumulated_strain = _strain_tensor(source)
        increment = accumulated_strain - previous_accumulated_strain
        previous_accumulated_strain = accumulated_strain
        state = update_weak_plane(state, increment, case.beta_deg, parameters)
        flac_stress = _stress_tensor_mpa(source)
        difference = state.stress_mpa - flac_stress
        stress_errors.extend(difference[np.triu_indices(3)].tolist())
        flac_sigma_n, _, flac_tau = resolved_traction(flac_stress, case.beta_deg)
        python_sigma_n, _, python_tau = resolved_traction(state.stress_mpa, case.beta_deg)
        traction_errors.extend([python_sigma_n - flac_sigma_n, python_tau - flac_tau])
        flac_state = int(source.state_bits)
        row: dict[str, Any] = {
            "case_id": case.case_id,
            "step": int(source.step),
            "beta_deg": case.beta_deg,
            "pressure_mpa": case.pressure_mpa,
            "path": case.path,
            "flac3d_state_bits": flac_state,
            "flac3d_joint_shear_now": int(bool(flac_state & FLAC3D_JOINT_SHEAR_NOW)),
            "flac3d_joint_shear_past": int(bool(flac_state & FLAC3D_JOINT_SHEAR_PAST)),
            "python_joint_shear_now": int(state.joint_shear_now),
            "python_joint_shear_past": int(state.joint_shear_past),
            "flac3d_sigma_n_mpa": flac_sigma_n,
            "python_sigma_n_mpa": python_sigma_n,
            "flac3d_tau_mpa": flac_tau,
            "python_tau_mpa": python_tau,
            "actual_delta_eps_xz": float(increment[0, 2]),
        }
        component_indices = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
        for name, indices in zip(STRESS_COMPONENTS, component_indices):
            row[f"flac3d_sigma_{name}_mpa"] = float(flac_stress[indices])
            row[f"python_sigma_{name}_mpa"] = float(state.stress_mpa[indices])
            row[f"error_sigma_{name}_mpa"] = float(difference[indices])
        rows.append(row)
    comparison = pd.DataFrame(rows)

    def onset(column: str) -> int:
        values = comparison[column].to_numpy(bool)
        return int(np.argmax(values) + 1) if np.any(values) else 0

    flac_onset = onset("flac3d_joint_shear_now")
    python_onset = onset("python_joint_shear_now")
    stress_error_array = np.asarray(stress_errors, dtype=float)
    traction_error_array = np.asarray(traction_errors, dtype=float)
    metrics = {
        "case_id": case.case_id,
        "beta_deg": case.beta_deg,
        "pressure_mpa": case.pressure_mpa,
        "path": case.path,
        "steps": len(frame),
        "stress_rmse_mpa": float(np.sqrt(np.mean(stress_error_array**2))),
        "maximum_absolute_stress_error_mpa": float(np.max(np.abs(stress_error_array))),
        "resolved_traction_rmse_mpa": float(np.sqrt(np.mean(traction_error_array**2))),
        "flac3d_yield_onset_step": flac_onset,
        "python_yield_onset_step": python_onset,
        "yield_onset_error_steps": abs(flac_onset - python_onset),
        "flac3d_joint_yielded": bool(flac_onset),
        "python_joint_yielded": bool(python_onset),
    }
    return comparison, metrics

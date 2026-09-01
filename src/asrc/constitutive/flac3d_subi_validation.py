from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asrc.constitutive.flac3d_uj_validation import (
    FLAC3D_JOINT_SHEAR_NOW,
    elastic_moduli_pa,
)
from asrc.constitutive.weak_plane import (
    STRESS_COMPONENTS,
    WeakPlaneParameters,
    WeakPlaneState,
    resolved_traction,
    update_weak_plane,
    weak_plane_normal,
)


@dataclass(frozen=True)
class SUBIValidationCase:
    case_id: str
    beta_deg: float
    pressure_mpa: float
    path: str


def subi_validation_cases(config: dict[str, Any]) -> list[SUBIValidationCase]:
    return [
        SUBIValidationCase(
            str(item["case_id"]),
            float(item["beta_deg"]),
            float(item["pressure_mpa"]),
            str(item["path"]),
        )
        for item in config["cases"]
    ]


def subi_table(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    joint = config["material"]["joint"]
    table = config["material"]["table"]
    kappa = np.linspace(
        0.0,
        float(table["maximum_plastic_shear"]),
        int(table["point_count"]),
    )
    cohesion = float(joint["residual_cohesion_mpa"]) + (
        float(joint["peak_cohesion_mpa"])
        - float(joint["residual_cohesion_mpa"])
    ) * np.exp(-kappa / float(joint["softening_scale"]))
    return kappa, cohesion


def subi_parameters(config: dict[str, Any]) -> WeakPlaneParameters:
    elastic = config["material"]["elastic"]
    joint = config["material"]["joint"]
    return WeakPlaneParameters(
        float(elastic["young_mpa"]),
        float(elastic["poisson"]),
        float(joint["peak_cohesion_mpa"]),
        float(joint["friction_deg"]),
        float(joint["dilation_deg"]),
        float(joint["tension_mpa"]),
    )


def _velocity_schedule(
    path_config: dict[str, Any],
    timestep: float,
) -> list[float]:
    amplitude = float(path_config["engineering_shear_increment"])
    velocities: list[float] = []
    for count, sign in path_config["phases"]:
        velocities.extend(
            [float(sign) * amplitude / timestep] * int(count)
        )
    return velocities


def _fish_functions(case: SUBIValidationCase, rows: int) -> list[str]:
    filename = f"{case.case_id}_flac3d.csv"
    return [
        "fish define asrc_subi_initialize",
        f"    global asrc_output = array.create({rows + 1})",
        "    global asrc_step = 0",
        "    asrc_output(1) = 'step,strain_inc_xx,strain_inc_yy,strain_inc_zz,strain_inc_xy,strain_inc_xz,strain_inc_yz,stress_xx_pa,stress_yy_pa,stress_zz_pa,stress_xy_pa,stress_xz_pa,stress_yz_pa,state_bits,plastic_shear_joint,joint_cohesion_pa'",
        "end",
        "[asrc_subi_initialize]",
        "fish define asrc_subi_record",
        "    asrc_step += 1",
        "    local current_zone = zone.near(0.5,0.5,0.5)",
        "    local strain_increment = zone.strain.inc(current_zone)",
        "    local stress = zone.stress(current_zone)",
        "    local state_bits = zone.state(current_zone,1)",
        "    local plastic_shear = zone.prop(current_zone,'strain-shear-plastic-joint')",
        "    local joint_cohesion = zone.prop(current_zone,'joint-cohesion')",
        "    asrc_output(asrc_step + 1) = string(asrc_step) + ',' ...",
        "        + string(comp.xx(strain_increment)) + ',' + string(comp.yy(strain_increment)) + ',' ...",
        "        + string(comp.zz(strain_increment)) + ',' + string(comp.xy(strain_increment)) + ',' ...",
        "        + string(comp.xz(strain_increment)) + ',' + string(comp.yz(strain_increment)) + ',' ...",
        "        + string(comp.xx(stress)) + ',' + string(comp.yy(stress)) + ',' ...",
        "        + string(comp.zz(stress)) + ',' + string(comp.xy(stress)) + ',' ...",
        "        + string(comp.xz(stress)) + ',' + string(comp.yz(stress)) + ',' ...",
        "        + string(state_bits) + ',' + string(plastic_shear) + ',' + string(joint_cohesion)",
        "end",
        "fish define asrc_subi_finalize",
        f"    local status = file.open('{filename}',1,1)",
        f"    status = file.write(asrc_output,{rows + 1})",
        "    status = file.close",
        "    asrc_output = array.delete(asrc_output)",
        f"    io.out('ASRC_CASE_COMPLETE case={case.case_id} rows={rows}')",
        "end",
    ]


def render_flac3d_subi_data_file(config: dict[str, Any]) -> str:
    elastic = config["material"]["elastic"]
    matrix = config["material"]["intact_matrix"]
    joint = config["material"]["joint"]
    flac = config["flac3d"]
    timestep = float(flac["dynamic_timestep"])
    bulk_pa, shear_pa = elastic_moduli_pa(elastic["young_mpa"], elastic["poisson"])
    kappa_table, cohesion_table = subi_table(config)
    lines = [
        "; Generated ASRC FLAC3D SUBI single-zone validation.",
        "model new",
        "model title 'ASRC SUBI single-zone validation'",
        "model large-strain off",
        "model deterministic on",
        "model configure dynamic",
        "fish automatic-create off",
        f"model dynamic timestep fix {timestep:.12g}",
        "table 'asrc_joint_cohesion' delete",
    ]
    for kappa, cohesion in zip(kappa_table, cohesion_table):
        lines.append(
            f"table 'asrc_joint_cohesion' add ({kappa:.16g},{cohesion * 1.0e6:.16g})"
        )
    lines.append("")
    for case in subi_validation_cases(config):
        schedule = _velocity_schedule(config["paths"][case.path], timestep)
        normal = weak_plane_normal(case.beta_deg)
        pressure_pa = case.pressure_mpa * 1.0e6
        lines.extend(
            [
                f"; Case {case.case_id}",
                "zone delete",
                "zone create brick size 1 1 1",
                "zone cmodel assign softening-ubiquitous",
                f"zone property density {float(flac['density_kg_m3']):.12g} bulk {bulk_pa:.12g} shear {shear_pa:.12g}",
                f"zone property cohesion {float(matrix['cohesion_mpa']) * 1.0e6:.12g} friction {float(matrix['friction_deg']):.12g} dilation {float(matrix['dilation_deg']):.12g} tension {float(matrix['tension_mpa']) * 1.0e6:.12g}",
                f"zone property joint-cohesion {float(joint['peak_cohesion_mpa']) * 1.0e6:.12g} joint-friction {float(joint['friction_deg']):.12g} joint-dilation {float(joint['dilation_deg']):.12g} joint-tension {float(joint['tension_mpa']) * 1.0e6:.12g}",
                "zone property table-joint-cohesion 'asrc_joint_cohesion'",
                f"zone property normal ({normal[0]:.16g},{normal[1]:.16g},{normal[2]:.16g})",
                f"zone initialize stress xx {-pressure_pa:.12g} yy {-pressure_pa:.12g} zz {-pressure_pa:.12g}",
                "zone gridpoint fix velocity",
                *_fish_functions(case, len(schedule)),
            ]
        )
        previous: float | None = None
        for velocity in schedule:
            if velocity != previous:
                lines.append(
                    f"zone gridpoint initialize velocity-x {velocity:.16g} range position-z 1"
                )
                lines.append(
                    "zone gridpoint initialize velocity-z 0 range position-z 1"
                )
                previous = velocity
            lines.extend(["model cycle 1", "[asrc_subi_record]"])
        lines.extend(["[asrc_subi_finalize]", ""])
    lines.extend(
        [
            "fish define asrc_subi_complete",
            f"    io.out('ASRC_RESULT cases={len(config['cases'])} status=complete')",
            "end",
            "[asrc_subi_complete]",
            "program return",
            "",
        ]
    )
    return "\n".join(lines)


def load_subi_case(path: str | Path, expected_steps: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if len(frame) != expected_steps:
        raise ValueError(
            f"FLAC3D SUBI output has {len(frame)} rows; expected {expected_steps}."
        )
    return frame


def _tensor_from_row(
    row: Any,
    prefix: str,
    scale: float = 1.0,
    suffix: str = "",
) -> np.ndarray:
    def value(component: str) -> float:
        return float(getattr(row, f"{prefix}{component}{suffix}"))

    return scale * np.asarray(
        [
            [value("xx"), value("xy"), value("xz")],
            [value("xy"), value("yy"), value("yz")],
            [value("xz"), value("yz"), value("zz")],
        ],
        dtype=float,
    )


def compare_subi_case(
    frame: pd.DataFrame,
    case: SUBIValidationCase,
    parameters: WeakPlaneParameters,
    kappa_table: np.ndarray,
    cohesion_table: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    state = WeakPlaneState(stress_mpa=-case.pressure_mpa * np.eye(3))
    previous_strain = np.zeros((3, 3))
    rows: list[dict[str, Any]] = []
    stress_errors: list[float] = []
    for source in frame.itertuples(index=False):
        accumulated_strain = _tensor_from_row(source, "strain_inc_")
        increment = accumulated_strain - previous_strain
        previous_strain = accumulated_strain
        cohesion_before = float(
            np.interp(
                state.accumulated_plastic_shear,
                kappa_table,
                cohesion_table,
            )
        )
        state = update_weak_plane(
            state,
            increment,
            case.beta_deg,
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
        flac_stress = _tensor_from_row(source, "stress_", 1.0e-6, "_pa")
        difference = state.stress_mpa - flac_stress
        stress_errors.extend(difference[np.triu_indices(3)].tolist())
        flac_state = int(source.state_bits)
        rows.append(
            {
                "case_id": case.case_id,
                "step": int(source.step),
                "flac3d_joint_shear_now": int(
                    bool(flac_state & FLAC3D_JOINT_SHEAR_NOW)
                ),
                "python_joint_shear_now": int(state.joint_shear_now),
                "flac3d_plastic_shear_joint": float(source.plastic_shear_joint),
                "python_plastic_shear_joint": state.accumulated_plastic_shear,
                "flac3d_joint_cohesion_mpa": float(source.joint_cohesion_pa) * 1.0e-6,
                "python_joint_cohesion_before_mpa": cohesion_before,
                "python_joint_cohesion_after_mpa": float(
                    np.interp(
                        state.accumulated_plastic_shear,
                        kappa_table,
                        cohesion_table,
                    )
                ),
                "maximum_absolute_stress_error_mpa": float(
                    np.max(np.abs(difference))
                ),
            }
        )
    comparison = pd.DataFrame(rows)

    def onset(column: str) -> int:
        values = comparison[column].to_numpy(bool)
        return int(np.argmax(values) + 1) if np.any(values) else 0

    stress_error = np.asarray(stress_errors)
    metrics = {
        "case_id": case.case_id,
        "beta_deg": case.beta_deg,
        "pressure_mpa": case.pressure_mpa,
        "path": case.path,
        "steps": len(frame),
        "stress_rmse_mpa": float(np.sqrt(np.mean(stress_error**2))),
        "maximum_absolute_stress_error_mpa": float(np.max(np.abs(stress_error))),
        "maximum_plastic_shear_error": float(
            np.max(
                np.abs(
                    comparison["flac3d_plastic_shear_joint"]
                    - comparison["python_plastic_shear_joint"]
                )
            )
        ),
        "maximum_cohesion_error_mpa": float(
            np.max(
                np.abs(
                    comparison["flac3d_joint_cohesion_mpa"]
                    - comparison["python_joint_cohesion_after_mpa"]
                )
            )
        ),
        "flac3d_yield_onset_step": onset("flac3d_joint_shear_now"),
        "python_yield_onset_step": onset("python_joint_shear_now"),
    }
    metrics["yield_onset_error_steps"] = abs(
        metrics["flac3d_yield_onset_step"] - metrics["python_yield_onset_step"]
    )
    return comparison, metrics

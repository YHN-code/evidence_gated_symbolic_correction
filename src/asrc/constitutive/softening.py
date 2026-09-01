from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from asrc.constitutive.weak_plane import (
    STRESS_COMPONENTS,
    WeakPlaneParameters,
    WeakPlaneState,
    resolved_traction,
    tensor_to_components,
    update_weak_plane,
)


@dataclass(frozen=True)
class CohesionEvolution:
    family: str
    coefficients: tuple[float, ...]
    complexity: int

    def cohesion(self, plastic_shear: float | np.ndarray) -> float | np.ndarray:
        kappa = np.maximum(np.asarray(plastic_shear, dtype=float), 0.0)
        c = self.coefficients
        if self.family == "constant":
            value = np.full_like(kappa, c[0], dtype=float)
        elif self.family == "linear_clipped":
            value = np.maximum(c[0], c[1] - c[2] * kappa)
        elif self.family == "exponential":
            value = c[0] + c[1] * np.exp(-kappa / c[2])
        elif self.family == "rational":
            value = c[0] + c[1] / (1.0 + kappa / c[2])
        elif self.family == "stretched_exponential":
            value = c[0] + c[1] * np.exp(
                -np.power(kappa / c[2], c[3])
            )
        elif self.family == "bilinear":
            value = c[0] + c[1] * np.maximum(1.0 - kappa / c[2], 0.0)
        elif self.family == "polynomial":
            value = c[0] + c[1] * kappa + c[2] * kappa**2
        else:
            raise ValueError(f"Unknown cohesion evolution family: {self.family}")
        return float(value) if np.ndim(value) == 0 else value

    def strength(
        self,
        sigma_n_mpa: float,
        _mean_compression_mpa: float,
        _beta_deg: float,
        state: WeakPlaneState,
        friction_deg: float,
    ) -> float:
        return float(
            self.cohesion(state.accumulated_plastic_shear)
            - sigma_n_mpa * np.tan(np.deg2rad(friction_deg))
        )

    @property
    def peak_cohesion_mpa(self) -> float:
        return float(self.cohesion(0.0))

    @property
    def residual_cohesion_mpa(self) -> float:
        if self.family in {
            "linear_clipped",
            "exponential",
            "rational",
            "stretched_exponential",
            "bilinear",
        }:
            return float(self.coefficients[0])
        if self.family == "constant":
            return float(self.coefficients[0])
        return float("nan")

    @property
    def softening_scale(self) -> float | None:
        if self.family in {
            "exponential",
            "rational",
            "stretched_exponential",
            "bilinear",
        }:
            return float(self.coefficients[2])
        return None

    @property
    def formula(self) -> str:
        c = self.coefficients
        if self.family == "constant":
            return f"c(kappa) = {c[0]:.8g}"
        if self.family == "linear_clipped":
            return f"c(kappa) = max({c[0]:.8g}, {c[1]:.8g} - {c[2]:.8g} kappa)"
        if self.family == "exponential":
            return f"c(kappa) = {c[0]:.8g} + {c[1]:.8g} exp(-kappa/{c[2]:.8g})"
        if self.family == "rational":
            return f"c(kappa) = {c[0]:.8g} + {c[1]:.8g}/(1 + kappa/{c[2]:.8g})"
        if self.family == "stretched_exponential":
            return (
                f"c(kappa) = {c[0]:.8g} + {c[1]:.8g} "
                f"exp(-(kappa/{c[2]:.8g})^{c[3]:.8g})"
            )
        if self.family == "bilinear":
            return f"c(kappa) = {c[0]:.8g} + {c[1]:.8g} max(1 - kappa/{c[2]:.8g}, 0)"
        return f"c(kappa) = {c[0]:.8g} + {c[1]:.8g} kappa + {c[2]:.8g} kappa^2"


@dataclass(frozen=True)
class EvolutionCandidate:
    model: CohesionEvolution
    calibration_rmse_mpa: float
    group_cv_rmse_mpa: float
    score: float
    audit: dict[str, Any]


def softening_parameters(config: dict[str, Any]) -> WeakPlaneParameters:
    elastic = config["material"]["elastic"]
    joint = config["material"]["reference_joint"]
    return WeakPlaneParameters(
        young_mpa=float(elastic["young_mpa"]),
        poisson=float(elastic["poisson"]),
        cohesion_mpa=float(joint["peak_cohesion_mpa"]),
        friction_deg=float(joint["friction_deg"]),
        dilation_deg=float(joint["dilation_deg"]),
        tension_mpa=float(joint["tension_mpa"]),
    )


def reference_evolution(config: dict[str, Any]) -> Any:
    joint = config["material"]["reference_joint"]
    peak = float(joint["peak_cohesion_mpa"])
    residual = float(joint["residual_cohesion_mpa"])
    if "open_candidate" in joint:
        from asrc.constitutive.open_softening import (
            OpenCohesionEvolution,
            validate_open_candidate,
        )

        protocol = config.get("open_candidate_protocol")
        if not isinstance(protocol, dict):
            raise ValueError(
                "Open reference evolution requires open_candidate_protocol."
            )
        spec = validate_open_candidate(joint["open_candidate"], protocol)
        supplied = {
            str(name): float(value)
            for name, value in joint.get(
                "open_candidate_shape_parameters",
                {},
            ).items()
        }
        expected = {parameter.name for parameter in spec.parameters}
        if set(supplied) != expected:
            raise ValueError(
                "Open reference shape parameters do not match the candidate "
                f"specification: expected {sorted(expected)}, got "
                f"{sorted(supplied)}."
            )
        values = []
        for parameter in spec.parameters:
            value = supplied[parameter.name]
            if not parameter.lower <= value <= parameter.upper:
                raise ValueError(
                    f"Open reference parameter {parameter.name!r} is outside "
                    "its pre-registered role bounds."
                )
            values.append(value)
        return OpenCohesionEvolution(
            spec,
            (
                residual,
                peak - residual,
                float(joint["softening_scale"]),
                *values,
            ),
        )
    family = str(joint["evolution_family"])
    scale = float(joint["softening_scale"])
    if family == "constant":
        coefficients = (peak,)
        complexity = 1
    elif family == "linear_clipped":
        coefficients = (
            residual,
            peak,
            (peak - residual) / max(scale, 1.0e-12),
        )
        complexity = 3
    elif family == "stretched_exponential":
        coefficients = (
            residual,
            peak - residual,
            scale,
            float(joint.get("shape_exponent", 1.0)),
        )
        complexity = 4
    else:
        coefficients = (residual, peak - residual, scale)
        complexity = 3
    return CohesionEvolution(
        family=family,
        coefficients=coefficients,
        complexity=complexity,
    )


def _path_increments(path_config: dict[str, Any]) -> list[np.ndarray]:
    amplitude = float(path_config["engineering_shear_increment"])
    increments: list[np.ndarray] = []
    for count, sign in path_config["phases"]:
        for _ in range(int(count)):
            increment = np.zeros((3, 3), dtype=float)
            increment[0, 2] = increment[2, 0] = float(sign) * amplitude / 2.0
            increments.append(increment)
    return increments


def generate_subi_c2_dataset(config: dict[str, Any]) -> pd.DataFrame:
    parameters = softening_parameters(config)
    reference_model = reference_evolution(config)
    sampling = config["sampling"]
    partitions = [
        ("calibration", sampling["calibration_beta_deg"], sampling["calibration_pressure_mpa"], sampling["calibration_paths"]),
        ("locked_angle", sampling["locked_beta_deg"], sampling["calibration_pressure_mpa"], sampling["calibration_paths"]),
        ("locked_pressure", sampling["calibration_beta_deg"], sampling["locked_pressure_mpa"], sampling["calibration_paths"]),
        ("locked_path", sampling["calibration_beta_deg"], sampling["calibration_pressure_mpa"], sampling["locked_paths"]),
        ("locked_joint", sampling["locked_beta_deg"], sampling["locked_pressure_mpa"], sampling["locked_paths"]),
    ]
    rows: list[dict[str, Any]] = []
    for partition, beta_values, pressure_values, paths in partitions:
        for beta_deg in beta_values:
            for pressure_mpa in pressure_values:
                for path_name in paths:
                    trajectory_id = f"{partition}_b{float(beta_deg):g}_p{float(pressure_mpa):g}_{path_name}"
                    initial_stress = -float(pressure_mpa) * np.eye(3)
                    reference_state = WeakPlaneState(stress_mpa=initial_stress.copy())
                    baseline_state = WeakPlaneState(stress_mpa=initial_stress.copy())
                    total_strain = np.zeros((3, 3), dtype=float)
                    increments = _path_increments(sampling["path_definitions"][path_name])
                    for step, increment in enumerate(increments, start=1):
                        total_strain += increment
                        kappa_before = reference_state.accumulated_plastic_shear
                        cohesion_before = float(reference_model.cohesion(kappa_before))
                        reference_state = update_weak_plane(
                            reference_state,
                            increment,
                            float(beta_deg),
                            parameters,
                            strength_law=lambda sn, pm, beta, state: reference_model.strength(
                                sn, pm, beta, state, parameters.friction_deg
                            ),
                        )
                        baseline_state = update_weak_plane(
                            baseline_state,
                            increment,
                            float(beta_deg),
                            parameters,
                        )
                        sigma_n, _, tau = resolved_traction(reference_state.stress_mpa, float(beta_deg))
                        row: dict[str, Any] = {
                            "partition": partition,
                            "trajectory_id": trajectory_id,
                            "step": step,
                            "beta_deg": float(beta_deg),
                            "confining_pressure_mpa": float(pressure_mpa),
                            "path": path_name,
                            "reference_kappa_before": kappa_before,
                            "reference_kappa_after": reference_state.accumulated_plastic_shear,
                            "reference_cohesion_before_mpa": cohesion_before,
                            "reference_sigma_n_mpa": sigma_n,
                            "reference_tau_mpa": tau,
                            "reference_joint_shear_now": int(reference_state.joint_shear_now),
                            "reference_joint_tension_now": int(reference_state.joint_tension_now),
                            "reference_plastic_dissipation_mpa": reference_state.plastic_dissipation_mpa,
                        }
                        for name, value in tensor_to_components(increment, strain=True).items():
                            row[f"delta_eps_{name}"] = value
                        for name, value in tensor_to_components(total_strain, strain=True).items():
                            row[f"total_eps_{name}"] = value
                        for name, value in tensor_to_components(baseline_state.stress_mpa).items():
                            row[f"baseline_sigma_{name}_mpa"] = value
                        for name, value in tensor_to_components(reference_state.stress_mpa).items():
                            row[f"reference_sigma_{name}_mpa"] = value
                        rows.append(row)
    return pd.DataFrame(rows)

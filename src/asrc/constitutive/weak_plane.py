from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


STRESS_COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")
STRAIN_COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")


@dataclass(frozen=True)
class WeakPlaneParameters:
    young_mpa: float
    poisson: float
    cohesion_mpa: float
    friction_deg: float
    dilation_deg: float = 0.0
    tension_mpa: float = 1.0e6

    @property
    def shear_mpa(self) -> float:
        return self.young_mpa / (2.0 * (1.0 + self.poisson))


@dataclass
class WeakPlaneState:
    stress_mpa: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=float))
    accumulated_plastic_shear: float = 0.0
    plastic_normal_strain: float = 0.0
    plastic_dissipation_mpa: float = 0.0
    joint_shear_now: bool = False
    joint_shear_past: bool = False
    joint_tension_now: bool = False
    joint_tension_past: bool = False
    internal_variables: dict[str, float] = field(default_factory=dict)

    def copy(self) -> "WeakPlaneState":
        return WeakPlaneState(
            stress_mpa=np.asarray(self.stress_mpa, dtype=float).copy(),
            accumulated_plastic_shear=float(self.accumulated_plastic_shear),
            plastic_normal_strain=float(self.plastic_normal_strain),
            plastic_dissipation_mpa=float(self.plastic_dissipation_mpa),
            joint_shear_now=bool(self.joint_shear_now),
            joint_shear_past=bool(self.joint_shear_past),
            joint_tension_now=bool(self.joint_tension_now),
            joint_tension_past=bool(self.joint_tension_past),
            internal_variables=dict(self.internal_variables),
        )


StrengthLaw = Callable[[float, float, float, WeakPlaneState], float]


def validate_parameters(parameters: WeakPlaneParameters) -> None:
    if parameters.young_mpa <= 0.0:
        raise ValueError("young_mpa must be positive.")
    if not -1.0 < parameters.poisson < 0.5:
        raise ValueError("poisson must lie between -1 and 0.5.")
    if parameters.cohesion_mpa < 0.0 or parameters.tension_mpa < 0.0:
        raise ValueError("Joint cohesion and tension must be non-negative.")
    if not 0.0 <= parameters.friction_deg < 90.0:
        raise ValueError("friction_deg must lie in [0, 90).")
    if not 0.0 <= parameters.dilation_deg <= parameters.friction_deg:
        raise ValueError("dilation_deg must lie between zero and friction_deg.")


def weak_plane_normal(beta_deg: float) -> np.ndarray:
    """Return a unit normal for a plane dipping by beta in the x-z section."""
    beta = np.deg2rad(float(beta_deg))
    return np.asarray([-np.sin(beta), 0.0, np.cos(beta)], dtype=float)


def elastic_stress_increment(
    strain_increment: np.ndarray,
    young_mpa: float,
    poisson: float,
) -> np.ndarray:
    strain = np.asarray(strain_increment, dtype=float)
    if strain.shape != (3, 3):
        raise ValueError("strain_increment must be a 3 x 3 symmetric tensor.")
    shear = young_mpa / (2.0 * (1.0 + poisson))
    lame = young_mpa * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return lame * float(np.trace(strain)) * np.eye(3) + 2.0 * shear * strain


def resolved_traction(stress_mpa: np.ndarray, beta_deg: float) -> tuple[float, np.ndarray, float]:
    normal = weak_plane_normal(beta_deg)
    traction = np.asarray(stress_mpa, dtype=float) @ normal
    sigma_n = float(normal @ traction)
    shear_vector = traction - sigma_n * normal
    return sigma_n, shear_vector, float(np.linalg.norm(shear_vector))


def mohr_coulomb_joint_strength(
    sigma_n_mpa: float,
    _mean_compression_mpa: float,
    _beta_deg: float,
    _state: WeakPlaneState,
    parameters: WeakPlaneParameters,
) -> float:
    friction = np.tan(np.deg2rad(parameters.friction_deg))
    return float(parameters.cohesion_mpa - float(sigma_n_mpa) * friction)


def update_weak_plane(
    state: WeakPlaneState,
    strain_increment: np.ndarray,
    beta_deg: float,
    parameters: WeakPlaneParameters,
    *,
    enable_weak_plane: bool = True,
    strength_law: StrengthLaw | None = None,
) -> WeakPlaneState:
    """Advance a reduced 3-D perfect-plastic ubiquitous-joint material point.

    Compression is negative, matching FLAC3D. C1 uses zero dilation so the
    return mapping modifies only the weak-plane shear traction. The state
    contract already reserves normal plastic strain and internal variables for
    the path-dependent C2 extension.
    """
    validate_parameters(parameters)
    if abs(parameters.dilation_deg) > 1.0e-12:
        raise NotImplementedError("C1 supports zero joint dilation only.")
    result = state.copy()
    result.joint_shear_now = False
    result.joint_tension_now = False
    trial = result.stress_mpa + elastic_stress_increment(
        strain_increment,
        parameters.young_mpa,
        parameters.poisson,
    )
    result.stress_mpa = 0.5 * (trial + trial.T)
    if not enable_weak_plane:
        return result

    sigma_n, shear_vector, tau = resolved_traction(result.stress_mpa, beta_deg)
    mean_compression = max(-float(np.trace(result.stress_mpa)) / 3.0, 0.0)
    law = strength_law
    if law is None:
        law = lambda sn, pm, beta, current: mohr_coulomb_joint_strength(
            sn, pm, beta, current, parameters
        )
    strength = float(law(sigma_n, mean_compression, beta_deg, result))
    result.internal_variables["last_joint_strength_mpa"] = strength
    result.internal_variables["last_sigma_n_mpa"] = sigma_n
    result.internal_variables["last_tau_trial_mpa"] = tau

    friction = np.tan(np.deg2rad(parameters.friction_deg))
    apex_tension = (
        parameters.cohesion_mpa / friction
        if friction > 0.0
        else parameters.tension_mpa
    )
    effective_tension = min(parameters.tension_mpa, apex_tension)
    shear_yield = tau - strength
    tension_yield = sigma_n - effective_tension
    normalized_shear_yield = shear_yield / np.sqrt(1.0 + friction**2)
    branch_tolerance = 1.0e-10 * max(
        1.0,
        abs(normalized_shear_yield),
        abs(tension_yield),
    )

    if (
        shear_yield > 1.0e-12
        and normalized_shear_yield + branch_tolerance >= tension_yield
        and tau > 1.0e-14
    ):
        normal = weak_plane_normal(beta_deg)
        shear_direction = shear_vector / tau
        excess = shear_yield
        correction = excess * (
            np.outer(shear_direction, normal) + np.outer(normal, shear_direction)
        )
        result.stress_mpa = result.stress_mpa - correction
        plastic_increment = excess / (2.0 * parameters.shear_mpa)
        result.accumulated_plastic_shear += plastic_increment
        result.plastic_dissipation_mpa += 2.0 * max(strength, 0.0) * plastic_increment
        result.joint_shear_now = True
        result.joint_shear_past = True
    elif (
        tension_yield > 1.0e-12
        and tension_yield > normalized_shear_yield + branch_tolerance
    ):
        normal = weak_plane_normal(beta_deg)
        shear = parameters.shear_mpa
        lame = (
            parameters.young_mpa
            * parameters.poisson
            / ((1.0 + parameters.poisson) * (1.0 - 2.0 * parameters.poisson))
        )
        tangent_correction = -tension_yield * lame / (lame + 2.0 * shear)
        normal_projector = np.outer(normal, normal)
        result.stress_mpa += (
            tangent_correction * (np.eye(3) - normal_projector)
            - tension_yield * normal_projector
        )
        result.plastic_normal_strain += tension_yield / (lame + 2.0 * shear)
        result.joint_tension_now = True
        result.joint_tension_past = True

    sigma_n_after, shear_after, tau_after = resolved_traction(result.stress_mpa, beta_deg)
    if friction > 0.0 and sigma_n_after > apex_tension:
        normal = weak_plane_normal(beta_deg)
        shear_direction = (
            shear_after / tau_after if tau_after > 1.0e-14 else np.zeros(3)
        )
        shear_correction = tau_after * (
            np.outer(shear_direction, normal) + np.outer(normal, shear_direction)
        )
        shear = parameters.shear_mpa
        lame = (
            parameters.young_mpa
            * parameters.poisson
            / ((1.0 + parameters.poisson) * (1.0 - 2.0 * parameters.poisson))
        )
        normal_excess = sigma_n_after - apex_tension
        normal_projector = np.outer(normal, normal)
        result.stress_mpa -= shear_correction
        result.stress_mpa += (
            -normal_excess * lame / (lame + 2.0 * shear)
            * (np.eye(3) - normal_projector)
            - normal_excess * normal_projector
        )
        result.joint_tension_now = True
        result.joint_tension_past = True
    return result


def tensor_to_components(tensor: np.ndarray, *, strain: bool = False) -> dict[str, float]:
    value = np.asarray(tensor, dtype=float)
    shear_scale = 2.0 if strain else 1.0
    return {
        "xx": float(value[0, 0]),
        "yy": float(value[1, 1]),
        "zz": float(value[2, 2]),
        "xy": float(shear_scale * value[0, 1]),
        "xz": float(shear_scale * value[0, 2]),
        "yz": float(shear_scale * value[1, 2]),
    }


def components_to_strain(row: pd.Series | dict[str, Any], prefix: str = "delta_eps_") -> np.ndarray:
    return np.asarray(
        [
            [float(row[f"{prefix}xx"]), float(row[f"{prefix}xy"]) / 2.0, float(row[f"{prefix}xz"]) / 2.0],
            [float(row[f"{prefix}xy"]) / 2.0, float(row[f"{prefix}yy"]), float(row[f"{prefix}yz"]) / 2.0],
            [float(row[f"{prefix}xz"]) / 2.0, float(row[f"{prefix}yz"]) / 2.0, float(row[f"{prefix}zz"])],
        ],
        dtype=float,
    )


def _path_increments(path_name: str, path_config: dict[str, Any]) -> list[np.ndarray]:
    steps = int(path_config["steps"])
    amplitude = float(path_config["increment"])
    axial_ratio = float(path_config.get("axial_ratio", 0.0))
    if steps < 2 or amplitude <= 0.0:
        raise ValueError("Each loading path requires steps >= 2 and increment > 0.")

    if path_name in {"monotonic_shear", "compression_shear"}:
        signs = np.ones(steps, dtype=float)
    elif path_name == "reverse_shear":
        first = int(path_config.get("forward_steps", max(1, steps // 3)))
        if not 1 <= first < steps:
            raise ValueError("reverse_shear forward_steps must lie within the path.")
        signs = np.concatenate([np.ones(first), -np.ones(steps - first)])
    elif path_name == "cyclic_shear":
        phases = path_config.get("phase_steps")
        if phases is None:
            quarter = max(1, steps // 4)
            phases = [quarter, 2 * quarter, steps - 3 * quarter]
        phases = [int(value) for value in phases]
        if len(phases) != 3 or any(value <= 0 for value in phases) or sum(phases) != steps:
            raise ValueError("cyclic_shear phase_steps must contain three positive values summing to steps.")
        signs = np.concatenate(
            [np.ones(phases[0]), -np.ones(phases[1]), np.ones(phases[2])]
        )
    else:
        raise ValueError(f"Unknown loading path: {path_name}")

    increments: list[np.ndarray] = []
    for sign in signs:
        delta = np.zeros((3, 3), dtype=float)
        delta[0, 2] = delta[2, 0] = sign * amplitude / 2.0
        if path_name == "compression_shear":
            delta[2, 2] = -axial_ratio * amplitude
        increments.append(delta)
    return increments


def parameters_from_config(config: dict[str, Any]) -> WeakPlaneParameters:
    elastic = config["material"]["elastic"]
    joint = config["material"]["reference_joint"]
    return WeakPlaneParameters(
        young_mpa=float(elastic["young_mpa"]),
        poisson=float(elastic["poisson"]),
        cohesion_mpa=float(joint["cohesion_mpa"]),
        friction_deg=float(joint["friction_deg"]),
        dilation_deg=float(joint.get("dilation_deg", 0.0)),
        tension_mpa=float(joint.get("tension_mpa", 1.0e6)),
    )


def generate_uj_c1_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Generate calibration and four locked material-point partitions."""
    parameters = parameters_from_config(config)
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
                    initial = -float(pressure_mpa) * np.eye(3)
                    reference = WeakPlaneState(stress_mpa=initial.copy())
                    baseline = WeakPlaneState(stress_mpa=initial.copy())
                    total_strain = np.zeros((3, 3), dtype=float)
                    increments = _path_increments(path_name, sampling["path_definitions"][path_name])
                    for step, increment in enumerate(increments, start=1):
                        total_strain += increment
                        baseline = update_weak_plane(
                            baseline,
                            increment,
                            float(beta_deg),
                            parameters,
                            enable_weak_plane=False,
                        )
                        reference = update_weak_plane(
                            reference,
                            increment,
                            float(beta_deg),
                            parameters,
                            enable_weak_plane=True,
                        )
                        sigma_n, _, tau = resolved_traction(reference.stress_mpa, float(beta_deg))
                        row: dict[str, Any] = {
                            "partition": partition,
                            "trajectory_id": trajectory_id,
                            "step": step,
                            "beta_deg": float(beta_deg),
                            "confining_pressure_mpa": float(pressure_mpa),
                            "path": path_name,
                            "reference_sigma_n_mpa": sigma_n,
                            "reference_tau_mpa": tau,
                            "reference_mean_compression_mpa": max(
                                -float(np.trace(reference.stress_mpa)) / 3.0,
                                0.0,
                            ),
                            "reference_joint_shear_now": int(reference.joint_shear_now),
                            "reference_joint_shear_past": int(reference.joint_shear_past),
                            "reference_joint_tension_now": int(reference.joint_tension_now),
                            "reference_joint_tension_past": int(reference.joint_tension_past),
                            "reference_accumulated_plastic_shear": reference.accumulated_plastic_shear,
                            "reference_plastic_dissipation_mpa": reference.plastic_dissipation_mpa,
                            "baseline_mean_compression_mpa": max(
                                -float(np.trace(baseline.stress_mpa)) / 3.0,
                                0.0,
                            ),
                        }
                        for name, value in tensor_to_components(increment, strain=True).items():
                            row[f"delta_eps_{name}"] = value
                        for name, value in tensor_to_components(total_strain, strain=True).items():
                            row[f"total_eps_{name}"] = value
                        for name, value in tensor_to_components(baseline.stress_mpa).items():
                            row[f"baseline_sigma_{name}_mpa"] = value
                        for name, value in tensor_to_components(reference.stress_mpa).items():
                            row[f"reference_sigma_{name}_mpa"] = value
                        rows.append(row)
    return pd.DataFrame(rows)

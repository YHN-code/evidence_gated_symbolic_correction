from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from asrc.constitutive.weak_plane import (
    WeakPlaneParameters,
    WeakPlaneState,
    tensor_to_components,
    update_weak_plane,
)
from asrc.utils.io import read_yaml


class StructuralSofteningArtifactError(ValueError):
    """Raised when a frozen structural-revision artifact is not admissible."""


@dataclass(frozen=True)
class FrozenStructuralSoftening:
    task_id: str
    location: str
    edit_type: str
    family_id: str
    expression: str
    amplitude: float
    rate: float
    baseline_amplitude: float
    source_run_id: str
    source_checkpoint_sha256: str


@dataclass(frozen=True)
class StructuralSofteningLaw:
    law_id: str
    label: str
    family: str
    amplitude_mpa: float
    rate: float
    damage_scale_plastic_shear: float
    provenance: str

    def cohesion_from_damage(self, damage: float | np.ndarray) -> float | np.ndarray:
        values = np.maximum(np.asarray(damage, dtype=float), 0.0)
        if self.family == "linear_clipped":
            cohesion = self.amplitude_mpa * np.maximum(1.0 - values, 0.0)
        elif self.family == "exponential_decay":
            cohesion = self.amplitude_mpa * np.exp(-self.rate * values)
        else:
            raise ValueError(f"Unsupported structural softening family: {self.family}")
        return float(cohesion) if np.ndim(cohesion) == 0 else cohesion

    def cohesion_from_plastic_shear(
        self, plastic_shear: float | np.ndarray
    ) -> float | np.ndarray:
        return self.cohesion_from_damage(
            np.asarray(plastic_shear, dtype=float) / self.damage_scale_plastic_shear
        )

    def table(
        self, *, maximum_damage: float, point_count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if maximum_damage <= 0.0 or point_count < 2:
            raise ValueError("Softening-table extent and point count must be positive.")
        damage = np.linspace(0.0, float(maximum_damage), int(point_count))
        return (
            damage * self.damage_scale_plastic_shear,
            np.asarray(self.cohesion_from_damage(damage), dtype=float),
        )

    @property
    def formula(self) -> str:
        if self.family == "linear_clipped":
            return f"c(d) = {self.amplitude_mpa:.8g} max(1 - d, 0)"
        return f"c(d) = {self.amplitude_mpa:.8g} exp(-{self.rate:.8g} d)"


@dataclass(frozen=True)
class StructuralSofteningCase:
    case_id: str
    beta_deg: float
    pressure_mpa: float
    path: str


def load_frozen_structural_softening(
    path: str | Path,
) -> FrozenStructuralSoftening:
    payload = read_yaml(path)
    artifact = payload.get("artifact")
    if not isinstance(artifact, Mapping):
        raise StructuralSofteningArtifactError("Frozen artifact section is missing.")
    required_truths = {
        "result_status": "accepted",
        "adequacy_accepted": True,
        "expected_library_coverage": True,
        "location": "cohesion",
        "edit_type": "replace_component",
        "family_id": "exponential_decay",
    }
    for field, expected in required_truths.items():
        if artifact.get(field) != expected:
            raise StructuralSofteningArtifactError(
                f"Frozen artifact requires {field}={expected!r}; "
                f"received {artifact.get(field)!r}."
            )
    if int(artifact.get("physical_violations", -1)) != 0:
        raise StructuralSofteningArtifactError(
            "Frozen artifact contains physical violations."
        )
    parameters = artifact.get("selected_parameters")
    if not isinstance(parameters, Mapping) or set(parameters) != {"a0", "a1"}:
        raise StructuralSofteningArtifactError(
            "Exponential structural revision requires exactly a0 and a1."
        )
    amplitude = float(parameters["a0"])
    rate = float(parameters["a1"])
    baseline_amplitude = float(artifact["baseline_amplitude"])
    if min(amplitude, rate, baseline_amplitude) <= 0.0:
        raise StructuralSofteningArtifactError(
            "Frozen softening amplitudes and rate must be positive."
        )
    return FrozenStructuralSoftening(
        task_id=str(artifact["task_id"]),
        location=str(artifact["location"]),
        edit_type=str(artifact["edit_type"]),
        family_id=str(artifact["family_id"]),
        expression=str(artifact["selected_expression"]),
        amplitude=amplitude,
        rate=rate,
        baseline_amplitude=baseline_amplitude,
        source_run_id=str(artifact["source_run_id"]),
        source_checkpoint_sha256=str(artifact["source_checkpoint_sha256"]),
    )


def build_application_laws(
    frozen: FrozenStructuralSoftening,
    application: Mapping[str, Any],
) -> tuple[StructuralSofteningLaw, ...]:
    peak = float(application["peak_cohesion_mpa"])
    scale = float(application["damage_scale_plastic_shear"])
    oracle_rate = float(application["oracle_rate"])
    if min(peak, scale, oracle_rate) <= 0.0:
        raise ValueError("Application peak, damage scale, and oracle rate must be positive.")
    discovered_peak = peak * frozen.amplitude / frozen.baseline_amplitude
    return (
        StructuralSofteningLaw(
            law_id="baseline_linear",
            label="Linear baseline",
            family="linear_clipped",
            amplitude_mpa=peak,
            rate=1.0,
            damage_scale_plastic_shear=scale,
            provenance="declared_baseline",
        ),
        StructuralSofteningLaw(
            law_id="discovered_exponential",
            label="Discovered exponential",
            family=frozen.family_id,
            amplitude_mpa=discovered_peak,
            rate=frozen.rate,
            damage_scale_plastic_shear=scale,
            provenance=f"{frozen.source_run_id}:{frozen.task_id}",
        ),
        StructuralSofteningLaw(
            law_id="oracle_exponential",
            label="Oracle exponential",
            family="exponential_decay",
            amplitude_mpa=peak,
            rate=oracle_rate,
            damage_scale_plastic_shear=scale,
            provenance="controlled_reference_not_used_for_discovery",
        ),
    )


def structural_softening_cases(
    config: Mapping[str, Any],
) -> tuple[StructuralSofteningCase, ...]:
    return tuple(
        StructuralSofteningCase(
            case_id=str(item["case_id"]),
            beta_deg=float(item["beta_deg"]),
            pressure_mpa=float(item["pressure_mpa"]),
            path=str(item["path"]),
        )
        for item in config["cases"]
    )


def path_shear_increments(path: Mapping[str, Any]) -> tuple[float, ...]:
    amplitude = float(path["engineering_shear_increment"])
    return tuple(
        float(sign) * amplitude
        for count, sign in path["phases"]
        for _ in range(int(count))
    )


def application_parameters(config: Mapping[str, Any]) -> WeakPlaneParameters:
    elastic = config["material"]["elastic"]
    joint = config["material"]["joint"]
    return WeakPlaneParameters(
        young_mpa=float(elastic["young_mpa"]),
        poisson=float(elastic["poisson"]),
        cohesion_mpa=float(config["application"]["peak_cohesion_mpa"]),
        friction_deg=float(joint["friction_deg"]),
        dilation_deg=float(joint["dilation_deg"]),
        tension_mpa=float(joint["tension_mpa"]),
    )


def replay_material_point(
    law: StructuralSofteningLaw,
    case: StructuralSofteningCase,
    parameters: WeakPlaneParameters,
    shear_increments: Iterable[float],
) -> pd.DataFrame:
    state = WeakPlaneState(stress_mpa=-case.pressure_mpa * np.eye(3))
    accumulated_shear = 0.0
    rows: list[dict[str, Any]] = []
    for step, engineering_shear in enumerate(shear_increments, start=1):
        increment = np.zeros((3, 3), dtype=float)
        increment[0, 2] = increment[2, 0] = float(engineering_shear) / 2.0
        accumulated_shear += float(engineering_shear)
        cohesion_before = float(
            law.cohesion_from_plastic_shear(state.accumulated_plastic_shear)
        )
        state = update_weak_plane(
            state,
            increment,
            case.beta_deg,
            parameters,
            strength_law=lambda sigma_n, _mean, _beta, current: float(
                law.cohesion_from_plastic_shear(
                    current.accumulated_plastic_shear
                )
                - sigma_n * np.tan(np.deg2rad(parameters.friction_deg))
            ),
        )
        row: dict[str, Any] = {
            "law_id": law.law_id,
            "case_id": case.case_id,
            "step": step,
            "engineering_shear": accumulated_shear,
            "plastic_shear": state.accumulated_plastic_shear,
            "damage": state.accumulated_plastic_shear
            / law.damage_scale_plastic_shear,
            "cohesion_before_mpa": cohesion_before,
            "cohesion_after_mpa": float(
                law.cohesion_from_plastic_shear(
                    state.accumulated_plastic_shear
                )
            ),
            "joint_shear_now": int(state.joint_shear_now),
            "plastic_dissipation_mpa": state.plastic_dissipation_mpa,
        }
        row.update(
            {
                f"stress_{name}_mpa": value
                for name, value in tensor_to_components(state.stress_mpa).items()
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def softening_curve_frame(
    laws: Iterable[StructuralSofteningLaw], *, maximum_damage: float, point_count: int
) -> pd.DataFrame:
    damage = np.linspace(0.0, float(maximum_damage), int(point_count))
    rows = []
    for law in laws:
        for value, cohesion in zip(damage, law.cohesion_from_damage(damage)):
            rows.append(
                {
                    "law_id": law.law_id,
                    "label": law.label,
                    "damage": float(value),
                    "cohesion_mpa": float(cohesion),
                }
            )
    return pd.DataFrame(rows)

from __future__ import annotations

from dataclasses import dataclass
from math import atan, degrees
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from asrc.constitutive.weak_plane import (
    STRESS_COMPONENTS,
    WeakPlaneParameters,
    WeakPlaneState,
    components_to_strain,
    parameters_from_config,
    resolved_traction,
    update_weak_plane,
)


@dataclass(frozen=True)
class StrengthModel:
    family: str
    coefficients: tuple[float, ...]
    complexity: int

    def evaluate(
        self,
        sigma_n_mpa: float,
        mean_compression_mpa: float,
        beta_deg: float,
        _state: WeakPlaneState,
    ) -> float:
        p_n = -float(sigma_n_mpa)
        mobilization = abs(np.sin(np.deg2rad(beta_deg)) * np.cos(np.deg2rad(beta_deg)))
        c = self.coefficients
        if self.family == "constant":
            return c[0]
        if self.family == "normal_linear":
            return c[0] + c[1] * p_n
        if self.family == "normal_quadratic":
            return c[0] + c[1] * p_n + c[2] * p_n**2
        if self.family == "global_pressure_linear":
            return c[0] + c[1] * max(float(mean_compression_mpa), 0.0)
        if self.family == "orientation_linear":
            return c[0] + c[1] * mobilization
        if self.family == "generic_polynomial":
            return c[0] + c[1] * p_n + c[2] * p_n**2 + c[3] * mobilization + c[4] * p_n * mobilization
        raise ValueError(f"Unknown strength family: {self.family}")

    @property
    def formula(self) -> str:
        c = self.coefficients
        if self.family == "constant":
            return f"tau_y = {c[0]:.8g}"
        if self.family == "normal_linear":
            return f"tau_y = {c[0]:.8g} - {c[1]:.8g} sigma_n"
        if self.family == "normal_quadratic":
            return f"tau_y = {c[0]:.8g} + {c[1]:.8g} p_n + {c[2]:.8g} p_n^2"
        if self.family == "global_pressure_linear":
            return f"tau_y = {c[0]:.8g} + {c[1]:.8g} p_mean"
        if self.family == "orientation_linear":
            return f"tau_y = {c[0]:.8g} + {c[1]:.8g} |sin(beta) cos(beta)|"
        return (
            f"tau_y = {c[0]:.8g} + {c[1]:.8g} p_n + {c[2]:.8g} p_n^2 "
            f"+ {c[3]:.8g} m_beta + {c[4]:.8g} p_n m_beta"
        )


@dataclass(frozen=True)
class CandidateResult:
    model: StrengthModel
    calibration_rmse_mpa: float
    group_cv_rmse_mpa: float
    score: float
    audit: dict[str, Any]


def _yield_observations(frame: pd.DataFrame) -> pd.DataFrame:
    pure_shear = frame["reference_joint_shear_now"].astype(bool)
    if "reference_joint_tension_now" in frame:
        pure_shear &= ~frame["reference_joint_tension_now"].astype(bool)
    yielded = frame.loc[pure_shear].copy()
    if yielded.empty:
        raise ValueError("No yielded weak-plane observations are available for fitting.")
    yielded["p_n"] = -yielded["reference_sigma_n_mpa"].to_numpy(float)
    yielded["p_mean"] = yielded["reference_mean_compression_mpa"].to_numpy(float)
    beta = np.deg2rad(yielded["beta_deg"].to_numpy(float))
    yielded["m_beta"] = np.abs(np.sin(beta) * np.cos(beta))
    return yielded


def _design_matrix(observations: pd.DataFrame, family: str) -> np.ndarray:
    ones = np.ones(len(observations), dtype=float)
    p_n = observations["p_n"].to_numpy(float)
    p_mean = observations["p_mean"].to_numpy(float)
    m_beta = observations["m_beta"].to_numpy(float)
    columns: dict[str, list[np.ndarray]] = {
        "constant": [ones],
        "normal_linear": [ones, p_n],
        "normal_quadratic": [ones, p_n, p_n**2],
        "global_pressure_linear": [ones, p_mean],
        "orientation_linear": [ones, m_beta],
        "generic_polynomial": [ones, p_n, p_n**2, m_beta, p_n * m_beta],
    }
    try:
        return np.column_stack(columns[family])
    except KeyError as exc:
        raise ValueError(f"Unknown strength family: {family}") from exc


def fit_strength_model(frame: pd.DataFrame, family: str, *, constrained: bool = True) -> StrengthModel:
    observations = _yield_observations(frame)
    design = _design_matrix(observations, family)
    target = observations["reference_tau_mpa"].to_numpy(float)
    if constrained:
        result = lsq_linear(design, target, bounds=(0.0, np.inf), lsmr_tol="auto")
        coefficients = result.x
    else:
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    return StrengthModel(
        family=family,
        coefficients=tuple(float(value) for value in coefficients),
        complexity=int(design.shape[1]),
    )


def replay_dataset(
    frame: pd.DataFrame,
    model: StrengthModel | None,
    parameters: WeakPlaneParameters,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trajectory_id, group in frame.groupby("trajectory_id", sort=False):
        ordered = group.sort_values("step")
        first = ordered.iloc[0]
        pressure = float(first["confining_pressure_mpa"])
        beta = float(first["beta_deg"])
        state = WeakPlaneState(stress_mpa=-pressure * np.eye(3))
        for _, source in ordered.iterrows():
            state = update_weak_plane(
                state,
                components_to_strain(source),
                beta,
                parameters,
                enable_weak_plane=model is not None,
                strength_law=None if model is None else model.evaluate,
            )
            sigma_n, _, tau = resolved_traction(state.stress_mpa, beta)
            row: dict[str, Any] = {
                "trajectory_id": trajectory_id,
                "step": int(source["step"]),
                "predicted_sigma_n_mpa": sigma_n,
                "predicted_tau_mpa": tau,
                "predicted_mean_compression_mpa": max(
                    -float(np.trace(state.stress_mpa)) / 3.0,
                    0.0,
                ),
                "predicted_joint_shear_now": int(state.joint_shear_now),
                "predicted_accumulated_plastic_shear": state.accumulated_plastic_shear,
                "predicted_plastic_dissipation_mpa": state.plastic_dissipation_mpa,
            }
            for i, name in enumerate(STRESS_COMPONENTS):
                indices = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
                row[f"predicted_sigma_{name}_mpa"] = float(state.stress_mpa[indices[i]])
            rows.append(row)
    return pd.DataFrame(rows)


def stress_rmse(frame: pd.DataFrame, predictions: pd.DataFrame) -> float:
    reference_values = frame[[f"reference_sigma_{name}_mpa" for name in STRESS_COMPONENTS]].to_numpy(float)
    predicted_values = predictions[[f"predicted_sigma_{name}_mpa" for name in STRESS_COMPONENTS]].to_numpy(float)
    return float(np.sqrt(np.mean((reference_values - predicted_values) ** 2)))


def audit_strength_model(
    model: StrengthModel,
    parameters: WeakPlaneParameters,
    evaluation_frame: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    pressures = np.linspace(0.0, 25.0, 101)
    state = WeakPlaneState()
    strengths = np.asarray([model.evaluate(-p, p, 37.0, state) for p in pressures], dtype=float)
    violations: list[str] = []
    if not np.all(np.isfinite(strengths)):
        violations.append("non_finite_strength")
    if float(np.nanmin(strengths)) < -1.0e-9:
        violations.append("negative_joint_strength")
    if float(np.nanmin(np.diff(strengths))) < -1.0e-7:
        violations.append("strength_decreases_with_confinement")
    if model.family == "normal_linear" and model.coefficients[1] > np.tan(np.deg2rad(80.0)):
        violations.append("implausibly_large_friction_coefficient")
    min_dissipation_increment = None
    max_yield_drift = None
    if evaluation_frame is not None and predictions is not None:
        merged = evaluation_frame.reset_index(drop=True).join(predictions.reset_index(drop=True), rsuffix="_pred")
        increments = merged.groupby("trajectory_id")["predicted_plastic_dissipation_mpa"].diff().fillna(
            merged["predicted_plastic_dissipation_mpa"]
        )
        min_dissipation_increment = float(increments.min())
        if min_dissipation_increment < -1.0e-9:
            violations.append("negative_plastic_dissipation")
        strengths_pred = np.asarray(
            [
                model.evaluate(
                    float(row.predicted_sigma_n_mpa),
                    float(row.predicted_mean_compression_mpa),
                    float(row.beta_deg),
                    state,
                )
                for row in merged.itertuples()
            ]
        )
        max_yield_drift = float(
            np.max(
                merged["predicted_tau_mpa"].to_numpy(float)
                - np.maximum(strengths_pred, 0.0)
            )
        )
        if max_yield_drift > 1.0e-6:
            violations.append("return_mapping_yield_drift")
    return {
        "violation_count": len(violations),
        "violations": violations,
        "minimum_strength_mpa": float(np.nanmin(strengths)),
        "minimum_strength_slope": float(np.nanmin(np.diff(strengths))),
        "minimum_dissipation_increment_mpa": min_dissipation_increment,
        "maximum_yield_drift_mpa": max_yield_drift,
    }


def _group_cv_rmse(
    calibration: pd.DataFrame,
    family: str,
    parameters: WeakPlaneParameters,
) -> float:
    fold_errors: list[float] = []
    for column in ("beta_deg", "confining_pressure_mpa", "path"):
        for value in calibration[column].drop_duplicates().tolist():
            test = calibration.loc[calibration[column] == value]
            train = calibration.loc[calibration[column] != value]
            try:
                model = fit_strength_model(train, family, constrained=True)
            except ValueError:
                continue
            fold_errors.append(stress_rmse(test, replay_dataset(test, model, parameters)))
    return float(np.mean(fold_errors)) if fold_errors else float("inf")


def run_strength_family_search(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[CandidateResult, list[CandidateResult]]:
    calibration = frame.loc[frame["partition"] == "calibration"].copy()
    parameters = parameters_from_config(config)
    search = config["search"]
    candidates: list[CandidateResult] = []
    for family in search["bounded_families"]:
        model = fit_strength_model(calibration, family, constrained=True)
        predictions = replay_dataset(calibration, model, parameters)
        rmse = stress_rmse(calibration, predictions)
        cv_rmse = _group_cv_rmse(calibration, family, parameters)
        audit = audit_strength_model(model, parameters, calibration, predictions)
        score = (
            cv_rmse
            + float(search["complexity_penalty_mpa"]) * model.complexity
            + float(search["violation_penalty_mpa"]) * audit["violation_count"]
        )
        candidates.append(CandidateResult(model, rmse, cv_rmse, score, audit))
    candidates.sort(key=lambda item: (item.score, item.model.complexity))
    return candidates[0], candidates


def method_metrics(
    frame: pd.DataFrame,
    model: StrengthModel | None,
    parameters: WeakPlaneParameters,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for partition, subset in frame.groupby("partition", sort=False):
        predictions = replay_dataset(subset, model, parameters)
        predictions.insert(0, "method", method)
        predictions.insert(1, "partition", partition)
        prediction_parts.append(predictions)
        reference_yield = subset.groupby("trajectory_id")["reference_joint_shear_now"].apply(
            lambda values: int(np.argmax(values.to_numpy() > 0) + 1) if np.any(values.to_numpy() > 0) else 0
        )
        predicted_yield = predictions.groupby("trajectory_id")["predicted_joint_shear_now"].apply(
            lambda values: int(np.argmax(values.to_numpy() > 0) + 1) if np.any(values.to_numpy() > 0) else 0
        )
        onset_error = float(np.mean(np.abs(reference_yield.to_numpy() - predicted_yield.to_numpy())))
        audit = (
            {"violation_count": 0, "violations": []}
            if model is None
            else audit_strength_model(model, parameters, subset, predictions)
        )
        metric_rows.append(
            {
                "method": method,
                "partition": partition,
                "stress_rmse_mpa": stress_rmse(subset, predictions),
                "yield_onset_mae_steps": onset_error,
                "physical_violation_count": int(audit["violation_count"]),
                "physical_violations": ";".join(audit["violations"]),
                "complexity": 0 if model is None else model.complexity,
            }
        )
    return pd.concat(prediction_parts, ignore_index=True), pd.DataFrame(metric_rows)


def model_summary(model: StrengthModel) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "family": model.family,
        "coefficients": list(model.coefficients),
        "complexity": model.complexity,
        "formula": model.formula,
    }
    if model.family == "normal_linear":
        summary["recovered_cohesion_mpa"] = model.coefficients[0]
        summary["recovered_friction_deg"] = degrees(atan(model.coefficients[1]))
    return summary

from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from asrc.constitutive.softening import (
    CohesionEvolution,
    EvolutionCandidate,
    softening_parameters,
)
from asrc.constitutive.weak_plane import (
    STRESS_COMPONENTS,
    WeakPlaneParameters,
    WeakPlaneState,
    components_to_strain,
    resolved_traction,
    update_weak_plane,
)


def _softening_observations(
    frame: pd.DataFrame,
    friction_deg: float,
) -> pd.DataFrame:
    mask = frame["reference_joint_shear_now"].astype(bool)
    if "reference_joint_tension_now" in frame:
        mask &= ~frame["reference_joint_tension_now"].astype(bool)
    observations = frame.loc[mask].copy()
    if observations.empty:
        raise ValueError("No pure weak-plane shear observations are available.")
    friction = np.tan(np.deg2rad(friction_deg))
    observations["observed_cohesion_mpa"] = (
        observations["reference_tau_mpa"].to_numpy(float)
        + observations["reference_sigma_n_mpa"].to_numpy(float) * friction
    )
    return observations


def _initial_coefficients(family: str, kappa: np.ndarray, cohesion: np.ndarray) -> np.ndarray:
    peak = max(float(np.max(cohesion)), 1.0e-6)
    residual = max(float(np.min(cohesion)), 0.0)
    scale = max(float(np.quantile(kappa[kappa > 0.0], 0.5)) if np.any(kappa > 0.0) else 1.0e-4, 1.0e-6)
    if family == "constant":
        return np.asarray([float(np.mean(cohesion))])
    if family == "linear_clipped":
        rate = max((peak - residual) / max(float(np.max(kappa)), 1.0e-6), 1.0)
        return np.asarray([residual, peak, rate])
    if family in {"exponential", "rational", "bilinear"}:
        return np.asarray([residual, max(peak - residual, 1.0e-6), scale])
    if family == "stretched_exponential":
        return np.asarray(
            [residual, max(peak - residual, 1.0e-6), scale, 1.0]
        )
    if family == "polynomial":
        return np.asarray([peak, -1.0, 0.0])
    raise ValueError(f"Unknown cohesion evolution family: {family}")


def fit_evolution_model(
    frame: pd.DataFrame,
    family: str,
    friction_deg: float,
    *,
    maximum_function_evaluations: int = 2000,
    constrained: bool = True,
) -> CohesionEvolution:
    observations = _softening_observations(frame, friction_deg)
    kappa = observations["reference_kappa_before"].to_numpy(float)
    target = observations["observed_cohesion_mpa"].to_numpy(float)
    initial = _initial_coefficients(family, kappa, target)
    if family == "polynomial" and not constrained:
        design = np.column_stack([np.ones(len(kappa)), kappa, kappa**2])
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    else:
        if family == "constant":
            lower = np.asarray([0.0])
            upper = np.asarray([10.0])
            complexity = 1
        elif family == "linear_clipped":
            lower = np.asarray([0.0, 0.0, 0.0])
            upper = np.asarray([10.0, 10.0, 1.0e7])
            complexity = 3
        elif family == "stretched_exponential":
            lower = np.asarray([0.0, 0.0, 1.0e-7, 0.25])
            upper = np.asarray([10.0, 10.0, 1.0, 4.0])
            complexity = 4
        else:
            lower = np.asarray([0.0, 0.0, 1.0e-7])
            upper = np.asarray([10.0, 10.0, 1.0])
            complexity = 3

        def residuals(coefficients: np.ndarray) -> np.ndarray:
            model = CohesionEvolution(family, tuple(coefficients), complexity)
            return np.asarray(model.cohesion(kappa), dtype=float) - target

        result = least_squares(
            residuals,
            np.clip(initial, lower + 1.0e-12, upper - 1.0e-12),
            bounds=(lower, upper),
            max_nfev=int(maximum_function_evaluations),
            xtol=1.0e-13,
            ftol=1.0e-13,
            gtol=1.0e-13,
        )
        coefficients = result.x
    return CohesionEvolution(
        family=family,
        coefficients=tuple(float(value) for value in coefficients),
        complexity=(
            1
            if family == "constant"
            else 4
            if family == "stretched_exponential"
            else 3
        ),
    )


def replay_softening_dataset(
    frame: pd.DataFrame,
    model: CohesionEvolution | None,
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
            kappa_before = state.accumulated_plastic_shear
            if model is None:
                law = None
                cohesion_before = parameters.cohesion_mpa
            else:
                law = lambda sn, pm, beta_value, current: model.strength(
                    sn, pm, beta_value, current, parameters.friction_deg
                )
                cohesion_before = float(model.cohesion(kappa_before))
            state = update_weak_plane(
                state,
                components_to_strain(source),
                beta,
                parameters,
                strength_law=law,
            )
            sigma_n, _, tau = resolved_traction(state.stress_mpa, beta)
            row: dict[str, Any] = {
                "trajectory_id": trajectory_id,
                "step": int(source["step"]),
                "predicted_kappa_before": kappa_before,
                "predicted_kappa_after": state.accumulated_plastic_shear,
                "predicted_cohesion_before_mpa": cohesion_before,
                "predicted_sigma_n_mpa": sigma_n,
                "predicted_tau_mpa": tau,
                "predicted_joint_shear_now": int(state.joint_shear_now),
                "predicted_joint_tension_now": int(state.joint_tension_now),
                "predicted_plastic_dissipation_mpa": state.plastic_dissipation_mpa,
            }
            indices = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
            for name, index in zip(STRESS_COMPONENTS, indices):
                row[f"predicted_sigma_{name}_mpa"] = float(state.stress_mpa[index])
            rows.append(row)
    return pd.DataFrame(rows)


def softening_stress_rmse(frame: pd.DataFrame, predictions: pd.DataFrame) -> float:
    reference = frame[[f"reference_sigma_{name}_mpa" for name in STRESS_COMPONENTS]].to_numpy(float)
    predicted = predictions[[f"predicted_sigma_{name}_mpa" for name in STRESS_COMPONENTS]].to_numpy(float)
    return float(np.sqrt(np.mean((reference - predicted) ** 2)))


def audit_evolution_model(
    model: CohesionEvolution,
    maximum_kappa: float,
) -> dict[str, Any]:
    domain_end = max(float(maximum_kappa) * 1.5, 1.0e-5)
    kappa = np.linspace(0.0, domain_end, 301)
    cohesion = np.asarray(model.cohesion(kappa), dtype=float)
    violations: list[str] = []
    if not np.all(np.isfinite(cohesion)):
        violations.append("non_finite_cohesion")
    if float(np.nanmin(cohesion)) < -1.0e-9:
        violations.append("negative_cohesion")
    if float(np.nanmax(np.diff(cohesion))) > 1.0e-7:
        violations.append("cohesion_heals_with_plastic_shear")
    if isfinite(model.residual_cohesion_mpa) and model.residual_cohesion_mpa > model.peak_cohesion_mpa + 1.0e-9:
        violations.append("residual_exceeds_peak")
    return {
        "violation_count": len(violations),
        "violations": violations,
        "minimum_cohesion_mpa": float(np.nanmin(cohesion)),
        "maximum_cohesion_slope": float(np.nanmax(np.diff(cohesion))),
        "audit_kappa_max": domain_end,
    }


def _group_cv_rmse(
    calibration: pd.DataFrame,
    family: str,
    parameters: WeakPlaneParameters,
    maximum_function_evaluations: int,
) -> float:
    errors: list[float] = []
    for column in ("beta_deg", "confining_pressure_mpa", "path"):
        for value in calibration[column].drop_duplicates().tolist():
            train = calibration.loc[calibration[column] != value]
            test = calibration.loc[calibration[column] == value]
            try:
                model = fit_evolution_model(
                    train,
                    family,
                    parameters.friction_deg,
                    maximum_function_evaluations=maximum_function_evaluations,
                )
            except ValueError:
                continue
            errors.append(
                softening_stress_rmse(
                    test,
                    replay_softening_dataset(test, model, parameters),
                )
            )
    return float(np.mean(errors)) if errors else float("inf")


def run_evolution_search(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[EvolutionCandidate, list[EvolutionCandidate]]:
    calibration = frame.loc[frame["partition"] == "calibration"].copy()
    parameters = softening_parameters(config)
    search = config["search"]
    max_kappa = float(calibration["reference_kappa_after"].max())
    candidates: list[EvolutionCandidate] = []
    for family in search["bounded_families"]:
        model = fit_evolution_model(
            calibration,
            family,
            parameters.friction_deg,
            maximum_function_evaluations=int(search["maximum_function_evaluations"]),
        )
        predictions = replay_softening_dataset(calibration, model, parameters)
        calibration_rmse = softening_stress_rmse(calibration, predictions)
        cv_rmse = _group_cv_rmse(
            calibration,
            family,
            parameters,
            int(search["maximum_function_evaluations"]),
        )
        audit = audit_evolution_model(model, max_kappa)
        score = (
            cv_rmse
            + float(search["complexity_penalty_mpa"]) * model.complexity
            + float(search["violation_penalty_mpa"]) * audit["violation_count"]
        )
        candidates.append(
            EvolutionCandidate(model, calibration_rmse, cv_rmse, score, audit)
        )
    candidates.sort(key=lambda result: (result.score, result.model.complexity))
    return candidates[0], candidates


def softening_method_metrics(
    frame: pd.DataFrame,
    model: CohesionEvolution | None,
    parameters: WeakPlaneParameters,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions_all: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    maximum_kappa = float(frame["reference_kappa_after"].max())
    audit = (
        {"violation_count": 0, "violations": []}
        if model is None
        else audit_evolution_model(model, maximum_kappa)
    )
    for partition, subset in frame.groupby("partition", sort=False):
        predictions = replay_softening_dataset(subset, model, parameters)
        predictions.insert(0, "method", method)
        predictions.insert(1, "partition", partition)
        predictions_all.append(predictions)
        metrics.append(
            {
                "method": method,
                "partition": partition,
                "stress_rmse_mpa": softening_stress_rmse(subset, predictions),
                "cohesion_rmse_mpa": float(
                    np.sqrt(
                        np.mean(
                            (
                                subset["reference_cohesion_before_mpa"].to_numpy(float)
                                - predictions["predicted_cohesion_before_mpa"].to_numpy(float)
                            )
                            ** 2
                        )
                    )
                ),
                "maximum_kappa_error": float(
                    np.max(
                        np.abs(
                            subset["reference_kappa_after"].to_numpy(float)
                            - predictions["predicted_kappa_after"].to_numpy(float)
                        )
                    )
                ),
                "physical_violation_count": int(audit["violation_count"]),
                "physical_violations": ";".join(audit["violations"]),
                "complexity": 0 if model is None else model.complexity,
            }
        )
    return pd.concat(predictions_all, ignore_index=True), pd.DataFrame(metrics)


def evolution_summary(model: CohesionEvolution) -> dict[str, Any]:
    residual = model.residual_cohesion_mpa
    return {
        "family": model.family,
        "coefficients": list(model.coefficients),
        "complexity": model.complexity,
        "formula": model.formula,
        "peak_cohesion_mpa": model.peak_cohesion_mpa,
        "residual_cohesion_mpa": residual if isfinite(residual) else None,
        "softening_scale": model.softening_scale,
    }

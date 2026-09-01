from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from asrc.constitutive.softening import CohesionEvolution
from asrc.constitutive.softening_search import audit_evolution_model
from asrc.constitutive.weak_plane import (
    STRESS_COMPONENTS,
    WeakPlaneParameters,
    WeakPlaneState,
    components_to_strain,
    update_weak_plane,
)


@dataclass(frozen=True)
class C3Candidate:
    model: CohesionEvolution
    calibration_rmse_mpa: float
    group_cv_rmse_mpa: float
    score: float
    audit: dict[str, Any]


def c3_known_parameters(config: dict[str, Any]) -> WeakPlaneParameters:
    elastic = config["material"]["elastic"]
    joint = config["material"]["known_joint"]
    return WeakPlaneParameters(
        young_mpa=float(elastic["young_mpa"]),
        poisson=float(elastic["poisson"]),
        cohesion_mpa=float(joint["baseline_cohesion_mpa"]),
        friction_deg=float(joint["friction_deg"]),
        dilation_deg=float(joint["dilation_deg"]),
        tension_mpa=float(joint["tension_mpa"]),
    )


def _initial_coefficients(
    family: str,
    config: dict[str, Any],
) -> np.ndarray:
    guess = config["search"]["initial_guess"]
    peak = float(guess["peak_cohesion_mpa"])
    residual = float(guess["residual_cohesion_mpa"])
    scale = float(guess["softening_scale"])
    if family == "constant":
        return np.asarray([peak], dtype=float)
    if family == "linear_clipped":
        rate = max((peak - residual) / max(scale, 1.0e-8), 1.0)
        return np.asarray([residual, peak, rate], dtype=float)
    if family in {"exponential", "rational", "bilinear"}:
        return np.asarray([residual, peak - residual, scale], dtype=float)
    if family == "stretched_exponential":
        return np.asarray([residual, peak - residual, scale, 1.0], dtype=float)
    if family == "polynomial":
        return np.asarray([peak, -1000.0, 0.0], dtype=float)
    raise ValueError(f"Unknown C3 evolution family: {family}")


def _coefficient_bounds(
    family: str,
    constrained: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if family == "constant":
        return np.asarray([0.0]), np.asarray([10.0])
    if family == "linear_clipped":
        return (
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([10.0, 10.0, 1.0e7]),
        )
    if family in {"exponential", "rational", "bilinear"}:
        return (
            np.asarray([0.0, 0.0, 1.0e-7]),
            np.asarray([10.0, 10.0, 0.1]),
        )
    if family == "stretched_exponential":
        return (
            np.asarray([0.0, 0.0, 1.0e-7, 0.25]),
            np.asarray([10.0, 10.0, 0.1, 4.0]),
        )
    if family == "polynomial" and not constrained:
        return (
            np.asarray([-10.0, -1.0e6, -1.0e10]),
            np.asarray([20.0, 1.0e6, 1.0e10]),
        )
    raise ValueError(f"Unsupported C3 family/bounds: {family}")


def replay_c3_paths(
    frame: pd.DataFrame,
    model: CohesionEvolution | None,
    parameters: WeakPlaneParameters,
) -> pd.DataFrame:
    """Replay every complete trajectory and preserve the input row order."""
    rows: list[dict[str, Any]] = []
    working = frame.reset_index(drop=False).rename(columns={"index": "_source_index"})
    for trajectory_id, group in working.groupby("trajectory_id", sort=False):
        ordered = group.sort_values("step")
        first = ordered.iloc[0]
        beta = float(first["beta_deg"])
        pressure = float(first["confining_pressure_mpa"])
        state = WeakPlaneState(stress_mpa=-pressure * np.eye(3))
        for _, source in ordered.iterrows():
            if model is None:
                law = None
                cohesion_before = parameters.cohesion_mpa
            else:
                law = lambda sn, pm, beta_value, current: model.strength(
                    sn,
                    pm,
                    beta_value,
                    current,
                    parameters.friction_deg,
                )
                cohesion_before = float(
                    model.cohesion(state.accumulated_plastic_shear)
                )
            state = update_weak_plane(
                state,
                components_to_strain(source),
                beta,
                parameters,
                strength_law=law,
            )
            row: dict[str, Any] = {
                "_source_index": int(source["_source_index"]),
                "trajectory_id": trajectory_id,
                "step": int(source["step"]),
                "predicted_kappa": state.accumulated_plastic_shear,
                "predicted_cohesion_mpa": cohesion_before,
                "predicted_joint_shear_now": int(state.joint_shear_now),
            }
            indices = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
            for component, index in zip(STRESS_COMPONENTS, indices):
                row[f"predicted_sigma_{component}_mpa"] = float(
                    state.stress_mpa[index]
                )
            rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("_source_index")
        .drop(columns="_source_index")
        .reset_index(drop=True)
    )


def observed_residual_vector(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    state_weight_mpa: float,
) -> np.ndarray:
    residuals: list[np.ndarray] = []
    for component in STRESS_COMPONENTS:
        column = f"observed_sigma_{component}_mpa"
        if column not in frame:
            continue
        observed = frame[column].to_numpy(float)
        mask = np.isfinite(observed)
        if np.any(mask):
            predicted = predictions[
                f"predicted_sigma_{component}_mpa"
            ].to_numpy(float)
            residuals.append(predicted[mask] - observed[mask])
    if "observed_kappa" in frame:
        observed_state = frame["observed_kappa"].to_numpy(float)
        state_mask = np.isfinite(observed_state)
        if np.any(state_mask):
            predicted_state = predictions["predicted_kappa"].to_numpy(float)
            residuals.append(
                float(state_weight_mpa)
                * (predicted_state[state_mask] - observed_state[state_mask])
            )
    if not residuals:
        raise ValueError("C3 frame contains no finite stress or state observations.")
    return np.concatenate(residuals)


def observed_stress_rmse(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
) -> float:
    residuals: list[np.ndarray] = []
    for component in STRESS_COMPONENTS:
        column = f"observed_sigma_{component}_mpa"
        if column not in frame:
            continue
        observed = frame[column].to_numpy(float)
        mask = np.isfinite(observed)
        if np.any(mask):
            predicted = predictions[
                f"predicted_sigma_{component}_mpa"
            ].to_numpy(float)
            residuals.append(predicted[mask] - observed[mask])
    if not residuals:
        raise ValueError("C3 frame contains no finite stress observations.")
    values = np.concatenate(residuals)
    return float(np.sqrt(np.mean(values**2)))


def c3_candidate_diagnostics(
    frame: pd.DataFrame,
    model: CohesionEvolution,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Summarize calibration residual structure without hidden truth columns."""
    predictions = replay_c3_paths(
        frame,
        model,
        c3_known_parameters(config),
    )
    residual_columns: dict[str, np.ndarray] = {}
    for component in STRESS_COMPONENTS:
        observed_column = f"observed_sigma_{component}_mpa"
        if observed_column not in frame:
            continue
        observed = frame[observed_column].to_numpy(float)
        predicted = predictions[
            f"predicted_sigma_{component}_mpa"
        ].to_numpy(float)
        residual_columns[component] = predicted - observed
    if not residual_columns:
        raise ValueError("C3 diagnostics require observed stress components.")

    residual_matrix = np.column_stack(list(residual_columns.values()))
    valid_rows = np.any(np.isfinite(residual_matrix), axis=1)
    squared = np.where(np.isfinite(residual_matrix), residual_matrix**2, np.nan)
    row_rmse = np.full(len(frame), np.nan, dtype=float)
    row_bias = np.full(len(frame), np.nan, dtype=float)
    row_rmse[valid_rows] = np.sqrt(
        np.nanmean(squared[valid_rows], axis=1)
    )
    row_bias[valid_rows] = np.nanmean(
        residual_matrix[valid_rows],
        axis=1,
    )
    kappa = predictions["predicted_kappa"].to_numpy(float)
    observed_kappa = kappa[valid_rows]
    quantiles = np.quantile(observed_kappa, [0.0, 0.25, 0.5, 0.75, 1.0])
    bins = []
    for index in range(4):
        lower = float(quantiles[index])
        upper = float(quantiles[index + 1])
        if index == 3:
            mask = valid_rows & (kappa >= lower) & (kappa <= upper)
        else:
            mask = valid_rows & (kappa >= lower) & (kappa < upper)
        if not np.any(mask):
            continue
        xz = residual_columns.get("xz")
        bins.append(
            {
                "quantile": index + 1,
                "predicted_kappa_min": lower,
                "predicted_kappa_max": upper,
                "observed_row_count": int(mask.sum()),
                "all_component_rmse_mpa": float(
                    np.sqrt(np.nanmean(residual_matrix[mask] ** 2))
                ),
                "all_component_signed_bias_mpa": float(
                    np.nanmean(row_bias[mask])
                ),
                "xz_signed_bias_mpa": (
                    float(np.nanmean(xz[mask])) if xz is not None else None
                ),
                "xz_rmse_mpa": (
                    float(np.sqrt(np.nanmean(xz[mask] ** 2)))
                    if xz is not None
                    else None
                ),
            }
        )

    diagnostic_frame = frame[
        ["beta_deg", "confining_pressure_mpa", "path"]
    ].copy()
    diagnostic_frame["row_observed"] = valid_rows
    diagnostic_frame["row_stress_rmse_mpa"] = row_rmse
    grouped = {}
    for column in ("beta_deg", "confining_pressure_mpa", "path"):
        rows = []
        for value, group in diagnostic_frame.groupby(column, sort=True):
            observed = group.loc[group["row_observed"]]
            rows.append(
                {
                    column: (
                        float(value)
                        if column != "path"
                        else str(value)
                    ),
                    "observed_row_count": int(len(observed)),
                    "calibration_stress_rmse_mpa": float(
                        np.sqrt(
                            np.mean(
                                observed["row_stress_rmse_mpa"].to_numpy(
                                    float
                                )
                                ** 2
                            )
                        )
                    ),
                }
            )
        grouped[column] = rows
    return {
        "data_scope": "calibration_only",
        "locked_data_used": False,
        "candidate_family": model.family,
        "candidate_formula": model.formula,
        "residual_by_predicted_kappa_quantile": bins,
        "grouped_calibration_residuals": grouped,
        "signed_residual_definition": "predicted_minus_observed",
    }


def fit_c3_evolution(
    frame: pd.DataFrame,
    family: str,
    config: dict[str, Any],
    *,
    constrained: bool = True,
) -> CohesionEvolution:
    parameters = c3_known_parameters(config)
    initial = _initial_coefficients(family, config)
    lower, upper = _coefficient_bounds(family, constrained)
    complexity = (
        1
        if family == "constant"
        else 4
        if family == "stretched_exponential"
        else 3
    )
    state_weight = float(config["search"]["state_residual_weight_mpa"])

    def residuals(coefficients: np.ndarray) -> np.ndarray:
        model = CohesionEvolution(
            family,
            tuple(float(value) for value in coefficients),
            complexity,
        )
        predictions = replay_c3_paths(frame, model, parameters)
        return observed_residual_vector(frame, predictions, state_weight)

    result = least_squares(
        residuals,
        np.clip(initial, lower + 1.0e-12, upper - 1.0e-12),
        bounds=(lower, upper),
        max_nfev=int(config["search"]["maximum_function_evaluations"]),
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
    )
    return CohesionEvolution(
        family,
        tuple(float(value) for value in result.x),
        complexity,
    )


def _trajectory_folds(
    frame: pd.DataFrame,
    fold_count: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    trajectories = sorted(frame["trajectory_id"].astype(str).unique().tolist())
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold in range(min(int(fold_count), len(trajectories))):
        test_ids = {
            trajectory
            for index, trajectory in enumerate(trajectories)
            if index % int(fold_count) == fold
        }
        train = frame.loc[~frame["trajectory_id"].isin(test_ids)].copy()
        test = frame.loc[frame["trajectory_id"].isin(test_ids)].copy()
        if not train.empty and not test.empty:
            folds.append((train, test))
    return folds


def c3_group_cv_rmse(
    frame: pd.DataFrame,
    family: str,
    config: dict[str, Any],
    *,
    constrained: bool = True,
) -> float:
    fold_errors = []
    for train, test in _trajectory_folds(
        frame,
        int(config["search"]["group_cv_folds"]),
    ):
        model = fit_c3_evolution(
            train,
            family,
            config,
            constrained=constrained,
        )
        predictions = replay_c3_paths(
            test,
            model,
            c3_known_parameters(config),
        )
        fold_errors.append(observed_stress_rmse(test, predictions))
    return float(np.mean(fold_errors)) if fold_errors else float("inf")


def fit_c3_candidate(
    frame: pd.DataFrame,
    family: str,
    config: dict[str, Any],
    *,
    constrained: bool = True,
) -> C3Candidate:
    model = fit_c3_evolution(
        frame,
        family,
        config,
        constrained=constrained,
    )
    predictions = replay_c3_paths(frame, model, c3_known_parameters(config))
    calibration_rmse = observed_stress_rmse(frame, predictions)
    group_cv_rmse = c3_group_cv_rmse(
        frame,
        family,
        config,
        constrained=constrained,
    )
    state_values = frame["observed_kappa"].dropna().to_numpy(float)
    maximum_kappa = float(np.max(state_values)) if len(state_values) else 0.003
    audit = audit_evolution_model(model, maximum_kappa)
    score = (
        group_cv_rmse
        + float(config["search"]["complexity_penalty_mpa"]) * model.complexity
        + float(config["search"]["violation_penalty_mpa"])
        * int(audit["violation_count"])
    )
    return C3Candidate(
        model=model,
        calibration_rmse_mpa=calibration_rmse,
        group_cv_rmse_mpa=group_cv_rmse,
        score=score,
        audit=audit,
    )


def run_c3_candidate_search(
    frame: pd.DataFrame,
    config: dict[str, Any],
    families: list[str] | None = None,
) -> tuple[C3Candidate, list[C3Candidate]]:
    requested = list(families or config["search"]["bounded_families"])
    allowed = {str(item) for item in config["search"]["bounded_families"]}
    unknown = set(requested).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown bounded C3 families: {sorted(unknown)}")
    candidates = [
        fit_c3_candidate(frame, family, config) for family in requested
    ]
    candidates.sort(
        key=lambda candidate: (
            candidate.score,
            candidate.model.complexity,
            candidate.model.family,
        )
    )
    return candidates[0], candidates


def evaluate_c3_model(
    frame: pd.DataFrame,
    model: CohesionEvolution | None,
    config: dict[str, Any],
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions_all: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    parameters = c3_known_parameters(config)
    maximum_kappa = float(frame["reference_kappa_after"].max())
    audit = (
        {"violation_count": 0, "violations": []}
        if model is None
        else audit_evolution_model(model, maximum_kappa)
    )
    reference_columns = [
        f"reference_sigma_{component}_mpa" for component in STRESS_COMPONENTS
    ]
    predicted_columns = [
        f"predicted_sigma_{component}_mpa" for component in STRESS_COMPONENTS
    ]
    for partition, subset in frame.groupby("partition", sort=False):
        predictions = replay_c3_paths(subset, model, parameters)
        predictions.insert(0, "method", method)
        predictions.insert(1, "partition", partition)
        predictions_all.append(predictions)
        stress_error = (
            subset[reference_columns].to_numpy(float)
            - predictions[predicted_columns].to_numpy(float)
        )
        observed_rmse = (
            observed_stress_rmse(subset, predictions)
            if subset["stress_observed"].any()
            else float("nan")
        )
        metrics.append(
            {
                "method": method,
                "partition": partition,
                "truth_stress_rmse_mpa": float(
                    np.sqrt(np.mean(stress_error**2))
                ),
                "observed_stress_rmse_mpa": observed_rmse,
                "truth_kappa_rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                subset["reference_kappa_after"].to_numpy(float)
                                - predictions["predicted_kappa"].to_numpy(float)
                            )
                            ** 2
                        )
                    )
                ),
                "physical_violation_count": int(audit["violation_count"]),
                "physical_violations": ";".join(audit["violations"]),
                "complexity": 0 if model is None else model.complexity,
            }
        )
    return pd.concat(predictions_all, ignore_index=True), pd.DataFrame(metrics)

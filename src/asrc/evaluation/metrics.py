from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricResult:
    rmse: float
    mae: float
    mape: float
    r2: float
    max_abs_error: float

    def to_dict(self) -> dict[str, float]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "r2": self.r2,
            "max_abs_error": self.max_abs_error,
        }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricResult:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(err) / denom) * 100.0)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return MetricResult(rmse, mae, mape, r2, float(np.max(np.abs(err))))


def evaluate_formula(case: Any, correction: np.ndarray) -> dict[str, float]:
    y_true = case.frame[case.y_true].to_numpy(float)
    y_base = case.frame[case.y_base].to_numpy(float)
    y_corr = y_base + np.asarray(correction, dtype=float)
    return regression_metrics(y_true, y_corr).to_dict()


def add_prediction_columns(case: Any, correction: np.ndarray, method: str) -> pd.DataFrame:
    df = case.frame.copy()
    df["method"] = method
    df["correction_MPa"] = np.asarray(correction, dtype=float)
    df["corrected_strength_MPa"] = df[case.y_base].astype(float) + df["correction_MPa"]
    df["residual_after_correction_MPa"] = df[case.y_true].astype(float) - df["corrected_strength_MPa"]
    return df

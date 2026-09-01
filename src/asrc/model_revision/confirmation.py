from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np
import pandas as pd


def integrated_error_by_cell(
    trajectory: pd.DataFrame,
    *,
    value_column: str = "locked_rmse",
) -> pd.DataFrame:
    """Integrate an error trajectory over acquisition budget for each paired cell."""
    keys = ["task_id", "data_seed", "strategy"]
    rows: list[dict[str, object]] = []
    for key, group in trajectory.groupby(keys, sort=True):
        ordered = group.sort_values("new_observation_count")
        budgets = ordered["new_observation_count"].to_numpy(float)
        values = ordered[value_column].to_numpy(float)
        if len(budgets) < 2 or np.any(np.diff(budgets) <= 0.0):
            raise ValueError("Each trajectory must contain increasing unique budgets.")
        span = float(budgets[-1] - budgets[0])
        if span <= 0.0 or not np.all(np.isfinite(values)):
            raise ValueError("Integrated-error inputs must be finite with positive span.")
        rows.append(
            {
                **dict(zip(keys, key)),
                "integrated_locked_rmse": float(np.trapezoid(values, budgets) / span),
            }
        )
    return pd.DataFrame(rows)


def _one_sided_sign_flip_pvalue(cluster_differences: np.ndarray) -> float:
    values = np.asarray(cluster_differences, dtype=float)
    observed = float(np.mean(values))
    if len(values) <= 16:
        statistics = np.asarray(
            [
                np.mean(values * np.asarray(signs))
                for signs in product((-1.0, 1.0), repeat=len(values))
            ]
        )
    else:
        rng = np.random.default_rng(73191)
        signs = rng.choice((-1.0, 1.0), size=(100000, len(values)))
        statistics = np.mean(signs * values[None, :], axis=1)
    return float((np.count_nonzero(statistics <= observed) + 1) / (len(statistics) + 1))


def paired_cluster_comparisons(
    cell_metrics: pd.DataFrame,
    *,
    treatment: str,
    comparators: Iterable[str],
    value_column: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Compare strategies with paired differences and a seed-cluster bootstrap."""
    pivot = cell_metrics.pivot(
        index=["task_id", "data_seed"],
        columns="strategy",
        values=value_column,
    )
    rows: list[dict[str, object]] = []
    for offset, comparator in enumerate(comparators):
        if treatment not in pivot or comparator not in pivot:
            raise ValueError("Treatment or comparator is absent from paired metrics.")
        differences = (pivot[treatment] - pivot[comparator]).dropna()
        if len(differences) != len(pivot):
            raise ValueError("Paired comparison contains missing task-seed cells.")
        difference_frame = differences.rename("difference").reset_index()
        clusters = difference_frame.groupby("data_seed")["difference"].mean().to_numpy(float)
        rng = np.random.default_rng(int(bootstrap_seed) + offset)
        samples = rng.choice(
            clusters,
            size=(int(bootstrap_replicates), len(clusters)),
            replace=True,
        ).mean(axis=1)
        rows.append(
            {
                "treatment": treatment,
                "comparator": comparator,
                "metric": value_column,
                "paired_cell_count": len(differences),
                "seed_cluster_count": len(clusters),
                "mean_difference": float(differences.mean()),
                "median_difference": float(differences.median()),
                "bootstrap_ci_lower": float(np.quantile(samples, 0.025)),
                "bootstrap_ci_upper": float(np.quantile(samples, 0.975)),
                "one_sided_sign_flip_pvalue": _one_sided_sign_flip_pvalue(clusters),
                "treatment_win_count": int((differences < 0.0).sum()),
                "tie_count": int(np.isclose(differences, 0.0, atol=1e-12).sum()),
            }
        )
    return pd.DataFrame(rows)

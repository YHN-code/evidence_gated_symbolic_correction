from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import pandas as pd


def pairwise_jaccard(structure_sets: list[set[str]]) -> float:
    values: list[float] = []
    for left, right in combinations(structure_sets, 2):
        union = left | right
        values.append(float(len(left & right) / len(union)) if union else 1.0)
    return float(sum(values) / len(values)) if values else 1.0


def summarize_stability(
    run_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    *,
    required_recovery_fraction: float,
    minimum_valid_candidate_fraction: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_id, task_runs in run_frame.groupby("task_id", sort=True):
        if candidate_frame.empty:
            task_candidates = pd.DataFrame(columns=["replicate", "structural_key"])
        else:
            task_candidates = candidate_frame.loc[candidate_frame["task_id"].eq(task_id)]
        structures_by_replicate = {
            int(replicate): set(group["structural_key"].astype(str))
            for replicate, group in task_candidates.groupby("replicate", sort=True)
        }
        structure_sets = [
            structures_by_replicate.get(int(replicate), set())
            for replicate in sorted(task_runs["replicate"].unique())
        ]
        replicate_count = int(len(task_runs))
        recovery_count = int(task_runs["corrected_behavior_recovered"].astype(bool).sum())
        required_count = int(
            math.ceil(required_recovery_fraction * replicate_count - 1.0e-12)
        )
        candidate_count = int(task_runs["candidate_count"].sum())
        valid_count = int(task_runs["valid_candidate_count"].sum())
        valid_fraction = float(valid_count / candidate_count) if candidate_count else 0.0
        locked_values = pd.to_numeric(task_runs["locked_rmse"], errors="coerce").dropna()
        selected_formulas = task_runs["selected_formula"].fillna("").astype(str)
        selected_formulas = selected_formulas.loc[selected_formulas.str.len().gt(0)]
        rows.append(
            {
                "task_id": task_id,
                "replicate_count": replicate_count,
                "required_recovery_count": required_count,
                "behavior_recovery_count": recovery_count,
                "behavior_recovery_fraction": float(recovery_count / replicate_count),
                "oracle_pool_count": int(
                    task_runs["oracle_patch_in_candidate_pool"].astype(bool).sum()
                ),
                "exact_selected_count": int(
                    task_runs["exact_patch_expression_match"].astype(bool).sum()
                ),
                "proposal_acceptance_fraction": float(
                    task_runs["accepted_proposal_count"].sum()
                    / max(1, task_runs["raw_proposal_count"].sum())
                ),
                "valid_candidate_fraction": valid_fraction,
                "locked_rmse_min": (
                    float(locked_values.min()) if not locked_values.empty else float("nan")
                ),
                "locked_rmse_median": (
                    float(locked_values.median())
                    if not locked_values.empty
                    else float("nan")
                ),
                "locked_rmse_max": (
                    float(locked_values.max()) if not locked_values.empty else float("nan")
                ),
                "unique_structure_count": (
                    int(task_candidates["structural_key"].astype(str).nunique())
                    if not task_candidates.empty
                    else 0
                ),
                "mean_pairwise_candidate_jaccard": pairwise_jaccard(structure_sets),
                "selected_formula_count": int(selected_formulas.nunique()),
                "stability_gate_passed": bool(
                    recovery_count >= required_count
                    and valid_fraction >= minimum_valid_candidate_fraction
                ),
            }
        )
    return pd.DataFrame(rows)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from asrc.constitutive.softening import CohesionEvolution
from asrc.constitutive.softening_search import audit_evolution_model


C2_AGENT_ACTIONS = (
    "inspect_softening_state",
    "compare_evolution_family",
    "test_unseen_path",
    "challenge_monotonicity",
    "accept_current_best",
)


@dataclass(frozen=True)
class ConstitutiveToolResult:
    action: str
    evidence: dict[str, Any]
    changed_search_state: bool


def execute_c2_constitutive_tool(
    action: str,
    *,
    dataset: pd.DataFrame,
    candidate: CohesionEvolution | None = None,
    candidate_table: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
) -> ConstitutiveToolResult:
    """Execute one auditable C2 tool selected by an external agent policy."""
    if action not in C2_AGENT_ACTIONS:
        raise ValueError(f"Unknown C2 constitutive action: {action}")
    if action == "inspect_softening_state":
        yielded = dataset["reference_joint_shear_now"].astype(bool)
        evidence = {
            "maximum_reference_kappa": float(dataset["reference_kappa_after"].max()),
            "yielded_fraction": float(yielded.mean()),
            "cohesion_drop_mpa": float(
                dataset["reference_cohesion_before_mpa"].max()
                - dataset["reference_cohesion_before_mpa"].min()
            ),
            "paths": sorted(dataset["path"].astype(str).unique().tolist()),
        }
    elif action == "compare_evolution_family":
        if candidate_table is None:
            raise ValueError("compare_evolution_family requires candidate_table.")
        evidence = {
            "candidate_count": len(candidate_table),
            "ranking": candidate_table.to_dict(orient="records"),
        }
    elif action == "test_unseen_path":
        if metrics is None:
            raise ValueError("test_unseen_path requires metrics.")
        locked = metrics.loc[metrics["partition"].astype(str).str.startswith("locked_")]
        evidence = {"locked_metrics": locked.to_dict(orient="records")}
    elif action == "challenge_monotonicity":
        if candidate is None:
            raise ValueError("challenge_monotonicity requires candidate.")
        evidence = audit_evolution_model(
            candidate,
            float(dataset["reference_kappa_after"].max()),
        )
    else:
        if candidate is None or metrics is None:
            raise ValueError("accept_current_best requires candidate and metrics.")
        evidence = {
            "requested_family": candidate.family,
            "zero_physical_violations": not bool(
                metrics["physical_violation_count"].max()
            ),
            "locked_joint_rmse_mpa": float(
                metrics.loc[metrics["partition"] == "locked_joint", "stress_rmse_mpa"].iloc[0]
            ),
            "verifier_controls_acceptance": True,
        }
    return ConstitutiveToolResult(
        action=action,
        evidence=evidence,
        changed_search_state=action == "compare_evolution_family",
    )

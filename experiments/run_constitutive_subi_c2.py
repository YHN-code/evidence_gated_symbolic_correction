from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asrc.constitutive.softening import (
    generate_subi_c2_dataset,
    reference_evolution,
    softening_parameters,
)
from asrc.constitutive.softening_search import (
    evolution_summary,
    fit_evolution_model,
    run_evolution_search,
    softening_method_metrics,
)
from asrc.utils.io import ensure_run_dir, read_yaml


STAGE = "constitutive_subi_c2"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _gate_report(
    metrics: pd.DataFrame,
    selected: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    acceptance = config["acceptance"]
    reference = reference_evolution(config)
    selected_summary = evolution_summary(selected.model)
    corrected = metrics.loc[
        (metrics["method"] == "constitutive_c2_asrc")
        & (metrics["partition"] == "locked_joint")
    ].iloc[0]
    baseline = metrics.loc[
        (metrics["method"] == "perfect_plastic_baseline")
        & (metrics["partition"] == "locked_joint")
    ].iloc[0]
    checks = {
        "selected_reference_family": selected.model.family
        == str(acceptance["selected_family"]),
        "locked_joint_rmse_within_limit": float(corrected["stress_rmse_mpa"])
        <= float(acceptance["maximum_locked_joint_rmse_mpa"]),
        "peak_cohesion_recovered": abs(
            selected.model.peak_cohesion_mpa - reference.peak_cohesion_mpa
        )
        / reference.peak_cohesion_mpa
        <= float(acceptance["maximum_relative_peak_error"]),
        "residual_cohesion_recovered": abs(
            selected.model.residual_cohesion_mpa - reference.residual_cohesion_mpa
        )
        / reference.residual_cohesion_mpa
        <= float(acceptance["maximum_relative_residual_error"]),
        "softening_scale_recovered": bool(
            selected.model.softening_scale
            and reference.softening_scale
            and abs(selected.model.softening_scale - reference.softening_scale)
            / reference.softening_scale
            <= float(acceptance["maximum_relative_scale_error"])
        ),
        "zero_physical_violations": int(corrected["physical_violation_count"])
        <= int(acceptance["maximum_physical_violations"]),
        "improves_perfect_plastic_baseline": float(corrected["stress_rmse_mpa"])
        < float(baseline["stress_rmse_mpa"]),
    }
    return {
        "protocol": config["protocol"],
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "selected_model": selected_summary,
        "reference_model": evolution_summary(reference),
        "selected_group_cv_rmse_mpa": selected.group_cv_rmse_mpa,
        "locked_joint_rmse_mpa": float(corrected["stress_rmse_mpa"]),
        "baseline_locked_joint_rmse_mpa": float(baseline["stress_rmse_mpa"]),
        "scientific_scope": (
            "C2 is a controlled recovery of a hidden path-dependent evolution "
            "law. It validates stateful search and held-out replay, not novel "
            "constitutive discovery."
        ),
        "next_gate": (
            "validate_frozen_c2_law_against_flac3d_subi_single_zone"
            if all(checks.values())
            else "revise_c2_state_or_evolution_search"
        ),
    }


def run(
    config_path: str,
    run_id: str,
    *,
    resume: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> Path:
    config = read_yaml(config_path)
    run_dir = ensure_run_dir(run_id)
    paths = {
        "dataset": run_dir / "data" / "constitutive_subi_c2_dataset.csv",
        "predictions": run_dir / "data" / "constitutive_subi_c2_predictions.csv",
        "metrics": run_dir / "metrics" / "constitutive_subi_c2_metrics.csv",
        "candidates": run_dir / "metrics" / "constitutive_subi_c2_candidates.csv",
        "models": run_dir / "formulas" / "constitutive_subi_c2_models.json",
        "report": run_dir / "reports" / "constitutive_subi_c2_gate.json",
    }
    if resume and not force and paths["report"].exists():
        if verbose:
            print(f"C2 already complete: {paths['report']}", flush=True)
        return run_dir
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[C2 1/4] Generating stateful calibration and locked paths...", flush=True)
    frame = generate_subi_c2_dataset(config)
    frame.to_csv(paths["dataset"], index=False)

    if verbose:
        print(
            f"[C2 2/4] Searching {len(config['search']['bounded_families'])} cohesion-evolution families...",
            flush=True,
        )
    selected, candidates = run_evolution_search(frame, config)
    candidate_rows = [
        {
            "family": item.model.family,
            "formula": item.model.formula,
            "complexity": item.model.complexity,
            "calibration_rmse_mpa": item.calibration_rmse_mpa,
            "group_cv_rmse_mpa": item.group_cv_rmse_mpa,
            "physical_violation_count": item.audit["violation_count"],
            "score": item.score,
            "selected": item.model.family == selected.model.family,
        }
        for item in candidates
    ]
    pd.DataFrame(candidate_rows).to_csv(paths["candidates"], index=False)

    calibration = frame.loc[frame["partition"] == "calibration"]
    parameters = softening_parameters(config)
    generic = fit_evolution_model(
        calibration,
        str(config["search"]["generic_family"]),
        parameters.friction_deg,
        constrained=False,
    )
    methods = [
        ("perfect_plastic_baseline", None),
        ("generic_state_polynomial", generic),
        ("constitutive_c2_asrc", selected.model),
    ]
    if verbose:
        print("[C2 3/4] Replaying frozen laws on all locked partitions...", flush=True)
    predictions_all: list[pd.DataFrame] = []
    metrics_all: list[pd.DataFrame] = []
    for method, model in methods:
        predictions, metrics = softening_method_metrics(
            frame,
            model,
            parameters,
            method,
        )
        predictions_all.append(predictions)
        metrics_all.append(metrics)
        if verbose:
            rmse = metrics.loc[
                metrics["partition"] == "locked_joint", "stress_rmse_mpa"
            ].iloc[0]
            print(f"  {method}: locked-joint RMSE={rmse:.6g} MPa", flush=True)
    predictions = pd.concat(predictions_all, ignore_index=True)
    metrics = pd.concat(metrics_all, ignore_index=True)
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    _write_json(
        paths["models"],
        {
            "reference": evolution_summary(reference_evolution(config)),
            "selected_constitutive_c2_asrc": evolution_summary(selected.model),
            "generic_state_polynomial": evolution_summary(generic),
            "bounded_candidates": candidate_rows,
            "agent_interface": config["agent"],
        },
    )
    report = _gate_report(metrics, selected, config)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["outputs"] = {
        name: str(path.relative_to(run_dir)) for name, path in paths.items()
    }
    _write_json(paths["report"], report)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[STAGE] = {
        "status": "complete",
        "gate_status": report["status"],
        "config": str(Path(config_path)),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, manifest)
    if verbose:
        print(f"[C2 4/4] Gate {report['status']}: {paths['report']}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run C2 path-dependent weak-plane softening recovery."
    )
    parser.add_argument("--config", default="configs/constitutive_subi_c2.yaml")
    parser.add_argument("--run-id", default="constitutive_subi_c2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(
        args.config,
        args.run_id,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

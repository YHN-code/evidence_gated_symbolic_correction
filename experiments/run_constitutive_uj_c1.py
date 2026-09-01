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

from asrc.constitutive.weak_plane import generate_uj_c1_dataset, parameters_from_config
from asrc.constitutive.weak_plane_search import (
    fit_strength_model,
    method_metrics,
    model_summary,
    run_strength_family_search,
)
from asrc.utils.io import ensure_run_dir, read_yaml


STAGE = "constitutive_uj_c1"


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _gate_report(
    metrics: pd.DataFrame,
    selected: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    acceptance = config["acceptance"]
    reference = config["material"]["reference_joint"]
    summary = model_summary(selected.model)
    locked = metrics.loc[
        (metrics["method"] == "constitutive_asrc")
        & metrics["partition"].astype(str).str.startswith("locked_")
    ]
    baseline = metrics.loc[
        (metrics["method"] == "intact_matrix_baseline")
        & (metrics["partition"] == "locked_joint"),
        "stress_rmse_mpa",
    ]
    corrected = metrics.loc[
        (metrics["method"] == "constitutive_asrc")
        & (metrics["partition"] == "locked_joint"),
        "stress_rmse_mpa",
    ]
    checks = {
        "selected_known_reference_family": selected.model.family == acceptance["selected_family"],
        "locked_joint_rmse_within_limit": bool(
            len(corrected)
            and float(corrected.iloc[0]) <= float(acceptance["maximum_locked_joint_rmse_mpa"])
        ),
        "cohesion_recovered": bool(
            "recovered_cohesion_mpa" in summary
            and abs(summary["recovered_cohesion_mpa"] - float(reference["cohesion_mpa"]))
            / float(reference["cohesion_mpa"])
            <= float(acceptance["maximum_relative_cohesion_error"])
        ),
        "friction_recovered": bool(
            "recovered_friction_deg" in summary
            and abs(summary["recovered_friction_deg"] - float(reference["friction_deg"]))
            <= float(acceptance["maximum_friction_error_deg"])
        ),
        "all_locked_partitions_present": set(locked["partition"])
        == {"locked_angle", "locked_pressure", "locked_path", "locked_joint"},
        "zero_physical_violations": bool(
            len(locked)
            and int(locked["physical_violation_count"].max())
            <= int(acceptance["maximum_physical_violations"])
        ),
        "improves_locked_joint_baseline": bool(
            len(baseline) and len(corrected) and float(corrected.iloc[0]) < float(baseline.iloc[0])
        ),
    }
    passed = all(checks.values())
    return {
        "protocol": config["protocol"],
        "status": "passed" if passed else "failed",
        "checks": checks,
        "selected_model": summary,
        "selected_group_cv_rmse_mpa": selected.group_cv_rmse_mpa,
        "locked_joint_rmse_mpa": None if corrected.empty else float(corrected.iloc[0]),
        "baseline_locked_joint_rmse_mpa": None if baseline.empty else float(baseline.iloc[0]),
        "scientific_scope": (
            "C1 is a controlled known-law recovery test. Passing supports the "
            "constitutive search and verifier implementation but does not establish "
            "novel constitutive discovery or LLM-agent superiority."
        ),
        "next_gate": (
            "verify_selected_law_against_flac3d_single_zone"
            if passed
            else "revise_c1_integrator_or_search_before_flac3d"
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
        "dataset": run_dir / "data" / "constitutive_uj_c1_dataset.csv",
        "predictions": run_dir / "data" / "constitutive_uj_c1_predictions.csv",
        "metrics": run_dir / "metrics" / "constitutive_uj_c1_metrics.csv",
        "candidates": run_dir / "metrics" / "constitutive_uj_c1_candidates.csv",
        "models": run_dir / "formulas" / "constitutive_uj_c1_models.json",
        "report": run_dir / "reports" / "constitutive_uj_c1_gate.json",
    }
    if resume and not force and paths["report"].exists():
        if verbose:
            print(f"C1 already complete: {paths['report']}", flush=True)
        return run_dir
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[C1 1/4] Generating calibration and locked material-point paths...", flush=True)
    frame = generate_uj_c1_dataset(config)
    frame.to_csv(paths["dataset"], index=False)

    if verbose:
        print(f"[C1 2/4] Searching {len(config['search']['bounded_families'])} bounded strength families...", flush=True)
    selected, candidates = run_strength_family_search(frame, config)
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
    generic = fit_strength_model(
        calibration,
        config["search"]["generic_family"],
        constrained=False,
    )
    methods = [
        ("intact_matrix_baseline", None),
        ("generic_constitutive_fit", generic),
        ("constitutive_asrc", selected.model),
    ]
    if verbose:
        print("[C1 3/4] Replaying selected laws on angle, pressure, path, and joint hold-outs...", flush=True)
    prediction_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    parameters = parameters_from_config(config)
    for method, model in methods:
        predictions, metrics = method_metrics(frame, model, parameters, method)
        prediction_frames.append(predictions)
        metric_frames.append(metrics)
        if verbose:
            joint_rmse = metrics.loc[metrics["partition"] == "locked_joint", "stress_rmse_mpa"].iloc[0]
            print(f"  {method}: locked-joint RMSE={joint_rmse:.6g} MPa", flush=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions.to_csv(paths["predictions"], index=False)
    metrics.to_csv(paths["metrics"], index=False)

    models = {
        "reference": {
            "family": "normal_linear",
            "cohesion_mpa": config["material"]["reference_joint"]["cohesion_mpa"],
            "friction_deg": config["material"]["reference_joint"]["friction_deg"],
            "formula": "tau_y = c - sigma_n tan(phi)",
        },
        "selected_constitutive_asrc": model_summary(selected.model),
        "generic_constitutive_fit": model_summary(generic),
        "bounded_candidates": candidate_rows,
    }
    _write_json(paths["models"], models)
    report = _gate_report(metrics, selected, config)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["outputs"] = {name: str(path.relative_to(run_dir)) for name, path in paths.items()}
    _write_json(paths["report"], report)

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest[STAGE] = {
        "status": "complete",
        "gate_status": report["status"],
        "config": str(Path(config_path)),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(manifest_path, manifest)
    if verbose:
        print(
            f"[C1 4/4] Gate {report['status']}: {paths['report']}",
            flush=True,
        )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C1 ubiquitous-joint known-law recovery.")
    parser.add_argument("--config", default="configs/constitutive_uj_c1.yaml")
    parser.add_argument("--run-id", default="constitutive_uj_c1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args.config, args.run_id, resume=args.resume, force=args.force, verbose=args.verbose)


if __name__ == "__main__":
    main()

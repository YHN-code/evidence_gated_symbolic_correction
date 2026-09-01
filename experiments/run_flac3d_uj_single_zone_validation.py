from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from asrc.constitutive.flac3d_uj_validation import (
    compare_flac3d_case,
    load_flac3d_case,
    render_flac3d_data_file,
    validation_cases,
    validation_parameters,
)
from asrc.solvers.flac3d import run_flac3d_data_file
from asrc.utils.io import ensure_run_dir, read_yaml


STAGE = "flac3d_uj_single_zone_validation"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _gate_report(
    metrics: pd.DataFrame,
    config: dict[str, Any],
    transcript_path: Path,
) -> dict[str, Any]:
    acceptance = config["acceptance"]
    overall_rmse = float(
        np.sqrt(
            np.mean(
                np.repeat(
                    metrics["stress_rmse_mpa"].to_numpy(float) ** 2,
                    metrics["steps"].to_numpy(int),
                )
            )
        )
    )
    yielded_cases = int(metrics["flac3d_joint_yielded"].sum())
    checks = {
        "all_cases_complete": len(metrics) == len(config["cases"]),
        "overall_stress_rmse_within_limit": overall_rmse
        <= float(acceptance["maximum_overall_stress_rmse_mpa"]),
        "case_stress_error_within_limit": float(
            metrics["maximum_absolute_stress_error_mpa"].max()
        )
        <= float(acceptance["maximum_case_stress_error_mpa"]),
        "resolved_traction_error_within_limit": float(
            metrics["resolved_traction_rmse_mpa"].max()
        )
        <= float(acceptance["maximum_resolved_traction_error_mpa"]),
        "yield_onset_matches": int(metrics["yield_onset_error_steps"].max())
        <= int(acceptance["maximum_yield_onset_error_steps"]),
        "sufficient_cases_reach_joint_yield": yielded_cases
        >= int(acceptance["require_joint_yield_in_cases"]),
    }
    passed = all(checks.values())
    return {
        "protocol": config["protocol"],
        "status": "passed" if passed else "failed",
        "checks": checks,
        "overall_stress_rmse_mpa": overall_rmse,
        "maximum_absolute_stress_error_mpa": float(
            metrics["maximum_absolute_stress_error_mpa"].max()
        ),
        "maximum_resolved_traction_rmse_mpa": float(
            metrics["resolved_traction_rmse_mpa"].max()
        ),
        "maximum_yield_onset_error_steps": int(
            metrics["yield_onset_error_steps"].max()
        ),
        "flac3d_joint_yielded_case_count": yielded_cases,
        "case_count": len(metrics),
        "flac3d_transcript": str(transcript_path),
        "scientific_scope": (
            "This gate validates solver semantics for the controlled C1 law. "
            "It is not a constitutive discovery experiment."
        ),
        "next_gate": (
            "start_c2_path_dependent_subi_recovery"
            if passed
            else "reconcile_python_and_flac3d_constitutive_semantics"
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
    work_dir = ROOT / "flac3d" / "runs" / run_id
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "reports" / "flac3d_uj_single_zone_gate.json"
    metrics_path = run_dir / "metrics" / "flac3d_uj_single_zone_metrics.csv"
    comparison_path = run_dir / "data" / "flac3d_uj_single_zone_comparison.csv"
    data_file = work_dir / "uj_single_zone_validation.dat"
    if resume and not force and report_path.exists():
        if verbose:
            print(f"FLAC3D UJ single-zone validation already complete: {report_path}", flush=True)
        return run_dir

    data_file.write_text(render_flac3d_data_file(config), encoding="ascii")
    cases = validation_cases(config)
    result_paths = [work_dir / f"{case.case_id}_flac3d.csv" for case in cases]
    if force:
        for path in result_paths:
            path.unlink(missing_ok=True)
    if verbose:
        print(
            f"[UJ-FLAC 1/3] Running {len(cases)} homogeneous paths in one FLAC3D session...",
            flush=True,
        )
    result = run_flac3d_data_file(
        config["flac3d"]["executable"],
        data_file,
        work_dir,
        timeout_seconds=float(config["flac3d"]["timeout_seconds"]),
        startup_timeout_seconds=float(config["flac3d"]["startup_timeout_seconds"]),
        verbose=verbose,
        progress_only=True,
        completion_check=lambda: all(path.exists() and path.stat().st_size > 0 for path in result_paths),
    )

    if verbose:
        print("[UJ-FLAC 2/3] Replaying actual FLAC3D strain increments in Python...", flush=True)
    parameters = validation_parameters(config)
    comparisons: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for case in cases:
        steps = int(config["paths"][case.path]["steps"])
        frame = load_flac3d_case(work_dir / f"{case.case_id}_flac3d.csv", steps)
        comparison, metrics = compare_flac3d_case(frame, case, parameters)
        comparisons.append(comparison)
        metric_rows.append(metrics)
        if verbose:
            print(
                f"  {case.case_id}: RMSE={metrics['stress_rmse_mpa']:.6g} MPa, "
                f"yield={metrics['flac3d_yield_onset_step']}/"
                f"{metrics['python_yield_onset_step']}",
                flush=True,
            )
    comparison_frame = pd.concat(comparisons, ignore_index=True)
    metrics_frame = pd.DataFrame(metric_rows)
    comparison_frame.to_csv(comparison_path, index=False)
    metrics_frame.to_csv(metrics_path, index=False)
    report = _gate_report(metrics_frame, config, result.transcript_path)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["outputs"] = {
        "metrics": str(metrics_path.relative_to(run_dir)),
        "comparison": str(comparison_path.relative_to(run_dir)),
        "flac3d_data_file": str(data_file),
    }
    _write_json(report_path, report)
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
        print(f"[UJ-FLAC 3/3] Gate {report['status']}: {report_path}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the C1 weak-plane integrator against a FLAC3D single zone."
    )
    parser.add_argument(
        "--config",
        default="configs/flac3d_uj_single_zone_validation.yaml",
    )
    parser.add_argument("--run-id", default="flac3d_uj_single_zone_validation")
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

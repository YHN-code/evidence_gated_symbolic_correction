from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401

from asrc.constitutive.flac3d_structural_softening_validation import (
    render_flac3d_structural_softening_data_file,
    structural_softening_output_name,
)
from asrc.constitutive.flac3d_subi_validation import (
    SUBIValidationCase,
    compare_subi_case,
    load_subi_case,
)
from asrc.constitutive.structural_softening_application import (
    StructuralSofteningLaw,
    application_parameters,
    build_application_laws,
    load_frozen_structural_softening,
    path_shear_increments,
    replay_material_point,
    softening_curve_frame,
    structural_softening_cases,
)
from asrc.solvers.flac3d import run_flac3d_data_file
from asrc.utils.io import (
    ensure_run_dir,
    read_yaml,
    write_json,
    write_table_bundle,
)
from asrc.utils.progress import mark_done, progress_message, should_skip


ROOT = Path(__file__).resolve().parents[1]
STAGE = "structural_softening_flac3d_validation"
STRESS_COLUMNS = [
    "stress_xx_mpa",
    "stress_yy_mpa",
    "stress_zz_mpa",
    "stress_xy_mpa",
    "stress_xz_mpa",
    "stress_yz_mpa",
]


def _python_evidence(
    config: dict[str, Any],
    laws: tuple[StructuralSofteningLaw, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = structural_softening_cases(config)
    parameters = application_parameters(config)
    traces = []
    for law in laws:
        for case in cases:
            traces.append(
                replay_material_point(
                    law,
                    case,
                    parameters,
                    path_shear_increments(config["paths"][case.path]),
                )
            )
    trace = pd.concat(traces, ignore_index=True)
    curve = softening_curve_frame(laws, maximum_damage=1.0, point_count=401)
    curve_wide = curve.pivot(index="damage", columns="law_id", values="cohesion_mpa")
    curve_rows = []
    oracle_curve = curve_wide["oracle_exponential"].to_numpy(float)
    for law_id in ("baseline_linear", "discovered_exponential"):
        error = curve_wide[law_id].to_numpy(float) - oracle_curve
        curve_rows.append(
            {"law_id": law_id, "curve_rmse_mpa": float(np.sqrt(np.mean(error**2)))}
        )
    curve_metrics = pd.DataFrame(curve_rows)

    response_rows = []
    oracle = trace.loc[trace["law_id"].eq("oracle_exponential")]
    for law_id in ("baseline_linear", "discovered_exponential"):
        selected = trace.loc[trace["law_id"].eq(law_id)]
        joined = selected.merge(
            oracle,
            on=["case_id", "step"],
            suffixes=("", "_oracle"),
            validate="one_to_one",
        )
        errors = np.concatenate(
            [
                joined[column].to_numpy(float)
                - joined[f"{column}_oracle"].to_numpy(float)
                for column in STRESS_COLUMNS
            ]
        )
        response_rows.append(
            {
                "law_id": law_id,
                "stress_response_rmse_mpa": float(np.sqrt(np.mean(errors**2))),
                "maximum_absolute_stress_error_mpa": float(np.max(np.abs(errors))),
            }
        )
    return trace, curve, curve_metrics.merge(pd.DataFrame(response_rows), on="law_id")


def _improvement(metrics: pd.DataFrame, column: str) -> float:
    values = metrics.set_index("law_id")[column]
    baseline = float(values["baseline_linear"])
    discovered = float(values["discovered_exponential"])
    return 1.0 - discovered / max(baseline, np.finfo(float).eps)


def _gate(
    config: dict[str, Any],
    python_metrics: pd.DataFrame,
    flac_metrics: pd.DataFrame | None,
) -> dict[str, Any]:
    limits = config["acceptance"]
    curve_improvement = _improvement(python_metrics, "curve_rmse_mpa")
    response_improvement = _improvement(
        python_metrics, "stress_response_rmse_mpa"
    )
    discovered_curve_rmse = float(
        python_metrics.set_index("law_id").loc[
            "discovered_exponential", "curve_rmse_mpa"
        ]
    )
    checks = {
        "discovered_curve_improves_baseline": curve_improvement
        >= float(limits["minimum_discovered_curve_improvement_fraction"]),
        "discovered_response_improves_baseline": response_improvement
        >= float(limits["minimum_discovered_response_improvement_fraction"]),
        "discovered_curve_matches_oracle": discovered_curve_rmse
        <= float(limits["maximum_discovered_oracle_curve_rmse_mpa"]),
    }
    if flac_metrics is not None:
        checks.update(
            {
                "flac3d_stress_state_matches": float(
                    flac_metrics["stress_rmse_mpa"].max()
                )
                <= float(limits["maximum_flac3d_stress_rmse_mpa"]),
                "flac3d_cohesion_state_matches": float(
                    flac_metrics["maximum_cohesion_error_mpa"].max()
                )
                <= float(limits["maximum_flac3d_cohesion_error_mpa"]),
                "flac3d_plastic_shear_state_matches": float(
                    flac_metrics["maximum_plastic_shear_error"].max()
                )
                <= float(limits["maximum_flac3d_plastic_shear_error"]),
            }
        )
    return {
        "status": (
            "passed" if all(checks.values()) and flac_metrics is not None
            else "python_pass_solver_not_run"
            if all(checks.values())
            else "failed"
        ),
        "checks": checks,
        "curve_improvement_fraction": curve_improvement,
        "stress_response_improvement_fraction": response_improvement,
        "discovered_curve_rmse_mpa": discovered_curve_rmse,
        "scientific_scope": (
            "This controlled chain validates transfer of the frozen SR03 edit "
            "to a path-dependent weak-plane law and FLAC3D state semantics. It "
            "does not constitute field calibration or independent discovery."
        ),
    }


def run(
    config_path: str,
    run_id: str,
    *,
    skip_flac3d: bool = False,
    resume: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> Path:
    config = read_yaml(config_path)
    run_dir = ensure_run_dir(run_id)
    report_path = run_dir / "reports" / "structural_softening_application_gate.json"
    outputs = [
        report_path,
        run_dir / "metrics" / "structural_softening_python_metrics.csv",
        run_dir / "data" / "structural_softening_python_trace.csv",
    ]
    if should_skip(run_dir, STAGE, outputs, resume, force):
        progress_message(run_dir, "Structural softening validation restored", verbose)
        return run_dir

    artifact_path = config["frozen_artifact"]
    frozen = load_frozen_structural_softening(artifact_path)
    laws = build_application_laws(frozen, config["application"])
    progress_message(
        run_dir,
        "Frozen structural softening loaded",
        verbose,
        edit=f"{frozen.location}:{frozen.edit_type}:{frozen.family_id}",
    )
    trace, curve, python_metrics = _python_evidence(config, laws)
    trace.to_csv(run_dir / "data" / "structural_softening_python_trace.csv", index=False)
    curve.to_csv(run_dir / "data" / "structural_softening_curves.csv", index=False)
    write_table_bundle(
        python_metrics,
        run_dir / "metrics" / "structural_softening_python_metrics",
    )
    write_json(
        run_dir / "formulas" / "frozen_structural_softening_laws.json",
        {
            "source_artifact": str(artifact_path),
            "source_checkpoint_sha256": frozen.source_checkpoint_sha256,
            "laws": [
                {
                    "law_id": law.law_id,
                    "label": law.label,
                    "family": law.family,
                    "formula": law.formula,
                    "provenance": law.provenance,
                }
                for law in laws
            ],
        },
    )
    progress_message(
        run_dir,
        "Python material-point comparison complete",
        verbose,
        curve_improvement=f"{_improvement(python_metrics, 'curve_rmse_mpa'):.3%}",
        response_improvement=f"{_improvement(python_metrics, 'stress_response_rmse_mpa'):.3%}",
    )

    flac_metrics: pd.DataFrame | None = None
    transcript: str | None = None
    if not skip_flac3d:
        work_dir = ROOT / "flac3d" / "runs" / run_id
        work_dir.mkdir(parents=True, exist_ok=True)
        data_file = work_dir / "structural_softening_single_zone.dat"
        data_file.write_text(
            render_flac3d_structural_softening_data_file(config, laws),
            encoding="ascii",
        )
        cases = structural_softening_cases(config)
        raw_paths = [
            work_dir / structural_softening_output_name(law, case)
            for law in laws
            for case in cases
        ]
        if force:
            for path in raw_paths:
                path.unlink(missing_ok=True)
        progress_message(
            run_dir,
            "FLAC3D single-zone replay started",
            verbose,
            units=len(raw_paths),
        )
        result = run_flac3d_data_file(
            config["flac3d"]["executable"],
            data_file,
            work_dir,
            timeout_seconds=float(config["flac3d"]["timeout_seconds"]),
            startup_timeout_seconds=float(
                config["flac3d"]["startup_timeout_seconds"]
            ),
            verbose=verbose,
            progress_only=True,
            completion_check=lambda: all(
                path.exists() and path.stat().st_size > 0 for path in raw_paths
            ),
        )
        transcript = str(result.transcript_path)
        parameters = application_parameters(config)
        rows = []
        for law in laws:
            kappa, cohesion = law.table(
                maximum_damage=float(config["table"]["maximum_damage"]),
                point_count=int(config["table"]["point_count"]),
            )
            for case in cases:
                expected = len(path_shear_increments(config["paths"][case.path]))
                raw = load_subi_case(
                    work_dir / structural_softening_output_name(law, case), expected
                )
                comparison, metrics = compare_subi_case(
                    raw,
                    SUBIValidationCase(
                        case.case_id, case.beta_deg, case.pressure_mpa, case.path
                    ),
                    parameters,
                    kappa,
                    cohesion,
                )
                comparison.insert(0, "law_id", law.law_id)
                comparison.to_csv(
                    run_dir
                    / "data"
                    / f"flac3d_{law.law_id}_{case.case_id}_comparison.csv",
                    index=False,
                )
                rows.append({"law_id": law.law_id, **metrics})
        flac_metrics = pd.DataFrame(rows)
        write_table_bundle(
            flac_metrics,
            run_dir / "metrics" / "structural_softening_flac3d_metrics",
        )
        progress_message(run_dir, "FLAC3D single-zone replay complete", verbose)

    report = _gate(config, python_metrics, flac_metrics)
    report.update(
        {
            "protocol": config["protocol"],
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "flac3d_transcript": transcript,
            "frozen_artifact": str(artifact_path),
        }
    )
    write_json(report_path, report)
    mark_done(run_dir, STAGE, {"gate_status": report["status"]})
    progress_message(
        run_dir,
        "Structural softening application validation complete",
        verbose,
        status=report["status"],
        output=run_dir,
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay the frozen SR03 softening revision in Python and FLAC3D."
    )
    parser.add_argument(
        "--config", default="configs/structural_softening_flac3d_validation.yaml"
    )
    parser.add_argument(
        "--run-id", default="structural_softening_flac3d_validation_v1"
    )
    parser.add_argument("--skip-flac3d", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.force and args.resume:
        parser.error("--force and --resume are mutually exclusive.")
    run(
        args.config,
        args.run_id,
        skip_flac3d=args.skip_flac3d,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

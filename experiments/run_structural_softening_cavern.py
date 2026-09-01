from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401

try:
    from generate_flac3d_cavern_v2_mesh import generate
    from run_flac3d_cavern_pilot import (
        build_pilot_cases,
        run_cavern_cases,
    )
except ModuleNotFoundError:
    from experiments.generate_flac3d_cavern_v2_mesh import generate
    from experiments.run_flac3d_cavern_pilot import (
        build_pilot_cases,
        run_cavern_cases,
    )

from asrc.constitutive.structural_softening_application import (
    build_application_laws,
    load_frozen_structural_softening,
)
from asrc.utils.io import ensure_run_dir, read_yaml, write_json, write_table_bundle
from asrc.utils.progress import progress_message


ROOT = Path(__file__).resolve().parents[1]
V3_TEMPLATE = ROOT / "flac3d" / "cavern_v2" / "cavern_case_v2.dat.in"
RESPONSE_COLUMNS = [
    "response_max_m",
    "response_crown_m",
    "response_wall_convergence_m",
    "response_invert_m",
]


def _existing_mesh(config: dict[str, Any]) -> Path:
    path = (
        ROOT
        / "flac3d"
        / "cavern_v2"
        / "generated"
        / f"{config['mesh']['name']}.f3grid"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Reusable cavern mesh is missing: {path}")
    return path


def evaluate_cavern_propagation(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required_models = {
        "softening_baseline_linear",
        "softening_discovered_exponential",
        "softening_oracle_exponential",
    }
    if set(frame["model"]) != required_models:
        raise ValueError("Cavern propagation output does not contain all three laws.")
    indexed = frame.set_index(["scenario_id", "stage", "model"])
    errors = []
    for method in ("softening_baseline_linear", "softening_discovered_exponential"):
        method_frame = indexed.xs(method, level="model").sort_index()
        oracle = indexed.xs("softening_oracle_exponential", level="model").sort_index()
        difference = np.concatenate(
            [
                method_frame[column].to_numpy(float)
                - oracle[column].to_numpy(float)
                for column in RESPONSE_COLUMNS
            ]
        )
        errors.append(
            {
                "model": method,
                "multi_response_rmse_m": float(np.sqrt(np.mean(difference**2))),
                "maximum_absolute_response_error_m": float(np.max(np.abs(difference))),
            }
        )
    metrics = pd.DataFrame(errors)
    values = metrics.set_index("model")["multi_response_rmse_m"]
    baseline_error = float(values["softening_baseline_linear"])
    discovered_error = float(values["softening_discovered_exponential"])
    improvement = 1.0 - discovered_error / max(baseline_error, np.finfo(float).eps)
    oracle = indexed.xs("softening_oracle_exponential", level="model")
    baseline = indexed.xs("softening_baseline_linear", level="model")
    oracle_vector = np.concatenate(
        [oracle[column].to_numpy(float) for column in RESPONSE_COLUMNS]
    )
    baseline_vector = np.concatenate(
        [baseline[column].to_numpy(float) for column in RESPONSE_COLUMNS]
    )
    signal = float(np.linalg.norm(oracle_vector - baseline_vector)) / max(
        float(np.linalg.norm(oracle_vector)), np.finfo(float).eps
    )
    all_converged = bool(
        (
            frame["ratio_selected"].to_numpy(float)
            <= float(config["response"]["convergence_ratio_average"])
        ).all()
    )
    limits = config["application_gate"]
    required_plastic = int(
        config.get("qualification_protocol", {}).get(
            "minimum_plastic_zone_count", 1
        )
    )
    plastic_by_model = {
        str(model): int(group["plastic_zone_count"].max())
        for model, group in frame.groupby("model", sort=True)
    }
    each_model_plastic = all(
        plastic_by_model[model] >= required_plastic for model in required_models
    )
    checks = {
        "all_stages_converged": all_converged,
        "softening_signal_is_resolved": signal
        >= float(limits["minimum_oracle_baseline_signal_fraction"]),
        "discovered_law_improves_baseline": improvement
        >= float(limits["minimum_discovered_response_improvement_fraction"]),
        "plasticity_is_activated": each_model_plastic
        if bool(limits.get("require_any_plastic_zone", True))
        else True,
    }
    return metrics, {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "oracle_baseline_signal_fraction": signal,
        "discovered_response_improvement_fraction": improvement,
        "maximum_plastic_zone_count_by_model": plastic_by_model,
        "minimum_plastic_zone_count": required_plastic,
        "response_metrics_admissible": all_converged,
        "scientific_scope": (
            "The cavern calculation propagates a frozen controlled softening "
            "revision to staged engineering responses. It is not field "
            "calibration and does not provide additional discovery evidence."
        ),
    }


def run(
    config_path: Path,
    run_id: str,
    *,
    reuse_mesh: bool,
    resume: bool,
    force: bool,
    verbose: bool,
) -> Path:
    config_path = config_path.resolve()
    config = read_yaml(config_path)
    if reuse_mesh:
        mesh_path = _existing_mesh(config)
    else:
        mesh_path, _ = generate(config_path)
    print(f"Structural-softening cavern mesh ready: {mesh_path}", flush=True)

    base_cases, material = build_pilot_cases(config)
    if len(base_cases) != 1:
        raise ValueError("The controlled cavern propagation requires one scenario.")
    softening = config["structural_softening"]
    frozen = load_frozen_structural_softening(softening["frozen_artifact"])
    laws = build_application_laws(frozen, softening["application"])
    table = softening["table"]
    cases = [
        replace(
            base_cases[0],
            model=f"softening_{law.law_id}",
            softening_law=law,
            softening_maximum_damage=float(table["maximum_damage"]),
            softening_table_point_count=int(table["point_count"]),
        )
        for law in laws
    ]
    solver = config["solver"]
    combined_path = run_cavern_cases(
        cases,
        material,
        run_id=run_id,
        config_path=config_path,
        flac3d_executable=Path(str(solver["executable"])),
        template_path=V3_TEMPLATE,
        timeout_seconds=float(solver["timeout_seconds"]),
        convergence_limit=cases[0].convergence_limit,
        admission_gate=dict(config["admission_gate"]),
        study_status=str(config["status"]),
        max_case_attempts=int(solver.get("max_case_attempts", 1)),
        case_retry_delay_seconds=float(solver.get("case_retry_delay_seconds", 5.0)),
        resume=resume,
        force=force,
        verbose=verbose,
    )
    frame = pd.read_csv(combined_path)
    metrics, report = evaluate_cavern_propagation(frame, config)
    output_dir = ensure_run_dir(run_id)
    write_table_bundle(
        metrics,
        output_dir / "metrics" / "structural_softening_cavern_metrics",
    )
    report.update(
        {
            "source_dataset": str(combined_path),
            "frozen_artifact": str(softening["frozen_artifact"]),
            "source_checkpoint_sha256": frozen.source_checkpoint_sha256,
        }
    )
    write_json(
        output_dir / "reports" / "structural_softening_cavern_gate.json",
        report,
    )
    progress_message(
        output_dir,
        "Structural softening cavern propagation complete",
        verbose,
        status=report["status"],
        improvement=f"{report['discovered_response_improvement_fraction']:.3%}",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propagate the frozen SR03 softening revision through a staged cavern."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/flac3d_structural_softening_cavern.yaml"),
    )
    parser.add_argument(
        "--run-id", default="structural_softening_cavern_pilot_v1"
    )
    parser.add_argument("--reuse-mesh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.force and args.resume:
        parser.error("--force and --resume are mutually exclusive.")
    run(
        args.config,
        args.run_id,
        reuse_mesh=args.reuse_mesh,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
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
    from run_flac3d_cavern_pilot import build_pilot_cases, run_cavern_cases
    from run_structural_softening_cavern import (
        RESPONSE_COLUMNS,
        V3_TEMPLATE,
        _existing_mesh,
        evaluate_cavern_propagation,
    )
except ModuleNotFoundError:
    from experiments.generate_flac3d_cavern_v2_mesh import generate
    from experiments.run_flac3d_cavern_pilot import (
        build_pilot_cases,
        run_cavern_cases,
    )
    from experiments.run_structural_softening_cavern import (
        RESPONSE_COLUMNS,
        V3_TEMPLATE,
        _existing_mesh,
        evaluate_cavern_propagation,
    )

from asrc.constitutive.structural_softening_application import (
    StructuralSofteningLaw,
    build_application_laws,
    load_frozen_structural_softening,
)
from asrc.utils.io import (
    ensure_run_dir,
    read_json,
    read_yaml,
    runs_root,
    write_json,
    write_table_bundle,
)
from asrc.utils.progress import progress_message


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/flac3d_structural_softening_cavern.yaml")
MODEL_BY_LAW = {
    "baseline_linear": "softening_baseline_linear",
    "discovered_exponential": "softening_discovered_exponential",
    "oracle_exponential": "softening_oracle_exponential",
}


def _laws(config: dict[str, Any]) -> dict[str, StructuralSofteningLaw]:
    softening = config["structural_softening"]
    frozen = load_frozen_structural_softening(softening["frozen_artifact"])
    return {
        law.law_id: law
        for law in build_application_laws(frozen, softening["application"])
    }


def _base_case(config: dict[str, Any]) -> tuple[Any, Any]:
    cases, material = build_pilot_cases(config)
    if len(cases) != 1:
        raise ValueError("The qualification protocol requires one base scenario.")
    return cases[0], material


def _case(
    base: Any,
    law: StructuralSofteningLaw,
    config: dict[str, Any],
    *,
    stress_mpa: float,
    scenario_id: str,
) -> Any:
    table = config["structural_softening"]["table"]
    return replace(
        base,
        scenario_id=scenario_id,
        model=MODEL_BY_LAW[law.law_id],
        major_stress_MPa=float(stress_mpa),
        softening_law=law,
        softening_maximum_damage=float(table["maximum_damage"]),
        softening_table_point_count=int(table["point_count"]),
    )


def _run_cases(
    cases: list[Any],
    material: Any,
    config: dict[str, Any],
    config_path: Path,
    run_id: str,
    *,
    resume: bool,
    force: bool,
    verbose: bool,
) -> Path:
    solver = config["solver"]
    return run_cavern_cases(
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


def evaluate_oracle_screen(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    limit = float(config["response"]["convergence_ratio_average"])
    protocol = config["qualification_protocol"]
    required_plastic = int(protocol["minimum_plastic_zone_count"])
    rows = []
    for scenario_id, group in frame.groupby("scenario_id", sort=False):
        ordered = group.sort_values("stage")
        expected_stages = list(range(1, 7))
        observed_stages = ordered["stage"].astype(int).tolist()
        converged = bool(
            observed_stages == expected_stages
            and (ordered["ratio_selected"].to_numpy(float) <= limit).all()
        )
        max_plastic = int(ordered["plastic_zone_count"].max())
        rows.append(
            {
                "scenario_id": scenario_id,
                "major_stress_mpa": float(ordered["major_stress_mpa"].iloc[0]),
                "all_stages_complete": observed_stages == expected_stages,
                "all_stages_converged": converged,
                "maximum_ratio_selected": float(ordered["ratio_selected"].max()),
                "maximum_plastic_zone_count": max_plastic,
                "maximum_response_m": float(ordered["response_max_m"].max()),
                "qualified": converged and max_plastic >= required_plastic,
            }
        )
    metrics = pd.DataFrame(rows).sort_values("major_stress_mpa")
    qualified = metrics.loc[metrics["qualified"]]
    selected = (
        float(qualified["major_stress_mpa"].max()) if len(qualified) else None
    )
    return metrics, {
        "status": "passed" if selected is not None else "failed_no_qualified_load",
        "selected_stress_mpa": selected,
        "selection_rule": protocol["selection_rule"],
        "discovered_law_evaluated": False,
        "checks": {
            "at_least_one_fully_converged_oracle_case_with_plasticity": selected
            is not None
        },
    }


def _pair_signal(frame: pd.DataFrame) -> float:
    indexed = frame.set_index(["scenario_id", "stage", "model"]).sort_index()
    baseline = indexed.xs("softening_baseline_linear", level="model")
    oracle = indexed.xs("softening_oracle_exponential", level="model")
    baseline_vector = np.concatenate(
        [baseline[column].to_numpy(float) for column in RESPONSE_COLUMNS]
    )
    oracle_vector = np.concatenate(
        [oracle[column].to_numpy(float) for column in RESPONSE_COLUMNS]
    )
    return float(np.linalg.norm(oracle_vector - baseline_vector)) / max(
        float(np.linalg.norm(oracle_vector)), np.finfo(float).eps
    )


def evaluate_pair_qualification(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    expected = {"softening_baseline_linear", "softening_oracle_exponential"}
    if set(frame["model"]) != expected:
        raise ValueError("Pair qualification requires baseline and oracle only.")
    expected_stages = list(range(1, 7))
    complete = all(
        group["stage"].astype(int).sort_values().tolist() == expected_stages
        for _, group in frame.groupby("model", sort=False)
    )
    limit = float(config["response"]["convergence_ratio_average"])
    converged = bool(
        complete and (frame["ratio_selected"].to_numpy(float) <= limit).all()
    )
    signal = _pair_signal(frame) if complete else 0.0
    protocol = config["qualification_protocol"]
    threshold = float(protocol["minimum_pair_signal_fraction"])
    required_plastic = int(protocol["minimum_plastic_zone_count"])
    plastic_by_model = {
        str(model): int(group["plastic_zone_count"].max())
        for model, group in frame.groupby("model", sort=True)
    }
    each_model_plastic = all(
        plastic_by_model[model] >= required_plastic for model in expected
    )
    checks = {
        "both_models_have_all_six_stages": complete,
        "all_stages_converged": converged,
        "pair_signal_is_resolved": complete and signal >= threshold,
        "plasticity_is_activated": each_model_plastic,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "multi_response_signal_fraction": signal,
        "minimum_signal_fraction": threshold,
        "maximum_plastic_zone_count_by_model": plastic_by_model,
        "minimum_plastic_zone_count": required_plastic,
        "discovered_law_evaluated": False,
    }


def _source_report(run_id: str, filename: str) -> dict[str, Any]:
    path = runs_root() / run_id / "reports" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required qualification report is missing: {path}")
    return read_json(path)


def _source_dataset(run_id: str) -> Path:
    path = ROOT / "flac3d" / "runs" / run_id / "flac3d_cavern_pilot_results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required FLAC3D dataset is missing: {path}")
    return path


def _prepare_mesh(
    config: dict[str, Any], config_path: Path, reuse_mesh: bool
) -> Path:
    path = _existing_mesh(config) if reuse_mesh else generate(config_path)[0]
    print(f"Structural-softening qualification mesh ready: {path}", flush=True)
    return path


def _protocol_fingerprint(
    config: dict[str, Any], config_path: Path, mesh_path: Path
) -> dict[str, str]:
    artifact_path = Path(str(config["structural_softening"]["frozen_artifact"]))
    if not artifact_path.is_absolute():
        artifact_path = ROOT / artifact_path
    return {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "frozen_artifact_sha256": hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest(),
    }


def _require_matching_protocol(
    source: dict[str, Any], current: dict[str, str]
) -> None:
    previous = source.get("protocol_fingerprint")
    if previous != current:
        raise ValueError(
            "Qualification protocol fingerprint differs from the preceding "
            "stage; do not continue with changed config, mesh, or artifact."
        )


def run_screen(
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
    mesh_path = _prepare_mesh(config, config_path, reuse_mesh)
    fingerprint = _protocol_fingerprint(config, config_path, mesh_path)
    base, material = _base_case(config)
    oracle = _laws(config)["oracle_exponential"]
    stresses = [
        float(value)
        for value in config["qualification_protocol"]["stress_candidates_mpa"]
    ]
    cases = [
        _case(
            base,
            oracle,
            config,
            stress_mpa=stress,
            scenario_id=f"oracle_screen_s{stress:g}",
        )
        for stress in stresses
    ]
    dataset = _run_cases(
        cases,
        material,
        config,
        config_path,
        run_id,
        resume=resume,
        force=force,
        verbose=verbose,
    )
    frame = pd.read_csv(dataset)
    # The solver export does not repeat the load magnitude, so attach the
    # pre-registered case value by scenario for the qualification audit.
    stress_by_scenario = {case.scenario_id: case.major_stress_MPa for case in cases}
    frame["major_stress_mpa"] = frame["scenario_id"].map(stress_by_scenario)
    metrics, report = evaluate_oracle_screen(frame, config)
    output = ensure_run_dir(run_id)
    write_table_bundle(metrics, output / "metrics" / "oracle_load_screen")
    report.update(
        {
            "stage": "oracle_load_screen",
            "source_dataset": str(dataset),
            "stress_candidates_mpa": stresses,
            "protocol_fingerprint": fingerprint,
        }
    )
    write_json(output / "reports" / "oracle_load_screen_gate.json", report)
    progress_message(
        output,
        "Oracle load screen complete",
        verbose,
        status=report["status"],
        selected_stress_mpa=report["selected_stress_mpa"],
    )
    return output


def run_pair(
    config_path: Path,
    run_id: str,
    screen_run_id: str,
    *,
    reuse_mesh: bool,
    resume: bool,
    force: bool,
    verbose: bool,
) -> Path:
    screen = _source_report(screen_run_id, "oracle_load_screen_gate.json")
    if screen.get("status") != "passed" or screen.get("selected_stress_mpa") is None:
        raise ValueError("Oracle load screen did not freeze an admissible stress.")
    stress = float(screen["selected_stress_mpa"])
    config_path = config_path.resolve()
    config = read_yaml(config_path)
    mesh_path = _prepare_mesh(config, config_path, reuse_mesh)
    fingerprint = _protocol_fingerprint(config, config_path, mesh_path)
    _require_matching_protocol(screen, fingerprint)
    base, material = _base_case(config)
    laws = _laws(config)
    cases = [
        _case(
            base,
            laws[law_id],
            config,
            stress_mpa=stress,
            scenario_id=f"pair_qualification_s{stress:g}",
        )
        for law_id in ("baseline_linear", "oracle_exponential")
    ]
    dataset = _run_cases(
        cases,
        material,
        config,
        config_path,
        run_id,
        resume=resume,
        force=force,
        verbose=verbose,
    )
    frame = pd.read_csv(dataset)
    report = evaluate_pair_qualification(frame, config)
    output = ensure_run_dir(run_id)
    report.update(
        {
            "stage": "baseline_oracle_pair_qualification",
            "selected_stress_mpa": stress,
            "source_screen_run_id": screen_run_id,
            "source_dataset": str(dataset),
            "protocol_fingerprint": fingerprint,
        }
    )
    write_json(output / "reports" / "baseline_oracle_pair_gate.json", report)
    progress_message(
        output,
        "Baseline-oracle pair qualification complete",
        verbose,
        status=report["status"],
        signal=f"{report['multi_response_signal_fraction']:.3%}",
    )
    return output


def run_confirmation(
    config_path: Path,
    run_id: str,
    pair_run_id: str,
    *,
    reuse_mesh: bool,
    resume: bool,
    force: bool,
    verbose: bool,
) -> Path:
    pair = _source_report(pair_run_id, "baseline_oracle_pair_gate.json")
    if pair.get("status") != "passed":
        raise ValueError("Baseline-oracle pair qualification did not pass.")
    stress = float(pair["selected_stress_mpa"])
    config_path = config_path.resolve()
    config = read_yaml(config_path)
    mesh_path = _prepare_mesh(config, config_path, reuse_mesh)
    fingerprint = _protocol_fingerprint(config, config_path, mesh_path)
    _require_matching_protocol(pair, fingerprint)
    base, material = _base_case(config)
    discovered = _laws(config)["discovered_exponential"]
    scenario = f"pair_qualification_s{stress:g}"
    cases = [
        _case(
            base,
            discovered,
            config,
            stress_mpa=stress,
            scenario_id=scenario,
        )
    ]
    dataset = _run_cases(
        cases,
        material,
        config,
        config_path,
        run_id,
        resume=resume,
        force=force,
        verbose=verbose,
    )
    pair_frame = pd.read_csv(_source_dataset(pair_run_id))
    discovered_frame = pd.read_csv(dataset)
    combined = pd.concat([pair_frame, discovered_frame], ignore_index=True)
    metrics, report = evaluate_cavern_propagation(combined, config)
    output = ensure_run_dir(run_id)
    combined.to_csv(output / "data" / "qualified_cavern_three_law_results.csv", index=False)
    write_table_bundle(
        metrics,
        output / "metrics" / "qualified_structural_softening_cavern_metrics",
    )
    report.update(
        {
            "stage": "discovered_law_confirmation",
            "selected_stress_mpa": stress,
            "source_pair_run_id": pair_run_id,
            "pair_dataset": str(_source_dataset(pair_run_id)),
            "discovered_dataset": str(dataset),
            "protocol_fingerprint": fingerprint,
        }
    )
    write_json(
        output / "reports" / "qualified_structural_softening_cavern_gate.json",
        report,
    )
    progress_message(
        output,
        "Discovered-law cavern confirmation complete",
        verbose,
        status=report["status"],
        improvement=f"{report['discovered_response_improvement_fraction']:.3%}",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run blinded cavern load screening, pair qualification, and confirmation."
    )
    parser.add_argument("--stage", choices=("screen", "pair", "confirm"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--screen-run-id")
    parser.add_argument("--pair-run-id")
    parser.add_argument("--reuse-mesh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.force and args.resume:
        parser.error("--force and --resume are mutually exclusive.")
    common = {
        "reuse_mesh": args.reuse_mesh,
        "resume": args.resume,
        "force": args.force,
        "verbose": args.verbose,
    }
    if args.stage == "screen":
        run_screen(args.config, args.run_id, **common)
    elif args.stage == "pair":
        if not args.screen_run_id:
            parser.error("--screen-run-id is required for --stage pair.")
        run_pair(args.config, args.run_id, args.screen_run_id, **common)
    else:
        if not args.pair_run_id:
            parser.error("--pair-run-id is required for --stage confirm.")
        run_confirmation(args.config, args.run_id, args.pair_run_id, **common)


if __name__ == "__main__":
    main()

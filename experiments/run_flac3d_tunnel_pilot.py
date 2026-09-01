from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from asrc.solvers.flac3d import FLAC3DError, run_flac3d_data_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTABLE = Path(r"flac3d700_console.exe")
DEFAULT_TEMPLATE = ROOT / "flac3d" / "tunnel_pilot" / "tunnel_case.dat.in"
DEFAULT_ANGLE_PAIRS = ((15.0, 0.0), (45.0, 30.0), (75.0, 60.0))
MODELS = ("baseline", "reference")


@dataclass(frozen=True)
class MeshSpec:
    name: str
    circumferential_zones: int
    radial_zones: int
    radial_ratio: float


MESHES = {
    "coarse": MeshSpec("coarse", 8, 14, 1.18),
    "medium": MeshSpec("medium", 12, 20, 1.12),
    "fine": MeshSpec("fine", 18, 30, 1.077),
    "very_fine": MeshSpec("very_fine", 24, 40, 1.057),
}


@dataclass(frozen=True)
class TunnelCase:
    beta_deg: float
    psi_deg: float
    model: str
    mesh: MeshSpec = MESHES["medium"]

    @property
    def case_id(self) -> str:
        def angle_slug(value: float) -> str:
            if value.is_integer():
                return f"{int(value):03d}"
            return f"{value:05.1f}".replace(".", "p")

        return (
            f"b{angle_slug(self.beta_deg)}_p{angle_slug(self.psi_deg)}_"
            f"{self.model}_{self.mesh.name}"
        )


def _material_block(model: str, beta_deg: float) -> str:
    common = """zone property bulk 1.0e10 shear 7.0e9 cohesion 2.0e5
zone property friction 40 dilation 0 tension 2.4e5"""
    if model == "baseline":
        return "zone cmodel assign mohr-coulomb\n" + common
    if model == "reference":
        return (
            "zone cmodel assign ubiquitous-joint\n"
            + common
            + f"\nzone property dip {beta_deg:.8g} dip-direction 90 joint-cohesion 1.0e5\n"
            "zone property joint-friction 30 joint-dilation 0 joint-tension 2.0e5"
        )
    raise ValueError(f"Unknown FLAC3D tunnel model: {model}")


def render_tunnel_case(template: str, case: TunnelCase) -> str:
    replacements = {
        "__CASE_ID__": case.case_id,
        "__BETA_DEG__": f"{case.beta_deg:.8g}",
        "__PSI_DEG__": f"{case.psi_deg:.8g}",
        "__MODEL_NAME__": case.model,
        "__MESH_NAME__": case.mesh.name,
        "__CIRCUMFERENTIAL_ZONES__": str(case.mesh.circumferential_zones),
        "__RADIAL_ZONES__": str(case.mesh.radial_zones),
        "__RADIAL_RATIO__": f"{case.mesh.radial_ratio:.8g}",
        "__MATERIAL_BLOCK__": _material_block(case.model, case.beta_deg),
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    unresolved = sorted({part for part in rendered.split() if part.startswith("__")})
    if unresolved:
        raise ValueError(f"Unresolved FLAC3D template tokens: {unresolved}")
    return rendered


def _parse_angle_pair(value: str) -> tuple[float, float]:
    try:
        beta, psi = (float(item.strip()) for item in value.split(",", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Angle pairs must use BETA,PSI, for example 45,30.") from exc
    if not (0.0 <= beta <= 90.0 and 0.0 <= psi <= 90.0):
        raise argparse.ArgumentTypeError("BETA and PSI must both be in [0, 90] degrees.")
    return beta, psi


def _read_case_result(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise FLAC3DError(f"Expected one FLAC3D result row in {path}, found {len(rows)}.")
    return rows[0]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _validate_resume_manifest(path: Path, expected: dict[str, object]) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"Cannot resume because the manifest is missing: {path}")
    existing = json.loads(path.read_text(encoding="utf-8"))
    audit_keys = sorted(set(expected) - {"created_utc"})
    changed = [key for key in audit_keys if existing.get(key) != expected.get(key)]
    if changed:
        raise SystemExit(
            "Resume configuration differs from the saved manifest for: "
            + ", ".join(changed)
            + ". Use a new --run-id or --force."
        )
    return existing


def _write_mesh_sensitivity_summary(run_dir: Path, rows: list[dict[str, str]]) -> Path | None:
    if len({row["mesh"] for row in rows}) < 2:
        return None
    response_columns = ("response_max_m", "response_crown_m", "response_springline_m")
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["beta_deg"], row["psi_deg"], row["model"])
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, str | float]] = []
    for group in grouped.values():
        finest = max(group, key=lambda item: int(item["radial_zones"]))
        for row in sorted(group, key=lambda item: int(item["radial_zones"])):
            summary: dict[str, str | float] = dict(row)
            for column in response_columns:
                reference = float(finest[column])
                relative = abs(float(row[column]) - reference) / max(abs(reference), 1e-15) * 100.0
                summary[f"{column}_difference_vs_finest_pct"] = relative
            summary_rows.append(summary)

    output_path = run_dir / "mesh_sensitivity_summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    return output_path


def _prepare_run_dir(run_dir: Path, force: bool) -> None:
    runs_root = (ROOT / "flac3d" / "runs").resolve()
    resolved = run_dir.resolve()
    if runs_root not in resolved.parents:
        raise SystemExit(f"Refusing to modify a run directory outside {runs_root}: {resolved}")
    if resolved.exists() and force:
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def run_tunnel_cases(
    cases: list[TunnelCase],
    *,
    run_id: str,
    flac3d_executable: Path = DEFAULT_EXECUTABLE,
    template_path: Path = DEFAULT_TEMPLATE,
    timeout_seconds: float = 900.0,
    resume: bool = False,
    force: bool = False,
    verbose: bool = False,
    manifest_extra: dict[str, object] | None = None,
    results_filename: str = "tunnel_pilot_results.csv",
) -> Path:
    if force and resume:
        raise ValueError("force and resume are mutually exclusive.")
    if not cases:
        raise ValueError("At least one FLAC3D tunnel case is required.")
    run_dir = ROOT / "flac3d" / "runs" / run_id
    if run_dir.exists() and not (force or resume):
        raise SystemExit(f"Run directory already exists: {run_dir}. Use --resume or --force.")
    _prepare_run_dir(run_dir, force)

    template_path = template_path.resolve()
    template = template_path.read_text(encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "flac3d_executable": str(flac3d_executable.resolve()),
        "template": str(template_path),
        "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "execution": "serial",
        "cases": [asdict(case) | {"case_id": case.case_id} for case in cases],
        **(manifest_extra or {}),
    }
    manifest_path = run_dir / "manifest.json"
    if resume:
        manifest = _validate_resume_manifest(manifest_path, manifest)
    else:
        _write_json(manifest_path, manifest)

    completed_rows: list[dict[str, str]] = []
    print(f"FLAC3D tunnel run started: {run_dir} ({len(cases)} cases, serial)", flush=True)
    for index, case in enumerate(cases, start=1):
        case_dir = run_dir / case.case_id
        result_csv = case_dir / "tunnel_case_result.csv"
        if resume and result_csv.is_file():
            row = _read_case_result(result_csv)
            completed_rows.append(row)
            print(f"[{index}/{len(cases)}] resumed {case.case_id}", flush=True)
            continue

        case_dir.mkdir(parents=True, exist_ok=True)
        data_file = case_dir / "tunnel_case.dat"
        data_file.write_text(render_tunnel_case(template, case), encoding="utf-8")
        print(f"[{index}/{len(cases)}] solving {case.case_id}", flush=True)

        def case_output_complete() -> bool:
            if not result_csv.is_file() or not (case_dir / f"{case.case_id}.sav").is_file():
                return False
            try:
                _read_case_result(result_csv)
            except (FLAC3DError, OSError, ValueError):
                return False
            return True

        try:
            result = run_flac3d_data_file(
                flac3d_executable,
                data_file,
                case_dir,
                timeout_seconds=timeout_seconds,
                verbose=verbose,
                completion_check=case_output_complete,
            )
        except FLAC3DError as exc:
            raise SystemExit(f"{case.case_id} failed: {exc}") from exc
        row = _read_case_result(result_csv)
        completed_rows.append(row)
        print(
            f"[{index}/{len(cases)}] complete {case.case_id} | "
            f"max_disp={float(row['response_max_m']):.6g} m | "
            f"ratio={float(row['ratio_local']):.3g} | {result.elapsed_seconds:.1f}s",
            flush=True,
        )
        _write_json(
            run_dir / "progress.json",
            {"completed": [item["case_id"] for item in completed_rows], "total": len(cases)},
        )

    combined_path = run_dir / results_filename
    fieldnames = list(completed_rows[0])
    with combined_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(completed_rows)
    sensitivity_path = _write_mesh_sensitivity_summary(run_dir, completed_rows)
    print(f"FLAC3D tunnel run complete: {combined_path}", flush=True)
    if sensitivity_path:
        print(f"Mesh sensitivity summary: {sensitivity_path}", flush=True)
    return combined_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the serial FLAC3D 7 tunnel pilot matrix.")
    parser.add_argument("--run-id", default="tunnel_pilot_v1")
    parser.add_argument("--flac3d-exe", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--angle-pair",
        action="append",
        type=_parse_angle_pair,
        help="Repeatable BETA,PSI pair. Defaults to 15,0; 45,30; 75,60.",
    )
    parser.add_argument("--model", action="append", choices=MODELS)
    parser.add_argument(
        "--mesh",
        action="append",
        choices=tuple(MESHES),
        help="Repeatable mesh level. Defaults to medium.",
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.force and args.resume:
        parser.error("--force and --resume are mutually exclusive.")

    angle_pairs = tuple(args.angle_pair or DEFAULT_ANGLE_PAIRS)
    models = tuple(dict.fromkeys(args.model or MODELS))
    meshes = tuple(MESHES[name] for name in dict.fromkeys(args.mesh or ("medium",)))
    cases = [
        TunnelCase(beta, psi, model, mesh)
        for beta, psi in angle_pairs
        for model in models
        for mesh in meshes
    ]
    run_tunnel_cases(
        cases,
        run_id=args.run_id,
        flac3d_executable=args.flac3d_exe,
        template_path=args.template,
        timeout_seconds=args.timeout,
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
        manifest_extra={"run_type": "pilot"},
    )


if __name__ == "__main__":
    main()

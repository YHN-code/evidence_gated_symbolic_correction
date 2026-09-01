from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from asrc.constitutive.structural_softening_application import StructuralSofteningLaw
from asrc.solvers.flac3d import (
    FLAC3DError,
    FLAC3DProcessCleanupError,
    run_flac3d_data_file,
)
from asrc.utils.io import read_yaml

try:
    from run_flac3d_tunnel_pilot import _prepare_run_dir, _validate_resume_manifest, _write_json
except ModuleNotFoundError:
    from experiments.run_flac3d_tunnel_pilot import (
        _prepare_run_dir,
        _validate_resume_manifest,
        _write_json,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "flac3d_hydropower_cavern_pilot.yaml"
DEFAULT_TEMPLATE = ROOT / "flac3d" / "cavern_pilot" / "cavern_case.dat.in"
MODELS = ("elastic", "baseline", "reference", "anisotropic")
_TOKEN = re.compile(r"__[A-Z0-9_]+__")


@dataclass(frozen=True)
class CavernGeometry:
    cavern_width_m: float
    cavern_height_m: float
    domain_half_width_m: float
    domain_half_height_m: float
    plane_strain_thickness_m: float
    excavation_stages: int


@dataclass(frozen=True)
class CavernMesh:
    name: str
    axial_zones: int
    circumferential_zones: int
    radial_zones: int
    radial_ratio: float


@dataclass(frozen=True)
class CavernSupport:
    installation_delay_stages: int
    thickness_m: float
    young_Pa: float
    poisson: float
    density_kg_m3: float


@dataclass(frozen=True)
class CavernMaterial:
    bulk_Pa: float
    shear_Pa: float
    cohesion_Pa: float
    friction_deg: float
    dilation_deg: float
    tension_Pa: float
    joint_cohesion_Pa: float
    joint_friction_deg: float
    joint_dilation_deg: float
    joint_tension_Pa: float
    young_plane_Pa: float | None = None
    young_normal_Pa: float | None = None
    poisson_plane: float | None = None
    poisson_normal: float | None = None
    shear_normal_Pa: float | None = None


@dataclass(frozen=True)
class CavernCase:
    scenario_id: str
    beta_deg: float
    psi_deg: float
    stress_ratio: float
    model: str
    major_stress_MPa: float
    geometry: CavernGeometry
    mesh: CavernMesh
    support: CavernSupport
    convergence_limit: float
    convergence_metric: str
    initial_cycles: int
    stage_cycles: int
    softening_law: StructuralSofteningLaw | None = None
    softening_maximum_damage: float = 0.0
    softening_table_point_count: int = 0

    @property
    def case_id(self) -> str:
        scenario = re.sub(r"[^a-zA-Z0-9]+", "_", self.scenario_id).strip("_").lower()
        return f"{scenario}_{self.model}_{self.mesh.name}"


def _require_mapping(config: dict[str, object], key: str) -> dict[str, object]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Cavern pilot config field {key!r} must be a mapping.")
    return value


def build_pilot_cases(config: dict[str, object]) -> tuple[list[CavernCase], CavernMaterial]:
    geometry_config = _require_mapping(config, "geometry")
    mesh_config = _require_mapping(config, "mesh")
    loading_config = _require_mapping(config, "loading")
    matrix_config = _require_mapping(config, "matrix")
    material_config = _require_mapping(config, "material")
    support_config = _require_mapping(config, "support")
    solver_config = _require_mapping(config, "solver")
    response_config = _require_mapping(config, "response")

    geometry = CavernGeometry(
        cavern_width_m=float(geometry_config["cavern_width_m"]),
        cavern_height_m=float(geometry_config["cavern_height_m"]),
        domain_half_width_m=float(geometry_config["domain_half_width_m"]),
        domain_half_height_m=float(geometry_config["domain_half_height_m"]),
        plane_strain_thickness_m=float(geometry_config["plane_strain_thickness_m"]),
        excavation_stages=int(geometry_config["excavation_stages"]),
    )
    if geometry.excavation_stages != 6:
        raise ValueError("The v2 cavern template requires exactly six excavation stages.")
    if geometry.cavern_width_m <= 0 or geometry.cavern_height_m <= geometry.cavern_width_m:
        raise ValueError("The cavern pilot requires a positive high-wall geometry.")
    if geometry.domain_half_width_m <= geometry.cavern_width_m / 2:
        raise ValueError("The cavern domain must extend beyond the excavation width.")
    if geometry.domain_half_height_m <= geometry.cavern_height_m / 2:
        raise ValueError("The cavern domain must extend beyond the excavation height.")
    mesh = CavernMesh(
        name=str(mesh_config["name"]),
        axial_zones=int(mesh_config["axial_zones"]),
        circumferential_zones=int(mesh_config["circumferential_zones"]),
        radial_zones=int(mesh_config["radial_zones"]),
        radial_ratio=float(mesh_config["radial_ratio"]),
    )
    if mesh.axial_zones < 1 or mesh.circumferential_zones < 4 or mesh.radial_zones < 8:
        raise ValueError("The body-fitted cavern mesh is too coarse for the pilot.")
    if mesh.radial_ratio < 1.0:
        raise ValueError("The cavern radial mesh ratio must be at least one.")
    support = CavernSupport(
        installation_delay_stages=int(support_config["installation_delay_stages"]),
        thickness_m=float(support_config["thickness_m"]),
        young_Pa=float(support_config["young_Pa"]),
        poisson=float(support_config["poisson"]),
        density_kg_m3=float(support_config["density_kg_m3"]),
    )
    if support.installation_delay_stages != 0:
        raise ValueError("The v2 pilot currently requires immediate stage support.")
    if support.thickness_m <= 0 or support.young_Pa <= 0 or support.density_kg_m3 <= 0:
        raise ValueError("The cavern shell support properties must be positive.")
    if not (0.0 <= support.poisson < 0.5):
        raise ValueError("The cavern shell Poisson ratio must lie in [0, 0.5).")
    anisotropic_config = config.get("anisotropic_material", {})
    if not isinstance(anisotropic_config, dict):
        raise ValueError(
            "Cavern pilot config field 'anisotropic_material' must be a mapping."
        )
    material = CavernMaterial(
        bulk_Pa=float(material_config["bulk_Pa"]),
        shear_Pa=float(material_config["shear_Pa"]),
        cohesion_Pa=float(material_config["cohesion_Pa"]),
        friction_deg=float(material_config["friction_deg"]),
        dilation_deg=float(material_config["dilation_deg"]),
        tension_Pa=float(material_config["tension_Pa"]),
        joint_cohesion_Pa=float(material_config["joint_cohesion_Pa"]),
        joint_friction_deg=float(material_config["joint_friction_deg"]),
        joint_dilation_deg=float(material_config["joint_dilation_deg"]),
        joint_tension_Pa=float(material_config["joint_tension_Pa"]),
        young_plane_Pa=(
            float(anisotropic_config["young_plane_Pa"])
            if "young_plane_Pa" in anisotropic_config
            else None
        ),
        young_normal_Pa=(
            float(anisotropic_config["young_normal_Pa"])
            if "young_normal_Pa" in anisotropic_config
            else None
        ),
        poisson_plane=(
            float(anisotropic_config["poisson_plane"])
            if "poisson_plane" in anisotropic_config
            else None
        ),
        poisson_normal=(
            float(anisotropic_config["poisson_normal"])
            if "poisson_normal" in anisotropic_config
            else None
        ),
        shear_normal_Pa=(
            float(anisotropic_config["shear_normal_Pa"])
            if "shear_normal_Pa" in anisotropic_config
            else None
        ),
    )
    major_stress = float(loading_config["major_stress_MPa"])
    convergence_metric = str(response_config.get("convergence_metric", "local")).lower()
    if convergence_metric not in {"average", "maximum", "local"}:
        raise ValueError("Cavern convergence_metric must be average, maximum, or local.")
    convergence_limit = float(
        response_config.get(
            f"convergence_ratio_{convergence_metric}",
            response_config.get("convergence_ratio_local", 0.0),
        )
    )
    initial_cycles = int(solver_config["initial_cycles"])
    stage_cycles = int(solver_config["stage_cycles"])
    if convergence_limit <= 0 or initial_cycles <= 0 or stage_cycles <= 0:
        raise ValueError("Cavern convergence and cycle controls must be positive.")

    models = [str(model) for model in matrix_config.get("models", MODELS)]
    unknown_models = sorted(set(models) - set(MODELS))
    if unknown_models:
        raise ValueError(f"Unknown cavern models: {unknown_models}")
    if "anisotropic" in models:
        ti_values = {
            "young_plane_Pa": material.young_plane_Pa,
            "young_normal_Pa": material.young_normal_Pa,
            "poisson_plane": material.poisson_plane,
            "poisson_normal": material.poisson_normal,
            "shear_normal_Pa": material.shear_normal_Pa,
        }
        missing = [name for name, value in ti_values.items() if value is None]
        if missing:
            raise ValueError(
                "The anisotropic cavern model requires anisotropic_material fields: "
                + ", ".join(missing)
            )
        if material.young_plane_Pa <= 0 or material.young_normal_Pa <= 0:
            raise ValueError("Anisotropic Young's moduli must be positive.")
        if material.shear_normal_Pa <= 0:
            raise ValueError("Anisotropic shear_normal_Pa must be positive.")
        if not (-1.0 < material.poisson_plane < 0.5):
            raise ValueError("Anisotropic poisson_plane must lie in (-1, 0.5).")
        if not (-1.0 < material.poisson_normal < 0.5):
            raise ValueError("Anisotropic poisson_normal must lie in (-1, 0.5).")
    scenarios = matrix_config.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("The cavern pilot requires at least one matrix scenario.")

    cases: list[CavernCase] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("Each cavern matrix scenario must be a mapping.")
        beta = float(scenario["beta_deg"])
        psi = float(scenario["psi_deg"])
        stress_ratio = float(scenario["stress_ratio"])
        if not (0.0 <= beta <= 90.0 and 0.0 <= psi <= 90.0):
            raise ValueError("Cavern beta_deg and psi_deg must lie in [0, 90].")
        if not (0.0 < stress_ratio <= 1.0):
            raise ValueError("Cavern stress_ratio must lie in (0, 1].")
        for model in models:
            cases.append(
                CavernCase(
                    scenario_id=str(scenario["scenario_id"]),
                    beta_deg=beta,
                    psi_deg=psi,
                    stress_ratio=stress_ratio,
                    model=model,
                    major_stress_MPa=major_stress,
                    geometry=geometry,
                    mesh=mesh,
                    support=support,
                    convergence_limit=convergence_limit,
                    convergence_metric=convergence_metric,
                    initial_cycles=initial_cycles,
                    stage_cycles=stage_cycles,
                )
            )
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Cavern scenario identifiers do not produce unique case IDs.")
    return cases, material


def _material_block(case: CavernCase, material: CavernMaterial) -> str:
    common = (
        f"zone property bulk {material.bulk_Pa:.10g} shear {material.shear_Pa:.10g} "
        f"cohesion {material.cohesion_Pa:.10g}\n"
        f"zone property friction {material.friction_deg:.10g} "
        f"dilation {material.dilation_deg:.10g} tension {material.tension_Pa:.10g}"
    )
    if case.softening_law is not None:
        if case.softening_maximum_damage <= 0.0 or case.softening_table_point_count < 2:
            raise ValueError("Cavern softening-table controls are incomplete.")
        law = case.softening_law
        kappa, cohesion = law.table(
            maximum_damage=case.softening_maximum_damage,
            point_count=case.softening_table_point_count,
        )
        table_name = f"asrc_{law.law_id}"
        table_lines = [f"table '{table_name}' delete"]
        table_lines.extend(
            f"table '{table_name}' add ({x_value:.16g},{y_value * 1.0e6:.16g})"
            for x_value, y_value in zip(kappa, cohesion)
        )
        table_lines.extend(
            [
                "zone cmodel assign softening-ubiquitous",
                common,
                f"zone property dip {case.beta_deg:.10g} dip-direction 90 "
                f"joint-cohesion {law.amplitude_mpa * 1.0e6:.10g}",
                f"zone property joint-friction {material.joint_friction_deg:.10g} "
                f"joint-dilation {material.joint_dilation_deg:.10g} "
                f"joint-tension {material.joint_tension_Pa:.10g}",
                f"zone property table-joint-cohesion '{table_name}'",
            ]
        )
        return "\n".join(table_lines)
    if case.model == "elastic":
        return (
            "zone cmodel assign elastic\n"
            f"zone property bulk {material.bulk_Pa:.10g} shear {material.shear_Pa:.10g}"
        )
    if case.model == "baseline":
        return "zone cmodel assign mohr-coulomb\n" + common
    if case.model == "reference":
        return (
            "zone cmodel assign ubiquitous-joint\n"
            + common
            + f"\nzone property dip {case.beta_deg:.10g} dip-direction 90 "
            f"joint-cohesion {material.joint_cohesion_Pa:.10g}\n"
            f"zone property joint-friction {material.joint_friction_deg:.10g} "
            f"joint-dilation {material.joint_dilation_deg:.10g} "
            f"joint-tension {material.joint_tension_Pa:.10g}"
        )
    if case.model == "anisotropic":
        if any(
            value is None
            for value in (
                material.young_plane_Pa,
                material.young_normal_Pa,
                material.poisson_plane,
                material.poisson_normal,
                material.shear_normal_Pa,
            )
        ):
            raise ValueError("Anisotropic material properties are incomplete.")
        return (
            "zone cmodel assign anisotropic\n"
            f"zone property young-plane {material.young_plane_Pa:.10g} "
            f"young-normal {material.young_normal_Pa:.10g}\n"
            f"zone property poisson-plane {material.poisson_plane:.10g} "
            f"poisson-normal {material.poisson_normal:.10g} "
            f"shear-normal {material.shear_normal_Pa:.10g}\n"
            f"zone property dip {case.beta_deg:.10g} dip-direction 90"
        )
    raise ValueError(f"Unknown cavern model: {case.model}")


def render_cavern_case(template: str, case: CavernCase, material: CavernMaterial) -> str:
    geometry = case.geometry
    springline_z = geometry.cavern_height_m / 2 - geometry.cavern_width_m / 2
    fish_ratio_name = {
        "average": "avg",
        "maximum": "max",
        "local": "local",
    }[case.convergence_metric]
    replacements = {
        "__CASE_ID__": case.case_id,
        "__SCENARIO_ID__": case.scenario_id,
        "__BETA_DEG__": f"{case.beta_deg:.10g}",
        "__PSI_DEG__": f"{case.psi_deg:.10g}",
        "__STRESS_RATIO__": f"{case.stress_ratio:.10g}",
        "__MAJOR_STRESS_PA__": f"{case.major_stress_MPa * 1e6:.10g}",
        "__CAVERN_HALF_WIDTH_M__": f"{geometry.cavern_width_m / 2:.10g}",
        "__CAVERN_HALF_HEIGHT_M__": f"{geometry.cavern_height_m / 2:.10g}",
        "__SPRINGLINE_Z_M__": f"{springline_z:.10g}",
        "__DOMAIN_HALF_WIDTH_M__": f"{geometry.domain_half_width_m:.10g}",
        "__DOMAIN_HALF_HEIGHT_M__": f"{geometry.domain_half_height_m:.10g}",
        "__PLANE_STRAIN_THICKNESS_M__": f"{geometry.plane_strain_thickness_m:.10g}",
        "__MODEL_NAME__": case.model,
        "__MESH_NAME__": case.mesh.name,
        "__AXIAL_ZONES__": str(case.mesh.axial_zones),
        "__CIRCUMFERENTIAL_ZONES__": str(case.mesh.circumferential_zones),
        "__RADIAL_ZONES__": str(case.mesh.radial_zones),
        "__RADIAL_RATIO__": f"{case.mesh.radial_ratio:.10g}",
        "__MESH_FILE__": (
            ROOT / "flac3d" / "cavern_v2" / "generated" / f"{case.mesh.name}.f3grid"
        ).resolve().as_posix(),
        "__BOUNDARY_TOLERANCE_M__": "0.05",
        "__CONVERGENCE_LIMIT__": f"{case.convergence_limit:.10g}",
        "__CONVERGENCE_METRIC__": case.convergence_metric,
        "__CONVERGENCE_RATIO_FISH__": f"zone.mech.ratio.{fish_ratio_name}",
        "__INITIAL_CYCLES__": str(case.initial_cycles),
        "__MATERIAL_BLOCK__": _material_block(case, material),
        "__STAGE_COMMANDS__": _stage_commands(case),
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    if "model precision" not in rendered:
        rendered = rendered.replace("model new", "model new\nmodel precision 10", 1)
    unresolved = sorted(set(_TOKEN.findall(rendered)))
    if unresolved:
        raise ValueError(f"Unresolved FLAC3D cavern template tokens: {unresolved}")
    return rendered


def _stage_commands(case: CavernCase, start_stage: int = 1) -> str:
    if not (1 <= start_stage <= 7):
        raise ValueError("start_stage must lie in [1, 7].")
    bounds = (44.35, 29.5666667, 14.7833333, 0.0, -14.7833333, -29.5666667, -44.35)
    commands: list[str] = []
    for stage in range(start_stage, 7):
        label = f"Stage{stage:02d}"
        lower = bounds[stage]
        upper = bounds[stage - 1]
        commands.extend(
            [
                f"zone cmodel assign null range group '{label}' slot 'ExcavationStage'",
                f"structure shell create by-zone-face id 1 group 'Support{label}' ...",
                f"    range group 'CavernBoundary' position-z {lower:.10g} {upper:.10g}",
                (
                    "structure shell property "
                    f"isotropic {case.support.young_Pa:.10g} {case.support.poisson:.10g} "
                    f"thickness {case.support.thickness_m:.10g} "
                    f"density {case.support.density_kg_m3:.10g} "
                    f"range group 'Support{label}'"
                ),
                "structure node fix velocity-y rotation-x rotation-z range position-y 0.0",
                (
                    "structure node fix velocity-y rotation-x rotation-z "
                    f"range position-y {case.geometry.plane_strain_thickness_m:.10g}"
                ),
                (
                    f"model solve ratio-{case.convergence_metric} {case.convergence_limit:.10g} "
                    f"cycles {case.stage_cycles}"
                ),
                f"[measure_cavern_response({stage})]",
                f"model save '{case.case_id}_stage_{stage:02d}'",
                f"[write_cavern_progress({stage})]",
                "",
            ]
        )
    return "\n".join(commands).rstrip()


def render_cavern_resume(case: CavernCase, start_stage: int) -> str:
    completed_stage = start_stage - 1
    if completed_stage < 1:
        raise ValueError("A cavern resume requires at least one completed stage.")
    stage_commands = _stage_commands(case, start_stage)
    if stage_commands:
        stage_commands += "\n"
    return (
        f"model restore '{case.case_id}_stage_{completed_stage:02d}'\n"
        "model precision 10\n"
        "fish automatic-create off\n"
        f"{stage_commands}"
        "[write_cavern_progress(6)]\n"
        f"io.out('ASRC_RESULT case={case.case_id} stages=6 max_disp=' + "
        "string(stage_max(6)) + ' ratio_selected=' + string(stage_ratio(6)))\n"
        f"model save '{case.case_id}_final'\n"
        "program return\n"
    )


def _read_case_rows(
    path: Path,
    expected_stages: int,
    *,
    allow_partial: bool = False,
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    valid_count = 1 <= len(rows) <= expected_stages if allow_partial else len(rows) == expected_stages
    if not valid_count:
        raise FLAC3DError(
            f"Expected {'1-' if allow_partial else ''}{expected_stages} stage rows in "
            f"{path}, found {len(rows)}."
        )
    stages = [int(row["stage"]) for row in rows]
    if stages != list(range(1, len(rows) + 1)):
        raise FLAC3DError(f"Cavern stages are incomplete or out of order in {path}: {stages}")
    support_column = next(
        (name for name in ("support_element_count", "support_zone_count") if name in rows[0]),
        None,
    )
    if support_column and min(
        int(float(row[support_column])) for row in rows
    ) <= 0:
        raise FLAC3DError(f"The staged support selected no elements in {path}.")
    return rows


def _row_convergence_value(row: dict[str, str]) -> float:
    ratio_column = "ratio_selected" if "ratio_selected" in row else "ratio_local"
    return float(row[ratio_column])


def _latest_stage_checkpoint(case_dir: Path, case_id: str, expected_stages: int) -> int:
    completed = 0
    for stage in range(1, expected_stages + 1):
        if (case_dir / f"{case_id}_stage_{stage:02d}.sav").is_file():
            completed = stage
        else:
            break
    return completed


def _write_paired_stage_residuals(
    run_dir: Path,
    rows: list[dict[str, str]],
    convergence_limit: float,
    *,
    baseline_model: str = "baseline",
    reference_model: str = "reference",
) -> tuple[Path, list[dict[str, object]]]:
    """Write auditable residuals for every configured model pair and stage."""
    grouped: dict[tuple[str, int], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (row["scenario_id"], int(row["stage"]))
        grouped.setdefault(key, {})[row["model"]] = row

    paired: list[dict[str, object]] = []
    for (scenario_id, stage), models in sorted(grouped.items()):
        if not {baseline_model, reference_model}.issubset(models):
            continue
        baseline = models[baseline_model]
        reference = models[reference_model]
        baseline_response = float(baseline["response_max_m"])
        reference_response = float(reference["response_max_m"])
        signed_residual = reference_response - baseline_response
        paired.append(
            {
                "scenario_id": scenario_id,
                "beta_deg": float(reference["beta_deg"]),
                "psi_deg": float(reference["psi_deg"]),
                "stress_ratio": float(reference["stress_ratio"]),
                "stage": stage,
                "baseline_model": baseline_model,
                "reference_model": reference_model,
                "baseline_response_m": baseline_response,
                "reference_response_m": reference_response,
                "signed_residual_m": signed_residual,
                "absolute_residual_fraction": abs(signed_residual)
                / max(abs(baseline_response), 1e-15),
                "baseline_ratio_local": float(baseline["ratio_local"]),
                "reference_ratio_local": float(reference["ratio_local"]),
                "baseline_ratio_selected": _row_convergence_value(baseline),
                "reference_ratio_selected": _row_convergence_value(reference),
                "pair_converged": (
                    _row_convergence_value(baseline) <= convergence_limit
                    and _row_convergence_value(reference) <= convergence_limit
                ),
            }
        )

    path = run_dir / "paired_stage_residuals.csv"
    fieldnames = [
        "scenario_id",
        "beta_deg",
        "psi_deg",
        "stress_ratio",
        "stage",
        "baseline_model",
        "reference_model",
        "baseline_response_m",
        "reference_response_m",
        "signed_residual_m",
        "absolute_residual_fraction",
        "baseline_ratio_local",
        "reference_ratio_local",
        "baseline_ratio_selected",
        "reference_ratio_selected",
        "pair_converged",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(paired)
    return path, paired


def _build_qualification_report(
    rows: list[dict[str, str]],
    cases: list[CavernCase],
    paired_rows: list[dict[str, object]],
    convergence_limit: float,
    admission_gate: dict[str, object],
) -> dict[str, object]:
    """Evaluate solver-pair qualification without claiming paper admission."""
    requested_models = {case.model for case in cases}
    baseline_model = str(admission_gate.get("baseline_model", "baseline"))
    reference_model = str(admission_gate.get("reference_model", "reference"))
    pair_models_available = {baseline_model, reference_model}.issubset(
        requested_models
    )
    scenario_count = len({case.scenario_id for case in cases})
    expected_stage_pairs = (
        scenario_count * cases[0].geometry.excavation_stages
        if pair_models_available
        else 0
    )
    all_stage_converged = all(
        _row_convergence_value(row) <= convergence_limit for row in rows
    )

    response_monotonic = True
    support_non_decreasing = True
    for case_id in {row["case_id"] for row in rows}:
        case_rows = sorted(
            (row for row in rows if row["case_id"] == case_id),
            key=lambda row: int(row["stage"]),
        )
        responses = [float(row["response_max_m"]) for row in case_rows]
        supports = [int(float(row["support_element_count"])) for row in case_rows]
        response_monotonic &= all(
            next_value + 1e-12 >= value
            for value, next_value in zip(responses, responses[1:])
        )
        support_non_decreasing &= all(
            next_value >= value for value, next_value in zip(supports, supports[1:])
        )

    maximum_residual_fraction = max(
        (float(row["absolute_residual_fraction"]) for row in paired_rows),
        default=None,
    )
    minimum_residual_fraction = float(
        admission_gate.get("minimum_material_residual_fraction", 0.0)
    )
    complete_stage_pairing = (
        pair_models_available and len(paired_rows) == expected_stage_pairs
    )
    residual_signal_detected = (
        maximum_residual_fraction is not None
        and maximum_residual_fraction >= minimum_residual_fraction
    )
    gates = {
        "all_stages_converged": all_stage_converged,
        "complete_stage_pairing": complete_stage_pairing,
        "monotonic_cumulative_response": response_monotonic,
        "non_decreasing_support_count": support_non_decreasing,
        "material_residual_signal": residual_signal_detected,
    }
    required = {
        "all_stages_converged": bool(
            admission_gate.get("require_all_stages_converged", True)
        ),
        "complete_stage_pairing": bool(
            admission_gate.get("require_complete_stage_pairing", True)
        ),
        "monotonic_cumulative_response": bool(
            admission_gate.get("require_monotonic_cumulative_response", True)
        ),
        "non_decreasing_support_count": bool(
            admission_gate.get("require_non_decreasing_support_count", True)
        ),
        "material_residual_signal": minimum_residual_fraction > 0.0,
    }
    failed_required_gates = [
        name for name, value in gates.items() if required[name] and not value
    ]
    qualified = pair_models_available and not failed_required_gates
    if not pair_models_available:
        status = "not_evaluated_pair_unavailable"
    elif qualified:
        status = "solver_pair_qualified"
    else:
        status = "solver_pair_failed"
    return {
        "qualification_status": status,
        "eligible_for_matrix_expansion": qualified,
        "paper_admitted": False,
        "requested_models": sorted(requested_models),
        "baseline_model": baseline_model,
        "reference_model": reference_model,
        "expected_stage_pairs": expected_stage_pairs,
        "observed_stage_pairs": len(paired_rows),
        "convergence_limit": convergence_limit,
        "convergence_metric": cases[0].convergence_metric,
        "minimum_material_residual_fraction": minimum_residual_fraction,
        "maximum_material_residual_fraction": maximum_residual_fraction,
        "gates": gates,
        "failed_required_gates": failed_required_gates,
        "pending_after_qualification": list(
            admission_gate.get("pending_after_qualification", [])
        ),
    }


def run_cavern_cases(
    cases: list[CavernCase],
    material: CavernMaterial,
    *,
    run_id: str,
    config_path: Path,
    flac3d_executable: Path,
    template_path: Path = DEFAULT_TEMPLATE,
    timeout_seconds: float = 1800.0,
    startup_timeout_seconds: float = 120.0,
    convergence_limit: float = 1e-5,
    admission_gate: dict[str, object] | None = None,
    study_status: str = "pilot_only_not_for_paper_claims",
    max_case_attempts: int = 1,
    case_retry_delay_seconds: float = 5.0,
    resume: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> Path:
    if force and resume:
        raise ValueError("force and resume are mutually exclusive.")
    if not cases:
        raise ValueError("At least one FLAC3D cavern case is required.")
    if max_case_attempts < 1:
        raise ValueError("max_case_attempts must be positive.")
    if case_retry_delay_seconds < 0:
        raise ValueError("case_retry_delay_seconds cannot be negative.")
    run_dir = ROOT / "flac3d" / "runs" / run_id
    if run_dir.exists() and not (force or resume):
        raise SystemExit(f"Run directory already exists: {run_dir}. Use --resume or --force.")
    _prepare_run_dir(run_dir, force)

    template_path = template_path.resolve()
    template = template_path.read_text(encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": "hydropower_cavern_pilot",
        "paper_claim_status": study_status,
        "flac3d_executable": str(flac3d_executable.resolve()),
        "config": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "template": str(template_path),
        "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "execution": "serial",
        "max_case_attempts": max_case_attempts,
        "case_retry_delay_seconds": case_retry_delay_seconds,
        "convergence_limit": convergence_limit,
        "convergence_metric": cases[0].convergence_metric,
        "material": asdict(material),
        "cases": [asdict(case) | {"case_id": case.case_id} for case in cases],
    }
    if "__MESH_FILE__" in template:
        mesh_files = {
            case.mesh.name: (
                ROOT / "flac3d" / "cavern_v2" / "generated" / f"{case.mesh.name}.f3grid"
            ).resolve()
            for case in cases
        }
        missing_mesh_files = [str(path) for path in mesh_files.values() if not path.is_file()]
        if missing_mesh_files:
            raise FileNotFoundError(f"Generated cavern V2 grids are missing: {missing_mesh_files}")
        manifest["mesh_artifacts"] = {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in mesh_files.items()
        }
    manifest_path = run_dir / "manifest.json"
    if resume:
        manifest = _validate_resume_manifest(manifest_path, manifest)
    else:
        _write_json(manifest_path, manifest)

    completed_rows: list[dict[str, str]] = []

    def write_progress() -> None:
        _write_json(
            run_dir / "progress.json",
            {
                "completed_cases": sorted(
                    {row["case_id"] for row in completed_rows}
                ),
                "total_cases": len(cases),
                "completed_stage_rows": len(completed_rows),
                "nonconverged_cases": sorted(
                    {
                        row["case_id"]
                        for row in completed_rows
                        if _row_convergence_value(row) > convergence_limit
                    }
                ),
            },
        )

    print(f"FLAC3D cavern pilot started: {run_dir} ({len(cases)} cases, serial)", flush=True)
    for index, case in enumerate(cases, start=1):
        case_dir = run_dir / case.case_id
        result_csv = case_dir / "cavern_case_result.csv"
        if resume and result_csv.is_file():
            rows = _read_case_rows(
                result_csv,
                case.geometry.excavation_stages,
                allow_partial=True,
            )
            if len(rows) == case.geometry.excavation_stages:
                completed_rows.extend(rows)
                write_progress()
                print(f"[{index}/{len(cases)}] resumed completed {case.case_id}", flush=True)
                continue

        case_dir.mkdir(parents=True, exist_ok=True)
        data_file = case_dir / "cavern_case.dat"

        def case_output_complete() -> bool:
            if not result_csv.is_file():
                return False
            if not (
                case_dir / f"{case.case_id}_stage_{case.geometry.excavation_stages:02d}.sav"
            ).is_file():
                return False
            try:
                _read_case_rows(result_csv, case.geometry.excavation_stages)
            except (FLAC3DError, OSError, ValueError):
                return False
            return True

        result = None
        for attempt in range(1, max_case_attempts + 1):
            completed_stage = (
                _latest_stage_checkpoint(
                    case_dir,
                    case.case_id,
                    case.geometry.excavation_stages,
                )
                if resume or attempt > 1
                else 0
            )
            if completed_stage:
                data_file.write_text(
                    render_cavern_resume(case, completed_stage + 1),
                    encoding="utf-8",
                )
                resume_note = f" resume_from_stage={completed_stage}"
            else:
                data_file.write_text(
                    render_cavern_case(template, case, material),
                    encoding="utf-8",
                )
                resume_note = ""
            attempt_note = (
                f" attempt={attempt}/{max_case_attempts}"
                if max_case_attempts > 1
                else ""
            )
            print(
                f"[{index}/{len(cases)}] solving {case.case_id} | "
                f"beta={case.beta_deg:g} psi={case.psi_deg:g} "
                f"K={case.stress_ratio:g}{resume_note}{attempt_note}",
                flush=True,
            )
            try:
                result = run_flac3d_data_file(
                    flac3d_executable,
                    data_file,
                    case_dir,
                    timeout_seconds=timeout_seconds,
                    startup_timeout_seconds=startup_timeout_seconds,
                    verbose=verbose,
                    progress_only=verbose,
                    completion_check=case_output_complete,
                )
                break
            except FLAC3DProcessCleanupError as exc:
                raise SystemExit(
                    f"{case.case_id} failed and its FLAC3D process could not be "
                    "confirmed as closed. Stop the residual console process before "
                    f"resuming this batch: {exc}"
                ) from exc
            except FLAC3DError as exc:
                transcript = case_dir / "flac3d_console.log"
                if transcript.is_file():
                    shutil.copy2(
                        transcript,
                        case_dir / f"flac3d_console_attempt_{attempt:02d}_failed.log",
                    )
                if attempt >= max_case_attempts:
                    raise SystemExit(
                        f"{case.case_id} failed after {max_case_attempts} "
                        f"attempt(s): {exc}"
                    ) from exc
                checkpoint = _latest_stage_checkpoint(
                    case_dir,
                    case.case_id,
                    case.geometry.excavation_stages,
                )
                print(
                    f"[{index}/{len(cases)}] FLAC3D exited during "
                    f"{case.case_id} attempt {attempt}; retrying from "
                    f"{'stage ' + str(checkpoint) if checkpoint else 'the start'} "
                    f"after {case_retry_delay_seconds:g}s | {exc}",
                    flush=True,
                )
                if case_retry_delay_seconds:
                    time.sleep(case_retry_delay_seconds)
        if result is None:
            raise RuntimeError(f"No FLAC3D result was returned for {case.case_id}.")
        rows = _read_case_rows(result_csv, case.geometry.excavation_stages)
        completed_rows.extend(rows)
        final = rows[-1]
        converged = all(_row_convergence_value(row) <= convergence_limit for row in rows)
        print(
            f"[{index}/{len(cases)}] complete {case.case_id} | "
            f"final_max_disp={float(final['response_max_m']):.6g} m | "
            f"plastic_zones={int(float(final['plastic_zone_count']))} | "
            f"ratio_{case.convergence_metric}={_row_convergence_value(final):.3g} | "
            f"status={'converged' if converged else 'nonconverged'} | "
            f"{result.elapsed_seconds:.1f}s",
            flush=True,
        )
        write_progress()

    combined_path = run_dir / "flac3d_cavern_pilot_results.csv"
    with combined_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(completed_rows[0]))
        writer.writeheader()
        writer.writerows(completed_rows)
    pair_gate = admission_gate or {}
    paired_path, paired_rows = _write_paired_stage_residuals(
        run_dir,
        completed_rows,
        convergence_limit,
        baseline_model=str(pair_gate.get("baseline_model", "baseline")),
        reference_model=str(pair_gate.get("reference_model", "reference")),
    )
    qualification = _build_qualification_report(
        completed_rows,
        cases,
        paired_rows,
        convergence_limit,
        pair_gate,
    )
    _write_json(run_dir / "qualification_summary.json", qualification)

    all_stage_converged = all(
        _row_convergence_value(row) <= convergence_limit for row in completed_rows
    )
    summary = {
        "solver_cases": len(cases),
        "stage_rows": len(completed_rows),
        "all_stage_converged": all_stage_converged,
        "pilot_status": "solver_pass" if all_stage_converged else "solver_fail_nonconverged",
        "convergence_limit": convergence_limit,
        "convergence_metric": cases[0].convergence_metric,
        "maximum_response_m": max(float(row["response_max_m"]) for row in completed_rows),
        "maximum_material_residual_fraction": (
            qualification["maximum_material_residual_fraction"]
        ),
        "paired_stage_count": len(paired_rows),
        "dataset": str(combined_path),
        "paired_dataset": str(paired_path),
        "qualification_status": qualification["qualification_status"],
        "eligible_for_matrix_expansion": qualification["eligible_for_matrix_expansion"],
        "paper_claim_status": study_status,
    }
    _write_json(run_dir / "pilot_summary.json", summary)
    if not all_stage_converged:
        print(
            "WARNING: one or more excavation stages exceeded the convergence limit; "
            "the pilot result is retained but is not admissible for paper claims.",
            flush=True,
        )
    print(f"FLAC3D cavern pilot complete: {combined_path}", flush=True)
    return combined_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the serial FLAC3D hydropower-inspired cavern pilot."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", default="hydropower_cavern_pilot_v1")
    parser.add_argument("--scenario-id", action="append")
    parser.add_argument("--model", action="append", choices=MODELS)
    parser.add_argument("--flac3d-exe", type=Path)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.force and args.resume:
        parser.error("--force and --resume are mutually exclusive.")

    config_path = args.config.resolve()
    config = read_yaml(config_path)
    cases, material = build_pilot_cases(config)
    if args.scenario_id:
        selected = set(args.scenario_id)
        known = {case.scenario_id for case in cases}
        unknown = sorted(selected - known)
        if unknown:
            parser.error(f"Unknown --scenario-id values: {unknown}")
        cases = [case for case in cases if case.scenario_id in selected]
    if args.model:
        selected_models = set(args.model)
        cases = [case for case in cases if case.model in selected_models]

    solver = _require_mapping(config, "solver")
    executable = args.flac3d_exe or Path(str(solver["executable"]))
    timeout = args.timeout or float(solver["timeout_seconds"])
    admission_gate = _require_mapping(config, "admission_gate")
    run_cavern_cases(
        cases,
        material,
        run_id=args.run_id,
        config_path=config_path,
        flac3d_executable=executable,
        timeout_seconds=timeout,
        convergence_limit=cases[0].convergence_limit,
        admission_gate=admission_gate,
        study_status=str(config.get("status", "pilot_only_not_for_paper_claims")),
        resume=args.resume,
        force=args.force,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

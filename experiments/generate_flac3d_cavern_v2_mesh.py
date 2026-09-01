from __future__ import annotations

import argparse
import json
from pathlib import Path

from asrc.solvers.cavern_v2_mesh import CavernV2MeshSpec, write_cavern_v2_grid
from asrc.utils.io import read_yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "flac3d_hydropower_cavern_v2_coarse.yaml"


def generate(config_path: Path, output: Path | None = None) -> tuple[Path, Path]:
    config = read_yaml(config_path)
    geometry = config["geometry"]
    mesh = config["mesh"]
    spec = CavernV2MeshSpec(
        cavern_width_m=float(geometry["cavern_width_m"]),
        cavern_height_m=float(geometry["cavern_height_m"]),
        domain_half_width_m=float(geometry["domain_half_width_m"]),
        domain_half_height_m=float(geometry["domain_half_height_m"]),
        thickness_m=float(geometry["plane_strain_thickness_m"]),
        core_horizontal_zones=int(mesh["core_horizontal_zones"]),
        stage_vertical_zones=int(mesh["stage_vertical_zones"]),
        radial_layers=int(mesh["radial_layers"]),
        arch_zones=int(mesh.get("arch_zones", 6)),
        crown_half_width_m=float(mesh.get("crown_half_width_m", 2.0)),
        radial_bias=float(mesh.get("radial_bias", 1.5)),
    )
    mesh_name = str(mesh["name"])
    output_path = output or ROOT / "flac3d" / "cavern_v2" / "generated" / f"{mesh_name}.f3grid"
    summary = write_cavern_v2_grid(output_path, spec)
    minimum_area_ratio = float(mesh.get("minimum_to_mean_area_ratio", 0.0))
    if summary.minimum_to_mean_area_ratio < minimum_area_ratio:
        raise ValueError(
            "Generated cavern mesh failed the normalized area gate: "
            f"{summary.minimum_to_mean_area_ratio:.6g} < {minimum_area_ratio:.6g}."
        )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary.__dict__, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a stage-conforming cavern V2 grid.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    mesh_path, summary_path = generate(args.config.resolve(), args.output)
    print(f"Cavern V2 grid: {mesh_path}")
    print(f"Mesh summary: {summary_path}")


if __name__ == "__main__":
    main()

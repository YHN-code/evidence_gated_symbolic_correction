from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CavernV2MeshSpec:
    cavern_width_m: float
    cavern_height_m: float
    domain_half_width_m: float
    domain_half_height_m: float
    thickness_m: float
    core_horizontal_zones: int
    stage_vertical_zones: int
    radial_layers: int
    arch_zones: int = 6
    crown_half_width_m: float = 2.0
    radial_bias: float = 1.5


@dataclass(frozen=True)
class CavernV2MeshSummary:
    gridpoints: int
    zones: int
    brick_zones: int
    wedge_zones: int
    rock_zones: int
    excavation_zones: int
    minimum_cross_section_area_m2: float
    minimum_to_mean_area_ratio: float
    stage_zone_counts: dict[str, int]


class _MeshBuilder:
    def __init__(self, thickness_m: float) -> None:
        self.thickness_m = thickness_m
        self._points_2d: dict[tuple[float, float], tuple[float, float]] = {}
        self._node_ids: dict[tuple[float, float, float], int] = {}
        self.nodes: list[tuple[int, float, float, float]] = []
        self.zones: list[tuple[str, int, tuple[int, ...]]] = []
        self.groups: dict[tuple[str, str], list[int]] = {}
        self.minimum_area = float("inf")
        self.total_area = 0.0

    @staticmethod
    def _point_key(point: tuple[float, float]) -> tuple[float, float]:
        return round(point[0], 11), round(point[1], 11)

    def register_point(self, point: tuple[float, float]) -> tuple[float, float]:
        key = self._point_key(point)
        self._points_2d.setdefault(key, (float(point[0]), float(point[1])))
        return self._points_2d[key]

    def _node(self, point: tuple[float, float], y_m: float) -> int:
        point = self.register_point(point)
        key = round(point[0], 11), round(y_m, 11), round(point[1], 11)
        if key not in self._node_ids:
            node_id = len(self.nodes) + 1
            self._node_ids[key] = node_id
            self.nodes.append((node_id, point[0], y_m, point[1]))
        return self._node_ids[key]

    @staticmethod
    def _signed_area(points: list[tuple[float, float]]) -> float:
        return 0.5 * sum(
            x0 * z1 - x1 * z0
            for (x0, z0), (x1, z1) in zip(points, points[1:] + points[:1])
        )

    def _record_groups(self, zone_id: int, groups: tuple[tuple[str, str], ...]) -> None:
        for group in groups:
            self.groups.setdefault(group, []).append(zone_id)

    def add_quad(
        self,
        points: list[tuple[float, float]],
        groups: tuple[tuple[str, str], ...],
    ) -> None:
        if len(points) != 4:
            raise ValueError("A quadrilateral requires four points.")
        area = self._signed_area(points)
        if area < 0.0:
            points = [points[0], points[3], points[2], points[1]]
            area = -area
        if area <= 1e-8:
            raise ValueError(f"Degenerate quadrilateral with area {area:g}: {points}")
        a, b, c, d = (self.register_point(point) for point in points)
        y0, y1 = 0.0, self.thickness_m
        node_ids = (
            self._node(a, y0),
            self._node(b, y0),
            self._node(a, y1),
            self._node(d, y0),
            self._node(b, y1),
            self._node(d, y1),
            self._node(c, y0),
            self._node(c, y1),
        )
        zone_id = len(self.zones) + 1
        self.zones.append(("B8", zone_id, node_ids))
        self._record_groups(zone_id, groups)
        self.minimum_area = min(self.minimum_area, area)
        self.total_area += area

    def add_triangle(
        self,
        points: list[tuple[float, float]],
        groups: tuple[tuple[str, str], ...],
    ) -> None:
        if len(points) != 3:
            raise ValueError("A triangle requires three points.")
        area = self._signed_area(points)
        if area < 0.0:
            points = [points[0], points[2], points[1]]
            area = -area
        if area <= 1e-8:
            raise ValueError(f"Degenerate triangle with area {area:g}: {points}")
        a, b, c = (self.register_point(point) for point in points)
        y0, y1 = 0.0, self.thickness_m
        node_ids = (
            self._node(a, y0),
            self._node(c, y0),
            self._node(b, y0),
            self._node(a, y1),
            self._node(c, y1),
            self._node(b, y1),
        )
        zone_id = len(self.zones) + 1
        self.zones.append(("W6", zone_id, node_ids))
        self._record_groups(zone_id, groups)
        self.minimum_area = min(self.minimum_area, area)
        self.total_area += area


def _stage_levels(spec: CavernV2MeshSpec) -> tuple[np.ndarray, np.ndarray]:
    half_height = spec.cavern_height_m / 2.0
    boundaries = np.linspace(-half_height, half_height, 7)
    levels: list[float] = []
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        strip = np.linspace(lower, upper, spec.stage_vertical_zones + 1)
        levels.extend(strip[:-1].tolist())
    levels.append(float(boundaries[-1]))
    springline = half_height - spec.cavern_width_m / 2.0
    levels.append(springline)
    arch_radius = spec.cavern_width_m / 2.0
    levels.extend(
        springline + arch_radius * np.sin(angle)
        for angle in np.linspace(0.0, np.pi / 2.0, spec.arch_zones + 1)
    )
    return np.asarray(sorted(set(round(value, 10) for value in levels))), boundaries


def _half_width(
    z_m: float,
    half_width: float,
    springline: float,
    crown_half_width: float,
) -> float:
    if z_m <= springline:
        return half_width
    circular_width = float(np.sqrt(max(half_width**2 - (z_m - springline) ** 2, 0.0)))
    return max(circular_width, crown_half_width)


def _stage_name(z_m: float, boundaries: np.ndarray) -> str:
    index = int(np.searchsorted(boundaries[1:], z_m, side="right"))
    return f"Stage{max(6 - index, 1):02d}"


def _outer_intersection(
    point: tuple[float, float],
    center_z: float,
    domain_half_width: float,
    domain_half_height: float,
) -> tuple[float, float]:
    dx = point[0]
    dz = point[1] - center_z
    scales: list[float] = []
    if abs(dx) > 1e-12:
        scales.append(domain_half_width / abs(dx))
    if dz > 1e-12:
        scales.append((domain_half_height - center_z) / dz)
    elif dz < -1e-12:
        scales.append((-domain_half_height - center_z) / dz)
    scale = min(value for value in scales if value >= 1.0)
    return dx * scale, center_z + dz * scale


def build_cavern_v2_mesh(spec: CavernV2MeshSpec) -> tuple[_MeshBuilder, CavernV2MeshSummary]:
    if spec.core_horizontal_zones < 4 or spec.core_horizontal_zones % 2:
        raise ValueError("core_horizontal_zones must be an even integer of at least four.")
    if spec.stage_vertical_zones < 1 or spec.radial_layers < 2 or spec.arch_zones < 2:
        raise ValueError("The vertical and radial mesh counts are too small.")
    if spec.radial_bias <= 0.0:
        raise ValueError("radial_bias must be positive.")

    builder = _MeshBuilder(spec.thickness_m)
    half_width = spec.cavern_width_m / 2.0
    half_height = spec.cavern_height_m / 2.0
    springline = half_height - half_width
    levels, boundaries = _stage_levels(spec)
    if not (0.0 < spec.crown_half_width_m < half_width):
        raise ValueError("crown_half_width_m must lie inside the cavern half-width.")
    widths = [
        _half_width(float(z_m), half_width, springline, spec.crown_half_width_m)
        for z_m in levels
    ]
    rows = [
        [
            builder.register_point((float(u * width), float(z_m)))
            for u in np.linspace(-1.0, 1.0, spec.core_horizontal_zones + 1)
        ]
        for z_m, width in zip(levels, widths)
    ]

    excavation_group = (("Excavation", "Default"),)
    for row_index in range(len(rows) - 1):
        lower_row = rows[row_index]
        upper_row = rows[row_index + 1]
        midpoint_z = 0.5 * (levels[row_index] + levels[row_index + 1])
        groups = excavation_group + ((_stage_name(float(midpoint_z), boundaries), "ExcavationStage"),)
        for column in range(spec.core_horizontal_zones):
            builder.add_quad(
                [
                    lower_row[column],
                    lower_row[column + 1],
                    upper_row[column + 1],
                    upper_row[column],
                ],
                groups,
            )

    floor_row = rows[0]
    perimeter: list[tuple[float, float]] = list(floor_row)
    perimeter.extend(row[-1] for row in rows[1:])
    perimeter.extend(reversed(rows[-1][:-1]))
    perimeter.extend(row[0] for row in reversed(rows[1:-1]))
    if len({builder._point_key(point) for point in perimeter}) != len(perimeter):
        raise ValueError("The cavern perimeter contains duplicate non-closing points.")

    outer = [
        _outer_intersection(
            point,
            springline,
            spec.domain_half_width_m,
            spec.domain_half_height_m,
        )
        for point in perimeter
    ]
    radial_positions = [
        (index / spec.radial_layers) ** spec.radial_bias
        for index in range(spec.radial_layers + 1)
    ]
    rings: list[list[tuple[float, float]]] = []
    for radial_position in radial_positions:
        rings.append(
            [
                builder.register_point(
                    (
                        inner[0] + radial_position * (outside[0] - inner[0]),
                        inner[1] + radial_position * (outside[1] - inner[1]),
                    )
                )
                for inner, outside in zip(perimeter, outer)
            ]
        )

    rock_groups = (("Rock", "MaterialRegion"),)
    for radial_index in range(spec.radial_layers):
        inner_ring = rings[radial_index]
        outer_ring = rings[radial_index + 1]
        for index in range(len(perimeter)):
            next_index = (index + 1) % len(perimeter)
            builder.add_quad(
                [
                    inner_ring[index],
                    outer_ring[index],
                    outer_ring[next_index],
                    inner_ring[next_index],
                ],
                rock_groups,
            )

    stage_counts = {
        f"Stage{stage:02d}": len(
            builder.groups.get((f"Stage{stage:02d}", "ExcavationStage"), [])
        )
        for stage in range(1, 7)
    }
    summary = CavernV2MeshSummary(
        gridpoints=len(builder.nodes),
        zones=len(builder.zones),
        brick_zones=sum(zone_type == "B8" for zone_type, _, _ in builder.zones),
        wedge_zones=sum(zone_type == "W6" for zone_type, _, _ in builder.zones),
        rock_zones=len(builder.groups.get(("Rock", "MaterialRegion"), [])),
        excavation_zones=len(builder.groups.get(("Excavation", "Default"), [])),
        minimum_cross_section_area_m2=builder.minimum_area,
        minimum_to_mean_area_ratio=(
            builder.minimum_area / (builder.total_area / len(builder.zones))
        ),
        stage_zone_counts=stage_counts,
    )
    return builder, summary


def write_cavern_v2_grid(path: Path, spec: CavernV2MeshSpec) -> CavernV2MeshSummary:
    builder, summary = build_cavern_v2_mesh(spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "* ASRC hydropower cavern V2 generated grid",
        "* Stage interfaces are explicit horizontal mesh lines.",
        "* GRIDPOINTS",
    ]
    lines.extend(
        f"G {node_id} {x_m:.12g} {y_m:.12g} {z_m:.12g}"
        for node_id, x_m, y_m, z_m in builder.nodes
    )
    lines.append("* ZONES")
    lines.extend(
        f"Z {zone_type} {zone_id} " + " ".join(str(node_id) for node_id in node_ids)
        for zone_type, zone_id, node_ids in builder.zones
    )
    lines.append("* ZONE GROUPS")
    for (group_name, slot_name), zone_ids in sorted(builder.groups.items()):
        lines.append(f'ZGROUP "{group_name}" SLOT "{slot_name}"')
        for start in range(0, len(zone_ids), 16):
            lines.append(" " + " ".join(str(value) for value in zone_ids[start : start + 16]))
    lines.extend(["* FACES", "* FACE GROUPS", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return summary

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
from asrc.plotting.style import set_paper_style

def save_figure(fig: plt.Figure, output_base: Path) -> list[str]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in [".png", ".pdf", ".svg"]:
        path = output_base.with_suffix(suffix)
        if suffix == ".png":
            metadata = {"Software": "ASRC"}
        elif suffix == ".pdf":
            metadata = {"CreationDate": None, "ModDate": None}
        else:
            metadata = {"Date": None}
        fig.savefig(path, bbox_inches="tight", metadata=metadata)
        if suffix == ".svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
        paths.append(str(path))
    plt.close(fig)
    return paths

def plot_flac3d_cavern_setup(
    field_data: pd.DataFrame,
    stage_results: pd.DataFrame,
    fig_dir: Path,
    *,
    beta_deg: float = 45.0,
    psi_deg: float = 30.0,
    major_stress_mpa: float = 10.0,
    output_stem: str = "Fig08_hydropower_cavern_setup",
) -> list[str]:
    """Plot the body-fitted hydropower cavern model and converged fields."""
    required = {
        "zone_id",
        "model",
        "gp_id",
        "x_m",
        "z_m",
        "disp_x_m",
        "disp_z_m",
        "stress_xx_Pa",
        "stress_zz_Pa",
        "stress_xz_Pa",
        "state",
    }
    missing = required.difference(field_data.columns)
    if missing:
        raise ValueError(f"FLAC3D cavern field data are missing columns: {sorted(missing)}")
    if stage_results.empty or int(stage_results["stage"].max()) != 6:
        raise ValueError("FLAC3D cavern plotting requires all six excavation stages.")

    set_paper_style()
    active = field_data.loc[field_data["model"].astype(str).str.lower() != "null"].copy()
    polygons: list[np.ndarray] = []
    displacements: list[float] = []
    major_compression: list[float] = []
    yielded: list[bool] = []
    for _, group in active.groupby("zone_id", sort=True):
        vertices = group[["x_m", "z_m"]].drop_duplicates().to_numpy(float)
        if len(vertices) < 3:
            continue
        center = vertices.mean(axis=0)
        order = np.argsort(np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0]))
        polygons.append(vertices[order])
        displacements.append(float(np.hypot(group["disp_x_m"], group["disp_z_m"]).mean() * 1_000.0))
        first = group.iloc[0]
        stress = np.array(
            [
                [float(first["stress_xx_Pa"]), float(first["stress_xz_Pa"])],
                [float(first["stress_xz_Pa"]), float(first["stress_zz_Pa"])],
            ]
        )
        major_compression.append(float(max(0.0, -np.linalg.eigvalsh(stress).min() / 1.0e6)))
        yielded.append(int(float(first["state"])) != 0)

    points = active.drop_duplicates("gp_id").copy()
    points["displacement_mm"] = np.hypot(points["disp_x_m"], points["disp_z_m"]) * 1_000.0
    springline = 27.35
    floor = -44.35
    radius = np.hypot(points["x_m"], points["z_m"] - springline)
    on_arch = (points["z_m"] >= springline - 0.05) & np.isclose(radius, 17.0, atol=0.08)
    on_wall = (
        points["z_m"].between(floor, springline)
        & np.isclose(points["x_m"].abs(), 17.0, atol=0.08)
    )
    on_floor = np.isclose(points["z_m"], floor, atol=0.08) & (points["x_m"].abs() <= 17.08)
    boundary = points.loc[on_arch | on_wall | on_floor]
    if boundary.empty:
        raise ValueError("No body-fitted cavern boundary points were found in the field export.")
    maximum = boundary.loc[boundary["displacement_mm"].idxmax()]

    arch_angles = np.linspace(np.pi, 0.0, 80)
    outline_x = np.r_[-17.0, -17.0, 17.0 * np.cos(arch_angles), 17.0, 17.0, -17.0]
    outline_z = np.r_[floor, springline, springline + 17.0 * np.sin(arch_angles), springline, floor, floor]
    vertices = np.column_stack((outline_x, outline_z))
    codes = np.full(len(vertices), MplPath.LINETO, dtype=np.uint8)
    codes[0] = MplPath.MOVETO
    codes[-1] = MplPath.CLOSEPOLY
    cavern_path = MplPath(vertices, codes)

    fig = plt.figure(figsize=(7.45, 5.9), layout="constrained")
    grid = fig.add_gridspec(2, 4, width_ratios=(1.0, 0.032, 1.0, 0.032))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 2])
    ax_c = fig.add_subplot(grid[1, 0])
    displacement_colorbar_ax = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[1, 2])
    stress_colorbar_ax = fig.add_subplot(grid[1, 3])
    mesh_color = "#697985"
    support_color = "#087E8B"
    weak_plane_color = "#168C80"
    load_color = "#B6534B"
    stage_colors = plt.get_cmap("cividis")(np.linspace(0.18, 0.88, 6))
    model_name = str(stage_results.iloc[-1]["model"]).strip().lower()
    is_anisotropic_model = model_name in {
        "reference",
        "anisotropic",
        "softening_baseline_linear",
        "softening_discovered_exponential",
        "softening_oracle_exponential",
    }
    has_yielded_zones = any(yielded)

    # (a) Geometry, staged excavation and loading orientation.
    ax_a.add_patch(Rectangle((-55, -55), 110, 110, facecolor="#F2F3F1", edgecolor="#353A40", lw=0.9))
    clip_patch = PathPatch(cavern_path, transform=ax_a.transData)
    stage_bounds = np.array([44.35, 29.5666667, 14.7833333, 0.0, -14.7833333, -29.5666667, floor])
    for index, (upper, lower, color) in enumerate(zip(stage_bounds[:-1], stage_bounds[1:], stage_colors), start=1):
        band = Rectangle((-17, lower), 34, upper - lower, facecolor=color, edgecolor="white", lw=0.4)
        band.set_clip_path(clip_patch)
        ax_a.add_patch(band)
        ax_a.text(0, (upper + lower) / 2, f"S{index}", ha="center", va="center", fontsize=7.0,
                  color="#17212A" if index == 6 else "white", fontweight="bold")
    ax_a.plot(outline_x, outline_z, color="#20252B", lw=1.1)
    if is_anisotropic_model:
        beta = np.deg2rad(beta_deg)
        joint_center = np.array([-37.0, -14.0])
        joint_direction = np.array([np.cos(beta), np.sin(beta)])
        joint_start = joint_center - 13 * joint_direction
        joint_end = joint_center + 13 * joint_direction
        ax_a.plot(
            [joint_start[0], joint_end[0]],
            [joint_start[1], joint_end[1]],
            color=weak_plane_color,
            lw=1.4,
        )
    psi = np.deg2rad(psi_deg)
    center = np.array([-38.0, 37.0])
    direction = np.array([np.cos(psi), np.sin(psi)])
    for sign in (-1.0, 1.0):
        ax_a.annotate(
            "",
            xy=center + sign * 2.0 * direction,
            xytext=center + sign * 10.0 * direction,
            arrowprops={"arrowstyle": "-|>", "color": load_color, "lw": 1.2},
        )
    loading_label = (
        rf"Loading: $\sigma_1={major_stress_mpa:g}$ MPa; "
        rf"$\sigma_3/\sigma_1=0.65$; $\psi={psi_deg:g}^\circ$"
    )
    if not is_anisotropic_model:
        loading_label += "  (Mohr--Coulomb baseline)"
    ax_a.text(-52, 51, loading_label, ha="left", va="top", fontsize=5.8, color=load_color)
    if is_anisotropic_model:
        ax_a.text(
            -50,
            -27,
            rf"$\beta={beta_deg:g}^\circ$",
            ha="left",
            va="top",
            fontsize=5.5,
            color=weak_plane_color,
        )
    width_dimension_z = floor - 5.0
    height_dimension_x = 27.0
    dimension_color = "#20252B"
    ax_a.plot([-17, 17], [width_dimension_z, width_dimension_z], color=dimension_color, lw=0.7)
    ax_a.plot([-17, -17], [floor, width_dimension_z - 1.0], color=dimension_color, lw=0.7)
    ax_a.plot([17, 17], [floor, width_dimension_z - 1.0], color=dimension_color, lw=0.7)
    ax_a.text(0, width_dimension_z - 1.4, "34 m", ha="center", va="top", fontsize=6.3)
    ax_a.plot([height_dimension_x, height_dimension_x], [floor, 44.35], color=dimension_color, lw=0.7)
    ax_a.plot([17, height_dimension_x + 1.0], [floor, floor], color=dimension_color, lw=0.7)
    ax_a.plot([17, height_dimension_x + 1.0], [44.35, 44.35], color=dimension_color, lw=0.7)
    ax_a.text(
        height_dimension_x + 3.0,
        0,
        "88.7 m",
        rotation=90,
        ha="center",
        va="center",
        fontsize=6.3,
    )

    # (b) Exact body-fitted mesh and installed shell support.
    mesh = PolyCollection(polygons, facecolors="#FAFAF8", edgecolors=mesh_color, linewidths=0.18)
    ax_b.add_collection(mesh)
    ax_b.fill(outline_x, outline_z, color="white", zorder=4)
    ax_b.plot(outline_x, outline_z, color=support_color, lw=1.6, zorder=5)
    ax_b.text(
        0.03,
        0.97,
        f"{len(polygons):,} zones; {int(stage_results.iloc[-1]['support_element_count'])} shell elements",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2},
    )
    ax_b.annotate(
        "shell support",
        xy=(11.5, 39.0),
        xytext=(27.0, 48.0),
        ha="left",
        va="center",
        fontsize=5.8,
        color=support_color,
        arrowprops={"arrowstyle": "-", "color": support_color, "lw": 0.7},
    )

    # (c) Converged displacement magnitude.
    displacement_collection = PolyCollection(
        polygons,
        edgecolors="face",
        linewidths=0.12,
        antialiaseds=False,
        cmap="viridis",
    )
    displacement_collection.set_array(np.asarray(displacements))
    displacement_collection.set_clim(0.0, max(displacements))
    ax_c.add_collection(displacement_collection)
    ax_c.fill(outline_x, outline_z, color="white", zorder=4)
    ax_c.plot(outline_x, outline_z, color="#20252B", lw=0.8, zorder=5)
    ax_c.text(0.03, 0.97, rf"$|u|_{{\max}}={maximum['displacement_mm']:.1f}$ mm",
              transform=ax_c.transAxes, ha="left", va="top", fontsize=6.2,
              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2})
    displacement_colorbar = fig.colorbar(displacement_collection, cax=displacement_colorbar_ax)
    displacement_colorbar.set_label("Displacement magnitude (mm)", fontsize=6.8, labelpad=4)
    displacement_colorbar.ax.tick_params(labelsize=6.3, width=0.6, length=2.5)
    displacement_colorbar.outline.set_linewidth(0.6)

    # (d) Major compressive stress. Plastic references may additionally show yielding.
    stress_collection = PolyCollection(
        polygons,
        edgecolors="face",
        linewidths=0.12,
        antialiaseds=False,
        cmap="magma",
    )
    stress_collection.set_array(np.asarray(major_compression))
    ax_d.add_collection(stress_collection)
    yielded_polygons = [polygon for polygon, is_yielded in zip(polygons, yielded) if is_yielded]
    if has_yielded_zones:
        ax_d.add_collection(PolyCollection(yielded_polygons, facecolors="none", edgecolors="#00B8D9", linewidths=0.45))
        ax_d.text(
            0.03,
            0.97,
            "Cyan outlines: yielded zones",
            transform=ax_d.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color="#007C91",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2},
        )
    ax_d.fill(outline_x, outline_z, color="white", zorder=4)
    ax_d.plot(outline_x, outline_z, color="#20252B", lw=0.8, zorder=5)
    stress_colorbar = fig.colorbar(stress_collection, cax=stress_colorbar_ax)
    stress_colorbar.set_label("Major compressive stress (MPa)", fontsize=6.8, labelpad=4)
    stress_colorbar.ax.tick_params(labelsize=6.3, width=0.6, length=2.5)
    stress_colorbar.outline.set_linewidth(0.6)

    stress_title = "(d) Major stress"
    if has_yielded_zones:
        stress_title = "(d) Stress and yielding"
    for axis, title in zip(
        (ax_a, ax_b, ax_c, ax_d),
        ("(a) Geometry and stages", "(b) Mesh and support", "(c) Displacement", stress_title),
    ):
        axis.set_title(title, loc="left", fontsize=7.6, fontweight="bold", pad=5)
        axis.set_xlim(-56, 56)
        axis.set_ylim(-56, 56)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("$x$ (m)")
        axis.set_ylabel("$z$ (m)")
        axis.set_xticks([-50, 0, 50])
        axis.set_yticks([-50, 0, 50])

    return save_figure(fig, fig_dir / output_stem)

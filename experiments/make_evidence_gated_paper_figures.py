from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd

from asrc.plotting.style import set_paper_style
from asrc.constitutive.structural_softening_application import (
    build_application_laws,
    load_frozen_structural_softening,
)
from asrc.utils.io import read_yaml


DEFAULT_OUTPUT = ROOT / "paper_assets" / "asrc_evidence_gated" / "figures"
ENGINEERING_CONTEXT = ROOT / "paper_assets" / "engineering_context" / "Fig06_hydropower_cavern_context_source"
CONFIRMATION = ROOT / "outputs" / "runs" / "semantic_evidence_guided_revision_confirmation_v1"
GATE_AUDIT = ROOT / "outputs" / "runs" / "evidence_gate_confirmation_audit_v1"
GATE1D = ROOT / "outputs" / "runs" / "gate1d_joint_information_confirmation_v1"
GATE2A = ROOT / "outputs" / "runs" / "gate2a_rate_state_external_confirmation_v1"
C1 = ROOT / "outputs" / "runs" / "constitutive_uj_c1"
C2 = ROOT / "outputs" / "runs" / "constitutive_subi_c2"
FLAC_UJ = ROOT / "outputs" / "runs" / "flac3d_uj_single_zone_validation"
FLAC_SUBI = ROOT / "outputs" / "runs" / "flac3d_subi_single_zone_validation"
CAVERN_SOFTENING = ROOT / "outputs" / "runs" / "structural_softening_cavern_confirmation_v3"
CAVERN_SOFTENING_CONFIG = ROOT / "configs" / "flac3d_structural_softening_cavern_qualification_v3.yaml"


COLORS = {
    "navy": "#284B63",
    "blue": "#3C78A8",
    "teal": "#168C80",
    "green": "#3A8064",
    "amber": "#C58A20",
    "red": "#B6534B",
    "purple": "#7564A5",
    "gray": "#69737D",
    "light_blue": "#E8F0F5",
    "light_green": "#E7F1EC",
    "light_amber": "#F7F0DF",
    "light_gray": "#F2F4F5",
    "ink": "#20262C",
}

STRATEGY_LABELS = {
    "no_acquisition": "No acquisition",
    "conditional_space_filling": "Space filling",
    "evidence_guided_predictive_disagreement": "Predictive disagreement",
    "joint_structure_parameter_information_gain": "Joint information",
    "parameter_information_gain": "Parameter information",
    "model_information_gain": "Model information",
    "expected_parameter_information_gain": "Parameter information",
    "expected_information_gain": "Model information",
    "space_filling_design": "Space filling",
    "random_design": "Random",
}

STRATEGY_COLORS = {
    "No acquisition": COLORS["gray"],
    "Space filling": COLORS["amber"],
    "Predictive disagreement": COLORS["teal"],
    "Joint information": COLORS["teal"],
    "Parameter information": COLORS["blue"],
    "Model information": COLORS["purple"],
    "Random": COLORS["gray"],
}


def _require(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required frozen result files are missing:\n" + "\n".join(missing))


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.11,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=COLORS["ink"],
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = 600
        elif suffix == "svg":
            # Keep labels editable in vector-graphics software.
            plt.rcParams["svg.fonttype"] = "none"
        fig.savefig(path, **kwargs)
        if suffix == "svg":
            svg_text = path.read_text(encoding="utf-8")
            path.write_text(
                "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
                encoding="utf-8",
            )
        paths.append(path)
    plt.close(fig)
    return paths


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
    *,
    linewidth: float = 0.85,
    linestyle: str | tuple[float, tuple[float, ...]] = "-",
    title_size: float = 7.0,
    body_size: float = 6.2,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.007",
        linewidth=linewidth,
        linestyle=linestyle,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.014,
        y + height - 0.040,
        title,
        fontsize=title_size,
        fontweight="bold",
        va="top",
        color=COLORS["ink"],
    )
    ax.text(
        x + 0.014,
        y + height - 0.105,
        body,
        fontsize=body_size,
        va="top",
        linespacing=1.22,
        color="#46515B",
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#64727E") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.15,
            color=color,
            connectionstyle="arc3,rad=0",
        )
    )


def figure_architecture(output_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.4, 2.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # A restrained primary path keeps the evidence gate as the only emphasized module.
    y, height = 0.58, 0.34
    neutral_face = "#FAFBFB"
    neutral_edge = "#929DA5"
    _box(
        ax,
        (0.035, y),
        0.125,
        height,
        "Baseline",
        r"$M_0(\mathbf{x})$" "\nobservations\ndomain limits",
        neutral_face,
        neutral_edge,
    )
    _box(
        ax,
        (0.205, y),
        0.165,
        height,
        "Candidate pool",
        "PySR\noptional LLM\ntemplates",
        neutral_face,
        neutral_edge,
    )
    _box(
        ax,
        (0.415, y),
        0.185,
        height,
        "Fit and verify",
        "typed checks\ncoefficient fitting\ngrouped validation\nphysical checks",
        neutral_face,
        neutral_edge,
    )
    _box(
        ax,
        (0.645, y),
        0.165,
        height,
        "Evidence gate",
        "near-equivalent set\nresponse spread\nimpact threshold",
        "#EDF3F6",
        COLORS["navy"],
        linewidth=1.35,
        title_size=7.25,
    )
    _box(
        ax,
        (0.855, y),
        0.110,
        height,
        "Outcome",
        "accepted\nprovisional\nrejected",
        neutral_face,
        neutral_edge,
    )

    center_y = y + height / 2
    for start, end in (
        ((0.160, center_y), (0.205, center_y)),
        ((0.370, center_y), (0.415, center_y)),
        ((0.600, center_y), (0.645, center_y)),
        ((0.810, center_y), (0.855, center_y)),
    ):
        _arrow(ax, start, end)

    # The lower branch is subordinate: it is entered only when the gate remains unresolved.
    acq_x, acq_y, acq_w, acq_h = 0.490, 0.12, 0.320, 0.22
    _box(
        ax,
        (acq_x, acq_y),
        acq_w,
        acq_h,
        "Acquire evidence",
        "prespecified acquisition policy",
        "#FBFCFC",
        "#70818D",
        linewidth=0.9,
        linestyle=(0, (3, 2)),
        title_size=6.9,
        body_size=5.95,
    )
    _arrow(ax, (0.728, y), (0.728, acq_y + acq_h), COLORS["navy"])
    _arrow(ax, (0.555, acq_y + acq_h), (0.555, y), COLORS["navy"])
    ax.text(0.739, 0.460, "if unresolved", fontsize=6.1, color=COLORS["navy"], ha="left", va="center")
    ax.text(0.544, 0.460, "new evidence", fontsize=6.1, color=COLORS["navy"], ha="right", va="center")
    return _save(fig, output_dir, "Fig01_evidence_gated_architecture")


def figure_patch_algebra(output_dir: Path) -> list[Path]:
    x = np.linspace(0.0, 1.0, 240)
    baseline = 0.18 + 0.65 * x
    correction = 0.11 * np.sin(np.pi * x)
    revised = (
        baseline + correction,
        baseline * (1.0 + 0.24 * np.sin(np.pi * x)),
        0.18 + 0.65 * np.sqrt(x),
    )
    titles = ("Additive correction", "Multiplicative correction", "Structural replacement")
    formulas = (
        r"$M=M_0+e(\mathbf{x};\theta)$",
        r"$M=M_0[1+e(\mathbf{x};\theta)]$",
        r"$M=\mathcal{P}(M_0,e,\mathcal{T})$",
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.65), sharex=True, sharey=True)
    for idx, (ax, y, title, formula) in enumerate(zip(axes, revised, titles, formulas)):
        ax.plot(x, baseline, color=COLORS["gray"], linestyle="--", label="Baseline $M_0$")
        ax.plot(x, y, color=(COLORS["teal"], COLORS["blue"], COLORS["purple"])[idx], label="Revised $M$")
        ax.fill_between(x, baseline, y, color=(COLORS["teal"], COLORS["blue"], COLORS["purple"])[idx], alpha=0.10)
        ax.set_title(title, pad=16, fontweight="bold")
        ax.text(0.5, 1.04, formula, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.2)
        ax.set_xlabel("Declared input domain")
        ax.grid(axis="y", color="#D9DEE2", linewidth=0.5)
        _panel_label(ax, f"({chr(97 + idx)})")
    axes[0].set_ylabel("Model response")
    axes[0].legend(loc="upper left", frameon=False)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.79, bottom=0.18, wspace=0.16)
    return _save(fig, output_dir, "Fig02_typed_patch_algebra")


def figure_confirmation(output_dir: Path) -> list[Path]:
    result_path = CONFIRMATION / "metrics" / "evidence_guided_confirmation_results.csv"
    gate_audit_path = GATE_AUDIT / "tables" / "gate_audit_summary.csv"
    _require([result_path, gate_audit_path])
    data = pd.read_csv(result_path)
    gate_audit = pd.read_csv(gate_audit_path)
    order = ["no_acquisition", "conditional_space_filling", "evidence_guided_predictive_disagreement"]
    labels = ["No acquisition", "Gated space filling", "Gated disagreement"]
    colors = [COLORS["gray"], COLORS["amber"], COLORS["teal"]]

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
    ax = axes[0, 0]
    values = [data.loc[data["strategy"] == strategy, "normalized_locked_rmse"].to_numpy(float) for strategy in order]
    box = ax.boxplot(values, positions=np.arange(3), widths=0.52, patch_artist=True, showfliers=False, medianprops={"color": "white", "linewidth": 1.4})
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.92)
    rng = np.random.default_rng(20260812)
    for i, (arr, color) in enumerate(zip(values, colors)):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(arr)), arr, s=9, color=color, alpha=0.38, edgecolors="none")
    ax.set_xticks(
        range(3),
        ["No\nacquisition", "Gated space\nfilling", "Gated\ndisagreement"],
    )
    ax.set_ylabel("Normalized locked-test RMSE")
    ax.set_title("Held-out prediction")
    _panel_label(ax, "(a)")

    ax = axes[0, 1]
    pivot = data.pivot_table(index="data_seed", columns="strategy", values="normalized_locked_rmse", aggfunc="mean")
    diff = pivot[order[2]] - pivot[order[0]]
    y = np.arange(len(diff))
    ax.axvline(0.0, color="#9AA3AA", linewidth=0.8)
    ax.barh(y, diff, color=np.where(diff < 0, COLORS["teal"], COLORS["red"]), height=0.66)
    ax.set_yticks(y, [f"Seed {i + 1}" for i in y])
    ax.invert_yaxis()
    ax.set_xlabel("Predictive disagreement - no acquisition")
    ax.set_title("Seed-cluster paired contrast")
    _panel_label(ax, "(b)")

    ax = axes[1, 0]
    recovery = [int(data.loc[data["strategy"] == strategy, "corrected_behavior_recovered"].astype(bool).sum()) for strategy in order]
    ambiguous = [int((data.loc[data["strategy"] == strategy, "final_ambiguity_class"] == "structurally_ambiguous").sum()) for strategy in order]
    xx = np.arange(3)
    ax.bar(xx - 0.18, recovery, width=0.36, color=COLORS["teal"], label="Adequacy met")
    ax.bar(xx + 0.18, ambiguous, width=0.36, color=COLORS["amber"], label="Structurally ambiguous")
    ax.set_xticks(
        xx,
        ["No\nacquisition", "Gated space\nfilling", "Gated\ndisagreement"],
    )
    ax.set_ylabel("Confirmation cases (of 32)")
    ax.set_ylim(0, 25.5)
    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=6.7,
        handlelength=1.2,
        handletextpad=0.45,
        labelspacing=0.35,
    )
    ax.set_title("Evidence outcomes")
    _panel_label(ax, "(c)")

    ax = axes[1, 1]
    primary_summary = data.groupby("strategy", as_index=True).agg(
        mean_rmse=("normalized_locked_rmse", "mean"),
        mean_queries=("acquisitions_used", "mean"),
    )
    unconditional = gate_audit.loc[
        gate_audit["strategy"] == "unconditional_space_filling"
    ].iloc[0]
    tradeoff = [
        (
            "No acquisition",
            float(primary_summary.loc[order[0], "mean_queries"]),
            float(primary_summary.loc[order[0], "mean_rmse"]),
            COLORS["gray"],
            "o",
        ),
        (
            "Gated disagreement",
            float(primary_summary.loc[order[2], "mean_queries"]),
            float(primary_summary.loc[order[2], "mean_rmse"]),
            COLORS["teal"],
            "D",
        ),
        (
            "Gated space filling",
            float(primary_summary.loc[order[1], "mean_queries"]),
            float(primary_summary.loc[order[1], "mean_rmse"]),
            COLORS["amber"],
            "s",
        ),
        (
            "Unconditional space filling",
            float(unconditional["mean_acquisitions"]),
            float(unconditional["mean_normalized_locked_rmse"]),
            COLORS["blue"],
            "^",
        ),
    ]
    for label, mean_queries, mean_rmse, color, marker in tradeoff:
        ax.scatter(
            mean_queries,
            mean_rmse,
            s=55,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
            label=label,
        )
    ax.legend(
        frameon=False,
        fontsize=6.7,
        loc="upper right",
        handletextpad=0.45,
        labelspacing=0.35,
    )
    ax.set_xlim(-0.12, 3.18)
    ax.set_ylim(0.085, 0.205)
    ax.set_xlabel("Mean added observations per case")
    ax.set_ylabel("Mean normalized locked-test RMSE")
    ax.set_title("Error-query tradeoff")
    _panel_label(ax, "(d)")

    for ax in axes.flat:
        ax.grid(axis="y", color="#D9DEE2", linewidth=0.5, zorder=0)
    fig.tight_layout(pad=1.1, w_pad=1.0, h_pad=1.2)
    return _save(fig, output_dir, "Fig03_confirmation_results")


def _trajectory_panel(ax: plt.Axes, path: Path, title: str) -> None:
    data = pd.read_csv(path)
    cluster = data.groupby(["strategy", "data_seed", "new_observation_count"], as_index=False)["locked_rmse"].mean()
    summary = cluster.groupby(["strategy", "new_observation_count"])["locked_rmse"].agg(["mean", "std", "count"]).reset_index()
    for strategy, rows in summary.groupby("strategy", sort=False):
        label = STRATEGY_LABELS.get(strategy, strategy.replace("_", " ").title())
        color = STRATEGY_COLORS.get(label, COLORS["gray"])
        rows = rows.sort_values("new_observation_count")
        x = rows["new_observation_count"].to_numpy(float)
        mean = rows["mean"].to_numpy(float)
        se = rows["std"].fillna(0).to_numpy(float) / np.sqrt(rows["count"].to_numpy(float))
        ax.plot(x, mean, marker="o", color=color, label=label)
        ax.fill_between(x, np.maximum(mean - se, 0), mean + se, color=color, alpha=0.10)
    ax.set_title(title)
    ax.set_xlabel("New observations")
    ax.set_ylabel("Locked-test RMSE")
    ax.set_xticks(range(0, 9, 2))
    ax.grid(color="#D9DEE2", linewidth=0.5)


def figure_experimental_design(output_dir: Path) -> list[Path]:
    path1 = GATE1D / "metrics" / "gate1d_confirmation_trajectory.csv"
    path2 = GATE2A / "metrics" / "gate2a_rate_state_trajectory.csv"
    _require([path1, path2])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.45))
    _trajectory_panel(axes[0], path1, "Six symbolic ambiguity tasks")
    _trajectory_panel(axes[1], path2, "Rate-and-state mechanisms")
    _panel_label(axes[0], "(a)")
    _panel_label(axes[1], "(b)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7.2,
        columnspacing=1.2,
        handlelength=2.0,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.89, bottom=0.29, wspace=0.22)
    return _save(fig, output_dir, "Fig04_experimental_design")


def figure_constitutive_flac3d(output_dir: Path) -> list[Path]:
    c1_json = C1 / "formulas" / "constitutive_uj_c1_models.json"
    c2_json = C2 / "formulas" / "constitutive_subi_c2_models.json"
    uj_csv = FLAC_UJ / "data" / "flac3d_uj_single_zone_comparison.csv"
    subi_csv = FLAC_SUBI / "data" / "flac3d_subi_single_zone_comparison.csv"
    _require([c1_json, c2_json, uj_csv, subi_csv])
    c1 = json.loads(c1_json.read_text(encoding="utf-8"))
    c2 = json.loads(c2_json.read_text(encoding="utf-8"))
    uj = pd.read_csv(uj_csv)
    subi = pd.read_csv(subi_csv)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4))
    ax = axes[0, 0]
    p = np.linspace(0, 16, 200)
    ref = c1["reference"]
    fit = c1["selected_constitutive_asrc"]
    y_ref = float(ref["cohesion_mpa"]) + p * np.tan(np.deg2rad(float(ref["friction_deg"])))
    y_fit = float(fit["coefficients"][0]) + p * float(fit["coefficients"][1])
    ax.plot(p, y_ref, color=COLORS["ink"], linewidth=2.2, label="Prescribed")
    ax.plot(p, y_fit, color=COLORS["teal"], linestyle="--", linewidth=1.5, label="Recovered")
    ax.set_xlabel(r"Normal compression, $-\sigma_n$ (MPa)")
    ax.set_ylabel("Weak-plane shear strength (MPa)")
    ax.set_title("Ubiquitous-joint strength")
    ax.legend(frameon=False)
    _panel_label(ax, "(a)")

    ax = axes[0, 1]
    kappa = np.linspace(0, 0.003, 300)
    cref = c2["reference"]["coefficients"]
    cfit = c2["selected_constitutive_c2_asrc"]["coefficients"]
    poly = c2["generic_state_polynomial"]["coefficients"]
    ref_curve = cref[0] + cref[1] * np.exp(-kappa / cref[2])
    fit_curve = cfit[0] + cfit[1] * np.exp(-kappa / cfit[2])
    poly_curve = poly[0] + poly[1] * kappa + poly[2] * kappa**2
    ax.plot(kappa * 1e3, ref_curve, color=COLORS["ink"], linewidth=2.2, label="Prescribed")
    ax.plot(kappa * 1e3, fit_curve, color=COLORS["teal"], linestyle="--", label="Recovered")
    ax.plot(kappa * 1e3, poly_curve, color=COLORS["red"], linestyle=":", label="Polynomial proxy")
    ax.set_xlabel(r"Accumulated plastic shear strain, $\kappa$ ($\times10^{-3}$)")
    ax.set_ylabel("Cohesion (MPa)")
    ax.set_title("Path-dependent cohesion")
    ax.legend(frameon=False)
    _panel_label(ax, "(b)")

    ax = axes[1, 0]
    flac_values: list[np.ndarray] = []
    python_values: list[np.ndarray] = []
    for component in ("xx", "yy", "zz", "xy", "xz", "yz"):
        flac_values.append(uj[f"flac3d_sigma_{component}_mpa"].to_numpy(float))
        python_values.append(uj[f"python_sigma_{component}_mpa"].to_numpy(float))
    xval = np.concatenate(flac_values)
    yval = np.concatenate(python_values)
    lim = [min(xval.min(), yval.min()), max(xval.max(), yval.max())]
    ax.scatter(xval, yval, s=8, color=COLORS["blue"], alpha=0.42, edgecolors="none")
    ax.plot(lim, lim, color=COLORS["ink"], linestyle="--", linewidth=0.9)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("FLAC3D stress (MPa)")
    ax.set_ylabel("Python replay stress (MPa)")
    ax.set_title("Ubiquitous-joint replay")
    ax.text(0.04, 0.94, r"RMSE = $4.00\times10^{-6}$ MPa", transform=ax.transAxes, va="top", fontsize=7.4)
    _panel_label(ax, "(c)")

    ax = axes[1, 1]
    for case_id, rows in subi.groupby("case_id"):
        rows = rows.sort_values("step")
        ax.plot(rows["flac3d_plastic_shear_joint"] * 1e3, rows["flac3d_joint_cohesion_mpa"], color=COLORS["gray"], alpha=0.55)
        ax.plot(rows["python_plastic_shear_joint"] * 1e3, rows["python_joint_cohesion_after_mpa"], color=COLORS["teal"], linestyle="--", alpha=0.85)
    ax.set_xlabel(r"Plastic shear state ($\times10^{-3}$)")
    ax.set_ylabel("Cohesion (MPa)")
    ax.set_title("Strain-softening replay")
    ax.text(0.04, 0.94, r"Stress RMSE = $7.61\times10^{-6}$ MPa", transform=ax.transAxes, va="top", fontsize=7.4)
    _panel_label(ax, "(d)")

    for ax in axes.flat:
        ax.grid(color="#D9DEE2", linewidth=0.5)
    fig.tight_layout(pad=1.05, w_pad=1.0, h_pad=1.15)
    return _save(fig, output_dir, "Fig05_constitutive_flac3d")


def figure_cavern_softening_propagation(output_dir: Path) -> list[Path]:
    data_path = CAVERN_SOFTENING / "data" / "qualified_cavern_three_law_results.csv"
    report_path = CAVERN_SOFTENING / "reports" / "qualified_structural_softening_cavern_gate.json"
    _require([data_path, report_path, CAVERN_SOFTENING_CONFIG])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed" or not report.get("response_metrics_admissible"):
        raise ValueError("The frozen cavern propagation result did not pass its admission gate.")

    data = pd.read_csv(data_path)
    config = read_yaml(CAVERN_SOFTENING_CONFIG)
    artifact_path = Path(config["structural_softening"]["frozen_artifact"])
    if not artifact_path.is_absolute():
        artifact_path = ROOT / artifact_path
    frozen = load_frozen_structural_softening(artifact_path)
    laws = build_application_laws(frozen, config["structural_softening"]["application"])
    laws_by_id = {law.law_id: law for law in laws}
    model_order = [
        "softening_baseline_linear",
        "softening_discovered_exponential",
        "softening_oracle_exponential",
    ]
    labels = {
        "softening_baseline_linear": "Linear baseline",
        "softening_discovered_exponential": "Transferred law",
        "softening_oracle_exponential": "Prescribed reference",
    }
    colors = {
        "softening_baseline_linear": COLORS["gray"],
        "softening_discovered_exponential": COLORS["teal"],
        "softening_oracle_exponential": COLORS["ink"],
    }
    linestyles = {
        "softening_baseline_linear": ":",
        "softening_discovered_exponential": "-",
        "softening_oracle_exponential": "--",
    }
    markers = {
        "softening_baseline_linear": "o",
        "softening_discovered_exponential": "s",
        "softening_oracle_exponential": "^",
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.45))
    ax = axes[0, 0]
    damage = np.linspace(0.0, 2.5, 301)
    law_to_model = {
        "baseline_linear": "softening_baseline_linear",
        "discovered_exponential": "softening_discovered_exponential",
        "oracle_exponential": "softening_oracle_exponential",
    }
    for law_id in ("baseline_linear", "discovered_exponential", "oracle_exponential"):
        model = law_to_model[law_id]
        ax.plot(
            damage,
            laws_by_id[law_id].cohesion_from_damage(damage),
            color=colors[model],
            linestyle=linestyles[model],
            linewidth=2.0 if model == "softening_discovered_exponential" else 1.5,
            label=labels[model],
        )
    ax.set_xlabel(r"Normalized plastic-shear state, $d$")
    ax.set_ylabel("Weak-plane cohesion (MPa)")
    ax.set_title("Transferred cohesion-decay shape")
    ax.set_xlim(0.0, 2.5)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False, loc="upper right", handlelength=2.3)
    _panel_label(ax, "(a)")

    def response_panel(ax: plt.Axes, column: str, title: str, ylabel: str) -> None:
        for model in model_order:
            rows = data.loc[data["model"] == model].sort_values("stage")
            ax.plot(
                rows["stage"],
                rows[column] * 1000.0,
                color=colors[model],
                linestyle=linestyles[model],
                marker=markers[model],
                markerfacecolor="white" if model != "softening_discovered_exponential" else colors[model],
                markeredgewidth=0.8,
                linewidth=2.0 if model == "softening_discovered_exponential" else 1.35,
                label=labels[model],
            )
        ax.set_xticks(range(1, 7))
        ax.set_xlabel("Excavation stage")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    response_panel(axes[0, 1], "response_max_m", "Maximum displacement", "Displacement (mm)")
    _panel_label(axes[0, 1], "(b)")
    response_panel(
        axes[1, 0],
        "response_wall_convergence_m",
        "Wall convergence",
        "Convergence (mm)",
    )
    _panel_label(axes[1, 0], "(c)")

    ax = axes[1, 1]
    indexed = data.set_index(["stage", "model"]).sort_index()
    oracle = indexed.xs("softening_oracle_exponential", level="model")
    response_columns = [
        "response_max_m",
        "response_crown_m",
        "response_wall_convergence_m",
        "response_invert_m",
    ]
    category_labels = ["Combined", "Maximum", "Crown", "Wall", "Invert"]

    def response_rmse(model: str) -> np.ndarray:
        method = indexed.xs(model, level="model")
        per_channel = [
            float(np.sqrt(np.mean((method[column] - oracle[column]) ** 2)))
            for column in response_columns
        ]
        joint = float(
            np.sqrt(
                np.mean(
                    np.concatenate(
                        [
                            (method[column] - oracle[column]).to_numpy(float)
                            for column in response_columns
                        ]
                    )
                    ** 2
                )
            )
        )
        return np.asarray([joint, *per_channel]) * 1000.0

    baseline_rmse = response_rmse("softening_baseline_linear")
    discovered_rmse = response_rmse("softening_discovered_exponential")
    improvement = 1.0 - discovered_rmse / baseline_rmse
    x = np.arange(len(category_labels))
    width = 0.36
    ax.bar(
        x - width / 2,
        baseline_rmse,
        width,
        color=COLORS["gray"],
        label="Linear baseline",
    )
    bars = ax.bar(
        x + width / 2,
        discovered_rmse,
        width,
        color=COLORS["teal"],
        label="Transferred law",
    )
    for bar, value in zip(bars, improvement):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{100 * value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7.6,
            color=COLORS["teal"],
        )
    ax.set_xticks(x, category_labels)
    ax.set_ylabel("RMSE relative to reference (mm)")
    ax.set_title("Boundary-response error")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0.0, max(baseline_rmse) * 1.25)
    _panel_label(ax, "(d)")

    for ax in axes.flat:
        ax.grid(axis="y", color="#D9DEE2", linewidth=0.5, zorder=0)
    fig.tight_layout(pad=1.05, w_pad=1.0, h_pad=1.15)
    return _save(fig, output_dir, "Fig07_cavern_softening_propagation")


def figure_literature(output_dir: Path) -> list[Path]:
    datasets = ["Le.Gs", "My.Sc", "Longmaxi", "Jixi"]
    baseline = np.array([2.3771, 2.1954, 0.2131, 0.0214])
    revised = np.array([2.4542, 1.5255, 0.1350, 0.0223])
    accepted = np.array([False, True, True, False])

    fig = plt.figure(figsize=(7.4, 3.45))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.18, 1.0, 1.0], wspace=0.44)
    ax = fig.add_subplot(grid[0, 0])
    x = np.arange(4)
    width = 0.36
    ax.bar(x - width / 2, np.ones(4), width, color=COLORS["gray"])
    ax.bar(x + width / 2, revised / baseline, width, color=np.where(accepted, COLORS["teal"], COLORS["amber"]))
    ax.axhline(1.0, color=COLORS["ink"], linewidth=0.8)
    ax.set_xticks(x, datasets, rotation=20, ha="right")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Group-CV RMSE / baseline RMSE")
    ax.set_title("Grouped validation")
    literature_legend = [
        Patch(facecolor=COLORS["gray"], label="Baseline"),
        Patch(facecolor=COLORS["teal"], label="Accepted"),
        Patch(facecolor=COLORS["amber"], label="Provisional"),
    ]
    _panel_label(ax, "(a)")

    ax = fig.add_subplot(grid[0, 1])
    beta = np.linspace(0, 90, 121)
    psi = np.linspace(0, 90, 121)
    bb, pp = np.meshgrid(np.deg2rad(beta), np.deg2rad(psi), indexing="ij")
    mpsi = np.abs(np.sin(pp) * np.cos(pp))
    residual = 0.0111 - 2.096 * np.sin(bb) + 3.717 * np.sin(pp) - 11.15 * mpsi**2
    levels = np.linspace(-np.max(np.abs(residual)), np.max(np.abs(residual)), 13)
    contour = ax.contourf(beta, psi, residual.T, levels=levels, cmap="RdBu_r", extend="both")
    ax.set_xlabel(r"$\beta$ (deg)")
    ax.set_ylabel(r"$\psi$ (deg)")
    ax.set_title("Dinh My.Sc")
    cax = ax.inset_axes([0.04, -0.33, 0.92, 0.055])
    cbar = fig.colorbar(contour, cax=cax, orientation="horizontal")
    cbar.set_ticks(np.linspace(levels[0], levels[-1], 5))
    cbar.set_label("Correction (MPa)", labelpad=2)
    cbar.ax.tick_params(labelsize=6.5, pad=1)
    _panel_label(ax, "(b)")

    ax = fig.add_subplot(grid[0, 2])
    beta_rad = np.deg2rad(beta)
    mbeta = np.abs(np.sin(beta_rad) * np.cos(beta_rad))
    longmaxi = 0.00440 + 22.49 * mbeta**2 / (1 + 3 * mbeta**2) - 2.945 * np.sin(2 * beta_rad)
    ax.plot(beta, longmaxi, color=COLORS["teal"], linewidth=2.0)
    ax.axhline(0.0, color="#9AA3AA", linewidth=0.8)
    ax.fill_between(beta, 0.0, longmaxi, color=COLORS["teal"], alpha=0.12)
    ax.set_xlabel(r"$\beta$ (deg)")
    ax.set_ylabel("Residual correction (MPa)")
    ax.set_title("Ma Longmaxi shale")
    _panel_label(ax, "(c)")

    for plot_ax in (fig.axes[0], fig.axes[2]):
        plot_ax.grid(color="#D9DEE2", linewidth=0.45)
    fig.legend(
        handles=literature_legend,
        loc="lower center",
        bbox_to_anchor=(0.205, 0.015),
        ncol=3,
        frameon=False,
        fontsize=6.5,
        handlelength=1.0,
        handletextpad=0.4,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.28)
    return _save(fig, output_dir, "Fig08_literature_validation")


def copy_engineering_context(output_dir: Path) -> list[Path]:
    """Copy the frozen FLAC3D cavern field export into the new paper bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ("png", "pdf", "svg"):
        source = ENGINEERING_CONTEXT.with_suffix(f".{suffix}")
        _require([source])
        target = output_dir / f"Fig06_hydropower_cavern_context.{suffix}"
        shutil.copy2(source, target)
        paths.append(target)
    return paths


def make_all(output_dir: Path) -> list[Path]:
    set_paper_style()
    paths: list[Path] = []
    builders = [
        ("Fig. 1 evidence-gated architecture", figure_architecture),
        ("Fig. 2 typed patch algebra", figure_patch_algebra),
        ("Fig. 3 confirmation results", figure_confirmation),
        ("Fig. 4 experimental design", figure_experimental_design),
        ("Fig. 5 constitutive replay", figure_constitutive_flac3d),
        ("Fig. 7 cavern propagation", figure_cavern_softening_propagation),
        ("Fig. 8 literature evaluation", figure_literature),
        ("Fig. 6 engineering context", copy_engineering_context),
    ]
    for label, builder in builders:
        print(f"Generating {label}...", flush=True)
        paths.extend(builder(output_dir))
        gc.collect()
    for suffix in ("png", "pdf", "svg"):
        (output_dir / f"Fig07_literature_validation.{suffix}").unlink(missing_ok=True)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evidence-gated ASRC manuscript figures.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    paths = make_all(output_dir)
    print(f"Generated {len(paths)} figure files in {output_dir}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()

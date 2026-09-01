from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from asrc.constitutive.softening import generate_subi_c2_dataset
from asrc.constitutive.weak_plane import STRESS_COMPONENTS


IDENTIFIER_COLUMNS = [
    "partition",
    "trajectory_id",
    "step",
    "beta_deg",
    "confining_pressure_mpa",
    "path",
]
STRAIN_COLUMNS = [
    "delta_eps_xx",
    "delta_eps_yy",
    "delta_eps_zz",
    "delta_eps_xy",
    "delta_eps_xz",
    "delta_eps_yz",
]


def generate_subi_c3_dataset(
    config: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Add reproducible sparse observations to the hidden C2 reference paths."""
    frame = generate_subi_c2_dataset(config).copy()
    observation = config["observations"]
    stress_stride = int(observation["stress_sample_stride"])
    state_stride = int(observation["plastic_shear_sample_stride"])
    if stress_stride < 1 or state_stride < 1:
        raise ValueError("C3 observation strides must be positive.")
    rng = np.random.default_rng(int(seed))
    frame["stress_observed"] = False
    frame["plastic_shear_observed"] = False
    frame["observed_kappa"] = np.nan
    frame["declared_stress_noise_std_mpa"] = float(
        observation["stress_noise_std_mpa"]
    )
    frame["declared_plastic_shear_noise_std"] = float(
        observation["plastic_shear_noise_std"]
    )
    for component in STRESS_COMPONENTS:
        frame[f"observed_sigma_{component}_mpa"] = np.nan

    stress_components = [
        str(item) for item in observation["observed_stress_components"]
    ]
    unknown = set(stress_components).difference(STRESS_COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown observed stress components: {sorted(unknown)}")
    if len(stress_components) != len(set(stress_components)):
        raise ValueError("Observed stress components must not contain duplicates.")

    for _, group in frame.groupby("trajectory_id", sort=False):
        ordered = group.sort_values("step")
        steps = ordered["step"].to_numpy(int)
        stress_mask = steps % stress_stride == 0
        state_mask = steps % state_stride == 0
        if bool(observation.get("always_observe_first_and_last", True)):
            stress_mask[[0, -1]] = True
            state_mask[-1] = True
        stress_indices = ordered.index[stress_mask]
        state_indices = ordered.index[
            state_mask
            & (ordered["reference_kappa_after"].to_numpy(float) > 0.0)
        ]
        frame.loc[stress_indices, "stress_observed"] = True
        frame.loc[state_indices, "plastic_shear_observed"] = True
        for component in stress_components:
            source = frame.loc[
                stress_indices,
                f"reference_sigma_{component}_mpa",
            ].to_numpy(float)
            noise = rng.normal(
                0.0,
                float(observation["stress_noise_std_mpa"]),
                size=len(stress_indices),
            )
            frame.loc[
                stress_indices,
                f"observed_sigma_{component}_mpa",
            ] = source + noise
        state_noise = rng.normal(
            0.0,
            float(observation["plastic_shear_noise_std"]),
            size=len(state_indices),
        )
        frame.loc[state_indices, "observed_kappa"] = np.maximum(
            frame.loc[
                state_indices,
                "reference_kappa_after",
            ].to_numpy(float)
            + state_noise,
            0.0,
        )
    frame["observation_seed"] = int(seed)
    return frame


def c3_search_view(frame: pd.DataFrame) -> pd.DataFrame:
    """Return calibration-only columns permitted in fitting and agent tools."""
    observed_stress = [
        f"observed_sigma_{component}_mpa" for component in STRESS_COMPONENTS
    ]
    columns = [
        *IDENTIFIER_COLUMNS,
        *STRAIN_COLUMNS,
        "stress_observed",
        "plastic_shear_observed",
        "observed_kappa",
        "observation_seed",
        "declared_stress_noise_std_mpa",
        "declared_plastic_shear_noise_std",
        *observed_stress,
    ]
    view = frame.loc[frame["partition"] == "calibration", columns].copy()
    forbidden = [
        column
        for column in view.columns
        if column.startswith("reference_") or column.startswith("baseline_")
    ]
    if forbidden:
        raise AssertionError(f"Hidden C3 columns leaked into search view: {forbidden}")
    return view.reset_index(drop=True)


def c3_observation_summary(frame: pd.DataFrame) -> dict[str, Any]:
    stress_rows = frame["stress_observed"].astype(bool)
    state_rows = frame["plastic_shear_observed"].astype(bool)
    return {
        "data_scope": "calibration_only",
        "trajectory_count": int(frame["trajectory_id"].nunique()),
        "row_count": int(len(frame)),
        "stress_observation_count": int(stress_rows.sum()),
        "stress_observed_fraction": float(stress_rows.mean()),
        "plastic_state_observation_count": int(state_rows.sum()),
        "plastic_state_observed_fraction": float(state_rows.mean()),
        "declared_stress_noise_std_mpa": float(
            frame["declared_stress_noise_std_mpa"].iloc[0]
        ),
        "declared_plastic_shear_noise_std": float(
            frame["declared_plastic_shear_noise_std"].iloc[0]
        ),
        "angles_deg": sorted(frame["beta_deg"].astype(float).unique().tolist()),
        "pressures_mpa": sorted(
            frame["confining_pressure_mpa"].astype(float).unique().tolist()
        ),
        "paths": sorted(frame["path"].astype(str).unique().tolist()),
        "observed_stress_components": [
            component
            for component in STRESS_COMPONENTS
            if frame[f"observed_sigma_{component}_mpa"].notna().any()
        ],
        "locked_data_visible": False,
    }

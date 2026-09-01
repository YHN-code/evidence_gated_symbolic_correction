from __future__ import annotations

import numpy as np
import pandas as pd


def make_angle_features(
    df: pd.DataFrame,
    beta_col: str = "beta_deg",
    psi_col: str | None = "psi_deg",
    include_forbidden: bool = False,
) -> pd.DataFrame:
    """Create bounded weak-plane angle features.

    The default feature set avoids endpoint-singular expressions. Forbidden
    terms are available only for unconstrained ablation checks.
    """
    out = pd.DataFrame(index=df.index)
    beta_deg = df[beta_col].astype(float).to_numpy()
    beta = np.deg2rad(beta_deg)
    sin_beta = np.sin(beta)
    cos_beta = np.cos(beta)
    m_beta = np.abs(sin_beta * cos_beta)

    out["beta_deg"] = beta_deg
    out["sin_beta"] = sin_beta
    out["cos_beta"] = cos_beta
    out["sin2_beta"] = sin_beta**2
    out["cos2_beta"] = cos_beta**2
    out["m_beta"] = m_beta
    out["m2_beta"] = m_beta**2
    out["m4_beta"] = m_beta**4
    out["sat_m2_beta_b3"] = out["m2_beta"] / (1.0 + 3.0 * out["m2_beta"])
    out["sin2beta"] = np.sin(2.0 * beta)
    out["cos2beta"] = np.cos(2.0 * beta)

    if include_forbidden:
        eps = 1e-12
        out["tan_beta"] = np.tan(beta)
        out["reciprocal_sin_beta"] = 1.0 / np.where(np.abs(sin_beta) < eps, np.nan, sin_beta)
        out["reciprocal_cos_beta"] = 1.0 / np.where(np.abs(cos_beta) < eps, np.nan, cos_beta)

    if psi_col is not None and psi_col in df.columns:
        psi_deg = df[psi_col].astype(float).to_numpy()
        psi = np.deg2rad(psi_deg)
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        m_psi = np.abs(sin_psi * cos_psi)
        out["psi_deg"] = psi_deg
        out["sin_psi"] = sin_psi
        out["cos_psi"] = cos_psi
        out["sin2_psi"] = sin_psi**2
        out["cos2_psi"] = cos_psi**2
        out["m_psi"] = m_psi
        out["m2_psi"] = m_psi**2
        out["m4_psi"] = m_psi**4
        out["sin2psi"] = np.sin(2.0 * psi)
        out["cos2psi"] = np.cos(2.0 * psi)
        out["m_beta_m_psi"] = m_beta * m_psi
        out["m2_beta_m2_psi"] = out["m2_beta"] * out["m2_psi"]
        out["sin2_beta_sin2_psi"] = out["sin2_beta"] * out["sin2_psi"]
        out["sin2_beta_cos2_psi"] = out["sin2_beta"] * out["cos2_psi"]
        out["cos2_beta_sin2_psi"] = out["cos2_beta"] * out["sin2_psi"]
        out["cos2_beta_cos2_psi"] = out["cos2_beta"] * out["cos2_psi"]
        out["m_beta_sin2_psi"] = out["m_beta"] * out["sin2_psi"]
        out["m_beta_cos2_psi"] = out["m_beta"] * out["cos2_psi"]
        out["m_psi_sin2_beta"] = out["m_psi"] * out["sin2_beta"]
        out["m_psi_cos2_beta"] = out["m_psi"] * out["cos2_beta"]
        out["m2_beta_m_psi"] = out["m2_beta"] * out["m_psi"]
        out["m_beta_m2_psi"] = out["m_beta"] * out["m2_psi"]
        out["m2_beta_sin2_psi"] = out["m2_beta"] * out["sin2_psi"]
        out["m2_beta_cos2_psi"] = out["m2_beta"] * out["cos2_psi"]
        out["m2_psi_sin2_beta"] = out["m2_psi"] * out["sin2_beta"]
        out["m2_psi_cos2_beta"] = out["m2_psi"] * out["cos2_beta"]

        angle_sum = beta + psi
        angle_difference = beta - psi
        for harmonic in (2, 4, 6):
            out[f"sin_{harmonic}_beta_plus_psi"] = np.sin(harmonic * angle_sum)
            out[f"cos_{harmonic}_beta_plus_psi"] = np.cos(harmonic * angle_sum)
            out[f"sin_{harmonic}_beta_minus_psi"] = np.sin(harmonic * angle_difference)
            out[f"cos_{harmonic}_beta_minus_psi"] = np.cos(harmonic * angle_difference)

    return out

from __future__ import annotations

from itertools import combinations


def candidate_terms(feature_names: list[str], mode: str, has_psi: bool) -> list[str]:
    if mode == "baseline":
        return []
    if mode == "rule_based":
        terms = ["m2_beta"]
        if has_psi:
            terms += ["m_beta_m_psi", "m2_psi"]
        else:
            terms += ["sat_m2_beta_b3"]
        return [term for term in terms if term in feature_names]
    if mode in {"asrc", "asrc_no_critic"}:
        preferred = ["m_beta", "m2_beta", "m4_beta", "sat_m2_beta_b3", "sin_beta", "cos_beta"]
        if has_psi:
            preferred += ["m_psi", "m2_psi", "m_beta_m_psi", "sin_psi", "cos_psi"]
        preferred += ["sin2_beta", "cos2_beta"]
        if has_psi:
            preferred += ["sin2_psi", "cos2_psi"]
        return [term for term in preferred if term in feature_names]
    if mode == "generic":
        generic = ["sin_beta", "cos_beta", "sin2_beta", "cos2_beta", "m_beta", "m2_beta", "m4_beta"]
        if has_psi:
            generic += ["sin_psi", "cos_psi", "sin2_psi", "cos2_psi", "m_psi", "m2_psi", "m_beta_m_psi"]
        generic += ["tan_beta", "reciprocal_sin_beta", "reciprocal_cos_beta"]
        return [term for term in generic if term in feature_names]
    raise ValueError(f"Unknown candidate mode: {mode}")


def enumerate_term_sets(terms: list[str], max_terms: int) -> list[list[str]]:
    output: list[list[str]] = [["1"]]
    for size in range(1, max_terms + 1):
        for combo in combinations(terms, size):
            output.append(["1", *combo])
    return output

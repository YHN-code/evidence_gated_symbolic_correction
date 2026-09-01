from __future__ import annotations

import pandas as pd

from asrc.model_revision.mechanism_transfer_benchmarks import (
    build_mechanism_transfer_suite,
)
from asrc.utils.io import project_root, read_yaml
from experiments.run_semantic_evidence_guided_confirmation import (
    NO_ACQUISITION,
    PREDICTIVE,
    SPACE_FILLING,
    _validate_protocol,
    _verdict,
    _workflow_config,
)


CONFIG_PATH = (
    project_root()
    / "configs"
    / "semantic_evidence_guided_revision_confirmation_v1.yaml"
)


def test_confirmation_registry_is_fresh_and_large_enough_for_sign_flip() -> None:
    config = read_yaml(CONFIG_PATH)
    seeds = [int(item) for item in config["confirmation"]["generation_data_seeds"]]
    prior = {int(item) for item in config["prior_development_data_seeds"]}
    suite = build_mechanism_transfer_suite(config, seed=seeds[0])
    _validate_protocol(config, suite)
    assert len(seeds) == 8
    assert not prior.intersection(seeds)
    assert len(seeds) * len(suite) == 32


def test_confirmation_uses_only_registered_acquisition_controls() -> None:
    config = read_yaml(CONFIG_PATH)
    assert _workflow_config(config, NO_ACQUISITION).maximum_new_observations == 0
    assert (
        _workflow_config(config, SPACE_FILLING).acquisition_strategy
        == "space_filling_design"
    )
    assert (
        _workflow_config(config, PREDICTIVE).acquisition_strategy
        == "predictive_disagreement"
    )


def _passing_verdict_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparisons = pd.DataFrame(
        [
            {
                "comparator": NO_ACQUISITION,
                "mean_difference": -0.10,
                "bootstrap_ci_upper": -0.02,
                "one_sided_sign_flip_pvalue": 0.03,
            },
            {
                "comparator": SPACE_FILLING,
                "mean_difference": 0.01,
                "bootstrap_ci_upper": 0.04,
                "one_sided_sign_flip_pvalue": 0.70,
            },
        ]
    )
    results = pd.DataFrame(
        [
            {
                "strategy": PREDICTIVE,
                "stability_violations": 0,
            }
            for _ in range(32)
        ]
    )
    contrasts = pd.DataFrame(
        [
            {
                "initial_gate_triggered": True,
                "predictive_minus_no_acquisition": -0.1,
                "ambiguity_resolved_or_prediction_equivalent": True,
            }
            for _ in range(32)
        ]
    )
    return comparisons, results, contrasts


def test_confirmation_verdict_ignores_descriptive_space_filling_superiority() -> None:
    config = read_yaml(CONFIG_PATH)
    comparisons, results, contrasts = _passing_verdict_frames()
    verdict = _verdict(comparisons, results, contrasts, config)
    assert verdict["confirmation_success"] is True


def test_confirmation_verdict_requires_registered_cluster_pvalue() -> None:
    config = read_yaml(CONFIG_PATH)
    comparisons, results, contrasts = _passing_verdict_frames()
    comparisons.loc[
        comparisons["comparator"].eq(NO_ACQUISITION),
        "one_sided_sign_flip_pvalue",
    ] = 0.08
    verdict = _verdict(comparisons, results, contrasts, config)
    assert verdict["one_sided_sign_flip_pass"] is False
    assert verdict["confirmation_success"] is False

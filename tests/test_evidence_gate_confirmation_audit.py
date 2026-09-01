from __future__ import annotations

import pandas as pd
import pytest

from asrc.model_revision.mechanism_transfer_benchmarks import (
    build_mechanism_transfer_suite,
)
from asrc.utils.io import project_root, read_yaml
from experiments.audit_evidence_gate_confirmation import (
    _exact_candidate_ids,
    _gate_summary,
    _validate_config,
)


def test_audit_config_is_explicitly_post_confirmation() -> None:
    config = read_yaml(
        project_root()
        / "configs"
        / "evidence_gate_confirmation_audit_v1.yaml"
    )

    _validate_config(config)
    assert config["coverage_audit"]["report_as_algorithm_performance"] is False


def test_exact_candidate_ids_use_strict_canonical_expression() -> None:
    config = read_yaml(
        project_root()
        / "configs"
        / "semantic_evidence_guided_revision_confirmation_v1.yaml"
    )
    seed = int(config["confirmation"]["generation_data_seeds"][0])
    item = build_mechanism_transfer_suite(config, seed=seed)[0]

    assert _exact_candidate_ids(item, (item.task.oracle_repair,)) == (
        item.task.oracle_repair.proposal_id,
    )


def test_gate_summary_reports_error_and_observation_cost() -> None:
    frame = pd.DataFrame(
        [
            {
                "strategy": "conditional",
                "task_id": "a",
                "normalized_locked_rmse": 0.2,
                "acquisitions_used": 1,
                "corrected_behavior_recovered": True,
            },
            {
                "strategy": "conditional",
                "task_id": "b",
                "normalized_locked_rmse": 0.4,
                "acquisitions_used": 2,
                "corrected_behavior_recovered": False,
            },
        ]
    )

    row = _gate_summary(frame).iloc[0]
    assert row["mean_normalized_locked_rmse"] == pytest.approx(0.3)
    assert row["total_acquisitions"] == 3
    assert row["behavior_recovery_count"] == 1

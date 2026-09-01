from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from asrc.model_revision import candidate_archives
from asrc.model_revision.benchmarks import build_gate1_suite
from asrc.model_revision.confirmation import (
    integrated_error_by_cell,
    paired_cluster_comparisons,
)
from asrc.utils.io import read_yaml
from experiments.run_symbolic_model_revision_gate1d import _validate_protocol_config


def test_gate1d_protocol_matches_frozen_gate1c_algorithm() -> None:
    config = read_yaml("configs/symbolic_model_revision_gate1d.yaml")
    _validate_protocol_config(config)
    changed = deepcopy(config)
    changed["design"]["maximum_new_observations"] = 9
    with pytest.raises(ValueError, match="maximum_new_observations"):
        _validate_protocol_config(changed)


def test_gate1d_protocol_rejects_candidate_generation_seed() -> None:
    config = read_yaml("configs/symbolic_model_revision_gate1d.yaml")
    changed = deepcopy(config)
    changed["data_seeds"][0] = changed["candidate_archives"]["gate1c"][
        "source_data_seed"
    ]
    with pytest.raises(ValueError, match="unseen"):
        _validate_protocol_config(changed)


def test_integrated_error_uses_the_complete_budget_trajectory() -> None:
    frame = pd.DataFrame(
        {
            "task_id": ["T01"] * 3,
            "data_seed": [11] * 3,
            "strategy": ["joint"] * 3,
            "new_observation_count": [0, 1, 2],
            "locked_rmse": [3.0, 2.0, 1.0],
        }
    )
    result = integrated_error_by_cell(frame)
    assert result.loc[0, "integrated_locked_rmse"] == pytest.approx(2.0)


def test_seed_cluster_comparison_preserves_pairing() -> None:
    rows = []
    for task_id in ("T01", "T02"):
        for data_seed in (11, 12, 13, 14):
            rows.extend(
                [
                    {
                        "task_id": task_id,
                        "data_seed": data_seed,
                        "strategy": "joint",
                        "metric": float(data_seed) - 1.0,
                    },
                    {
                        "task_id": task_id,
                        "data_seed": data_seed,
                        "strategy": "random",
                        "metric": float(data_seed),
                    },
                ]
            )
    result = paired_cluster_comparisons(
        pd.DataFrame(rows),
        treatment="joint",
        comparators=["random"],
        value_column="metric",
        bootstrap_replicates=2000,
        bootstrap_seed=19,
    )
    assert result.loc[0, "mean_difference"] == pytest.approx(-1.0)
    assert result.loc[0, "bootstrap_ci_upper"] == pytest.approx(-1.0)
    assert result.loc[0, "treatment_win_count"] == 8
    assert np.isfinite(result.loc[0, "one_sided_sign_flip_pvalue"])


def test_development_archive_loader_uses_frozen_ranking(
    tmp_path, monkeypatch
) -> None:
    gate1_config = read_yaml("configs/symbolic_model_revision_gate1.yaml")
    task = build_gate1_suite(gate1_config, seed=31)[0]
    source_dir = tmp_path / "source_run"
    (source_dir / "reports").mkdir(parents=True)
    (source_dir / "metrics").mkdir(parents=True)
    (source_dir / "formulas" / "gate1").mkdir(parents=True)
    (source_dir / "reports" / "gate1_pilot_summary.json").write_text(
        json.dumps({"seed": 17, "methods": ["union"]}),
        encoding="utf-8",
    )
    oracle = deepcopy(task.definition.oracle_payload)
    oracle["proposal_id"] = "candidate_oracle"
    alternative = {
        "proposal_id": "candidate_linear",
        "source": "grammar",
        "edit_type": "add_term",
        "target": "model_output",
        "expression": {
            "op": "multiply",
            "arguments": [
                {"op": "parameter", "name": "p0"},
                {"op": "variable", "name": "x1"},
            ],
        },
        "rationale": "Frozen test alternative.",
        "expected_signature": "Linear response.",
    }
    (source_dir / "formulas" / "gate1" / "T01_union_proposals.json").write_text(
        json.dumps({"accepted": [oracle, alternative], "rejected": []}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "task_id": "T01",
                "method": "union",
                "proposal_id": "candidate_linear",
                "source": "grammar",
                "formula": "linear",
                "status": "valid",
                "selection_score": 0.2,
                "validation_rmse": 0.2,
                "complexity": 3,
            },
            {
                "task_id": "T01",
                "method": "union",
                "proposal_id": "candidate_oracle",
                "source": "replay",
                "formula": "quadratic",
                "status": "valid",
                "selection_score": 0.1,
                "validation_rmse": 0.1,
                "complexity": 5,
            },
        ]
    ).to_csv(source_dir / "metrics" / "gate1_candidate_results.csv", index=False)
    monkeypatch.setattr(candidate_archives, "runs_root", lambda: tmp_path)
    selected, audit, metadata = candidate_archives.load_development_matrix_candidates(
        {
            "source_run_id": "source_run",
            "source_data_seed": 17,
            "source_method": "union",
            "task_ids": ["T01"],
            "maximum_structures_per_task": 2,
        },
        {"T01": task},
    )
    assert len(selected["T01"]) == 2
    assert audit["source_proposal_id"].tolist() == [
        "candidate_oracle",
        "candidate_linear",
    ]
    assert len(metadata["candidate_set_sha256"]) == 64

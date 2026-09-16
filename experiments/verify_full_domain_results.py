"""Check frozen Section S12 results without search, fitting, or external APIs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def verify():
    base = ROOT / "data/full_domain"
    data = pd.read_csv(base / "evidence_guided_confirmation_results.csv")
    methods = ["no_acquisition", "conditional_space_filling",
               "evidence_guided_predictive_disagreement"]
    assert set(data.strategy) == set(methods)
    assert len(data) == 96
    assert not data.duplicated(["task_id", "data_seed", "strategy"]).any()
    assert data.task_id.nunique() == 4 and data.data_seed.nunique() == 8
    assert (data.groupby(["task_id", "data_seed"]).size() == 3).all()
    summary = {}
    for method in methods:
        rows = data.loc[data.strategy == method]
        assert len(rows) == 32
        assert np.isclose(rows.normalized_locked_rmse.mean(), 0.02043966, atol=5e-9)
        assert int(rows.corrected_behavior_recovered.sum()) == 31
        assert int(rows.selected_strict_structure.sum()) == 4
        assert int(rows.strict_structure_candidate_covered.sum()) == 11
        assert int(rows.acquisitions_used.sum()) == 0
        assert int(rows.initial_gate_triggered.sum()) == 0
        assert int(rows.stability_violations.sum()) == 0
        summary[method] = {"cases": 32, "mean_normalized_rmse": float(rows.normalized_locked_rmse.mean()),
                           "adequacy": 31, "selected_structure_matches": 4, "added_observations": 0}
    pivot = data.pivot(index=["task_id","data_seed"], columns="strategy", values="normalized_locked_rmse")
    for method in methods[1:]:
        assert np.array_equal(pivot[methods[0]].to_numpy(), pivot[method].to_numpy())
    audit = json.loads((base / "structure_audit.json").read_text())
    for method in methods:
        selected = [r for r in audit["rows"] if r["strategy"] == method
                    and str(r["selected_representative"]).lower() == "true"]
        assert len(selected) == 32
        assert sum(r["status"] == "local_family_identity_certified" for r in selected) == 4
    print(json.dumps({"status": "passed", "scope": "frozen-result consistency only",
                      "methods": summary}, indent=2))
    return summary

if __name__ == "__main__":
    verify()

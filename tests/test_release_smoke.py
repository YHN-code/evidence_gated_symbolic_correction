from __future__ import annotations

import json
from pathlib import Path

import asrc


ROOT = Path(__file__).resolve().parents[1]
FROZEN_FILES = [
    "outputs/runs/semantic_evidence_guided_revision_confirmation_v1/metrics/evidence_guided_confirmation_results.csv",
    "outputs/runs/evidence_gate_confirmation_audit_v1/tables/gate_audit_summary.csv",
    "outputs/runs/gate1d_joint_information_confirmation_v1/metrics/gate1d_confirmation_trajectory.csv",
    "outputs/runs/gate2a_rate_state_external_confirmation_v1/metrics/gate2a_rate_state_trajectory.csv",
    "outputs/runs/constitutive_uj_c1/formulas/constitutive_uj_c1_models.json",
    "outputs/runs/constitutive_subi_c2/formulas/constitutive_subi_c2_models.json",
    "outputs/runs/flac3d_uj_single_zone_validation/data/flac3d_uj_single_zone_comparison.csv",
    "outputs/runs/flac3d_subi_single_zone_validation/data/flac3d_subi_single_zone_comparison.csv",
    "outputs/runs/structural_softening_cavern_confirmation_v3/data/qualified_cavern_three_law_results.csv",
    "outputs/runs/structural_softening_cavern_confirmation_v3/reports/qualified_structural_softening_cavern_gate.json",
    "outputs/runs/structural_ambiguity_resolution_confirmation_v2/checkpoints/structural_ambiguity/seed_263171_noise_1p000_SR03.json",
]


def test_package_imports_from_release_tree() -> None:
    assert Path(asrc.__file__).resolve().is_relative_to(ROOT)


def test_frozen_figure_inputs_are_present() -> None:
    missing = [path for path in FROZEN_FILES if not (ROOT / path).is_file()]
    assert not missing


def test_cavern_gate_is_frozen_as_passed() -> None:
    path = ROOT / "outputs/runs/structural_softening_cavern_confirmation_v3/reports/qualified_structural_softening_cavern_gate.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["response_metrics_admissible"] is True


def test_private_llm_configuration_is_absent() -> None:
    assert not (ROOT / "configs/llm_agent.local.yaml").exists()

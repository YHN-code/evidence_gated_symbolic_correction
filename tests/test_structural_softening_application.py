from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from asrc.constitutive.flac3d_structural_softening_validation import (
    render_flac3d_structural_softening_data_file,
)
from asrc.constitutive.structural_softening_application import (
    StructuralSofteningArtifactError,
    application_parameters,
    build_application_laws,
    load_frozen_structural_softening,
    path_shear_increments,
    replay_material_point,
    structural_softening_cases,
)
from asrc.utils.io import read_yaml
from experiments.run_flac3d_cavern_pilot import (
    build_pilot_cases,
    render_cavern_case,
)
from experiments.run_structural_softening_cavern import (
    V3_TEMPLATE,
    evaluate_cavern_propagation,
)
from experiments.run_structural_softening_cavern_qualification import (
    _require_matching_protocol,
    evaluate_oracle_screen,
    evaluate_pair_qualification,
)


def test_frozen_sr03_artifact_builds_dimensionally_transferred_laws() -> None:
    config = read_yaml("configs/structural_softening_flac3d_validation.yaml")
    frozen = load_frozen_structural_softening(config["frozen_artifact"])
    baseline, discovered, oracle = build_application_laws(
        frozen, config["application"]
    )

    assert frozen.location == "cohesion"
    assert frozen.edit_type == "replace_component"
    assert discovered.family == "exponential_decay"
    assert discovered.amplitude_mpa == pytest.approx(
        1.5 * frozen.amplitude / frozen.baseline_amplitude
    )
    assert discovered.cohesion_from_damage(1.0) == pytest.approx(
        discovered.amplitude_mpa * np.exp(-frozen.rate)
    )
    assert baseline.cohesion_from_damage(1.1) == 0.0
    assert oracle.rate == pytest.approx(2.2)


def test_frozen_artifact_rejects_nonaccepted_or_wrong_location(tmp_path) -> None:
    config = read_yaml("configs/frozen_sr03_structural_softening.yaml")
    payload = copy.deepcopy(config)
    payload["artifact"]["location"] = "model_output"
    path = tmp_path / "invalid.yaml"
    import yaml

    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(StructuralSofteningArtifactError, match="location"):
        load_frozen_structural_softening(path)


def test_discovered_material_point_response_tracks_oracle_better_than_linear() -> None:
    config = read_yaml("configs/structural_softening_flac3d_validation.yaml")
    frozen = load_frozen_structural_softening(config["frozen_artifact"])
    laws = build_application_laws(frozen, config["application"])
    case = structural_softening_cases(config)[0]
    increments = path_shear_increments(config["paths"][case.path])
    parameters = application_parameters(config)
    traces = {
        law.law_id: replay_material_point(law, case, parameters, increments)
        for law in laws
    }
    oracle = traces["oracle_exponential"]["stress_xz_mpa"].to_numpy(float)
    baseline_error = np.sqrt(
        np.mean(
            (traces["baseline_linear"]["stress_xz_mpa"].to_numpy(float) - oracle)
            ** 2
        )
    )
    discovered_error = np.sqrt(
        np.mean(
            (
                traces["discovered_exponential"]["stress_xz_mpa"].to_numpy(float)
                - oracle
            )
            ** 2
        )
    )
    assert discovered_error < 0.05 * baseline_error


def test_flac3d_renderer_contains_three_laws_and_all_case_outputs() -> None:
    config = read_yaml("configs/structural_softening_flac3d_validation.yaml")
    frozen = load_frozen_structural_softening(config["frozen_artifact"])
    laws = build_application_laws(frozen, config["application"])
    text = render_flac3d_structural_softening_data_file(config, laws)

    assert "model precision 10" in text
    assert text.count("zone cmodel assign softening-ubiquitous") == 6
    assert "table 'asrc_baseline_linear'" in text
    assert "table 'asrc_discovered_exponential'" in text
    assert "table 'asrc_oracle_exponential'" in text
    assert text.count("ASRC_CASE_COMPLETE") == 6


def test_cavern_renderer_assigns_frozen_softening_table() -> None:
    config = read_yaml("configs/flac3d_structural_softening_cavern.yaml")
    base_cases, material = build_pilot_cases(config)
    frozen = load_frozen_structural_softening(
        config["structural_softening"]["frozen_artifact"]
    )
    discovered = build_application_laws(
        frozen, config["structural_softening"]["application"]
    )[1]
    case = replace(
        base_cases[0],
        model="softening_discovered_exponential",
        softening_law=discovered,
        softening_maximum_damage=12.0,
        softening_table_point_count=101,
    )

    rendered = render_cavern_case(
        V3_TEMPLATE.read_text(encoding="utf-8"), case, material
    )

    assert "model precision 10" in rendered
    assert "zone cmodel assign softening-ubiquitous" in rendered
    assert "table 'asrc_discovered_exponential'" in rendered
    assert "table-joint-cohesion 'asrc_discovered_exponential'" in rendered
    assert "joint-friction 20" in rendered


def test_cavern_propagation_gate_rewards_discovered_multi_response_match() -> None:
    config = read_yaml("configs/flac3d_structural_softening_cavern.yaml")
    rows = []
    for model, offset in (
        ("softening_baseline_linear", 0.003),
        ("softening_discovered_exponential", 0.0002),
        ("softening_oracle_exponential", 0.0),
    ):
        for stage in range(1, 7):
            scale = stage / 6.0
            rows.append(
                {
                    "scenario_id": "oblique_softening_propagation",
                    "stage": stage,
                    "model": model,
                    "response_max_m": 0.03 * scale + offset,
                    "response_crown_m": 0.02 * scale + offset,
                    "response_wall_convergence_m": 0.015 * scale + offset,
                    "response_invert_m": 0.01 * scale + offset,
                    "plastic_zone_count": 5,
                    "ratio_selected": 9.0e-6,
                }
            )

    metrics, report = evaluate_cavern_propagation(pd.DataFrame(rows), config)

    assert len(metrics) == 2
    assert report["status"] == "passed"
    assert report["discovered_response_improvement_fraction"] > 0.9


def test_cavern_propagation_requires_plasticity_in_every_model() -> None:
    config = read_yaml("configs/flac3d_structural_softening_cavern.yaml")
    rows = []
    for model, offset, plastic_zones in (
        ("softening_baseline_linear", 0.003, 5),
        ("softening_discovered_exponential", 0.0002, 0),
        ("softening_oracle_exponential", 0.0, 5),
    ):
        for stage in range(1, 7):
            scale = stage / 6.0
            rows.append(
                {
                    "scenario_id": "oblique_softening_propagation",
                    "stage": stage,
                    "model": model,
                    "response_max_m": 0.03 * scale + offset,
                    "response_crown_m": 0.02 * scale + offset,
                    "response_wall_convergence_m": 0.015 * scale + offset,
                    "response_invert_m": 0.01 * scale + offset,
                    "plastic_zone_count": plastic_zones,
                    "ratio_selected": 9.0e-6,
                }
            )

    _, report = evaluate_cavern_propagation(pd.DataFrame(rows), config)

    assert report["status"] == "failed"
    assert report["checks"]["plasticity_is_activated"] is False
    assert report["maximum_plastic_zone_count_by_model"][
        "softening_discovered_exponential"
    ] == 0


def _qualification_rows(
    model: str,
    *,
    scenario_id: str,
    stress_mpa: float,
    ratio: float,
    response_offset: float = 0.0,
) -> list[dict[str, object]]:
    rows = []
    for stage in range(1, 7):
        scale = stage / 6.0
        rows.append(
            {
                "scenario_id": scenario_id,
                "stage": stage,
                "model": model,
                "major_stress_mpa": stress_mpa,
                "response_max_m": 0.030 * scale + response_offset,
                "response_crown_m": 0.020 * scale + response_offset,
                "response_wall_convergence_m": 0.015 * scale + response_offset,
                "response_invert_m": 0.010 * scale + response_offset,
                "plastic_zone_count": 8,
                "ratio_selected": ratio,
            }
        )
    return rows


def test_oracle_screen_freezes_highest_converged_plastic_load() -> None:
    config = read_yaml("configs/flac3d_structural_softening_cavern.yaml")
    rows = []
    for stress, ratio in ((12.0, 8.0e-6), (15.0, 9.0e-6), (18.0, 2.0e-4)):
        rows.extend(
            _qualification_rows(
                "softening_oracle_exponential",
                scenario_id=f"oracle_screen_s{stress:g}",
                stress_mpa=stress,
                ratio=ratio,
            )
        )

    metrics, report = evaluate_oracle_screen(pd.DataFrame(rows), config)

    assert report["status"] == "passed"
    assert report["selected_stress_mpa"] == 15.0
    assert report["discovered_law_evaluated"] is False
    assert metrics.set_index("major_stress_mpa").loc[18.0, "qualified"] == 0


def test_v2_oracle_screen_uses_lower_bracketing_loads() -> None:
    config = read_yaml(
        "configs/flac3d_structural_softening_cavern_qualification_v2.yaml"
    )

    protocol = config["qualification_protocol"]
    assert protocol["version"] == "1.1-lower-bracket-after-v1-screen-failure"
    assert protocol["stress_candidates_mpa"] == [6.0, 8.0, 10.0]


def test_v3_oracle_screen_changes_only_cycle_budget_and_screen_protocol() -> None:
    v2 = read_yaml(
        "configs/flac3d_structural_softening_cavern_qualification_v2.yaml"
    )
    v3 = read_yaml(
        "configs/flac3d_structural_softening_cavern_qualification_v3.yaml"
    )

    v2_protocol = v2.pop("qualification_protocol")
    v3_protocol = v3.pop("qualification_protocol")
    v2_cycles = v2["solver"].pop("stage_cycles")
    v3_cycles = v3["solver"].pop("stage_cycles")
    assert v2 == v3
    assert v2_cycles == 60000
    assert v3_cycles == 240000
    assert v2_protocol["stress_candidates_mpa"] == [6.0, 8.0, 10.0]
    assert v3_protocol["stress_candidates_mpa"] == [6.0]
    assert v3_protocol["source_failed_screen_run_id"] == (
        "structural_softening_cavern_oracle_screen_v2"
    )


def test_pair_qualification_uses_joint_multi_response_signal() -> None:
    config = read_yaml("configs/flac3d_structural_softening_cavern.yaml")
    rows = _qualification_rows(
        "softening_baseline_linear",
        scenario_id="pair_qualification_s15",
        stress_mpa=15.0,
        ratio=9.0e-6,
        response_offset=0.003,
    )
    rows.extend(
        _qualification_rows(
            "softening_oracle_exponential",
            scenario_id="pair_qualification_s15",
            stress_mpa=15.0,
            ratio=9.0e-6,
        )
    )

    report = evaluate_pair_qualification(pd.DataFrame(rows), config)

    assert report["status"] == "passed"
    assert report["multi_response_signal_fraction"] > 0.01
    assert report["discovered_law_evaluated"] is False


def test_pair_qualification_rejects_unresolved_signal() -> None:
    config = read_yaml("configs/flac3d_structural_softening_cavern.yaml")
    rows = _qualification_rows(
        "softening_baseline_linear",
        scenario_id="pair_qualification_s15",
        stress_mpa=15.0,
        ratio=9.0e-6,
        response_offset=1.0e-8,
    )
    rows.extend(
        _qualification_rows(
            "softening_oracle_exponential",
            scenario_id="pair_qualification_s15",
            stress_mpa=15.0,
            ratio=9.0e-6,
        )
    )

    report = evaluate_pair_qualification(pd.DataFrame(rows), config)

    assert report["status"] == "failed"
    assert report["checks"]["pair_signal_is_resolved"] is False


def test_pair_qualification_requires_plasticity_in_both_models() -> None:
    config = read_yaml("configs/flac3d_structural_softening_cavern.yaml")
    rows = _qualification_rows(
        "softening_baseline_linear",
        scenario_id="pair_qualification_s6",
        stress_mpa=6.0,
        ratio=9.0e-6,
        response_offset=0.003,
    )
    for row in rows:
        row["plastic_zone_count"] = 0
    rows.extend(
        _qualification_rows(
            "softening_oracle_exponential",
            scenario_id="pair_qualification_s6",
            stress_mpa=6.0,
            ratio=9.0e-6,
        )
    )

    report = evaluate_pair_qualification(pd.DataFrame(rows), config)

    assert report["status"] == "failed"
    assert report["checks"]["plasticity_is_activated"] is False
    assert report["maximum_plastic_zone_count_by_model"][
        "softening_baseline_linear"
    ] == 0


def test_pair_qualification_rejects_missing_stage() -> None:
    config = read_yaml("configs/flac3d_structural_softening_cavern.yaml")
    rows = _qualification_rows(
        "softening_baseline_linear",
        scenario_id="pair_qualification_s15",
        stress_mpa=15.0,
        ratio=9.0e-6,
        response_offset=0.003,
    )
    rows.extend(
        _qualification_rows(
            "softening_oracle_exponential",
            scenario_id="pair_qualification_s15",
            stress_mpa=15.0,
            ratio=9.0e-6,
        )[:-1]
    )

    report = evaluate_pair_qualification(pd.DataFrame(rows), config)

    assert report["status"] == "failed"
    assert report["checks"]["both_models_have_all_six_stages"] is False
    assert report["checks"]["all_stages_converged"] is False


def test_qualification_rejects_changed_protocol_fingerprint() -> None:
    source = {
        "protocol_fingerprint": {
            "config_sha256": "config-a",
            "mesh_sha256": "mesh-a",
            "frozen_artifact_sha256": "artifact-a",
        }
    }
    with pytest.raises(ValueError, match="fingerprint differs"):
        _require_matching_protocol(
            source,
            {
                "config_sha256": "config-a",
                "mesh_sha256": "mesh-b",
                "frozen_artifact_sha256": "artifact-a",
            },
        )

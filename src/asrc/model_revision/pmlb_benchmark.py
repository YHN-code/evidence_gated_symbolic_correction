from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from asrc.model_revision.ast import evaluate_expression
from asrc.model_revision.benchmarks import Gate1Task, Gate1TaskDefinition
from asrc.model_revision.proposals import (
    RepairContract,
    RepairRequest,
    validate_expression_ast,
    validate_typed_repair,
)


class PMLBBenchmarkUnavailableError(RuntimeError):
    """Raised when the optional PMLB dependency or a dataset is unavailable."""


@dataclass(frozen=True)
class PMLBRevisionTask:
    task: Gate1Task
    dataset_name: str
    subset: str
    motif: str
    baseline_normalized_rmse: float
    row_count: int
    feature_count: int

    def metadata_row(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "dataset_name": self.dataset_name,
            "subset": self.subset,
            "motif": self.motif,
            "row_count": self.row_count,
            "feature_count": self.feature_count,
            "baseline_normalized_rmse": self.baseline_normalized_rmse,
        }


def load_pmlb_summary() -> pd.DataFrame:
    try:
        import pmlb
    except ImportError as exc:
        raise PMLBBenchmarkUnavailableError(
            "PMLB support is optional. Install the project with the 'pmlb' extra."
        ) from exc
    summary_path = Path(pmlb.__file__).resolve().parent / "all_summary_stats.tsv"
    if not summary_path.exists():
        raise PMLBBenchmarkUnavailableError(
            f"PMLB summary metadata is missing: {summary_path}"
        )
    return pd.read_csv(summary_path, sep="\t")


def eligible_pmlb_regression_metadata(
    *,
    minimum_rows: int,
    maximum_rows: int,
    minimum_features: int,
    maximum_features: int,
) -> pd.DataFrame:
    summary = load_pmlb_summary()
    selected = summary.loc[
        summary["task"].eq("regression")
        & summary["n_instances"].between(minimum_rows, maximum_rows)
        & summary["n_features"].between(minimum_features, maximum_features)
        & summary["n_categorical_features"].eq(0)
    ].copy()
    selected["benchmark_stratum"] = np.where(
        selected["dataset"].str.contains("_fri_", regex=False),
        "friedman_synthetic",
        "real_black_box",
    )
    return selected.sort_values("dataset").reset_index(drop=True)


def fetch_pmlb_regression_frame(
    dataset_name: str,
    *,
    cache_dir: str | Path,
) -> pd.DataFrame:
    try:
        from pmlb import fetch_data
    except ImportError as exc:
        raise PMLBBenchmarkUnavailableError(
            "PMLB support is optional. Install the project with the 'pmlb' extra."
        ) from exc
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    try:
        frame = fetch_data(dataset_name, local_cache_dir=str(cache_path))
    except Exception as exc:
        raise PMLBBenchmarkUnavailableError(
            f"Unable to fetch PMLB dataset {dataset_name!r}: {exc}"
        ) from exc
    if not isinstance(frame, pd.DataFrame) or "target" not in frame.columns:
        raise PMLBBenchmarkUnavailableError(
            f"PMLB dataset {dataset_name!r} did not return a target DataFrame."
        )
    return frame


def _linear_expression(coefficients: np.ndarray) -> dict[str, Any]:
    terms: list[dict[str, Any]] = [
        {"op": "constant", "value": float(coefficients[0])}
    ]
    for index, coefficient in enumerate(coefficients[1:], start=1):
        terms.append(
            {
                "op": "multiply",
                "arguments": [
                    {"op": "constant", "value": float(coefficient)},
                    {"op": "variable", "name": f"x{index}"},
                ],
            }
        )
    return {"op": "add", "arguments": terms}


def _residual_evidence(
    observed: pd.DataFrame,
    variables: tuple[str, ...],
) -> dict[str, Any]:
    fit = observed.loc[observed["partition"].eq("fit")]
    residual = fit["target"].to_numpy(float) - fit["baseline"].to_numpy(float)
    transforms: dict[str, np.ndarray] = {}
    for name in variables:
        values = fit[name].to_numpy(float)
        transforms[name] = values
        transforms[f"{name}_squared"] = values**2
        transforms[f"abs_{name}"] = np.abs(values)
    for left_index, left in enumerate(variables):
        for right in variables[left_index + 1 :]:
            transforms[f"{left}_times_{right}"] = (
                fit[left].to_numpy(float) * fit[right].to_numpy(float)
            )
    correlations: dict[str, float] = {}
    for name, values in transforms.items():
        if np.std(values) <= 1.0e-12 or np.std(residual) <= 1.0e-12:
            correlations[name] = 0.0
        else:
            correlations[name] = float(np.corrcoef(values, residual)[0, 1])
    probe_indices = np.unique(
        np.linspace(0, len(fit) - 1, min(16, len(fit)), dtype=int)
    )
    probes = []
    for index in probe_indices:
        row = fit.iloc[int(index)]
        probes.append(
            {
                **{name: float(row[name]) for name in variables},
                "baseline": float(row["baseline"]),
                "residual": float(row["target"] - row["baseline"]),
            }
        )
    return {
        "fit_count": int(len(fit)),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "residual_quantiles": {
            "q10": float(np.quantile(residual, 0.10)),
            "q50": float(np.quantile(residual, 0.50)),
            "q90": float(np.quantile(residual, 0.90)),
        },
        "diagnostic_correlations": correlations,
        "fit_only_residual_probes": probes,
    }


def _stable_task_seed(seed: int, dataset_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{dataset_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def build_pmlb_revision_task(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    task_id: str,
    seed: int,
    maximum_candidates: int,
    maximum_rows: int = 2000,
    locked_fraction: float = 0.20,
    audit_fraction: float = 0.20,
    fit_fraction: float = 0.75,
) -> PMLBRevisionTask:
    if not 0.10 <= locked_fraction <= 0.35:
        raise ValueError("locked_fraction must lie in [0.10, 0.35].")
    if not 0.10 <= audit_fraction <= 0.35:
        raise ValueError("audit_fraction must lie in [0.10, 0.35].")
    if not 0.50 <= fit_fraction < 1.0:
        raise ValueError("fit_fraction must lie in [0.50, 1.0).")
    if "target" not in frame.columns:
        raise ValueError("PMLB frame must contain a target column.")

    clean = frame.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    feature_columns = [column for column in clean.columns if column != "target"]
    if not feature_columns:
        raise ValueError("PMLB frame must contain at least one feature.")
    if len(clean) < 100:
        raise ValueError("PMLB revision tasks require at least 100 finite rows.")
    rng = np.random.default_rng(_stable_task_seed(seed, dataset_name))
    if len(clean) > maximum_rows:
        indices = np.sort(rng.choice(len(clean), size=maximum_rows, replace=False))
        clean = clean.iloc[indices].reset_index(drop=True)

    raw_x = clean[feature_columns].to_numpy(float)
    robust_center = np.median(raw_x, axis=0)
    robust_scale = np.quantile(raw_x, 0.75, axis=0) - np.quantile(
        raw_x, 0.25, axis=0
    )
    fallback_scale = np.std(raw_x, axis=0)
    robust_scale = np.where(robust_scale > 1.0e-12, robust_scale, fallback_scale)
    robust_scale = np.where(robust_scale > 1.0e-12, robust_scale, 1.0)
    shell_distance = np.sqrt(
        np.mean(((raw_x - robust_center) / robust_scale) ** 2, axis=1)
    )
    tie_breaker = rng.random(len(clean)) * 1.0e-12
    locked_count = max(20, int(round(len(clean) * locked_fraction)))
    locked_indices = np.argsort(shell_distance + tie_breaker)[-locked_count:]
    locked_mask = np.zeros(len(clean), dtype=bool)
    locked_mask[locked_indices] = True
    available_indices = np.flatnonzero(~locked_mask)
    rng.shuffle(available_indices)
    audit_count = max(20, int(round(len(available_indices) * audit_fraction)))
    audit_indices = available_indices[:audit_count]
    observed_indices = available_indices[audit_count:]
    rng.shuffle(observed_indices)
    fit_count = int(round(len(observed_indices) * fit_fraction))
    fit_indices = observed_indices[:fit_count]
    validation_indices = observed_indices[fit_count:]
    if min(len(fit_indices), len(validation_indices), len(audit_indices)) < 10:
        raise ValueError("PMLB split produced fewer than 10 rows in a public partition.")

    x_center = np.mean(raw_x[observed_indices], axis=0)
    x_scale = np.std(raw_x[observed_indices], axis=0)
    x_scale = np.where(x_scale > 1.0e-12, x_scale, 1.0)
    x = (raw_x - x_center) / x_scale
    raw_y = clean["target"].to_numpy(float)
    y_center = float(np.mean(raw_y[fit_indices]))
    y_scale = float(np.std(raw_y[fit_indices]))
    if y_scale <= 1.0e-12:
        raise ValueError("PMLB target is constant on the fit partition.")
    y = (raw_y - y_center) / y_scale

    design = np.column_stack((np.ones(len(fit_indices)), x[fit_indices]))
    coefficients, *_ = np.linalg.lstsq(design, y[fit_indices], rcond=None)
    baseline_expression_payload = _linear_expression(coefficients)
    variables = tuple(f"x{index}" for index in range(1, x.shape[1] + 1))
    contract = RepairContract(
        allowed_variables=frozenset(variables),
        allowed_targets=frozenset({"response"}),
    )
    baseline_expression = validate_expression_ast(
        baseline_expression_payload,
        contract,
    )
    baseline = evaluate_expression(
        baseline_expression,
        {name: x[:, index] for index, name in enumerate(variables)},
    )

    def partition_frame(indices: np.ndarray, partition: str) -> pd.DataFrame:
        payload = pd.DataFrame(
            {name: x[indices, index] for index, name in enumerate(variables)}
        )
        payload["target"] = y[indices]
        payload["baseline"] = baseline[indices]
        payload["partition"] = partition
        return payload

    observed = pd.concat(
        (
            partition_frame(fit_indices, "fit"),
            partition_frame(validation_indices, "validation"),
        ),
        ignore_index=True,
    )
    audit = partition_frame(audit_indices, "id_test")
    locked = partition_frame(locked_indices, "locked")
    oracle_payload = {
        "proposal_id": "unavailable_black_box_oracle",
        "source": "replay",
        "edit_type": "add_term",
        "target": "response",
        "expression": {"op": "constant", "value": 0.0},
        "rationale": "Black-box task has no declared exact repair.",
        "expected_signature": "Not available for black-box data.",
    }
    oracle_repair = validate_typed_repair(oracle_payload, contract)
    observed_ranges = {
        name: (float(observed[name].min()), float(observed[name].max()))
        for name in variables
    }
    locked_ranges = {
        name: (float(locked[name].min()), float(locked[name].max()))
        for name in variables
    }
    definition = Gate1TaskDefinition(
        task_id=task_id,
        task_family="pmlb_black_box_revision",
        target="response",
        variable_descriptions={
            name: f"anonymous standardized input {index}"
            for index, name in enumerate(variables, start=1)
        },
        observed_ranges=observed_ranges,
        locked_ranges=locked_ranges,
        baseline_expression=baseline_expression,
        oracle_payload=oracle_payload,
        oracle_parameters={},
        noise_std=0.0,
        constraints=(
            "The corrected response must remain finite on the observed domain.",
        ),
    )
    request = RepairRequest(
        baseline_expression=baseline_expression,
        residual_evidence={
            "task_id": task_id,
            "target_kind": "anonymous standardized response",
            "variables": dict(definition.variable_descriptions),
            **_residual_evidence(observed, variables),
        },
        constraints=definition.constraints,
        maximum_candidates=maximum_candidates,
    )
    task = Gate1Task(
        definition=definition,
        contract=contract,
        baseline_expression=baseline_expression,
        oracle_repair=oracle_repair,
        observed=observed,
        locked=locked,
        audit=audit,
        request=request,
    )
    locked_scale = max(float(np.std(locked["target"])), 1.0e-12)
    baseline_normalized_rmse = float(
        np.sqrt(np.mean((locked["target"] - locked["baseline"]) ** 2))
        / locked_scale
    )
    subset = (
        "pmlb_friedman_synthetic"
        if "_fri_" in dataset_name
        else "pmlb_real_black_box"
    )
    return PMLBRevisionTask(
        task=task,
        dataset_name=dataset_name,
        subset=subset,
        motif="black_box_residual",
        baseline_normalized_rmse=baseline_normalized_rmse,
        row_count=len(clean),
        feature_count=len(feature_columns),
    )

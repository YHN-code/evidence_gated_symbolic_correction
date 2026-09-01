from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from asrc.model_revision.ast import (
    ExpressionEvaluationError,
    evaluate_expression,
    expression_node_count,
    expression_to_text,
    parameter_names,
)
from asrc.model_revision.benchmarks import Gate1Task, Gate1TaskDefinition
from asrc.model_revision.proposals import (
    RepairContract,
    RepairRequest,
    validate_expression_ast,
    validate_typed_repair,
)


OFFICIAL_LLM_SRBENCH_REPOSITORY = "nnheui/llm-srbench"
SUPPORTED_SUBSETS = (
    "lsr_synth_bio_pop_growth",
    "lsr_synth_chem_react",
    "lsr_synth_matsci",
    "lsr_synth_phys_osc",
    "lsr_transform",
)
_SUBSET_GROUPS = {
    "lsr_synth_bio_pop_growth": "bio_pop_growth",
    "lsr_synth_chem_react": "chem_react",
    "lsr_synth_matsci": "matsci",
    "lsr_synth_phys_osc": "phys_osc",
}


def _hdf5_group_path(subset: str, instance_id: str) -> str:
    if subset == "lsr_transform":
        return f"/lsr_transform/{instance_id}"
    return f"/lsr_synth/{_SUBSET_GROUPS[subset]}/{instance_id}"


class ExternalBenchmarkUnavailableError(RuntimeError):
    """Raised when optional LLM-SRBench dependencies or data are unavailable."""


class ExternalTaskRejectedError(ValueError):
    """Raised when a record fails a preregistered admissibility rule."""


@dataclass(frozen=True)
class LLMSRBenchRecord:
    instance_id: str
    subset: str
    expression: str
    symbols: tuple[str, ...]
    train_data: np.ndarray
    id_test_data: np.ndarray
    ood_test_data: np.ndarray
    evaluation_regime: str = "official_id_and_ood"

    @property
    def input_count(self) -> int:
        return int(self.train_data.shape[1] - 1)


@dataclass(frozen=True)
class DatasetSnapshot:
    repository: str
    resolved_root: str
    requested_revision: str | None
    hdf5_partition_keys: Mapping[str, str]
    records: tuple[LLMSRBenchRecord, ...]


@dataclass(frozen=True)
class BaselineCorruption:
    operator: str
    selected_path: str
    removed_subtree: str
    baseline_expression: dict[str, Any]


@dataclass(frozen=True)
class ExternalRevisionTask:
    task: Gate1Task
    official_instance_id: str
    subset: str
    original_expression: str
    exact_expression: dict[str, Any]
    corruption: BaselineCorruption
    motif: str
    baseline_normalized_rmse: float
    exact_data_normalized_rmse: float
    calibrated_truth_parameters: Mapping[str, float]
    calibration_fit_normalized_rmse: float
    evaluation_regime: str = "official_id_and_ood"

    def metadata_row(self) -> dict[str, Any]:
        payload = {
            "task_id": self.task.task_id,
            "official_instance_id": self.official_instance_id,
            "subset": self.subset,
            "evaluation_regime": self.evaluation_regime,
            "input_count": len(self.task.variables),
            "motif": self.motif,
            "original_expression": self.original_expression,
            "exact_ast": self.exact_expression,
            "exact_node_count": expression_node_count(self.exact_expression),
            "corruption_operator": self.corruption.operator,
            "corruption_path": self.corruption.selected_path,
            "removed_subtree": self.corruption.removed_subtree,
            "baseline_ast": self.corruption.baseline_expression,
            "baseline_formula": expression_to_text(self.corruption.baseline_expression),
            "baseline_normalized_rmse": self.baseline_normalized_rmse,
            "exact_data_normalized_rmse": self.exact_data_normalized_rmse,
        }
        if self.calibrated_truth_parameters:
            payload.update(
                {
                    "calibrated_truth_parameter_count": len(
                        self.calibrated_truth_parameters
                    ),
                    "calibrated_truth_parameters": dict(
                        self.calibrated_truth_parameters
                    ),
                    "calibration_fit_normalized_rmse": self.calibration_fit_normalized_rmse,
                }
            )
        return payload


@dataclass(frozen=True)
class ExternalTaskSelection:
    tasks: tuple[ExternalRevisionTask, ...]
    audit: tuple[dict[str, Any], ...]


def _as_2d_float(values: Any, field_name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[0] < 2 or result.shape[1] < 2:
        raise ExternalTaskRejectedError(
            f"{field_name} must be a two-dimensional output-plus-input array."
        )
    if not np.all(np.isfinite(result)):
        raise ExternalTaskRejectedError(f"{field_name} contains non-finite values.")
    return result


def _metadata_value(entry: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in entry:
            return entry[name]
    raise ExternalTaskRejectedError(
        f"Metadata is missing every supported field in {names!r}."
    )


def _read_hdf5_partition(
    group: Any,
    canonical_name: str,
    aliases: Sequence[str],
) -> tuple[np.ndarray, str]:
    for key in aliases:
        if key in group:
            return _as_2d_float(group[key][...], canonical_name), str(key)
    available = sorted(str(key) for key in group.keys())
    raise ExternalBenchmarkUnavailableError(
        f"HDF5 group is missing {canonical_name!r}; supported keys are "
        f"{list(aliases)!r}, available keys are {available!r}."
    )


def _split_transform_test(
    values: np.ndarray,
    *,
    instance_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Split one official LSR-Transform test set into two IID audits.

    The split is deterministic and disjoint.  It does not create an OOD set;
    callers must retain the ``official_test_hash_split_iid`` regime label.
    """

    samples = _as_2d_float(values, "test_data")
    if len(samples) < 4:
        raise ExternalBenchmarkUnavailableError(
            "LSR-Transform test data requires at least four rows."
        )
    digest = hashlib.sha256(str(instance_id).encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    permutation = rng.permutation(len(samples))
    midpoint = len(samples) // 2
    return samples[permutation[:midpoint]], samples[permutation[midpoint:]]


def _dataset_access_error_message(error: Exception) -> str:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    detail = str(error).lower()
    dataset_url = "https://huggingface.co/datasets/nnheui/llm-srbench"
    if status_code == 403 or "authorized list" in detail or "403 client" in detail:
        return (
            "Hugging Face authentication succeeded, but this account is not yet "
            "authorized for official LLM-SRBench. While signed in with the same "
            f"account used by `hf auth login`, visit {dataset_url}, submit/accept "
            "the gated-data access form, and wait until access is granted. Repeated "
            "downloads cannot bypass a 403 response."
        )
    if status_code == 401 or "401 client" in detail or "unauthorized" in detail:
        return (
            "Hugging Face did not receive a valid login for official LLM-SRBench. "
            f"Visit {dataset_url}, accept its access conditions, run `hf auth login`, "
            "and retry."
        )
    return (
        "Unable to load official LLM-SRBench from Hugging Face. Verify network "
        f"access and the dataset authorization at {dataset_url}."
    )


def load_official_llm_srbench(
    *,
    cache_dir: str | Path,
    subsets: Sequence[str] = SUPPORTED_SUBSETS,
    revision: str | None = None,
    token: str | None = None,
) -> DatasetSnapshot:
    """Load official metadata and HDF5 samples from a local HF cache."""

    unknown = sorted(set(subsets) - set(SUPPORTED_SUBSETS))
    if unknown:
        raise ValueError(f"Unsupported LLM-SRBench subsets: {unknown}")
    try:
        import datasets
        import h5py
        from huggingface_hub import snapshot_download
    except (ImportError, ModuleNotFoundError) as exc:
        raise ExternalBenchmarkUnavailableError(
            'Install optional dependencies with python -m pip install -e ".[external-benchmark]".'
        ) from exc

    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    snapshot_kwargs: dict[str, Any] = {
        "repo_id": OFFICIAL_LLM_SRBENCH_REPOSITORY,
        "repo_type": "dataset",
        "cache_dir": str(cache_path),
    }
    dataset_kwargs: dict[str, Any] = {
        "path": OFFICIAL_LLM_SRBENCH_REPOSITORY,
        "cache_dir": str(cache_path),
    }
    if revision:
        snapshot_kwargs["revision"] = revision
        dataset_kwargs["revision"] = revision
    if token:
        snapshot_kwargs["token"] = token
        dataset_kwargs["token"] = token
    try:
        resolved_root = Path(snapshot_download(**snapshot_kwargs))
        metadata = datasets.load_dataset(**dataset_kwargs)
    except Exception as exc:
        raise ExternalBenchmarkUnavailableError(
            _dataset_access_error_message(exc)
        ) from exc

    hdf5_path = resolved_root / "lsr_bench_data.hdf5"
    if not hdf5_path.exists():
        raise ExternalBenchmarkUnavailableError(
            f"Official snapshot does not contain {hdf5_path.name}."
        )
    records: list[LLMSRBenchRecord] = []
    observed_schemas: set[tuple[tuple[str, str], ...]] = set()
    with h5py.File(hdf5_path, "r") as sample_file:
        for subset in subsets:
            if subset not in metadata:
                raise ExternalBenchmarkUnavailableError(
                    f"Official metadata does not contain split {subset!r}."
                )
            for entry in metadata[subset]:
                instance_id = str(_metadata_value(entry, "name", "instance_id"))
                expression = str(_metadata_value(entry, "expression", "gt_expression"))
                symbols = tuple(str(item) for item in entry["symbols"])
                group_path = _hdf5_group_path(subset, instance_id)
                if group_path not in sample_file:
                    raise ExternalBenchmarkUnavailableError(
                        f"HDF5 samples are missing {group_path}."
                    )
                group = sample_file[group_path]
                train_data, train_key = _read_hdf5_partition(
                    group, "train_data", ("train_data", "train")
                )
                if subset == "lsr_transform":
                    official_test, official_test_key = _read_hdf5_partition(
                        group, "test_data", ("test",)
                    )
                    id_test_data, ood_test_data = _split_transform_test(
                        official_test,
                        instance_id=instance_id,
                    )
                    id_test_key = f"{official_test_key}:hash_half_1"
                    ood_test_key = f"{official_test_key}:hash_half_2"
                    evaluation_regime = "official_test_hash_split_iid"
                else:
                    id_test_data, id_test_key = _read_hdf5_partition(
                        group, "id_test_data", ("id_test_data", "test")
                    )
                    ood_test_data, ood_test_key = _read_hdf5_partition(
                        group, "ood_test_data", ("ood_test_data", "ood_test")
                    )
                    evaluation_regime = "official_id_and_ood"
                observed_schemas.add(
                    tuple(
                        sorted(
                            {
                                "train_data": train_key,
                                "id_test_data": id_test_key,
                                "ood_test_data": ood_test_key,
                            }.items()
                        )
                    )
                )
                records.append(
                    LLMSRBenchRecord(
                        instance_id=instance_id,
                        subset=subset,
                        expression=expression,
                        symbols=symbols,
                        train_data=train_data,
                        id_test_data=id_test_data,
                        ood_test_data=ood_test_data,
                        evaluation_regime=evaluation_regime,
                    )
                )
    if len(observed_schemas) != 1:
        raise ExternalBenchmarkUnavailableError(
            f"Official HDF5 snapshot uses inconsistent partition schemas: {observed_schemas!r}."
        )
    hdf5_partition_keys = dict(next(iter(observed_schemas)))
    return DatasetSnapshot(
        repository=OFFICIAL_LLM_SRBENCH_REPOSITORY,
        resolved_root=str(resolved_root),
        requested_revision=revision,
        hdf5_partition_keys=hdf5_partition_keys,
        records=tuple(records),
    )


def expression_string_to_ast(
    expression: str,
    input_symbols: Sequence[str],
    aliases: Sequence[str],
    *,
    allow_free_parameters: bool = False,
) -> dict[str, Any]:
    """Parse a trusted benchmark expression into the bounded public AST."""

    if len(input_symbols) != len(aliases):
        raise ValueError("input_symbols and aliases must have equal length.")
    try:
        import sympy
    except (ImportError, ModuleNotFoundError) as exc:
        raise ExternalBenchmarkUnavailableError("SymPy is required.") from exc

    internal_symbols = {
        str(name): sympy.Symbol(f"asrc_input_{index}", real=True)
        for index, name in enumerate(input_symbols)
    }
    normalized = (
        str(expression).strip().replace("^", "**").replace("π", "pi").replace("·", "*")
    )
    if "=" in normalized:
        normalized = normalized.split("=", 1)[1].strip()
    # Dynamical-system metadata writes sampled state columns as P(t), x(t), etc.
    # The HDF5 arrays already provide those states as independent input columns.
    argument_pattern = "|".join(
        sorted((re.escape(str(name)) for name in input_symbols), key=len, reverse=True)
    )
    if argument_pattern:
        for name in sorted((str(item) for item in input_symbols), key=len, reverse=True):
            normalized = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\(\s*(?:{argument_pattern})\s*\)",
                name,
                normalized,
            )

    # Some official LSR-Synth expressions retain generated coefficient labels such
    # as ``0.18_s``. They are free scientific constants, not numeric literals.
    numeric_suffix_pattern = re.compile(
        r"(?<![A-Za-z0-9_.])(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?_[A-Za-z][A-Za-z0-9_]*"
    )
    coefficient_tokens = tuple(dict.fromkeys(numeric_suffix_pattern.findall(normalized)))
    coefficient_aliases = {
        token: f"asrc_free_coefficient_{index}"
        for index, token in enumerate(coefficient_tokens)
    }
    for token in sorted(coefficient_aliases, key=len, reverse=True):
        normalized = normalized.replace(token, coefficient_aliases[token])

    function_names = {
        "ln",
        "abs",
        "Abs",
        "sin",
        "cos",
        "exp",
        "log",
        "tanh",
        "sqrt",
        "pi",
        "E",
    }
    identifiers = tuple(dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", normalized)))
    free_names = tuple(
        name
        for name in identifiers
        if name not in internal_symbols and name not in function_names
    )
    free_symbols = {
        name: sympy.Symbol(f"asrc_parameter_{index}", real=True)
        for index, name in enumerate(free_names)
    }
    local_symbols = {
        **internal_symbols,
        **free_symbols,
        "ln": sympy.log,
        "abs": sympy.Abs,
        "Abs": sympy.Abs,
        "sqrt": sympy.sqrt,
    }
    try:
        parsed = sympy.sympify(normalized, locals=local_symbols, evaluate=False)
    except Exception as exc:
        raise ExternalTaskRejectedError(f"Expression parsing failed: {exc}") from exc
    symbol_to_alias = {
        symbol: str(alias) for symbol, alias in zip(internal_symbols.values(), aliases)
    }
    symbol_to_parameter = {
        symbol: f"truth_parameter_{index}"
        for index, symbol in enumerate(free_symbols.values())
    }

    def convert(node: Any) -> dict[str, Any]:
        if isinstance(node, sympy.Symbol):
            if node in symbol_to_alias:
                return {"op": "variable", "name": symbol_to_alias[node]}
            if node in symbol_to_parameter and allow_free_parameters:
                return {"op": "parameter", "name": symbol_to_parameter[node]}
            raise ExternalTaskRejectedError(
                f"Expression contains unsupported free symbol {node!s}."
            )
        if bool(getattr(node, "is_number", False)) and not node.free_symbols:
            value = float(node)
            if not math.isfinite(value):
                raise ExternalTaskRejectedError("Expression has a non-finite constant.")
            return {"op": "constant", "value": value}
        if isinstance(node, sympy.Add):
            return {"op": "add", "arguments": [convert(item) for item in node.args]}
        if isinstance(node, sympy.Mul):
            return {"op": "multiply", "arguments": [convert(item) for item in node.args]}
        if isinstance(node, sympy.Pow):
            return {"op": "power", "left": convert(node.args[0]), "right": convert(node.args[1])}
        function_map = {
            sympy.sin: "sin",
            sympy.cos: "cos",
            sympy.exp: "exp",
            sympy.log: "log",
            sympy.tanh: "tanh",
            sympy.Abs: "abs",
        }
        if node.func in function_map and len(node.args) == 1:
            return {"op": function_map[node.func], "argument": convert(node.args[0])}
        raise ExternalTaskRejectedError(
            f"Expression uses unsupported operation {node.func!s}."
        )

    return convert(parsed)


def _substitute_expression_parameters(
    expression: dict[str, Any],
    values: Mapping[str, float],
) -> dict[str, Any]:
    operation = str(expression["op"])
    if operation == "parameter":
        name = str(expression["name"])
        if name not in values:
            raise ExternalTaskRejectedError(
                f"Calibrated truth parameter {name!r} is unavailable."
            )
        return {"op": "constant", "value": float(values[name])}
    if operation in {"variable", "constant", "baseline"}:
        return dict(expression)
    if operation in {"negate", "abs", "exp", "log", "sin", "cos", "tanh"}:
        return {
            "op": operation,
            "argument": _substitute_expression_parameters(expression["argument"], values),
        }
    if operation in {"subtract", "divide", "power"}:
        return {
            "op": operation,
            "left": _substitute_expression_parameters(expression["left"], values),
            "right": _substitute_expression_parameters(expression["right"], values),
        }
    return {
        "op": operation,
        "arguments": [
            _substitute_expression_parameters(item, values)
            for item in expression["arguments"]
        ],
    }


def _calibrate_truth_parameters(
    expression: dict[str, Any],
    fit: pd.DataFrame,
    variables: tuple[str, ...],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, float], float]:
    names = tuple(parameter_names(expression))
    if not names:
        prediction = evaluate_expression(
            expression,
            {name: fit[name].to_numpy(float) for name in variables},
        )
        return expression, {}, _normalized_rmse(fit["target"].to_numpy(float), prediction)

    target = fit["target"].to_numpy(float)
    inputs = {name: fit[name].to_numpy(float) for name in variables}
    scale = max(float(np.std(target)), 1.0e-12)

    def residual(vector: np.ndarray) -> np.ndarray:
        parameters = {name: float(vector[index]) for index, name in enumerate(names)}
        try:
            prediction = evaluate_expression(expression, inputs, parameters)
        except ExpressionEvaluationError:
            return np.full(len(fit), 1.0e6, dtype=float)
        return (prediction - target) / scale

    rng = np.random.default_rng(int(seed))
    starts = [
        np.zeros(len(names), dtype=float),
        np.ones(len(names), dtype=float),
        np.full(len(names), 0.1, dtype=float),
        np.full(len(names), -1.0, dtype=float),
    ]
    starts.extend(rng.uniform(-3.0, 3.0, len(names)) for _ in range(12))
    lower = np.full(len(names), -100.0, dtype=float)
    upper = np.full(len(names), 100.0, dtype=float)
    best_vector: np.ndarray | None = None
    best_loss = float("inf")
    for start in starts:
        try:
            result = least_squares(
                residual,
                np.clip(start, lower + 1.0e-10, upper - 1.0e-10),
                bounds=(lower, upper),
                max_nfev=3000,
                x_scale="jac",
                xtol=1.0e-12,
                ftol=1.0e-12,
                gtol=1.0e-12,
            )
        except (ValueError, FloatingPointError):
            continue
        loss = float(np.mean(residual(result.x) ** 2))
        if np.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_vector = np.asarray(result.x, dtype=float)
    if best_vector is None:
        raise ExternalTaskRejectedError("Fit-only calibration of truth constants failed.")
    parameters = {
        name: float(best_vector[index]) for index, name in enumerate(names)
    }
    calibrated = _substitute_expression_parameters(expression, parameters)
    return calibrated, parameters, float(np.sqrt(best_loss))


def _children(node: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    operation = node["op"]
    if operation in {"negate", "abs", "exp", "log", "sin", "cos", "tanh"}:
        return [("argument", node["argument"])]
    if operation in {"subtract", "divide", "power"}:
        return [("left", node["left"]), ("right", node["right"])]
    if operation in {"add", "multiply"}:
        return [(f"arguments[{index}]", child) for index, child in enumerate(node["arguments"])]
    return []


def _contains_variable(node: dict[str, Any]) -> bool:
    return node["op"] == "variable" or any(
        _contains_variable(child) for _, child in _children(node)
    )


def _simplify_neutral(node: dict[str, Any]) -> dict[str, Any]:
    operation = node["op"]
    if operation in {"variable", "parameter", "constant", "baseline"}:
        return dict(node)
    if operation in {"negate", "abs", "exp", "log", "sin", "cos", "tanh"}:
        return {"op": operation, "argument": _simplify_neutral(node["argument"])}
    if operation in {"subtract", "divide", "power"}:
        return {
            "op": operation,
            "left": _simplify_neutral(node["left"]),
            "right": _simplify_neutral(node["right"]),
        }
    arguments = [_simplify_neutral(item) for item in node["arguments"]]
    neutral = 0.0 if operation == "add" else 1.0
    retained = [
        item
        for item in arguments
        if not (item["op"] == "constant" and float(item["value"]) == neutral)
    ]
    if not retained:
        return {"op": "constant", "value": neutral}
    if len(retained) == 1:
        return retained[0]
    return {"op": operation, "arguments": retained}


def neutralize_proper_subtree(
    expression: dict[str, Any],
    *,
    task_key: str,
) -> BaselineCorruption:
    """Mechanically remove one nonterminal additive or multiplicative child."""

    candidates: list[tuple[tuple[str, ...], str, dict[str, Any]]] = []

    def visit(node: dict[str, Any], path: tuple[str, ...]) -> None:
        operation = node["op"]
        for slot, child in _children(node):
            child_path = (*path, slot)
            if (
                operation in {"add", "multiply"}
                and expression_node_count(child) >= 2
                and _contains_variable(child)
            ):
                candidates.append((child_path, operation, child))
            visit(child, child_path)

    visit(expression, ())
    if not candidates:
        raise ExternalTaskRejectedError(
            "Expression has no admissible nonterminal subtree for neutralization."
        )
    candidates.sort(key=lambda item: (len(item[0]), item[0]))
    digest = hashlib.sha256(str(task_key).encode("utf-8")).digest()
    selected_path, parent_operation, removed = candidates[
        int.from_bytes(digest[:8], "big") % len(candidates)
    ]

    def replace(node: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
        if not path:
            return {"op": "constant", "value": 0.0 if parent_operation == "add" else 1.0}
        head, *tail = path
        result = dict(node)
        if head.startswith("arguments["):
            index = int(head[len("arguments[") : -1])
            arguments = [dict(item) for item in node["arguments"]]
            arguments[index] = replace(arguments[index], tuple(tail))
            result["arguments"] = arguments
        else:
            result[head] = replace(node[head], tuple(tail))
        return result

    baseline = _simplify_neutral(replace(expression, selected_path))
    return BaselineCorruption(
        operator="deterministic_proper_subtree_neutralization_v1",
        selected_path="/".join(selected_path),
        removed_subtree=expression_to_text(removed),
        baseline_expression=baseline,
    )


def classify_expression_motif(expression: dict[str, Any]) -> str:
    operators: list[str] = []

    def visit(node: dict[str, Any]) -> None:
        operators.append(str(node["op"]))
        for _, child in _children(node):
            visit(child)

    visit(expression)
    if any(item in operators for item in ("sin", "cos", "tanh")):
        return "bounded_or_periodic"
    if any(item in operators for item in ("exp", "log")):
        return "exponential_or_logarithmic"
    if "power" in operators:
        return "power_law"
    if operators.count("variable") >= 2 and "multiply" in operators:
        return "interaction"
    return "algebraic"


def _normalized_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    scale = max(float(np.std(observed)), 1.0e-12)
    return float(np.sqrt(np.mean((observed - predicted) ** 2)) / scale)


def _residual_evidence(observed: pd.DataFrame, variables: tuple[str, ...]) -> dict[str, Any]:
    fit = observed.loc[observed["partition"].eq("fit")].copy()
    residual = fit["target"].to_numpy(float) - fit["baseline"].to_numpy(float)
    diagnostics: dict[str, np.ndarray] = {}
    for name in variables:
        values = fit[name].to_numpy(float)
        diagnostics[name] = values
        diagnostics[f"{name}_squared"] = values**2
        diagnostics[f"abs_{name}"] = np.abs(values)
    for left_index, left in enumerate(variables):
        for right in variables[left_index + 1 :]:
            diagnostics[f"{left}_times_{right}"] = (
                fit[left].to_numpy(float) * fit[right].to_numpy(float)
            )
    correlations = {}
    for name, values in diagnostics.items():
        if np.std(values) <= 1.0e-12 or np.std(residual) <= 1.0e-12:
            correlations[name] = 0.0
        else:
            correlations[name] = float(np.corrcoef(values, residual)[0, 1])
    ordered = fit.sort_values(list(variables)).reset_index(drop=True)
    indices = np.unique(np.linspace(0, len(ordered) - 1, min(16, len(ordered)), dtype=int))
    probes = [
        {
            **{name: float(ordered.iloc[index][name]) for name in variables},
            "baseline": float(ordered.iloc[index]["baseline"]),
            "residual": float(ordered.iloc[index]["target"] - ordered.iloc[index]["baseline"]),
        }
        for index in indices
    ]
    return {
        "fit_count": len(fit),
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


def _frame_from_samples(samples: np.ndarray, variables: tuple[str, ...]) -> pd.DataFrame:
    if samples.shape[1] != len(variables) + 1:
        raise ExternalTaskRejectedError(
            "Sample width does not match the output-plus-input symbol count."
        )
    frame = pd.DataFrame(samples[:, 1:], columns=list(variables))
    frame["target"] = samples[:, 0]
    return frame


def _deterministic_subsample(
    frame: pd.DataFrame,
    *,
    maximum_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    if maximum_rows is None or len(frame) <= int(maximum_rows):
        return frame.reset_index(drop=True)
    if int(maximum_rows) < 2:
        raise ExternalTaskRejectedError("Sample caps must be at least two.")
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(len(frame), size=int(maximum_rows), replace=False))
    return frame.iloc[indices].reset_index(drop=True)


def build_external_revision_task(
    record: LLMSRBenchRecord,
    *,
    fit_fraction: float,
    split_seed: int,
    maximum_candidates: int,
    minimum_baseline_normalized_rmse: float,
    exact_data_tolerance: float,
    minimum_nodes: int,
    maximum_nodes: int,
    minimum_inputs: int,
    maximum_inputs: int,
    calibrate_free_constants: bool = False,
    maximum_train_samples: int | None = None,
    maximum_id_test_samples: int | None = None,
    maximum_locked_samples: int | None = None,
) -> ExternalRevisionTask:
    if not minimum_inputs <= record.input_count <= maximum_inputs:
        raise ExternalTaskRejectedError(
            f"input_count={record.input_count} is outside the registered range."
        )
    if len(record.symbols) != record.input_count + 1:
        raise ExternalTaskRejectedError(
            "Symbol count does not match the output-plus-input sample width."
        )
    variables = tuple(f"x{index}" for index in range(1, record.input_count + 1))
    parameterized_expression = expression_string_to_ast(
        record.expression,
        record.symbols[1:],
        variables,
        allow_free_parameters=bool(calibrate_free_constants),
    )
    node_count = expression_node_count(parameterized_expression)
    if not minimum_nodes <= node_count <= maximum_nodes:
        raise ExternalTaskRejectedError(
            f"exact_node_count={node_count} is outside the registered range."
        )
    contract = RepairContract(
        allowed_variables=frozenset(variables),
        allowed_targets=frozenset({"response"}),
        maximum_depth=12,
        maximum_nodes=max(64, maximum_nodes + 16),
    )
    parameterized_expression = validate_expression_ast(parameterized_expression, contract)

    train = _deterministic_subsample(
        _frame_from_samples(record.train_data, variables),
        maximum_rows=maximum_train_samples,
        seed=int(split_seed) + 11,
    )
    id_test = _deterministic_subsample(
        _frame_from_samples(record.id_test_data, variables),
        maximum_rows=maximum_id_test_samples,
        seed=int(split_seed) + 23,
    )
    ood_test = _deterministic_subsample(
        _frame_from_samples(record.ood_test_data, variables),
        maximum_rows=maximum_locked_samples,
        seed=int(split_seed) + 37,
    )
    rng = np.random.default_rng(int(split_seed))
    permutation = rng.permutation(len(train))
    fit_count = min(len(train) - 1, max(2, int(round(float(fit_fraction) * len(train)))))
    train["partition"] = "validation"
    train.loc[permutation[:fit_count], "partition"] = "fit"
    calibration_fit = train.loc[train["partition"].eq("fit")]
    expression, calibrated_parameters, calibration_fit_error = (
        _calibrate_truth_parameters(
            parameterized_expression,
            calibration_fit,
            variables,
            seed=int(split_seed) + 65537,
        )
    )
    expression = validate_expression_ast(expression, contract)
    corruption = neutralize_proper_subtree(expression, task_key=record.instance_id)
    baseline_expression = validate_expression_ast(corruption.baseline_expression, contract)
    train_variables = {name: train[name].to_numpy(float) for name in variables}
    exact_train = evaluate_expression(expression, train_variables)
    exact_error = _normalized_rmse(train["target"].to_numpy(float), exact_train)
    if exact_error > exact_data_tolerance:
        raise ExternalTaskRejectedError(
            f"exact_data_normalized_rmse={exact_error:.6g} exceeds tolerance."
        )
    baseline_train = evaluate_expression(baseline_expression, train_variables)
    baseline_error = _normalized_rmse(train["target"].to_numpy(float), baseline_train)
    if baseline_error < minimum_baseline_normalized_rmse:
        raise ExternalTaskRejectedError(
            f"baseline_normalized_rmse={baseline_error:.6g} is too small."
        )

    id_test["partition"] = "id_test"
    ood_test["partition"] = "locked"
    for frame in (train, id_test, ood_test):
        frame_variables = {name: frame[name].to_numpy(float) for name in variables}
        frame["baseline"] = evaluate_expression(baseline_expression, frame_variables)

    ranges = {name: (float(train[name].min()), float(train[name].max())) for name in variables}
    locked_ranges = {name: (float(ood_test[name].min()), float(ood_test[name].max())) for name in variables}
    oracle_payload = {
        "proposal_id": "official_exact_expression",
        "source": "replay",
        "edit_type": "replace_subtree",
        "target": "response",
        "expression": expression,
        "rationale": "Official expression retained for post-selection audit only.",
        "expected_signature": "Official expression retained for audit only.",
    }
    scale = max(float(np.std(train["target"])), 1.0)
    definition = Gate1TaskDefinition(
        task_id=record.instance_id,
        task_family=f"llm_srbench_{record.subset}",
        target="response",
        variable_descriptions={
            name: f"anonymous dimensionless input coordinate {index}"
            for index, name in enumerate(variables, start=1)
        },
        observed_ranges=ranges,
        locked_ranges=locked_ranges,
        baseline_expression=baseline_expression,
        oracle_payload=oracle_payload,
        oracle_parameters={},
        noise_std=max(
            float(np.sqrt(np.mean((train["target"].to_numpy(float) - exact_train) ** 2))),
            1.0e-10 * scale,
        ),
    )
    request = RepairRequest(
        baseline_expression=baseline_expression,
        residual_evidence=_residual_evidence(train, variables),
        constraints=("Corrected response must remain finite on supplied data.",),
        maximum_candidates=int(maximum_candidates),
    )
    task = Gate1Task(
        definition=definition,
        contract=contract,
        baseline_expression=baseline_expression,
        oracle_repair=validate_typed_repair(oracle_payload, contract),
        observed=train,
        locked=ood_test,
        audit=id_test,
        request=request,
    )
    return ExternalRevisionTask(
        task=task,
        official_instance_id=record.instance_id,
        subset=record.subset,
        original_expression=record.expression,
        exact_expression=expression,
        corruption=corruption,
        motif=classify_expression_motif(expression),
        baseline_normalized_rmse=baseline_error,
        exact_data_normalized_rmse=exact_error,
        calibrated_truth_parameters=calibrated_parameters,
        calibration_fit_normalized_rmse=calibration_fit_error,
        evaluation_regime=record.evaluation_regime,
    )


def select_external_revision_tasks(
    records: Iterable[LLMSRBenchRecord],
    *,
    task_count: int,
    selection_seed: int,
    build_options: Mapping[str, Any],
    excluded_task_ids: Iterable[str] = (),
) -> ExternalTaskSelection:
    """Apply fixed admissibility rules and balance domains and motifs greedily."""

    if task_count < 1:
        raise ValueError("task_count must be positive.")
    excluded_ids = frozenset(str(item) for item in excluded_task_ids)
    eligible: list[ExternalRevisionTask] = []
    admissible_ids: set[str] = set()
    observed_ids: set[str] = set()
    audit: list[dict[str, Any]] = []
    ordered_records = sorted(records, key=lambda item: (item.subset, item.instance_id))
    for index, record in enumerate(ordered_records):
        observed_ids.add(record.instance_id)
        row: dict[str, Any] = {
            "official_instance_id": record.instance_id,
            "subset": record.subset,
            "selected": False,
            "admissible": False,
            "rejection_reason": "",
            "excluded_from_selection": record.instance_id in excluded_ids,
            "exclusion_reason": (
                "frozen_development_task"
                if record.instance_id in excluded_ids
                else ""
            ),
        }
        try:
            task = build_external_revision_task(
                record,
                split_seed=int(selection_seed) + 1009 * index,
                **dict(build_options),
            )
        except (ExternalTaskRejectedError, ValueError) as exc:
            row["rejection_reason"] = str(exc)
        else:
            admissible_ids.add(record.instance_id)
            row.update(
                {
                    "admissible": True,
                    "motif": task.motif,
                    "input_count": len(task.task.variables),
                    "exact_node_count": expression_node_count(task.exact_expression),
                    "baseline_normalized_rmse": task.baseline_normalized_rmse,
                }
            )
            if record.instance_id not in excluded_ids:
                eligible.append(task)
        audit.append(row)

    missing_exclusions = sorted(excluded_ids - observed_ids)
    if missing_exclusions:
        raise ExternalTaskRejectedError(
            f"Excluded task ids are absent from the requested dataset: {missing_exclusions}."
        )
    inadmissible_exclusions = sorted(excluded_ids - admissible_ids)
    if inadmissible_exclusions:
        raise ExternalTaskRejectedError(
            "Frozen development tasks no longer satisfy the confirmation "
            f"admissibility contract: {inadmissible_exclusions}."
        )

    if len(eligible) < task_count:
        raise ExternalTaskRejectedError(
            f"Only {len(eligible)} admissible tasks remain; {task_count} requested."
        )

    def stable_rank(task: ExternalRevisionTask) -> str:
        value = f"{selection_seed}:{task.official_instance_id}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    remaining = sorted(eligible, key=stable_rank)
    selected: list[ExternalRevisionTask] = []
    domain_counts: dict[str, int] = {}
    motif_counts: dict[str, int] = {}
    while remaining and len(selected) < task_count:
        best = min(
            remaining,
            key=lambda item: (
                domain_counts.get(item.subset, 0),
                motif_counts.get(item.motif, 0),
                stable_rank(item),
            ),
        )
        selected.append(best)
        remaining.remove(best)
        domain_counts[best.subset] = domain_counts.get(best.subset, 0) + 1
        motif_counts[best.motif] = motif_counts.get(best.motif, 0) + 1

    selected_ids = {item.official_instance_id for item in selected}
    for row in audit:
        row["selected"] = row["official_instance_id"] in selected_ids
    return ExternalTaskSelection(tuple(selected), tuple(audit))


def selection_manifest(selection: ExternalTaskSelection) -> list[dict[str, Any]]:
    return [task.metadata_row() for task in selection.tasks]

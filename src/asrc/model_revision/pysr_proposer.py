from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from asrc.model_revision.proposals import (
    ProposalBatch,
    RepairContract,
    RepairRequest,
    StaticRepairProposer,
)


class PySRRepairUnavailableError(RuntimeError):
    """Raised when the optional PySR proposer cannot be initialized."""


def replay_pysr_hall_of_fame(
    path: str | Path,
    request: RepairRequest,
    contract: RepairContract,
) -> ProposalBatch:
    """Reconstruct the typed PySR frontier without rerunning Julia."""

    try:
        import sympy
    except (ImportError, ModuleNotFoundError) as exc:
        raise PySRRepairUnavailableError("SymPy is required for PySR replay.") from exc
    source_path = Path(path)
    if not source_path.exists():
        raise PySRRepairUnavailableError(
            f"PySR hall-of-fame checkpoint does not exist: {source_path}"
        )
    equations = pd.read_csv(source_path)
    columns = {str(name).lower(): str(name) for name in equations.columns}
    equation_column = columns.get("equation")
    if equation_column is None:
        raise PySRRepairUnavailableError(
            f"PySR hall of fame has no Equation column: {source_path}"
        )
    sort_columns = [
        columns[name]
        for name in ("loss", "complexity")
        if name in columns
    ]
    ordered = equations.sort_values(sort_columns) if sort_columns else equations
    payloads: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(ordered.iterrows(), start=1):
        try:
            expression = sympy.sympify(str(row[equation_column]))
            ast = sympy_expression_to_ast(expression, set(contract.allowed_variables))
        except (TypeError, ValueError, KeyError, sympy.SympifyError):
            continue
        payloads.append(
            {
                "proposal_id": f"pysr_{index:03d}",
                "source": "pysr",
                "edit_type": "add_term",
                "target": sorted(contract.allowed_targets)[0],
                "expression": ast,
                "rationale": "Replayed PySR residual-expression frontier candidate.",
                "expected_signature": "Low residual error under the shared verifier.",
            }
        )
        if len(payloads) >= request.maximum_candidates:
            break
    return StaticRepairProposer("pysr", payloads).propose(request, contract)


def _supported_kwargs(regressor: type[Any], configured: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(regressor.__init__).parameters
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return configured
    return {key: value for key, value in configured.items() if key in parameters}


def sympy_expression_to_ast(expression: Any, allowed_variables: set[str]) -> dict[str, Any]:
    """Convert a trusted PySR/SymPy expression into the public repair AST."""

    try:
        import sympy
    except (ImportError, ModuleNotFoundError) as exc:
        raise PySRRepairUnavailableError("SymPy is required for PySR AST conversion.") from exc

    parameter_index = 0

    def parameter() -> dict[str, Any]:
        nonlocal parameter_index
        parameter_index += 1
        return {"op": "parameter", "name": f"pysr_coefficient_{parameter_index}"}

    def convert(node: Any, *, exponent: bool = False) -> dict[str, Any]:
        if isinstance(node, sympy.Symbol):
            name = str(node)
            if name not in allowed_variables:
                raise ValueError(f"PySR expression used unknown variable {name!r}.")
            return {"op": "variable", "name": name}
        if bool(getattr(node, "is_Number", False)):
            value = float(node)
            if exponent or value in {-1.0, 0.0, 1.0}:
                return {"op": "constant", "value": value}
            return parameter()
        if isinstance(node, sympy.Add):
            return {"op": "add", "arguments": [convert(item) for item in node.args]}
        if isinstance(node, sympy.Mul):
            return {"op": "multiply", "arguments": [convert(item) for item in node.args]}
        if isinstance(node, sympy.Pow):
            return {
                "op": "power",
                "left": convert(node.args[0]),
                "right": convert(node.args[1], exponent=True),
            }
        function_map = {
            sympy.sin: "sin",
            sympy.cos: "cos",
            sympy.exp: "exp",
            sympy.log: "log",
            sympy.tanh: "tanh",
            sympy.Abs: "abs",
        }
        for function, operation in function_map.items():
            if node.func == function:
                return {"op": operation, "argument": convert(node.args[0])}
        raise ValueError(f"Unsupported PySR/SymPy operation {node.func!r}.")

    return convert(expression)


class PySRRepairProposer:
    source = "pysr"

    def __init__(
        self,
        *,
        observed: Any,
        variable_names: tuple[str, ...],
        seed: int,
        output_directory: Path,
        config: Mapping[str, Any],
    ) -> None:
        self.observed = observed
        self.variable_names = variable_names
        self.seed = int(seed)
        self.output_directory = Path(output_directory)
        self.config = dict(config)
        self.wall_time_seconds = 0.0
        self.internal_iterations = int(self.config.get("niterations", 40))

    def propose(
        self,
        request: RepairRequest,
        contract: RepairContract,
    ) -> ProposalBatch:
        try:
            import sympy
            from pysr import PySRRegressor
        except Exception as exc:
            raise PySRRepairUnavailableError(
                "PySR is unavailable. Activate the project venv and verify the Julia/PySR runtime."
            ) from exc

        from time import perf_counter

        fit = self.observed.loc[self.observed["partition"].eq("fit")]
        x_fit = fit.loc[:, list(self.variable_names)]
        residual = fit["target"].to_numpy(float) - fit["baseline"].to_numpy(float)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        configured = {
            "niterations": self.internal_iterations,
            "populations": int(self.config.get("populations", 4)),
            "population_size": int(self.config.get("population_size", 24)),
            "binary_operators": list(self.config.get("binary_operators", ["+", "-", "*", "/"])),
            "unary_operators": list(self.config.get("unary_operators", ["sin", "cos", "tanh", "exp"])),
            "model_selection": "best",
            "maxsize": int(self.config.get("maxsize", 20)),
            "random_state": self.seed,
            "deterministic": True,
            "parallelism": "serial",
            "progress": bool(self.config.get("progress", False)),
            "verbosity": int(self.config.get("verbosity", 0)),
            "warm_start": False,
            "output_directory": str(self.output_directory),
        }
        if self.config.get("constraints"):
            configured["constraints"] = {
                str(operator): tuple(int(value) for value in limits)
                for operator, limits in dict(self.config["constraints"]).items()
            }
        model = PySRRegressor(**_supported_kwargs(PySRRegressor, configured))
        started = perf_counter()
        model.fit(x_fit, residual)
        self.wall_time_seconds = perf_counter() - started
        equations = getattr(model, "equations_", None)
        if equations is None or len(equations) == 0:
            return ProposalBatch((), ())
        sort_columns = [
            column for column in ["loss", "complexity"] if column in equations.columns
        ]
        ordered = equations.sort_values(sort_columns) if sort_columns else equations
        payloads = []
        for index, (row_index, row) in enumerate(ordered.iterrows(), start=1):
            try:
                expression = row.get("sympy_format")
                if expression is None or (
                    isinstance(expression, float) and np.isnan(expression)
                ):
                    try:
                        expression = model.sympy(index=row_index)
                    except TypeError:
                        expression = sympy.sympify(str(row["equation"]))
                ast = sympy_expression_to_ast(expression, set(self.variable_names))
            except (TypeError, ValueError, KeyError):
                continue
            payloads.append(
                {
                    "proposal_id": f"pysr_{index:03d}",
                    "source": "pysr",
                    "edit_type": "add_term",
                    "target": sorted(contract.allowed_targets)[0],
                    "expression": ast,
                    "rationale": "PySR residual-expression frontier candidate.",
                    "expected_signature": "Low residual error under the shared verifier.",
                }
            )
            if len(payloads) >= request.maximum_candidates:
                break
        return StaticRepairProposer("pysr", payloads).propose(request, contract)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from asrc.utils.io import project_root, read_yaml


@dataclass(frozen=True)
class CaseData:
    case_name: str
    data_type: str
    frame: pd.DataFrame
    config: dict[str, Any]
    x_columns: list[str]
    y_true: str
    y_base: str
    residual: str
    group: str | None = None

    def subset(self, **filters: Any) -> "CaseData":
        df = self.frame.copy()
        for column, allowed in filters.items():
            values = allowed if isinstance(allowed, (list, tuple, set)) else [allowed]
            df = df[df[column].isin(values)].copy()
        return CaseData(
            case_name=self.case_name,
            data_type=self.data_type,
            frame=df.reset_index(drop=True),
            config=self.config,
            x_columns=self.x_columns,
            y_true=self.y_true,
            y_base=self.y_base,
            residual=self.residual,
            group=self.group,
        )


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root() / path


def load_case(config_path: str | Path) -> CaseData:
    config = read_yaml(config_path)
    if config.get("case_name") == "ma2018_brazilian_bts":
        return load_ma2018_case(config_path)

    columns = config["columns"]
    input_file = config.get("input_file")
    if not input_file:
        raise ValueError(f"{config_path} does not define input_file; generate synthetic data first.")

    df = pd.read_csv(_resolve_path(input_file))
    for column, allowed in config.get("filters", {}).items():
        df = df[df[column].isin(allowed)].copy()

    y_true = columns["y_true"]
    y_base = columns["y_base"]
    residual = columns.get("residual", "residual")
    if residual not in df.columns:
        df[residual] = df[y_true].astype(float) - df[y_base].astype(float)

    numeric_columns = list(columns["x"]) + [y_true, y_base, residual]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="raise")

    return CaseData(
        case_name=config["case_name"],
        data_type=config["data_type"],
        frame=df.reset_index(drop=True),
        config=config,
        x_columns=list(columns["x"]),
        y_true=y_true,
        y_base=y_base,
        residual=residual,
        group=columns.get("group"),
    )


def load_ma2018_case(config_path: str | Path) -> CaseData:
    config = read_yaml(config_path)
    columns = config["columns"]
    source = config["source_columns"]
    df_long = pd.read_csv(_resolve_path(config["input_file"]))
    for column, allowed in config.get("filters", {}).items():
        df_long = df_long[df_long[column].isin(allowed)].copy()

    wide = (
        df_long.pivot_table(
            index=["dataset", source["angle"]],
            columns=source["series"],
            values=source["value"],
            aggfunc="first",
        )
        .reset_index()
        .rename(
            columns={
                source["angle"]: "beta_deg",
                source["mean_series"]: columns["y_true"],
                source["predicted_series"]: columns["y_base"],
            }
        )
    )
    keep = ["dataset", "beta_deg", columns["y_true"], columns["y_base"]]
    df = wide[keep].dropna(subset=[columns["y_true"], columns["y_base"]]).copy()
    df[columns["residual"]] = df[columns["y_true"]].astype(float) - df[columns["y_base"]].astype(float)
    for column in columns["x"] + [columns["y_true"], columns["y_base"], columns["residual"]]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    return CaseData(
        case_name=config["case_name"],
        data_type=config["data_type"],
        frame=df.reset_index(drop=True),
        config=config,
        x_columns=list(columns["x"]),
        y_true=columns["y_true"],
        y_base=columns["y_base"],
        residual=columns["residual"],
        group=columns.get("group"),
    )

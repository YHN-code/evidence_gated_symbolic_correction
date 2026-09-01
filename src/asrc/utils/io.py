from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _to_markdown(df: pd.DataFrame, index: bool = False) -> str:
    view = df if index else df.reset_index(drop=True)
    headers = ([view.index.name or "index"] if index else []) + [str(col) for col in view.columns]
    rows = []
    for idx, row in view.iterrows():
        values = ([idx] if index else []) + row.tolist()
        rows.append([_format_cell(value) for value in values])
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    header_line = "| " + " | ".join(header.ljust(width) for header, width in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body]) + "\n"


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _latex_escape(value: Any) -> str:
    text = _format_cell(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _to_latex(df: pd.DataFrame, index: bool = False) -> str:
    view = df if index else df.reset_index(drop=True)
    headers = ([view.index.name or "index"] if index else []) + [str(col) for col in view.columns]
    align = "l" * len(headers)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    lines.append(" & ".join(_latex_escape(header) for header in headers) + r" \\")
    lines.append(r"\midrule")
    for idx, row in view.iterrows():
        values = ([idx] if index else []) + row.tolist()
        lines.append(" & ".join(_latex_escape(value) for value in values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def runs_root() -> Path:
    override = os.environ.get("ASRC_RUNS_ROOT", "").strip()
    return Path(override).expanduser().resolve() if override else project_root() / "outputs" / "runs"


def read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = project_root() / path
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_run_dir(run_id: str) -> Path:
    run_dir = runs_root() / run_id
    for name in ["agent_context", "agent_decisions", "agent_memory", "agent_tools", "data", "figures", "formulas", "logs", "metrics", "reports", "tables"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        write_json(
            manifest,
            {
                "run_id": run_id,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "project": "ASRC paper experiments",
            },
        )
    return run_dir


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def write_json_atomic(
    path: str | Path,
    payload: Any,
    *,
    replace_attempts: int = 12,
    initial_wait_seconds: float = 0.05,
) -> Path:
    """Write JSON through a unique temporary file with Windows lock retries."""

    path = Path(path)
    if replace_attempts < 1:
        raise ValueError("replace_attempts must be positive.")
    if initial_wait_seconds < 0.0:
        raise ValueError("initial_wait_seconds must be non-negative.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(replace_attempts):
            try:
                os.replace(temporary_path, path)
                return path
            except PermissionError:
                if attempt + 1 >= replace_attempts:
                    raise
                time.sleep(
                    min(
                        float(initial_wait_seconds) * (2.0**attempt),
                        1.0,
                    )
                )
    finally:
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except PermissionError:
                pass
    return path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_table_bundle(df: pd.DataFrame, output_base: Path, index: bool = False) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    csv_path = output_base.with_suffix(".csv")
    md_path = output_base.with_suffix(".md")
    tex_path = output_base.with_suffix(".tex")
    df.to_csv(csv_path, index=index)
    md_path.write_text(_to_markdown(df, index=index), encoding="utf-8")
    tex_path.write_text(_to_latex(df, index=index), encoding="utf-8")
    paths.extend([csv_path, md_path, tex_path])
    return paths

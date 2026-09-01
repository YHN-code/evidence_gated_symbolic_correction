from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from asrc.utils.io import read_json, write_json


def should_skip(run_dir: Path, stage: str, outputs: list[Path], resume: bool, force: bool) -> bool:
    if force or not resume:
        return False
    progress_path = run_dir / "progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {}
    stage_done = progress.get(stage, {}).get("status") == "done"
    return stage_done and all(path.exists() for path in outputs)


def mark_done(run_dir: Path, stage: str, extra: dict | None = None) -> None:
    progress_path = run_dir / "progress.json"
    progress = read_json(progress_path) if progress_path.exists() else {}
    progress[stage] = {
        "status": "done",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    write_json(progress_path, progress)


def progress_message(run_dir: Path, message: str, verbose: bool = True, **fields: Any) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    suffix = ""
    if fields:
        suffix = " | " + " ".join(f"{key}={value}" for key, value in fields.items())
    line = f"[{timestamp}] {message}{suffix}"
    log_path = run_dir / "logs" / "progress.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    if verbose:
        print(line, flush=True)

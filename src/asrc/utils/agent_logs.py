from __future__ import annotations

from pathlib import Path
from typing import Any

from asrc.utils.io import write_json


def save_agent_decision(run_dir: Path, decision: dict[str, Any], name: str) -> Path:
    return write_json(run_dir / "agent_decisions" / f"{name}.json", decision)

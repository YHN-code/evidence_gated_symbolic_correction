from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolate_generated_runs():
    previous = os.environ.get("ASRC_RUNS_ROOT")
    run_root = Path(__file__).resolve().parents[1] / ".pytest_cache" / f"asrc_runs_{os.getpid()}"
    shutil.rmtree(run_root, ignore_errors=True)
    run_root.mkdir(parents=True, exist_ok=True)
    os.environ["ASRC_RUNS_ROOT"] = str(run_root)
    yield
    shutil.rmtree(run_root, ignore_errors=True)
    if previous is None:
        os.environ.pop("ASRC_RUNS_ROOT", None)
    else:
        os.environ["ASRC_RUNS_ROOT"] = previous


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="run slow full-pipeline integration tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test skipped by default; use --runslow to run it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

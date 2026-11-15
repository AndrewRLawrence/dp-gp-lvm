"""Tests of the CI helper scripts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "ci" / "junit_badge.py"


@pytest.fixture(scope="module")
def junit_badge() -> ModuleType:
    spec = importlib.util.spec_from_file_location("junit_badge", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, tests: int, failures: int = 0, errors: int = 0, skipped: int = 0) -> Path:
    path.write_text(
        f'<testsuites><testsuite name="pytest" tests="{tests}" failures="{failures}" '
        f'errors="{errors}" skipped="{skipped}"/></testsuites>'
    )
    return path


def test_badge_for_passing_reports(junit_badge: ModuleType, tmp_path: Path) -> None:
    reports = [write(tmp_path / "a.xml", 10, skipped=1), write(tmp_path / "b.xml", 5)]
    output = tmp_path / "tests.json"
    assert junit_badge.main(["--output", str(output), *map(str, reports)]) == 0
    assert json.loads(output.read_text()) == {
        "schemaVersion": 1,
        "label": "tests",
        "message": "14 passed",
        "color": "brightgreen",
    }


def test_badge_for_failing_reports(junit_badge: ModuleType, tmp_path: Path) -> None:
    root_suite = tmp_path / "suite.xml"
    root_suite.write_text('<testsuite tests="4" failures="1" errors="1" skipped="0"/>')
    totals = junit_badge.count([root_suite])
    assert junit_badge.badge(totals, reports=1)["message"] == "2 failed, 2 passed"
    assert junit_badge.badge(totals, reports=1)["color"] == "red"


def test_badge_without_reports(junit_badge: ModuleType, tmp_path: Path) -> None:
    output = tmp_path / "tests.json"
    junit_badge.main(["--output", str(output), str(tmp_path / "missing.xml")])
    assert json.loads(output.read_text())["message"] == "unknown"

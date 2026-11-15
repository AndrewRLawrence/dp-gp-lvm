"""Write a shields.io endpoint file with the test count from JUnit XML reports.

Usage: ``python3 scripts/ci/junit_badge.py --output tests.json reports/*/junit.xml``

The README badge reads the file from the ``badges`` branch. See https://shields.io/badges/endpoint-badge.
This script needs only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree


def count(paths: list[Path]) -> dict[str, int]:
    """Return the number of tests, failures, errors, and skipped tests in the reports."""
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for path in paths:
        # The reports come from this CI run, so they are trusted input.
        root = ElementTree.parse(path).getroot()  # noqa: S314
        suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
        for suite in suites:
            for key in totals:
                totals[key] += int(suite.get(key, "0"))
    return totals


def badge(totals: dict[str, int], *, reports: int) -> dict[str, object]:
    """Return the shields.io endpoint data for the totals."""
    if reports == 0:
        return {"schemaVersion": 1, "label": "tests", "message": "unknown", "color": "lightgrey"}
    failed = totals["failures"] + totals["errors"]
    passed = totals["tests"] - failed - totals["skipped"]
    if failed:
        return {"schemaVersion": 1, "label": "tests", "message": f"{failed} failed, {passed} passed", "color": "red"}
    return {"schemaVersion": 1, "label": "tests", "message": f"{passed} passed", "color": "brightgreen"}


def main(argv: list[str] | None = None) -> int:
    """Read the reports and write the endpoint file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="*", type=Path, help="JUnit XML files.")
    parser.add_argument("--output", type=Path, required=True, help="The endpoint JSON file to write.")
    args = parser.parse_args(argv)
    reports = [p for p in args.reports if p.exists()]
    data = badge(count(reports), reports=len(reports))
    args.output.write_text(json.dumps(data) + "\n")
    print(json.dumps(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

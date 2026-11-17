"""Tests of the CI helper scripts."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from packaging.version import Version

SCRIPTS = Path(__file__).parents[1] / "scripts" / "ci"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"ci_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses look up the module of a class in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def junit_badge() -> ModuleType:
    return load("junit_badge")


@pytest.fixture(scope="module")
def release() -> ModuleType:
    return load("release")


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


# The tests of scripts/ci/release.py.

REPOSITORY = "https://github.com/octo/example"


def write_repo(
    root: Path,
    *,
    version: str = "1.2.0",
    heading: str = "## [1.2.0] - 2026-11-17",
    unreleased: str = "## [Unreleased]",
    citation_version: str | None = "1.2.0",
    citation_date: str | None = "'2026-11-17'",
) -> Path:
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "dp-gp-lvm"\nversion = "{version}"\n\n[project.urls]\nRepository = "{REPOSITORY}"\n'
    )
    (root / "CHANGELOG.md").write_text(
        f"# Changelog\n\n{unreleased}\n\n### Added\n\n"
        "- A [file](docs/a.md), an [anchor](#added), and a [site](https://example.org).\n\n"
        f"{heading}\n\n### Fixed\n\n- A fix.\n\n"
        f"[Unreleased]: {REPOSITORY}/compare/v1.2.0...HEAD\n[1.2.0]: {REPOSITORY}/releases/tag/v1.2.0\n"
    )
    citation = ["cff-version: 1.2.0", 'title: "Example"']
    if citation_version is not None:
        citation.append(f"version: {citation_version}")
    if citation_date is not None:
        citation.append(f"date-released: {citation_date}")
    citation += ["preferred-citation:", "  version: 9.9.9", "  date-released: 2019-06-09"]
    (root / "CITATION.cff").write_text("\n".join(citation) + "\n")
    return root


def test_metadata_of_this_repository_agrees(release: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    assert release.main(["check-metadata"]) == 0
    assert "agrees" in capsys.readouterr().out


def test_read_metadata(release: ModuleType, tmp_path: Path) -> None:
    metadata = release.read_metadata(write_repo(tmp_path))
    assert metadata.version == "1.2.0"
    assert metadata.repository == REPOSITORY
    assert metadata.citation_version == "1.2.0"
    assert metadata.citation_date == "2026-11-17"


@pytest.mark.parametrize(
    ("base", "tags", "commits", "expected"),
    [
        ("1.0.0", [], 13, "1.0.0.dev13"),
        ("1.0.0", ["v1.0.0"], 15, "1.0.1.dev15"),
        ("1.1.0", ["v1.0.0"], 20, "1.1.0.dev20"),
        ("1.1.0rc1", [], 30, "1.1.0rc1.dev30"),
        ("1.1.0rc1", ["v1.1.0rc0"], 30, "1.1.0rc1.dev30"),
        ("1.1.0rc1", ["v1.1.0rc1"], 31, "1.1.0rc2.dev31"),
        ("1.1.0b0", ["v1.1.0a2", "v1.1.0b0"], 31, "1.1.0b1.dev31"),
        ("1.1.0", ["v1.1.0rc1"], 32, "1.1.0rc2.dev32"),
        (
            "1.1.0",
            ["v1.1.0rc0", "v1.1.0b3", "v1.1.0rc1", "v1.0.0rc9", "v1.1.0rc1x", "legacy-tf1"],
            32,
            "1.1.0rc2.dev32",
        ),
        ("1.1.0", ["v1.1.0rc1", "v1.1.0"], 33, "1.1.1.dev33"),
    ],
)
def test_dev_version(release: ModuleType, base: str, tags: list[str], commits: int, expected: str) -> None:
    assert release.dev_version(base, tags, commits) == expected


def test_dev_versions_sort_between_the_releases(release: ModuleType) -> None:
    history = [
        release.dev_version("1.0.0", [], 13),
        "1.0.0",
        release.dev_version("1.0.0", ["v1.0.0"], 15),
        release.dev_version("1.1.0", ["v1.0.0"], 20),
        "1.1.0",
        release.dev_version("1.1.0", ["v1.0.0", "v1.1.0"], 21),
    ]
    versions = [Version(item) for item in history]
    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions)


def test_dev_versions_sort_between_the_prereleases(release: ModuleType) -> None:
    history = [
        release.dev_version("1.0.0", [], 13),
        release.dev_version("1.0.0rc0", [], 14),
        "1.0.0rc0",
        release.dev_version("1.0.0rc0", ["v1.0.0rc0"], 15),
        release.dev_version("1.0.0", ["v1.0.0rc0"], 16),
        "1.0.0",
        release.dev_version("1.0.0", ["v1.0.0rc0", "v1.0.0"], 17),
    ]
    assert history == [
        "1.0.0.dev13",
        "1.0.0rc0.dev14",
        "1.0.0rc0",
        "1.0.0rc1.dev15",
        "1.0.0rc1.dev16",
        "1.0.0",
        "1.0.1.dev17",
    ]
    versions = [Version(item) for item in history]
    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions)


@pytest.mark.parametrize("version", ["1.0", "v1.0.0", "1.0.0.dev1", "1.0.0.post1", "01.0.0a"])
def test_parse_version_rejects_other_forms(release: ModuleType, version: str) -> None:
    with pytest.raises(release.ReleaseError, match="is not a version"):
        release.parse_version(version)


@pytest.mark.parametrize(
    ("version", "expected"),
    [("1.0.0", False), ("1.0.0a1", True), ("1.0.0b2", True), ("1.0.0rc1", True), ("1.0.1.dev3", True)],
)
def test_is_prerelease(release: ModuleType, version: str, expected: bool) -> None:
    assert release.is_prerelease(version) is expected


@pytest.mark.parametrize(
    ("ref", "tags", "expected"),
    [
        ("refs/tags/v1.2.0", [], ["version=1.2.0", "prerelease=false"]),
        ("refs/heads/main", ["v1.2.0"], ["version=1.2.1.dev7", "prerelease=true"]),
        ("refs/heads/main", ["v1.2.0rc1"], ["version=1.2.0rc2.dev7", "prerelease=true"]),
        ("refs/pull/9/merge", [], ["version=1.2.0.dev7", "prerelease=true"]),
    ],
)
def test_version_command(
    release: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], ref: str, tags: list[str], expected: str
) -> None:
    root = write_repo(tmp_path)
    assert release.main(["--root", str(root), "version", "--ref", ref, "--commits", "7", "--tags", *tags]) == 0
    assert capsys.readouterr().out.splitlines() == expected


def test_version_of_a_prerelease_tag(release: ModuleType, tmp_path: Path) -> None:
    root = write_repo(tmp_path, version="1.2.0rc1", heading="## [1.2.0rc1] - 2026-11-17", citation_version="1.2.0rc1")
    metadata = release.read_metadata(root)
    assert release.release_version(metadata, ref="refs/tags/v1.2.0rc1", commits=1, tags=[]) == "1.2.0rc1"


@pytest.mark.parametrize(
    ("ref", "repo", "message"),
    [
        ("refs/tags/v1.3.0", {}, "The tag is v1.3.0, but pyproject.toml has the version 1.2.0"),
        ("refs/tags/1.2.0", {}, "The tag is 1.2.0"),
        ("refs/tags/legacy-tf1", {}, "The tag is legacy-tf1"),
        (
            "refs/tags/v1.2.0",
            {"heading": "## [1.2.0] - Unreleased", "citation_date": None},
            "no release date for 1.2.0",
        ),
    ],
)
def test_version_command_rejects_a_bad_tag(
    release: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    ref: str,
    repo: dict[str, Any],
    message: str,
) -> None:
    root = write_repo(tmp_path, **repo)
    assert release.main(["--root", str(root), "version", "--ref", ref, "--commits", "7"]) == 1
    assert message in capsys.readouterr().err


def test_unreleased_version_agrees(release: ModuleType, tmp_path: Path) -> None:
    root = write_repo(tmp_path, heading="## [1.2.0] - Unreleased", citation_date=None)
    release.check_metadata(release.read_metadata(root))


@pytest.mark.parametrize(
    ("repo", "message"),
    [
        ({"version": "1.2"}, "'1.2' is not a version"),
        ({"citation_version": "1.1.0"}, "CITATION.cff has the version 1.1.0, but pyproject.toml has 1.2.0"),
        ({"citation_version": None}, "CITATION.cff has the version None"),
        ({"unreleased": "## [Unreleased] - 2026-11-17"}, "must have the heading '## [Unreleased]'"),
        ({"unreleased": "## Unreleased"}, "must have the heading '## [Unreleased]'"),
        ({"heading": "## [1.1.0] - 2026-11-17"}, "no heading for the version 1.2.0"),
        ({"heading": "## [1.2.0] - Unreleased"}, "has a date-released, but CHANGELOG.md says that 1.2.0 is unreleased"),
        ({"heading": "## [1.2.0]"}, "must end in ' - YYYY-MM-DD' or ' - Unreleased'"),
        ({"heading": "## [1.2.0] - 17 November 2026"}, "must end in ' - YYYY-MM-DD' or ' - Unreleased'"),
        ({"citation_date": "2026-11-16"}, "date-released 2026-11-16, but CHANGELOG.md has 2026-11-17"),
        ({"citation_date": None}, "date-released None, but CHANGELOG.md has 2026-11-17"),
    ],
)
def test_check_metadata_rejects_disagreement(
    release: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    repo: dict[str, Any],
    message: str,
) -> None:
    root = write_repo(tmp_path, **repo)
    assert release.main(["--root", str(root), "check-metadata"]) == 1
    assert message in capsys.readouterr().err


def test_release_notes(release: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = write_repo(tmp_path)
    metadata = release.read_metadata(root)
    assert release.release_notes(metadata, "1.2.0") == "### Fixed\n\n- A fix.\n"
    assert release.release_notes(metadata, "Unreleased") == (
        f"### Added\n\n- A [file]({REPOSITORY}/blob/vUnreleased/docs/a.md), an [anchor](#added), "
        "and a [site](https://example.org).\n"
    )
    assert release.main(["--root", str(root), "notes", "1.2.0"]) == 0
    assert capsys.readouterr().out == "### Fixed\n\n- A fix.\n"


def test_release_notes_of_the_last_section(release: ModuleType, tmp_path: Path) -> None:
    root = write_repo(tmp_path)
    (root / "CHANGELOG.md").write_text("## [Unreleased]\n\n## [1.2.0] - 2026-11-17\n\n- The first release.\n")
    assert release.release_notes(release.read_metadata(root), "1.2.0") == "- The first release.\n"


def test_release_notes_of_a_missing_version(
    release: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = write_repo(tmp_path)
    assert release.main(["--root", str(root), "notes", "0.9.0"]) == 1
    assert "no section for 0.9.0" in capsys.readouterr().err


PACKAGE_FILES = ["__init__.py", "kernels.py", "py.typed"]


def make_dist(
    directory: Path,
    version: str = "1.2.0",
    *,
    wheel_extra: tuple[str, ...] = (),
    sdist_extra: tuple[str, ...] = (),
    drop: tuple[str, ...] = (),
    metadata_version: str | None = None,
    stated_version: bool = True,
    link: bool = False,
) -> Path:
    directory.mkdir(exist_ok=True)
    stated = f"Version: {metadata_version or version}\n" if stated_version else ""
    info = f"dp_gp_lvm-{version}.dist-info"
    wheel_files = {f"dp_gp_lvm/{name}": "" for name in PACKAGE_FILES}
    wheel_files |= {f"{info}/{name}": "" for name in ("WHEEL", "RECORD", "licenses/LICENSE")}
    wheel_files[f"{info}/METADATA"] = f"Metadata-Version: 2.4\nName: dp-gp-lvm\n{stated}"
    wheel_files |= dict.fromkeys(wheel_extra, "")
    with zipfile.ZipFile(directory / f"dp_gp_lvm-{version}-py3-none-any.whl", "w") as wheel:
        wheel.writestr("dp_gp_lvm/", "")
        for name, text in wheel_files.items():
            if name not in drop:
                wheel.writestr(name, text)

    prefix = f"dp_gp_lvm-{version}/"
    sdist_files = {prefix + name: "" for name in ("pyproject.toml", "pyproject.toml.orig", "README.md", "LICENSE")}
    sdist_files[prefix + "PKG-INFO"] = f"Metadata-Version: 2.4\nName: dp-gp-lvm\n{stated}"
    sdist_files |= {f"{prefix}src/dp_gp_lvm/{name}": "" for name in PACKAGE_FILES}
    sdist_files |= {prefix + name: "" for name in sdist_extra}
    with tarfile.open(directory / f"dp_gp_lvm-{version}.tar.gz", "w:gz") as sdist:
        for name in (prefix, f"{prefix}src", f"{prefix}src/dp_gp_lvm"):
            entry = tarfile.TarInfo(name.rstrip("/"))
            entry.type = tarfile.DIRTYPE
            sdist.addfile(entry)
        for name, text in sdist_files.items():
            if name not in drop:
                data = text.encode()
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                sdist.addfile(entry, io.BytesIO(data))
        if link:
            entry = tarfile.TarInfo(f"{prefix}data")
            entry.type = tarfile.SYMTYPE
            entry.linkname = "../../data"
            sdist.addfile(entry)
    (directory / ".gitignore").write_text("*\n")
    return directory


def test_check_dist_accepts_the_package(
    release: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dist = make_dist(tmp_path / "dist", "1.2.1.dev7")
    assert release.main(["check-dist", str(dist), "--version", "1.2.1.dev7"]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert "dp_gp_lvm/__init__.py" in printed
    assert "dp_gp_lvm-1.2.1.dev7/src/dp_gp_lvm/py.typed" in printed


def test_check_dist_of_the_real_build(release: ModuleType, tmp_path: Path) -> None:
    sdist = make_dist(tmp_path / "dist") / "dp_gp_lvm-1.2.0.tar.gz"
    assert "dp_gp_lvm-1.2.0/pyproject.toml.orig" in release.check_sdist(sdist, "1.2.0")


@pytest.mark.parametrize(
    ("dist", "message"),
    [
        ({"wheel_extra": ("data/frey_faces/frey.npy",)}, "The wheel contains files that must not be in it: data/"),
        ({"wheel_extra": ("dp_gp_lvm/notes.txt",)}, "The wheel contains files that must not be in it"),
        ({"sdist_extra": ("tests/test_models.py",)}, "The sdist contains files that must not be in it"),
        ({"sdist_extra": ("data/README.md",)}, "The sdist contains files that must not be in it"),
        ({"drop": ("dp_gp_lvm/py.typed",)}, "The wheel does not contain these files: dp_gp_lvm/py.typed"),
        ({"drop": ("dp_gp_lvm-1.2.0/LICENSE",)}, "The sdist does not contain these files: dp_gp_lvm-1.2.0/LICENSE"),
        ({"metadata_version": "1.0.0"}, "The wheel METADATA has the version 1.0.0, not 1.2.0"),
        ({"link": True}, "The sdist contains links or special files: dp_gp_lvm-1.2.0/data"),
    ],
)
def test_check_dist_rejects_bad_contents(
    release: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    dist: dict[str, Any],
    message: str,
) -> None:
    directory = make_dist(tmp_path / "dist", **dist)
    assert release.main(["check-dist", str(directory), "--version", "1.2.0"]) == 1
    assert message in capsys.readouterr().err


def test_check_sdist_rejects_a_wrong_version(release: ModuleType, tmp_path: Path) -> None:
    sdist = make_dist(tmp_path / "dist", metadata_version="1.0.0") / "dp_gp_lvm-1.2.0.tar.gz"
    with pytest.raises(release.ReleaseError, match=r"PKG-INFO has the version 1\.0\.0, not 1\.2\.0"):
        release.check_sdist(sdist, "1.2.0")


def test_check_dist_needs_a_version_field(release: ModuleType, tmp_path: Path) -> None:
    directory = make_dist(tmp_path / "dist", stated_version=False)
    with pytest.raises(release.ReleaseError, match="METADATA has the version None"):
        release.check_wheel(directory / "dp_gp_lvm-1.2.0-py3-none-any.whl", "1.2.0")
    with pytest.raises(release.ReleaseError, match="PKG-INFO has the version None"):
        release.check_sdist(directory / "dp_gp_lvm-1.2.0.tar.gz", "1.2.0")


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("dp_gp_lvm-1.0.0.tar.gz", "must contain only dp_gp_lvm-1.2.0-py3-none-any.whl and dp_gp_lvm-1.2.0.tar.gz"),
        (None, "not: nothing"),
    ],
)
def test_check_dist_needs_exactly_the_two_files(
    release: ModuleType, tmp_path: Path, extra: str | None, message: str
) -> None:
    directory = tmp_path / "dist"
    if extra is None:
        directory.mkdir()
    else:
        make_dist(directory)
        (directory / extra).write_bytes(b"")
    with pytest.raises(release.ReleaseError, match=message.replace(".", r"\.")):
        release.check_dist(directory, "1.2.0")

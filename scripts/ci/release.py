"""Get the release version, check the release metadata and the distributions, and print the release notes.

The release workflow ``.github/workflows/release.yml`` runs these commands:

- ``python3 scripts/ci/release.py version --ref "$GITHUB_REF" --commits N --tags v1.0.0 ...``
  prints the version to build. A tag ``vX.Y.Z`` gives ``X.Y.Z``. Other refs give a dev version, for example
  ``X.Y.Z.devN``.
- ``python3 scripts/ci/release.py check-metadata`` checks that ``pyproject.toml``, ``CHANGELOG.md``, and
  ``CITATION.cff`` agree.
- ``python3 scripts/ci/release.py check-dist dist --version X.Y.Z`` checks the files in the wheel and the sdist.
- ``python3 scripts/ci/release.py notes X.Y.Z`` prints the CHANGELOG section of the version.

CONTRIBUTING.md gives the release steps. This script needs only the Python standard library.
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "dp_gp_lvm"

# The release versions that this project uses: X.Y.Z, or a pre-release X.Y.ZaN, X.Y.ZbN, or X.Y.ZrcN (PEP 440).
VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?$")
PHASES = ("a", "b", "rc")  # The order of the pre-release phases.
HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\](?: - (?P<date>.+?))?\s*$", re.MULTILINE)
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LINK_REFERENCE = re.compile(r"^\[[^\]]+\]: ", re.MULTILINE)
RELATIVE_LINK = re.compile(r"\]\((?!https?://|#|mailto:)([^)\s]+)\)")
UNRELEASED = "Unreleased"


class ReleaseError(Exception):
    """A release check failed."""


@dataclass(frozen=True)
class Metadata:
    """The release metadata of the repository."""

    version: str
    repository: str
    changelog: str
    citation_version: str | None
    citation_date: str | None


def citation_field(text: str, key: str) -> str | None:
    """Return the value of a top-level key of ``CITATION.cff``, without quotes."""
    match = re.search(rf"^{re.escape(key)}:[ \t]*(.+?)[ \t]*$", text, re.MULTILINE)
    return match.group(1).strip("\"'") if match else None


def read_metadata(root: Path) -> Metadata:
    """Read the version, the CHANGELOG, and the CITATION fields from the repository."""
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    citation = (root / "CITATION.cff").read_text()
    return Metadata(
        version=project["version"],
        repository=project["urls"]["Repository"],
        changelog=(root / "CHANGELOG.md").read_text(),
        citation_version=citation_field(citation, "version"),
        citation_date=citation_field(citation, "date-released"),
    )


def parse_version(version: str) -> tuple[int, int, int]:
    """Return the release numbers ``(X, Y, Z)`` of a version, or raise ``ReleaseError``."""
    match = VERSION.match(version)
    if match is None:
        raise ReleaseError(f"{version!r} is not a version of the form X.Y.Z, X.Y.ZaN, X.Y.ZbN, or X.Y.ZrcN.")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def is_prerelease(version: str) -> bool:
    """Return True for an alpha, beta, release candidate, or dev version."""
    match = VERSION.match(version)
    return match is None or match.group(4) is not None


def changelog_dates(changelog: str) -> dict[str, str | None]:
    """Return the text after each ``## [version] - `` heading of the CHANGELOG, by version."""
    return {match.group("version"): match.group("date") for match in HEADING.finditer(changelog)}


def last_prerelease(version: str, tags: list[str]) -> tuple[str, int] | None:
    """Return the phase and the number of the last pre-release tag of ``version``, or None.

    Only the release numbers ``X.Y.Z`` of ``version`` are used. For example, the tags ``v1.0.0b1`` and ``v1.0.0rc0``
    give ``("rc", 0)`` for ``1.0.0`` and for ``1.0.0rc0``.
    """
    major, minor, patch = parse_version(version)
    tagged = re.compile(rf"^v{major}\.{minor}\.{patch}(a|b|rc)(\d+)$")
    parts = [(match.group(1), int(match.group(2))) for tag in tags if (match := tagged.match(tag))]
    return max(parts, key=lambda part: (PHASES.index(part[0]), part[1]), default=None)


def dev_version(base: str, tags: list[str], commits: int) -> str:
    """Return the dev version for a commit that is not a release.

    The dev version sorts after the last release tag of ``base`` and before the next release (PEP 440):

    - ``base`` has no tag, and no pre-release of it has a tag: ``<base>.devN``.
    - ``base`` is a release ``X.Y.Z`` with a tag: ``X.Y.(Z+1).devN``.
    - In the other cases, the last pre-release tag of ``X.Y.Z`` is ``X.Y.ZrcK``: ``X.Y.Zrc(K+1).devN``. Alpha and
      beta versions are the same. For example, after the tag ``v1.0.0rc0``, the dev versions are ``1.0.0rc1.devN``
      for the base ``1.0.0rc0`` and also for the base ``1.0.0``.

    ``commits`` is the number of commits in the history. On the linear ``main`` branch, each commit has a different
    number.
    """
    major, minor, patch = parse_version(base)
    tagged = f"v{base}" in tags
    if is_prerelease(base) and not tagged:
        return f"{base}.dev{commits}"
    if not is_prerelease(base) and tagged:
        return f"{major}.{minor}.{patch + 1}.dev{commits}"
    last = last_prerelease(base, tags)
    if last is None:
        return f"{base}.dev{commits}"
    phase, number = last
    return f"{major}.{minor}.{patch}{phase}{number + 1}.dev{commits}"


def check_metadata(metadata: Metadata) -> None:
    """Check that ``pyproject.toml``, ``CHANGELOG.md``, and ``CITATION.cff`` agree."""
    version = metadata.version
    parse_version(version)
    if metadata.citation_version != version:
        raise ReleaseError(
            f"CITATION.cff has the version {metadata.citation_version}, but pyproject.toml has {version}."
        )
    dates = changelog_dates(metadata.changelog)
    if UNRELEASED not in dates or dates[UNRELEASED] is not None:
        raise ReleaseError("CHANGELOG.md must have the heading '## [Unreleased]'.")
    if version not in dates:
        raise ReleaseError(f"CHANGELOG.md has no heading for the version {version}.")
    date = dates[version]
    if date == UNRELEASED:
        if metadata.citation_date is not None:
            raise ReleaseError(f"CITATION.cff has a date-released, but CHANGELOG.md says that {version} is unreleased.")
    elif date is None or DATE.match(date) is None:
        raise ReleaseError(f"The CHANGELOG.md heading of {version} must end in ' - YYYY-MM-DD' or ' - Unreleased'.")
    elif metadata.citation_date != date:
        raise ReleaseError(f"CITATION.cff has the date-released {metadata.citation_date}, but CHANGELOG.md has {date}.")


def release_version(metadata: Metadata, *, ref: str, commits: int, tags: list[str]) -> str:
    """Return the version to build for a git ref.

    A tag ``vX.Y.Z`` gives ``X.Y.Z``. The tag must agree with ``pyproject.toml``, and the CHANGELOG and
    ``CITATION.cff`` must give the release date. Other refs give a dev version.
    """
    check_metadata(metadata)
    if not ref.startswith("refs/tags/"):
        return dev_version(metadata.version, tags, commits)
    tag = ref.removeprefix("refs/tags/")
    if tag != f"v{metadata.version}":
        raise ReleaseError(f"The tag is {tag}, but pyproject.toml has the version {metadata.version}.")
    if changelog_dates(metadata.changelog)[metadata.version] == UNRELEASED:
        raise ReleaseError(f"CHANGELOG.md has no release date for {metadata.version}.")
    return metadata.version


def metadata_version(text: str) -> str | None:
    """Return the ``Version`` field of a core metadata file (``METADATA`` or ``PKG-INFO``)."""
    match = re.search(r"^Version: (.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def check_members(kind: str, members: list[str], allowed: re.Pattern[str], required: list[str]) -> None:
    """Check that each member is allowed and that each required member is present."""
    unexpected = [name for name in members if not allowed.fullmatch(name)]
    if unexpected:
        raise ReleaseError(f"The {kind} contains files that must not be in it: {', '.join(unexpected)}")
    missing = [name for name in required if name not in members]
    if missing:
        raise ReleaseError(f"The {kind} does not contain these files: {', '.join(missing)}")


def check_wheel(path: Path, version: str) -> list[str]:
    """Check the files and the version of the wheel, and return its file names."""
    info = f"{PACKAGE}-{version}.dist-info"
    allowed = re.compile(
        rf"{PACKAGE}/\w+\.py|{PACKAGE}/py\.typed|{re.escape(info)}/(METADATA|WHEEL|RECORD|licenses/LICENSE)"
    )
    with zipfile.ZipFile(path) as wheel:
        members = [name for name in wheel.namelist() if not name.endswith("/")]
        required = [f"{PACKAGE}/__init__.py", f"{PACKAGE}/py.typed", f"{info}/METADATA", f"{info}/licenses/LICENSE"]
        check_members("wheel", members, allowed, required)
        found = metadata_version(wheel.read(f"{info}/METADATA").decode())
    if found != version:
        raise ReleaseError(f"The wheel METADATA has the version {found}, not {version}.")
    return members


def check_sdist(path: Path, version: str) -> list[str]:
    """Check the files and the version of the sdist, and return its file names."""
    prefix = f"{PACKAGE}-{version}/"
    allowed = re.compile(
        rf"{re.escape(prefix)}(PKG-INFO|pyproject\.toml|pyproject\.toml\.orig|README\.md|LICENSE"
        rf"|src/{PACKAGE}/\w+\.py|src/{PACKAGE}/py\.typed)"
    )
    with tarfile.open(path, "r:gz") as sdist:
        entries = sdist.getmembers()
        links = [entry.name for entry in entries if not (entry.isfile() or entry.isdir())]
        if links:
            raise ReleaseError(f"The sdist contains links or special files: {', '.join(links)}")
        members = [entry.name for entry in entries if entry.isfile()]
        required = [prefix + name for name in ("PKG-INFO", "pyproject.toml", "README.md", "LICENSE")]
        required += [f"{prefix}src/{PACKAGE}/__init__.py", f"{prefix}src/{PACKAGE}/py.typed"]
        check_members("sdist", members, allowed, required)
        pkg_info = sdist.extractfile(f"{prefix}PKG-INFO")
        found = metadata_version(pkg_info.read().decode()) if pkg_info else None
    if found != version:
        raise ReleaseError(f"The sdist PKG-INFO has the version {found}, not {version}.")
    return members


def check_dist(directory: Path, version: str) -> list[str]:
    """Check that the directory contains only the wheel and the sdist of the version, and check both."""
    wheel = f"{PACKAGE}-{version}-py3-none-any.whl"
    sdist = f"{PACKAGE}-{version}.tar.gz"
    # `uv build` also writes a .gitignore file. The artifact upload skips hidden files.
    found = sorted(path.name for path in directory.iterdir() if not path.name.startswith("."))
    if found != sorted([wheel, sdist]):
        raise ReleaseError(f"{directory} must contain only {wheel} and {sdist}, not: {', '.join(found) or 'nothing'}")
    return check_wheel(directory / wheel, version) + check_sdist(directory / sdist, version)


def release_notes(metadata: Metadata, version: str) -> str:
    """Return the CHANGELOG section of the version, with absolute links to the files at the release tag."""
    headings = list(HEADING.finditer(metadata.changelog))
    starts = [heading.start() for heading in headings[1:]] + [len(metadata.changelog)]
    sections = {
        heading.group("version"): metadata.changelog[heading.end() : end]
        for heading, end in zip(headings, starts, strict=True)
    }
    if version not in sections:
        raise ReleaseError(f"CHANGELOG.md has no section for {version}.")
    section = sections[version]
    reference = LINK_REFERENCE.search(section)
    section = section[: reference.start()] if reference else section
    base = f"{metadata.repository.rstrip('/')}/blob/v{version}/"
    return RELATIVE_LINK.sub(lambda link: f"]({base}{link.group(1)})", section).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run one command. Return 0 if it succeeds and 1 if a check fails."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=ROOT, help="The repository root.")
    commands = parser.add_subparsers(dest="command", required=True)
    version = commands.add_parser("version", help="Print the version to build, as GitHub Actions outputs.")
    version.add_argument("--ref", required=True, help="The git ref, for example refs/heads/main or refs/tags/v1.0.0.")
    version.add_argument("--commits", type=int, required=True, help="The number of commits in the history.")
    version.add_argument("--tags", nargs="*", default=[], help="The release tags that exist.")
    commands.add_parser("check-metadata", help="Check that pyproject.toml, CHANGELOG.md, and CITATION.cff agree.")
    dist = commands.add_parser("check-dist", help="Check the files in the wheel and the sdist.")
    dist.add_argument("directory", type=Path, help="The directory that contains the distributions.")
    dist.add_argument("--version", required=True, help="The version of the distributions.")
    notes = commands.add_parser("notes", help="Print the CHANGELOG section of a version.")
    notes.add_argument("version", help="The version.")
    args = parser.parse_args(argv)
    try:
        if args.command == "check-dist":
            for name in check_dist(args.directory, args.version):
                print(name)
            return 0
        metadata = read_metadata(args.root)
        if args.command == "version":
            built = release_version(metadata, ref=args.ref, commits=args.commits, tags=args.tags)
            print(f"version={built}")
            print(f"prerelease={str(is_prerelease(built)).lower()}")
        elif args.command == "check-metadata":
            check_metadata(metadata)
            print(f"The release metadata of {metadata.version} agrees.")
        else:
            print(release_notes(metadata, args.version), end="")
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

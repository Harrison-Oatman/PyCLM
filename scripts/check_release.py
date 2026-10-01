"""
Check that a release tag, the package version and the changelog agree.

    python scripts/check_release.py v1.1.0

Passes when the tag is ``v`` + ``[project] version`` of ``pyproject.toml``,
``CHANGELOG.md`` has a ``## <version>`` heading that is no longer marked
unreleased, and ``CITATION.cff`` (when present) gives the same version. Run
by the publish workflow before anything is uploaded, and worth running by
hand before tagging. Standard library only.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def problems(tag: str, root: Path = ROOT) -> list[str]:
    with open(root / "pyproject.toml", "rb") as f:
        version = tomllib.load(f)["project"]["version"]
    out = []
    if tag.removeprefix("v") != version:
        out.append(
            f"the tag {tag!r} does not match pyproject.toml's version {version!r} "
            f"(tag v{version}, or bump the version in a release pull request)"
        )
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = re.search(rf"^## {re.escape(version)}\b(.*)$", changelog, re.MULTILINE)
    if heading is None:
        out.append(f"CHANGELOG.md has no '## {version}' section")
    elif "unreleased" in heading.group(1).lower():
        out.append(
            f"CHANGELOG.md still marks {version} as unreleased: "
            f"'## {version}{heading.group(1)}'; put the release date there"
        )
    citation = root / "CITATION.cff"
    if citation.exists():
        cited = re.search(
            r"^version:\s*['\"]?([^'\"\s]+)",
            citation.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        if cited is None or cited.group(1) != version:
            have = cited.group(1) if cited else "none"
            out.append(f"CITATION.cff gives version {have}, pyproject.toml {version}")
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: check_release.py <tag>", file=sys.stderr)
        return 2
    found = problems(argv[0])
    for problem in found:
        print(f"::error::{problem}")
    if not found:
        print(f"release {argv[0]}: tag, version and changelog agree")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

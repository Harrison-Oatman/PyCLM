"""scripts/check_release.py: a release's tag, version and changelog must agree."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_release.py"
spec = importlib.util.spec_from_file_location("check_release", SCRIPT)
check_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_release)


def _project(tmp_path, version, heading):
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n')
    (tmp_path / "CHANGELOG.md").write_text(
        f"# Changelog\n\n{heading}\n\n- a fix\n", encoding="utf-8"
    )
    return tmp_path


def test_agreeing_release_passes(tmp_path):
    root = _project(tmp_path, "1.2.0", "## 1.2.0 — 2026-11-02")
    assert check_release.problems("v1.2.0", root) == []


def test_mismatches_are_named(tmp_path):
    root = _project(tmp_path, "1.2.0", "## 1.2.0 — unreleased")
    found = check_release.problems("v1.2.1", root)
    assert len(found) == 2
    assert "does not match" in found[0]
    assert "unreleased" in found[1]
    root = _project(tmp_path, "1.2.0", "## 1.1.0 — 2026-10-01")
    assert "no '## 1.2.0' section" in check_release.problems("v1.2.0", root)[0]


def test_citation_version_must_match(tmp_path):
    root = _project(tmp_path, "1.2.0", "## 1.2.0 — 2026-11-02")
    (root / "CITATION.cff").write_text("cff-version: 1.2.0\nversion: 1.1.0\n")
    found = check_release.problems("v1.2.0", root)
    assert "CITATION.cff gives version 1.1.0" in found[0]
    (root / "CITATION.cff").write_text("cff-version: 1.2.0\nversion: 1.2.0\n")
    assert check_release.problems("v1.2.0", root) == []

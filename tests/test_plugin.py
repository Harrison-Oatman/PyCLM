"""
The Claude Code plugin in ``plugin/``: its manifest and hook files are valid,
the start-of-session note reads a lab context (role, this computer's
microscope, the project pointer) and says one line without one, the lab
context template parses, and the shared skills match the repository's.
"""

import filecmp
import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PLUGIN = REPO / "plugin"
TEMPLATE = PLUGIN / "templates" / "lab_context"


def _hook():
    spec = importlib.util.spec_from_file_location(
        "session_start", PLUGIN / "hooks" / "session_start.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hook(tmp_path, monkeypatch):
    module = _hook()
    monkeypatch.setattr(module, "USER_FILE", tmp_path / "no-user-file.toml")
    monkeypatch.delenv("PYCLM_LAB_CONTEXT", raising=False)
    monkeypatch.delenv("PYCLM_PERSON", raising=False)
    return module


@pytest.fixture
def lab(tmp_path):
    root = tmp_path / "lab"
    (root / "microscopes" / "scope-a").mkdir(parents=True)
    (root / "lab.md").write_text("# Lab\n")
    (root / "microscopes" / "scope-a" / "note.md").write_text("# Scope A\n")
    (root / "lab.toml").write_text(
        '[roles]\nadmins = ["Ada"]\n\n[microscopes.scope-a]\nhosts = ["SCOPE-PC"]\n'
    )
    return root


def test_manifest_and_hooks_are_valid():
    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    assert manifest["name"] == "pyclm"
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())
    command = hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    assert "${CLAUDE_PLUGIN_ROOT}/hooks/session_start.py" in command


def test_every_skill_has_a_name_and_description():
    for skill in sorted((PLUGIN / "skills").iterdir()):
        text = (skill / "SKILL.md").read_text(encoding="utf-8")
        head = text.split("---")[1]
        assert f"name: {skill.name}" in head
        assert "description: " in head


def test_shared_skills_match_the_repository():
    sys.path.insert(0, str(PLUGIN))
    try:
        from sync_skills import SHARED
    finally:
        sys.path.pop(0)
    for name in SHARED:
        cmp = filecmp.dircmp(
            REPO / ".claude" / "skills" / name, PLUGIN / "skills" / name
        )
        hint = f"plugin/skills/{name} differs; run python plugin/sync_skills.py"
        assert not cmp.left_only, hint
        assert not cmp.right_only, hint
        assert not cmp.diff_files, hint


def test_template_parses():
    lab = tomllib.loads((TEMPLATE / "lab.toml").read_text())
    assert "admins" in lab["roles"]
    for path in ("lab.md", "microscopes/_template/note.md", "proposals/README.md"):
        assert (TEMPLATE / path).exists()


def test_no_lab_context_is_one_line(hook, tmp_path):
    out = hook.note(cwd=tmp_path, host="ANY")
    assert out == hook.SETUP_LINE
    assert "\n" not in out


def test_a_user_on_a_known_microscope(hook, lab, tmp_path, monkeypatch):
    monkeypatch.setenv("PYCLM_LAB_CONTEXT", str(lab))
    monkeypatch.setenv("PYCLM_PERSON", "Ben")
    project = tmp_path / "project" / "day1"
    project.mkdir(parents=True)
    (tmp_path / "project" / "pyclm_project.md").write_text("# Project\n")
    out = hook.note(cwd=project, host="scope-pc")
    assert "role: user" in out
    assert str(lab / "microscopes" / "scope-a" / "note.md") in out
    assert "pyclm_project.md" in out
    assert "refer them to Ada" in out
    assert "Never edit lab.md" in out
    assert len(out.splitlines()) <= 20


def test_an_admin_elsewhere(hook, lab, tmp_path):
    user_file = tmp_path / "lab_context.toml"
    user_file.write_text(f'path = "{lab.as_posix()}"\nname = "ada"\n')
    hook.USER_FILE = user_file
    out = hook.note(cwd=tmp_path, host="LAPTOP")
    assert "role: admin" in out
    assert "no microscope note for this computer" in out
    assert "refer them" not in out
    assert "only when this person asks" in out


def test_the_hook_never_fails(tmp_path):
    env = {"PYCLM_LAB_CONTEXT": str(tmp_path / "missing"), "PATH": ""}
    result = subprocess.run(
        [sys.executable, str(PLUGIN / "hooks" / "session_start.py")],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    assert result.returncode == 0
    assert "no lab.md" in result.stdout


def test_marketplace_lists_the_plugin():
    market = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text())
    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    (entry,) = market["plugins"]
    assert entry["name"] == manifest["name"]
    assert (REPO / entry["source"] / ".claude-plugin" / "plugin.json").exists()


def test_mini_toml_reads_the_lab_files(hook, lab):
    data = hook._mini_toml((lab / "lab.toml").read_text())
    assert data == tomllib.loads((lab / "lab.toml").read_text())
    template = (TEMPLATE / "lab.toml").read_text()
    assert hook._mini_toml(template) == tomllib.loads(template)
    user = 'path = "C:/Lab Context"\nname = "Ada"  # comment\n'
    assert hook._mini_toml(user) == {"path": "C:/Lab Context", "name": "Ada"}


def test_launcher_sets_what_the_hook_reads():
    launcher = (TEMPLATE / "launcher" / "launch.ps1").read_text(encoding="utf-8")
    for name in ("CLAUDE_CONFIG_DIR", "PYCLM_PERSON", "PYCLM_LAB_CONTEXT"):
        assert f"$env:{name} =" in launcher
    assert "microscope.lock" in launcher  # pyclm.lock.LOCK_NAME
    bat = (TEMPLATE / "launcher" / "Start-PyCLM.bat").read_bytes()
    assert b"\r\n" in bat and b"launch.ps1" in bat
    json.loads((TEMPLATE / "launcher" / "local.example.json").read_text())

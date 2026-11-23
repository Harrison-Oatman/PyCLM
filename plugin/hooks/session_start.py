"""
The PyCLM plugin's SessionStart hook: a short note for the agent at the start
of every Claude Code session.

With a lab context configured for this person, the note names the files to
read (the lab file, this computer's microscope note, the project pointer if
the session starts in a project folder), the person's role, and the three
rules: read, record, propose. Without one it is a single line offering setup.

Where the lab context is found, first match wins:

1. ``PYCLM_LAB_CONTEXT``: the folder holding ``lab.md`` and ``lab.toml``;
2. ``~/.pyclm/lab_context.toml``: ``path = "<folder>"`` and ``name = "<person>"``
   (written by the pyclm-lab-setup skill).

Standard library only, so it runs with whatever ``python`` is on the PATH.
Whatever goes wrong, it prints at most one line and exits 0: a hook must
never stop a session from starting.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = None

USER_FILE = Path.home() / ".pyclm" / "lab_context.toml"
PROJECT_POINTER = "pyclm_project.md"
SETUP_LINE = (
    "PyCLM plugin: no lab context is set up on this computer. If the person "
    "wants to set up PyCLM (or Claude Code) for optogenetics in their lab, "
    "use the pyclm-lab-setup skill."
)


def _load_toml(path: Path) -> dict:
    if tomllib is None:
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def user_settings() -> dict:
    """``path`` and ``name`` for this person, from the environment or the user file."""
    settings: dict = {}
    if USER_FILE.exists():
        try:
            settings.update(_load_toml(USER_FILE))
        except Exception:
            pass
    if os.environ.get("PYCLM_LAB_CONTEXT"):
        settings["path"] = os.environ["PYCLM_LAB_CONTEXT"]
    if os.environ.get("PYCLM_PERSON"):
        settings["name"] = os.environ["PYCLM_PERSON"]
    return settings


def microscope_for_host(lab: dict, host: str) -> str | None:
    for name, scope in (lab.get("microscopes") or {}).items():
        hosts = [h.lower() for h in scope.get("hosts", [])]
        if host.lower() in hosts:
            return name
    return None


def role_of(lab: dict, person: str | None) -> str:
    admins = [a.lower() for a in (lab.get("roles") or {}).get("admins", [])]
    if person and person.lower() in admins:
        return "admin"
    return "user"


def find_project_pointer(start: Path) -> Path | None:
    for folder in (start, *start.parents):
        candidate = folder / PROJECT_POINTER
        if candidate.exists():
            return candidate
    return None


def note(cwd: Path | None = None, host: str | None = None) -> str:
    settings = user_settings()
    if not settings.get("path"):
        return SETUP_LINE
    root = Path(settings["path"]).expanduser()
    if not (root / "lab.md").exists():
        return (
            f"PyCLM plugin: the lab context folder {root} has no lab.md "
            "(check ~/.pyclm/lab_context.toml or PYCLM_LAB_CONTEXT)."
        )
    try:
        lab = _load_toml(root / "lab.toml") if (root / "lab.toml").exists() else {}
    except Exception:
        lab = {}
    host = platform.node() if host is None else host
    person = settings.get("name")
    role = role_of(lab, person)
    scope = microscope_for_host(lab, host)
    pointer = find_project_pointer(Path.cwd() if cwd is None else cwd)
    admins = ", ".join((lab.get("roles") or {}).get("admins", [])) or "the lab admin"

    lines = [
        f"PyCLM lab context: {root}",
        f"Working for: {person or 'unknown (ask their name)'}; role: {role}.",
        "Read now, before planning anything:",
        f"- {root / 'lab.md'}",
    ]
    if scope:
        lines.append(
            f"- {root / 'microscopes' / scope / 'note.md'} (this computer, {host}); "
            "read its 'Check at session start' list out to the person"
        )
    else:
        lines.append(
            f"- no microscope note for this computer ({host}); if this is a "
            "microscope PC, ask which microscope it is"
        )
    if pointer:
        lines.append(f"- {pointer} (this project)")
    lines += [
        "Rules:",
        "- Never edit lab.md, lab.toml or a microscope note directly; write a "
        f"proposal in {root / 'proposals'} (or a pull request if this person "
        "can push), and say so.",
        "- Every run writes run_record.json; at the end of a session, use the "
        "pyclm-close-session skill.",
    ]
    if role == "user":
        lines.append(
            "- This person is a user: for the admin-only decisions listed in "
            f"lab.md, do not ask them to decide; stop and refer them to {admins}."
        )
    return "\n".join(lines)


def main() -> int:
    try:
        print(note())
    except Exception as e:  # never block a session
        print(f"PyCLM plugin: lab context not loaded ({type(e).__name__}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

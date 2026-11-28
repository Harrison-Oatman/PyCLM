"""
Copy the repository's skills (``.claude/skills/``) into the plugin
(``plugin/skills/``), so the plugin ships them without a second copy to edit.

The repository's copies are the ones to edit; run this after changing them:

    uv run python plugin/sync_skills.py

``tests/test_plugin.py`` fails while the two differ.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / ".claude" / "skills"
TARGET = REPO / "plugin" / "skills"
SHARED = (
    "pyclm-design-experiment",
    "pyclm-microscope-bringup",
    "pyclm-write-pattern-method",
)


def sync() -> list[Path]:
    written = []
    for name in SHARED:
        dest = TARGET / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(SOURCE / name, dest)
        written.append(dest)
    return written


if __name__ == "__main__":
    for path in sync():
        print(f"synced {path.relative_to(REPO)}")
    sys.exit(0)

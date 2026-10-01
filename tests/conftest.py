"""Shared test setup."""

import os

# The viewer tests start Qt. Without a display (CI runners, SSH sessions) the
# default platform plugin aborts, so render offscreen unless told otherwise.
# Set before any test module imports Qt; subprocesses inherit it.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

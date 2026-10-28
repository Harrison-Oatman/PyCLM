"""Console set-up shared by the ``pyclm`` command and :func:`pyclm.run_pyclm`."""

import sys


def utf8_console() -> None:
    """
    Make stdout and stderr write UTF-8, replacing what they cannot encode.
    On Windows a console redirected to a file defaults to cp1252, and
    pymmcore-plus's device table (``core.describe()``) prints an emoji that
    would otherwise end the run before the first frame.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass

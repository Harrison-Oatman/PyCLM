"""Nox sessions for local development tasks."""

import nox

nox.options.sessions = ["docs"]

DOCS_SRC = "documentation"
DOCS_OUT = "documentation/_build/html"


@nox.session(venv_backend="none")
def docs(session: nox.Session) -> None:
    """Build the HTML documentation (mirrors the ReadTheDocs build).

    Pass extra sphinx-build flags after ``--``, e.g.::

        nox -s docs -- -W --keep-going
    """
    session.run(
        "uv",
        "run",
        "--group",
        "docs",
        "sphinx-build",
        "-b",
        "html",
        *session.posargs,
        DOCS_SRC,
        DOCS_OUT,
        external=True,
    )
    session.log("Docs built → %s/index.html", DOCS_OUT)


@nox.session(venv_backend="none")
def docs_clean(session: nox.Session) -> None:
    """Remove the previous HTML build, then rebuild from scratch."""
    import shutil

    shutil.rmtree(DOCS_OUT, ignore_errors=True)
    session.log("Removed %s", DOCS_OUT)
    docs(session)

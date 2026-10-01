"""
Sphinx directive ``pyclm-schema``: a table of the keys of one configuration
model from :mod:`pyclm.schema`, generated at build time so the TOML
reference cannot drift from the code.

Usage in a MyST page::

    ```{pyclm-schema} ExperimentConfig
    ```
"""

from __future__ import annotations

from docutils import nodes
from docutils.parsers.rst import Directive

HEADERS = ("Key", "Type", "Default", "Meaning")


def _cell(text: str) -> nodes.entry:
    entry = nodes.entry()
    entry += nodes.paragraph(text=text)
    return entry


def _table(rows: list[dict]) -> nodes.table:
    table = nodes.table()
    group = nodes.tgroup(cols=len(HEADERS))
    table += group
    for width in (18, 22, 14, 46):
        group += nodes.colspec(colwidth=width)
    head = nodes.thead()
    group += head
    row = nodes.row()
    for h in HEADERS:
        row += _cell(h)
    head += row
    body = nodes.tbody()
    group += body
    for r in rows:
        row = nodes.row()
        for key in ("key", "type", "default", "description"):
            row += _cell(str(r[key]))
        body += row
    return table


class SchemaTable(Directive):
    required_arguments = 1
    has_content = False

    def run(self):
        from pyclm.schema import MODELS, field_table

        name = self.arguments[0]
        model = MODELS.get(name)
        if model is None:
            error = self.state_machine.reporter.error(
                f"pyclm-schema: unknown model {name!r} (known: {', '.join(sorted(MODELS))})",
                line=self.lineno,
            )
            return [error]
        return [_table(field_table(model))]


def setup(app):
    app.add_directive("pyclm-schema", SchemaTable)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}

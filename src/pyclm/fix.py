"""
``pyclm check --fix``: rewrite an experiment directory's TOMLs from old
pattern-method names and keywords to the current ones.

What changes comes from the methods themselves: a renamed method's alias
(:func:`~pyclm.core.patterns.pattern.renamed_method`) names its new name,
and :meth:`~pyclm.core.patterns.pattern.PatternMethod.migrate_kwargs`
gives the arguments in their current names, which is also what a run does
with an old TOML. The file is edited line by line, so comments and layout
survive; the original is kept next to it as ``<name>.toml.bak``. A table
the line editor cannot handle safely (an inline table, a changed value that
spans lines) is reported and left alone.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

_HEADER = re.compile(r"^\s*\[([^\[\]]+)\]\s*(#.*)?$")
_KEY = re.compile(r"^(\s*)([A-Za-z0-9_-]+)(\s*=\s*)(.*)$")


@dataclass
class FileFix:
    """What ``fix_file`` did to one TOML."""

    path: Path
    changes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    written: bool = False


def format_value(value) -> str:
    """A Python value as a TOML value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(round(value, 9))
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(format_value(v) for v in value) + "]"
    raise TypeError(f"cannot write {value!r} as a TOML value")


def _split_comment(text: str) -> tuple[str, str]:
    """``value  # comment`` -> (value, '  # comment'), ignoring # inside strings."""
    quote = None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            j = i
            while j > 0 and text[j - 1] in " \t":
                j -= 1
            return text[:j], text[j:]
    return text.rstrip(), ""


def _pattern_tables(data: dict) -> list[tuple[str, dict]]:
    """``(header, table)`` for ``[pattern]`` and each nested method table under it."""
    pattern = data.get("pattern")
    if not isinstance(pattern, dict):
        return []
    out = [("pattern", pattern)]
    for key, value in pattern.items():
        if isinstance(value, dict) and "method" in value:
            out.append((f"pattern.{key}", value))
    return out


def plan_table(table: dict, registry: dict) -> tuple[str | None, dict, list[str], dict]:
    """
    The current method name (None if unchanged), the arguments in current
    names, the notes, and the plain renames (old key -> new key), for one
    TOML method table.
    """
    from .check import migrate_pattern_kwargs
    from .core.patterns.pattern import _merged_class_dicts

    name = str(table.get("method", ""))
    cls = registry.get(name)
    kwargs = {
        k: v
        for k, v in table.items()
        if k != "method" and not (isinstance(v, dict) and "method" in v)
    }
    if cls is None:
        return None, kwargs, [], {}
    new_kwargs, notes, error = migrate_pattern_kwargs(cls, kwargs)
    if error:
        raise ValueError(error)
    new_name = getattr(cls, "renamed_to", "") or None
    if new_name:
        notes = [f'method "{name}" is now "{new_name}"', *notes]
    renames = {
        old: new
        for old, new in _merged_class_dicts(cls, "renamed_kwargs").items()
        if old in kwargs and new in new_kwargs and new_kwargs[new] == kwargs[old]
    }
    return new_name, new_kwargs, notes, renames


def fix_text(text: str, registry: dict) -> tuple[str, list[str], list[str]]:
    """The rewritten TOML text, the changes made and the problems left alone."""
    data = tomllib.loads(text)
    lines = text.splitlines(keepends=True)
    changes: list[str] = []
    problems: list[str] = []
    headers = {}
    current = None
    for i, line in enumerate(lines):
        m = _HEADER.match(line)
        if m:
            current = m.group(1).strip()
            headers.setdefault(current, i)
    edits: dict[
        int, list[str] | None
    ] = {}  # line index -> replacement lines (None: drop)
    for header, table in _pattern_tables(data):
        try:
            new_name, new_kwargs, notes, renames = plan_table(table, registry)
        except ValueError as e:
            problems.append(f"[{header}]: {e}")
            continue
        if not notes:
            continue
        if header not in headers:
            problems.append(
                f"[{header}] is an inline table; update it by hand: {'; '.join(notes)}"
            )
            continue
        start = headers[header]
        end = next(
            (j for j in range(start + 1, len(lines)) if _HEADER.match(lines[j])),
            len(lines),
        )
        old_kwargs = {
            k: v
            for k, v in table.items()
            if k != "method" and not (isinstance(v, dict) and "method" in v)
        }
        # a plain rename keeps its line, value and comment; only the key changes
        removed = [k for k in old_kwargs if k not in new_kwargs and k not in renames]
        added = [
            k for k in new_kwargs if k not in old_kwargs and k not in renames.values()
        ]
        changed = [
            k for k in new_kwargs if k in old_kwargs and new_kwargs[k] != old_kwargs[k]
        ]
        table_edits: dict[int, list[str] | None] = {}
        first_removed = None
        method_line = None
        ok = True
        for j in range(start + 1, end):
            m = _KEY.match(lines[j])
            if not m:
                continue
            indent, key, eq, rest = m.groups()
            value_text, comment = _split_comment(rest.rstrip("\r\n"))
            newline = "\n" if lines[j].endswith("\n") else ""
            touched = key in removed or key in changed or (key == "method" and new_name)
            if touched:
                try:
                    tomllib.loads(f"x = {value_text}")
                except tomllib.TOMLDecodeError:
                    problems.append(
                        f"[{header}] {key} spans several lines; update it by hand: "
                        + "; ".join(notes)
                    )
                    ok = False
                    break
            if key == "method":
                method_line = j
                if new_name:
                    table_edits[j] = [
                        f"{indent}{key}{eq}{format_value(new_name)}{comment}{newline}"
                    ]
            elif key in renames:
                table_edits[j] = [f"{indent}{renames[key]}{eq}{rest}{newline}"]
            elif key in removed:
                if first_removed is None:
                    first_removed = j
                table_edits[j] = None
            elif key in changed:
                table_edits[j] = [
                    f"{indent}{key}{eq}{format_value(new_kwargs[key])}{comment}{newline}"
                ]
        if not ok:
            continue
        new_lines = [f"{k} = {format_value(new_kwargs[k])}\n" for k in added]
        if new_lines:
            if first_removed is not None:
                at = first_removed
                table_edits[at] = new_lines
            else:
                at = method_line if method_line is not None else start
                table_edits[at] = (table_edits.get(at) or [lines[at]]) + new_lines
        edits.update(table_edits)
        changes += [f"[{header}] {n}" for n in notes]
    out = []
    for i, line in enumerate(lines):
        if i in edits:
            if edits[i] is not None:
                out += edits[i]
        else:
            out.append(line)
    return "".join(out), changes, problems


def fix_file(path, registry: dict, write: bool = True) -> FileFix:
    """Rewrite one experiment TOML; the original is kept as ``<name>.toml.bak``."""
    path = Path(path)
    result = FileFix(path)
    text = path.read_text(encoding="utf-8")
    try:
        new_text, result.changes, result.problems = fix_text(text, registry)
    except tomllib.TOMLDecodeError as e:
        result.problems.append(f"not valid TOML: {e}")
        return result
    if new_text != text and write:
        tomllib.loads(new_text)  # never write a file that no longer parses
        path.with_name(path.name + ".bak").write_text(text, encoding="utf-8")
        path.write_text(new_text, encoding="utf-8")
        result.written = True
    return result


def fix_directory(directory, registry: dict | None = None) -> list[FileFix]:
    """``fix_file`` on every experiment TOML of a directory."""
    from .core.patterns import known_models
    from .directories import experiment_tomls

    registry = dict(known_models) if registry is None else registry
    return [fix_file(p, registry) for p in experiment_tomls(directory).values()]

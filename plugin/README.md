# PyCLM plugin for Claude Code (in progress)

This branch (`plugin`) holds the Claude Code plugin for PyCLM. It stays off
`main` until the release, which comes with full documentation. Nothing here
changes PyCLM itself; the engine features the plugin uses (`pyclm run
--detached`, `pyclm watch`, `run_record.json`, the microscope lock) are on
`main` with their own documentation.

## What is here so far

| Path | What it is |
| --- | --- |
| `.claude-plugin/plugin.json` | the plugin manifest |
| `hooks/session_start.py` | the start-of-session note: which lab, microscope and project files to read, the person's role, and the rules (read, record, propose); one line when no lab context is set up |
| `skills/pyclm-lab-setup/` | set up a lab or core facility: the interview, the lab context folder, connecting a person |
| `skills/pyclm-close-session/` | run records to the project; learnings to proposals |
| `skills/pyclm-design-experiment/`, `pyclm-microscope-bringup/`, `pyclm-write-pattern-method/` | copies of the repository's skills (`.claude/skills/`, edited there; `python plugin/sync_skills.py` copies them) |
| `templates/lab_context/` | the lab context folder: `lab.md`, `lab.toml`, a microscope note, a project pointer, the proposals inbox |

## Still to come

1. Run chain: back-to-back runs on `pyclm run --detached` and `pyclm watch`, with a session log.
2. Design builder (with the design skill) and the live-analysis skill.
3. Scouting and picking: plate map, scout scoring, a pick page.
4. Dashboard and analysis cards.
5. Documentation: install, each skill, lab and facility setup, roles, the proposals inbox.

## Trying it before the release

```bash
claude --plugin-dir plugin
```

starts Claude Code with the plugin from this checkout. The hook needs a
`python` (3.11 or later) on the `PATH`.

# PyCLM plugin for Claude Code (in progress)

This branch (`plugin`) holds the Claude Code plugin for PyCLM. It stays off
`main` until the release, which comes with full documentation. Nothing here
changes PyCLM itself; the engine features the plugin uses (`pyclm run
--detached`, `pyclm watch`, `run_record.json`, the microscope lock) are on
`main` with their own documentation, and in PyPI releases from 1.2.0.

## Install

On the computer that runs the microscope, with Claude Code installed:

```bash
claude plugin marketplace add Harrison-Oatman/PyCLM@plugin
claude plugin install pyclm@pyclm-plugins
```

or, inside a Claude Code session, `/plugin install pyclm --marketplace
Harrison-Oatman/PyCLM@plugin`. Start a new session afterwards; the start
note and the `pyclm-*` skills come with it. `claude plugin marketplace
update pyclm-plugins` picks up later changes to the branch.

The start-of-session note needs `python` (3.11 or later; 3.8 to 3.10 also
work) on the `PATH` of the shell Claude Code uses: check `python
--version` there. On Windows, the Python from python.org with "Add to
PATH" ticked is the simplest. Python from the Microsoft Store works too;
the Store's placeholder alias, with no Python installed behind it, does not.
Without it, the skills still work, but sessions start without the note.

PyCLM itself is installed separately, by the bring-up skill or by hand:

```bash
pip install "closed-loop-microscopy[gui]>=1.2"
```

On a microscope PC that several people share, the lab context's
`launcher\Start-PyCLM.bat` does the plugin install for each person's
profile the first time they use it (`templates/lab_context/README.md`).

## Where to start

Say what you want in plain words; the skills match on it. For a new
microscope, "set up PyCLM on this microscope and run an experiment today"
starts `pyclm-first-day`, which orders the rest:

1. `pyclm-lab-setup`: the lab context folder (rules, roles, data,
   microscope notes) and this person's connection to it.
2. `pyclm-microscope-bringup`: install, device interface, `.cfg`,
   calibration, first light.
3. `pyclm-design-experiment`: the idea to a checked experiment directory.
4. `pyclm-run`: start detached, follow with `pyclm watch`, flags, chains.
5. `pyclm-close-session`: run records and proposals.

The lab decides what needs approval before light goes on cells; the
plugin follows `lab.md` and sets no rule of its own.

## What is here so far

| Path | What it is |
| --- | --- |
| `.claude-plugin/plugin.json` | the plugin manifest |
| `hooks/session_start.py` | the start-of-session note: which lab, microscope and project files to read, the person's role, and the rules (read, record, propose); one line when no lab context is set up |
| `skills/pyclm-lab-setup/` | set up a lab or core facility: the interview, the lab context folder, connecting a person |
| `skills/pyclm-first-day/` | the order of a first day on a new microscope, what to do before it, what to cut |
| `skills/pyclm-run/` | start a run detached, follow it with `pyclm watch`, report flags, pause or stop on request, chain runs with `--then` |
| `skills/pyclm-session-pages/` | read-only HTML pages in the session folder (Design, Positions, Running, others as needed), built from files with one style (`template.html`); nothing on a page controls the run |
| `skills/pyclm-close-session/` | run records to the project (or a summary from `status.json` without one); learnings to proposals |
| `skills/pyclm-design-experiment/`, `pyclm-microscope-bringup/`, `pyclm-write-pattern-method/` | copies of the repository's skills (`.claude/skills/`, edited there; `python plugin/sync_skills.py` copies them) |
| `templates/lab_context/` | the lab context folder: `lab.md`, `lab.toml`, a microscope note, a project pointer, the proposals inbox, and `launcher/` (`Start-PyCLM.bat`, the desktop start file for a microscope PC several people share: own Claude sign-in, sessions and plugin per person; new, continue or resume; `Start-MicroManager.bat`, the Micro-Manager GUI with the scope's own start-up steps, refused while a run holds the microscope) |

## Still to come

1. Run chain beyond `--then`: a session log of planned and finished runs.
2. Design builder (with the design skill) and the live-analysis skill.
3. Scouting and picking: plate map, scout scoring, a pick page.
4. Dashboard and analysis cards.
5. Documentation pages: each skill, lab and facility setup, roles, the proposals inbox.

## Trying it from a checkout

```bash
claude --plugin-dir plugin
```

starts Claude Code with the plugin from this checkout, without installing
it. `claude plugin validate .` and `claude plugin validate plugin` check the
marketplace and the plugin.

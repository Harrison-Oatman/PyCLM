# Using PyCLM with Claude Code

PyCLM comes with a plugin for [Claude Code](https://claude.com/claude-code),
a coding agent that works in a terminal on the microscope PC. With the
plugin, the agent can walk a lab through setting PyCLM up on a microscope,
turn an experiment idea into a checked experiment folder, start the run,
follow it, and write up the session. It runs the `pyclm` commands itself
and asks people only what it cannot know: the microscope, the sample and
the science.

The plugin adds three things to every Claude Code session:

- **Skills**: step-by-step guides the agent follows (below).
- **A start-of-session note**: which lab, microscope and project files to
  read first, and the person's role, from the lab's own context folder.
- **Session pages**: read-only HTML pages in the session folder that show
  a design, the picked positions or a running experiment at a glance.

The lab decides what needs a person's approval, for example before light
goes on cells. The plugin follows the lab's rules (`lab.md`, below) and
sets none of its own.

## Install

You need, on the computer that runs the microscope:

- Claude Code ([install](https://code.claude.com/docs/en/quickstart)) and
  a Claude account to sign in with;
- PyCLM 1.2 or later (`pip install "closed-loop-microscopy[gui]>=1.2"`, or
  let the bring-up skill install it);
- `python` 3.8 or later on the `PATH` of the shell Claude Code uses, for
  the start-of-session note. On Windows, Python from python.org with "Add
  to PATH" ticked is the simplest; the Microsoft Store's placeholder alias,
  with no Python behind it, does not work. Without it the skills still
  work, but sessions start without the note.

Then, in a terminal:

```bash
claude plugin marketplace add Harrison-Oatman/PyCLM
claude plugin install pyclm@pyclm-plugins
```

Start a new Claude Code session afterwards. To update later:

```bash
claude plugin marketplace update pyclm-plugins
claude plugin update pyclm@pyclm-plugins
```

On a microscope PC that several people share, use the launcher instead
(below); it installs and updates the plugin for each person.

## Where to start

Say what you want in plain words; the agent picks the skill. For a
microscope that has never run PyCLM, *"set up PyCLM on this microscope and
run an experiment today"* starts the first-day guide, which orders the rest.

| Skill | What it does |
| --- | --- |
| `pyclm-first-day` | The order of a first day on a new microscope, what to prepare before it, and what to cut when time runs short. |
| `pyclm-lab-setup` | The lab context folder (rules, roles, data, microscope notes) and each person's connection to it. |
| `pyclm-microscope-bringup` | Install, the Micro-Manager device interface, the `.cfg` presets, `pyclm_config.toml` (DMD calibration, camera ROI, pixel size, focus lock) and first light. |
| `pyclm-design-experiment` | An experiment idea to a checked experiment directory, a pattern preview and a dry run. |
| `pyclm-write-pattern-method` | A light rule described in words to a tested pattern method. |
| `pyclm-run` | Start the run detached (`pyclm run --detached`), follow it with `pyclm watch`, report flags, chain runs, pause or stop on request. |
| `pyclm-session-pages` | Design, Positions and Running pages, and others as needed. |
| `pyclm-close-session` | Run records to the project, and what was learned as proposals for the lab. |

Some things the agent cannot do, and asks a person for:

- **Focus and positions.** The person focuses each position in
  Micro-Manager (with the focus lock, if there is one), saves the Stage
  Position List as `PositionList.pos` in the experiment folder, and closes
  Micro-Manager. The GUI and a PyCLM run cannot share the hardware.
- **Anything physical**: the sample, the calibration slide, filters,
  objectives.
- **Decisions that change the science**: light dose, timing, controls. The
  agent proposes; the researcher decides.

## The lab context folder

A lab keeps what every session should know in one folder, made from the
template in the plugin (`templates/lab_context/`) by the `pyclm-lab-setup`
skill:

| File | Holds |
| --- | --- |
| `lab.md` | approval rules, roles, admin-only decisions, data conventions, notifications |
| `lab.toml` | the same facts for scripts: admins, which computer is which microscope |
| `microscopes/<name>/note.md` | how to start PyCLM there, focus, data paths, quirks, timings, dated temporary issues |
| `microscopes/<name>/pyclm_config.toml` | the calibrated configuration in use |
| `proposals/` | changes proposed by sessions, for an admin or the microscope owner to review |
| `launcher/` | start files for a shared microscope PC |

Sessions read these files and do not edit them. What a session learns
becomes a proposal, which the owner accepts or not. The folder can be a git
repository or a plain shared folder.

## A microscope PC that several people share

Put shortcuts to two files from the lab context's `launcher\` folder on the
desktop (Windows):

- **`Start-PyCLM.bat`** asks who you are, then which project, then whether
  to start a new Claude Code session, continue the last one or pick an
  earlier one. Each person has a profile with their own Claude sign-in,
  settings, session history and plugin install; it offers to update the
  plugin at each start, and shows when a PyCLM run holds the microscope. An
  admin can allow a guest option for visitors, whose profile is deleted
  when they close Claude.
- **`Start-MicroManager.bat`** starts the Micro-Manager GUI the way that
  microscope needs it, with any steps that must come first (set in
  `launcher\local.json`), and will not start while a run holds the
  microscope unless you insist.

Settings for one PC (where `claude.exe` is, the projects folder, guests,
Micro-Manager start-up steps) go in `launcher\local.json`, copied from
`local.example.json`.

## Session pages

At the moments a person has to look at something, the agent writes a page
into the session folder and opens it in the browser:

- **Design**: the pattern of each condition on a sample image, the light
  program as a time strip, and the `pyclm check` result, before the design
  is approved.
- **Positions**: the stitched overview with the picked fields outlined and
  numbered, before the positions are approved.
- **Running**: timepoint of total, end time, flags, the live analysis
  figure and the newest frame of every position, rebuilt every few minutes.

Pages are built only from files the run and the `pyclm` commands write.
They never control the run: approvals are always typed in the
conversation. Each page is one HTML file plus an `img/` folder, opens from
the file system, and has a "Figure view" for slides.

## Without the plugin

The three skills that are not specific to a lab
(`pyclm-microscope-bringup`, `pyclm-design-experiment`,
`pyclm-write-pattern-method`) are also in the repository, in
[`.claude/skills/`](https://github.com/Harrison-Oatman/PyCLM/tree/main/.claude/skills).
Start Claude Code in a clone of PyCLM and they are available:

```bash
git clone https://github.com/Harrison-Oatman/PyCLM.git
cd PyCLM
claude
```

To use them from another folder, copy those skill folders into that
folder's `.claude/skills/`, or into `~/.claude/skills/` to have them
everywhere. The skills are plain Markdown, so other agents, or a person,
can follow them too.

The agent never changes your Micro-Manager configuration; it tells you
what to change. Its work on the virtual microscope (`--dry`) is safe to
try before going to the scope.

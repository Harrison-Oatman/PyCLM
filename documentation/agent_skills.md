# Using PyCLM with Claude Code

The easiest way to use PyCLM is to ask for what you want. The PyCLM plugin
for [Claude Code](https://claude.com/claude-code), a coding agent that runs
in a terminal on the microscope PC, knows how to set PyCLM up on a
microscope, turn an experiment idea into a checked experiment folder, start
the run, follow it, and write up the session. It runs the `pyclm` commands
itself and asks you only what it cannot know: your microscope, your sample
and your science. Before light goes on cells, it shows you the design on a
page and waits for your go-ahead.

## Install

On the computer that runs the microscope, with Claude Code installed
([instructions](https://code.claude.com/docs/en/quickstart)):

```bash
claude plugin marketplace add Harrison-Oatman/PyCLM
claude plugin install pyclm@pyclm-plugins
```

Then start Claude Code (`claude`) in a folder for your project and say what
you want. The agent installs PyCLM itself if it is missing. On a microscope
PC that several people share, use the start file instead (see "A
microscope PC that several people share", below).

## Example: a 30-minute moving-bar run

This example is from a test on an MDCK monolayer on a Nikon Ti2 with a DMD,
which had run PyCLM before. The researcher had focused one field in Micro-Manager
with Perfect Focus on, saved it as `PositionList.pos` in an empty project
folder, closed Micro-Manager, started Claude Code there, and typed:

> I want a 30-minute moving-bar run in well B7: a few fields near the saved
> position, at least one of them unlit. Ask me what you need to know about
> the design. Show me the design and the positions on session pages before
> asking me to approve them, and keep a Running page while it runs. Do not
> scout beyond this one well.

**What the agent did first, on its own.** It read the lab's rules and the
microscope's note (which configuration, which data folder, what needs
approval), checked that the installed PyCLM was 1.2 or later, and saw that the
saved position was the unlit field of an earlier run.

**What it asked**, in one message: the bar (width, spacing, speed,
direction), the light dose and the imaging settings, which it proposed to
copy from the earlier run; the cadence (it suggested 15 timepoints every
2 minutes; the researcher chose 30 every minute); and whether to scout the
well first or place fields at fixed offsets. The researcher answered in one
reply.

**Then, without further questions,** it imaged a 3 × 3 grid around the saved
position with no light (24 seconds), wrote the experiment folder, ran
`pyclm check`, previewed the pattern at every timepoint on the real scout
frames, rehearsed three timepoints on the virtual microscope, and opened two
pages in the browser:

```{figure} imgs/plugin/page_design.jpg
:alt: The Design page

**Design page.** The bar on each lit field and the unlit control at t = 0,
the light program as a strip of 30 exposures, the settings, the pattern
over the whole run, and the `pyclm check` output.
```

```{figure} imgs/plugin/page_positions.jpg
:alt: The Positions page

**Positions page.** The 3 × 3 scout with the three picked fields outlined in
their condition's color and numbered, and why they were picked: the saved
position stays unlit, its two neighbors get the bar.
```

It asked for approval in the conversation. The researcher typed *"Approve the
positions and the design"*, and the agent started the run detached
(`pyclm run --detached`, so closing the terminal does not stop it),
started `pyclm watch` to look for problems every minute, and opened the
Running page, which it rebuilt every five minutes:

```{figure} imgs/plugin/page_running.jpg
:alt: The Running page at the end of the run

**Running page**, at the end of the run. Midway through, the researcher asked
for reporter intensity against position relative to the bar; the agent
wrote the analysis, read the run's data as it was written, and put the
figure at the top of the page. Below it are the newest frame of every field
and each position's health.
```

The run finished 30 of 30 timepoints at all three positions with no errors or
dropped frames. At the end the agent wrote a record of every point where it
needed the researcher (three, beyond the first message) and three proposals
for the lab, for example to note in the microscope's file that this PC's
PyCLM 1.2 lives in its own folder. An admin accepts or declines proposals;
sessions never change the lab's files themselves.

## More things to ask

| Say something like | What happens |
| --- | --- |
| *"Set up PyCLM on this microscope and run an experiment today."* | The first-day guide: the lab's context folder, install and the Micro-Manager device interface, DMD calibration on a fluorescent slide, first light, then a short run on cells. It tells you what to prepare, orders the day, and says what to cut when time runs short. |
| *"Light the outward half of each cell, and leave every other well dark."* | A design: the agent picks a built-in pattern method from the [method zoo](method_zoo.md), or writes one, and adds segmentation if the light depends on the image. |
| *"Light cells whose nuclear/cytosolic ratio is below 0.8."* | A new pattern method, written as code, tested on your images and previewed before it runs. |
| *"Run the plan in this folder."* | A design made earlier on another computer: the agent swaps in this microscope's configuration and today's positions, checks it, and shows you what changed before asking to start. |
| *"How is the run going?"* | Timepoint of total, on time or how late, and any flags (focus, brightness, errors), with the evidence. Flags never stop a run on their own. |
| *"Run this again overnight with the bar at half speed."* | A second run chained after the first (`--then`), with both shown for one approval. |
| *"We're done for today."* | Run records copied to the project, and what was learned written as proposals for the lab. |

Some things the agent cannot do, and asks you for:

- **Focus and positions.** You focus in Micro-Manager (with the focus lock,
  if there is one), save the Stage Position List as `PositionList.pos`,
  and close Micro-Manager before the run; the two cannot share the
  hardware. The agent can then scout around your position without light.
- **Anything physical**: the sample, the calibration slide, filters,
  objectives.
- **Decisions that change the science**: dose, timing, controls. The agent
  proposes; you decide.

## What the plugin adds

- **Skills**, step-by-step guides the agent follows:

  | Skill | What it covers |
  | --- | --- |
  | `pyclm-first-day` | the order of a first day on a new microscope |
  | `pyclm-lab-setup` | the lab context folder and each person's connection to it |
  | `pyclm-microscope-bringup` | install, device interface, `.cfg` presets, `pyclm_config.toml` (DMD calibration, camera ROI, pixel size, focus lock), first light |
  | `pyclm-design-experiment` | an idea to a checked experiment directory, a preview and a dry run |
  | `pyclm-write-pattern-method` | a light rule in words to a tested pattern method |
  | `pyclm-run` | start detached, follow with `pyclm watch`, flags, chains, pause or stop on request |
  | `pyclm-session-pages` | the Design, Positions and Running pages, and others as needed |
  | `pyclm-close-session` | run records and proposals |

- **A start-of-session note**: which lab, microscope and project files to
  read first, and the person's role.
- **Session pages**: HTML files in the session folder, built only from files
  the run and the `pyclm` commands write. They open from the file system,
  never control the run, and have a "Figure view" for slides. Approvals are
  always typed in the conversation.

## The lab context folder

A lab keeps what every session should know in one folder, made from the
template in the plugin by the `pyclm-lab-setup` skill (*"set up PyCLM for our
lab"*):

| File | Holds |
| --- | --- |
| `lab.md` | approval rules (for example, "show me the design before light goes on cells"), roles, data conventions, notifications |
| `lab.toml` | the same facts for scripts: admins, which computer is which microscope |
| `microscopes/<name>/note.md` | how PyCLM runs there, focus, data paths, quirks, timings, dated temporary issues |
| `microscopes/<name>/pyclm_config.toml` | the calibrated configuration in use |
| `proposals/` | changes proposed by sessions, for an admin or the microscope's owner to review |
| `launcher/` | start files for a shared microscope PC |

The plugin sets no approval rules of its own; it follows `lab.md`.

## A microscope PC that several people share

Put shortcuts to two files from the lab context's `launcher\` folder on the
Windows desktop. **`Start-PyCLM.bat`** gives each person their own Claude
sign-in, settings, session history and plugin, and starts Claude Code in a
project folder:

```text
MICROSCOPE IN USE: a run of D:\data\2026-11-27_run2_B7, started 2026-11-27T18:16:20 by Sam.
You can plan, design or look at data; a second run will refuse to start.

Who are you?
  1. Sam
  2. Priya
  a. Add me
Choice: 1
Update the PyCLM plugin first? (y, or Enter to skip):

Which project?
  1. wave-speed
  2. plugin-test
  n. New project
  s. No project (setup, a question, looking around)
Choice: 2

Session:
  n. New session
  c. Continue my last session here
  r. Pick one of my earlier sessions here
Choice: c
```

An admin can also offer a guest option for visitors, whose profile is
deleted when they close Claude. **`Start-MicroManager.bat`** opens the
Micro-Manager GUI with any steps that microscope needs first, and refuses
while a PyCLM run holds the microscope unless you insist. Settings for one PC
go in `launcher\local.json` (copied from `local.example.json`).

## Requirements and updates

- PyCLM 1.2 or later on the microscope PC (the agent can install it:
  `pip install "closed-loop-microscopy[gui]>=1.2"`).
- `python` 3.8 or later on the `PATH` of the shell Claude Code uses, for the
  start-of-session note. On Windows, Python from python.org with "Add to
  PATH" ticked is simplest; the Microsoft Store's placeholder alias, with
  no Python behind it, does not work. Without it the skills still work, but
  sessions start without the note.
- To update: `claude plugin marketplace update pyclm-plugins`, then
  `claude plugin update pyclm@pyclm-plugins` (the start file offers this).

## Without the plugin

Three of the skills (`pyclm-microscope-bringup`, `pyclm-design-experiment`,
`pyclm-write-pattern-method`) are also in the repository, in
[`.claude/skills/`](https://github.com/Harrison-Oatman/PyCLM/tree/main/.claude/skills).
Start Claude Code in a clone of PyCLM and they are available; copy them into
another folder's `.claude/skills/`, or `~/.claude/skills/`, to use them
there. They are plain Markdown, so other agents, or a person, can follow
them too. Everything the agent does is a `pyclm` command you can also run
yourself ([the `pyclm` command](command_line.md)).

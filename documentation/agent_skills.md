# Using PyCLM with Claude Code

PyCLM comes with a plugin for [Claude Code](https://claude.com/claude-code),
a coding agent that runs in a terminal on the microscope PC. With the
plugin, Claude can set PyCLM up on a microscope, write and check an
experiment, run it, follow it and export the data. It runs the `pyclm`
commands described elsewhere in this documentation. Light goes on cells
only after you approve a design, and decisions about the science stay
with you.

```{figure} imgs/plugin/overview_draft.svg
:alt: Overview of PyCLM with Claude Code (draft)

Overview (draft).
```

## Installation and setup

On the microscope PC:

1. Install Claude Code ([instructions](https://code.claude.com/docs/en/quickstart))
   and sign in with your Claude account.
2. Install the plugin:

   ```bash
   claude plugin marketplace add Harrison-Oatman/PyCLM
   claude plugin install pyclm@pyclm-plugins
   ```

3. Make sure `python` (3.8 or later) is on the `PATH`. The plugin's
   start-of-session note needs it. On Windows, use Python from python.org
   with "Add to PATH" ticked.

PyCLM itself (1.2 or later) is installed by Claude if it is missing, or by
hand with `pip install "closed-loop-microscopy[gui]>=1.2"`.

Then start Claude Code (`claude`) and set up once per lab and once per
microscope:

- *"Set up PyCLM for our lab."* creates the **lab context folder**: the
  lab's approval rules, roles and data conventions (`lab.md`), a note and
  the calibrated `pyclm_config.toml` for each microscope, and a `proposals/`
  folder. Every session reads it. Sessions never edit it; they write
  proposals that an admin accepts or declines. The plugin sets no approval
  rules of its own; it follows `lab.md`.
- *"Set up PyCLM on this microscope."* covers the install, the Micro-Manager
  device interface, the `.cfg` presets, DMD calibration on a fluorescent
  slide, and first light ([First-time setup](first_time_setup.md) is the
  same work by hand).

To update the plugin: `claude plugin marketplace update pyclm-plugins`,
then `claude plugin update pyclm@pyclm-plugins`.

## Launching

On a PC that one person uses, start `claude` in a project folder. On a
microscope PC that several people share, the lab context's
`launcher\Start-PyCLM.bat` (a desktop shortcut) asks who you are, which
project, and which session, then starts Claude Code there:

```{figure} imgs/plugin/hierarchy.svg
:alt: Person, project and session

Each **person** has a profile with their own Claude sign-in, settings,
history and plugin. A **project** is a folder shared by everyone on the PC.
A **session** is one conversation, new or continued.
```

The start file also shows when a PyCLM run holds the microscope, offers to
update the plugin, and can offer a guest profile that is deleted on exit.
`launcher\Start-MicroManager.bat` opens Micro-Manager with that microscope's
start-up steps and refuses while a run holds the microscope. Settings for one
PC go in `launcher\local.json` (copy `local.example.json`).

## Running an experiment

```{figure} imgs/plugin/six_steps.svg
:alt: The six steps of a run

The six steps of a run, and where you come in.
```

The exchanges below follow an illustrative bar-speed experiment. The
Example at the end is a real run.

### 1. Describe

Say what you want in plain words. Claude reads the lab and microscope notes,
then asks only what it cannot know, usually in one message: the pattern, the
dose, the timing and the controls. It proposes values; you decide.

```{admonition} You
:class: chat-you
I have optoEGFR cells in wells B2-C4, and I want to compare moving bars of
light ranging from 15 to 120 µm/h. Do 3 replicates per bar speed and image
one dark position per well.
```

```{admonition} Claude
:class: chat-claude
I'm setting up a directory for today's bar experiments now.
```

### 2. Position

If a **plate map** for your vessel and a focus from a previous session are
stored, Claude asks whether to reuse the z and focus-lock settings, use the
sample as it is in focus now, or open Micro-Manager so you can focus. It then
scouts around the well without light and proposes fields on the **Positions
page**.

A plate map records where one vessel format's wells sit on the stage and where
focus is. When there is none, Claude asks whether you want to help set one up:
you focus one or two wells and say which they are. You can also skip it and
use the well in focus. Claude adds a confirmation each time a well turns up
where the map says, and relies on the map once positions have proven
consistent between sessions. Maps are kept with the microscope in the lab
context folder.

```{admonition} Claude
:class: chat-claude
Scouted the six wells with no light and picked 18 fields: 3 per bar speed
(15, 30, 60 and 120 µm/h), spread over the wells, and one dark field in each
well. The Positions page is open.
```

```{admonition} You
:class: chat-you
Approve the positions.
```

### 3. Review

Claude writes the experiment folder, runs `pyclm check`, previews the pattern
on real frames at every timepoint, and rehearses a few timepoints on the
virtual microscope. The **Design page** shows the result. What needs your
approval is set in `lab.md`.

```{admonition} Claude
:class: chat-claude
`pyclm check`: no errors, 1 warning (a 40x pixel-size preset, unused at
20x). Preview and rehearsal look right; the Design page is open. Start the
run?
```

```{admonition} You
:class: chat-you
Approve the design and start.
```

### 4. Run

Claude starts the run with `pyclm run --detached`, so it continues if the
terminal or Claude is closed, and checks that the first timepoint arrives.
One run holds the microscope at a time.

```{admonition} You
:class: chat-you
Plot KTR activity over time, live, grouped by condition.
```

```{admonition} Claude
:class: chat-claude
Started detached (pid 47572), so it keeps going if this window closes. t 1
of 30, light at the 12 bar fields only, no errors. The KTR plot, lit vs
unlit, is at the top of the Running page and updates with each timepoint.
```

### 5. Follow

`pyclm watch` checks the run every minute for lateness, errors, focus and
brightness. Flags are reported with their evidence and never stop a run on
their own. The **Running page** is rebuilt every 5 minutes and can lead with a
live analysis figure.

```{admonition} You
:class: chat-you
Send a notification to my phone if any field goes out of focus.
```

```{admonition} Claude
:class: chat-claude
Will do; I check the watcher's flags every minute. t 15 of 30, at most 5 s
late, no flags so far. KTR activity in the lit fields has been falling since
t 6; the unlit field is flat.
```

Notifications to a phone need Remote Control on (Connecting to Claude Code
projects, below), and the Claude session to stay open.

### 6. Export

When the run ends, Claude exports and packs the data and writes the session
record. The run record (`run_record.json`) and what was learned go to the
project; changes for the lab go to `proposals/`.

```{admonition} You
:class: chat-you
We're done. Export it for Fiji and copy it to the lab drive.
```

```{admonition} Claude
:class: chat-claude
Exported ImageJ hyperstacks (`pyclm export`) and a movie per field with the
pattern overlaid (`pyclm export --movies`), packed and verified the stores
on the lab drive (`pyclm pack`), and wrote the run record.
```

## Connecting to Claude Code projects

A session on the microscope PC can be followed and driven from elsewhere
with Remote Control, for example from a laptop, a phone, or a project on
claude.ai where the experiment was planned:

- **From a session already running:** type `/remote-control` (or `/rc`). The
  conversation so far carries over; open it at claude.ai/code or in the
  Claude app.
- **From a terminal on the microscope PC:** `claude remote-control --name
  <scope>` in the project folder waits for a connection from claude.ai/code
  or the Claude app.
- **From a Claude project:** in the Claude desktop app on the microscope PC,
  turn on Remote Control (Settings, Claude Code, Remote Control) and add the
  projects folder. A conversation in the project can then start a session on
  that PC, with the plugin and the lab context as they are there.

While Remote Control is on, Claude can send push notifications to the Claude
app on your phone (turn them on in `/config`).

The session runs on the microscope PC under the account that started it,
and the PC must stay on. Approvals are still typed in the conversation, and
a plan made in the project is a folder Claude checks against this microscope
before asking to run it (*"Run the plan in this folder."*).

## Example

A 30-minute moving-bar run on an MDCK monolayer, on a Nikon Ti2 with a DMD
that had run PyCLM before. The researcher had focused one field in well B7
and asked for the run
with an unlit control, the design and positions shown on pages, and no
scouting outside B7. Claude asked about the bar, cadence and scouting in one
message, scouted a 3 × 3 grid with no light, and opened two pages:

```{figure} imgs/plugin/page_design.jpg
:alt: The Design page

**Design page**: the bar on each lit field and the unlit control at t = 0,
the light program, the settings, and the `pyclm check` output.
```

```{figure} imgs/plugin/page_positions.jpg
:alt: The Positions page

**Positions page**: the scout with the picked fields outlined and numbered
in their condition's color, and why they were picked.
```

After the researcher approved both, Claude started the run detached and kept
the Running page up to date. Midway, the researcher asked for reporter
intensity against position relative to the bar, and Claude added it to the
page:

```{figure} imgs/plugin/page_running.jpg
:alt: The Running page at the end of the run

**Running page** at the end: the live figure, the newest frame of every
field, and each position's health.
```

The run finished 30 of 30 timepoints at all three positions with no errors or
dropped frames. Claude needed the researcher three times beyond the first
message: the design questions, the approval, and which PyCLM install to use.

## Without the plugin

Three of the skills (`pyclm-microscope-bringup`, `pyclm-design-experiment`,
`pyclm-write-pattern-method`) are also in
[`.claude/skills/`](https://github.com/Harrison-Oatman/PyCLM/tree/main/.claude/skills).
Start Claude Code in a clone of PyCLM to use them, or copy them into another
folder's `.claude/skills/`. Everything Claude does is a `pyclm` command you
can run yourself ([the `pyclm` command](command_line.md)).

---
name: pyclm-lab-setup
description: Set up a lab (or a core facility) to run PyCLM with Claude Code - create the lab context folder (lab file, roles, data rules, microscope notes) and connect this person to it. Use when someone says "set up PyCLM / Claude Code for optogenetics in my lab", joins a lab that already has one, or wants to change the lab's rules or roles.
---

# Set up a lab for PyCLM

The lab context is a small folder of text files that every PyCLM session
in the lab reads at its start: `lab.md` (rules, roles, data), `lab.toml`
(the same facts for scripts) and one note per microscope. Templates are in
this plugin's `templates/lab_context/`. Its `README.md` says what each file
is for; read it first.

This skill sets the folder up and connects one person to it. It does not
install PyCLM or calibrate a microscope: that is `pyclm-microscope-bringup`.
What bring-up finds (how to start PyCLM there, the calibrated config,
quirks, timings) goes into the microscope's note.

## 1. Find out which case this is

Ask one question: is there already a lab context folder (a git repository,
or a shared folder) for this lab?

- **Yes:** go to step 4 (connect this person).
- **No:** steps 2 to 4. The person doing this becomes an admin unless they
  say otherwise.
- **A core facility:** the same, but the folder belongs to the facility
  staff, users only read it, and `lab.md` becomes `facility.md` in spirit
  (keep the file name `lab.md`; say "facility" in its title). A lab using
  the core can add its own folder; where the two disagree about hardware,
  the facility's file wins.

## 2. The interview (new lab only)

Ask these, a few at a time, in plain words. Every answer can be changed
later; say so, so nobody agonizes. Offer the template's example where the
person has no opinion, and write "<not decided>" rather than inventing one.

1. Who approves what before light goes on cells? (Offer the design summary
   plus pattern preview as one example, not as a default. The lab sets its
   own rules; do not suggest permission rules for agents.)
2. Who are the admins? Which decisions are admin-only? (Start from the
   template's list.)
3. Where do data go, and how are folders named?
4. How should people hear about flags from `pyclm watch`?
5. Will the folder be a git repository, and who can push to it?
6. Which microscopes, and the computer name of each PC that runs one
   (`python -c "import platform; print(platform.node())"` on that PC)?

## 3. Create the folder

Copy `templates/lab_context/` to where the person chose. Fill `lab.md` and
`lab.toml` from the answers, dated today. Replace the template's example
`[microscopes.my-scope]` table, or delete it if no microscope is known
yet (setting up on a laptop first is fine: add the microscope later, from
its PC). For each microscope, copy
`microscopes/_template/` to `microscopes/<name>/` and add a
`[microscopes.<name>]` table with its hosts to `lab.toml`; the note itself
is filled by the owner, with what `pyclm-microscope-bringup` found. If it is a git
repository, commit, and push only if the person asks.

Show the person `lab.md` and ask them to read it before going on.

## 4. Connect this person

Write `~/.pyclm/lab_context.toml` on this computer, for this person's login:

```toml
path = "<the lab context folder>"
name = "<the person's name, as written in lab.toml>"
```

(`PYCLM_LAB_CONTEXT` and `PYCLM_PERSON` override it, for example on a
shared login.) On a microscope PC that several people use, offer the
launcher instead: a desktop shortcut to the lab context's
`launcher\Start-PyCLM.bat` gives each person their own Claude sign-in,
session history and plugin install, and sets both variables
(`templates/lab_context/README.md`, "Several people on one microscope
PC"). Windows only. Ask how Micro-Manager is started on that PC (anything
that must run or be switched on first, which `.cfg` to pick) and write it
into `launcher\local.json` under `"micromanager"`, with a desktop shortcut
to `launcher\Start-MicroManager.bat` beside the PyCLM one, so the GUI
always starts the same way (`templates/lab_context/README.md`, "Starting
Micro-Manager"). The next Claude Code session starts with a short note naming
the files to read and the person's role. Tell the person to start a new
session to check it.

## 5. A project, when one starts

When the person starts a project on a scope, offer a `pyclm_project.md`
pointer in the project's folder (template in `projects/_template/`): which
microscope, where data and run records go, and how data are stored long
term and exported. Do not create other project files; the person's own
Claude memory and their project's notes are theirs.

## Changing the rules later

An admin may edit `lab.md` and `lab.toml` directly. Anyone else writes a
proposal in `proposals/` (or a pull request) and tells an admin.

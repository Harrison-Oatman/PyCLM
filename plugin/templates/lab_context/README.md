# Lab context

The shared context for PyCLM sessions in one lab (or one core facility).
Claude Code sessions with the PyCLM plugin read `lab.md`, the note of the
microscope they run on, and the project pointer of the project they work in.
Nothing else here is loaded at the start of a session.

| Path | Holds | Changed by |
| --- | --- | --- |
| `lab.md` | approval rules, roles, admin-only decisions, data conventions, notifications | the lab admin, from proposals |
| `lab.toml` | the same facts a script needs: admins, which computer is which microscope | the lab admin |
| `microscopes/<name>/note.md` | how to start PyCLM there, data paths, quirks, measured timings, dated temporary issues | the microscope owner, from proposals |
| `microscopes/<name>/pyclm_config.toml` | the calibrated configuration in use | the microscope owner |
| `microscopes/<name>/plates/<format>.toml` | plate maps: where each vessel format's wells sit on the stage, and its focus | sessions add maps and confirmations; the owner may correct |
| `proposals/` | proposed changes from sessions, one file each | sessions add; the admin or owner reviews |
| `launcher/` | `Start-PyCLM.bat`: the start file for a microscope PC that several people share; `Start-MicroManager.bat`: the Micro-Manager GUI with this scope's start-up steps (below) | the lab admin |

Project pointers (`pyclm_project.md`) live in each project's own folder,
not here (template in `projects/_template/`).

Keep each note to about one page. Replace old lines instead of adding
history; the date on each line says when it was last confirmed. Never put
credentials, raw data or analysis code here.

## Several people on one microscope PC

Copy a shortcut to `launcher\Start-PyCLM.bat` onto the microscope PC's
desktop. It asks who you are, then which project, then whether to start a
new session, continue the last one or pick an earlier one, and starts the
Claude Code command line (not the desktop app) for that person:

- each person gets a profile (`CLAUDE_CONFIG_DIR`): their own Claude
  sign-in, settings, session history, and the PyCLM plugin, installed the
  first time;
- the session's start note names that person and their role
  (`PYCLM_PERSON`), and git commits carry their name if they gave one;
- it says when a PyCLM run holds the microscope (PyCLM 1.2 and later).
- with `"allow_guest": true` in `launcher\local.json` (an admin's choice),
  it also offers "Guest": a visitor signs in with their own Claude account
  for one session, has the role of a user, and their profile (sign-in,
  settings, session history) is deleted when Claude closes; files they wrote
  in the project folder stay.

Project folders go in `C:\Users\Public\pyclm-projects` unless
`launcher\local.json` (copied from `local.example.json`) names another
place; it can also name `claude.exe`, the profiles folder and extra folders
every session may read. A lab that wants the same Claude Code settings in
every profile (permissions, environment variables) puts a
`launcher\profile_settings.json` here; new profiles start from it. The
plugin ships none.

## Starting Micro-Manager

Put a shortcut to `launcher\Start-MicroManager.bat` next to the PyCLM one.
It starts the Micro-Manager GUI the way this microscope needs it, so
nobody has to remember the order. Steps that must come first (a
controller program, a device server, an environment variable, a wait) go
under `"micromanager"` in `launcher\local.json`. A PC with several
set-ups (one `.cfg` per camera, say) lists several entries and gets a menu.
It will not start while a PyCLM run holds the microscope unless you type
`open anyway`. Close Micro-Manager before a PyCLM run starts: the GUI and
a run cannot share the hardware.

This folder can be a git repository (recommended: changes are pull
requests) or a plain shared folder (changes come through `proposals/`).

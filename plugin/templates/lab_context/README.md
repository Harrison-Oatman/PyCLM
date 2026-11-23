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
| `proposals/` | proposed changes from sessions, one file each | sessions add; the admin or owner reviews |

Project pointers (`pyclm_project.md`) live in each project's own folder,
not here (template in `projects/_template/`).

Keep each note to about one page. Replace old lines instead of adding
history; the date on each line says when it was last confirmed. Never put
credentials, raw data or analysis code here.

This folder can be a git repository (recommended: changes are pull
requests) or a plain shared folder (changes come through `proposals/`).

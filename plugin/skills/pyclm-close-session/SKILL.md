---
name: pyclm-close-session
description: Close a PyCLM session - collect the run records, copy them to the project, and turn what this session learned into proposals for the microscope note or lab file. Use when the person says the session or experiment is over, when the last run of the day ends, or when they ask what should be written down.
---

# Close a PyCLM session

Two jobs: keep the record of what was run, and pass on what was learned to
whoever needs it next. Do both before the person leaves; it takes minutes.

## 1. Run records

Every `pyclm run` writes `run_record.json` in its directory when it ends
(PyCLM versions with the run record; older ones do not). For each run of
this session:

- Read the record and tell the person, in one line per run: outcome,
  timepoints acquired of planned, errors and watch flags, data size.
- With no `run_record.json`, build the same summary from `status.json`
  (timepoints, errors, done or not), the end of `log.log`, and
  `watch/flags_log.jsonl` if a watcher ran, and say it came from those.
- If the project has a pointer (`pyclm_project.md`) naming a place for run
  records (for example `runs/` in the project's repository), copy into
  `<that place>/<date>_<run directory name>/`:
  the experiment and schedule TOMLs, `multipoints.xml`, `run_record.json`
  (or `status.json` without one),
  and the `pyclm_config.toml` used; add `log.log` and
  `watch/flags_log.jsonl` only if the run did not finish. Never copy data,
  movies, parquet tables or exports. Commit only if the person allows it.
- With no pointer, the record stays with the data; say so.

## 2. What was learned

List what this session found out that a later session would otherwise
have to find again: a workaround, a fault, a setting that worked, a timing,
a documentation gap. For each item ask: who else would need it?

1. **A different project on this microscope:** a proposal against the
   microscope note.
2. **Anyone in the lab:** a proposal against `lab.md`.
3. **Anyone using PyCLM:** draft an issue for the PyCLM repository; show it
   to the person; file it only if they say so.
4. **Only this project:** the project's own notes or the person's memory;
   nothing to do here.

A temporary problem with the microscope (a laser not firing, a stage
error) goes on the note's "Check at session start" list: what, the date,
who saw it, when to check again. A fixed one is cleared the same way, with
the evidence.

## 3. Write the proposals

Never edit `lab.md`, `lab.toml` or a microscope note directly, unless the
person is an admin (or the note's owner) and asks you to. Otherwise:

- With push access to the lab context repository: one branch and one pull
  request per note, the evidence in its description.
- Without: one file per item in the lab context's `proposals/`, named
  `<date>_<short-topic>.md`, saying what to change, why, and the evidence
  (a run directory, a record, a log line).

Show the person the list, then tell them who reviews it (the note's owner,
or an admin from `lab.md`).

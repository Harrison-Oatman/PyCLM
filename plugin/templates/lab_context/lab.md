# Lab: <lab name>

Read in every PyCLM session in this lab, on any microscope and in any
project. Changes come from proposals (see "Changing this file"). Each line
ends with the date it was last confirmed.

## Who decides what

- Admins: <names> (also in `lab.toml`). (<date>)
- A user never decides the admin-only items below; the agent stops, says
  why in one line, names an admin, and drafts a note the user can send.
- Approval before running on cells: <for example "the researcher approves
  the design summary and pattern preview"; "none"; or the lab's own rule>.
  (<date>)
- Routine steps that need no approval: building experiment files,
  `pyclm check`, `pyclm preview`, dry runs, watching a run, exports. (<date>)

## Admin-only decisions

Edit this list for your lab. (<date>)

- Calibration: the affine, camera ROI, anything in `pyclm_config.toml` or
  the Micro-Manager `.cfg`.
- Installing or updating PyCLM, pymmcore, Micro-Manager or drivers.
- Light power, exposure or stage travel beyond the limits in the microscope
  note.
- Deleting or moving anyone else's data; editing this folder directly.

## Data

- Where data go: <root, and the folder rule, for example
  `<root>/<person>/<project>/<date>_<run>`>. (<date>)
- Raw data are never edited; removing raw data needs a confirmed backup and
  the owner's yes. (<date>)
- Long-term storage and export are decided per project, in its pointer.

## Notifications

- How a person hears about flags from `pyclm watch`: <for example a command
  for `--on-flag`, or "they watch the dashboard">. (<date>)

## Changing this file

- A session that learns something every project in the lab needs writes a
  proposal: a file in `proposals/` named `<date>_<short-topic>.md`, with
  what to change, why, and the evidence (a run directory, a log line).
  With push access, a pull request instead.
- An admin accepts or declines proposals and deletes the file.

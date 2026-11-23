# Microscope: <name> (<make and model>; PC <computer name>)

Read for every PyCLM session on this microscope, whatever the cells or the
project. No project or cell-line specifics here. About one page; each line
ends with the date it was last confirmed and who confirmed it.

Owner (reviews proposals): <name>.

## Check at session start

Temporary issues. The agent reads these out before planning. Each item:
what, the date seen, who saw it, when to check again. An item past its
check date is asked about, not dropped. Clearing one is a proposal with the
evidence.

- (none as of <date>)

## Starting PyCLM here

- PyCLM install and how to run it: <for example
  `uv run --project <path> pyclm run <dir> --detached`>. (<date>)
- Configuration in use: `pyclm_config.toml` in this folder (calibrated
  <date>, by <name>). (<date>)
- Data root for runs on this microscope, beyond the lab rule: <path>.
  (<date>)

## Defaults for users

Settings a user is not asked about; the agent uses these and says so.

- Binning: <n>. Exposure and intensity per preset: <...>. (<date>)
- Acquisition speed-ups in use (`[acquisition]`): <...>. (<date>)
- Segmentation default: <method and model>. (<date>)

## Limits

Stage and light limits are kept in Micro-Manager; note here where they are
set and their values, so an agent can explain a refused move.

- <...> (<date>)

## Quirks and measured timings

- <for example "the focus lock loses lock above 30 C; re-engage before a
  run"> (<date>, <who>)
- Measured time per position visit: <imaging, light only, move + focus>.
  (<date>)

## Never on this microscope

- <action>: <what it broke>. (<date>)

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

## Finding focus

How a sample gets in focus here. When a z and lock setting are stored
from a previous session, the agent asks whether to reuse them, use what
is in focus now, or open Micro-Manager for a person to focus.

- Focus lock: <none / Perfect Focus / Definite Focus / ...>, device name in
  Micro-Manager <...>, `position_mover` <basic / pfs>. (<date>)
- Typical z and lock offset per vessel: <for example "96-well glass
  bottom: z 4640-4750, PFS offset ~10000">. (<date>)
- Who focuses: <for example "the agent, from the plate map's z and the
  PFS offset; Sam prefers to focus by hand for new samples">. (<date>)
- Plate maps: `plates/` in this folder, one file per vessel format; see
  `plates/example-96well.toml`. (<date>)
- Starting Micro-Manager: `launcher\Start-MicroManager.bat` (its steps
  are in `launcher\local.json`, "micromanager"); which `.cfg` to pick in
  its start-up dialog: <...>. (<date>)
- Micro-Manager and PyCLM cannot both hold the hardware: close the
  Micro-Manager GUI before a run starts. (<date>)

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

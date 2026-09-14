# 01: the environment

## What must be generated live

Nothing to plate. The microscope on, all devices powered, MicroManager
closed (PyCLM needs the devices). `config_template.toml` filled in.

## Steps

```
uv sync
uv run mmcore info                # the Micro-Manager directory and pymmcore version
uv run wet prepare 01             # copies the config here and runs pyclm check
```

Read the check output: the device-interface line, `pyclm_config.toml: ok`,
the `.cfg` read with presets checked (no "presets not checked"), and the
two `camera_roi` lines. **Copy the suggested `camera_roi` into
`config_template.toml`** so every later item uses it. Then a snap:

```
uv run pyclm preview experiments/01_environment probe --snap
```

and open `preview/probe.preview/raw_545.tif` and `pattern_overlay.png`.

## Pass means

- the configuration loads with no device-interface error (if it fails,
  the message names the interface pymmcore speaks; see the pin in the
  PyCLM `pyproject.toml`);
- `pyclm check` reports zero errors;
- the snapped frame is the size of the camera ROI and the DMD overlay
  reaches every edge of it.

Record: paste the check output into the notes of `results/01_environment.md`
(`uv run wet verify 01` writes the file).

# 11: the camera ROI

## What must be generated live

- `camera_roi` in `config_template.toml` set to what item 01 suggested.
- A **fluorescent slide** (uniform), and a **`PositionList.pos` with one
  position** named `fullon-1`.

## Steps

```
uv run wet run 11
```

## Pass means

- frames have the ROI's size and the stimulation frame is lit to its
  edges (the `full_on` pattern through the DMD covers the whole ROI);
- by eye: `stim/DMD` in the viewer shows no dark border.

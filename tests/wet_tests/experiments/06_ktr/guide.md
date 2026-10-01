# 06: two segmentations of one channel (KTR clamp)

## What must be generated live

- **KTR reporter cells**, and a MicroManager channel preset named `ktr`
  in the `Channel` group (or edit `presets` in `ktr.toml` to the preset
  you image the reporter with; keep `channel` in `[pattern]` the same).
- **`PositionList.pos` with one position** named `ktr-1`.
- The two cellpose segmentations use the same model; if you have a
  nuclear model, set `model` under `[segmentation.nuclei]`.

## Steps

```
uv run wet run 06
```

## Pass means

- both label images exist (`labels/nuclei`, `labels/cells`) and the
  pattern is graded (more than two levels), so the ratio was finite for
  most cells.

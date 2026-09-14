# 16: the viewer and the export

## What must be generated live

The outputs of items 02, 04 and 12.

## Steps

```
uv run pyclm gui experiments/02_open_loop_bar
uv run pyclm gui experiments/12_grid
uv run pyclm export experiments/02_open_loop_bar
uv run pyclm export experiments/04_closed_loop_cellpose
uv run pyclm export experiments/12_grid
```

and, in Python:

```python
import pyclm.io
for d in ('02_open_loop_bar', '04_closed_loop_cellpose', '12_grid'):
    for store in pyclm.io.find_experiments(f'experiments/{d}'):
        with pyclm.io.open(store) as exp:
            print(exp.name, exp.current_t, exp.frames.num_rows, exp.pattern_at(0) is not None,
                  exp.camera_pattern_at(0) is not None, exp.grid)
```

## Pass means

- the viewer: the position axis switches positions, follow-the-run is
  off after a run, the minimap shows the positions with the field of
  view, the status line reads `done`;
- the hyperstacks open in ImageJ with the raw, label (04) and pattern
  channels;
- the readers return frames, both pattern spaces and, for 12, the grid.
  Record by hand in `results/16_viewer_export.md`.

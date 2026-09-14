# 03: HDF5 format 1

## What must be generated live

The `PositionList.pos` of item 02: copy it here.

## Steps

```
uv run wet run 03
```

Then open one `.hdf5` in the lab's existing analysis notebook.

## Pass means

- the automatic checks pass (files written, `current_t_index` 3, the
  `dmd` datasets gzip-compressed);
- the old notebook reads the frames and the DMD patterns unchanged.

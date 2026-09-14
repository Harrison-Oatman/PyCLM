# 15: restart safety

## What must be generated live

The completed outputs of item 02, left in place.

## Steps

```
uv run wet run 02
```

again, without `wet clean 02` first.

## Pass means

- the run refuses before touching the hardware: `pyclm check` lists the
  existing outputs as an error and nothing is overwritten (compare the
  store's modification time); `--force` does not bypass this either
  (the Controller raises `FileExistsError`). Then `uv run wet clean 02`
  and the run starts. Record by hand in `results/15_restart_safety.md`
  (`uv run wet verify 15` writes the skeleton).

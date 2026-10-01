# 09: the red / far-red switch

## What must be generated live

- A **far-red channel preset** named `farred` in the `Channel` group, and
  the names of the red and far-red laser devices and their intensity
  property (`red_device`, `farred_device`, `property` in `switch.toml`).
  If the microscope has no such pair, run the item anyway with two
  properties that exist (for example two Sola power settings); the
  point is the switching, not the wavelength.
- **`PositionList.pos` with one position** named `switch-1`.

## Steps

```
uv run wet run 09
```

## Pass means

- laser properties switched at least six times (`events.parquet`,
  kind `property`), the whole field lit in the camera-space pattern;
- by eye: the `1100` programme repeats its last character, so after
  t = 3 the far-red laser stays on.

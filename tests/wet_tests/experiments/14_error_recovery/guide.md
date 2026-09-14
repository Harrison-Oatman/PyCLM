# 14: a hardware error does not kill the run

## What must be generated live

The `PositionList.pos` of item 02 (copy it here).

## Steps

```
uv run wet run 14
```

During timepoint 3 or 4, provoke exactly one failure. The least risky:
from a second terminal send a property the device will reject,

```
uv run python -c "from pyclm.commands import write_command; write_command('experiments/14_error_recovery', {'command': 'set_property', 'experiment': 'bar.1', 'channel': 'stimulation', 'device': 'Sola', 'property': 'Power', 'value': 'not-a-number'})"
```

(the Manager refuses a badly typed value at the boundary and records
it; to reach the microscope's own error guard instead, briefly unplug
the DMD's USB during one stimulation and plug it back in).

## Pass means

- the run completes (`done: true`), the error is counted in
  `status.json` and appears as an `acquisition_error` (or a `refused`
  settings event, if you used the property route) in `events.parquet`;
- the timepoints after the error are normal.

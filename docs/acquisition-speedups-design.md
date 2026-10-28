# Acquisition speed-ups: design note and testing guide

Status: Harrison asked on 2026-10-28 for items 1, 2, 4 and 6 of the open
speed-up list (built from session 3 run 1's timing, `run1_timing`, and the
session 3 recommendations, item 20), "as a table in schedule or pyclm
config", to test on the scope. Implemented with the decisions below; each
option is to be kept, changed or made the default after the scope test.

## The table

```toml
# pyclm_config.toml
[acquisition]
settle = "after_move"           # default "every_frame"
camera_capture = "sequence"     # default "snap"
stimulation_timing = "shutter"  # default "camera"
reapply_camera_settings = false # default false: exposure / binning set only on change
```

## Decisions

1. **`pyclm_config.toml`, not `schedule.toml`.** These describe the rig
   (camera, shutter, stage), not the timing of an experiment; one
   schedule moves between microscopes, one config does not.
   `settle_time_seconds` and `reapply_config_groups` stay where they are
   (top level) so existing configs keep working.
2. **Opt-in, except item 6.** Settle, capture and stimulation timing change
   what the hardware does and need the scope test; their defaults are
   today's behavior. Item 6 only skips a set to the value the camera
   already reports, the same rule as #49's config-group cache, so it is on
   (`reapply_camera_settings = true` turns it off).
3. **Item 6 reads the camera, it does not remember.** `getExposure()` and
   the camera's `Binning` property are compared with the wanted value, so
   a change made by a config preset or a device property is still
   corrected. A read that fails sets the value as before.
4. **Item 4, `settle = "after_move"`.** `waitForSystem` still runs before
   every frame; only the `settle_time_seconds` sleep is limited to the
   first frame after a stage move (or a grid tile). The settle was meant
   for stage vibration; a filter or channel change is covered by
   `waitForSystem`. Saves the settle (0.2 s on the lab's rig) for every
   further frame of a visit: in run 1, an imaging visit has three frames
   (545, 638, the stimulation), two of them after the first.
5. **Item 2, `camera_capture = "sequence"`.** Each frame is a one-image
   `startSequenceAcquisition(1, 0, False)`, read with `popNextImage`;
   exposure, binning and channel can still change between frames. A
   sequence that gives no image within the exposure plus 5 s is stopped and
   raises (the microscope's error guard counts it). A camera left running
   continuously was not built: every frame changes channel or exposure.
   How much this saves depends on the camera's adapter, which is what the
   scope test measures (run 1's planned test D).
6. **Item 1, `stimulation_timing = "shutter"`, only where no frame is
   needed.** A stimulation is timed by the shutter only when its frame is
   not saved (`[stimulation] save = false`) and no consumer demands the
   stimulation channel's raw frames (`Router.demanded("raw")`; the writer's
   record-only subscription does not count). Otherwise the camera times it
   as before, and `pyclm check` says so per experiment. The shutter is
   Micro-Manager's current shutter (`getShutterDevice()`, set by the
   stimulation's config groups through `Core-Shutter`); none configured
   falls back to the camera with one warning. The light is
   `setShutterOpen(True)`, `waitForDevice`, sleep the exposure,
   `setShutterOpen(False)`, `waitForDevice`. The stimulation is published
   with `data = None`; the writers use the frame only when it is saved.
7. **The dose can change with item 1.** With the camera, the auto-shutter
   stays open for the camera's overhead as well as the exposure; with the
   shutter, the light is on for the exposure plus the shutter's own
   open/close time. The setup docs say to measure it before using the
   option for an experiment.
8. **A run ends with a summary line** from the microscope (settle mode,
   capture, timing; config groups and camera values not set again;
   stimulations by shutter; blank stimulations skipped), so a scope test
   can confirm the options were active.

## Testing

Dry (done): `tests/test_acquisition_speedups.py` covers each option against
the simulated core, the fallbacks, the schema and a dry run with every
option on. The simulated core does not model camera or shutter timing.

At the microscope (Harrison, 2026-10-28 or later): run one short schedule
four times, adding one option at a time, and compare per-visit times in
`frames.parquet` (completion stamps) and the log's `acquiring image` /
`image acquired` stamps against run 1's numbers (light visit 2.3 s,
imaging visit 6.2 s):

1. defaults (item 6 is on);
2. `settle = "after_move"`: imaging visits should shorten by about 0.2 s
   per frame after the first; check the first frame after a channel change
   is still sharp;
3. `camera_capture = "sequence"`: every frame should shorten if the
   Hamamatsu adapter starts sequences faster than it snaps; check the
   frames are complete and the exposure is right;
4. `stimulation_timing = "shutter"` with `[stimulation] save = false`:
   light visits should shorten by about 0.5 s; check the DMD shows the
   pattern while the shutter is open (the Polygon may need the camera's
   trigger), and measure the dose.

The last line of each run's log says which options were active.

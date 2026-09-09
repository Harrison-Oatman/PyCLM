# Experiment TOMLs

Each experiment (one per imaging position) is described by one TOML file in
the experiment directory. The file's stem is the experiment name that
position labels refer to: position `bar10.pos1` uses `bar10.toml`. The
timing shared by every experiment lives in `schedule.toml`.

A complete example:

```toml
# Optional timing offsets in timepoints. They must come before the first
# table: a key written after a table header belongs to that table.
t_delay = 0               # wait this many timepoints before this experiment starts
t_stop = 0                # stop after this many of its own timepoints (0 = run to the end)

# Applied to every acquisition of this experiment
[config_groups]
Objective = "1-Plan Apo LmbdD0.80 20x"

[device_properties]
"Cam-Gain" = 2            # "<device>-<property>" = value

# Defaults for the imaging channels
[imaging]
exposure = 50             # ms
every_t = 5               # acquire every 5th timepoint (counted from t_delay)
save = true
binning = 1

    [imaging.config_groups]
    Laser-Intensity = "50%"
    LightPath = "Fluor"

# Which presets of which config group are the imaging channels
[channels]
group = "Channel"
presets = ["545", "638"]

    [channels.638]         # optional per-preset overrides
    exposure = 100
    every_t = 10

# The stimulation (light delivery) event
[stimulation]
exposure = 200
every_t = 1
save = false              # keep the camera frame taken during stimulation?

    [stimulation.config_groups]
    Channel = "DMD"
    LightPath = "DMD"

# Optional: segmentation of the frames a pattern method asks for
[segmentation]
method = "cellpose"       # a built-in or registered SegmentationMethod
save_output = true        # record the label images
model = "cpsam"           # remaining keys are the method's keyword arguments

# Optional: further segmentations of the same frames, by name
[segmentation.nuclei]
method = "cellpose"       # its own method and keyword arguments
model = "nuclei"
save = true               # record these labels too (labels/nuclei)

# Optional: tracking of the segmented objects across timepoints
[tracking]
method = "centroid"       # a built-in or registered TrackingMethod
save = true               # record tracked labels and the tracks table
segmentation = "nuclei"   # which segmentation to track (default: the default table)
max_distance_um = 20      # remaining keys are the method's keyword arguments

# The pattern (light) method and its parameters
[pattern]
method = "move_out"       # a built-in or registered PatternMethod
every_t = 1               # regenerate the pattern every timepoint
channel = "545"           # remaining keys are the method's keyword arguments
```

## Sections

**`[config_groups]`, `[device_properties]`** (optional). MicroManager
configuration applied for every acquisition of this experiment: a table of
`group = preset` pairs and a table of `"device-property" = value` pairs. The
same two tables may appear inside `[imaging]`, inside a channel table and
inside `[stimulation]`, where they apply to those acquisitions only.

**`[imaging]`**. Defaults for all imaging channels: `exposure` (ms,
default 10), `every_t` (default 1), `save` (default true) and `binning`
(default 1; the same for every channel).

**`[channels]`**. `group` names the MicroManager config group that switches
channels (default `Channel`) and `presets` lists the presets to image, in
acquisition order. A table named after a preset (`[channels.638]`)
overrides `exposure`, `every_t`, `config_groups` and `device_properties`
for that channel. An experiment that only stimulates leaves `presets`
empty or omits the table: the DMD frame is then its only image (saved when
`[stimulation] save = true`), `pyclm check` says so, and `pyclm preview`
uses the stimulation frame as its probe.

**`[stimulation]`**. The light-delivery event: `exposure` (ms; `0` means no
stimulation), `every_t` (default 1), `save` (default true: whether the
camera frame taken during stimulation is stored) and `binning` (default:
the imaging binning; it also sets the pattern method's pixel grid). Its
`config_groups` normally select the DMD light path; the preset named in
the channel group (`Channel = "DMD"` above) is how the stimulation channel
appears in the plan and the outputs.

**`[segmentation]`** (optional). `method` is a built-in name (`cellpose`,
`embryo_resizing`) or one registered with `run_pyclm(segmentation_methods=...)`;
`save_output` (default true) records the label images; every other key is
passed to the method's constructor. Segmentation runs for a channel only
when the pattern method asks for it (`add_requirement(channel, seg=True)`)
or when tracking needs it; a `[segmentation]` table nobody uses is a
warning at start-up. A sub-table `[segmentation.<name>]` configures a
further segmentation of the same frames with its own method and keyword
arguments (`save`, default true, records its labels as `labels/<name>`); a
pattern method asks for it with `add_requirement(channel, seg="<name>")`,
and several can run on one channel, for instance nuclei and whole cells of
a biosensor channel (see [custom pattern methods](custom_pattern_methods.md)).
Each runs only when asked for, at the cadence it is asked for.

**`[tracking]`** (optional). `method` is a built-in name (`centroid`) or one
registered with `run_pyclm(tracking_methods=...)`; `save` (default true)
records the tracked labels and the tracks table; `segmentation` names the
segmentation table whose objects are tracked (default: the default
`[segmentation]`); every other key is passed to the method. Tracking runs
for a channel only when the pattern method
asks for tracks (`add_requirement(channel, tracks=True)`), and then
segmentation of that channel runs on every acquired frame. See
[Tracking](tracking.md).

**`[pattern]`**. `method` is a built-in name (see the [method zoo](method_zoo.md))
or one registered with `run_pyclm(pattern_methods=...)`; `every_t` (default
1) is how often the pattern is regenerated, in timepoints; every other key
is passed to the method's constructor. The pattern is actually regenerated
every least common multiple of `every_t` and the `every_t` of the channels
the method requires, so a method needing a channel imaged every 5
timepoints runs every 5.

**`t_delay`, `t_stop`** (optional, timepoints). The experiment starts at
timepoint `t_delay` and its cadences are counted from there; with `t_stop`
greater than 0 it ends after that many of its own timepoints. Write them
at the top of the file, before the first table; written after `[pattern]`
they would be keys of that table, and `pyclm check` says so.

## `schedule.toml`

```toml
[timing]
steps = 120                   # number of timepoints
interval_seconds = 60         # between timepoints
setup_time_seconds = 5        # how early events are prepared before each timepoint
time_between_positions = 10   # offset between successive experiments within a timepoint
```

The plan PyCLM derives from these files is written to `plan.useq.yaml` in
the experiment directory and embedded in every output (see
[Data format](data_format.md)).

## Reference

Generated from the schema (`pyclm.schema`) at build time, so it matches the
code. Unknown keys are errors everywhere except in the three method tables,
whose other keys are the method's arguments; `pyclm check` compares those
with the method's constructor.

### The experiment file

```{pyclm-schema} ExperimentConfig
```

`[imaging]`:

```{pyclm-schema} ImagingDefaults
```

`[channels]`:

```{pyclm-schema} Channels
```

`[channels.<preset>]`:

```{pyclm-schema} ChannelOverride
```

`[stimulation]`:

```{pyclm-schema} Stimulation
```

`[segmentation]` and `[segmentation.<name>]`:

```{pyclm-schema} SegmentationTable
```

`[tracking]`:

```{pyclm-schema} TrackingTable
```

`[pattern]`:

```{pyclm-schema} PatternTable
```

### `schedule.toml`

```{pyclm-schema} ScheduleConfig
```

`[timing]`:

```{pyclm-schema} Timing
```

### `pyclm_config.toml`

```{pyclm-schema} PyclmConfig
```

`[output]`:

```{pyclm-schema} OutputConfig
```

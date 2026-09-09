# The configuration schema, in between the summary and the code

*A reader's guide to `src/pyclm/schema.py` (Stage 5). Prepared 2026-09-08.*

## 1. What the schema is

Every PyCLM configuration file is a TOML document: a tree of tables holding
keys. The schema is a matching tree of small Python classes (pydantic
*models*), one per table, each listing the keys the table may hold, their
types, defaults, limits and one-line meanings. Loading a file is now three
steps rather than one:

1. **Read** the TOML into a plain dictionary (as before).
2. **Validate** the dictionary against the model tree. Every key is checked
   for presence, type and range, unknown keys are flagged, and a few rules
   that span keys are applied. Every problem in the file is collected.
3. **Build** the same runtime objects the old parser built (`Experiment`,
   the timing arguments of `ExperimentSchedule`, the config values
   `run_pyclm` needs) from the validated model.

Nothing after step 3 changed. The Manager, the plan, the writer and the
methods receive exactly the objects they always did.

## 2. The three trees

```
ExperimentConfig  (bar10.toml)
├── format_version, t_delay, t_stop           top-level keys
├── config_groups   { group = "preset" }      tables of plain pairs
├── device_properties { "Dev-Prop" = value }
├── imaging: ImagingDefaults                  exposure, every_t, save, binning, + the two tables
├── channels: Channels                        group, presets,
│   └── overrides: { preset -> ChannelOverride }   exposure, every_t, + the two tables
├── stimulation: Stimulation                  exposure (required), every_t, save, binning, + tables
├── segmentation: SegmentationTable | none    method, save, *kwargs
├── segmentations: { name -> SegmentationTable }   the [segmentation.<name>] tables
├── tracking: TrackingTable | none            method, save, segmentation, *kwargs
└── pattern: PatternTable                     method, save, every_t, *kwargs

ScheduleConfig  (schedule.toml)
└── timing: Timing                            steps, interval_seconds (both required),
                                              setup_time_seconds, time_between_positions

PyclmConfig  (pyclm_config.toml)
├── config_path, affine_transform, slm_shape_h, slm_shape_w (required)
├── focus_device, settle_time_seconds
└── output: OutputConfig                      format, pattern_policy, export_imagej
```

Two shapes need a word:

- **Nested tables that are keyed by user-chosen names.** `[channels.638]`
  and `[segmentation.nuclei]` have names the schema cannot know in
  advance. A small step before validation moves them out of their parent
  table into a dictionary (`overrides`, `segmentations`), so the parent
  keeps a fixed set of keys and each named entry is validated by the same
  model as its siblings.
- **Method tables are open.** `[segmentation]`, `[tracking]` and `[pattern]`
  are the only tables that accept keys the schema does not list, because
  those keys are the method's own arguments (`duty_cycle`, `max_distance_um`,
  `model`). The model names its own keys (`method`, `save`, `every_t`,
  `segmentation`) and exposes the rest as `kwargs`. The schema cannot judge
  them; `pyclm check` does, against the method's constructor.

## 3. What gets checked, in order

1. **Structure.** Required keys present; no unknown keys outside the method
   tables. An unknown key is compared with the allowed ones and reported
   with a suggestion: `unknown key 'exposur' in [imaging] (did you mean
   'exposure'?); allowed: binning, config_groups, device_properties,
   every_t, exposure, save`.
2. **Types and ranges**, declared next to each key: `exposure` is a number
   greater than 0, `every_t` an integer of at least 1, `format` one of
   `"ome-zarr"` or `"hdf5"`, `affine_transform` a 2 × 3 matrix. A violation
   reads `[imaging] exposure: must be greater than 0 (got -5)`.
3. **Table rules**, written as small functions on the model: a device
   property key must be `"Device-Property"`; `binning` is refused inside a
   channel table with a sentence saying it is set once under `[imaging]`;
   `save_output` is accepted as a spelling of `save`; `t_delay` or `t_stop`
   inside a method table is refused with the explanation that a key after
   a table header belongs to that table.
4. **Cross-table rules**: every `[channels.<x>]` names a listed preset,
   presets are unique, `[tracking].segmentation` names a segmentation table
   that exists, `format_version` is not newer than the code.

Pydantic collects all violations in one pass; PyCLM rewrites each into one
line in the file's vocabulary (`[channels.638]`, not the internal
`channels.overrides.638`) and raises a single `ConfigError` carrying the
list. The old parser stopped at the first `KeyError` and said nothing
about the rest.

## 4. What the schema does not do

It does not look at the microscope, the methods or the other files.
Whether the preset `Fluor` exists, whether `bar_sped` is an argument of the
bar method, whether every position has a TOML: those need the registry of
methods, the `.cfg` file and the position list, and live in `pyclm check`,
which calls the schema first and then does the rest.

## 5. What this buys, compared with the hand-written parser

- **Mistakes are caught where they are made, not where they hurt.** A
  misspelled key used to be ignored (or silently handed to a method that
  swallowed it); a missing key was a `KeyError` at start-up naming a
  Python line, not a file. Now the message names the file, the table and
  the key, and offers the likely spelling. Two real bugs surfaced the day
  the schema went in: per-channel overrides that were never applied, and
  `t_delay` written where the documentation put it, inside `[pattern]`.
- **One place holds the truth.** Defaults, limits and meanings are declared
  once, next to the key, and used for validation, for building the runtime
  objects, for the generated reference tables in the documentation (a
  Sphinx directive reads the models at build time, so the reference cannot
  drift), and later for forms in the interactive setup.
- **Files are self-describing and versionable.** `format_version` lets a
  future change to a table be migrated rather than guessed at.
- **The rest of the code got simpler and typed.** `run_pyclm` reads
  `config.output.format` instead of `config.get("output", {}).get("format",
  ...)`, with the default applied once, in the schema, rather than in each
  caller with a chance to differ.
- **Required means required.** `steps` and `interval_seconds` used to
  default silently to 10 and 10; a misspelled `intervl_seconds` would have
  run a 100-second experiment. Both are now required, and a misspelling is
  reported.

## 6. What it costs

- Strictness is a behaviour change: a file with a stray or misspelled key
  that used to run now stops at `pyclm check` until fixed. That is the
  point, but old files may need a minute of attention.
- The message rewriting in `describe_errors` is a layer between pydantic's
  wording and PyCLM's; a new kind of validation error may need a line
  there to read well.
- Method arguments stay unchecked by the schema itself, by design; the
  signature check in `pyclm check` is the complement, and a method whose
  constructor takes only `**kwargs` gives it nothing to check against.

# Multi-channel segmentation and Cellpose auto-normalization: design note

Status: requested by Harrison on 2026-10-28 ("segmentation methods can get
multiple channels from the same position (e.g. 545 and 638) ... pass in a
color image to cellpose"; "the cellpose segmentation methods should be able
to run with auto-normalization"). Implemented on the branch after PR #49
with every decision as recommended below; the decisions are open to change
in review.

## What a user writes

```toml
[channels]
group = "Channel"
presets = ["545", "638"]

[segmentation]
method = "cellpose"
channels = ["545", "638"]
auto_normalize = true          # or normlow = [100, 200], normhigh = [3000, 8000]

[pattern]
method = "all_cells"
channel = "545"                # the masks belong to 545
```

## Decisions

1. **`channels` is a key of the `[segmentation]` table, not a method
   argument.** The router has to know which frames to deliver before any
   method is built, and every method gets the feature without parsing
   anything. *Alternative:* a Cellpose kwarg; it would hide routing in a
   method's arguments and need repeating in every custom method.
2. **The method gets one channels-last `(H, W, C)` array, in the listed
   order.** That is what Cellpose's `convert_image(..., channel_axis=-1)`
   and most color-image models take. A table without `channels` still
   hands over the 2D frame, so existing methods are unchanged.
3. **The masks belong to the channel a consumer names**
   (`add_requirement("545", seg=True)`), which must be one of the listed
   channels; `pyclm check` says so otherwise. The label image carries that
   channel's event, so storage, the dock and tracking need no change.
4. **All listed channels are imaging channels at one `every_t`** (a schema
   error otherwise); binning is shared by the experiment already. Frames of
   different shapes are an error naming them. A timepoint whose frames never
   all arrive (a channel errored) is dropped with a warning when a later
   timepoint comes in, so the buffer cannot grow.
5. **A method says how many channels it takes with a class attribute,
   `max_channels`** (default 1, Cellpose 3, the channels Cellpose pads to).
   `pyclm check` refuses a table listing more.
6. **Cellpose `auto_normalize = true` scales each frame by its own
   percentiles** (`percentile_low` 1, `percentile_high` 99; Cellpose's
   own `normalize: True, percentile: [...]`, applied per channel). Off by
   default so existing experiments keep their fixed `normlow`/`normhigh`,
   which may now be lists, one per listed channel.
7. **`pyclm preview --image 545=a.tif --image 638=b.tif`** gives each
   channel its own TIF; a single `--image` still serves every channel.
8. **Recording never widens production (router).** The writer's
   record-only subscriptions are added only for keys a demanding consumer
   already receives, or that a source makes (a producer with no inputs,
   like the microscope's `skipped` records). Before, "produced" was inferred from the producer's
   inputs being routed, which a multi-channel segmentation breaks: 638's
   raw frames reach the segmentation process for 545's masks, and that
   must not make it segment 638 as well.

## Not done

- Per-channel exposure or presets are unchanged: each listed channel is
  acquired as configured; the gathering only waits for both frames.
- No multi-channel input for tracking or pattern methods: they read
  several channels already through `add_requirement`.

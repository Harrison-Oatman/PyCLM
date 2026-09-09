"""
``pyclm new``: an experiment directory from a template, ready for
``pyclm check``. Two templates, taken from the lab's experiments: an open
loop (a moving bar, no segmentation) and a closed loop (Cellpose
segmentation and a per-cell pattern). Nothing is overwritten.
"""

from __future__ import annotations

from pathlib import Path

SCHEDULE = """# Timing shared by every experiment in this directory
[timing]
steps = 60                    # number of timepoints
interval_seconds = 60         # seconds between timepoints
setup_time_seconds = 3.0      # seconds before each timepoint at which its events are prepared
time_between_positions = 3.0  # seconds between successive positions within a timepoint
"""

CONFIG = """# Hardware and output configuration (edit config_path and the DMD calibration)
config_path = "C:\\\\Program Files\\\\Micro-Manager-2.0\\\\MMConfig.cfg"

# camera -> DMD affine [[a, b, tx], [c, d, ty]] from the MicroManager Projector calibration
affine_transform = [[-0.289, -0.001, 938.103], [0.003, -0.58, 1500.888]]
slm_shape_h = 1140
slm_shape_w = 912

focus_device = "ZDrive"       # MicroManager focus (Z) device
settle_time_seconds = 1.0     # wait after the hardware reports ready, before each snap

[output]
format = "ome-zarr"           # or "hdf5"
pattern_policy = "on_change"  # DMD patterns stored: on_change | all | none
export_imagej = true          # ImageJ hyperstacks when the run ends
"""

OPEN_LOOP = """# {name}.toml: an open-loop experiment (a bar of light moving across the field)
# Positions named "{name}.<anything>" in the position list use this file.
format_version = 1

[config_groups]               # applied to every acquisition
Objective = "20x"

[imaging]                     # defaults for the imaging channels
exposure = 200                # ms
every_t = 5                   # image every 5th timepoint
save = true
binning = 1

    [imaging.config_groups]
    LightPath = "Fluor"

[channels]                    # which presets of which config group to image
group = "Channel"
presets = ["545"]

[stimulation]                 # the light delivery (DMD) event
exposure = 500
every_t = 1
save = false                  # keep the camera frame taken during stimulation?

    [stimulation.config_groups]
    Channel = "DMD"
    LightPath = "DMD"

[pattern]
method = "bar"                # see the method zoo in the documentation
duty_cycle = 0.2
bar_speed = 1.0               # um per minute
period = 100                  # um
"""

CLOSED_LOOP = """# {name}.toml: a closed-loop experiment (segment cells, light the half of each cell facing out)
# Positions named "{name}.<anything>" in the position list use this file.
format_version = 1

[config_groups]               # applied to every acquisition
Objective = "20x"

[imaging]                     # defaults for the imaging channels
exposure = 200                # ms
every_t = 5                   # image every 5th timepoint
save = true
binning = 1

    [imaging.config_groups]
    LightPath = "Fluor"

[channels]                    # which presets of which config group to image
group = "Channel"
presets = ["545"]

[stimulation]                 # the light delivery (DMD) event
exposure = 500
every_t = 1
save = false

    [stimulation.config_groups]
    Channel = "DMD"
    LightPath = "DMD"

[segmentation]                # runs on the frames the pattern method asks for
method = "cellpose"
model = "cpsam"

# [tracking]                  # uncomment to give cells lasting identities
# method = "centroid"
# max_distance_um = 20

[pattern]
method = "move_out"           # per-cell: light the outward-facing half of each cell
channel = "545"
# tracks = true               # run the per-cell loop on tracked cells (needs [tracking])
"""

README = """This directory was created by `pyclm new`.

  {name}.toml        the experiment: channels, stimulation, segmentation, pattern
  schedule.toml      timepoints and interval
  pyclm_config.toml  the microscope: MicroManager .cfg path, DMD calibration, output format

Next steps
  1. Export a position list from MicroManager as PositionList.pos into this
     directory. Name each position "{name}.<something>" so it uses {name}.toml.
  2. Edit the three TOML files (presets, exposures, the method and its arguments).
  3. pyclm check <this directory>      reports anything that is wrong
  4. pyclm preview <this directory> {name} --image some_frame.tif
                                       shows what the pattern method would do
  5. pyclm run <this directory>        runs it (add --dry to rehearse on the virtual microscope)
"""

TEMPLATES = {
    "open-loop": OPEN_LOOP,
    "closed-loop": CLOSED_LOOP,
}


def create(
    directory: Path, template: str = "closed-loop", name: str = "experiment"
) -> list[Path]:
    """Write the template files into ``directory`` (created if needed); existing files are kept."""
    if template not in TEMPLATES:
        raise ValueError(
            f"unknown template {template!r}; choose from {sorted(TEMPLATES)}"
        )
    if not name or "." in name or "/" in name or "\\" in name:
        raise ValueError(
            "the experiment name must be a plain word (no dots or slashes)"
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        f"{name}.toml": TEMPLATES[template].format(name=name),
        "schedule.toml": SCHEDULE,
        "pyclm_config.toml": CONFIG,
        "README.txt": README.format(name=name),
    }
    written = []
    for filename, text in files.items():
        path = directory / filename
        if path.exists():
            continue
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written

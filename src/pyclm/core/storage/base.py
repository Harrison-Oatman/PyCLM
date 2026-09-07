"""
Frame writers and the cadence-group model shared by writers and readers.

A *cadence group* is the set of channels of one experiment that are acquired
on the same ``every_t``. Each group gets its own image with a compact time
axis (one slot per acquisition of that group), so readers never see blank
frames for timepoints a channel was not scheduled at. In the common
configuration (all imaging channels on one cadence, stimulation frame not
saved) an experiment has a single group called ``imaging``.

See docs/stage2-storage-options.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core_interface import MicroscopeCoreInterface
from ..datatypes import AcquisitionData, SegmentationData
from ..plan import AcquisitionPlan

STORAGE_FORMATS = ("hdf5", "ome-zarr")
PATTERN_POLICIES = ("on_change", "all", "none")


@dataclass(frozen=True)
class CadenceGroup:
    """Channels of one experiment that share a cadence, and their compact time axis."""

    name: str
    experiment: str
    every_t: int
    t_delay: int
    t_stop: int
    channels: tuple[str, ...]
    stim_channel: str | None  # the stimulation channel's name if it is in this group
    timepoints: int  # slots on the compact time axis
    binning: int
    plan_timepoints: int

    def local_index(self, t: int) -> int | None:
        """Slot on the compact axis for plan timepoint ``t``, or None if not scheduled."""
        this_t = t - self.t_delay
        if this_t < 0 or this_t % self.every_t != 0:
            return None
        if self.t_stop > 0 and this_t >= self.t_stop:
            return None
        return this_t // self.every_t

    def global_t(self, i: int) -> int:
        return self.t_delay + i * self.every_t

    def channel_index(self, channel: str) -> int:
        return self.channels.index(channel)


def cadence_groups(plan: AcquisitionPlan, experiment: str) -> list[CadenceGroup]:
    """
    Group the saved channels of ``experiment`` by ``every_t``.

    The stimulation camera frame is included only when ``stimulation.save`` is
    set; it joins the group of its cadence (its own group if no imaging
    channel shares it). Naming: the group holding the first imaging channel
    is ``imaging``; a group holding only the stimulation frame is ``stim``;
    any other is ``imaging_every<N>``.
    """
    exp = plan.schedule.experiments[experiment]
    stim_name = plan.stim_channel(experiment)
    meta_t_delay = int(exp.t_delay)
    meta_t_stop = int(exp.t_stop)

    by_cadence: dict[int, list[str]] = {}
    for channel in plan.channels(experiment):
        is_stim = channel == stim_name
        if is_stim and not exp.stimulation.save:
            continue
        by_cadence.setdefault(plan.every_t(experiment, channel), []).append(channel)

    imaging_channels = [c for c in plan.channels(experiment) if c != stim_name]
    first_imaging = imaging_channels[0] if imaging_channels else None

    groups = []
    # the group holding the first imaging channel comes first, then by cadence
    ordered = sorted(
        by_cadence.items(), key=lambda kv: (first_imaging not in kv[1], kv[0])
    )
    for every_t, channels in ordered:
        # stimulation frame last within its group
        channels = [c for c in channels if c != stim_name] + [
            c for c in channels if c == stim_name
        ]
        if first_imaging in channels:
            name = "imaging"
        elif channels == [stim_name]:
            name = "stim"
        else:
            name = f"imaging_every{every_t}"

        binnings = {plan.imaging_config(experiment, c).binning for c in channels}
        if len(binnings) != 1:
            raise ValueError(
                f"{experiment}: channels {channels} share cadence {every_t} but "
                f"have different binning {sorted(binnings)}"
            )

        timepoints = sum(
            1
            for t in range(plan.timepoints)
            if plan.is_scheduled(experiment, channels[0], t)
        )
        groups.append(
            CadenceGroup(
                name=name,
                experiment=experiment,
                every_t=int(every_t),
                t_delay=meta_t_delay,
                t_stop=meta_t_stop,
                channels=tuple(channels),
                stim_channel=stim_name if stim_name in channels else None,
                timepoints=timepoints,
                binning=int(binnings.pop()),
                plan_timepoints=plan.timepoints,
            )
        )
    return groups


def image_shape(core: MicroscopeCoreInterface, binning: int = 1) -> tuple[int, int]:
    """(height, width) of a frame at the given binning, from the camera ROI."""
    roi = core.getROI()
    h, w = roi[3], roi[2]
    return (h // binning, w // binning)


class FrameWriter(ABC):
    """
    Persists what the pipeline produces for one run.

    ``open`` prepares every output for the plan (readers may open the outputs
    as soon as it returns); ``write_frame`` receives raw and stimulation
    frames, ``write_labels`` segmentation masks; ``close`` finalises. Writers
    must never block on readers.
    """

    format: str = "base"

    def __init__(self):
        self.base_path: Path = Path.cwd()
        self.plan: AcquisitionPlan | None = None
        self.is_open = False

    @abstractmethod
    def open(
        self,
        plan: AcquisitionPlan,
        core: MicroscopeCoreInterface,
        base_path: Path,
        affine_transform: np.ndarray | None = None,
        slm_shape: tuple[int, int] | None = None,
    ) -> list[tuple[str, str]]:
        """Create the outputs. Returns ``(path, layer)`` pairs for the live GUI."""

    @abstractmethod
    def write_frame(self, data: AcquisitionData) -> None: ...

    @abstractmethod
    def write_labels(self, data: SegmentationData) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def output_paths(self) -> dict[str, Path]:
        """Experiment name → the file or directory holding its data."""

    @abstractmethod
    def planned_paths(self, names, base_path: Path) -> dict[str, Path]:
        """Experiment name → output path that ``open`` would create."""

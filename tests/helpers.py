"""
Shared builders for unit tests: minimal experiments, schedules, and a fake
image source for the simulated microscope core.
"""

from queue import Empty

import numpy as np

from pyclm.core.experiments import (
    ConfigGroup,
    Experiment,
    ExperimentSchedule,
    ImagingConfig,
    MicroscopePosition,
    PatternConfig,
    SegmentationConfig,
)
from pyclm.core.plan import AcquisitionPlan


class FakeImageSource:
    """Minimal stand-in for TimeSeriesImageSource: constant zero frames."""

    def __init__(self, shape=(16, 16)):
        self._shape = tuple(shape)
        self.snaps = 0

    @property
    def shape(self):
        return self._shape

    def next_frame(self, pos):
        self.snaps += 1
        return np.zeros(self._shape, dtype=np.uint16)


def make_experiment(
    name,
    *,
    channel="545",
    every_t=1,
    exposure=10,
    binning=1,
    stim_exposure=10,
    stim_every_t=1,
    stim_binning=None,
    pattern_method="full_on",
    pattern_kwargs=None,
    segmentation_method="none",
    t_delay=0,
    t_stop=0,
) -> Experiment:
    channel_cfg = ImagingConfig(
        name,
        exposure_ms=exposure,
        every_t=every_t,
        binning=binning,
        config_groups=[ConfigGroup("Channel", channel)],
    )
    stim_cfg = ImagingConfig(
        name,
        exposure_ms=stim_exposure,
        every_t=stim_every_t,
        binning=binning if stim_binning is None else stim_binning,
        config_groups=[ConfigGroup("Channel", "DMD")],
    )
    return Experiment(
        name,
        {channel: channel_cfg},
        stim_cfg,
        SegmentationConfig(segmentation_method),
        PatternConfig(pattern_method, **(pattern_kwargs or {})),
        t_delay=t_delay,
        t_stop=t_stop,
    )


def make_schedule(
    experiments, *, steps=3, interval=0.02, setup=0.0, between=0.0
) -> ExperimentSchedule:
    positions = {
        exp.experiment_name: MicroscopePosition(
            x=float(i * 100), y=0.0, z=0.0, label=exp.experiment_name
        )
        for i, exp in enumerate(experiments)
    }
    return ExperimentSchedule(
        {exp.experiment_name: exp for exp in experiments},
        positions,
        t_count=steps,
        t_interval=interval,
        t_setup=setup,
        t_between=between,
    )


def make_plan(schedule, requirements=None) -> AcquisitionPlan:
    return AcquisitionPlan.from_schedule(schedule, requirements)


def drain(queue) -> list:
    """Return every item currently in a queue."""
    items = []
    while True:
        try:
            items.append(queue.get_nowait())
        except Empty:
            return items

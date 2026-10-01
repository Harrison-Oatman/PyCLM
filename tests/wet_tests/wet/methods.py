"""
The custom methods the wet tests register with ``run_pyclm``. The three
documented examples are copied from the PyCLM documentation
(``documentation/examples``) so this directory stands alone.
"""

import numpy as np

from pyclm import PatternMethod

from .examples.farred_patterns import RedFarRedSwitch
from .examples.ktr_patterns import KTRClamp
from .examples.tracking_patterns import IntensityProgram, LeaderCells


class ExposureRamp(PatternMethod):
    """Item 07: raises the stimulation exposure and a laser property every timepoint."""

    name = "exposure_ramp"

    def __init__(self, device="Sola", prop="Power", start=10.0, step=5.0, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.prop = prop
        self.start = float(start)
        self.step = float(step)

    def generate(self, context):
        context.set_exposure("stimulation", self.start + self.step * context.t)
        context.set_property("stimulation", self.device, self.prop, 10 + 5 * context.t)
        return np.ones(self.pattern_shape, np.float32)


class OddOff(PatternMethod):
    """
    Item 10: full field at even timepoints, nothing at odd ones.

    An open-loop pattern generated at ``t`` reaches the DMD at ``t + 1``
    (the SLM handshake at ``t`` answers with what the buffer already holds),
    so the method decides for the next timepoint, as the red / far-red
    example does for its settings.
    """

    name = "odd_off"

    def generate(self, context):
        value = 0.0 if (context.t + 1) % 2 else 1.0
        return np.full(self.pattern_shape, value, np.float32)


PATTERN_METHODS = {
    ExposureRamp.name: ExposureRamp,
    OddOff.name: OddOff,
    RedFarRedSwitch.name: RedFarRedSwitch,
    KTRClamp.name: KTRClamp,
    LeaderCells.name: LeaderCells,
    IntensityProgram.name: IntensityProgram,
}

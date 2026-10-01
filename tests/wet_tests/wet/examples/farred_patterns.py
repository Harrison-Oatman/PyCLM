"""
A pattern method that changes an acquisition setting of its own experiment
while the run is in progress, included verbatim in the user documentation
(documentation/custom_pattern_methods.md) and exercised by
tests/test_doc_examples.py.

Register it with::

    run_pyclm(experiment_dir, pattern_methods={"red_farred_switch": RedFarRedSwitch})
"""

import numpy as np

from pyclm import PatternMethod


class RedFarRedSwitch(PatternMethod):
    """
    A red / far-red optogenetic switch driven by a programme string: one
    character per timepoint, ``1`` = red light on (the tool switches on),
    ``0`` = far-red light on (the tool switches off). The last character
    holds for the rest of the run.

    Red light goes through the DMD (the stimulation channel) and far-red
    through an ordinary channel of the experiment; both lasers are turned
    on and off with a device property. The pattern itself is the whole
    field, so the programme alone decides what the cells receive.
    """

    name = "red_farred_switch"

    def __init__(
        self,
        program="11110000",
        farred_channel="farred",
        red_device="LaserRed",
        farred_device="LaserFarRed",
        property="Intensity",
        on=100.0,
        off=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not program or set(program) - {"0", "1"}:
            raise ValueError("program must be a non-empty string of 0s and 1s")
        self.program = program
        self.farred_channel = farred_channel
        self.red_device = red_device
        self.farred_device = farred_device
        self.property = property
        self.on = on
        self.off = off
        # no image data needed: this is an open-loop programme

    def generate(self, context) -> np.ndarray:
        # settings take effect from the next timepoint, so read the programme there
        t_next = context.t + 1
        red_on = self.program[min(t_next, len(self.program) - 1)] == "1"

        context.set_property(
            "stimulation",
            self.red_device,
            self.property,
            self.on if red_on else self.off,
        )
        context.set_property(
            self.farred_channel,
            self.farred_device,
            self.property,
            self.off if red_on else self.on,
        )

        # the whole field; the lasers decide which colour the cells get
        return np.ones(self.pattern_shape, np.float32)

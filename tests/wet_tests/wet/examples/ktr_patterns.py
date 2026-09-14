"""
A pattern method on two segmentations of one channel, included verbatim in
the user documentation (documentation/custom_pattern_methods.md) and
exercised by tests/test_doc_examples.py.

Register it with::

    run_pyclm(experiment_dir, pattern_methods={"ktr_clamp": KTRClamp})
"""

import numpy as np

from pyclm import PatternMethod, nuclear_cytosolic_ratio


class KTRClamp(PatternMethod):
    """
    Hold every cell's kinase translocation reporter (KTR) readout, the
    nuclear over cytosolic intensity of the biosensor channel, at a target.

    The biosensor channel is segmented twice, into nuclei and into whole
    cells (``[segmentation.nuclei]`` and ``[segmentation.cells]``). Light
    activates the pathway, which moves the reporter out of the nucleus and
    lowers the ratio, so a cell above its target receives more light and a
    cell below it less. The controller is proportional around a 50 % duty.
    """

    name = "ktr_clamp"

    def __init__(self, channel="ktr", target=1.0, gain=2.0, **kwargs):
        super().__init__(**kwargs)
        self.channel = channel
        self.target = float(target)
        self.gain = float(gain)
        # the raw biosensor frame plus its two segmentations
        self.add_requirement(channel, raw=True, seg=["nuclei", "cells"])

    def generate(self, context) -> np.ndarray:
        nuclei = context.regions(self.channel, "nuclei")
        cells = context.regions(self.channel, "cells")
        readout = nuclear_cytosolic_ratio(nuclei, cells, context.raw(self.channel))

        # one duty per nucleus; nuclei outside any cell have a NaN ratio and get none
        duty = 0.5 + self.gain * (readout.ratio - self.target) / self.target
        duty = np.clip(np.nan_to_num(duty, nan=0.0), 0.0, 1.0)

        # light the whole cell that owns each nucleus
        return cells.paint(duty, ids=readout.cell_ids)

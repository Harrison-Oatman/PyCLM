Intensity-control patterns (``cell_intensity_patterns``)
========================================================

Methods that read a per-cell intensity from a reporter channel and make a
binary or graded illumination decision per cell.

.. autoclass:: pyclm.core.patterns.cell_intensity_patterns.NucleusControlMethod
   :members: generate, process_prop
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.cell_intensity_patterns.BinaryNucleusClampModel
   :members: process_prop
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.cell_intensity_patterns.GlobalCycleModel
   :members: generate
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.cell_intensity_patterns.CenteredImageModel
   :members: generate
   :show-inheritance:

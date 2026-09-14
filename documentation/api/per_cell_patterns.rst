Per-cell feedback patterns (``fbc_cell_movement``)
==================================================

These methods use a segmentation to direct light to a sub-cellular region
of every cell, chosen by the cell's position in the field. All take a
``channel`` argument naming the segmented imaging channel.

.. autoclass:: pyclm.core.patterns.fbc_cell_movement.PerCellPatternMethod
   :members: generate, process_prop
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.fbc_cell_movement.RotateCcwModel
   :members: process_prop
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.fbc_cell_movement.MoveOutModel
   :members: process_prop
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.fbc_cell_movement.MoveInModel
   :members: process_prop
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.fbc_cell_movement.MoveDownModel
   :members: process_prop
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.fbc_cell_movement.BounceModel
   :members: generate, process_prop
   :show-inheritance:

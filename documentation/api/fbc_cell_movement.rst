Feedback Control Patterns
=========================

These patterns use per-cell segmentation data to direct light to specific
sub-cellular regions based on each cell's position within the field.
All require a ``channel`` argument naming a segmented imaging channel.

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


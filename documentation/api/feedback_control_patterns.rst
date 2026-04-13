Feedback Control Patterns
=========================

These patterns use per-cell segmentation data to direct light to specific
sub-cellular regions based on each cell's position within the field.
All require a ``channel`` argument naming a segmented imaging channel.

.. autoclass:: pyclm.core.patterns.feedback_control_patterns.PerCellPatternMethod
   :members: generate, process_prop
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.feedback_control_patterns.RotateCcwModel
   :members: process_prop
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.feedback_control_patterns.MoveOutModel
   :members: process_prop
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.feedback_control_patterns.MoveInModel
   :members: process_prop
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.feedback_control_patterns.MoveDownModel
   :members: process_prop
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.feedback_control_patterns.BounceModel
   :members: generate, process_prop
   :show-inheritance:


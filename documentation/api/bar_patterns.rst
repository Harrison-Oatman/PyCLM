Bar patterns (``bar_patterns``)
===============================

Open-loop bars: still, moving, bouncing, sawtooth, and rotating spokes. ``bar`` in a
TOML selects :class:`BarPatternBase`, which dispatches on its arguments.

.. autoclass:: pyclm.core.patterns.bar_patterns.BarPatternBase
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.bar_patterns.StationaryBarPattern
   :members: generate
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.bar_patterns.BarPattern
   :members: generate
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.bar_patterns.BouncingBarPattern
   :members: generate
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.bar_patterns.SawToothMethod
   :members: generate
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.bar_patterns.SpokesPattern
   :members: generate
   :show-inheritance:

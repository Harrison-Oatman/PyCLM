Pulse trains (``pulse_patterns``)
=================================

Open-loop stimulation that varies in time but never moves: a uniform field
gated on a period (``global_pulse``) or on an explicit list of intervals
(``pulse_schedule``), and the stationary bar's stripes under the same
periodic gate (``blinking_bar``).

.. autoclass:: pyclm.core.patterns.pulse_patterns.GlobalPulsePattern
   :members: generate, is_on, effective_phase_min
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.pulse_patterns.PulseSchedulePattern
   :members: generate, is_on
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.pulse_patterns.BlinkingBarPattern
   :members: generate, is_on, spatial_mask
   :show-inheritance:

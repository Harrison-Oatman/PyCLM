Pattern Method API Reference
============================

Each pattern method is a subclass of :class:`~pyclm.core.patterns.pattern.PatternMethod`.
Implement :meth:`~pyclm.core.patterns.pattern.PatternMethod.generate` to return a
float32 array in ``[0, 1]`` with shape equal to the camera ROI.

.. toctree::
   :maxdepth: 1

   pattern_base
   bar_patterns
   wave_patterns
   static_patterns
   feedback_control_patterns
   ktr_patterns

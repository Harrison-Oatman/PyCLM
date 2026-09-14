Pattern methods
===============

Subclass :class:`~pyclm.core.patterns.pattern.PatternMethod`, declare what
the method needs with ``add_requirement``, and return from ``generate`` a
float array in ``[0, 1]`` with shape ``self.pattern_shape`` (the camera
ROI at the stimulation binning; for a grid position, the stitched frame).
See :doc:`../custom_pattern_methods` for the walkthrough. Both classes
are importable from ``pyclm``.

.. autoclass:: pyclm.core.patterns.pattern.PatternMethod
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.pattern.PatternContext
   :members:

.. autoclass:: pyclm.core.patterns.pattern.AcquiredImageRequest
   :members:

.. autoclass:: pyclm.core.patterns.pattern.CameraProperties
   :members:

.. autoclass:: pyclm.core.patterns.pattern.ROI
   :members:

Gallery metadata for the :doc:`../method_zoo`:

.. autoclass:: pyclm.core.patterns.zoo.ZooMeta
   :members:

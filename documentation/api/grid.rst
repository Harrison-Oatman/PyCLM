Grids (``pyclm.core.grid``)
===========================

A position made of tiles laid out by MicroManager's Create Grid, imaged as
one stitched frame; see the first-time setup for how to make one and
:doc:`../custom_pattern_methods` for what a method sees. A pattern method
reaches the geometry through ``context.grid()``.

.. autoclass:: pyclm.core.grid.GridGeometry
   :members:

.. autofunction:: pyclm.core.grid.group_tiles

.. autofunction:: pyclm.core.grid.geometry_for

.. autofunction:: pyclm.core.grid.stitch

.. autofunction:: pyclm.core.grid.cut

Segmentation methods
====================

Subclass :class:`~pyclm.core.segmentation.SegmentationMethod` and register
it with :meth:`Controller.register_segmentation_method
<pyclm.Controller.register_segmentation_method>`; select it with a
``[segmentation]`` or ``[segmentation.<name>]`` table. ``SegmentationMethod``
is importable from ``pyclm``.

.. autoclass:: pyclm.core.segmentation.SegmentationMethod
   :members:

.. autoclass:: pyclm.core.segmentation.segmentation.SharedSegmentationResource
   :members:

.. autoclass:: pyclm.core.segmentation.segmentation.SharedSegmentationResourceRequest
   :members:

Shipped methods
---------------

.. autoclass:: pyclm.core.segmentation.CellposeSegmentationMethod
   :members:
   :show-inheritance:

.. autoclass:: pyclm.core.segmentation.cellpose_segmentation.EmbryoSegmentationMethod
   :members:
   :show-inheritance:

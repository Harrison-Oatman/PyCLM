Reading and exporting data (``pyclm.io``)
=========================================

``pyclm.io`` reads every format PyCLM has written (HDF5 format 1 and
OME-Zarr formats 2 and 3) behind one API and exports ImageJ hyperstacks.
See :doc:`../data_format` for the layouts.

.. autofunction:: pyclm.io.open

.. autofunction:: pyclm.io.find_experiments

.. autoclass:: pyclm.io.ExperimentData
   :members:

.. autoclass:: pyclm.io.GroupData
   :members:

.. autofunction:: pyclm.io.export_imagej

.. autofunction:: pyclm.io.export_group

.. autofunction:: pyclm.io.export.pattern_overlay

.. autofunction:: pyclm.io.pattern_to_camera

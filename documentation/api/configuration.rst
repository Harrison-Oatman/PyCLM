Configuration schema (``pyclm.schema``)
=======================================

The pydantic models behind the three kinds of file. The keys, defaults
and limits are tabulated in :doc:`../experiment_tomls`; this page is the
programmatic side: load a file, get an ``Experiment``, or generate the
tables.

.. autoclass:: pyclm.schema.ExperimentConfig
   :members: to_experiment, from_file, channel_names, segmentation_names, config_groups_used

.. autoclass:: pyclm.schema.ScheduleConfig
   :members: from_file, timing_kwargs

.. autoclass:: pyclm.schema.PyclmConfig
   :members: from_file, affine, slm_shape, dmd_footprint

.. autoclass:: pyclm.schema.ConfigError
   :members:

.. autofunction:: pyclm.schema.load_model

.. autofunction:: pyclm.schema.describe_errors

.. autofunction:: pyclm.schema.field_table

The MicroManager configuration file, read as text for ``pyclm check``:

.. autoclass:: pyclm.mmconfig.MMConfig
   :members:

The pipeline (for developers)
=============================

The processes a run is made of and the objects that pass between them.
The runtime map is in ``docs/architecture-notes.md`` in the repository;
this page is the reference for the names it uses.

The plan
--------

.. autoclass:: pyclm.core.plan.AcquisitionPlan
   :members: from_schedule, events_at, datasets_at, expected_datasets, is_scheduled, pattern_due, every_t, stop_experiment, estimate_timepoint_s, over_budget, to_yaml, from_yaml

.. autoclass:: pyclm.core.plan.PlannedEvent
   :members:

.. autoclass:: pyclm.core.experiments.ExperimentSchedule
   :members:

.. autoclass:: pyclm.core.experiments.Experiment
   :members:

.. autoclass:: pyclm.core.experiments.ImagingConfig
   :members:

.. autoclass:: pyclm.core.experiments.MicroscopePosition
   :members:

Processes
---------

.. autoclass:: pyclm.core.base_process.PipelineProcess
   :members:

.. autoclass:: pyclm.core.manager.Manager
   :members: initialize, dispatch, apply_settings, apply_command, poll_commands, status, next_timepoint

.. autoclass:: pyclm.core.manager.SLMBuffer
   :members: initialize, pattern_to_slm

.. autofunction:: pyclm.core.manager.compose_affine

.. autoclass:: pyclm.core.microscope.MicroscopeProcess
   :members: handle_acquisition_event, handle_update_pattern_event, handle_update_position_event, set_binning, declare_slm

.. autofunction:: pyclm.core.microscope.pattern_is_blank

.. autoclass:: pyclm.core.pattern_process.PatternProcess
   :members: register_method, request_method, initialize_models

.. autoclass:: pyclm.core.segmentation_process.SegmentationProcess
   :members: register_method, request_method

.. autoclass:: pyclm.core.tracking_process.TrackingProcess
   :members: register_method, request_method

.. autoclass:: pyclm.core.writer_process.WriterProcess
   :members: initialize, subscriptions

The router
----------

.. autoclass:: pyclm.core.router.Router
   :members: add, resolve, publish, end_stream, as_dict, demanded, demanded_kinds

.. autoclass:: pyclm.core.router.Subscription
   :members:

.. autoclass:: pyclm.core.router.RoutingError

Events and data
---------------

.. autoclass:: pyclm.core.events.AcquisitionEvent
   :members:

.. autoclass:: pyclm.core.datatypes.AcquisitionData
   :members:

.. autoclass:: pyclm.core.datatypes.StimulationData
   :members:

.. autoclass:: pyclm.core.datatypes.SegmentationData
   :members:

.. autoclass:: pyclm.core.datatypes.CameraPattern
   :members:

.. autoclass:: pyclm.core.datatypes.SkippedAcquisition
   :members:

Storage
-------

.. autoclass:: pyclm.core.storage.base.FrameWriter
   :members:

.. autoclass:: pyclm.core.storage.base.CadenceGroup
   :members:

.. autofunction:: pyclm.core.storage.base.cadence_groups

.. autoclass:: pyclm.core.storage.ome_zarr.OMEZarrWriter
   :show-inheritance:

.. autoclass:: pyclm.core.storage.hdf5_v1.HDF5WriterV1
   :show-inheritance:

.. autoclass:: pyclm.core.storage.events.EventLog
   :members: record, table, close

The microscope core
-------------------

.. autoclass:: pyclm.core.core_interface.MicroscopeCoreInterface
   :members:

.. autoclass:: pyclm.core.real_core.RealMicroscopeCore
   :show-inheritance:

.. autoclass:: pyclm.core.virtual_microscope.simulated_core.SimulatedMicroscopeCore
   :show-inheritance:

.. autoclass:: pyclm.core.virtual_microscope.simulated_source.TimeSeriesImageSource
   :members: from_mapping, next_frame, nearest

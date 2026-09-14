Running and preparing experiments
=================================

The Python entry points behind the ``pyclm`` command (see
:doc:`../command_line`). ``run_pyclm``, ``Controller`` and the position
movers are importable from ``pyclm``.

Running
-------

.. autofunction:: pyclm.run_pyclm

.. autoclass:: pyclm.Controller
   :members: register_pattern_method, register_segmentation_method, register_tracking_method, add_process, initialize, run

Position movers
---------------

How the stage reaches a position; chosen with ``position_mover`` in
``pyclm_config.toml`` or passed to ``run_pyclm``.

.. autoclass:: pyclm.core.position_mover.PositionMover
   :members:

.. autoclass:: pyclm.core.position_mover.BasicPositionMover
   :show-inheritance:

.. autoclass:: pyclm.core.position_mover.PFSPositionMover
   :members:
   :show-inheritance:

.. autofunction:: pyclm.core.position_mover.resolve_mover

Checking
--------

.. autofunction:: pyclm.check.check_directory

.. autoclass:: pyclm.check.CheckReport
   :members:

.. autoclass:: pyclm.check.Finding
   :members:

.. autoclass:: pyclm.check.CheckFailed

.. autofunction:: pyclm.check.check_kwargs

Preview
-------

.. autofunction:: pyclm.preview.preview

.. autoclass:: pyclm.preview.PreviewResult
   :members:

Templates
---------

.. autofunction:: pyclm.templates.create

Commands to a running experiment
--------------------------------

.. autoclass:: pyclm.commands.Command
   :members: describe, to_changes

.. autofunction:: pyclm.commands.write_command

.. autofunction:: pyclm.commands.pending

The experiment directory
------------------------

.. autofunction:: pyclm.directories.schedule_from_directory

.. autofunction:: pyclm.directories.dry_schedule_from_directory

.. autofunction:: pyclm.directories.positions_from_pos

.. autofunction:: pyclm.directories.positions_from_xml

.. autofunction:: pyclm.directories.write_position_list

.. autofunction:: pyclm.directories.find_schedule

.. autofunction:: pyclm.directories.find_config_in

.. autofunction:: pyclm.directories.experiment_tomls

.. autoclass:: pyclm.directories.AmbiguousFileError

.. autoclass:: pyclm.directories.DrySettings
   :members:

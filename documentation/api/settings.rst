Runtime settings (``pyclm.core.settings``)
==========================================

Setting changes a pattern method requests for its own experiment through
the context (``set_exposure``, ``set_config``, ``set_property``,
``set_position``), and that commands apply from outside; see
:doc:`../custom_pattern_methods` and :doc:`../command_line`.

.. autoclass:: pyclm.core.settings.SettingChange
   :members:

.. autofunction:: pyclm.core.settings.check_change

.. autofunction:: pyclm.core.settings.exposure

.. autofunction:: pyclm.core.settings.config

.. autofunction:: pyclm.core.settings.device_property

.. autofunction:: pyclm.core.settings.position

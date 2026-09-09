Runtime settings (``pyclm.core.settings``)
==========================================

Setting changes a pattern method requests for its own experiment through
the context (``set_exposure``, ``set_config``, ``set_property``,
``set_position``); see :doc:`../custom_pattern_methods`.

.. autoclass:: pyclm.core.settings.SettingChange
   :members:

.. autofunction:: pyclm.core.settings.check_change

.. autoclass:: pyclm.core.storage.events.EventLog
   :members: record, table, close

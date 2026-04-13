KTR / Nucleus Control Patterns
================================

Patterns that use per-cell intensity from a reporter channel (e.g. a
kinase translocation reporter) to make binary or graded illumination
decisions per cell.

.. autoclass:: pyclm.core.patterns.ktr_patterns.NucleusControlMethod
   :members: generate, process_prop
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.ktr_patterns.BinaryNucleusClampModel
   :members: process_prop
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.ktr_patterns.GlobalCycleModel
   :members: generate
   :show-inheritance:

.. autoclass:: pyclm.core.patterns.ktr_patterns.CenteredImageModel
   :members: generate
   :show-inheritance:

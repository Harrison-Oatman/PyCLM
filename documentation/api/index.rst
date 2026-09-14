API reference
=============

The reference is organised by what you are doing. Everything a user needs
is importable from ``pyclm`` or ``pyclm.io``; the pipeline internals are
documented last, for developers.

**Running and preparing experiments**

.. toctree::
   :maxdepth: 1

   running
   configuration

**Writing methods**

.. toctree::
   :maxdepth: 1

   pattern_methods
   measure
   settings
   segmentation
   tracking
   grid

**Reading data**

.. toctree::
   :maxdepth: 1

   io

**The shipped pattern methods**

Each is a :class:`~pyclm.core.patterns.pattern.PatternMethod`; the
:doc:`../method_zoo` shows what they look like and the TOML that selects
them.

.. toctree::
   :maxdepth: 1

   static_patterns
   bar_patterns
   wave_patterns
   per_cell_patterns
   intensity_patterns
   embryo_patterns

**The pipeline (for developers)**

.. toctree::
   :maxdepth: 1

   pipeline

"""``python -m pyclm`` and the ``pyclm`` command: see :mod:`pyclm.cli`."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())

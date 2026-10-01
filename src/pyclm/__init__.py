import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    # the distribution is closed-loop-microscopy; the import name is pyclm
    __version__ = _version("closed-loop-microscopy")
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "unknown"

from .controller import Controller
from .core import (
    AcquisitionPlan,
    MicroscopePosition,
    PatternContext,
    PatternMethod,
    SegmentationMethod,
    TrackingMethod,
    Tracks,
)
from .core.measure import PerTrack, Regions, nuclear_cytosolic_ratio
from .core.position_mover import BasicPositionMover, PFSPositionMover, PositionMover
from .run_pyclm import run_pyclm

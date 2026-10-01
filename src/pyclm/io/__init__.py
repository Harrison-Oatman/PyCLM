"""
Read and export PyCLM data without the acquisition pipeline.

    import pyclm.io
    exp = pyclm.io.open("experiment_dir/bar10.pos1.zarr")   # or .hdf5
    pyclm.io.export_imagej(exp)                             # ImageJ hyperstacks next to the data
"""

from .export import export_group, export_imagej, pattern_to_camera
from .readers import (
    ExperimentData,
    GroupData,
    HDF5Experiment,
    ZarrExperiment,
    find_experiments,
    open,
)

__all__ = [
    "ExperimentData",
    "GroupData",
    "HDF5Experiment",
    "ZarrExperiment",
    "export_group",
    "export_imagej",
    "find_experiments",
    "open",
    "pattern_to_camera",
]

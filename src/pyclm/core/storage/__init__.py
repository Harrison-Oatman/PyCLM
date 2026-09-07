"""Frame writers: the HDF5 v1 layout and the OME-Zarr layout (format 2)."""

from .base import (
    PATTERN_POLICIES,
    STORAGE_FORMATS,
    CadenceGroup,
    FrameWriter,
    cadence_groups,
    image_shape,
)
from .hdf5_v1 import HDF5WriterV1
from .ome_zarr import OMEZarrWriter


def make_writer(
    storage_format: str = "hdf5", pattern_policy: str = "on_change"
) -> FrameWriter:
    if storage_format == "hdf5":
        return HDF5WriterV1()
    if storage_format == "ome-zarr":
        return OMEZarrWriter(pattern_policy=pattern_policy)
    raise ValueError(
        f"unknown storage format {storage_format!r}; expected one of {STORAGE_FORMATS}"
    )


__all__ = [
    "PATTERN_POLICIES",
    "STORAGE_FORMATS",
    "CadenceGroup",
    "FrameWriter",
    "HDF5WriterV1",
    "OMEZarrWriter",
    "cadence_groups",
    "image_shape",
    "make_writer",
]

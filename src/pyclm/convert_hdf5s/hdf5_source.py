import json
import logging
from pathlib import Path

import h5py
import numpy as np
from natsort import natsorted

logger = logging.getLogger(__name__)


class ExperimentFile:
    """
    SWMR-safe read access to a single experiment HDF5 file.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._file: h5py.File | None = None

    def __enter__(self) -> "ExperimentFile":
        self._file = h5py.File(self.path, mode="r", libver="latest", swmr=True)
        return self

    def __exit__(self, *exc_info):
        if self._file is not None:
            self._file.close()
            self._file = None

    def _timepoint_keys(self) -> list[str]:
        return natsorted(
            key for key in self._file if isinstance(self._file[key], h5py.Group)
        )

    def read_binning(self, channel_key: str) -> int:
        """
        Extract binning from experiment_metadata, or return default 1.
        """
        if "experiment_metadata" not in self._file.attrs:
            return 1

        try:
            meta = json.loads(self._file.attrs["experiment_metadata"])
            if channel_key.startswith("channel_"):
                short_name = channel_key.removeprefix("channel_")
                return meta.get("channels", {}).get(short_name, {}).get("binning", 1)
            return meta.get("segmentation", {}).get("binning", 1)
        except Exception:
            logger.warning(
                "Could not read binning from metadata for %s",
                channel_key,
                exc_info=True,
            )
            return 1

    def read_camera_roi(self) -> tuple[int, int, int, int] | None:
        """
        Read the camera ROI (x_offset, y_offset, width, height) recorded at
        acquisition time, or None if the file predates this metadata.
        """
        roi = self._file.attrs.get("camera_roi")
        if roi is None:
            return None
        x0, y0, w, h = roi
        return int(x0), int(y0), int(w), int(h)

    def has_seg(self, channel_key: str) -> bool:
        for t_key in self._timepoint_keys():
            group = self._file[t_key]
            if channel_key in group and "seg" in group[channel_key]:
                return True
        return False

    def iter_frames(self, channel_key: str):
        """
        Yield (t_key, data, seg, dmd) for each timepoint with data for
        `channel_key`.

        `seg` is `None` if no timepoint for this channel has segmentation
        output, otherwise an array (zeros if this particular timepoint is
        missing it) - so every yielded frame has the same stack depth.

        `dmd` is the stim_aq DMD pattern (SLM coordinates) for this
        timepoint, or `None` if it wasn't acquired/saved.
        """
        has_seg = self.has_seg(channel_key)

        for t_key in self._timepoint_keys():
            group = self._file[t_key]

            if channel_key not in group or "data" not in group[channel_key]:
                continue

            try:
                data_dset = group[channel_key]["data"]
                data_dset.refresh()
                data = np.array(data_dset)

                if data.shape == (0, 0):
                    continue

                seg = None
                if has_seg:
                    if "seg" in group[channel_key]:
                        seg_dset = group[channel_key]["seg"]
                        seg_dset.refresh()
                        seg = np.array(seg_dset)
                    else:
                        seg = np.zeros(data.shape, dtype=data.dtype)

                dmd = None
                if "stim_aq" in group and "dmd" in group["stim_aq"]:
                    dmd_dset = group["stim_aq"]["dmd"]
                    dmd_dset.refresh()
                    dmd = np.array(dmd_dset)
            except Exception:
                # likely a frame still being written (SWMR)
                continue

            yield t_key, data, seg, dmd

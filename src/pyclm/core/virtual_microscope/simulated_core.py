from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Dict
import numpy as np

from ..core_interface import MicroscopeCoreInterface


class SimulatedMicroscopeCore(MicroscopeCoreInterface):
    PFS_STATUS_LOCKED = "0000001100001010"

    def __init__(
        self,
        image_source,
        pixel_size_um: float = 0.108,
        camera_name: str = "SimulatedCamera",
        slm_device: str = "SimulatedSLM",
        slm_shape: Tuple[int, int] = (1140, 900),
        initial_xy: Tuple[float, float] = (0.0, 0.0),
        initial_z: float = 0.0,
    ):
        self._image_source = image_source
        self._camera_name = camera_name
        self._pixel_size_um = float(pixel_size_um)

        h, w = image_source.shape[:2]
        self._roi = (0, 0, int(w), int(h))

        self._last_image: Optional[np.ndarray] = None
        self._exposure_ms: float = 0.0

        self._properties: Dict[str, Dict[str, Any]] = {camera_name: {"Binning": "1x1"}}
        self._config_groups: Dict[str, str] = {}

        self._x, self._y = float(initial_xy[0]), float(initial_xy[1])
        self._z = float(initial_z)
        self._autofocus_offset = 0.0

        self._properties.setdefault("PFS", {})
        self._properties["PFS"].setdefault("FocusMaintenance", "Off")
        self._properties["PFS"].setdefault("PFS Status", self.PFS_STATUS_LOCKED)

        self._slm_device = slm_device  # set None to simulate no SLM device
        self._slm_h, self._slm_w = int(slm_shape[0]), int(slm_shape[1])
        self._slm_image: Optional[np.ndarray] = None

    def loadSystemConfiguration(self, configuration) -> None:
        cfg = str(configuration)
        self._loaded_configuration = cfg
    
    # SLM-related
    def getSLMDevice(self) -> str:
        return "" if self._slm_device is None else str(self._slm_device)

    def getSLMHeight(self, device: str) -> int:
        return self._slm_h

    def getSLMWidth(self, device: str) -> int:
        return self._slm_w

    def setSLMImage(self, device: str, image: Any) -> None:
        self._slm_image = np.asarray(image)

    # Config/device properties
    def setConfig(self, group: str, config: str) -> None:
        self._config_groups[str(group)] = str(config)

    def getAllowedPropertyValues(self, device: str, prop_name: str) -> Sequence[str]:
        if device == self._camera_name and prop_name == "Binning":
            return ["1x1", "2x2", "4x4"]
        if device == "PFS" and prop_name == "FocusMaintenance":
            return ["On", "Off"]
        return []

    def setProperty(self, label: str, name: str, value: Any) -> None:
        self._properties.setdefault(label, {})
        self._properties[label][name] = value
        if label == "PFS" and name == "FocusMaintenance" and str(value) == "On":
            self._properties["PFS"]["PFS Status"] = self.PFS_STATUS_LOCKED

    def getProperty(self, label: str, name: str) -> str:
        return str(self._properties.get(label, {}).get(name, ""))

    # Camera-related
    def getCameraDevice(self) -> str:
        return self._camera_name

    def setExposure(self, exposure_ms: float) -> None:
        self._exposure_ms = float(exposure_ms)

    def snapImage(self) -> None:
        frame = np.asarray(self._image_source.next_frame())
        x, y, w, h = self._roi
        frame = frame[y : y + h, x : x + w, ...] if frame.ndim >= 2 else frame

        binning = self._properties.get(self._camera_name, {}).get("Binning", "1x1")
        try:
            b = int(str(binning).split("x")[0])
        except Exception:
            b = 1

        if b > 1 and frame.ndim >= 2:
            frame = self._bin2d(frame, b)

        self._last_image = frame

    def getImage(self) -> Any:
        if self._last_image is None:
            self.snapImage()
        return self._last_image

    def getPixelSizeUm(self) -> float:
        return self._pixel_size_um
    
    def getROI(self):
        return self._roi

    # Stage/focus/positioning
    def getZPosition(self) -> float:
        return float(self._z)

    def setPosition(self, z: float) -> None:
        self._z = float(z)

    def setXYPosition(self, x: float, y: float) -> None:
        self._x, self._y = float(x), float(y)

    def setAutoFocusOffset(self, offset: float) -> None:
        self._autofocus_offset = float(offset)
        if self.getProperty("PFS", "FocusMaintenance") == "On":
            self._properties["PFS"]["PFS Status"] = self.PFS_STATUS_LOCKED

    # Synchronization
    def waitForSystem(self) -> None:
        return None

    @staticmethod
    def _bin2d(frame: np.ndarray, b: int) -> np.ndarray:
        if b <= 1:
            return frame
        if frame.ndim == 2:
            h, w = frame.shape
            h2, w2 = (h // b) * b, (w // b) * b
            f = frame[:h2, :w2].reshape(h2 // b, b, w2 // b, b).mean(axis=(1, 3))
            return f.astype(frame.dtype) if np.issubdtype(frame.dtype, np.integer) else f
        if frame.ndim == 3:
            h, w, c = frame.shape
            h2, w2 = (h // b) * b, (w // b) * b
            f = frame[:h2, :w2, :].reshape(h2 // b, b, w2 // b, b, c).mean(axis=(1, 3))
            return f.astype(frame.dtype) if np.issubdtype(frame.dtype, np.integer) else f
        return frame
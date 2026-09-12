from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pymmcore_plus import CMMCorePlus

from .core_interface import MicroscopeCoreInterface


class RealMicroscopeCore(MicroscopeCoreInterface):
    """
    Delegates everything to an internal CMMCorePlus.
    """

    def __init__(self):
        self._core = CMMCorePlus()

    def loadSystemConfiguration(self, configuration):
        try:
            self._core.loadSystemConfiguration(str(configuration))
        except Exception as e:
            if "interface" in str(e).lower():
                raise RuntimeError(f"{e}" + chr(10) + device_interface_hint()) from e
            raise


def device_interface() -> int | None:
    """The Micro-Manager device interface version this pymmcore was built for."""
    try:
        import pymmcore

        return int(pymmcore.__version__.split(".")[3])
    except Exception:
        return None


def device_interface_hint() -> str:
    di = device_interface()
    return (
        f"pymmcore speaks Micro-Manager device interface {di}: every device "
        "adapter DLL it loads must be built for the same interface. Install a "
        "Micro-Manager build of that interface (or the vendor's matching adapter), "
        "or change the pymmcore pin (see pyproject.toml, [tool.uv])."
    )

    # SLM-related
    def getSLMDevice(self) -> str:
        return self._core.getSLMDevice()

    def getSLMHeight(self, device: str) -> int:
        return self._core.getSLMHeight(device)

    def getSLMWidth(self, device: str) -> int:
        return self._core.getSLMWidth(device)

    def setSLMImage(self, device: str, image: Any) -> None:
        self._core.setSLMImage(device, image)

    # Config/device properties
    def setConfig(self, group: str, config: str) -> None:
        self._core.setConfig(group, config)

    def setProperty(self, label: str, name: str, value: Any) -> None:
        self._core.setProperty(label, name, value)

    def getProperty(self, label: str, name: str) -> str:
        return self._core.getProperty(label, name)

    def getAllowedPropertyValues(self, device: str, prop_name: str) -> Sequence[str]:
        return self._core.getAllowedPropertyValues(device, prop_name)

    def describe(self) -> str:
        return self._core.describe()

    def setFocusDevice(self, label: str) -> None:
        self._core.setFocusDevice(label)

    def getAvailableConfigGroups(self) -> Sequence[str]:
        return self._core.getAvailableConfigGroups()

    def getConfigGroupObject(self, group: str, include_read_only: bool = False):
        return self._core.getConfigGroupObject(group, include_read_only)

    # Camera-related
    def getCameraDevice(self) -> str:
        return self._core.getCameraDevice()

    def setExposure(self, exposure_ms: float) -> None:
        self._core.setExposure(exposure_ms)

    def snapImage(self) -> None:
        self._core.snapImage()

    def getImage(self) -> Any:
        return self._core.getImage()

    def getPixelSizeUm(self) -> float:
        return self._core.getPixelSizeUm()

    def getROI(self):
        return self._core.getROI()

    def setROI(self, x: int, y: int, width: int, height: int) -> None:
        self._core.setROI(int(x), int(y), int(width), int(height))

    # Stage/focus/positioning
    def getXYPosition(self) -> tuple[float, float]:
        x, y = self._core.getXYPosition()
        return float(x), float(y)

    def getZPosition(self) -> float:
        return self._core.getZPosition()

    def setPosition(self, z: float) -> None:
        self._core.setPosition(z)

    def setXYPosition(self, x: float, y: float) -> None:
        self._core.setXYPosition(x, y)

    def setAutoFocusOffset(self, offset: float) -> None:
        self._core.setAutoFocusOffset(offset)

    # Synchronization
    def waitForSystem(self) -> None:
        self._core.waitForSystem()

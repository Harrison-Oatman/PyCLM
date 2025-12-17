from __future__ import annotations

from typing import Any, Sequence
from pymmcore_plus import CMMCorePlus

from .core_interface import MicroscopeCoreInterface


class RealMicroscopeCore(MicroscopeCoreInterface):
    """
    Delegates everything to an internal CMMCorePlus.
    """

    def __init__(self):
        self._core = CMMCorePlus()

    def loadSystemConfiguration(self, configuration):
        self._core.loadSystemConfiguration(str(configuration))

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

    # Stage/focus/positioning
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
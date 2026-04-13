"""
Abstractions for moving the microscope stage to a MicroscopePosition.

Subclass PositionMover to implement hardware-specific focus maintenance.
Register a custom mover via Controller(position_mover=...) or by passing it
to run_pyclm().
"""

import logging
from abc import ABC, abstractmethod
from time import time

logger = logging.getLogger(__name__)


class PositionMover(ABC):
    """
    Abstract base for stage movement.  Implement ``move_to`` for your hardware.

    Returns
    -------
    (z_moved, new_z) : tuple[bool, float]
        ``z_moved`` is True when a z-axis adjustment was made.
        ``new_z`` is the actual z coordinate reported by the core after the move.
    """

    @abstractmethod
    def move_to(self, position, core) -> tuple[bool, float]:
        """Move the stage to *position* using *core*."""


class BasicPositionMover(PositionMover):
    """
    Simple XYZ mover with no focus-maintenance polling.

    Moves XY then Z and returns the position reported by the core.
    Suitable for microscopes without hardware autofocus or when using the
    simulated core in dry-run mode.
    """

    def move_to(self, position, core) -> tuple[bool, float]:
        start = time()

        logger.info(f"moving to xy ({position.x}, {position.y})")
        core.setXYPosition(position.x, position.y)

        logger.info(f"moving to z {position.z}")
        core.setPosition(position.z)

        logger.info(f"move took {time() - start:.3f}s")
        return True, core.getZPosition()


class PFSPositionMover(PositionMover):
    """
    Position mover for Nikon microscopes with the Perfect Focus System (PFS).

    Moves to XY/Z, applies an optional PFS offset stored in
    ``position.extras["PFSOffset"]``, then polls the PFS status property until
    focus is confirmed locked.

    The y-axis is negated on XY movement to match the Nikon stage convention.
    Override ``PFS_DEVICE``, ``PFS_MAINTENANCE_PROPERTY``, ``PFS_STATUS_PROPERTY``,
    and ``PFS_LOCKED_VALUE`` on a subclass if your hardware differs.
    """

    PFS_DEVICE = "PFS"
    PFS_MAINTENANCE_PROPERTY = "FocusMaintenance"
    PFS_STATUS_PROPERTY = "PFS Status"
    PFS_LOCKED_VALUE = "0000001100001010"

    def move_to(self, position, core) -> tuple[bool, float]:
        start = time()

        pfs_offset = position.extras.get("PFSOffset")

        # Move Z down first if needed to avoid objective collision on XY slew
        if position.z < core.getZPosition():
            core.setPosition(position.z)

        logger.info(f"moving to xy ({position.x}, {position.y})")
        core.setXYPosition(position.x, -position.y)

        logger.info(f"moving to z {position.z}")
        core.setPosition(position.z)

        if pfs_offset is not None:
            logger.info(f"setting PFS offset {pfs_offset}")
            core.setAutoFocusOffset(pfs_offset)

        core.setProperty(self.PFS_DEVICE, self.PFS_MAINTENANCE_PROPERTY, "On")

        while (
            core.getProperty(self.PFS_DEVICE, self.PFS_STATUS_PROPERTY)
            != self.PFS_LOCKED_VALUE
        ):
            pass

        logger.info(f"move+focus took {time() - start:.3f}s")
        return True, core.getZPosition()

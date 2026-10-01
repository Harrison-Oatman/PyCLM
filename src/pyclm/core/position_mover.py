"""
Abstractions for moving the microscope stage to a MicroscopePosition.

Subclass PositionMover to implement hardware-specific focus maintenance.
Register a custom mover via Controller(position_mover=...) or by passing it
to run_pyclm().
"""

import inspect
import logging
from abc import ABC, abstractmethod
from time import sleep, time

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
    focus is confirmed locked, raising ``TimeoutError`` after ``PFS_TIMEOUT_S``.

    The PFS can switch itself off (the interface was lost, or someone pressed
    the button) and then ignore a repeated "On"; while waiting, the mover
    switches focus maintenance off and on again every ``PFS_RETRY_S``.
    Override ``PFS_DEVICE``, ``PFS_MAINTENANCE_PROPERTY``, ``PFS_STATUS_PROPERTY``,
    and ``PFS_LOCKED_VALUE`` on a subclass if your hardware differs.
    """

    PFS_DEVICE = "PFS"
    PFS_MAINTENANCE_PROPERTY = "FocusMaintenance"
    PFS_STATUS_PROPERTY = "PFS Status"
    PFS_LOCKED_VALUE = "0000001100001010"
    PFS_TIMEOUT_S = 30.0
    PFS_POLL_S = 0.01
    PFS_RETRY_S = 5.0  # no lock after this long: switch focus maintenance off and on

    def move_to(self, position, core) -> tuple[bool, float]:
        start = time()

        pfs_offset = position.extras.get("PFSOffset")

        # Move Z down first if needed to avoid objective collision on XY slew
        if position.z < core.getZPosition():
            core.setPosition(position.z)

        logger.info(f"moving to xy ({position.x}, {position.y})")
        core.setXYPosition(position.x, position.y)

        logger.info(f"moving to z {position.z}")
        core.setPosition(position.z)

        if pfs_offset is not None:
            logger.info(f"setting PFS offset {pfs_offset}")
            core.setAutoFocusOffset(pfs_offset)

        core.setProperty(self.PFS_DEVICE, self.PFS_MAINTENANCE_PROPERTY, "On")

        lock_start = time()
        seen: dict[str, int] = {}
        retries = 0
        next_retry = lock_start + self.PFS_RETRY_S
        while (
            status := core.getProperty(self.PFS_DEVICE, self.PFS_STATUS_PROPERTY)
        ) != self.PFS_LOCKED_VALUE:
            seen[status] = seen.get(status, 0) + 1
            if time() - lock_start > self.PFS_TIMEOUT_S:
                # the status strings seen say why: searching, out of range, off
                history = ", ".join(f"{s!r} x{n}" for s, n in seen.items())
                raise TimeoutError(
                    f"PFS did not report focus lock within {self.PFS_TIMEOUT_S}s "
                    f"at x={position.x}, y={position.y}, z={position.z} "
                    f"(offset {pfs_offset}, z now {core.getZPosition():.2f}); "
                    f"status seen: {history}; focus maintenance was switched "
                    f"off and on {retries} time(s)"
                )
            if time() >= next_retry:
                retries += 1
                next_retry = time() + self.PFS_RETRY_S
                logger.warning(
                    f"PFS not locked after {time() - lock_start:.1f}s (status "
                    f"{status!r}); switching focus maintenance off and on "
                    f"(attempt {retries})"
                )
                self._restart_focus_maintenance(core)
            sleep(self.PFS_POLL_S)

        if retries:
            logger.warning(
                f"PFS locked after {retries} restart(s) of focus maintenance"
            )

        logger.info(f"move+focus took {time() - start:.3f}s")
        return True, core.getZPosition()

    def _restart_focus_maintenance(self, core) -> None:
        """Off then on: a repeated "On" alone does not restart a PFS that switched itself off."""
        try:
            core.setProperty(self.PFS_DEVICE, self.PFS_MAINTENANCE_PROPERTY, "Off")
            sleep(0.2)
            core.setProperty(self.PFS_DEVICE, self.PFS_MAINTENANCE_PROPERTY, "On")
        except Exception as e:
            logger.warning(f"restarting PFS focus maintenance failed: {e}")


MOVERS: dict[str, type[PositionMover]] = {
    "basic": BasicPositionMover,
    "pfs": PFSPositionMover,
}


def resolve_mover(spec: str) -> type[PositionMover]:
    """
    The :class:`PositionMover` subclass named by ``spec``: one of
    :data:`MOVERS` (``"basic"``, ``"pfs"``) or a dotted path
    ``"package.module:ClassName"`` to a custom subclass.
    """
    key = (spec or "basic").strip()
    if key.lower() in MOVERS:
        return MOVERS[key.lower()]
    if ":" not in key:
        raise ValueError(
            f"position_mover {spec!r} is not one of {sorted(MOVERS)} and is not a "
            "'package.module:ClassName' path"
        )
    module_name, _, class_name = key.partition(":")
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ValueError(
            f"position_mover {spec!r}: cannot import {module_name!r} ({e})"
        ) from e
    cls = getattr(module, class_name, None)
    if (
        cls is None
        or not (isinstance(cls, type) and issubclass(cls, PositionMover))
        or inspect.isabstract(cls)
    ):
        raise ValueError(
            f"position_mover {spec!r}: {class_name!r} in {module_name!r} is not a "
            "concrete PositionMover subclass"
        )
    return cls

"""
Contains classes for passing messages between processes.

These messages are allowed to contain small bits of information, but
should not pass numpy arrays (see datatypes.py)
"""

from .events import AcquisitionEvent, UpdatePatternEvent, UpdateStagePositionEvent


class Message:
    # will always be a string
    message = "BASE_MESSAGE"

    def __repr__(self):
        return f"message: {self.message}"


class CloseMessage(Message):
    """Sent by the manager to every process once the schedule is complete."""

    message = "close"


class AcquisitionEventMessage(Message):
    message = "acquisition_event"

    def __init__(self, event: AcquisitionEvent):
        self.event = event


class UpdatePatternEventMessage(Message):
    message = "update_pattern_event"

    def __init__(self, event: UpdatePatternEvent):
        self.event = event


class UpdatePositionEventMessage(Message):
    message = "update_position_event"

    def __init__(self, event: UpdateStagePositionEvent):
        self.event = event


class StreamCloseMessage(Message):
    message = "stream_close"


class UpdateZPositionMessage(Message):
    message = "update_z_position"

    def __init__(self, new_z_position, experiment_name):
        self.new_z_position = new_z_position
        self.experiment_name = experiment_name


class SettingsRequestMessage(Message):
    """
    Setting changes a pattern method asked for during ``generate`` at
    timepoint ``t_requested``, sent by the pattern process to the Manager
    (``changes`` are :class:`~pyclm.core.settings.SettingChange`).
    """

    message = "settings_request"

    def __init__(self, experiment_name: str, t_requested: int, changes: list):
        self.experiment_name = experiment_name
        self.t_requested = int(t_requested)
        self.changes = list(changes)


class EventDoneMessage(Message):
    """
    The microscope's acknowledgement of one acquisition event: when it was
    scheduled, when it completed, and the error text if it failed.
    """

    message = "event_done"

    def __init__(self, event: AcquisitionEvent, error: str | None = None):
        self.event_id = event.id
        self.experiment_name = event.experiment_name
        self.index = dict(event.index)
        self.channel = event.index.get("c")
        self.scheduled_time = event.scheduled_time
        self.completed_time = event.completed_time
        self.error = error

    @property
    def t_index(self) -> int:
        return int(self.index.get("t", 0))

    @property
    def lateness_s(self) -> float | None:
        if self.completed_time is None or not self.scheduled_time:
            return None
        return float(self.completed_time - self.scheduled_time)

"""
Contains classes for passing messages between processes.

These messages are allowed to contain small bits of information, but
should not pass numpy arrays (see datatypes.py)
"""

from abc import ABC
from .events import *


class Message(ABC):

    # will always be a string
    message = "BASE_MESSAGE"

    def __repr__(self):
        return f"message: {self.message}"


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

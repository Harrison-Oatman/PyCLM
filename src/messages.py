"""
Contains classes for passing messages between processes.

These messages are allowed to contain small bits of information, but
should not pass numpy arrays (see datatypes.py)
"""

from abc import ABC


class Message(ABC):

    # will always be a string
    message = "BASE_MESSAGE"


class AcquisitionEventMessage(Message):

    message = "acquisition_event"

    def __init__(self):
        pass

"""
Contains classes for passing messages between processes.

These messages are allowed to contain small bits of information, but
should not pass data that should be represented by a numpy array (see datatypes.py)
"""

from abc import ABC


class Message(ABC):

    # will always be a string
    message = "BASE_MESSAGE"

    # optional, unrestricted
    field = None


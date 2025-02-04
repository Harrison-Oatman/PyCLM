"""
Contains classes for passing structured data between processes
"""


import numpy as np


class EventSLMPattern:

    def __init__(self, event_id, pattern):
        self.event_id = event_id
        self.pattern = pattern

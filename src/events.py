import numpy as np

class AcquisitionEvent:

    def __init__(self):


class Position:

    def __init__(self, x=None, y=None, z=None, pfs=None, label=None):

        self.label=label
        self.x = x
        self.y = y
        self.z = z
        self.pfs = pfs


    def get_xy(self):
        if not ((self.x is None) or (self.y is None)):
            return [self.x, self.y]

        return None

    def get_z(self):
        return self.z

    def get_pfs(self):
        return self.get_pfs



# class PositionGrid:
#     """
#     Contains a single grid of xy(z) positions
#     """
#
#     def __init__(self, label):
#         self.label = label
#
#     def add_positions(self, positions):

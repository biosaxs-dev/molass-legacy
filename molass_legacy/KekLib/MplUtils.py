"""
    MplUtils.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from matplotlib.colors import BoundaryNorm as ParentNorm

class BoundaryNorm(ParentNorm):
    def __init__(self, *args, **kwargs):
        self.inv_ret = kwargs.pop("inv_ret", -9999)     # not sure for -9999, which seems ok for CorMap
        ParentNorm.__init__(self, *args, **kwargs)

    def inverse(self, value):
        try:
            return ParentNorm.inverse(self, value)
        except ValueError:
            # ignore because it is known as below.
            # BoundaryNorm is not invertible, so calling this method will always raise an error
            # users should set inv_ret appropriately
            pass

        return self.inv_ret

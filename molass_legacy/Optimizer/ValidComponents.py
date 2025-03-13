"""
    ValidComponents.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

VALID_COMPONENTS_THRESHOLD = 0.01

class ValidComponents:
    def __init__(self, nc):
        self.nc = nc

    def update(self, optimizer):
        self.vc_vector = optimizer.minima_props > VALID_COMPONENTS_THRESHOLD

    def get_valid_vector(self):
        return np.ones(self.nc, dtype=bool)
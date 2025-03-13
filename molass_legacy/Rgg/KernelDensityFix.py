# coding: utf-8
"""
    Rgg.KernelDensityDemo.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from pomegranate.distributions import (KernelDensity, UniformKernelDensity as UniformKernelDensityOrig)

class UniformKernelDensity(UniformKernelDensityOrig):
    def __init__(self,  points=[], bandwidth=1, weights=None, frozen=False):
        super(self.__class__, self).__init__()
        self.points = points
        self.bandwidth = bandwidth
        self.weights = weights
        self.frozen = frozen
        self.scale_fix = 1/(self.bandwidth * 2 * len(self.points))

    def probability(self, X):
        p = UniformKernelDensityOrig.probability(self, X)
        return p * self.scale_fix

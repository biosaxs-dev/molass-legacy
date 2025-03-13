"""
    Optimizer.BH.Customize.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from scipy._lib._util import check_random_state
from .BasinHopping import AdaptiveStepsize, RandomDisplacement

class DefaultTakeStep(AdaptiveStepsize):
    def __init__(self, stepsize=0.5, interval=50, disp=False, seed=None):

        # set up the np.random generator
        self.rng = check_random_state(seed)
        displace = RandomDisplacement(stepsize=stepsize, random_gen=self.rng)

        AdaptiveStepsize.__init__(self, displace, interval=interval, verbose=disp)

    def get_rng(self):
        return self.rng

class BoundedRandomDisplacement(RandomDisplacement):
    def __init__(self, **kwargs):
        RandomDisplacement.__init__(self, **kwargs)

class CustomTakeStep(AdaptiveStepsize):
    def __init__(self, stepsize=0.5, interval=50, disp=False, seed=None):

        # set up the np.random generator
        self.rng = check_random_state(seed)
        displace = RandomDisplacement(stepsize=stepsize, random_gen=self.rng)

        AdaptiveStepsize.__init__(self, displace, interval=interval, verbose=disp)

    def get_rng(self):
        return self.rng

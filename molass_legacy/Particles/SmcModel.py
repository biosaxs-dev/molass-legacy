"""
    Particles.SmcModel.py

    Copyright (c) 2024, SAXS Team, KEK-PF    
"""
from particles import smc_samplers as ssp

class SmcModel(ssp.StaticModel):
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def logpyt(self, theta, t):  # density of Y_t given theta and Y_{0:t-1}
        # 
        return stats.norm.logpdf(self.data[t], loc=theta['mu'],
                                 scale = theta['sigma']) 
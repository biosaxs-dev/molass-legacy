# coding: utf-8
"""
    Pomegranate.EghDistribution.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from pomegranate import *
from molass_legacy.Peaks.ElutionModels import egh_pdf, compute_egh_params
import molass_legacy.KekLib.DebugPlot as plt

class EghDistribution:
    def __init__(self, tR, sigma, tau=0):
        self.name = 'EghDistribution'
        self.frozen = False
        self.tR = tR
        self.sigma = sigma
        self.tau = tau
        self.parameters = (self.tR, self.sigma, self.tau)
        self.d = 1
        self.summaries = numpy.zeros(4)

    def to_dict(self):
        return {
            'class' : 'Distribution',
            'name'  : self.name,
            'parameters' : self.parameters,
            'frozen' : self.frozen
        }

    def probability(self, X):
        X = X.reshape(X.shape[0])
        return egh_pdf(X, self.tR, self.sigma, self.tau)

    def log_probability(self, X):
        X = X.reshape(X.shape[0])
        return np.log(egh_pdf(X, self.tR, self.sigma, self.tau))

    def summarize(self, X, w=None):
        if w is None:
            w = numpy.ones(X.shape[0])

        X = X.reshape(X.shape[0])
        self.summaries[0] += w.sum()
        w_ = w/self.summaries[0]
        self.summaries[1] += X.dot(w_)
        self.summaries[2] += ((X - self.summaries[1])** 2).dot(w_)
        self.summaries[3] += ((X - self.summaries[1])** 3).dot(w_)

    def from_summaries(self, inertia=0.0):
        init_params = (self.summaries[1], np.sqrt(self.summaries[2]), 0)
        self.tR, self.sigma, tau = compute_egh_params(init_params, self.summaries[1:])
        moderate_tau = min(self.sigma, max(-self.sigma, tau))
        self.tau = moderate_tau
        self.parameters = (self.tR, self.sigma, moderate_tau)
        self.clear_summaries()

    def clear_summaries(self, inertia=0.0):
        self.summaries = numpy.zeros(4)

    @classmethod
    def from_samples(cls, X, weights=None):
        d = EghDistribution(0, 1)
        d.summarize(X, weights)
        d.from_summaries()
        return d

    @classmethod
    def blank(cls):
        return EghDistribution(0, 1)

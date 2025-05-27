# coding: utf-8
"""
    Rgg.RggEghDistribution.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
import scipy.stats as stats
import seaborn as sns
from pomegranate import GaussianKernelDensity
from .RggUtils import normal_pdf
from .KernelDensityFix import UniformKernelDensity
from molass_legacy.Peaks.ElutionModels import egh_pdf, compute_egh_params

import molass_legacy.KekLib.DebugPlot as plt

USE_KERNEL_DENSITY = True
TRIM_IMPOSSILE = False
KD_BANDWIDTH_RATIO = 1
# KD_DISTRIBUTION = GaussianKernelDensity
KD_DISTRIBUTION = UniformKernelDensity
VERY_SMALL_VALUE = 1e-10
USE_SIMPLE_PRODUCT = False

class RggEghDistribution:
    def __init__(self, tR, sigma, tau=0, rg=30, use_copula=True):
        print("use_copula=", use_copula)
        self.name = 'RggEghDistribution'
        self.frozen = False
        self.tR = tR
        self.sigma = sigma
        self.tau = tau
        self.rg = rg
        self.rg_sigma = KD_BANDWIDTH_RATIO
        self.parameters = (self.tR, self.sigma, self.tau, self.rg, self.rg_sigma)
        if USE_KERNEL_DENSITY:
            self.rg_dist = KD_DISTRIBUTION([self.rg], bandwidth=self.rg_sigma)

        self.d = 2
        self.summaries = np.zeros(6)
        if use_copula:
            self.norm_ppf = stats.distributions.norm().ppf
            self.copula = stats.multivariate_normal(mean=[0, 0], cov=[[1., 0.1], 
                                                                      [0.1, 1.]])
        self.use_copula = use_copula
        self.debug_weights = False
        self.debug_probability = False
        self.update_area_params()

    def to_dict(self):
        return {
            'class' : 'Distribution',
            'name'  : self.name,
            'parameters' : self.parameters,
            'frozen' : self.frozen
        }

    def update_area_params(self):
        # self.egh_limits = (self.tR - self.sigma*3, self.tR + self.sigma*3)
        self.rg_limits = (self.rg - self.rg_sigma, self.rg + self.rg_sigma)
        self.ehg_scale = 1/(self.rg_sigma*2)

    def probability(self, X):
        if self.use_copula:
            P0 = egh_pdf(X[:,0], self.tR, self.sigma, self.tau)
            if USE_KERNEL_DENSITY:
                X1 = X[:,1]
                P1 = self.rg_dist.probability(X1)
                if USE_SIMPLE_PRODUCT:
                    return P0*P1

                if TRIM_IMPOSSILE:
                    invalid = np.logical_or(X1 < self.rg_limits[0], X1 > self.rg_limits[1])
                    P0[invalid] = 0
            else:
                P1 = normal_pdf(X[:,1], self.rg, self.rg_sigma)
            N0 = self.norm_ppf(P0)
            N1 = np.zeros(X.shape[0])
            positive = P1 > VERY_SMALL_VALUE
            N1[positive] = self.norm_ppf(P1[positive])
            if False:
                plt.push()
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,7))
                ax1.plot(X[:,0])
                ax2.plot(X[:,1])
                plt.show()
                plt.pop()
                plt.push()
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,7))
                ax1.plot(P0, P1)
                ax2.plot(N0, N1)
                plt.show()
                plt.pop()
            P_ = self.copula.pdf(np.array([N0, N1]).T)
        else:
            P0 = egh_pdf(X[:,0], self.tR, self.sigma, self.tau)
            X1 = X[:,1]
            P1 = self.rg_dist.probability(X1)
            invalid = np.logical_or(X1 < self.rg_limits[0], X1 > self.rg_limits[1])
            P0[invalid] = 0
            P_ = self.ehg_scale * P0

        if self.debug_probability:
            plt.push()
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            bb_ = np.zeros(X.shape[0])
            ax.bar3d(X[:,0], X[:,1], bb_, 1, 0.1, P_, shade=True, edgecolor='green')
            ax.set_ylim(0, 50)
            fig.tight_layout()
            self.debug_probability = plt.show()
            plt.pop()
        return P_

    def log_probability(self, X):
        p = self.probability(X)
        p[p < VERY_SMALL_VALUE] = VERY_SMALL_VALUE
        return np.log(p)

    def summarize(self, X, w=None):
        if w is None:
            w = numpy.ones(X.shape[0])

        if self.debug_weights:
            plt.push()
            fig, ax = plt.subplots()
            ax.plot(w)
            ret = plt.show()
            plt.pop()
            self.debug_weights = ret

        X_ = X[:,0]
        self.summaries[0] += w.sum()
        if np.isnan(self.summaries[0]):
            raise StopIteration("summaries have included NaN's")

        w_ = w/self.summaries[0]
        self.summaries[1] += X_.dot(w_)
        self.summaries[2] += ((X_ - self.summaries[1])** 2).dot(w_)
        self.summaries[3] += ((X_ - self.summaries[1])** 3).dot(w_)
        X1 = X[:,1]
        self.summaries[4] += X1.dot(w_)
        self.summaries[5] += ((X1 - self.summaries[4])**2).dot(w_)
        print("self.summaries=", self.summaries)

    def from_summaries(self, inertia=0.0):
        init_params = (self.summaries[1], np.sqrt(self.summaries[2]), 0)
        self.tR, self.sigma, tau = compute_egh_params(init_params, self.summaries[1:4])
        moderate_tau = min(self.sigma, max(-self.sigma, tau))
        self.tau = moderate_tau
        self.rg = self.summaries[4]
        self.rg_sigma = np.sqrt(self.summaries[5])
        if USE_KERNEL_DENSITY:
            self.rg_dist = KD_DISTRIBUTION([self.rg], bandwidth=self.rg_sigma*KD_BANDWIDTH_RATIO)
        self.parameters = (self.tR, self.sigma, moderate_tau, self.rg, self.rg_sigma)
        print("self.parameters=", self.parameters)
        self.update_area_params()
        self.clear_summaries()

    def clear_summaries(self, inertia=0.0):
        self.summaries = np.zeros(6)

    @classmethod
    def from_samples(cls, X, weights=None):
        d = RggEghDistribution(100, 30, 0, 23)
        d.summarize(X, weights)
        d.from_summaries()
        return d

    @classmethod
    def blank(cls):
        return RggEghDistribution(100, 30, 0, 23)

def demo():
    x = np.linspace(0, 400, 100)
    dx = (x[1] - x[0])/4
    y = np.linspace(0, 50, 100)
    dy = (y[1] - y[0])/4
    xx, yy = np.meshgrid(x, y)

    xx_ = xx.flatten()
    yy_ = yy.flatten()
    X = np.array([xx_, yy_]).T
    bb_ = np.zeros(len(xx_))

    fig = plt.figure(figsize=(21,7))
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132, projection='3d')
    ax2 = fig.add_subplot(133, projection='3d')
    d1 = RggEghDistribution(200, 30, tau=10, rg=30, use_copula=True)
    d2 = RggEghDistribution(200, 30, tau=10, rg=30, use_copula=False)

    ax0.plot(x, egh_pdf(x, 200, 30, 10))

    for ax, d in [(ax1, d1), (ax2, d2)]:
        dz = d.probability(X)
        print("dz.shape=", dz.shape)
        ax.set_xlim(x[0], x[-1])
        ax.bar3d(xx_, yy_, bb_, dx, dy, dz, shade=True, edgecolor='green')

    fig.tight_layout()
    plt.show()

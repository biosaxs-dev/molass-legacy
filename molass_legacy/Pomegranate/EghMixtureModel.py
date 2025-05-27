# coding: utf-8
"""
    Pomegranate.EghMixtureModel.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from scipy.interpolate import UnivariateSpline
from molass_legacy.KekLib.SciPyCookbook import smooth
from pomegranate import *
from molass_legacy.Peaks.ElutionModels import egh_pdf, compute_egh_params
from Prob.GaussianMixture import hist_to_source
from .EghDistribution import EghDistribution
import molass_legacy.KekLib.DebugPlot as plt

def sort_dist_params_lists(distributions, weights):
        components = sorted(list(zip(distributions, weights)), key=lambda x: x[0].parameters[0])
        distributions, weights = zip(*components)
        return distributions, weights

class FitResult:
    def __init__(self, model):
        distributions = model.distributions
        weights = np.exp(model.weights)     # note that weights have been converted to log
        distributions, weights = sort_dist_params_lists(distributions, weights)
        self.model = model = GeneralMixtureModel(distributions, weights)
        self.params_list = [d.parameters for d in distributions]

class EghMixtureModel:
    def __init__(self, x, y, num_components=None):
        self.logger = logging.getLogger(__name__)
        sy = smooth(y)
        self.X = np.expand_dims(hist_to_source(x, sy),1)
        self.num_components = num_components

    def fit(self, init_result=None, peaks=None):
        if init_result is None and peaks is None:
            model = GeneralMixtureModel.from_samples(EghDistribution, self.num_components, self.X)
        else:
            if init_result is None:
                distributions,weights = self.get_init_params_from_peaks(peaks)
            else:
                distributions = init_result.model.distributions
                weights = init_result.model.weights
            model = GeneralMixtureModel(distributions, weights)
            model.fit(self.X)
        return FitResult(model)

    def get_components(self, fit_result, x, y):
        model = fit_result.model
        ty = np.zeros(len(x))
        cy_list = []
        for w, dist in zip(np.exp(model.weights), model.distributions):
            tR, sigma, tau = dist.parameters
            print(w, tR, sigma, tau)
            cy = w * egh_pdf(x, tR, sigma, tau)
            cy_list.append(cy)
            ty += cy

        scale = np.max(y)/np.max(ty)
        return [scale*cy for cy in cy_list], scale*ty

    def get_init_params_from_peaks(self, rb_peaks):
        distributions = []
        heights = []
        x = rb_peaks.x
        y = rb_peaks.y
        peaks = rb_peaks.get_peaks()
        for peak in peaks:
            ls, pt, rs = peak
            tR = x[pt]
            sigma = (x[pt] - x[ls] + x[rs] - x[pt])/2
            tau = 0
            dist = EghDistribution(tR, sigma, tau)
            distributions.append(dist)
            heights.append(y[pt])

        distributions, weights = sort_dist_params_lists(distributions, heights)
        weights = np.array(weights)
        weights /= np.sum(weights)

        num_peaks = len(peaks)
        num_components = self.num_components
        num_sub_peaks = num_components - num_peaks
        assert num_sub_peaks >= 0
        if num_sub_peaks == 0:
            return distributions, weights

        spline = UnivariateSpline(x, y, s=0, ext=3)
        model = GeneralMixtureModel(distributions, weights)
        model.fit(self.X)

        taus = [d.parameters[2] for d in model.distributions]
        if abs(taus[0]) > abs(taus[-1]):
            sides = [-1, 1]
        else:
            sides = [1, -1]

        distributions = list(distributions)

        for side in sides:
            if side < 0:
                # left side
                i = 0
                dist = model.distributions[i]
                parameters = dist.parameters
                peak_tR,peak_sigma, peak_tau = parameters
                tR = peak_tR - peak_sigma*2.5
                sigma = peak_sigma/4
                tau = peak_tau/4
            else:
                # right side
                i = -1
                dist = model.distributions[i]
                parameters = dist.parameters
                peak_tR,peak_sigma, peak_tau = parameters
                tR = peak_tR + peak_sigma*2.5
                sigma = peak_sigma/4
                tau = peak_tau/4
            h = spline(tR)
            sub_dist = EghDistribution(tR, sigma, tau)
            distributions.insert(i, sub_dist)
            heights.insert(i, h)
            num_peaks += 1
            if num_peaks == num_components:
                break

        distributions, weights = sort_dist_params_lists(distributions, heights)
        weights = np.array(weights)
        weights /= np.sum(weights)

        return distributions, weights

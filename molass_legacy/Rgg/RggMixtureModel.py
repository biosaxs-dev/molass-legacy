# coding: utf-8
"""
    Rgg.RggMixtureModel.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize
from molass_legacy.KekLib.SciPyCookbook import smooth
from Prob.GaussianMixture import gaussian_pdf
from molass_legacy.Peaks.ElutionModels import egh_pdf, compute_egh_params
from pomegranate import *
from .RggEghDistribution import RggEghDistribution
from Rgg.RggUtils import convert_to_probabilitic_data
from .Callbacks import VisualizationtCallback
import molass_legacy.KekLib.DebugPlot as plt

USE_RG = True
if not USE_RG:
    from Prob.GaussianMixture import hist_to_source

USE_GAUSSIAN_MIXTURE_MODEL = False
if USE_GAUSSIAN_MIXTURE_MODEL:
    from sklearn.mixture import GaussianMixture

def sort_dist_params_lists(distributions, weights):
        components = sorted(list(zip(distributions, weights)), key=lambda x: x[0].parameters[0])
        distributions, weights = zip(*components)
        return distributions, weights

class FitResult:
    def __init__(self, model):
        if USE_GAUSSIAN_MIXTURE_MODEL:
            """
            verify the differences of model parameters between 2D (USE_RG) and 1D (not USE_RG)
            """ 
            # print(dir(model))
            # print("model.means_=", model.means_)
            # print("model.covariances_", model.covariances_)
            self.weights = model.weights_
            if USE_RG:
                self.params_list = [(mu, std) for mu, std in model.means_]
            else:
                self.params_list = [(mu[0], np.sqrt(var)) for mu, var in zip(model.means_, model.covariances_)]
        else:
            distributions = model.distributions
            weights = np.exp(model.weights)     # note that weights have been converted to log
            distributions, weights = sort_dist_params_lists(distributions, weights)
            self.weights = weights
            self.model = model = GeneralMixtureModel(distributions, weights)
            self.params_list = [d.parameters for d in distributions]

    def get_params_for_refiner(self):
        ret_params = []
        for w, params in zip(self.weights, self.params_list):
            ret_params.append((w, *params[0:4]))
        return ret_params

class RggMixtureModel:
    def __init__(self, X, num_components=None):
        self.logger = logging.getLogger(__name__)
        self.X = X
        self.num_components = num_components
        if USE_GAUSSIAN_MIXTURE_MODEL:
            cv_type = "spherical"
            self.model = GaussianMixture(n_components=num_components, covariance_type=cv_type)

    def fit(self, init_result=None, peaks=None):
        for k in range(100):
            try:
                return self.fit_impl(init_result=init_result, peaks=peaks)
            except StopIteration as exc:
                # this does not seem to work due to
                # the internal exception handling of GeneralMixtureModel
                print([k], "retrying fit")

    def fit_impl(self, init_result=None, peaks=None):
        if USE_GAUSSIAN_MIXTURE_MODEL:
            self.model.fit(self.X)
            return FitResult(self.model)

        model = None
        if peaks is None:
            if init_result is None:
                model = GeneralMixtureModel.from_samples(RggEghDistribution, self.num_components, self.X)
            else:
                distributions = init_result.model.distributions
                weights = init_result.model.weights
        else:
            distributions, weights = self.get_init_params_from_peaks(peaks)

        if model is None:
            model = GeneralMixtureModel(distributions, weights)

        _ = model.fit(self.X, callbacks=[VisualizationtCallback()])
        return FitResult(model)

    def get_components(self, fit_result, x, y, simple_scale=False):
        ty = np.zeros(len(x))
        cy_list = []
        for w, params in zip(np.exp(fit_result.weights), fit_result.params_list):
            if USE_GAUSSIAN_MIXTURE_MODEL:
                mu, sigma = params
                cy = w * gaussian_pdf(x, mu, sigma)
            else:
                tR, sigma, tau, rg, _ = params
                print(tR, sigma, tau, rg)
                cy = w * egh_pdf(x, tR, sigma, tau)
            cy_list.append(cy)
            ty += cy

        scale = np.max(y)/np.max(ty)
        if simple_scale:
            return [scale*cy for cy in cy_list], scale*ty

        y_positive = y.copy()
        y_positive[y < 0] = 0

        cy_matrix = np.array(cy_list).T

        def obj_func(scales):
            scaled_matrix = cy_matrix * scales
            if False:
                plt.push()
                fig, ax = plt.subplots()
                ax.plot(x, y)
                ty = np.zeros(len(x))
                for cy in scaled_matrix.T:
                    ax.plot(x, cy, ':')
                    ty += cy
                ax.plot(x, ty, ':', color='red')
                fig.tight_layout()
                plt.show()
                plt.pop()
            return np.sum((np.sum(scaled_matrix, axis=1) - y_positive)**2)

        init_scales = np.ones(len(cy_list)) * scale

        result = minimize(obj_func, init_scales)

        ret_ty = np.zeros(len(x))
        ret_cy_list = []
        for scale, cy in zip(result.x, cy_list):
            cy_ = scale*cy
            ret_cy_list.append(cy_)
            ret_ty += cy_

        return ret_cy_list, ret_ty

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

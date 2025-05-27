# coding: utf-8
"""
    Pomegranate.MappingGmmDecomposer.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
from time import time
import numpy as np
from itertools import combinations
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline
import molass_legacy.KekLib.DebugPlot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from molass_legacy.Peaks.RobustPeaks import RobustPeaks
from molass_legacy.Peaks.ElutionModels import egh
from Prob.GaussianMixture import hist_to_source
from Prob.EghMixture import EghMixture
import OurStatsModels as sm

APPLY_PARAM_CONSTRAINTS = True
USE_WLS = True

class MappingGmmDecomposer:
    def __init__(self, uv_rp, xr_rp, num_components=None):
        self.uv_rp = uv_rp
        self.xr_rp = xr_rp
        self.uv_x = uv_rp.x
        self.uv_y = uv_rp.y
        self.xr_x = xr_rp.x
        self.xr_y = xr_rp.y
        self.num_components = num_components

    def decompose_separatley(self):
        from .EghMixtureModel import EghMixtureModel

        num_components = self.num_components
        x0 = self.uv_x
        y0 = self.uv_y
        mm0 = EghMixtureModel(x0, y0, num_components)
        x1 = self.xr_x
        y1 = self.xr_y
        mm1 = EghMixtureModel(x1, y1, num_components)

        best_pair = None
        best_r_value = None
        last_max_r_value = None
        for k in range(5):
            results0 = []
            if best_pair is None or True:
                res0, res1 = None, None
            else:
                res0, res1 = best_pair
            # print([k], res0, res1)
            print([k], "----")
            for i in range(2):
                results0.append(mm0.fit(init_result=res0))

            results1 = []
            for i in range(2):
                results1.append(mm1.fit(init_result=res1))

            max_pair, max_r_value, _ = self.get_best_combination(results0, results1)
            if last_max_r_value is not None and last_max_r_value > 0.95:
                if max_r_value < last_max_r_value:
                    pass

            if best_r_value is None or max_r_value > best_r_value:
                best_r_value = max_r_value
                best_pair = max_pair

            last_max_r_value = max_r_value

        res0, res1 = best_pair

        cy_list0, ty0 = mm0.get_components(res0, x0, y0)
        cy_list1, ty1 = mm1.get_components(res1, x1, y1)

        plot_mapped_states([
            [x0, y0, cy_list0, ty0],
            [x1, y1, cy_list1, ty1],
            ])

    def get_best_combination(self, results0, results1):

        max_r_value = None
        max_pair = None
        r_values = []
        for i, res0 in enumerate(results0):
            x = [params[0] for params in res0.params_list]
            for j, res1 in enumerate(results1):
                y = [params[0] for params in res1.params_list]
                r_value = linregress(x, y)[2]
                print((i,j), r_value)
                if max_r_value is None or r_value > max_r_value:
                    max_r_value = r_value
                    max_pair = (i, j)
                    r_values.append(r_value)

        i, j = max_pair
        return (results0[i], results1[j]), max_r_value, np.average(r_values)

def linear_regression(x, y, weights=None):
    if weights is None:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
    else:
        X = sm.add_constant(x)
        mod = sm.WLS(y, X, weights=weights)
        res = mod.fit()
        intercept, slope = res.params
    return slope, intercept

def plot_mapped_states(components_list, scale=False, step=None, mappinn_info=None):
    plt.push()
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(21,7))

    step_str = "" if step is None else " step %d" % step
    fig.suptitle("UV-Xray Decomposed Mapping Figures (Plan)%s" % step_str, fontsize=30)
    ax1.set_title("UV Elution", fontsize=20)
    ax2.set_title("Xray Elution", fontsize=20)
    ax3.set_title("Mapped Elution", fontsize=20)

    for ax, components, color in zip([ax1, ax2], components_list, ['C0', 'C1']):
        x, y, cy_list, ty = components
        if scale:
            s = np.max(y)/np.max(ty)
        else:
            s = 1

        ax.plot(x, y, color=color)
        ax.plot(x, s*ty, ':', color='red')
        for cy in cy_list:
            ax.plot(x, s*cy, ':')

    if mappinn_info is not None:
        mapping_params, uv_params, uv_pi, xr_params, xr_pi = mappinn_info
        slope, intercept = mapping_params
        x0, y0, cy_list0, ty0 = components_list[0]
        x1, y1, cy_list1, ty1 = components_list[1]
        ax3.plot(x1, y1, color='C1', label='Xray')
        spline = UnivariateSpline(x0, y0, s=0, ext=3)
        x_ = x1*slope + intercept
        y_ = spline(x_)
        scale = np.max(y1)/np.max(y_)
        my = y_*scale
        ax3.plot(x1, my, color='gray', alpha=0.2, label='simply mapped UV')
        smy = make_mapped_uv(x1, my, uv_params, uv_pi, xr_params, xr_pi)
        ax3.plot(x1, smy, ':', color='C0', label='scaling mapped UV')
        ax3.legend(fontsize=16, loc="upper left")

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.show()
    plt.pop()

def make_mapped_uv(x, mapped_y, uv_params, uv_pi, xr_params, xr_pi):
    w = np.zeros(len(mapped_y))
    for uv_p, xr_p, params in zip(uv_pi, xr_pi, xr_params):
        w += egh(x, xr_p/uv_p, *params)
    smy = mapped_y*w
    scale = np.max(mapped_y)/np.max(smy)
    return smy*scale

def spike_demo():
    x0 = np.arange(1000)
    y0 = np.zeros(len(x0))
    for params in [
            [1.0, 401, 50, 5],
            [0.7, 602, 60, 10],
            ]:
        y0 += egh(x0, *params)

    x1 = np.arange(500)
    y1 = np.zeros(len(x1))
    for params in [
            # [0.6, 200, 40, 10],
            [0.6, 200, 30, 5],
            [1.0, 300, 30, 10],
            ]:
        y1 += egh(x1, *params)

    spike_demo_impl(x0, y0, x1, y1, num_components=2)

def spike_demo_impl(x0, y0, x1, y1, num_components=None):
    rb0 = RobustPeaks(x0, y0)
    rb1 = RobustPeaks(x1, y1)
    md = MappingGmmDecomposer(rb0, rb1, num_components=num_components)
    md.decompose_separatley()
    # md.decompose()


def spike_demo_real(in_folder):
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    sp = StandardProcedure()
    sp.load(in_folder, debug=False)
    sd = sp.get_sd()
    uv_curve = sd.get_uv_curve()
    xr_curve = sd.get_xray_curve()
    spike_demo_impl(uv_curve.x, uv_curve.y, xr_curve.x, xr_curve.y, num_components=4)

# coding: utf-8
"""
    FewerPointMapper.py

    Copyright (c) 2021-2022, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from scipy import stats
from scipy.interpolate import UnivariateSpline
from matplotlib.gridspec import GridSpec
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
from molass_legacy.Elution.CurveUtils import simple_plot
from molass_legacy.Peaks.RobustCurve import RobustCurve, plot_curve, gaussian
from molass_legacy.Trimming.MatchingPeaks import compute_matching_indeces

class FewerPointMapper:
    def __init__(self, rb_curve1, rb_curve2):
        self.curves = [rb_curve1, rb_curve2]
        v = np.array([pt[0] for pt in rb_curve1.get_peak_tops()])
        u = np.array([pt[0] for pt in rb_curve2.get_peak_tops()])
        scale = len(rb_curve2.x)/len(rb_curve1.x)
        indeces = compute_matching_indeces(len(rb_curve1.x), v*scale, u)
        j_ = []
        i_ = []
        for j, i in zip(indeces[0], indeces[1]):
            if j is None or i is None:
                continue
            j_.append(j)
            i_.append(i)
        slope, intercept, r_value, p_value, std_err = stats.linregress(u[i_], v[j_])
        self.params = (slope, intercept)
        self.spline = UnivariateSpline(rb_curve1.x, rb_curve1.y, s=0, ext=3)
        self.matching_indeces = np.array([j_, i_])

    def get_params(self):
        return self.params

    def get_simply_mapped_y(self, u):
        slope, intercept = self.params
        return self.spline(u*slope + intercept)

    def get_scale_mapped_y(self, u):
        simple_y  = self.get_simply_mapped_y(u)

        peak_model_list = []
        for k, curve in enumerate(self.curves):
            models = curve.get_peak_models()
            matched = []
            for i in self.matching_indeces[k,:]:
                matched.append(models[i])
            peak_model_list.append(matched)

        slope, intercept = self.params
        mapping_params = []
        tw = np.zeros(len(u))
        for model1, model2 in zip(*peak_model_list):
            if len(model1.params) == 3:
                h1, m1, s1 = model1.params
                h2, m2, s2 = model2.params
                m1_ = (m1 - intercept)/slope
                s1_ = s1/slope
                w = model2.func(u, h2, m1_, s1_)
            elif len(model1.params) == 4:
                h1, m1, s1, t1 = model1.params
                h2, m2, s2, t2 = model2.params
                m1_ = (m1 - intercept)/slope
                s1_ = s1/slope
                t1_ = t1/slope
                w = model2.func(u, h2, m1_, s1_, t1_)
            else:
                assert False
            mapping_params.append((w, h2/h1))
            tw += w

        mapped_y = np.zeros(len(u))
        for w, scale in mapping_params:
            mapped_y += simple_y * w/tw *scale

        return mapped_y

def demo(sd, peak_model_class, scale_mapped=True):
    from DataUtils import get_in_folder
    from molass_legacy.Peaks.RobustCurve import LIFTED_BASE, MINIMIZE_METHOD

    peak_model_name = peak_model_class.name
    if peak_model_name == 'Hyperbola':
        scale_mapped = False

    uv_curve = sd.get_uv_curve()
    xr_curve = sd.get_xray_curve()

    rb_curves = []
    for curve in [uv_curve, xr_curve]:
        x = curve.x
        y = curve.y
        rbc = RobustCurve(x, y, peak_model=peak_model_class)
        rb_curves.append(rbc)

    fpm = FewerPointMapper(*rb_curves)

    plt.push()
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(21,7))

    text_args = []
    if LIFTED_BASE:
        text_args.append("lifted base")
    method = "default method" if MINIMIZE_METHOD is None else MINIMIZE_METHOD
    text_args.append(method)
    paren_text = "(%s)" % ', '.join(text_args)

    fig.suptitle("Fewer Point Mapping with Locally fitted %s for %s  %s" % (peak_model_name, get_in_folder(), paren_text), fontsize=20)
    ax1.set_title("UV Elution", fontsize=16)
    ax2.set_title("Xray Elution", fontsize=16)
    how = 'Scale' if scale_mapped else 'Simply'
    ax3.set_title("%s mapped Elution" % how, fontsize=16)

    plot_curve(ax1, rb_curves[0], color='C0')
    plot_curve(ax2, rb_curves[1], color='C1')

    x = xr_curve.x
    y = xr_curve.y


    if scale_mapped:
        my = fpm.get_scale_mapped_y(x)
        ax_ = ax3
    else:
        my = fpm.get_simply_mapped_y(x)
        axt = ax3.twinx()
        axt.grid(False)
        ax_ = axt

    ax3.plot(x, y, color='C1')
    ax_.plot(x, my, color='C0')

    for ax in [ax1, ax2, ax3]:
        ax.legend()

    fig.tight_layout()
    plt.show()
    plt.pop()

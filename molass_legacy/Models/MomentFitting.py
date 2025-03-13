"""
    Models.MomentFitting.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Elution.CurveUtils import simple_plot
from molass_legacy.Peaks.ElutionModels import compute_moments
from molass_legacy.Models.Stochastic.SDM import SDM

def sdm_guess_impl(model, x, y, moments):
    def objective(p):
        pass

def fit_a_component(model, x, y):
    M = compute_moments(x, y)
    sdm_guess_impl(model, x, y, M)

def spike(caller):
    model = SDM()
    sd = caller.serial_data
    uv_curve = sd.get_uv_curve()
    xr_curve = sd.get_xr_curve()
    x = xr_curve.x
    y = xr_curve.y
    fit_a_component(model, x, y)

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
        fig.suptitle("MomentFitting Spike", fontsize=20)
        simple_plot(ax1, uv_curve, color="blue")
        simple_plot(ax2, xr_curve, color="orange")
        fig.tight_layout()
        plt.show()
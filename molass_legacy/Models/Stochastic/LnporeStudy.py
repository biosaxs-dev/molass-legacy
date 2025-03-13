"""
    Models.Stochastic.LnporeStudy.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from matplotlib.widgets import Slider, Button
from molass_legacy.Models.Stochastic.MomentUtils import compute_egh_moments
from SecTheory.BasicModels import robust_single_pore_pdf
from importlib import reload
import Models.Stochastic.LognormalPoreFunc
reload(Models.Stochastic.LognormalPoreFunc)
from molass_legacy.Models.Stochastic.LognormalPoreFunc import distr_func, lognormal_pore_func
from molass_legacy.Models.Stochastic.LognormalUtils import compute_mu_sigma, compute_mode, compute_stdev
from molass_legacy.Models.Stochastic.SecModelUtils import DEFAULT_SIGMA, SIGMA_LOWER_BOUND
from molass_legacy.Models.Stochastic.ParamLimits import MAX_PORESIZE

def compute_curves_from_monopore_params(x, params, rgs):
    N, T, x0, me, mp, poresize = params[0:6]
    scales = params[6:]
    cy_list = []
    for k, (rg, scale) in enumerate(zip(rgs, scales)):
        rho = min(1, rg/poresize)
        ni_ = N * (1 - rho)**me
        ti_ = T * (1 - rho)**mp
        cy = scale*robust_single_pore_pdf(x - x0, ni_, ti_)
        cy_list.append(cy)
    ty = np.sum(cy_list, axis=0)
    return cy_list, ty, poresize

def compute_curves_from_lnpore_params(x, params, rgs):
    N, T, x0, me, mp, poresize = params[0:6]
    mu, sigma = compute_mu_sigma(poresize, poresize*0.2)
    sigma = max(sigma, SIGMA_LOWER_BOUND)
    scales = params[6:]
    cy_list = []
    for k, (rg, scale) in enumerate(zip(rgs, scales)):
        cy = lognormal_pore_func(x, scale, N, T, me, mp, mu, sigma, rg, x0)
        cy_list.append(cy)
    ty = np.sum(cy_list, axis=0)
    return cy_list, ty, mu, sigma

def study(x, y, baseline, model, peaks, peak_rgs, props):
    from importlib import reload
    import Models.Stochastic.RoughGuess
    reload(Models.Stochastic.RoughGuess)
    import Models.Stochastic.MonoporeGuess
    reload(Models.Stochastic.MonoporeGuess)
    from molass_legacy.Models.Stochastic.RoughGuess import guess_monopore_params_roughtly
    from molass_legacy.Models.Stochastic.MonoporeGuess import guess_monopore_params_using_moments

    moments_list = compute_egh_moments(peaks)
    rough_monopore_params = guess_monopore_params_roughtly(x, y, model, peaks, peak_rgs, props, moments_list, debug=False)
    print("monopore_params = ", rough_monopore_params)
    if rough_monopore_params is None:
        return
 
    better_monopore_params = guess_monopore_params_using_moments(x, y, moments_list, peak_rgs, props, rough_monopore_params)
    print("better_monopore_params=", better_monopore_params)
    if better_monopore_params is None:
        return

    def plot_elution_curves(ax, title, mnp_cy_list, mnp_ty, lnp_cy_list, lnp_ty):
        ax.set_title(title, fontsize=16)
        ax.plot(x, y, label="data")
        for k, (mnp_cy, lnp_cy) in enumerate(zip(mnp_cy_list, lnp_cy_list)):
            color = "C%d" % (k+1)
            ax.plot(x, mnp_cy, ":", color=color)
            ax.plot(x, lnp_cy, "-", color=color)
        ax.plot(x, mnp_ty, ":", color="red")
        ax.plot(x, lnp_ty, "-", color="red")

    rv = np.arange(800)
    def plot_psd(ax, title, poresize, mu, sigma):
        ax.set_title(title, fontsize=16)
        ax.plot(rv, distr_func(rv, mu, sigma))
        ax.axvline(poresize, color="green")

    mnp_cy_list1, mnp_ty1, poresize1 = compute_curves_from_monopore_params(x, rough_monopore_params, peak_rgs)
    lnp_cy_list1, lnp_ty1, mu1, sigma1 = compute_curves_from_lnpore_params(x, rough_monopore_params, peak_rgs)
    mnp_cy_list2, mnp_ty2, poresize2 = compute_curves_from_monopore_params(x, better_monopore_params, peak_rgs)
    lnp_cy_list2, lnp_ty2, mu2, sigma2 = compute_curves_from_lnpore_params(x, better_monopore_params, peak_rgs)

    with plt.Dp(button_spec=["OK", "Cancel"]):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))
        fig.suptitle("Lognormal Pore Model Study with Differenent Monopore Params", fontsize=20)
        ax1, ax2 = axes[0,:]
        plot_elution_curves(ax1, "Rough Monopore Elution", mnp_cy_list1, mnp_ty1, lnp_cy_list1, lnp_ty1)
        plot_elution_curves(ax2, "Better Monopore Elution", mnp_cy_list2, mnp_ty2, lnp_cy_list2, lnp_ty2)
        ax3, ax4 = axes[1,:]
        plot_psd(ax3, "Rough Monopore PSD", poresize1, mu1, sigma1)
        plot_psd(ax4, "Better Monopore PSD", poresize2, mu2, sigma2)
        fig.tight_layout()
        ret = plt.show()
    if not ret:
        return
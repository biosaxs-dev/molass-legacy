"""
    Models.Stochastic.MomentUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
import molass_legacy.KekLib.DebugPlot as plt
from SecTheory.BasicModels import robust_single_pore_pdf
from molass_legacy.Peaks.ElutionModels import compute_moments_from_egh_params
from molass_legacy.Models.Stochastic.LognormalPoreFunc import lognormal_pore_func

def compute_egh_moments(peaks):
    moments_list = []
    for params in peaks:
        tr, s, t = params[1:4]
        moments_list.append(compute_moments_from_egh_params(tr, s, t))
    return moments_list

def to_moderate_props(props):
    new_props = np.power(props, 0.5)
    return new_props/np.sum(new_props)

def moments_demo_from_rgdist(x, y, model, peaks, peak_rgs, props, monopore_params, logger=None, debug=False):
    if debug:
        from importlib import reload
        import Models.Stochastic.MonoporeGuess
        reload(Models.Stochastic.MonoporeGuess)
        import Models.Stochastic.LognormalGuess
        reload(Models.Stochastic.LognormalGuess)
    from molass_legacy.Models.Stochastic.MonoporeGuess import guess_monopore_params_using_moments
    from molass_legacy.Models.Stochastic.LognormalGuess import guess_lognormalpore_params_using_moments

    egh_moments_list = compute_egh_moments(peaks)
    if monopore_params is None:
        from importlib import reload
        import Models.Stochastic.RoughGuess
        reload(Models.Stochastic.RoughGuess)
        from molass_legacy.Models.Stochastic.RoughGuess import guess_monopore_params_roughtly
        monopore_params = guess_monopore_params_roughtly(x, y, model, peaks, peak_rgs, props, egh_moments_list)
        if logger is not None:
            logger.info("monopore_params in moments_demo_from_rgdist: %s", monopore_params)
    print("monopore_params=", monopore_params)
    if monopore_params is None:
        return

    better_monopore_params = guess_monopore_params_using_moments(x, y, egh_moments_list, peak_rgs, props, monopore_params)
    if logger is not None:
            logger.info("better_monopore_params in moments_demo_from_rgdist: %s", better_monopore_params)
    print("better_monopore_params=", better_monopore_params)
    if True:
        lognormalpore_params = guess_lognormalpore_params_using_moments(x, y, egh_moments_list, peak_rgs, props, better_monopore_params, debug=True)
    else:
        lognormalpore_params = None
    print("lognormalpore_params=", lognormalpore_params)

    with plt.Dp(button_spec=["OK", "Cancel"]):
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(ncols=4, figsize=(20,4))
        fig.suptitle("Stochastic Model Moments Study & Proof", fontsize=20)
        ax1.set_title("EGH", fontsize=16)
        ax1.plot(x, y, label='data')
        cy_list = []
        for k, params in enumerate(peaks):
            cy = model(x, params)
            cy_list.append(cy)
            ax1.plot(x, cy, ":", label='component-%d' % k)
        ty = np.sum(cy_list, axis=0)
        ax1.plot(x, ty, ":", color="red", label='model total')

        ax2.set_title("Rough Monopore", fontsize=16)

        def plot_monopore(ax, monopore_params):
            N, T, x0, me, mp, poresize = monopore_params[0:6]
            scales = monopore_params[6:]
            ax.plot(x, y, label='data')
            rho = peak_rgs/poresize
            rho[rho > 1] = 1
            cy_list = []
            for k, (r_, scale) in enumerate(zip(rho, scales)):
                ni_ = N*(1 - r_)**me
                ti_ = T*(1 - r_)**mp
                cy = scale*robust_single_pore_pdf(x - x0, ni_, ti_)
                cy_list.append(cy)
                ax.plot(x, cy, ":", label='component-%d' % k)
            ty = np.sum(cy_list, axis=0)
            ax.plot(x, ty, ":", color="red", label='model total')

        plot_monopore(ax2, monopore_params)

        ax3.set_title("Better Monopore", fontsize=16)
        plot_monopore(ax3, better_monopore_params)

        def plot_lognormalpore(ax, lognormalpore_params):
            N, T, x0, me, mp, mu, sigma = lognormalpore_params[0:7]
            scales = lognormalpore_params[7:]
            ax.plot(x, y, label='data')
            cy_list = []
            for k, (rg, scale) in enumerate(zip(peak_rgs, scales)):
                cy = lognormal_pore_func(x, scale, N, T, me, mp, mu, sigma, rg, x0)
                cy_list.append(cy)
                ax.plot(x, cy, ":", label='component-%d' % k)
            ty = np.sum(cy_list, axis=0)
            ax.plot(x, ty, ":", color="red", label='model total')

        ax4.set_title("Lognormal Pore", fontsize=16)
        if lognormalpore_params is not None:
            plot_lognormalpore(ax4, lognormalpore_params)

        fig.tight_layout()
        ret = plt.show()

    if ret:
        return better_monopore_params, lognormalpore_params
    else:
        return None
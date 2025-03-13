"""
    Models.Stochastic.LognormalGuess.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.integrate import quad
from importlib import reload
import molass_legacy.KekLib.DebugPlot as plt
from SecTheory.BasicModels import robust_single_pore_pdf
import Models.Stochastic.LognormalPoreFunc
reload(Models.Stochastic.LognormalPoreFunc)
from molass_legacy.Models.Stochastic.LognormalPoreFunc import distr_func, lognormal_pore_func
from molass_legacy.Models.Stochastic.LognormalUtils import compute_mu_sigma, compute_mode, compute_stdev
from molass_legacy.Models.ModelUtils import compute_raw_moments1, compute_area_props

def guess_lognormalpore_params_using_moments(x, y, egh_moments_list, peak_rgs, props, monopore_params, progress_cb_info=None, debug=False):
    from molass_legacy.Models.Stochastic.SecModelUtils import SIGMA_LOWER_BOUND
    import Simulative.LognormalPsd
    reload(Simulative.LognormalPsd)
    from Simulative.LognormalPsd import lognormalpore_model_interactive_impl

    # debug = True
    # print("monopore_params=", monopore_params, "peak_rgs=", peak_rgs)
    N, T, x0, me, mp, poresize = monopore_params[0:6]
    # init_sigma = DEFAULT_SIGMA
    # init_mu = np.log(poresize)
    init_mu, init_sigma = compute_mu_sigma(poresize, poresize*0.2)
    print("init_mu, init_sigma=", init_mu, init_sigma)
    init_sigma = max(init_sigma, SIGMA_LOWER_BOUND)
    print("init_mu, init_sigma=", init_mu, init_sigma)
    if debug:
        scales = monopore_params[6:]
        print("poresize=", poresize, compute_mode(init_mu, init_sigma))
        print("stdev=", compute_stdev(init_mu, init_sigma))
        with plt.Dp(button_spec=["OK", "Cancel"]):
            fig, ax = plt.subplots()
            ax.set_title("Lognormalpore vs. Monopore")
            ax.plot(x, y)
            lnp_cy_list = []
            mnp_cy_list = []
            for k, (rg, scale) in enumerate(zip(peak_rgs, scales)):
                color = "C%d" % (k+1)
                cy = lognormal_pore_func(x, scale, N, T, me, mp, init_mu, init_sigma, rg, x0)
                lnp_cy_list.append(cy)
                ax.plot(x, cy, alpha=0.3, color=color)
                rho = min(1, rg/poresize)
                ni_ = N * (1 - rho)**me
                ti_ = T * (1 - rho)**mp
                cy = scale*robust_single_pore_pdf(x - x0, ni_, ti_)
                mnp_cy_list.append(cy)
                ax.plot(x, cy, ":", color=color)
            lnp_ty = np.sum(lnp_cy_list, axis=0)
            mnp_ty = np.sum(mnp_cy_list, axis=0)
            ax.plot(x, lnp_ty, "-r", label="Lnpore")
            ax.plot(x, mnp_ty, "-b", label="Monopore")
            ax.legend()

            fig.tight_layout()
            ret = plt.show()
        if not ret:
            return

    abort = False
    use_multi_factor_fv = False
    if use_multi_factor_fv:
        raw_moments = np.array([M[0] for M in egh_moments_list])

    def mu_sigma_scales_objective(p, title=None):
        nonlocal abort
        x0_, mu, sigma = p[0:3]
        cy_list = []
        for rg, scale in zip(peak_rgs, p[3:]):
                cy = lognormal_pore_func(x, scale, N, T, me, mp, mu, sigma, rg, x0_)
                cy_list.append(cy)
        ty = np.sum(cy_list, axis=0)
        fv = np.sum((ty - y)**2)
        if use_multi_factor_fv:
            raw_moments_ = compute_raw_moments1(x, cy_list)
            props_ = compute_area_props(cy_list)
            fv = fv**2 + 0.0001*np.sum((raw_moments_ - raw_moments)**2)**2 + 0.0001*np.sum((props_ - props)**2)**2

        if False:
            title = "minimize iteration"
        if title is not None:
            print("%s fv=%.3g" % (title, fv))
            print("%s p=%s" % (title, repr(p)))
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title(title)
                ax.plot(x, y)
                for cy in cy_list:
                    ax.plot(x, cy, ":")
                ax.plot(x, ty, ":", color="red")
                fig.tight_layout()
                ret = plt.show()
            if not ret:
                abort = True
        if np.isnan(fv):
            fv = np.inf
        return fv
    
    num_components = len(peak_rgs)
    bounds = [(x0-50, x0+50), (0, None), (SIGMA_LOWER_BOUND, None)] + [(0, None)]*num_components
    init_scales = monopore_params[6:]
    init_params = np.concatenate([[x0, init_mu, init_sigma], init_scales])
    if debug:
        print("init_params =", init_params)
        mu_sigma_scales_objective(init_params, title="init debug")
        if abort:
            return
        demo_params = np.concatenate([[N, T, x0, me, mp], init_params[1:]])
        ret = lognormalpore_model_interactive_impl(x, y, demo_params, peak_rgs, title="guess_lognormalpore_params_using_moments debug")
        if not ret:
            return
        mu_sigma_scales_objective(init_params, title="before LNP minimize")
        if abort:
            return
    if progress_cb_info is None:
        callback = None
    else:
        minp, maxp, progress_cb = progress_cb_info
        max_iters = len(init_params) * 200
        call_counter = 0
        def callback_func(x, *args):
            nonlocal call_counter
            call_counter += 1
            if call_counter % 10 == 0:
                progress_cb(minp + (maxp-minp)*call_counter/max_iters)
        callback = callback_func
    res = minimize(mu_sigma_scales_objective, init_params, bounds=bounds, method="Nelder-Mead", callback=callback)
    if debug:
        print("res.x =", res.x)
        mu_sigma_scales_objective(res.x, title="after LNP minimize")
    x0, mu, sigma = res.x[0:3]
    opt_params = np.concatenate([[N, T, x0, me, mp, mu, sigma], res.x[3:]]) 
    return opt_params
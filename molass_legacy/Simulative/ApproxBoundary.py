"""
    Simulative.ApproxBoundary.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from SecTheory.BasicModels import robust_single_pore_pdf
from molass_legacy.Models.Stochastic.LognormalPoreFunc import lognormal_pore_pdf
from molass_legacy.Models.Stochastic.LognormalUtils import compute_mode, compute_stdev, compute_mu_sigma

def compute_lnp_curves(x, N, T, me, mp, mu, sigma, t0, rgs, scales):
    cy_list = []
    for rg, scale in zip(rgs, scales):
        cy = lognormal_pore_pdf(x, N, T, me, mp, mu, sigma, rg, t0)*scale
        cy_list.append(cy)
    ty = np.sum(cy_list, axis=0)
    return cy_list, ty

def compute_mnp_curves(x, N, T, me, mp, poresize, t0, rgs, scales):
    cy_list = []
    for rg, scale in zip(rgs, scales):
        rho = min(1, rg/poresize)
        ni_ = N * (1 - rho)**me
        ti_ = T * (1 - rho)**mp
        cy = robust_single_pore_pdf(x - t0, ni_, ti_)*scale
        cy_list.append(cy)
    ty = np.sum(cy_list, axis=0)
    return cy_list, ty

def compute_bounadry_ratio(x, N, T, t0, me, mp, poresize, rgs, scales, stdev_ratios=None):
    if stdev_ratios is None:
        stdev_ratios = np.arange(0.1, 1.001, 0.01)
    ty_mnp = compute_mnp_curves(x, N, T, me, mp, poresize, t0, rgs, scales)[1]
    area = np.sum(ty_mnp)
    ok_ratio = np.nan

    for j, ratio in enumerate(stdev_ratios):
        stdev = ratio * poresize
        mu_, sigma_ = compute_mu_sigma(poresize, stdev)
        ty_lnp = compute_lnp_curves(x, N, T, me, mp, mu_, sigma_, t0, rgs, scales)[1]
        dev_ratio = np.sum(np.abs(ty_lnp - ty_mnp))/area
        if dev_ratio > 0.3:
            break
        ok_ratio = ratio

    return ok_ratio

def compute_approx_boundary(x, y, params, rgs, quickly=True):
    N, T, t0, me, mp, mu, sigma = params[0:7]
    scales = params[7:]

    max_rg = np.max(rgs)
    rv = np.arange(int(round(max_rg + 1,-1)), 500, 10 if quickly else 1)
    stdev_ratio_list = np.arange(0.1, 1.001, 0.01 if quickly else 0.001)
    results = []
    for i, poresize in enumerate(rv):
        print(poresize)
        ok_ratio = compute_bounadry_ratio(x, N, T, t0, me, mp, poresize, rgs, scales, stdev_ratio_list)
        results.append(ok_ratio)

    if False:
        print("mu, sigma=", mu, sigma)
        np.savetxt("temp/boundary.dat", np.array([rv, results]).T)    

    results = np.array(results)
    return rv, results

def decide_allowance_color(rv, bv, mode, stdev_ratio):
    from bisect import bisect_right
    from scipy.interpolate import UnivariateSpline

    valid = np.isfinite(bv)    # NaN is harmful for UnivariateSpline
    i = bisect_right(rv, mode)
    if valid[i] and i+1 < len(valid) and valid[i+1]:
        spline = UnivariateSpline(rv[valid], bv[valid], s=0)
        boundary = spline(mode)
        color = "red" if stdev_ratio > boundary else "blue"
    else:
        color = "gray"
    return color

def plot_boundary_impl(ax2, rv, bv, mu, sigma):        
    ax2.set_title("Stdev Ratio vs. Pore Size", fontsize=16)
    ax2.set_xlabel(r"Pore Size ($\AA$)", fontsize=14)
    ax2.set_ylabel("Stdev Ratio", fontsize=14)
    ax2.plot(rv, bv, "-", linewidth=3, label="Approximation Bounadry")
    ymin, ymax = ax2.get_ylim()
    zeros = np.zeros(len(rv))
    tops = np.ones(len(rv))*ymax
    ax2.fill_between(rv, zeros, bv, color='green', alpha=0.2)
    ax2.fill_between(rv, bv, tops, color='pink', alpha=0.2)
    mode = compute_mode(mu, sigma)
    ax2.axvline(mode, color='blue', linestyle=':', label="Estimated Pore Size")
    stdev_ratio = compute_stdev(mu, sigma)/mode
    color = decide_allowance_color(rv, bv, mode, stdev_ratio)
    ax2.plot(mode, stdev_ratio, "o", color=color, markersize=10, label="Estimated Stdev Ratio")
    ax2.legend()
    return color

def plot_boundary(x, y, params, rgs, rv, bv, parent=None, fig_file=None):
    N, T, t0, me, mp, mu, sigma = params[0:7]
    scales = params[7:]

    cy_list, ty = compute_lnp_curves(x, N, T, me, mp, mu, sigma, t0, rgs, scales)
    if y is None:
        y = ty

    with plt.Dp(parent=parent, window_title="Demo", button_spec=["OK", "Cancel"]):
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,5))
        fig.suptitle("Monopore Approximation Boundary for %s" % get_in_folder(), fontsize=20)
        ax1.set_title("Experiment Data", fontsize=16)
        ax1.set_xlabel("Time (Frames)", fontsize=14)
        ax1.set_ylabel("Intensity", fontsize=14)
        ax1.plot(x, y)
        if cy_list is not None:
            for cy in cy_list:
                ax1.plot(x, cy, ":")
        color = plot_boundary_impl(ax2, rv, bv, mu, sigma)
        fig.tight_layout()
        if fig_file is None:
            plt.show()
        else:
            plt.show(block=False)
            fig.savefig(fig_file)
    return color

def demo_impl(x, y, params, rgs, parent=None, quickly=True, fig_file=None):
    rv, bv = compute_approx_boundary(x, y, params, rgs, quickly=quickly)
    color = plot_boundary(x, y, params, rgs, rv, bv, parent=parent, fig_file=fig_file)
    return color

def compute_boundary_impl(lrf_src, quickly=True, parent=None, fig_file=None):
    guess_info = lrf_src.guess_lnpore_params(return_rgs=True)
    x = lrf_src.xr_x
    y = lrf_src.xr_y
    params, rgs = guess_info
    rv, bv = compute_approx_boundary(x, y, params, rgs, quickly=quickly)
    color = plot_boundary(x, y, params, rgs, rv, bv, parent=parent, fig_file=fig_file)
    ret_params = np.concatenate(guess_info)
    return color, ret_params
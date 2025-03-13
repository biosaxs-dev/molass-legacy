"""
    SecTheory.PoresizeFreedom.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from DataUtils import get_in_folder
from ModelParams.EghParams import EghAdvansedParams
from .RetensionTime import compute_retention_time

def freedom_plot(sseq):
    in_folder = get_in_folder(sseq.get_in_folder().replace("\\", "/"))

    fv_array, x_array = sseq.get_info()[0:2]
    k = np.argmin(fv_array[:,1])
    pasrams_type = EghAdvansedParams(sseq.get_nc())
    separated = pasrams_type.split_params_simple(x_array[k,:])
    xr_params = separated[0]
    rg_params = separated[2][:-1]
    seccol_params = separated[-1]

    f_trs = xr_params[:,1]

    fig, axes = plt.subplots(ncols=5, figsize=(20, 4))
    fig.suptitle("Demo indicating Freedom of Pore Size using %s" % in_folder, fontsize=20)

    t0, k, rp, m = seccol_params

    poresizes = [60, 70, 80, 90, 100]
    fit_scores = []
    params_list = []
    i = 0
    for ax, ps in zip(axes, poresizes):
        ax.set_title("Pore Size=%.3g" % ps, fontsize=16)
        ax.plot(f_trs, rg_params, "-o", label="current fit")

        secc_params, min_ret = adjust_to_poresize(f_trs, rg_params, seccol_params, ps)
        params_list.append(secc_params)
        fit_scores.append(min_ret.fun)

        m_trs = compute_retention_time(secc_params, rg_params)
        ax.plot(m_trs, rg_params, "-o", label="stochastic model")
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        tx = (xmin + xmax)/2
        ty = (ymin + ymax)/2
        ax.text(tx, ty, ", ".join(["%.3g" % v for v in secc_params]), ha="center", alpha=0.5, fontsize=16)
        ax.legend()
        i += 1

    fig.tight_layout()
    fig.subplots_adjust(top=0.8)
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,6))
    fig.suptitle("SEC Parameters Dependence on Pore Size using %s" % in_folder, fontsize=20)

    ax1.set_title("Comparison of Curves with Different Poresizes", fontsize=16)
    ax1.set_xlabel("Retention Time", fontsize=16)
    ax1.set_ylabel("Rg", fontsize=16)

    ax1.plot(f_trs, rg_params, "-o", label="current fit")
    wider_rgs = np.linspace(10, 120, 100)
    for ps, params in zip(poresizes, params_list):
        m_trs = compute_retention_time(params, wider_rgs)
        ax1.plot(m_trs, wider_rgs, ":", label="poresize %g" % ps)
    ax1.legend()

    params_array = np.array(params_list)

    axt = ax2.twinx()
    axt.grid(False)

    ax2.set_title("t0, T, rp, m Variation vs. Poresizes", fontsize=16)
    ax2.set_xlabel("Pore Size", fontsize=16)

    ax2.plot(poresizes, params_array[:,0], label="t0")
    ax2.plot(poresizes, params_array[:,1], label="T")
    ax2.plot(poresizes, params_array[:,2], label="rp")
    axt.plot(poresizes, params_array[:,3], ":", label="m")
    ax2.legend()
    axt.legend()

    ax3.set_title("Fit Scores vs. Poresize", fontsize=16)
    ax3.set_xlabel("Pore Size", fontsize=16)
    ax3.set_ylabel("Fit Score", fontsize=16)

    ax3.plot(poresizes, fit_scores, "o")

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.show()

def adjust_to_poresize(f_trs, rgs, init_params, ps):
    x_ = np.zeros(4)
    x_[2] = ps
    def objective_func(x):
        x_[[0,1,3]] = x
        model_trs = compute_retention_time(x_, rgs)
        return np.sqrt(np.average((model_trs - f_trs)**2))

    ret = minimize(objective_func, init_params[[0,1,3]])
    x_[[0,1,3]] = ret.x

    return x_, ret

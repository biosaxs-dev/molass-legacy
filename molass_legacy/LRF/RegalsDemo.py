# coding: utf-8
"""
    RegalsDemo.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from .regals import component, concentration_class, profile_class, mixture, regals
from SvdDenoise import get_denoised_data
import molass_legacy.KekLib.DebugPlot as plt

def demo(sp, sd):
    pdata, popts = sp.get_lrf_args(sd)

    # print(pdata.paired_ranges)

    D, E, q, ecurve = sd.get_xr_data_separate_ly()
    print(D.shape)

    I = D
    sigma = E

    x = ecurve.x
    y = ecurve.y

    C_list = []
    decomp_info = pdata.decomp_info
    for rec in decomp_info.opt_recs:
        f = rec[1]
        C_list.append(f(x))

    C_ = np.array(C_list)
    D_ = get_denoised_data(D, rank=2)
    P_ = D_ @ np.linalg.pinv(C_)

    C1 = component(concentration_class('smooth', x, xmin=0, xmax=120), profile_class('simple', q))
    C2 = component(concentration_class('smooth', x, xmin=70, xmax=280), profile_class('simple', q))
    M = mixture([C1, C2])
    M.lambda_concentration = M.estimate_concentration_lambda(E, np.array([np.inf, np.inf]))
    R = regals(I, sigma)

    def stop_fun(num_iter, params):
        return [num_iter >= 100, 'max_iter']

    def update_fun(num_iter, mix, params, resid):
        print('%2d, x2 = %f, delta_profile = %s'
                % (num_iter, params['x2'], np.array2string(params['delta_profile'], precision=3)))

    [M1,params,resid,exit_cond] = R.run(M, stop_fun, update_fun)
    [I1, sigma1] = M1.extract_profile(I, sigma, 0)
    [I2, sigma2] = M1.extract_profile(I, sigma, 1)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,11))
    fig.suptitle("Comparison between _MOLASS and REGALS on Elution Decomposition", fontsize=20)
    ax1 = axes[0,0]
    ax2 = axes[0,1]
    ax3 = axes[1,0]
    ax4 = axes[1,1]
    ax1.set_title("Decomposition from molass_legacy._MOLASS", fontsize=16)
    ax2.set_title("Decomposition from REGALS", fontsize=16)
    ax3.set_title("Profiles from molass_legacy._MOLASS", fontsize=16)
    ax4.set_title("Profiles from REGALS", fontsize=16)

    for ax in [ax3, ax4]:
        ax.set_yscale('log')

    for ax in [ax1, ax2]:
        ax.plot(x, y, color='orange')

    for c in C_list:
        ax1.plot(x, c, ':')

    ax2t = ax2.twinx()
    ax2t.grid(False)
    for k, c in enumerate(M1.concentrations.T):
        ax2t.plot(x, c, ':')


    for p in P_.T:
        ax3.plot(q, p)

    for I_ in [I1, I2]:
        ax4.plot(q, I_)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()

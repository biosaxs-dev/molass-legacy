"""
    Models.Stochastic.MomentsCollation.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from molass_legacy.Models.Stochastic.MonoporeMoments import study_monopore_moments_impl

def plot_collation_state_impl(x, y, egh_cy_list, egh_moments_list, ax, moments_list):
    ax.plot(x, y, color="orange")

    for k, (cy, M, M_) in enumerate(zip(egh_cy_list, egh_moments_list, moments_list)):
        ax.plot(x, cy, ":")
        m, v = M[0:2]
        s = np.sqrt(v)
        ax.axvline(x=m, ls="-", color="blue")
        label = "EGH Moments" if k == 0 else None
        ax.axvspan(m-s, m+s, color="cyan", alpha=0.3, label=label)
        m_, v_ = M_[0:2]
        s_ = np.sqrt(v_)
        ax.axvline(x=m_, ls="-", color="red")
        label = "Stochastic Moments" if k == 0 else None
        ax.axvspan(m_-s_, m_+s_, color="pink", alpha=0.3, label=label)

    ax.legend(loc="upper left")

def moments_collation_illust_impl(lrf_src,  debug=False):
    print("moments_collation_illust_impl")

    rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = lrf_src.compute_rgs(debug=False)
    peaks = lrf_src.get_peaks()
    num_peaks = len(peaks)
    model = lrf_src.model
    egh_moments_list = lrf_src.get_egh_moments_list()
    x = lrf_src.xr_x
    y = lrf_src.xr_y
    egh_cy_list = []
    for params in peaks:
        cy = model(x, params)
        egh_cy_list.append(cy)
    egh_ty = np.sum(egh_cy_list, axis=0)

    def plot_collation_state(ax, moments_list):
        plot_collation_state_impl(x, y, egh_cy_list, egh_moments_list, ax, moments_list)

    init_moments_list  = study_monopore_moments_impl(lrf_src, return_init_moments=True)
    step1_moments_list = study_monopore_moments_impl(lrf_src, trust_all_rgs=True, use_basinhopping=False, return_step1_moments=True)
    step2_moments_list, unreliable_indeces = study_monopore_moments_impl(lrf_src, trust_max_num=2, return_step2_moments=True)

    with plt.Dp():
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))
        fig.suptitle("Moments Collation Method Illustration on %s" % get_in_folder(), fontsize=20)
        ax1.set_title("Initial Guess", fontsize=16)
        plot_collation_state(ax1, init_moments_list)
        ax2.set_title("Optimize Simply with Local Optimization", fontsize=16)
        plot_collation_state(ax2, step1_moments_list)
        ax3.set_title("Optimize with Rg Coorection %s, Time Scaling, BH" % (unreliable_indeces), fontsize=16)
        plot_collation_state(ax3, step2_moments_list)
        fig.tight_layout()
        plt.show()
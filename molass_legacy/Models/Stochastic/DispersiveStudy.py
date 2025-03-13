"""
    Models.Stochastic.DispersiveStudy.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.Stochastic.MomentUtils import compute_egh_moments
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

def study(x, y, baseline, model, peaks, peak_rgs, qualities, props, curent_info=None, debug=False):
    from importlib import reload
    import Models.Stochastic.DispersiveMonopore
    reload(Models.Stochastic.DispersiveMonopore)
    from molass_legacy.Models.Stochastic.DispersiveMonopore import guess_params_using_moments
    import Models.Stochastic.DispersiveUtils
    reload(Models.Stochastic.DispersiveUtils)
    from molass_legacy.Models.Stochastic.DispersiveUtils import compute_elution_curves

    egh_moments_list = compute_egh_moments(peaks)
    ret = guess_params_using_moments(x, y, egh_moments_list, peak_rgs, qualities, props, debug=debug)
    if ret is None:
        return
    sdm_params, temp_rgs, bounds = ret

    show_edm_info = False
    if curent_info is not None:
        curent_model, modelname, params_array = curent_info
        if modelname == "EDM":
            show_edm_info = True

    in_folder = get_in_folder()
    is_oa_ald = in_folder.find('sample_data') >= 0

    with plt.Dp():
        if show_edm_info:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20,4))
            fig.suptitle("Study on SDM-EDM Relation with " + in_folder, fontsize=20)
        else:
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))

        ax1.set_title("EGH Decomposition", fontsize=16)
        ax1.plot(x, y)
        cy_list = []
        for k, params in enumerate(peaks):
            cy = model(x, params)
            cy_list.append(cy)
            ax1.plot(x, cy, ":", label='component-%d' % k)
        ty = np.sum(cy_list, axis=0)
        ax1.plot(x, ty, ":", color="red", label='model total')
        ax1.legend()

        ax2.set_title("SDM Decomposition", fontsize=16)
        cy_list, ty = compute_elution_curves(x, sdm_params, temp_rgs)
        ax2.plot(x, y)
        for k, cy in enumerate(cy_list):
            ax2.plot(x, cy, ":", label='component-%d' % k)
        ax2.plot(x, ty, ":", color="red", label='model total')
        ax2.legend()

        if show_edm_info:
            ax3.set_title("EDM Decomposition", fontsize=16)
            ax3.plot(x, y)
            cy_list = []
            for k, params in enumerate(params_array):
                cy = curent_model(x - x[0], params)
                ax3.plot(x, cy, ":", label='component-%d' % k)
                cy_list.append(cy)
            ty = np.sum(cy_list, axis=0)
            ax3.plot(x, ty, ":", color="red", label='model total')
            ax3.legend()

            # N, K, x0, poresize, N0, tI
            poresize = sdm_params[3]
            # t0, u, a, b, e, Dz, cinj
            porosities = params_array[:,4]
            edm_rgs = poresize * (1 - np.power(porosities, 1/3))

            ind = np.arange(len(peak_rgs))  # the x locations for the groups
            width = 0.35  # the width of the bars

            ax4.set_title("$K_{SEC} \sim Porosity$ Proof with Poresize=%.3g" % poresize, fontsize=16)
            ax4.bar(ind - width/2, peak_rgs, width, label="SAXS $R_g$")
            ax4.bar(ind + width/2, edm_rgs, width, label="Kinetic $R_g$")

            ax4.set_ylabel("$R_g$")
            ax4.set_xticks(np.arange(len(peak_rgs)))
            if is_oa_ald:
                labels = ('OA', 'OA-r', 'ALD')
            else:
                labels = ['$P_%d$' % k for k in range(len(peak_rgs))]
            ax4.set_xticklabels(labels)
            ax4.set_ylim(0, max(60, np.max(peak_rgs)*1.1))
            ax4.legend(loc="upper left")
            axt = ax4.twinx()
            axt.grid(False)
            axt.set_ylabel("Porosity")
            axt.plot(ind, porosities, "o", color="red", label="Porosity")
            axt.set_ylim(0, 1)
            axt.legend()

        fig.tight_layout()
        plt.show()
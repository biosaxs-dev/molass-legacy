"""
    SSDC.SsdcAnalysisUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from matplotlib.gridspec import GridSpec
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.Stochastic.DispersiveUtils import NUM_SDMCUV_PARAMS, compute_elution_curves

def plot_ssdc_result(names, info_list, rg_info_list, unreliable_indeces,
                     sdm_params_info_list, mapping_list, mapping,
                     extra_button_specs=[]):

    gs = GridSpec(2,4)
    with plt.Dp(button_spec=["OK", "Cancel"], extra_button_specs=extra_button_specs):
        fig = plt.figure(figsize=(20,8))
        axes_list = []
        for i in range(2):
            axes_row = []
            for j in range(3):
                jslice = j if j < 2 else slice(2,4)
                ax = fig.add_subplot(gs[i,jslice])
                axes_row.append(ax)
            axes_list.append(axes_row)
        axes = np.array(axes_list)

        fig.suptitle("SSDC - Same Sample in Different Columns Analysis: %s vs. %s" % tuple(names), fontsize=24)

        for k, (in_folder, lrfsrc) in enumerate(info_list):
            rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = rg_info_list[k]
            x = lrfsrc.xr_x
            y = lrfsrc.xr_y
            model = lrfsrc.model

            ax = axes[k,0]
            ax.set_title("EGH Decomposition of %s" % names[k], fontsize=16)
            ax.plot(x, y)
            cy_list = []
            peaks = lrfsrc.xr_peaks[indeces]
            for k, params in enumerate(peaks):
                cy = model(x, params)
                ax.plot(x, cy, ":", label="$R_g$=%.3g" % peak_rgs[k])
            ax.legend()
        
        for k, (in_folder, lrfsrc) in enumerate(info_list):
            x = lrfsrc.xr_x
            y = lrfsrc.xr_y
            ret_params, temp_rgs = sdm_params_info_list[k][0:2]
            cy_list, ty = compute_elution_curves(x, ret_params, temp_rgs)

            ax = axes[k,1]
            ax.set_title("SDM Decomposition of %s" % names[k], fontsize=16)
            ax.plot(x, y)
            for k, cy in enumerate(cy_list):
                ax.plot(x, cy, ":", label="$R_g$=%.3g" % temp_rgs[k])
            texts = ax.legend().get_texts()
            for i in unreliable_indeces:
                texts[i].set_color("red")

        slope, intercept = mapping

        for k, (in_folder, lrfsrc) in enumerate(info_list):
            x = lrfsrc.xr_x
            y = lrfsrc.xr_y
            ret_params, temp_rgs = sdm_params_info_list[k][0:2]
            cy_list, ty = compute_elution_curves(x, ret_params, temp_rgs)
            N, K, x0, poresize, N0, tI = ret_params[0:NUM_SDMCUV_PARAMS]

            ax = axes[k,2]
            ax.set_title("SDM Decomposition of %s with Poresize=%.3g, $N_0$=%.0f" % (names[k], poresize, N0), fontsize=16)
            ax.plot(x, y)
            moments = mapping_list[k]
            for j, cy in enumerate(cy_list):
                ax.plot(x, cy, ":", label="$R_g$=%.3g" % temp_rgs[j])
                ax.axvline(x=moments[j], color="green", alpha=0.5)
            ax.axvline(x=x0, color="red", label="$t_0$")
            ax.axvline(x=tI, color="gray", label="$t_{inj}$")
            # ax.set_xlim(-350, 1400)
            if k > 0:
                ax.set_xlim(*[x_*slope + intercept for x_ in [xmin, xmax]])
            xmin, xmax = ax.get_xlim()
            print([k], xmin, xmax)
            texts = ax.legend(bbox_to_anchor=(0.3, 1)).get_texts()
            for i in unreliable_indeces:
                texts[i].set_color("red")
            
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.show()
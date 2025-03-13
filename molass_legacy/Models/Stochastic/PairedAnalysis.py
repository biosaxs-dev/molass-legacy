"""
    Models.Stochastic.PairedAnalysis.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from importlib import reload
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

def paired_analysis_main(info_list, exec_spec):
    from molass_legacy.Models.Stochastic.PairedAnalysisSpecs import get_components_to_use
    from molass_legacy.Models.Stochastic.MomentUtils import compute_egh_moments
    num_components  = exec_spec["num_components"]
    componets_to_use = get_components_to_use(exec_spec)
    print("paired_analysis_impl", len(info_list))

    def ssdc_analysis():
        from SSDC.SsdcAnalysis import SsdcAnalysis
        parent = plt.get_parent()
        dialog = SsdcAnalysis(parent, info_list, num_components=num_components)
        dialog.show()

    def ssdc_analysis_sdm():
        import SSDC.SsdcAnalysisSdm
        reload(SSDC.SsdcAnalysisSdm)
        from SSDC.SsdcAnalysisSdm import spike
        spike(info_list, num_components, exec_spec)
    
    extra_button_specs = [
                ("SSDC Analysis", ssdc_analysis),
                ("SSDC Analysis (SDM)", ssdc_analysis_sdm),
                ]

    with plt.Dp(button_spec=["OK", "Cancel"],
                extra_button_specs=extra_button_specs):
        fig, axes = plt.subplots(ncols=2, figsize=(12,5))

        average_sigma_list = []
        for k, (in_folder, lrfsrc) in enumerate(info_list):
            ax = axes[k]
            ax.set_title(get_in_folder(in_folder))
            select = componets_to_use[k]
            rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = lrfsrc.compute_rgs(want_num_components=num_components, select=select, debug=False)
            x = lrfsrc.xr_x
            y = lrfsrc.xr_y
            model = lrfsrc.model
            ax.plot(x, y)
            cy_list = []
            print([k], "indeces=", indeces)
            peaks = lrfsrc.xr_peaks[indeces]
            moments_list = compute_egh_moments(peaks)
            print([k], "props=", props)
            print([k], "moments_list=", moments_list)
            average_sigma = np.sum(props * np.sqrt(np.array(moments_list)[:,1]))
            average_sigma_list.append(average_sigma)
            for k, params in enumerate(peaks):
                cy = model(x, params)
                ax.plot(x, cy, ":", label="$R_g$=%.3g" % peak_rgs[k])
            ax.legend()
        
        print("sigma ratio=", average_sigma_list[0]/average_sigma_list[1])

        fig.tight_layout()
        plt.show()
"""
    SSDC.SsdcAnalysisSdm.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.stats import linregress
from molass_legacy._MOLASS.SerialSettings import set_setting
from molass_legacy.Models.Stochastic.MomentUtils import compute_egh_moments
from importlib import reload
import Models.Stochastic.DispersiveMonopore
reload(Models.Stochastic.DispersiveMonopore)
from molass_legacy.Models.Stochastic.DispersiveMonopore import guess_params_using_moments
import SSDC.SsdcAnalysisUtils
reload(SSDC.SsdcAnalysisUtils)
from SSDC.SsdcAnalysisUtils import plot_ssdc_result

def spike(info_list, num_components, exec_spec):
    print("spike")

    if False:
        exec_spec = {
        # "input_folders" :       [ PyTools + r"\Data\20230706\ALD_OA001",   PyTools + r"\Data\20230706\ALD_OA002"],
        "num_components" :      3,
        "unreliable_indeces" :  [1],
        "poresize_bounds" :      (75, 80),
        "init_N0" :             [50000, 7000],
        }

    names = [folder.split('\\')[-1] for folder, info in info_list]
    print("names=", names)

    rg_info_list = []
    sdm_params_info_list = []
    moments_list = []
    mapping_list = []
    for i, (in_folder, lrfsrc) in enumerate(info_list):
        set_setting('in_folder', in_folder)
        print([i], "getting rg info")
        info = lrfsrc.compute_rgs(want_num_components=num_components, debug=False)
        rg_info_list.append(info)

        print([i], "getting sdm info")
        rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = info
        peaks = lrfsrc.xr_peaks[indeces]
        egh_moments_list = compute_egh_moments(peaks)
        moments_list.append(egh_moments_list)
        mapping_list.append([M[0] for M in egh_moments_list])
        x = lrfsrc.xr_x
        y = lrfsrc.xr_y
        exec_spec_ = exec_spec.copy()
        exec_spec_["init_N0"] = exec_spec_["init_N0"][i]
        ret = guess_params_using_moments(x, y, egh_moments_list, peak_rgs, qualities, props, exec_spec=exec_spec_, debug=False)
        sdm_params_info_list.append(ret)

    mx = mapping_list[0]
    my = mapping_list[1]
    slope, intercept, r_value, p_value, std_err = linregress(mx, my)
    mapping = slope, intercept

    unreliable_indeces = exec_spec['unreliable_indeces']
    def paired_analysis():
        import Models.Stochastic.PairedAnalysisImpl
        reload(Models.Stochastic.PairedAnalysisImpl)
        from molass_legacy.Models.Stochastic.PairedAnalysisImpl import paired_analysis_impl
        paired_analysis_impl(names, num_components, info_list, rg_info_list,
                             unreliable_indeces, sdm_params_info_list, mapping_list, mapping, moments_list, exec_spec)

    extra_button_specs = [
                ("Paired Analysis", paired_analysis),
                ]

    plot_ssdc_result(names, info_list, rg_info_list, unreliable_indeces,
                     sdm_params_info_list, mapping_list, mapping,
                     extra_button_specs=extra_button_specs)
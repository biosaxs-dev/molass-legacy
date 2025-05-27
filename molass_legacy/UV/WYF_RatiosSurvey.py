"""
    UV.WYF_RatiosSurvey.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import os
import sys
import numpy as np

def get_wyf_ratios(in_folder, *args, **kwargv):
    from molass_legacy._MOLASS.SerialSettings import clear_temporary_settings, clear_v2_temporary_settings
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Batch.LiteBatch import LiteBatch
    from molass_legacy.UV.WYF_Ratios import compute_ratios_from_Puv
    from molass_legacy.SerialAnalyzer.DataUtils import cut_upper_folders

    global fh

    print(in_folder)
    folder_name = cut_upper_folders(in_folder)
    if False:
    # if folder_name < "20170506":
    # if folder_name.find("OA001") < 0:
        print("skip")
        return True, None

    try:
        clear_temporary_settings()
        clear_v2_temporary_settings()
        sp = StandardProcedure()
        sd = sp.load_old_way(in_folder)
        lb = LiteBatch()
        lb.prepare(sd)
        init_params = lb.get_init_estimate()

        lb.optimizer.prepare_for_optimization(init_params, minimally=True)
        lrf_info = lb.optimizer.objective_func(init_params, return_lrf_info=True)
        proportions = lrf_info.get_xr_proportions()
        print("proportions=", proportions)
        Pxr, Cxr, Puv, Cuv, mapped_UvD = lrf_info.matrices
        ret_list = compute_ratios_from_Puv(lb.optimizer.wvector, Puv[:,:-1], proportions=proportions)
    except:
        ret_list = [(np.nan, np.nan)]

    for k, ratios in enumerate(ret_list):
        peak_name = "%s-%d" % (folder_name, k)
        fh.write(",".join([peak_name, *["%.3g" % r for r in ratios]]) + "\n")

    fh.flush()

    return True, None

def collect_ratio_data(input_folder, result_folder):
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.SerialAnalyzer.DataUtils import serial_folder_walk
    global fh

    result_file = os.path.join(result_folder, "wyf-ratios.csv")

    fh = open(result_file, "w")

    serial_folder_walk(root_folder, get_wyf_ratios)

    fh.close()

if __name__ == '__main__':
    this_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.dirname(this_dir)
    sys.path.append(lib_dir)

    import molass_legacy.KekLib, SerialAnalyzer, DataStructure, GuinierAnalyzer, Decomposer, Extrapolation
    from molass_legacy.SerialAnalyzer.DataUtils import get_pytools_folder

    root_folder = get_pytools_folder()
    input_folder = os.path.join(root_folder, "Data")
    result_folder = r"D:\TODO\20231218\wyf-ratios"
    collect_ratio_data(input_folder, result_folder)

"""
    SecTheory.SecParamsSurvey.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import os
import sys
import numpy as np

columntype_id = None

def inspect_sec_params(in_folder, *args, **kwargv):
    from molass_legacy._MOLASS.SerialSettings import clear_temporary_settings, clear_v2_temporary_settings, set_setting, get_setting
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Batch.LiteBatch import LiteBatch
    from molass_legacy.SerialAnalyzer.DataUtils import cut_upper_folders
    from molass_legacy.Tools.EmbedDialog import open_dialog
    from Experiment.ColumnTypes import update_sec_settings_by_id

    global fh, count, columntype_id

    count += 1
    print([count], in_folder)
    set_setting('in_folder', in_folder)

    if count < start:
        return True

    folder_name = cut_upper_folders(in_folder)
    # if False:
    # if folder_name < "20170506":
    # if folder_name.find("OA001") < 0:

    if True:
        if folder_name.find("proteins5") >= 0:
            columntype_id = "ad6w"
        else:
            print("skip")
            return True

    try:
        clear_temporary_settings()
        clear_v2_temporary_settings()

        if columntype_id is None:
            columntype_id = "ad200w"
        update_sec_settings_by_id(columntype_id)
        poresize = get_setting('poresize')
        print("poresize (1)", poresize)

        sp = StandardProcedure()
        sd = sp.load_old_way(in_folder)

        lb = LiteBatch()
        lb.prepare(sd, num_peaks=6)

        init_params = lb.get_init_estimate()

        optimizer = lb.get_optimizer()
        optimizer.prepare_for_optimization(init_params, minimally=True)

        print("init_separate_params[7]", optimizer.init_separate_params[7])

        lrf_info = lb.optimizer.objective_func(init_params, return_lrf_info=True)

        # plot_sec_params(optimizer, init_params, lrf_info)

        def reload_wrapper():
            from importlib import reload
            import SecTheory.SecParamsPlot
            reload(SecTheory.SecParamsPlot)
            from SecTheory.SecParamsPlot import plot_sec_params
            plot_sec_params(optimizer, init_params, lrf_info)

        open_dialog(reload_wrapper)

    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(None, "inspect_sec_params: ", n=10)

    fh.flush()

    return False

def collect_sec_params(input_folder, result_folder):
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from DevelUtils.FolderWalkers import serial_folder_walk
    global fh, count, start

    result_file = os.path.join(result_folder, "sec-params.csv")

    fh = open(result_file, "w")
    count = -4
    start = 0

    serial_folder_walk(input_folder, inspect_sec_params, reverse=True)

    fh.close()

if __name__ == '__main__':
    this_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.dirname(this_dir)
    sys.path.append(lib_dir)

    import seaborn
    seaborn.set()

    import molass_legacy.KekLib, SerialAnalyzer, DataStructure, GuinierAnalyzer, Decomposer, Extrapolation
    from molass_legacy.SerialAnalyzer.DataUtils import get_pytools_folder

    root_folder = get_pytools_folder()
    input_folder = os.path.join(root_folder, "Data")
    for drive in ['D', 'E']:
        result_folder = r"%s:\TODO\20231218\sec-params" % drive
        if os.path.exists(result_folder):
            break
    collect_sec_params(input_folder, result_folder)

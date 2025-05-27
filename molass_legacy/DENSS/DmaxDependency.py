# coding: utf-8
"""
    DmaxDependency.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
from molass_legacy._MOLASS.SerialSettings import set_setting
from DataUtils import serial_folder_walk
from molass_legacy.Batch.StandardProcedure import StandardProcedure

sp = StandardProcedure()

dmax_variations = [80, 90, 100, 110, 120]

def run_datgnom(result):
    print('run_datgnom: result=', result)

def run_denss_for_survey(k, result, dmax):
    print(dmax)

num_folders = 0

def run_denss_varying_dmax(folder, *args, **kwargv):
    global num_folders
    num_folders += 1
    print([num_folders], folder)
    set_setting('in_folder', folder)

    sp.load(folder)
    sd = sp.get_corrected_sd()
    preview_results = sp.get_preview_results(sd)
    for result in preview_results:
        run_datgnom(result)
        continue

        for dmax in dmax_variations:
            for k in range(2):
                run_denss_for_survey(k, preview_results, dmax)

    return False, None

def collect_dependecy_data(root_folder):
    serial_folder_walk(root_folder, run_denss_varying_dmax)

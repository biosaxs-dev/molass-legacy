"""
    Optimizer.TheUtils.py

    Copyright (c) 2021-2024, SAXS Team, KEK-PF
"""
import os
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting

FILES = [
    "callback.txt",
    "in_data_info.txt",
    "init_params.txt",
    "optimizer.log",
    "pid.txt",
    "seed.txt",
    "trimming.txt",
    "bounds.txt",
    ]

def get_optimizer_folder():
    optimizer_folder = get_setting('optimizer_folder')
    if optimizer_folder is None:
        optimizer_folder = os.path.join(get_setting('analysis_folder'), 'optimized')
    return optimizer_folder

def get_treatment_path(optimizer_folder=None):
    from molass_legacy.Optimizer.TheUtils import get_optimizer_folder
    if optimizer_folder is None:
        optimizer_folder = get_optimizer_folder()
    return os.path.join(optimizer_folder, "treatment.txt")

def get_optjob_folder_impl():
    return os.path.join(get_optimizer_folder(), 'jobs').replace('\\', '/')

def get_analysis_folder_from_work_folder(work_folder):
    # analysis_folder <= analysis_folder/optimized/jobs/nnn
    return os.path.abspath(work_folder + '/../../..')

def guess_n_compnents_trial(dsets, logger):

    (xr_curve, D), rg_curve, (uv_curve, U) = dsets

    epeaks = xr_curve.get_emg_peaks()
    n = len(epeaks)
    n_compnents = n + 2

    logger.info("n_components guessed as %d from len(epeaks)=%d", n_compnents, n)

    if False:
        import molass_legacy.KekLib.DebugPlot as plt
        from molass_legacy.Elution.CurveUtils import simple_plot

        plt.push()
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,7))
        simple_plot(ax1, uv_curve, color='blue')
        simple_plot(ax2, xr_curve, color='orange')
        fig.tight_layout()
        plt.show()
        plt.pop()

    return n_compnents

def get_work_info(work_folder):
    import os
    import re
    from molass_legacy.KekLib.BasicUtils import Struct
    in_data_txt = os.path.join(work_folder, FILES[1])
    in_folder_re = re.compile(r"in_folder=(\S+)")
    in_folder = None
    with open(in_data_txt) as fh:
        for line in fh:
            m = in_folder_re.search(line)
            if m:
                in_folder = m.group(1)
                break

    init_params_txt = os.path.join(work_folder, FILES[2])
    init_params = np.loadtxt(init_params_txt)

    bounds_txt = os.path.join(work_folder, FILES[7])
    if os.path.exists(bounds_txt):
        real_bounds = np.loadtxt(bounds_txt)
    else:
        real_bounds = None

    return  Struct(work_folder=work_folder, in_folder=in_folder, init_params=init_params, real_bounds=real_bounds)

def get_work_info_for_demo():
    from molass_legacy._MOLASS.SerialSettings import get_setting

    work_folder = get_setting("optjob_folder")
    if work_folder is None:
        in_folder = get_setting('in_folder')
        if in_folder.find('20181203') > 0:
            work_folder = r'D:\TODO\optimizer\result-20210624\20181203\000'
        else:
            work_folder = r'D:\TODO\20210628\optimizer-results\temp-save-32\optimizer\000'

    return get_work_info(work_folder)

def convert_to_peak_significances(xr_curve, xr_params_list):
    score_list = []
    for h, mu, sigma, tau in xr_params_list:
        score_list.append(h*sigma)
    score_array = np.array(score_list)
    return score_array/np.sum(score_array)

def get_sd_from_folder_impl(in_folder, logger, ver_date=None):
    from molass_legacy.SerialAnalyzer.AnalyzerUtil import compute_conc_factor_util
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
    from molass_legacy.SecSaxs.DataTreatment import DataTreatment

    logger.info("loading data from %s using StandardProcedure", in_folder)
    sp = StandardProcedure()
    sd_raw = sp.load_old_way(in_folder)
    pre_recog = PreliminaryRecognition(sd_raw)

    treat = DataTreatment()
    treat.load()        # this requires "optimizer_folder" setting
    if ver_date is None or ver_date >= "2023-01-01":
        sd = treat.get_trimmed_sd(sd_raw, pre_recog)
        corrected_sd = treat.get_corrected_sd(sd_raw, pre_recog, sd)
    else:
        sd = treat.get_treated_sd(sd_raw, pre_recog)
        corrected_sd = sd
    sd.conc_factor = compute_conc_factor_util()
    return sd, corrected_sd, treat

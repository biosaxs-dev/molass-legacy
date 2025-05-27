# coding: utf-8
"""
    Optimizer.BoundsAnalysis.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
import glob
import molass_legacy.KekLib.DebugPlot as plt

def demo(parent, analysis_folder, trimming=True, correction=True):
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.Optimizer.TheUtils import get_work_info
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
    from molass_legacy.Baseline.BaselineUtils import get_corrected_sd_impl
    from DataUtils import get_in_folder
    from molass_legacy.Optimizer.StateSequence import read_callback_txt_impl

    set_setting("analysis_folder", analysis_folder)
    work_folder = analysis_folder + "/optimized/jobs/000"
    in_folder = get_work_info(work_folder).in_folder
    in_folder = in_folder.replace("E:", "D:")
    set_setting("in_folder", in_folder)
    if False:
        sp = StandardProcedure()
        sd = sp.load_old_way(in_folder)
        pre_recog = PreliminaryRecognition(sd)
        if trimming:
            sd_ = sd._get_analysis_copy_impl(pre_recog)
        else:
            sd_ = sd.get_copy()

        if correction:
            v2_copy = get_corrected_sd_impl(sd_, sd, pre_recog)
        else:
            v2_copy = sd_

        D, _, wv, ecurve = v2_copy.get_uv_data_separate_ly()

    for k, file in enumerate(glob.glob(analysis_folder + "/optimized/jobs/*/callback.txt")):
        print([k], file)
        fv_list, x_list = read_callback_txt_impl(file)

    with plt.Dp():
        in_folder = get_in_folder(in_folder)
        fig, ax = plt.subplots()
        ax.set_title("Bounds Analysis for %s" % in_folder, fontsize=20)
        # simple_plot(ax, ecurve, legend=False)
        ax.plot(0, 0, 'o')
        fig.tight_layout()
        plt.show()

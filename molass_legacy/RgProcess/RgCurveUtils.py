"""
    RgProcess.RgCurveUtils.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import os
from .RgCurve import check_rg_folder, RgCurve

rg_root_folder = None

def make_and_save_rgcurves(in_root_folder, out_root_folder):
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.SerialAnalyzer.DataUtils import serial_folder_walk
    global rg_root_folder

    for path in [in_root_folder, out_root_folder]:
        assert os.path.exists(path)

    rg_root_folder = out_root_folder

    set_setting("test_pattern", 0)
    print(in_root_folder)
    serial_folder_walk(in_root_folder, do_a_folder)

def rg_folder_from_in_folder(in_folder):
    global rg_root_folder

    if rg_root_folder is None:
        from molass_legacy.SerialAnalyzer.DataUtils import get_pytools_folder
        pytools = get_pytools_folder()
        rg_root_folder = os.path.join(pytools, "RgCurves")

    nodes = in_folder.replace("/", "\\").split("\\")
    rg_folder = os.path.join(rg_root_folder, "\\".join(nodes[3:]))
    return rg_folder

def do_a_folder(in_folder, uv_folder, plot):
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
    from SecSaxs.DataTreatment import DataTreatment

    print(in_folder)
    if False:
        if in_folder < r"E:\PyTools\Data/20191001":
        # if in_folder < r"E:\PyTools\Data/20211001":
            return True, None

    rg_folder = rg_folder_from_in_folder(in_folder)
    print(rg_folder)
    ok = check_rg_folder(rg_folder)
    if ok:
        return True, None

    if os.path.exists(rg_folder):
        from shutil import rmtree
        rmtree(rg_folder)
    os.makedirs(rg_folder)

    sp = StandardProcedure()
    sd = sp.load_old_way(in_folder)

    pre_recog = PreliminaryRecognition(sd)
    treat = DataTreatment(route="v2", trimming=1, correction=1)
    sd_copy = treat.get_treated_sd(sd, pre_recog)

    D, E, qv, ecurve = sd_copy.get_xr_data_separate_ly()
    rg_curve = RgCurve(qv, ecurve, D, E)
    rg_curve.export(rg_folder)

    return True, None

def make_an_rg_folder(in_folder, rg_folder):
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
    from SecSaxs.DataTreatment import DataTreatment
    from molass_legacy._MOLASS.SerialSettings import clear_temporary_settings

    clear_temporary_settings()

    sp = StandardProcedure()
    sd = sp.load_old_way(in_folder)

    pre_recog = PreliminaryRecognition(sd)
    treat = DataTreatment(route="v2", trimming=1, correction=1)
    sd_copy = treat.get_treated_sd(sd, pre_recog)

    D, E, qv, ecurve = sd_copy.get_xr_data_separate_ly()
    rg_curve = RgCurve(qv, ecurve, D, E)
    rg_curve.export(rg_folder)

def make_rg_folders(in_folder_list, out_root_folder):
    for k, in_folder in enumerate(in_folder_list):
        rg_folder = os.path.join(out_root_folder, '%03d' % k)
        os.makedirs(rg_folder)
        make_an_rg_folder(in_folder, rg_folder)

def plot_rg_curve(ax, rg_curve, color='gray', with_qualities=False, quality_scale=1, label=None):
    if label is None:
        label = "SAXS Rg Curve"
    for k, (x, y, rg) in enumerate(rg_curve.get_curve_segments()):
        label_ = label if k == 0 else None     # to avoid multiple appearances in the legend 
        ax.plot(x, rg, color=color, alpha=0.5, label=label_)
        if with_qualities:
            qu = rg_curve.qualities[k]
            label_ = "SAXS Rg Quality Curve" if k == 0 else None # to avoid multiple appearances in the legend 
            ax.plot(x, qu*quality_scale, color='green', alpha=0.5, label=label_)
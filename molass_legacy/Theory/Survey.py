"""
    Theory.Survey.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""

from molass_legacy.Batch.StandardProcedure import StandardProcedure
from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
from .SphericalFit import compute_bq_score

def compute_bq_scores_for_all_peaks(sd):
    M, E, qv, ecurve = sd.get_xr_data_separate_ly()

    ret_list = []
    for pno, range_ in enumerate(ecurve.get_ranges_by_ratio(0.5)):
        f, p, t = range_

        eslice = slice(f,t)

        if pno == ecurve.primary_peak_no:
            gf, gt = sd.pre_recog.get_gunier_interval()
            preRg = sd.pre_recog.get_rg()
        else:
            pre_rg = sd.pre_recog.pre_rg
            preRg = pre_rg.compute_rg(selected=p)
            sg = pre_rg.sg
            gf = sg.guinier_start
            gt = sg.guinier_stop
        gslice = slice(gf, gt)

        ret = compute_bq_score(M, E, qv, ecurve, gslice, eslice, preRg)
        print([pno], p, ret[-1])
        ret_list.append(ret)

    return ret_list

def get_sd(in_folder):
    sp = StandardProcedure()
    sp.load(in_folder, debug=False)
    sd = sp.get_corrected_sd(debug=False)
    pre_recog = PreliminaryRecognition(sd)
    sd_ = pre_recog.get_analysis_copy()
    return sd_
    compute_bq_scores_for_all_peaks(sd_)

def bq_scores_survey_for_a_folder(in_folder, fh=None):
    from molass_legacy._MOLASS.SerialSettings import clear_temporary_settings
    from DataUtils import cut_upper_folders

    print('Doing', in_folder)
    clear_temporary_settings()      # iterating without this can be buggy! be careful.

    try:
        sd = get_sd(in_folder)
        for pno, ret in enumerate(compute_bq_scores_for_all_peaks(sd)):
            if fh is None:
                print([pno], ret)
            else:
                fh.write(','.join([cut_upper_folders(in_folder), str(pno)] + ['%g' % v for v in ret]) + '\n')
                fh.flush()
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
        etb = ExceptionTracebacker()
        print(etb.last_lines())

def bq_scores_survey():
    from DataUtils import get_pytools_folder, serial_folder_walk

    fh = open('bq_scores.csv', 'w')

    def proc(in_folder, uv_folder, plot=False):
        bq_scores_survey_for_a_folder(in_folder, fh=fh)
        return True, None

    data_folder = get_pytools_folder() + '/Data'
    serial_folder_walk(data_folder, proc)
    fh.close()

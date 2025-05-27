"""
    SdUtils.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""

def get_sd(in_folder, trimming=True, correction=True):
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
    from molass_legacy.Baseline.BaselineUtils import get_corrected_sd_impl

    set_setting("in_folder", in_folder)

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

    return v2_copy

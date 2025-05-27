"""
    Batch.BatchUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import os
from molass_legacy._MOLASS.SerialSettings import clear_temporary_settings, set_setting
from molass_legacy.Batch.LiteBatch import LiteBatch     

def load_lrfsrc(in_folder, clear=True):
    if clear:
        clear_temporary_settings()
    set_setting('in_folder', in_folder)
    batch = LiteBatch()
    lrf_src = batch.get_lrf_source(in_folder=in_folder, clear=False)
    return lrf_src

def load_v2_result(job_folder):
    from molass_legacy.Optimizer.FullOptResult import FullOptResult

    in_data_info_txt = os.path.join(job_folder, "in_data_info.txt")
    with open(in_data_info_txt) as fh:
        for line in fh:
            in_folder = line.strip().split("=")[-1]
    print("in_folder=", in_folder)
    lrfsrc = load_lrfsrc(in_folder)
    sd = lrfsrc.sd
    print("type(sd)=", type(sd))
    return FullOptResult(sd, lrfsrc.pre_recog, job_folder, set_folder_setting=True)
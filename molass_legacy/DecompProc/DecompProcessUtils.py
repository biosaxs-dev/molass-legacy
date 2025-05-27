# coding: utf-8
"""
    DecompProcessUtils.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import os
import numpy as np

MAXNUM_STEPS = 20000

def run_decomp_impl(*args, **kwargs):
    pass

def get_decompfolder():
    from molass_legacy._MOLASS.SerialSettings import get_setting
    from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry

    decomp_folder = (get_setting('analysis_folder') + '/Decomp').replace('\\', '/')
    if not os.path.exists(decomp_folder):
        mkdirs_with_retry(decomp_folder)

    return decomp_folder

def get_outfolder(job_id=None):
    decomp_folder = get_decompfolder()

    import re
    if job_id is None:
        out_folder = decomp_folder + '/000'
        while True:
            if os.path.exists(out_folder):
                out_folder = re.sub(r'/(\d+)$', lambda m: '/%03d' % (int(m.group(1)) + 1), out_folder)
            else:
                break
    else:
        out_folder_init = decomp_folder + '/%03d' % job_id
        out_folder = out_folder_init
        i = 0
        while True:
            if os.path.exists(out_folder):
                i += 1
                out_folder =  out_folder_init + '_%d' % i
            else:
                break

    return out_folder

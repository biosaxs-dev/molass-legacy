"""
    SaxsCurveUtils.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np

def percentile_normalize(y, percentile=98):
    # select 98-percentille point as peak-point to avoid abnormal max
    # see also SimpleGuinier.py
    head_len = len(y)//4
    kth = int(head_len*percentile/100)
    arg_hy98 = np.argpartition(y[0:head_len], kth)
    m = arg_hy98[kth]
    return y/y[m]

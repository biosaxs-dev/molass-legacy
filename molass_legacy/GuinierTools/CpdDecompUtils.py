"""
    GuinierTools.CpdUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import ruptures as rpt

def compute_end_points(nc, x_, rgv):
    print("cpd_spike_impl")

    algo = rpt.Dynp(model="l2").fit(rgv)
    my_bkps = algo.predict(n_bkps=nc-1)
    print("my_bkps=", my_bkps)
    points = np.concatenate([[0], my_bkps[:-1], [len(x_)-1]])
    return x_[points].copy()
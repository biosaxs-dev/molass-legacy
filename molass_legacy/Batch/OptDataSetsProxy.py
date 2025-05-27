"""
    Batch.OptDataSetsProxy.py

    Copyright (c) 2023-2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.Optimizer.OptDataSets import OptDataSets, get_dsets_impl

class DummyRgCurve:
    def __init__(self):
        poresize =  get_setting('poresize')
        if poresize is None:
            set_setting('poresize', 100)    # poresize will be used to compute dummy rg_params in SecTheory.SecEstimator.guess_initial_secparams

    def get_rgs_from_trs(self, trs):
        return np.ones(len(trs)) * np.nan

class OptDataSetsProxy(OptDataSets):
    def __init__(self, sd, corrected_sd):
        dsets = get_dsets_impl(sd, corrected_sd, rg_info=False)
        self.dsets = (dsets[0], DummyRgCurve(), dsets[2])
        D = dsets[0][1]
        E = sd.intensity_array[:,:,2].T
        self.weight_info = self.compute_weight_info(1/(E + D/100))
        self.E = E

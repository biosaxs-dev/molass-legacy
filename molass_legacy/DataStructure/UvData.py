# coding: utf-8
"""
    UvData.py

    Copyright (c) 2019-2022, SAXS Team, KEK-PF
"""

import numpy as np
import logging
from bisect import bisect_right
from MatrixData import MatrixData
from SerialDataUtils import load_uv_array
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Trimming import FlowChange
from molass_legacy.UV.PlainCurve import make_secondary_e_curve_at

NORMAL_RATIO_LIMIT = 50     # < 89.5 for 20170304


class UvData(MatrixData):
    def __init__(self, folder, sd=None, array=None, vector=None):
        self.logger = logging.getLogger(__name__)
        if sd is None:
            array, vector, _, col_header = load_uv_array(folder, column_header=True)
        else:
            if sd is None:
                assert array is not None and vector is not None
            else:
                array = sd.conc_array
                vector = sd.lvector
            col_header = None
        self.vector = vector
        self.col_header = col_header
        MatrixData.__init__(self, array)
        self.set_elution_curve()
        self.check_lowest_wlen_anomaly()

    def plot(self, ax=None):
        MatrixData.plot(self, ax=ax, title="UV Absorbance", ec_color='blue')

    def set_elution_curve(self):
        pick_pos = get_setting('absorbance_picking')
        index = bisect_right(self.vector, pick_pos)
        pick_num = 1
        start = index
        stop  = index + 1
        MatrixData.set_elution_curve(self, index, slice(start, stop))

    def check_lowest_wlen_anomaly(self):
        max_y = np.max(self.data[0,:])
        ratio = max_y/self.e_curve.max_y
        if ratio > NORMAL_RATIO_LIMIT:
            # as in 20170304
            self.data[0,:] = self.data[1,:]
            self.logger.warning("UV data[0,:] have been modified due to an abnormal ratio=%.3g." % ratio)

    def set_restriction(self, xr_e_curve, flow_changes=None):
        if flow_changes is None:
            e_curve2 = make_secondary_e_curve_at(self.data, self.vector, self.e_curve, self.logger)
            fc = FlowChange(self.e_curve, e_curve2, xr_e_curve)
            flow_changes = fc.get_real_flow_changes()
        print('flow_changes=', flow_changes)
        self.i_slice = slice(0, None)
        if flow_changes[0] is None:
            flow_changes[0] = 0
        self.j_slice = slice(*flow_changes)
        return flow_changes

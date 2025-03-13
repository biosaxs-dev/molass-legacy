# coding: utf-8
"""
    XrayData.py

    Copyright (c) 2019-2021, SAXS Team, KEK-PF
"""

import numpy as np
from bisect import bisect_right
from MatrixData import MatrixData
from SerialDataUtils import load_intensity_files
from molass_legacy._MOLASS.SerialSettings import get_setting, get_xray_picking
from molass_legacy.Trimming.PreliminaryRg import get_default_angle_range_impl

class XrayData(MatrixData):
    def __init__(self, folder, array=None, data=None, q=None, error=None, sd=None, correct_elution=False):
        files = None
        if data is None:
            if array is None:
                if sd is None:
                    array, files = load_intensity_files(folder)
                    assert len(files) > 10      # temporary guard ot avoid confusing when given an invalid folder
                else:
                    array = sd.intensity_array
            q = array[0,:,0]
            data = array[:,:,1].T
            error = array[:,:,2].T
        else:
            assert q is not None
            assert error is not None

        self.vector = q
        MatrixData.__init__(self, data)
        self.error = MatrixData(error)
        self.set_elution_curve(correct=correct_elution)
        self.files = files

    def plot(self, ax=None, color=None, alpha=1, title="Xray Scattering", ec_color='orange'):
        MatrixData.plot(self, ax=ax, color=color, alpha=alpha, title=title, ec_color=ec_color)

    def set_elution_curve(self, correct=False):
        pick_pos = get_xray_picking()
        pick_num = get_setting('num_points_intensity')      # odd number
        index = bisect_right(self.vector, pick_pos)
        hwidth = pick_num//2
        # take max because (index - hwidth) can be negative, e.g. in 20190522/Backsub3_CytC
        start = max(0, index - hwidth)
        stop  = start + pick_num
        print('XrayData.set_elution_curve start, stop=', (start, stop))
        MatrixData.set_elution_curve(self, index, slice(start, stop), correct=correct)

    def set_restriction(self, mapping_params, flow_changes):
        angle_start, flange_limit, pre_rg = get_default_angle_range_impl(self.data, self.error.data, self.e_curve, self.vector, self.e_index, self.logger)
        print('angle_start, flange_limit=', angle_start, flange_limit)

        self.i_slice = slice(angle_start, flange_limit)

        slope, intercept = mapping_params
        A_, B_ = 1/slope, -intercept/slope
        mapped_fc = [None if j is None else min(self.data.shape[1], max(0, int(A_*j + B_ + 0.5))) for j in flow_changes]

        print('flow_changes=', flow_changes, 'mapped_fc=', mapped_fc)
        if mapped_fc[0] is None:
            mapped_fc[0] = 0
        self.j_slice = slice(*mapped_fc)

    def copy_for_mct(self):
        i_slice = self.i_slice
        data = self.data[i_slice,:]
        q = self.vector[i_slice]
        error = self.error.data[i_slice,:]
        copy = XrayData(None, data=data, q=q, error=error)
        # copy.plot()
        return copy

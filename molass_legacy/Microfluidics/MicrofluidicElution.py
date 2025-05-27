# coding: utf-8
"""
    MicrofluidicElution.py

    Copyright (c) 2018-2019, SAXS Team, KEK-PF
"""
import os
import glob
import re
import numpy as np
from FlowSimulator import FlowSimulator

class MicrofluidicElution(FlowSimulator):
    def __init__(self, filepath, num_files):
        FlowSimulator.__init__(self, filepath, num_files=num_files)
        self.set_params()

    def set_flow_changes_as_a_temp_fix(self, absorbance):
        fc = self.get_linear_slope_ends()
        absorbance.jump_j = fc[0]
        absorbance.jump_j_safe = fc[0]
        absorbance.right_jump_j = fc[1]
        absorbance.right_jump_j_safe = fc[1]

    def set_baseplane_params_as_a_temp_fix(self, absorbance):
        absorbance.baseplane_params = ( 0, 0, 0 )

    def create_flow_change_proxy(self):
        return SimpleFlowChangeProxy(self)

    def propose_background_range(self, size, xy=None, width=10, debug=False, auto=True):
        x, y = self.get_elution_data(size)
        ii = self.get_linear_slope_ends()
        slope_start = ii[0]
        if xy is not None:
            x, y = xy

        min_y = np.min(y)
        lim_y = min_y*0.8 + y[slope_start]*0.2
        bx = np.where(np.logical_and(y < lim_y, x < x[slope_start]))[0]
        length = len(bx)
        stop = bx[-1] - int(length*0.1)
        start = stop - width

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            import time
            print('length=', length)
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(y)
            for i in ii:
                ax.plot(i, y[i], 'o', color='red')
            ax.plot(bx, y[bx], 'o', color='yellow')
            ax.plot([start, stop], y[[start, stop]], 'o', color='green')
            fig.tight_layout()
            if auto:
                plt.show(block=False)
                time.sleep(0.5)
            else:
                plt.show()

        return (start, stop)

def get_mtd_elution(in_folder, num_files):
    from SerialDataUtils import get_mtd_filename
    filename = get_mtd_filename(in_folder)

    if filename is None:
        mtd_elution = None
    else:
        try:
            from MicrofluidicElution import MicrofluidicElution
            mtd_elution = MicrofluidicElution(filename, num_files)
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            print(etb)
            mtd_elution = None
    assert mtd_elution is not None    

    return mtd_elution

class CurveSimilarityProxy:
    def __init__(self, me):
        self.size  = len(me.x)
        self.slope  = 1
        self.intercept = 0
        self.mapped_info = [np.array([1, 0]), [1, 1], None]

    def inverse_int_value(self, j):
        i = int( (j - self.intercept)/self.slope + 0.5 )
        return max(0, min(self.size-1, i))

class SimpleFlowChangeProxy:
    def __init__(self, me):
        self.size = len(me.x)
        fc = me.get_linear_slope_ends()
        fc_ = fc[0:2]
        self.fc_list = fc_
        self.fc_safe_list = fc_
        self.similarity = CurveSimilarityProxy(me)

    def get_flow_changes(self):
        return self.fc_safe_list

    def get_real_flow_changes(self):
        flow_changes_src = self.fc_safe_list
        flow_changes = []
        for fc in flow_changes_src:
            if fc == 0 or fc == self.size - 1:
                fc_ = None
            else:
                fc_ = fc
            flow_changes.append( fc_ )
        return flow_changes

    def get_raw_flow_changes(self):
        return self.fc_list

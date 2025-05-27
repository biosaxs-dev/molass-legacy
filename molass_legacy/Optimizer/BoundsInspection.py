"""
    Optimizer.BoundsInspection.py

    Copyright (c) 2021-2022, SAXS Team, KEK-PF
"""
import os
import numpy as np
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from tksheet import Sheet

class BoundsInspection(Dialog):
    def __init__(self, parent, fullopt, params):
        self.parent = parent
        self.params = params
        self.param_names = fullopt.get_parameter_names()
        self.real_bounds = fullopt.real_bounds
        self.scale_shift = fullopt.scale_shift
        self.scale_slope = fullopt.scale_slope
        self.norm_params = fullopt.to_norm_params(params)
        Dialog.__init__(self, parent, "Bounds Inspection", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):

        num_columns = 7
        num_valid_rows = min(30, len(self.real_bounds))
        data_list = [ [name] + ["%g" % v for v in [param, *bpair, sshift, sslope, nparam]]
                        for name, param, bpair, sshift, sslope, nparam
                        in zip(self.param_names, self.params, self.real_bounds, self.scale_shift, self.scale_slope, self.norm_params)]

        column_width = 90
        width = column_width*num_columns + 60
        height = int(22*num_valid_rows) + 60
        self.sheet = Sheet(body_frame, width=width, height=height, data=data_list, show_selected_cells_border=False, column_width=column_width)
        self.sheet.pack()
        self.sheet.enable_bindings()

"""
    Optimizer.FvScoreDetails.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import re
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from tksheet import Sheet
from .BasicOptimizer import PARAMS_SCALE

class FvScoreDetails(Dialog):
    def __init__(self, parent, optimizer, params):
        self.parent = parent
        self.optimizer = optimizer
        self.params = params
        self.separate_params = optimizer.split_params_simple(params)
        self.lower_bounds = optimizer.lower_bounds
        self.upper_bounds = optimizer.upper_bounds

        Dialog.__init__(self, parent, "FV Score Details", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        fv, score_list, Pxr, Cxr, Puv, Cuv, mapped_UvD = self.optimizer.objective_func(self.params, return_full=True)
        score_names = self.optimizer.get_score_names()

        data_list = []
        data_list.append(["", "Score Name", "Score Value"])
        for name, value in zip(score_names, score_list):
            row = ["", name, "%g" % value]
            if name == "Guinier_deviation":
                dev1, dev2 = self.optimizer.gdev.compute_deviation(Pxr, Cxr, self.separate_params[2], return_details=True)
                row += ["", "%g" % dev1, "%g" % dev2]
            data_list.append(row)

        data_list[1][0] = "Score Details"

        data_list.append([])

        data_list.append(["FV", "%g" % fv])

        data_list.append([])
        data_list.append(["", "Parameter Name", "Real Value", "Normalized Value", "Bounds Mask", "Lower Bound", "Upper Bound", "Penalty Type"])
        start_row_params = len(data_list)

        delchars_re = re.compile(r"[\$\\_{}]")
        param_names = [re.sub(delchars_re, "", name) for name in self.optimizer.get_parameter_names()]
        norm_values = self.optimizer.to_norm_params(self.params)
        mask_index = 0
        for name, real_value, norm_value, bnd_mask in zip(param_names, self.params, norm_values, self.optimizer.bounds_mask):
            if bnd_mask:
                lower_bound = self.lower_bounds[mask_index]
                upper_bound = self.upper_bounds[mask_index]
                mark = "" if real_value >= lower_bound and real_value <= upper_bound  else "out_of_bounds"
                mask_index += 1
                lower_bound_ = "%g" % lower_bound
                upper_bound_ = "%g" % upper_bound
            else:
                mark = "negative" if (name not in ["x0", "t0", "xba", "xbb", "s1", "s2", "mpb", "", "mra", "mrb", "diffratio", "Ti"] and real_value < 0) else ""
                lower_bound_ = ""
                upper_bound_ = ""
            data_list.append(["", name, "%g" % real_value, "%g" % norm_value, str(bnd_mask), lower_bound_, upper_bound_, mark])

        data_list[start_row_params][0] = "Parameter Details"

        num_columns = 8
        num_valid_rows = min(30, len(score_list) + 1 + len(param_names))

        column_width = 150
        width = column_width*num_columns + 60
        height = int(22*num_valid_rows) + 60
        self.sheet = Sheet(body_frame, width=width, height=height, data=data_list, show_selected_cells_border=False, column_width=column_width)
        self.sheet.pack()
        self.sheet.enable_bindings()

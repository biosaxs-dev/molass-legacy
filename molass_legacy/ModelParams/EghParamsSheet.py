"""
    Optimizer.ParamSetType.EghParamsSheet.py

    Copyright (c) 2021-2023, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.OurTkinter import Tk
from tksheet import Sheet
from molass_legacy._MOLASS.SerialSettings import get_setting
from .ParamsSheetBase import ParamsSheetBase

# task : unify parameter names with those in EghParams.py

class EghParamsSheet(ParamsSheetBase):
    def __init__(self, parent, params, dsets, optimizer, model_func=None):
        self.composite = optimizer.composite
        self.model_func = model_func
        ParamsSheetBase.__init__(self, parent, params, dsets, optimizer)
        self.build_sheet(params, dsets, optimizer)

    def get_wanted_params(self, params, optimizer):
        return optimizer.split_params_simple(params) + [[], 0, (None, None)]

    def build_sheet(self, params, dsets, optimizer):
        self.n = optimizer.n_components

        body_frame = Tk.Frame(self)
        body_frame.pack()

        dark_cells = []
        xr_params, xr_baseparams, rgs, mapping, uv_params, uv_baseparams, mappable_range, seccol_params, dark_cell_j_list, num_extra_addresses, (Ti, Np) = self.get_wanted_params(params, optimizer)

        constrained_anyway = num_extra_addresses> 0
        num_addresses = len(params) + num_extra_addresses
        self.params_addr = [None] * num_addresses
        self.params_addr_inv = {}

        xr_base = 0
        xr_bp_base = xr_base + np.product(xr_params.shape)
        rg_base = xr_bp_base + len(xr_baseparams)
        mp_base = rg_base + len(rgs)
        uv_base = mp_base + len(mapping)
        uv_bp_base = uv_base + len(uv_params)
        mr_base = uv_bp_base + len(uv_baseparams)
        se_base = mr_base + len(mappable_range)

        if dsets is None:
            num_columns = 9
            extra_headers = []
            xr_proportions = None
            uv_proportions = None
        else:
            if self.model_func is None:
                from molass_legacy.Peaks.ElutionModels import egh
                model_func = egh
            else:
                model_func = self.model_func
            num_columns = 11
            extra_headers = ["", "", "proportion"]

            opt_lrf_info = optimizer.objective_func(params, return_lrf_info=True)
            xr_proportions = opt_lrf_info.get_xr_proportions()
            uv_proportions = opt_lrf_info.get_uv_proportions()

        if seccol_params is None:
            num_extended_rows = 0
        else:
            num_extended_rows = len(seccol_params) + 1

        data_list = [["" for c in range(num_columns)] for r in range(self.n*2 + 2 + 8 + num_extended_rows)]

        row_offset = 0
        for j, name in enumerate(["h", "mu (tR)", "sigma", "tau", "composite", "", "rg"] + extra_headers, start=1):
            data_list[row_offset][j] = name

        row_offset += 1
        n = xr_params.shape[0]
        xr_col_size = xr_params.shape[1]
        if self.composite is None or not self.composite.really_composite:
            composite_ids = None
            show_flags = None
            proportions = xr_proportions
        else:
            composite_ids = self.composite.get_composite_ids()
            show_flags = self.composite.get_composite_show_flags()
            proportions = self.composite.get_composite_proportions(xr_proportions)

        for i in range(n):
            if i == 0:
                v = "xr_params"
            else:
                v = ""
            data_list[row_offset+i][0] = v
            for j in range(xr_col_size):
                data_list[row_offset+i][j+1] = "%g" % xr_params[i,j]
                k = i*4 + j
                self.set_params_addr(xr_base+k, (row_offset+i, j+1))
                if j in dark_cell_j_list:
                    dark_cells.append((row_offset+i, j+1))

            if composite_ids is not None:
                data_list[row_offset+i][xr_col_size+1] = str(composite_ids[i]+1)

            show_value = show_flags is None or show_flags[i]
            if show_value:
                data_list[row_offset+i][xr_col_size+3] = "%g" % rgs[i]
            self.set_params_addr(rg_base+i, (row_offset+i, xr_col_size+3))

            if xr_proportions is not None:
                if i == 0:
                    data_list[row_offset+i][xr_col_size+5] = "xr area"
                if i < len(xr_proportions):
                    if show_value:
                        i_ = i if composite_ids is None else composite_ids[i]
                        data_list[row_offset+i][xr_col_size+6] = "%g" % proportions[i_]

        row_offset += n + 1
        col_names = ["slope", "intercept"]
        if len(xr_baseparams) == 3:
            col_names.append("fouling")

        for j, name in enumerate(col_names):
            data_list[row_offset][1+j] = name

        row_offset += 1
        for i in range(1):
            if i == 0:
                v = "xr_baseline"
            else:
                v = ""
            data_list[row_offset+i][0] = v
            num_params = len(xr_baseparams)
            for j in range(num_params):
                data_list[row_offset+i][j+1] = "%g" % xr_baseparams[j]
                self.set_params_addr(xr_bp_base+j, (row_offset+i, j+1))

            # data_list[row_offset+i][xr_col_size+2] = "%g" % rgs[n+i]
            # self.set_params_addr(rg_base+n+i, (row_offset+i, xr_col_size+2))

        row_offset += 2
        data_list[row_offset][1] = "h"
        if uv_proportions is not None:
            data_list[row_offset][5] = "composite"
            data_list[row_offset][10] = "proportion"

        row_offset += 1
        if self.composite is None:
            proportions = uv_proportions
        else:
            proportions = self.composite.get_composite_proportions(uv_proportions)
        for i in range(n):
            if i == 0:
                v = "uv_params"
            else:
                v = ""
            data_list[row_offset+i][0] = v
            data_list[row_offset+i][1] = "%g" % uv_params[i]
            self.set_params_addr(uv_base+i, (row_offset+i, 1))

            if composite_ids:
                data_list[row_offset+i][xr_col_size+1] = str(composite_ids[i]+1)

            show_value = show_flags is None or show_flags[i]

            if uv_proportions is not None:
                if i == 0:
                    data_list[row_offset+i][xr_col_size+5] = "uv area"
                if i < len(uv_proportions):
                    if show_value:
                        i_ = i if composite_ids is None else composite_ids[i]
                        data_list[row_offset+i][xr_col_size+6] = "%g" % proportions[i_]

        row_offset += n + 1
        col_names = ["L", "x0", "k", "b", "s1", "s2", "diff_ratio"]
        if len(uv_baseparams) == 8:
            col_names.append("fouling")
        for j, name in enumerate(col_names):
            data_list[row_offset][1+j] = name

        row_offset += 1
        for i in range(1):
            if i == 0:
                v = "uv_baseline"
            else:
                v = ""
            data_list[row_offset+i][0] = v
            num_params = len(uv_baseparams)
            for j in range(num_params):
                data_list[row_offset+i][j+1] = "%g" % uv_baseparams[j]
                self.set_params_addr(uv_bp_base+j, (row_offset+i, j+1))

        row_offset += 2
        for j, name in enumerate(["slope", "intercept", "", "", "from", "to"]):
            data_list[row_offset][1+j] = name

        row_offset += 1
        data_list[row_offset][0] = "mapping"
        data_list[row_offset][4] = "mappable_range"
        for j in range(2):
            data_list[row_offset][1+j] = "%g" % mapping[j]
            self.set_params_addr(mp_base+j, (row_offset, 1+j))
            data_list[row_offset][5+j] = "%g" % mappable_range[j]
            self.set_params_addr(mr_base+j, (row_offset, 5+j))

        if seccol_params is not None:
            row_offset += 1
            for name, i in zip(["Npc", "rp (pore size)", "tI", "t0", "P", "m"], [0, 1, 2, 3, 4, 5]):
                value = seccol_params[i]
                row_offset += 1
                data_list[row_offset][0] = name
                data_list[row_offset][1] = "%g" % value
                self.set_params_addr(se_base+i, (row_offset, 1))

            if constrained_anyway and False:
                c9d_base = se_base + 4
                tmp_offset = row_offset - 3
                for i, (name, value) in enumerate([("Ti", Ti), ("Np", Np)]):
                    data_list[tmp_offset][3] = name
                    data_list[tmp_offset][4] = "%g" % value
                    self.set_params_addr(c9d_base + i, (tmp_offset, 4))
                    tmp_offset += 1

        self.num_valid_rows = row_offset + 2
        self.data_list = data_list
        column_width = 90
        width = column_width*num_columns + 60
        height = int(22*self.num_valid_rows) + 60
        self.sheet = Sheet(body_frame, width=width, height=height, data=data_list, show_selected_cells_border=False, column_width=column_width)

        for r, c in dark_cells:
            self.sheet.highlight_cells(r, c, bg="gray", fg="white")

        self.sheet.pack()

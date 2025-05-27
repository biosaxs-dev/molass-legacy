"""
    Optimizer.ParamSetType.StcParamsSheet.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.OurTkinter import Tk
from tksheet import Sheet
from molass_legacy._MOLASS.SerialSettings import get_setting
from .ParamsSheetBase import ParamsSheetBase

class StcParamsSheet(ParamsSheetBase):
    def __init__(self, parent, params, dsets, optimizer):
        ParamsSheetBase.__init__(self, parent, params, dsets, optimizer)

        self.n = optimizer.n_components
        nc  = self.n - 1
        self.params_addr = [None]*len(params)
        self.params_addr_inv = {}

        body_frame = Tk.Frame(self)
        body_frame.pack()

        xr_params, xr_baseparams, rgs, mapping, uv_params, uv_baseparams, mappable_range, sec_params = optimizer.split_params_simple(params)
        t0, rp, N, me, T, mp = sec_params

        xr_base = 0
        xr_bp_base = xr_base + np.product(xr_params.shape)
        rg_base = xr_bp_base + len(xr_baseparams)
        mp_base = rg_base + len(rgs)
        uv_base = mp_base + len(mapping)
        uv_bp_base = uv_base + len(uv_params)
        mr_base = uv_bp_base + len(uv_baseparams)
        se_base = mr_base + len(mappable_range)

        if dsets is None:
            num_columns = 8
            extra_headers = []
            xr_proportions = None
            uv_proportions = None
        else:
            num_columns = 10
            extra_headers = ["", "", "proportion"]

            opt_lrf_info = optimizer.objective_func(params, return_lrf_info=True)
            xr_proportions = opt_lrf_info.get_xr_proportions()
            uv_proportions = opt_lrf_info.get_uv_proportions()

        K = N*T
        m = me + mp
        sec_params_disp = (rp, t0, K, m)
        num_extended_rows = len(sec_params_disp) + 1

        data_list = [["" for c in range(num_columns)] for r in range(self.n*2 + 2 + 8 + num_extended_rows)]

        row_offset = 0
        for j, name in enumerate(["h", "", "", "", "", "rg"] + extra_headers, start=1):
            data_list[row_offset][j] = name

        row_offset += 1
        n = xr_params.shape[0]
        xr_col_size = 4     # xr_params.shape[1] for egh
        for i in range(n):
            if i == 0:
                v = "xr_params"
            else:
                v = ""
            data_list[row_offset+i][0] = v
            data_list[row_offset+i][1] = "%g" % xr_params[i]
            self.set_params_addr(xr_base+i, (row_offset+i, 1))

            data_list[row_offset+i][xr_col_size+2] = "%g" % rgs[i]
            self.set_params_addr(rg_base+i, (row_offset+i, xr_col_size+2))

            if xr_proportions is not None:
                if i == 0:
                    data_list[row_offset+i][xr_col_size+4] = "xr area"
                if i < len(xr_proportions):
                    data_list[row_offset+i][xr_col_size+5] = "%g" % xr_proportions[i]

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
            data_list[row_offset][9] = "proportion"

        row_offset += 1
        for i in range(n):
            if i == 0:
                v = "uv_params"
            else:
                v = ""
            data_list[row_offset+i][0] = v
            data_list[row_offset+i][1] = "%g" % uv_params[i]
            self.set_params_addr(uv_base+i, (row_offset+i, 1))

            if uv_proportions is not None:
                if i == 0:
                    data_list[row_offset+i][xr_col_size+4] = "uv area"
                if i < len(uv_proportions):
                    data_list[row_offset+i][xr_col_size+5] = "%g" % uv_proportions[i]

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

        dark_cells = []

        if sec_params is not None:
            # note: 0    1  2  3   4  5
            #       t0, rp, N, me, T, mp = sec_params
            row_offset += 1
            for name, value, i in zip(["rp (pore size)", "t0", "K", "m"], sec_params_disp, [1, 0, -1, -1]):
                row_offset += 1
                data_list[row_offset][0] = name
                data_list[row_offset][1] = "%g" % value
                if i >= 0:
                    self.set_params_addr(se_base+i, (row_offset, 1))
                else:
                    dark_cells.append((row_offset, 0))
                    dark_cells.append((row_offset, 1))
                    if name == "K":
                        data_list[row_offset][3] = "N (nperm)"
                        data_list[row_offset][4] = "%g" % N
                        self.set_params_addr(se_base+2, (row_offset, 4))
                        data_list[row_offset][6] = "T (tperm)"
                        data_list[row_offset][7] = "%g" % T
                        self.set_params_addr(se_base+4, (row_offset, 7))
                    elif name == "m":
                        data_list[row_offset][3] = "me"
                        data_list[row_offset][4] = "%g" % me
                        self.set_params_addr(se_base+3, (row_offset, 4))
                        data_list[row_offset][6] = "mp"
                        data_list[row_offset][7] = "%g" % mp
                        self.set_params_addr(se_base+5, (row_offset, 7))
                    else:
                        assert False

        self.num_valid_rows = row_offset + 2
        self.data_list = data_list
        column_width = 90
        width = column_width*num_columns + 60
        height = int(22*self.num_valid_rows) + 60
        self.sheet = Sheet(body_frame, width=width, height=height, data=data_list, show_selected_cells_border=False, column_width=column_width)

        for r, c in dark_cells:
            self.sheet.highlight_cells(r, c, bg="gray", fg="white")

        self.sheet.pack()

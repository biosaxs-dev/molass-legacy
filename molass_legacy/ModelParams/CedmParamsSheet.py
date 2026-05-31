"""
    ModelParams.CedmParamsSheet.py

    Parameter inspection sheet for Constrained-EDM (G2020 / CedmParams).

    XR layout per component: [a, b, cinj]   (no per-component t0/u/e/Dz)
    Shared column params appended at end: [t0_sh, u_sh, e_sh, Dz_sh]

    Copyright (c) 2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.OurTkinter import Tk
from tksheet import Sheet
from .ParamsSheetBase import ParamsSheetBase


class CedmParamsSheet(ParamsSheetBase):
    def __init__(self, parent, params, dsets, optimizer):
        ParamsSheetBase.__init__(self, parent, params, dsets, optimizer)

        self.n = optimizer.n_components
        nc = self.n - 1
        self.params_addr = [None] * len(params)
        self.params_addr_inv = {}

        body_frame = Tk.Frame(self)
        body_frame.pack()

        # split_params_simple returns 8 elements for CEDM:
        # [xr_params (nc×3), xr_baseparams, rg_params, mapping,
        #  uv_params, uv_baseparams, mappable_range, cedm_colparams]
        (xr_params, xr_baseparams, rgs, mapping,
         uv_params, uv_baseparams, mappable_range,
         cedm_colparams) = optimizer.split_params_simple(params)

        # Base indices for params_addr mapping
        xr_base    = 0
        xr_bp_base = xr_base + np.prod(xr_params.shape)
        rg_base    = xr_bp_base + len(xr_baseparams)
        mp_base    = rg_base + len(rgs)
        uv_base    = mp_base + len(mapping)
        uv_bp_base = uv_base + len(uv_params)
        mr_base    = uv_bp_base + len(uv_baseparams)
        # cedm_colparams are the last 4 values (appended outside split_params)
        col_base   = mr_base + len(mappable_range)

        # Compute per-component area proportions when dsets available
        if dsets is None:
            num_columns = 7
            xr_proportions = None
            uv_proportions = None
        else:
            from molass_legacy.Models.RateTheory.EDM import edm_impl
            num_columns = 10
            t0_sh, u_sh, e_sh, Dz_sh = cedm_colparams
            a_mp, b_mp = mapping
            (xr_curve, xrD), rg_curve, (uv_curve, uvD) = dsets
            xr_x = xr_curve.x
            xr_proportions = []
            uv_proportions = []
            for k, (a_k, b_k, cinj_k) in enumerate(xr_params):
                xr_cy = edm_impl(xr_x, t0_sh, u_sh, a_k, b_k, e_sh, Dz_sh, cinj_k)
                xr_proportions.append(np.sum(xr_cy))
                uv_cy = uv_params[k] * xr_cy
                uv_proportions.append(np.sum(uv_cy))
            xr_proportions = np.array(xr_proportions) / max(np.sum(xr_proportions), 1e-15)
            uv_proportions = np.array(uv_proportions) / max(np.sum(uv_proportions), 1e-15)

        num_extended_rows = 2 + 4  # 2 blank separators + 4 cedm_colparam rows

        num_rows = self.n * 2 + 2 + 8 + num_extended_rows
        data_list = [["" for _ in range(num_columns)] for _ in range(num_rows)]

        row_offset = 0

        # --- XR header row ---
        xr_col_size = xr_params.shape[1]   # 3: a, b, cinj
        for j, name in enumerate(["a", "b", "cinj", "", "rg"], start=1):
            data_list[row_offset][j] = name
        if xr_proportions is not None:
            data_list[row_offset][xr_col_size + 4] = "xr area"
            data_list[row_offset][xr_col_size + 5] = "uv area"

        row_offset += 1

        # --- XR params rows ---
        for i in range(nc):
            data_list[row_offset + i][0] = "xr_params" if i == 0 else ""
            for j in range(xr_col_size):
                data_list[row_offset + i][j + 1] = "%g" % xr_params[i, j]
                self.set_params_addr(xr_base + i * xr_col_size + j, (row_offset + i, j + 1))

            data_list[row_offset + i][xr_col_size + 2] = "%g" % rgs[i]
            self.set_params_addr(rg_base + i, (row_offset + i, xr_col_size + 2))

            if xr_proportions is not None:
                data_list[row_offset + i][xr_col_size + 4] = "%g" % xr_proportions[i]
                data_list[row_offset + i][xr_col_size + 5] = "%g" % uv_proportions[i]

        row_offset += nc + 1

        # --- XR baseline header ---
        col_names = ["slope", "intercept"]
        if len(xr_baseparams) == 3:
            col_names.append("fouling")
        for j, name in enumerate(col_names):
            data_list[row_offset][1 + j] = name

        row_offset += 1
        data_list[row_offset][0] = "xr_baseline"
        for j in range(len(xr_baseparams)):
            data_list[row_offset][j + 1] = "%g" % xr_baseparams[j]
            self.set_params_addr(xr_bp_base + j, (row_offset, j + 1))

        row_offset += 2

        # --- UV params header ---
        data_list[row_offset][1] = "scale"
        if uv_proportions is not None:
            data_list[row_offset][xr_col_size + 5] = "uv area"

        row_offset += 1
        for i in range(nc):
            data_list[row_offset + i][0] = "uv_params" if i == 0 else ""
            data_list[row_offset + i][1] = "%g" % uv_params[i]
            self.set_params_addr(uv_base + i, (row_offset + i, 1))

        row_offset += nc + 1

        # --- UV baseline header ---
        col_names = ["L", "x0", "k", "b", "s1", "s2", "diff_ratio"]
        if len(uv_baseparams) == 8:
            col_names.append("fouling")
        for j, name in enumerate(col_names):
            data_list[row_offset][1 + j] = name

        row_offset += 1
        data_list[row_offset][0] = "uv_baseline"
        for j in range(len(uv_baseparams)):
            data_list[row_offset][j + 1] = "%g" % uv_baseparams[j]
            self.set_params_addr(uv_bp_base + j, (row_offset, j + 1))

        row_offset += 2

        # --- Mapping + mappable_range ---
        for j, name in enumerate(["slope", "intercept", "", "", "from", "to"]):
            data_list[row_offset][1 + j] = name

        row_offset += 1
        data_list[row_offset][0] = "mapping"
        data_list[row_offset][4] = "mappable_range"
        for j in range(2):
            data_list[row_offset][1 + j] = "%g" % mapping[j]
            self.set_params_addr(mp_base + j, (row_offset, 1 + j))
            data_list[row_offset][5 + j] = "%g" % mappable_range[j]
            self.set_params_addr(mr_base + j, (row_offset, 5 + j))

        # --- CEDM shared column params ---
        row_offset += 1
        for i, name in enumerate(["t0_sh", "u_sh", "e_sh", "Dz_sh"]):
            row_offset += 1
            data_list[row_offset][0] = name
            data_list[row_offset][1] = "%g" % cedm_colparams[i]
            self.set_params_addr(col_base + i, (row_offset, 1))

        self.num_valid_rows = row_offset + 2
        self.data_list = data_list
        column_width = 90
        width = column_width * num_columns + 60
        height = int(22 * self.num_valid_rows) + 60
        self.sheet = Sheet(
            body_frame, width=width, height=height,
            data=data_list, show_selected_cells_border=False,
            column_width=column_width,
        )
        self.sheet.pack()

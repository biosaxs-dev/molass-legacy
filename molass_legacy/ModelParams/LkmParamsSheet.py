"""
    ModelParams.LkmParamsSheet.py

    Parameter inspection sheet for LKM (Lumped Kinetic Model, G1400).

    XR layout: xr_params = [scale_0, ..., scale_{nc-1}]   (one scale per component)
    LKM column params appended at end:
        [Pe, t0, R_0, k_MT_0, R_1, k_MT_1, ..., R_{nc-1}, k_MT_{nc-1}]

    Copyright (c) 2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.OurTkinter import Tk
from tksheet import Sheet
from .ParamsSheetBase import ParamsSheetBase


class LkmParamsSheet(ParamsSheetBase):
    def __init__(self, parent, params, dsets, optimizer):
        ParamsSheetBase.__init__(self, parent, params, dsets, optimizer)

        self.n = optimizer.n_components
        nc = self.n - 1
        self.params_addr = [None] * len(params)
        self.params_addr_inv = {}

        body_frame = Tk.Frame(self)
        body_frame.pack()

        # split_params_simple returns 8 elements for LKM:
        # [xr_params (nc,), xr_baseparams, rg_params, mapping,
        #  uv_params (nc,), uv_baseparams, mappable_range, lkm_colparams]
        (xr_params, xr_baseparams, rgs, mapping,
         uv_params, uv_baseparams, mappable_range,
         lkm_colparams) = optimizer.split_params_simple(params)

        xr_params = np.asarray(xr_params)
        uv_params = np.asarray(uv_params)

        # Base indices for params_addr mapping into flat params vector
        xr_base    = 0
        xr_bp_base = xr_base + len(xr_params)
        rg_base    = xr_bp_base + len(xr_baseparams)
        mp_base    = rg_base + len(rgs)
        uv_base    = mp_base + len(mapping)
        uv_bp_base = uv_base + len(uv_params)
        mr_base    = uv_bp_base + len(uv_baseparams)
        # lkm_colparams are appended outside the decomp block
        col_base   = mr_base + len(mappable_range)

        # 9 columns: 0=label, 1..8 for UV baseline (up to 8 params with fouling)
        num_columns = 9
        # lkm_colparams: [Pe, t0, R_0, k_MT_0, ..., R_{nc-1}, k_MT_{nc-1}]
        num_colparam_rows = len(lkm_colparams)    # = 2 + 2*nc
        num_extended_rows = 1 + num_colparam_rows  # 1 blank separator + param rows

        # Row budget: self.n*2 + 2 + 8 + num_extended_rows + 4
        # = 2 headers + 2*nc data rows (xr + uv)
        # + 2 baseline section rows + 8 fixed rows
        # + blank separator + lkm_colparam rows + 4 safety margin
        num_rows = self.n * 2 + 2 + 8 + num_extended_rows + 4
        data_list = [["" for _ in range(num_columns)] for _ in range(num_rows)]

        row_offset = 0

        # --- XR header row ---
        for j, name in enumerate(["scale", "", "rg"]):
            data_list[row_offset][j + 1] = name

        row_offset += 1

        # --- XR params rows (one scale per component) ---
        for i in range(nc):
            data_list[row_offset + i][0] = "xr_params" if i == 0 else ""
            data_list[row_offset + i][1] = "%g" % xr_params[i]
            self.set_params_addr(xr_base + i, (row_offset + i, 1))

            data_list[row_offset + i][3] = "%g" % rgs[i]
            self.set_params_addr(rg_base + i, (row_offset + i, 3))

        row_offset += nc + 1   # nc data rows + 1 blank separator

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

        row_offset += 2   # data row + 1 blank separator

        # --- UV header row ---
        data_list[row_offset][1] = "scale"

        row_offset += 1

        # --- UV params rows ---
        for i in range(nc):
            data_list[row_offset + i][0] = "uv_params" if i == 0 else ""
            data_list[row_offset + i][1] = "%g" % uv_params[i]
            self.set_params_addr(uv_base + i, (row_offset + i, 1))

        row_offset += nc + 1   # nc data rows + 1 blank separator

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

        row_offset += 2   # data row + 1 blank separator

        # --- Mapping + mappable_range ---
        for j, name in enumerate(["slope", "intercept", "", "from", "to"]):
            data_list[row_offset][1 + j] = name

        row_offset += 1
        data_list[row_offset][0] = "mapping"
        data_list[row_offset][4] = "mappable_range"
        for j in range(2):
            data_list[row_offset][1 + j] = "%g" % mapping[j]
            self.set_params_addr(mp_base + j, (row_offset, 1 + j))
            data_list[row_offset][4 + j] = "%g" % mappable_range[j]
            self.set_params_addr(mr_base + j, (row_offset, 4 + j))

        # --- LKM column params: Pe, t0, R_0, k_MT_0, ..., R_{nc-1}, k_MT_{nc-1} ---
        # Build human-readable names matching lkm_colparams layout
        colparam_names = ["Pe", "t0"] + [
            f"R_{k}" if j == 0 else f"k_MT_{k}"
            for k in range(nc)
            for j in range(2)
        ]

        row_offset += 1   # blank separator
        for i, name in enumerate(colparam_names):
            row_offset += 1
            data_list[row_offset][0] = name
            data_list[row_offset][1] = "%g" % lkm_colparams[i]
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

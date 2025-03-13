"""
    Optimizer.ParamSetType.LjEghParamsSheet.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
from .EghParamsSheet import EghParamsSheet

class LjEghParamsSheet(EghParamsSheet):
    def __init__(self, parent, params, dsets, optimizer):
        EghParamsSheet.__init__(self, parent, params, dsets, optimizer)

    def get_wanted_params(self, params, optimizer):
        xr_params, xr_baseparams, rgs, mapping, uv_params, uv_baseparams, mappable_range, seccol_params = optimizer.split_params_simple(params)
        dark_cell_j_list = [2]
        num_extra_addresses = xr_params.shape[0]
        Npc, tI = seccol_params[[0,2]]
        return xr_params, xr_baseparams, rgs, mapping, uv_params, uv_baseparams, mappable_range, seccol_params, dark_cell_j_list, num_extra_addresses, (tI, Npc)

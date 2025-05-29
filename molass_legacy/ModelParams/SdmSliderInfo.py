"""
    ModelParams.SdmSliderInfo.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.ModelParams.BaseSliderInfo import BaseSliderInfo

class SdmSliderInfo(BaseSliderInfo):
    def __init__(self, nc): 
        cmpparam_names = ["xr_h", "Rg", "uv_h"]

        cmpparam_indeces = []
        rg_base = nc + 2
        uv_base = rg_base + nc + 2
        for k in range(nc):
            cmpparam_indeces.append([k, rg_base+k, uv_base+k])

        whlparam_names = ["ma", "mb", "t0", "poresize"]
        mapping_start = nc*2 + 2
        colparam_start = nc*3 + 2 + 2 + 7 + 2
        whlparam_indeces = [mapping_start, mapping_start+1, colparam_start+2, colparam_start+3]

        BaseSliderInfo.__init__(self,
                                cmpparam_names=cmpparam_names,
                                cmpparam_indeces=cmpparam_indeces,
                                whlparam_names=whlparam_names,
                                whlparam_indeces=whlparam_indeces)
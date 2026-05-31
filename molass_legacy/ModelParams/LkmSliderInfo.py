"""
    ModelParams.LkmSliderInfo.py

    Copyright (c) 2025, SAXS Team, KEK-PF
"""
from .BaseSliderInfo import BaseSliderInfo

class LkmSliderInfo(BaseSliderInfo):
    def __init__(self, nc):
        # LKM per-component params: [scale]  (n_=1)
        # Rg params follow at nc*n_ (same convention as EdmSliderInfo)
        cmpparam_names = ["scale", "Rg"]

        n_ = 1
        rg_base = nc * n_
        cmpparam_indeces = []
        for k in range(nc):
            cmpparam_indeces.append([k] + [rg_base + k])

        BaseSliderInfo.__init__(self,
                                cmpparam_names=cmpparam_names,
                                cmpparam_indeces=cmpparam_indeces)

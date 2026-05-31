"""
    ModelParams.CedmSliderInfo.py

    Copyright (c) 2025, SAXS Team, KEK-PF
"""
from .BaseSliderInfo import BaseSliderInfo

class CedmSliderInfo(BaseSliderInfo):
    def __init__(self, nc):
        # CEDM per-component params: [a, b, cinj]  (n_=3)
        # Rg params follow at nc*n_ (same convention as EdmSliderInfo)
        cmpparam_names = ["a", "b", "cinj", "Rg"]

        n_ = 3
        rg_base = nc * n_
        cmpparam_indeces = []
        for k in range(nc):
            cmpparam_indeces.append(list(range(n_)) + [rg_base + k])

        BaseSliderInfo.__init__(self,
                                cmpparam_names=cmpparam_names,
                                cmpparam_indeces=cmpparam_indeces)

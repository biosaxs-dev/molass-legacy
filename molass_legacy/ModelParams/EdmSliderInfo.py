"""
    ModelParams.EdmSliderInfo.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
from ModelParams.BaseSliderInfo import BaseSliderInfo

class EdmSliderInfo(BaseSliderInfo):
    def __init__(self, nc): 
        cmpparam_names = ["t0", "u", "a", "b", "e", "Dz", "cinj", "Rg"]

        cmpparam_indeces = []
        n_ = 7
        rg_base = nc*n_
        for k in range(nc):
            cmpparam_indeces.append(list(range(n_)) + [rg_base+k])

        BaseSliderInfo.__init__(self,
                                cmpparam_names=cmpparam_names,
                                cmpparam_indeces=cmpparam_indeces)